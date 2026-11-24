# CARMEN-Numeric

UniTTab-style encoder for **perioperative numerical data** — minute-level vital-sign
trends, ventilator and anesthetic-depth readings, TCI concentrations, intra-operative
labs and pre-operative demographics — as the numerical branch that will later be fused
with the [CARMEN](https://github.com/dlcjfgmlnasa/CARMEN) waveform encoder by
cross-attention.

The model follows **UniTTab** (Luetto et al., *One Transformer for All Time Series:
Representing and Training with Time-Dependent Heterogeneous Tabular Data*,
[arXiv 2302.06375](https://arxiv.org/abs/2302.06375); official code
[fabriziogaruti/UniTTab](https://github.com/fabriziogaruti/UniTTab)).

## What is implemented

| Paper component | Here |
| --- | --- |
| Heterogeneous rows (row types with different fields) | `TableSchema` / `RowType`; VitalDB uses `static`, `monitor` (1/min), `lab` (per blood draw) |
| Frequency-based numerical embedding γ(v), L = 8 | `FrequencyEncoding` + one Linear(2L, d) per field |
| Categorical embedding | per-field lookup |
| Field Transformer (1 layer, 8 heads, d = 72) | shared by all row types, run per type on dense blocks |
| Row-type projection W_h (Eq. 1) and S_h | `row_in[h]`, `row_out[h]` |
| Sequence Transformer (12 layers, 12 heads, m = 1080) + [CLS] | `seq_encoder` (`--preset paper`; smaller presets for quick runs) |
| Masked Token pretext task: field p_f = 0.15, row p_r = 0.1, joint time-stamp masking | `sample_mask` |
| Quantized numerical targets + neighborhood label smoothing (±5 bins, ε = 0.1) | `Normalizer` (quantile bins), `soft_targets`, `masked_token_loss` |

Deliberate choices where the paper is silent or the clinical setting differs:

* **Missing values** get a learned per-field `[MISSING]` embedding and are never
  prediction targets (clinical rows are sparse: NIBP every few minutes, ART only with an
  arterial line, labs a few times per case).
* **Input scaling** uses a *global* robust range ([0.5, 99.5] percentile of the training
  data) — not per patient — so the absolute physiological level is kept, the same
  principle as CARMEN's loc/scale conditioning.
* **Time** is relative (`t_min` since anesthesia start, `dt_min` since the previous
  row), frequency-encoded, instead of calendar fields.
* **Artifacts** outside a physically possible `valid_range` become missing; the
  physiological tails (hypotension, desaturation) are kept.
* Splits are **patient-level** (`subjectid`).

## Layout

```
carmen_numeric/
  schema.py        fields, row types
  normalize.py     input scaling, quantile bins, decoding
  model.py         UniTTab, masking, loss
  evaluate.py      reconstruction MAE in physical units vs. baselines
  data/vitaldb.py  VitalDB -> rows (static / monitor / lab)
  data/dataset.py  windows, collate, patient split
scripts/
  prepare_vitaldb.py   build the row cache (API download or local .vital files)
  pretrain.py          masked-token pre-training (CPU, 1 GPU or torchrun DDP)
  orch/orchctl.py      주경야독 GPU API client (from Biosignal-Foundation-Model)
  orch/node_pretrain.sh  prepare + train on the GPU node
tests/
```

## Quick start (VitalDB open data)

```bash
pip install -e ".[vitaldb,dev]"
python scripts/prepare_vitaldb.py --out data/vitaldb_numeric --max-cases 600 --workers 8
python scripts/pretrain.py --data data/vitaldb_numeric --out runs/small --preset small \
    --steps 3000 --batch-size 32 --lr 3e-4
pytest -q
```

Evaluation prints, for masked values of each field, the MAE in the field's unit for the
model (bin mode and expectation) next to two baselines on the same masks — the
training median and the **nearest observed value** in the sequence (forward/backward
fill). Beating `nearest` on dense vitals is the real bar.

## On the 주경야독 GPU node

`scripts/orch/orchctl.py` reads `ORCH_PAT` (a `pat_…` token from the portal) from the
environment or `~/.orch/orch.env`; the base URL defaults to
`https://koreahealth.io/compute`.

```bash
python scripts/orch/orchctl.py me
python scripts/orch/orchctl.py up --gpus 8
python scripts/orch/orchctl.py runfile scripts/orch/node_pretrain.sh
python scripts/orch/orchctl.py down
```

`node_pretrain.sh` builds the cache from `$ORCH_HOME/vitaldb/raw/*.vital` (and the
clinical / lab CSVs under `$ORCH_HOME/aki/csv/`) and launches `torchrun` detached.
Patient data stays on the node; only code goes through git.

## Next

* downstream probes (e.g. intra-operative hypotension, post-op AKI) on the `[CLS]` /
  row embeddings;
* SNUH OR numerical tracks with the same schema;
* cross-attention fusion with frozen CARMEN patch tokens on a shared time axis.
