"""Hypotension prediction from minute-level numbers (UniTTab) + CARMEN waveform tokens.

    python scripts/finetune_fusion.py --mode fusion --data $SIG --label-data $SIG \
        --ckpt runs/unittab_signal_v1/best.pt --tokens runs/../carmen_tokens --out runs/fusion
    python scripts/finetune_fusion.py --mode wave ...   # waveform tokens only
    python scripts/finetune_fusion.py --mode late ...   # concatenate pooled numbers + waveform

Samples, labels and splits are those of ``probe_ioh.py``; waveform tokens come from
``extract_carmen_tokens.py`` (frozen CARMEN, the 60 s ending at the prediction time,
``3 signals x 12 five-second tokens``) in the same sample order.

``fusion``: the numeric encoder (pre-trained, fine-tuned here) yields three query tokens
``[CLS, last row, mean of rows]``; two cross-attention blocks (self-attention among the
queries, then attention to the projected waveform tokens + a learned "no waveform" token)
update them, and an MLP gives one logit per horizon. ``wave``: the same blocks with learned
queries and no numeric encoder. ``late``: the numeric features concatenated with mean-pooled
waveform tokens. ``gated``: the numeric MLP head of ``finetune_ioh.py`` plus a residual waveform
branch (cross-attention blocks + MLP whose last layer starts at zero), so the model starts as the
numeric-only model; with ``--init-from`` (a ``finetune_ioh.py`` ``best.pt``) it starts from
that model's weights and the step-0 state competes in checkpoint selection, and with
``--freeze-numeric`` only the waveform branch is trained (the numeric model is kept as is, in
eval mode, so the checkpoints differ only by the waveform residual). ``bidir``: both
single-modality models frozen — the numeric model of ``--init-from`` and the waveform model of
``--wave-from`` (a ``--mode wave`` ``best.pt``) — mixed per signal subset actually present
(``a[s] * z_numeric + b[s] * z_wave + c[s]``, one weight per subset and label) plus a residual
cross-attention branch (numeric queries over waveform tokens, last layer zero). Each subset's
weights start at the single-modality model that has the higher val AUROC for that subset and
label, so the step-0 model is the better of the two per waveform condition — the "primary"
modality is chosen by the data, in either direction — and step 0 competes in checkpoint
selection. The frozen numeric encoder is run once per split (cached). CARMEN stays
frozen. The step with the best mean val AUROC is kept; test metrics, curves and logits
(``test_scores.npz``) are written.

Missing modalities: ``--drop-wave`` / ``--drop-signal`` hide all waveform tokens of a sample
/ the tokens of one signal at random during training (modality dropout); ``--subset-train``
draws a signal subset per sample instead. Test scores are computed for every signal subset
(``eap``, ``ea``, ``ep`` = no arterial line, ``ap``, ``e``, ``a``, ``p``, ``none``). With
``--token-mode joint`` subsets are made by masking tokens of a joint encoding (leaky: the kept
tokens saw the other signals); ``subsets`` uses a separate CARMEN encoding per subset and
``independent`` single-signal encodings (both exact). ``--eval-only CKPT`` scores an existing
``best.pt`` without training.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import downstream_tasks as T  # noqa: E402
import finetune_ioh as F  # noqa: E402
import probe_ioh as P  # noqa: E402
import viz_predictions as V  # noqa: E402

from carmen_numeric import TableSchema  # noqa: E402
from carmen_numeric.data.dataset import load_cases  # noqa: E402

N_SIG, N_CHUNK = 3, 12
SLOT = {"ecg": 0, "abp": 1, "ppg": 2}  # token slots = CARMEN signal types
# signal subsets as bitmasks (ECG 1, ABP 2, PPG 4) and their token-file names
SUBSET_NAME = {1: "e", 2: "a", 3: "ea", 4: "p", 5: "ep", 6: "ap", 7: "eap"}
COND_MASK = {"eap": 7, "ea": 3, "ep": 5, "ap": 6, "e": 1, "a": 2, "p": 4, "none": 0}
CONDITIONS = tuple(COND_MASK)
ALIASES = {"full": "eap", "no_wave": "none", "no_abp": "ep", "abp_only": "a"}


def slot_mask(bits: torch.Tensor) -> torch.Tensor:  # (B,) bitmask -> (B, 36) allowed slots
    sig = torch.stack([(bits >> k) & 1 for k in range(N_SIG)], -1).bool()  # (B, 3)
    return sig[..., None].expand(-1, -1, N_CHUNK).reshape(len(bits), N_SIG * N_CHUNK)


def apply_condition(valid: torch.Tensor, cond: str) -> torch.Tensor:  # (B, 36) -> (B, 36)
    """Mask-based condition (on jointly encoded tokens this leaks — see TokenSource)."""
    bits = torch.full((len(valid),), COND_MASK[ALIASES.get(cond, cond)], device=valid.device)
    return valid & slot_mask(bits)


class TokenSource:
    """CARMEN tokens of one split; ``get(idx, bits)`` gives tokens of a signal subset.

    ``joint``: one joint encoding of all signals; subsets by masking tokens (other signals'
    information stays in the kept tokens). ``subsets``: a separate encoding per subset
    (``<split>_<subset>_tokens.npy`` from ``extract_carmen_tokens.py --subsets all``) —
    exact. ``independent``: the single-signal encodings side by side; subsets by masking,
    exact because no signal saw another.
    """

    def __init__(self, path: Path, part: str, samples: dict, device, mode: str,
                 store=None) -> None:
        # store: where the token arrays live ("cpu" when they do not fit on the GPU; batches
        # are gathered there and moved to ``device``)
        self.device = device
        self.store = torch.device(store) if store is not None else device
        meta = np.load(path / f"{part}_meta.npz")
        rows = None  # token rows of the samples, when the token file covers more samples
        if not (np.array_equal(meta["caseid"], samples["caseid"])
                and np.allclose(meta["t"], samples["t"])):
            where = {(int(c), round(float(t), 3)): i
                     for i, (c, t) in enumerate(zip(meta["caseid"], meta["t"], strict=True))}
            try:
                rows = np.array([where[(int(c), round(float(t), 3))]
                                 for c, t in zip(samples["caseid"], samples["t"], strict=True)])
            except KeyError as e:
                raise ValueError(f"{part}: sample {e} has no tokens") from None

        def load(name: str) -> tuple[torch.Tensor, torch.Tensor]:
            global N_CHUNK
            tok = np.load(path / f"{name}_tokens.npy", mmap_mode="r")
            val = np.load(path / f"{name}_valid.npy")
            N_CHUNK = tok.shape[1] // N_SIG  # 12 (5 s tokens) unless --chunk-s was changed
            if rows is not None:
                tok, val = tok[rows], val[rows]
            return (torch.from_numpy(np.ascontiguousarray(tok)).to(self.store),
                    torch.from_numpy(val).to(self.store))
        self.mode = mode
        if mode == "joint":
            self.tok, self.valid = load(part)
        elif mode == "independent":
            parts = [load(f"{part}_{k}") for k in ("e", "a", "p")]
            self.tok = parts[0][0] + parts[1][0] + parts[2][0]  # slot-disjoint, zeros elsewhere
            self.valid = parts[0][1] | parts[1][1] | parts[2][1]
        elif mode == "subsets":
            loaded = [load(f"{part}_{SUBSET_NAME[b]}") for b in range(1, 8)]
            self.tok_all = torch.stack([t for t, _ in loaded])  # (7, n, 36, d)
            self.valid_all = torch.stack([v for _, v in loaded])
            self.tok, self.valid = self.tok_all[6], self.valid_all[6]  # all signals
        else:
            raise ValueError(mode)
        sig = self.valid.view(-1, N_SIG, N_CHUNK).any(-1).to(device)  # (n, 3) signal present
        self.present_bits = (sig.long() * torch.tensor([1, 2, 4], device=device)).sum(-1)

    def _out(self, tok: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tok.to(self.device), valid.to(self.device)

    def get(self, idx: torch.Tensor, bits: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        i = idx.to(self.store)
        if bits is None:
            return self._out(self.tok[i], self.valid[i])
        bits = bits & self.present_bits[idx]
        if self.mode != "subsets":
            tok, valid = self._out(self.tok[i], self.valid[i])
            return tok, valid & slot_mask(bits)
        k = (bits - 1).clamp(min=0).to(self.store)
        tok, valid = self._out(self.tok_all[k, i], self.valid_all[k, i])
        return tok, valid & (bits > 0)[:, None]


def sample_subsets(present: torch.Tensor, p_full: float, p_none: float,
                   gen: torch.Generator) -> torch.Tensor:
    """Per-sample signal subset for training: all available signals with ``p_full``, none
    with ``p_none``, otherwise a random non-empty subset of the available ones."""
    b = len(present)
    u = torch.rand(b, generator=gen, device=present.device)
    r = torch.randint(1, 8, (b,), generator=gen, device=present.device) & present
    r = torch.where(r == 0, present, r)
    return torch.where(u < p_full, present, torch.where(u > 1 - p_none, torch.zeros_like(r), r))


def modality_dropout(valid: torch.Tensor, p_wave: float, p_signal: float,
                     gen: torch.Generator) -> torch.Tensor:
    v = valid.clone().view(-1, N_SIG, N_CHUNK)
    b = v.shape[0]
    if p_signal > 0:
        drop = torch.rand(b, N_SIG, generator=gen, device=v.device) < p_signal
        v &= ~drop[..., None]
    if p_wave > 0:
        drop = torch.rand(b, generator=gen, device=v.device) < p_wave
        v &= ~drop[:, None, None]
    return v.view(-1, N_SIG * N_CHUNK)


class WaveTokens(nn.Module):
    """CARMEN tokens (B, 36, d_c) -> (B, 37, m) with signal / time embeddings + a null token."""

    def __init__(self, d_carmen: int, m: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_carmen), nn.Linear(d_carmen, m))
        self.sig = nn.Embedding(N_SIG, m)
        self.pos = nn.Embedding(N_CHUNK, m)
        self.null = nn.Parameter(torch.zeros(1, 1, m))
        idx = torch.arange(N_SIG * N_CHUNK)
        self.register_buffer("sig_idx", idx // N_CHUNK, persistent=False)
        self.register_buffer("pos_idx", idx % N_CHUNK, persistent=False)

    def forward(self, tok: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(tok.float()) + self.sig(self.sig_idx) + self.pos(self.pos_idx)
        b = x.shape[0]
        x = torch.cat([self.null.expand(b, -1, -1), x], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~valid], 1)
        return x, pad  # pad: True = ignore


class CrossBlocks(nn.Module):
    def __init__(self, m: int, layers: int = 2, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(m, heads, 4 * m, dropout, batch_first=True,
                                       norm_first=True, activation="gelu")
            for _ in range(layers)])
        self.norm = nn.LayerNorm(m)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            q = blk(q, mem, memory_key_padding_mask=pad)
        return self.norm(q)


class FusionModel(nn.Module):
    def __init__(self, mode: str, encoder, d_carmen: int, m: int, n_out: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.mode = mode
        self.frozen_numeric = False
        self.encoder = encoder if mode != "wave" else None
        self.wave = WaveTokens(d_carmen, m)
        if mode in ("fusion", "wave", "gated"):
            self.cross = CrossBlocks(m, dropout=dropout)
        if mode == "wave":
            self.queries = nn.Parameter(torch.randn(1, 3, m) * 0.02)
        width = {"fusion": 3 * m, "wave": 3 * m, "late": 4 * m, "gated": 3 * m}[mode]
        self.head = self._mlp(width, n_out, dropout)
        if mode == "gated":  # residual waveform branch, zero at init
            self.wave_head = self._mlp(3 * m, n_out, dropout)
            nn.init.zeros_(self.wave_head[-1].weight)
            nn.init.zeros_(self.wave_head[-1].bias)

    def freeze_numeric(self) -> None:
        """Keep the numeric encoder + head fixed (no gradients, eval-mode dropout)."""
        self.frozen_numeric = True
        for p in [*self.encoder.parameters(), *self.head.parameters()]:
            p.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_numeric:
            self.encoder.eval()
            self.head.eval()
        return self

    @staticmethod
    def _mlp(width: int, n_out: int, dropout: float) -> nn.Sequential:  # = IOHModel.head
        return nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout),
                             nn.Linear(width, 256), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(256, n_out))

    def numeric_queries(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # (B, 3, m)
        z = self.encoder.encode(batch)
        rows, mask = z["rows"].float(), z["row_mask"]
        n = mask.sum(1).clamp(min=1)
        last = rows[torch.arange(len(n), device=rows.device), n - 1]
        mean = (rows * mask[..., None]).sum(1) / n[:, None]
        return torch.stack([z["cls"].float(), last, mean], 1)

    def forward(self, batch, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mem, pad = self.wave(tok, valid)
        if self.mode == "late":
            q = self.numeric_queries(batch)
            keep = (~pad).float()[..., None]
            w = (mem * keep).sum(1) / keep.sum(1)  # null token is always kept
            return self.head(torch.cat([q.flatten(1), w], 1))
        if self.mode == "gated":
            q = self.numeric_queries(batch)
            return self.head(q.flatten(1)) + self.wave_head(self.cross(q, mem, pad).flatten(1))
        q = self.numeric_queries(batch) if self.mode == "fusion" \
            else self.queries.expand(tok.shape[0], -1, -1)
        return self.head(self.cross(q, mem, pad).flatten(1))


class NumericCache:
    """Frozen numeric model outputs per sample: queries ``(n, 3, m)`` and logits ``(n, H)``."""

    def __init__(self, model: FusionModel, win, n: int, device, batch_size: int = 512) -> None:
        qs, zs = [], []
        model.eval()
        with torch.no_grad():
            for s in range(0, n, batch_size):
                idx = torch.arange(s, min(s + batch_size, n), device=device)
                q = model.numeric_queries(win.batch(idx)).float()
                qs.append(q.half())
                zs.append(model.head(q.flatten(1)).float())
        self.q, self.z = torch.cat(qs), torch.cat(zs)

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"q": self.q[idx], "z": self.z[idx]}


class BidirModel(nn.Module):
    """``a[s] * z_numeric + b[s] * z_wave + c[s] + residual`` (``s`` = signal subset present)."""

    def __init__(self, wave_model: FusionModel, d_carmen: int, m: int, n_out: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.mode = "bidir"
        self.wave_model = wave_model
        for p in wave_model.parameters():
            p.requires_grad_(False)
        self.wave = WaveTokens(d_carmen, m)
        self.cross = CrossBlocks(m, dropout=dropout)
        self.res_head = FusionModel._mlp(3 * m, n_out, dropout)
        nn.init.zeros_(self.res_head[-1].weight)
        nn.init.zeros_(self.res_head[-1].bias)
        self.a = nn.Parameter(torch.ones(8, n_out))  # index = subset bitmask (ECG 1, ABP 2, PPG 4)
        self.b = nn.Parameter(torch.zeros(8, n_out))
        self.c = nn.Parameter(torch.zeros(8, n_out))
        self.register_buffer("bit", torch.tensor([1, 2, 4]), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.wave_model.eval()
        return self

    def present(self, valid: torch.Tensor) -> torch.Tensor:  # (B, 3 * N_CHUNK) -> (B,) bits
        return (valid.view(len(valid), N_SIG, -1).any(-1).long() * self.bit).sum(-1)

    def forward(self, batch, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            zw = self.wave_model(None, tok, valid).float()
        mem, pad = self.wave(tok, valid)
        res = self.res_head(self.cross(batch["q"].float(), mem, pad).flatten(1))
        s = self.present(valid)
        return self.a[s] * batch["z"] + self.b[s] * zw + self.c[s] + res

    @torch.no_grad()
    def init_mix(self, y: np.ndarray, cache: NumericCache, src: TokenSource, device,
                 min_pos: int = 10) -> list[dict]:
        """Start each (subset, label) at the single-modality model with the higher val AUROC,
        pooling val samples over the forced conditions by the subset actually present."""
        self.eval()
        n = len(y)
        zs, ws, ss = [], [], []
        for cond in CONDITIONS:
            for s0 in range(0, n, 512):
                idx = torch.arange(s0, min(s0 + 512, n), device=device)
                bits = torch.full((len(idx),), COND_MASK[cond], device=device, dtype=torch.long)
                tok, valid = src.get(idx, bits)
                zw = self.wave_model(None, tok, valid).float()
                zs.append(cache.z[idx].cpu())
                ws.append(zw.cpu())
                ss.append(self.present(valid).cpu())
        zn, zw, sub = torch.cat(zs).numpy(), torch.cat(ws).numpy(), torch.cat(ss).numpy()
        yy = np.tile(y, (len(CONDITIONS), 1))
        table = []
        for s in range(8):
            for k in range(y.shape[1]):
                ok = (sub == s) & ~np.isnan(yy[:, k])
                pos = int(np.nansum(yy[ok, k]))
                an = P.auroc(yy[ok, k], zn[ok, k]) if pos >= min_pos and pos < ok.sum() else np.nan
                aw = P.auroc(yy[ok, k], zw[ok, k]) if pos >= min_pos and pos < ok.sum() else np.nan
                wave_first = s > 0 and np.isfinite(aw) and aw > an
                self.a[s, k], self.b[s, k] = (0.0, 1.0) if wave_first else (1.0, 0.0)
                table.append({"subset": SUBSET_NAME.get(s, "none"), "label": k,
                              "n": int(ok.sum()), "pos": pos,
                              "val_auroc_numeric": None if np.isnan(an) else round(float(an), 4),
                              "val_auroc_wave": None if np.isnan(aw) else round(float(aw), 4),
                              "primary": "wave" if wave_first else "numeric"})
        return table


@torch.no_grad()
def predict_cond(model, win, src: TokenSource, cond: str, n: int, device,
                 batch_size: int = 512) -> np.ndarray:
    """Logits with the tokens of one signal subset (condition name or alias)."""
    model.eval()
    out = []
    bits_all = COND_MASK[ALIASES.get(cond, cond)]
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n), device=device)
        bits = torch.full((len(idx),), bits_all, device=device, dtype=torch.long)
        tok, valid = src.get(idx, bits)
        b = win.batch(idx) if win is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(b, tok, valid).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True,
                    choices=["fusion", "wave", "late", "gated", "bidir"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-data", required=True)
    ap.add_argument("--ckpt", required=True, help="pre-trained UniTTab (also sets d_model)")
    ap.add_argument("--tokens", required=True, help="extract_carmen_tokens.py output dir")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--example-cases", type=int, nargs="*", default=[])
    ap.add_argument("--drop-wave", type=float, default=0.0,
                    help="training: probability of hiding all waveform tokens of a sample")
    ap.add_argument("--drop-signal", type=float, default=0.0,
                    help="training: probability of hiding one signal's tokens")
    ap.add_argument("--eval-only", default=None,
                    help="best.pt of an earlier run: skip training, score test conditions")
    ap.add_argument("--token-mode", default="joint", choices=["joint", "subsets", "independent"])
    ap.add_argument("--token-device", default=None, choices=["cpu"],
                    help="keep CARMEN tokens in host memory and move batches to the GPU "
                         "(when the tokens do not fit next to other jobs)")
    ap.add_argument("--subset-train", nargs=2, type=float, default=None,
                    metavar=("P_FULL", "P_NONE"),
                    help="training: per-sample random signal subsets (instead of --drop-*)")
    ap.add_argument("--select-conds", nargs="+", default=["eap"],
                    help="val conditions whose mean AUROC picks the checkpoint")
    ap.add_argument("--task", default="ioh", choices=list(T.TASKS))
    ap.add_argument("--cases-csv", default=None, help="clinical_data.csv (case-level tasks)")
    ap.add_argument("--labs-csv", default=None, help="lab_data.csv (aki)")
    ap.add_argument("--arr-dir", default=None,
                    help="VitalDB Arrhythmia Database dir (task arrhythmia)")
    ap.add_argument("--split", default=None,
                    help="split json (default: <label-data>/split.json), e.g. a CV fold")
    ap.add_argument("--init-from", default=None,
                    help="--mode gated: finetune_ioh.py best.pt for the encoder + numeric head")
    ap.add_argument("--freeze-numeric", action="store_true",
                    help="--mode gated with --init-from: train only the waveform branch")
    ap.add_argument("--wave-from", default=None,
                    help="--mode bidir: finetune_fusion.py --mode wave best.pt (frozen)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.init_from and args.mode not in ("gated", "bidir"):
        ap.error("--init-from needs --mode gated or bidir")
    if args.mode == "bidir" and not (args.init_from and args.wave_from):
        ap.error("--mode bidir needs --init-from (numeric) and --wave-from (waveform)")
    if args.freeze_numeric and not args.init_from:
        ap.error("--freeze-numeric needs --init-from")

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, label_data, tok_dir = Path(args.data), Path(args.label_data), Path(args.tokens)
    lschema = TableSchema.load(label_data / "schema.json")
    split = json.loads(Path(args.split or label_data / "split.json").read_text())
    ids = {p: sorted(split[p])[: args.max_cases] for p in P.PARTS}
    t0 = time.time()
    samples = {p: T.build_task(args.task, load_cases(label_data, ids[p]), lschema,
                               args.cases_csv, args.labs_csv,
                               arr_dir=args.arr_dir) for p in P.PARTS}
    labels = samples["train"]["labels"]
    simple_name, simple_col, simple_sign = samples["train"]["simple"]
    counts = {p: len(samples[p]["t"]) for p in P.PARTS}
    toks = {p: TokenSource(tok_dir, p, samples[p], device, args.token_mode, args.token_device)
            for p in P.PARTS}
    d_carmen = toks["train"].tok.shape[-1]
    wave_rate = {p: float(toks[p].valid.any(1).float().mean()) for p in P.PARTS}
    print(f"samples {counts}; with waveform {wave_rate} ({time.time() - t0:.0f}s)", flush=True)

    encoder = P.load_model(args.ckpt, device)
    m = encoder.cfg.d_model
    win = {p: F.Windows(data, list(samples[p]["caseid"]), samples[p]["caseid"],
                        samples[p]["t"], encoder.cfg.max_rows, device)
           for p in P.PARTS} if args.mode != "wave" else {p: None for p in P.PARTS}
    mix_init = None
    if args.mode == "bidir":
        num = FusionModel("gated", encoder, d_carmen, m, len(labels))
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = num.load_state_dict(ck["model"], strict=False)
        if unexpected or any(k.startswith(("encoder.", "head.")) for k in missing):
            raise ValueError(f"{args.init_from}: not a finetune_ioh.py model of this size")
        num.to(device)
        win = {p: NumericCache(num, win[p], counts[p], device) for p in P.PARTS}
        del num, encoder
        wck = torch.load(args.wave_from, map_location="cpu", weights_only=False)
        if wck["mode"] != "wave":
            raise ValueError(f"{args.wave_from} is a {wck['mode']} model, not wave")
        wave_model = FusionModel("wave", None, d_carmen, m, len(labels))
        wave_model.load_state_dict(wck["model"])
        model = BidirModel(wave_model, d_carmen, m, len(labels)).to(device)
        mix_init = model.init_mix(samples["val"]["y"], win["val"], toks["val"], device)
        print("bidir start (primary per subset / label): " + "  ".join(
            f"{r['subset']}:{r['label']}={r['primary'][0]}" for r in mix_init), flush=True)
        print(f"numeric outputs cached ({time.time() - t0:.0f}s)", flush=True)
    else:
        model = FusionModel(args.mode, encoder, d_carmen, m, len(labels)).to(device)
    if args.init_from and args.mode != "bidir":
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if unexpected or any(k.startswith(("encoder.", "head.")) for k in missing):
            raise ValueError(f"{args.init_from}: not a finetune_ioh.py model of this size")
        model.to(device)
        if args.freeze_numeric:
            model.freeze_numeric()
    encoder_of = getattr(model, "encoder", None)
    enc_params = [p for p in encoder_of.parameters() if p.requires_grad] \
        if encoder_of is not None else []
    enc_ids = {id(p) for p in encoder_of.parameters()} if encoder_of is not None else set()
    mix = [model.a, model.b, model.c] if args.mode == "bidir" else []
    mix_ids = {id(p) for p in mix}
    other = [p for p in model.parameters()
             if id(p) not in enc_ids | mix_ids and p.requires_grad]
    groups = [{"params": other, "lr": args.lr_head}]
    if mix:  # mixing weights: no weight decay (it would pull the start model towards 0)
        groups.append({"params": mix, "lr": args.lr_head, "weight_decay": 0.0})
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lr = [g["lr"] for g in opt.param_groups]
    y = {p: torch.as_tensor(samples[p]["y"], dtype=torch.float32, device=device)
         for p in P.PARTS}
    gen = torch.Generator(device=device).manual_seed(args.seed)
    best, best_state, best_step, bad, log = -1.0, None, 0, 0, []
    model.train()

    def validate(step: int, loss: float) -> bool:
        """Log val AUROC, keep the best state; True when patience is used up."""
        nonlocal best, best_state, best_step, bad
        va = float(np.mean([
            F.mean_auroc(samples["val"]["y"], predict_cond(model, win["val"], toks["val"],
                                                           c, counts["val"], device))
            for c in args.select_conds]))
        log.append({"step": step, "loss": round(loss, 4), "val_auroc": round(va, 4)})
        print(json.dumps(log[-1]) + f" ({time.time() - t0:.0f}s)", flush=True)
        if va > best:
            best, best_step, bad = va, step, 0
            best_state = copy.deepcopy({k: v.detach().cpu()
                                        for k, v in model.state_dict().items()})
            return False
        bad += 1
        return bad >= args.patience

    if args.init_from and not args.eval_only:
        validate(0, float("nan"))  # the numeric-only starting point competes
    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        if ck["mode"] != args.mode:
            raise ValueError(f"{args.eval_only} is a {ck['mode']} model")
        best_state, best_step = ck["model"], ck["step"]
    for step in range(1, 0 if args.eval_only else args.steps + 1):
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g, lr in zip(opt.param_groups, base_lr, strict=True):
            g["lr"] = lr * max(f, 0.05)
        idx = torch.randint(counts["train"], (args.batch_size,), device=device, generator=gen)
        yy = y["train"][idx]
        ok = ~torch.isnan(yy)
        b = win["train"].batch(idx) if win["train"] is not None else None
        if args.subset_train:
            bits = sample_subsets(toks["train"].present_bits[idx], *args.subset_train, gen)
            tt, tv = toks["train"].get(idx, bits)
        else:
            tt, tv = toks["train"].get(idx)
            tv = modality_dropout(tv, args.drop_wave, args.drop_signal, gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(b, tt, tv).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logit[ok],
                                                              torch.nan_to_num(yy)[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step % args.eval_every == 0 or step == args.steps) and validate(step, loss.item()):
            print(f"early stop at {step}", flush=True)
            break
    model.load_state_dict(best_state)
    te = samples["test"]
    cond_logits = {}
    for cond in CONDITIONS:
        cond_logits[cond] = predict_cond(model, win["test"], toks["test"], cond,
                                         counts["test"], device)
        np.savez(out / f"test_scores_{cond}.npz", logit=cond_logits[cond], caseid=te["caseid"],
                 t=te["t"])
    logit = cond_logits["eap"]
    np.savez(out / "test_scores.npz", logit=logit, caseid=te["caseid"], t=te["t"],
             has_wave=toks["test"].valid.any(1).cpu().numpy())

    res: dict = {"task": args.task, "mode": args.mode, "best_step": best_step,
                 "best_val_auroc": round(best, 4), "log": log, "counts": counts,
                 "with_waveform": wave_rate, "args": vars(args), "horizons": {},
                 "conditions": {}}
    if mix_init is not None:
        res["mix"] = {"init": mix_init, **{k: getattr(model, k).detach().cpu().numpy()
                                           .round(4).tolist() for k in ("a", "b", "c")}}
    for cond, lg in cond_logits.items():
        res["conditions"][cond] = {}
        for k, h in enumerate(labels):
            ok = ~np.isnan(te["y"][:, k])
            res["conditions"][cond][str(h)] = P.bootstrap(
                te["y"][ok, k], lg[ok, k], te["caseid"][ok], n=500)
        print(f"[{cond}] " + "  ".join(
            f"{h}: AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f}"
            for h, r in res["conditions"][cond].items()), flush=True)
    for k, h in enumerate(labels):
        ok = {p: ~np.isnan(samples[p]["y"][:, k]) for p in P.PARTS}
        yt = te["y"][ok["test"], k]
        g = te["caseid"][ok["test"]]
        hand = {p: samples[p]["hand"][ok[p]] for p in P.PARTS}
        scores = {
            args.mode: 1 / (1 + np.exp(-logit[ok["test"], k])),
            "hand_gbm": P.gbm_probe(hand["train"], samples["train"]["y"][ok["train"], k],
                                    hand["test"]),
            simple_name: simple_sign * hand["test"][:, simple_col],
        }
        res["horizons"][str(h)] = {
            "n": int(len(yt)), "prevalence": round(float(yt.mean()), 4),
            "methods": {mm: {**P.bootstrap(yt, s, g), **V.curve_points(yt, s),
                             "operating": V.operating_points(yt, s,
                                                             per_hour=te["per_hour"])}
                        for mm, s in scores.items()},
            "calibration": {mm: V.calibration(yt, scores[mm]) for mm in (args.mode, "hand_gbm")},
        }
        r = res["horizons"][str(h)]["methods"]
        print(f"--- {h}: " + "  ".join(
            f"{mm} {v['auroc']:.3f} [{v['auroc_ci'][0]:.3f},{v['auroc_ci'][1]:.3f}] "
            f"AUPRC {v['auprc']:.3f}" for mm, v in r.items()), flush=True)
    res["examples"] = [
        {"caseid": int(c), "risk_t": V._r(te["t"][te["caseid"] == c], 1),
         "risk": {str(h): V._r(1 / (1 + np.exp(-logit[te["caseid"] == c, k])))
                  for k, h in enumerate(labels)}}
        for c in args.example_cases if (te["caseid"] == c).any()]
    (out / "result.json").write_text(json.dumps(res))
    if not args.eval_only:
        torch.save({"model": best_state, "mode": args.mode, "step": best_step,
                    "args": vars(args)}, out / "best.pt")
    print(f"wrote {out}/result.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
