"""Fine-tune the encoder end-to-end for intra-operative hypotension (arterial MAP < 65 mmHg
within 5 / 10 / 15 minutes).

    python scripts/finetune_ioh.py --data $SIG --label-data $SIG \
        --ckpt runs/unittab_signal_v1/best.pt --out runs/ft_ioh_signal
    python scripts/finetune_ioh.py ... --scratch --out runs/ft_ioh_scratch  # random init

Samples, labels and splits are those of ``probe_ioh.py`` (a prediction every 5 minutes
where the arterial MAP of the last 5 minutes is >= 65; windows end at the prediction time,
nothing from the future). The encoder output ``[CLS, last row, mean of rows]`` feeds a small
MLP with one logit per horizon (masked BCE where a horizon has no label). Encoder and head
are trained together; the step with the best mean val AUROC is kept and scored on test
patients, next to the current-MAP and hand-made-feature baselines on the same samples.
``--example-cases`` also writes the risk over time of those test cases.
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
import probe_ioh as P  # noqa: E402
import viz_predictions as V  # noqa: E402

from carmen_numeric import Normalizer, TableSchema, UniTTab  # noqa: E402
from carmen_numeric.data.dataset import GPUWindowBatcher, load_cases  # noqa: E402

HORIZONS = [5, 10, 15]


class Windows:
    """Device-resident cases of one split + (case index, last row) of each sample."""

    def __init__(self, data: Path, caseids: list[int], case_of: np.ndarray, t: np.ndarray,
                 max_rows: int, device: torch.device) -> None:
        schema = TableSchema.load(data / "schema.json")
        norm = Normalizer.load(schema, data / "normalizer.json")
        ctx = (schema.row_type_id["static"],) if "static" in schema.row_type_id else ()
        cases = load_cases(data, sorted(set(caseids)))
        self.batcher = GPUWindowBatcher(cases, norm, max_rows, ctx, device)
        index = {cid: i for i, cid in enumerate(self.batcher.caseids)}
        dyn_t = {cid: c["values"][~np.isin(c["row_type"], np.asarray(ctx))][:, 0]
                 for cid, c in cases.items()}
        self.case = torch.as_tensor(np.array([index[c] for c in case_of]), device=device)
        end = np.array([np.searchsorted(dyn_t[c], tt, side="right") - 1
                        for c, tt in zip(case_of, t, strict=True)])
        if (end < 0).any():
            raise ValueError("a prediction time precedes every row of its case")
        self.end = torch.as_tensor(end, device=device)

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.batcher.windows_ending_at(self.case[idx], self.end[idx])


class IOHModel(nn.Module):
    def __init__(self, encoder: UniTTab, n_out: int, hidden: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        m = encoder.cfg.d_model
        self.head = nn.Sequential(nn.LayerNorm(3 * m), nn.Dropout(dropout),
                                  nn.Linear(3 * m, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # (B, n_out) logits
        z = self.encoder.encode(batch)
        rows, mask = z["rows"].float(), z["row_mask"]
        n = mask.sum(1).clamp(min=1)
        last = rows[torch.arange(len(n), device=rows.device), n - 1]
        mean = (rows * mask[..., None]).sum(1) / n[:, None]
        return self.head(torch.cat([z["cls"].float(), last, mean], 1))


@torch.no_grad()
def predict(model: IOHModel, win: Windows, n: int, device: torch.device,
            batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(win.batch(idx)).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def mean_auroc(y: np.ndarray, logit: np.ndarray) -> float:
    vals = []
    for k in range(y.shape[1]):
        ok = ~np.isnan(y[:, k])
        vals.append(P.auroc(y[ok, k], logit[ok, k]))
    return float(np.nanmean(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="cache the encoder was trained on")
    ap.add_argument("--label-data", required=True, help="cache with art_mbp (signal cache)")
    ap.add_argument("--ckpt", required=True, help="pre-trained checkpoint (config + weights)")
    ap.add_argument("--scratch", action="store_true", help="same architecture, random init")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=6, help="evals without val improvement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=None, help="per split (smoke tests)")
    ap.add_argument("--example-cases", type=int, nargs="*", default=[])
    ap.add_argument("--task", default="ioh", choices=list(T.TASKS))
    ap.add_argument("--cases-csv", default=None, help="clinical_data.csv (case-level tasks)")
    ap.add_argument("--labs-csv", default=None, help="lab_data.csv (aki)")
    ap.add_argument("--arr-dir", default=None,
                    help="VitalDB Arrhythmia Database dir (task arrhythmia)")
    ap.add_argument("--split", default=None,
                    help="split json (default: <label-data>/split.json), e.g. a CV fold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, label_data = Path(args.data), Path(args.label_data)
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
    print(f"samples {counts} ({time.time() - t0:.0f}s)", flush=True)

    encoder = P.load_model(args.ckpt, device, random_init=args.scratch)
    win = {p: Windows(data, list(samples[p]["caseid"]), samples[p]["caseid"], samples[p]["t"],
                      encoder.cfg.max_rows, device) for p in P.PARTS}
    y = {p: torch.as_tensor(samples[p]["y"], dtype=torch.float32, device=device)
         for p in P.PARTS}
    model = IOHModel(encoder, len(labels)).to(device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr_enc},
        {"params": model.head.parameters(), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    base_lr = [g["lr"] for g in opt.param_groups]
    gen = torch.Generator(device=device).manual_seed(args.seed)
    n_tr = counts["train"]
    best, best_state, best_step, bad = -1.0, None, 0, 0
    log = []
    model.train()
    for step in range(1, args.steps + 1):
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g, lr in zip(opt.param_groups, base_lr, strict=True):
            g["lr"] = lr * max(f, 0.05)
        idx = torch.randint(n_tr, (args.batch_size,), device=device, generator=gen)
        yy = y["train"][idx]
        ok = ~torch.isnan(yy)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(win["train"].batch(idx)).float()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logit[ok], torch.nan_to_num(yy)[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            va = mean_auroc(samples["val"]["y"],
                            predict(model, win["val"], counts["val"], device, 512))
            log.append({"step": step, "loss": round(loss.item(), 4), "val_auroc": round(va, 4)})
            print(json.dumps(log[-1]) + f" ({time.time() - t0:.0f}s)", flush=True)
            if va > best:
                best, best_step, bad = va, step, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in
                                            model.state_dict().items()})
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at {step}", flush=True)
                    break
    model.load_state_dict(best_state)
    logit = predict(model, win["test"], counts["test"], device, 512)

    res: dict = {"task": args.task, "ckpt": args.ckpt, "scratch": args.scratch,
                 "best_step": best_step, "best_val_auroc": round(best, 4), "log": log,
                 "counts": counts,
                 "args": vars(args), "horizons": {}}
    te = samples["test"]
    for k, h in enumerate(labels):
        ok = {p: ~np.isnan(samples[p]["y"][:, k]) for p in P.PARTS}
        yt = te["y"][ok["test"], k]
        g = te["caseid"][ok["test"]]
        hand = {p: samples[p]["hand"][ok[p]] for p in P.PARTS}
        scores = {
            "finetuned": 1 / (1 + np.exp(-logit[ok["test"], k])),
            "hand_gbm": P.gbm_probe(hand["train"], samples["train"]["y"][ok["train"], k],
                                    hand["test"]),
            simple_name: simple_sign * hand["test"][:, simple_col],
        }
        res["horizons"][str(h)] = {
            "n": int(len(yt)), "prevalence": round(float(yt.mean()), 4),
            "methods": {m: {**P.bootstrap(yt, s, g), **V.curve_points(yt, s),
                            "operating": V.operating_points(yt, s,
                                                            per_hour=te["per_hour"])}
                        for m, s in scores.items()},
            "calibration": {m: V.calibration(yt, scores[m]) for m in ("finetuned", "hand_gbm")},
        }
        r = res["horizons"][str(h)]["methods"]
        print(f"--- {h}: " + "  ".join(
            f"{m} {v['auroc']:.3f} [{v['auroc_ci'][0]:.3f},{v['auroc_ci'][1]:.3f}] "
            f"AUPRC {v['auprc']:.3f}" for m, v in r.items()), flush=True)

    # risk over time of a few test cases (every minute with an arterial MAP in the last 5 min)
    examples = []
    mon, col = P.monitor_columns(lschema)
    for cid in args.example_cases if args.task == "ioh" else []:
        c = load_cases(label_data, [cid]).get(cid)
        if c is None:
            continue
        v = c["values"][c["row_type"] == mon]
        tm = v[:, col["t_min"]].astype(np.float64)
        mp = v[:, col["art_mbp"]].astype(np.float64)
        rows = [i for i in range(len(tm)) if tm[i] >= 15 and
                (~np.isnan(mp[np.searchsorted(tm, tm[i] - 5, side="right"): i + 1])).any()]
        w = Windows(data, [cid], np.full(len(rows), cid), tm[rows], encoder.cfg.max_rows, device)
        lg = predict(model, w, len(rows), device, 512)
        examples.append({"caseid": cid, "risk_t": V._r(tm[rows], 1),
                         "risk": {str(h): V._r(1 / (1 + np.exp(-lg[:, k])))
                                  for k, h in enumerate(labels)}})
    res["examples"] = examples
    (out / "result.json").write_text(json.dumps(res))
    torch.save({"model": best_state, "config": encoder.cfg.to_dict(), "step": best_step,
                "args": vars(args)}, out / "best.pt")
    print(f"wrote {out}/result.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
