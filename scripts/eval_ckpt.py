"""Evaluate pre-training checkpoints on VitalDB splits — or all of VitalDB — with both
masking regimes.

    python scripts/eval_ckpt.py --data $DATA --split test runs/a/best.pt runs/b/best.pt
    python scripts/eval_ckpt.py --data $DATA --split all  runs/a/best.pt     # every case

``--split all`` scores train, val and test separately and then pooled over all cases
(train patients were seen in pre-training, so their numbers are in-sample; val/test are
not). For each checkpoint and part: masked-token loss / accuracy and per-field MAE (model
vs. median, nearest-value and linear-interpolation baselines) under the UniTTab masks,
span masks (gap filling) and forecast masks (the last 15 rows of each window). Point
estimates use ``data.vitaldb.POINT_DECODE`` (predictive median for SpO2 and MAC,
expectation otherwise). ``--split heldout`` scores val and test only.

On a GPU the rows of every case sit on the device and windows are cut there, so the
host CPU (saturated on the shared node) is not in the loop. Writes
``<ckpt>.eval_<split>.json`` next to each checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carmen_numeric import (  # noqa: E402
    ForecastMaskConfig,
    Normalizer,
    SpanMaskConfig,
    TableSchema,
    UniTTab,
    UniTTabConfig,
)
from carmen_numeric.data.dataset import (  # noqa: E402
    GPUWindowBatcher,
    WindowDataset,
    collate,
    load_cases,
)
from carmen_numeric.data.vitaldb import POINT_DECODE  # noqa: E402
from carmen_numeric.evaluate import combine, evaluate, summarize_mae  # noqa: E402

FIELDS = ["hr", "spo2", "art_mbp", "nibp_mbp", "etco2", "rr", "bis", "bt", "tv", "pip",
          "fio2", "mac", "ppf_ce", "rftn_ce", "lab_hct", "lab_k", "lab_gluc", "lab_lac"]
SPAN = SpanMaskConfig(0.2, 5, 30, 0.5, keep_time=True)
FORE = ForecastMaskConfig(1.0, 15, 15)
# name -> evaluate() keyword arguments
REGIMES = (("unittab", {}), ("span", {"span": SPAN}), ("forecast", {"forecast": FORE}))


def batches_for(cases, norm, max_rows, ctx, device, batch_size, on_device):
    if on_device:
        b = GPUWindowBatcher(cases, norm, max_rows, ctx, device, keep_raw=True)
        it = b.eval_batches(batch_size)
        return it, it.num_windows
    ds = WindowDataset(cases, norm, max_rows, ctx, random_crop=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate), len(ds)


def report(title: str, res: dict) -> None:
    print(f"\n--- {title}")
    for name, r in res.items():
        print(f"  [{name}] windows={r['windows']} loss={r['loss']:.4f} acc_num={r['acc_num']:.3f}"
              f" acc_num_pm2={r['acc_num_pm2']:.3f} acc_cat={r['acc_cat']:.3f}")
        print(summarize_mae(r["mae"], FIELDS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val",
                    choices=["train", "val", "test", "all", "heldout"],
                    help="all = train/val/test + pooled; heldout = val/test + pooled")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu-data", action="store_true",
                    help="build windows on the CPU (DataLoader) even when a GPU is present")
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_device = device.type == "cuda" and not args.cpu_data
    data = Path(args.data)
    schema = TableSchema.load(data / "schema.json")
    norm = Normalizer.load(schema, data / "normalizer.json")
    split = json.loads((data / "split.json").read_text())
    ctx = (schema.row_type_id["static"],) if "static" in schema.row_type_id else ()
    parts = {"all": ["train", "val", "test"], "heldout": ["val", "test"]}.get(
        args.split, [args.split])
    amp = torch.bfloat16 if device.type == "cuda" else None

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = UniTTabConfig(**ck["config"])
        model = UniTTab(cfg)
        model.load_state_dict(ck["model"])
        model.to(device)
        print(f"\n=== {path}  (step {ck.get('step')}, split={args.split}, device={device}, "
              f"windows on {'device' if on_device else 'cpu'})", flush=True)
        results: dict[str, dict] = {}
        for part in parts:
            t0 = time.time()
            cases = load_cases(data, split[part])
            it, n_win = batches_for(cases, norm, cfg.max_rows, ctx, device, args.batch_size,
                                    on_device)
            res = {}
            for name, kw in REGIMES:
                # different seed per part so pooled masks are not the same pattern repeated
                res[name] = evaluate(model, it, norm, device, seed=args.seed + len(results),
                                     max_batches=args.max_batches, amp_dtype=amp,
                                     context_row_types=ctx, point_decode=POINT_DECODE, **kw)
            results[part] = res
            tag = " (in-sample: seen in pre-training)" if part == "train" else ""
            report(f"{part}: {len(cases)} cases, {n_win} windows, {time.time() - t0:.0f}s{tag}",
                   res)
            del it
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(parts) > 1:
            results["val+test"] = {name: combine([results[p][name] for p in ("val", "test")])
                                   for name, _ in REGIMES}
            report("val+test pooled (held-out patients)", results["val+test"])
        if "train" in parts:
            results["all"] = {name: combine([results[p][name] for p in parts])
                              for name, _ in REGIMES}
            report("all of VitalDB pooled", results["all"])
        out = Path(path).with_suffix(f".eval_{args.split}.json")
        out.write_text(json.dumps({"ckpt": str(path), "step": ck.get("step"),
                                   "point_decode": POINT_DECODE, **results}))
        print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
