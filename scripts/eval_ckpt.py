"""Evaluate pre-training checkpoints on a split with both masking regimes.

    python scripts/eval_ckpt.py --data $DATA --split test runs/a/best.pt runs/b/best.pt

For each checkpoint: masked-token loss / accuracy and per-field MAE (model vs. median and
nearest-value baselines) under the UniTTab masks and under span masks (gap filling).
Writes ``<ckpt>.eval_<split>.json`` next to each checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carmen_numeric import (  # noqa: E402
    Normalizer,
    SpanMaskConfig,
    TableSchema,
    UniTTab,
    UniTTabConfig,
)
from carmen_numeric.data.dataset import WindowDataset, collate, load_cases  # noqa: E402
from carmen_numeric.evaluate import evaluate, summarize_mae  # noqa: E402

FIELDS = ["hr", "spo2", "art_mbp", "nibp_mbp", "etco2", "bis", "bt", "tv", "fio2", "mac",
          "ppf_ce", "rftn_ce", "lab_hct", "lab_k", "lab_gluc", "lab_lac"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    schema = TableSchema.load(data / "schema.json")
    norm = Normalizer.load(schema, data / "normalizer.json")
    split = json.loads((data / "split.json").read_text())
    ctx = (schema.row_type_id["static"],) if "static" in schema.row_type_id else ()
    cases = load_cases(data, split[args.split])
    span = SpanMaskConfig(0.2, 5, 30, 0.5, keep_time=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = UniTTabConfig(**ck["config"])
        model = UniTTab(cfg)
        model.load_state_dict(ck["model"])
        model.to(device)
        ds = WindowDataset(cases, norm, cfg.max_rows, ctx, random_crop=False)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        amp = torch.bfloat16 if device.type == "cuda" else None
        res = {}
        for name, sp in (("unittab", None), ("span", span)):
            res[name] = evaluate(model, dl, norm, device, seed=args.seed,
                                 max_batches=args.max_batches, amp_dtype=amp, span=sp,
                                 context_row_types=ctx)
        print(f"\n=== {path}  (step {ck.get('step')}, {args.split}, {len(ds)} windows)")
        for name in res:
            r = res[name]
            print(f"  [{name}] loss={r['loss']:.4f} acc_num={r['acc_num']:.3f} "
                  f"acc_num_pm2={r['acc_num_pm2']:.3f} acc_cat={r['acc_cat']:.3f}")
            print(summarize_mae(r["mae"], FIELDS))
        out = Path(path).with_suffix(f".eval_{args.split}.json")
        out.write_text(json.dumps({"ckpt": str(path), "step": ck.get("step"), **res}))


if __name__ == "__main__":
    main()
