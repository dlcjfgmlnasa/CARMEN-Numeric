"""Paired comparison of hypotension predictors on the same test patients.

    python scripts/compare_ioh.py --data $SIG --label-data $SIG \
        --finetuned ft_signal runs/ft_ioh_signal/best.pt \
        --finetuned ft_scratch runs/ft_ioh_scratch/best.pt \
        --frozen frozen runs/unittab_signal_v1/best.pt --out runs/compare_ioh.json

Scores every method on the test samples of ``probe_ioh.py`` (current MAP, hand-made
features + GBM, frozen encoder + logistic regression, fine-tuned models from
``finetune_ioh.py``, precomputed logits such as ``finetune_fusion.py``'s
``test_scores.npz``) and bootstraps test patients once for all methods, so differences
in AUROC / AUPRC come with paired 95 % intervals (reference: ``--reference``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import finetune_ioh as F  # noqa: E402
import probe_ioh as P  # noqa: E402

from carmen_numeric import TableSchema, UniTTab, UniTTabConfig  # noqa: E402
from carmen_numeric.data.dataset import load_cases  # noqa: E402


def paired(y: np.ndarray, groups: np.ndarray, scores: dict[str, np.ndarray], ref: str,
           n: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    roc = {m: [] for m in scores}
    pr = {m: [] for m in scores}
    for _ in range(n):
        pick = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        for m, s in scores.items():
            roc[m].append(P.auroc(y[pick], s[pick]))
            pr[m].append(P.auprc(y[pick], s[pick]))
    out = {}
    for m, s in scores.items():
        r = {"auroc": P.auroc(y, s), "auprc": P.auprc(y, s),
             "auroc_ci": np.nanpercentile(roc[m], [2.5, 97.5]).tolist(),
             "auprc_ci": np.nanpercentile(pr[m], [2.5, 97.5]).tolist()}
        if m != ref:
            d_roc = np.asarray(roc[m]) - np.asarray(roc[ref])
            d_pr = np.asarray(pr[m]) - np.asarray(pr[ref])
            r.update({
                "d_auroc": r["auroc"] - P.auroc(y, scores[ref]),
                "d_auroc_ci": np.nanpercentile(d_roc, [2.5, 97.5]).tolist(),
                "p_auroc_le0": float(np.mean(d_roc <= 0)),
                "d_auprc": r["auprc"] - P.auprc(y, scores[ref]),
                "d_auprc_ci": np.nanpercentile(d_pr, [2.5, 97.5]).tolist(),
                "p_auprc_le0": float(np.mean(d_pr <= 0)),
            })
        out[m] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-data", required=True)
    ap.add_argument("--finetuned", nargs=2, action="append", default=[],
                    metavar=("NAME", "CKPT"))
    ap.add_argument("--scores", nargs=2, action="append", default=[],
                    metavar=("NAME", "NPZ"),
                    help="precomputed test logits (test_scores.npz of finetune_fusion.py)")
    ap.add_argument("--frozen", nargs=2, default=None, metavar=("NAME", "CKPT"),
                    help="pre-trained checkpoint for the frozen-encoder linear probe")
    ap.add_argument("--reference", default="hand_gbm")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, label_data = Path(args.data), Path(args.label_data)
    lschema = TableSchema.load(label_data / "schema.json")
    split = json.loads((label_data / "split.json").read_text())
    ids = {p: sorted(split[p])[: args.max_cases] for p in P.PARTS}
    samples = {p: P.build_samples(load_cases(label_data, ids[p]), lschema, F.HORIZONS, 5, 15.0)
               for p in P.PARTS}
    te = samples["test"]
    n_te = len(te["t"])

    logits: dict[str, np.ndarray] = {}  # (n_test, H)
    for name, ckpt in args.finetuned:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = F.IOHModel(UniTTab(UniTTabConfig(**ck["config"])), len(F.HORIZONS))
        model.load_state_dict(ck["model"])
        model.to(device)
        win = F.Windows(data, list(te["caseid"]), te["caseid"], te["t"], model.encoder.cfg.max_rows,
                        device)
        logits[name] = F.predict(model, win, n_te, device, 512)
        del model, win
    for name, path in args.scores:
        z = np.load(path)
        if not (np.array_equal(z["caseid"], te["caseid"]) and np.allclose(z["t"], te["t"])):
            raise ValueError(f"{path}: test samples do not match")
        logits[name] = z["logit"]
    frozen = None
    if args.frozen:
        fname, fckpt = args.frozen
        enc = P.load_model(fckpt, device)
        frozen = (fname, {p: P.encoder_features(enc, data, ids[p], samples[p], device, 512)
                          for p in P.PARTS})
        del enc

    res: dict = {"reference": args.reference, "horizons": {}}
    for k, h in enumerate(F.HORIZONS):
        ok = {p: ~np.isnan(samples[p]["y"][:, k]) for p in P.PARTS}
        y = {p: samples[p]["y"][ok[p], k] for p in P.PARTS}
        hand = {p: samples[p]["hand"][ok[p]] for p in P.PARTS}
        scores = {"map_only": -hand["test"][:, 1],
                  "hand_gbm": P.gbm_probe(hand["train"], y["train"], hand["test"])}
        if frozen is not None:
            f = frozen[1]
            s, _ = P.linear_probe(f["train"][ok["train"]], y["train"], f["val"][ok["val"]],
                                  y["val"], f["test"][ok["test"]], device)
            scores[frozen[0]] = s
        for name, lg in logits.items():
            scores[name] = lg[ok["test"], k]
        r = paired(y["test"], te["caseid"][ok["test"]], scores, args.reference, args.n_boot)
        res["horizons"][str(h)] = {"n": int(len(y["test"])),
                                   "prevalence": float(y["test"].mean()), "methods": r}
        print(f"--- {h} min (n={len(y['test'])}, prevalence {y['test'].mean():.3f}); "
              f"differences vs {args.reference}", flush=True)
        for m, v in r.items():
            line = f"  {m:<16} AUROC {v['auroc']:.3f}  AUPRC {v['auprc']:.3f}"
            if "d_auroc" in v:
                line += (f"  dAUROC {v['d_auroc']:+.3f} [{v['d_auroc_ci'][0]:+.3f},"
                         f"{v['d_auroc_ci'][1]:+.3f}]  dAUPRC {v['d_auprc']:+.3f} "
                         f"[{v['d_auprc_ci'][0]:+.3f},{v['d_auprc_ci'][1]:+.3f}]")
            print(line, flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
