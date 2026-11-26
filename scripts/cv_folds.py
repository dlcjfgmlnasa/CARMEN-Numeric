"""Patient-level cross-validation folds for small labelled cohorts, and pooling of fold results.

    # 5 folds over the VitalDB Arrhythmia Database cases present in the cache
    python scripts/cv_folds.py make --arr-dir $ARR --label-data $SIG --k 5 --out $R/folds_arr
    # per fold f: train / fine-tune with --split $R/folds_arr/fold{f}.json, then
    #   compare_ioh.py ... --split $R/folds_arr/fold{f}.json --dump cmp_f{f}.npz
    python scripts/cv_folds.py pool cmp_f0.npz ... cmp_f4.npz --reference numeric_ft \
        --label-data $SIG --out compare_arr_cv.json

``make``: folds are stratified group k-fold (groups = ``subjectid``, so all cases of a patient
share a fold; strata = the rarest arrhythmia class a case contains). Fold ``f`` is the test
set of ``fold{f}.json``, fold ``f + 1`` its validation set, the rest train. ``pool``: the
out-of-fold test predictions of all folds are concatenated (every case is tested once) and
compared with ``compare_ioh.paired`` (patients resampled once for all methods). Scores are
pooled as they are: methods must output comparable scores across folds (logits / GBM
probabilities are); AUROC of a pooled score also reflects calibration drift between folds.
The output has the format of ``compare_ioh.py`` (one entry per label).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_ioh as C  # noqa: E402

# rarest first: a case is stratified by the rarest class it contains
ARR_STRATA = [("AVB", 4), ("SND", 4), ("VT", 3), ("Patterned Ventricular Ectopy", 3),
              ("AFIB/AFL", 2)]


def make(args) -> None:
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    meta = pd.read_csv(Path(args.arr_dir) / "metadata.csv")
    idx = pd.read_csv(Path(args.label_data) / "index.csv").set_index("caseid")
    meta = meta[meta.case_id.isin(idx.index)].reset_index(drop=True)

    def stratum(classes: str) -> int:
        have = {c.strip() for c in str(classes).split(",")}
        return next((s for name, s in ARR_STRATA if name in have), 1)

    y = meta.rhythm_classes.map(stratum).to_numpy()
    groups = idx.loc[meta.case_id, "subjectid"].to_numpy()
    folds = np.zeros(len(meta), int)
    cv = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    for f, (_, te) in enumerate(cv.split(np.zeros(len(meta)), y, groups)):
        folds[te] = f
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cid = meta.case_id.to_numpy()
    for f in range(args.k):
        v = (f + 1) % args.k
        split = {"test": sorted(int(c) for c in cid[folds == f]),
                 "val": sorted(int(c) for c in cid[folds == v]),
                 "train": sorted(int(c) for c in cid[(folds != f) & (folds != v)])}
        (out / f"fold{f}.json").write_text(json.dumps(split))
    # one split holding every case (token extraction: one pass over all cases)
    (out / "all.json").write_text(json.dumps({"train": sorted(int(c) for c in cid),
                                              "val": [], "test": []}))
    tab = pd.crosstab(folds, y)
    print(f"{len(meta)} cases ({len(set(groups))} patients) in {args.k} folds; strata per fold "
          f"(1 none, 2 AF, 3 VA, 4 brady/conduction):\n{tab}")


def pool(args) -> None:
    zs = [np.load(p) for p in args.dumps]
    labels = [str(h) for h in zs[0]["labels"]]
    subject = None
    if args.label_data:  # resample patients (subjectid) rather than cases
        import pandas as pd
        subject = pd.read_csv(Path(args.label_data) / "index.csv").set_index("caseid").subjectid
    res: dict = {"task": args.task, "reference": args.reference, "folds": len(zs),
                 "horizons": {}}
    for h in labels:
        methods = [k.split("__", 2)[2] for k in zs[0].files if k.startswith(f"score__{h}__")]
        for z in zs[1:]:
            have = {k.split("__", 2)[2] for k in z.files if k.startswith(f"score__{h}__")}
            methods = [m for m in methods if m in have]
        y = np.concatenate([z[f"y__{h}"] for z in zs])
        cid = np.concatenate([z[f"caseid__{h}"] for z in zs])
        per_fold = [set(z[f"caseid__{h}"].tolist()) for z in zs]
        if sum(map(len, per_fold)) != len(set().union(*per_fold)):
            raise ValueError("a case is in the test set of more than one fold")
        scores = {m: np.concatenate([z[f"score__{h}__{m}"] for z in zs]) for m in methods}
        groups = subject.reindex(cid).fillna(-1).to_numpy() if subject is not None else cid
        r = C.paired(y, groups, scores, args.reference, args.n_boot)
        if args.pair_prefix:  # "<model>:<cond>" vs "<pair_prefix>:<cond>" (same draws)
            for m in methods:
                cond = m.split(":", 1)[1] if ":" in m else None
                ref = f"{args.pair_prefix}:{cond}"
                if cond is None or m == ref or ref not in scores:
                    continue
                q = C.paired(y, groups, {m: scores[m], ref: scores[ref]}, ref, args.n_boot)[m]
                r[m].update({f"{k}_vs_{args.pair_prefix}": q[k] for k in
                             ("d_auroc", "d_auroc_ci", "d_auprc", "d_auprc_ci")})
        res["horizons"][h] = {"n": int(len(y)), "prevalence": float(y.mean()),
                              "n_cases_pos": int(len(np.unique(cid[y == 1]))), "methods": r}
        print(f"--- {h} (n={len(y)}, prevalence {y.mean():.3f}, "
              f"{res['horizons'][h]['n_cases_pos']} positive cases)", flush=True)
        for m, v in r.items():
            line = f"  {m:<22} AUROC {v['auroc']:.3f}  AUPRC {v['auprc']:.3f}"
            if "d_auroc" in v:
                line += (f"  dAUROC {v['d_auroc']:+.3f} [{v['d_auroc_ci'][0]:+.3f},"
                         f"{v['d_auroc_ci'][1]:+.3f}]  dAUPRC {v['d_auprc']:+.3f} "
                         f"[{v['d_auprc_ci'][0]:+.3f},{v['d_auprc_ci'][1]:+.3f}]")
            if f"d_auroc_vs_{args.pair_prefix}" in v:
                k = f"_vs_{args.pair_prefix}"
                line += (f"  | vs {args.pair_prefix}: dAUROC {v['d_auroc' + k]:+.3f} "
                         f"[{v['d_auroc_ci' + k][0]:+.3f},{v['d_auroc_ci' + k][1]:+.3f}] "
                         f"dAUPRC {v['d_auprc' + k]:+.3f}")
            print(line, flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make", help="write fold<k>.json split files")
    mk.add_argument("--arr-dir", required=True, help="VitalDB Arrhythmia Database dir")
    mk.add_argument("--label-data", required=True, help="cache (index.csv: caseid, subjectid)")
    mk.add_argument("--k", type=int, default=5)
    mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--out", required=True)
    pl = sub.add_parser("pool", help="pool compare_ioh.py --dump files of all folds")
    pl.add_argument("dumps", nargs="+")
    pl.add_argument("--reference", default="numeric_ft")
    pl.add_argument("--label-data", default=None, help="resample patients via index.csv")
    pl.add_argument("--task", default="arrhythmia")
    pl.add_argument("--pair-prefix", default=None,
                    help="also compare every <model>:<cond> with <pair-prefix>:<cond>, e.g. "
                         "wave (adds d_auroc_vs_wave ...)")
    pl.add_argument("--n-boot", type=int, default=1000)
    pl.add_argument("--out", required=True)
    args = ap.parse_args()
    make(args) if args.cmd == "make" else pool(args)


if __name__ == "__main__":
    main()
