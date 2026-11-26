"""Re-score a pooled comparison on the samples whose waveforms actually exist.

Forced test conditions (``wave:a`` = only ABP allowed) keep every sample, so a sample without
ABP gives the waveform model nothing while a fused model can fall back on numerics. This
script splits that apart: for each condition it keeps only the samples where every signal of
the condition is present (``<split>_meta.npz`` from ``extract_carmen_tokens.py``) and compares
``<method>:<cond>`` with ``<pair>:<cond>`` there (paired patient bootstrap).

    python scripts/present_subset.py --dumps runs/arr_cmp_f{0,1,2,3,4}.npz \
        --meta tokens_arr/train_meta.npz --label-data $SIG --out runs/arr_bidir_present.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SIGNAL_INDEX = {"e": 0, "a": 1, "p": 2}  # presence columns: ECG, ABP, PPG


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dumps", nargs="+", required=True, help="compare_ioh.py --dump files")
    ap.add_argument("--meta", required=True, help="token meta npz (caseid, t, present)")
    ap.add_argument("--label-data", required=True, help="cache with index.csv (subjectid)")
    ap.add_argument("--method", default="bidir")
    ap.add_argument("--pair", default="wave")
    ap.add_argument("--reference", default="numeric_ft")
    ap.add_argument("--conds", nargs="+", default=["eap", "ep", "e", "ap", "p", "a", "ea"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = np.load(args.meta)
    present = {(int(c), int(t)): p
               for c, t, p in zip(meta["caseid"], meta["t"], meta["present"], strict=True)}
    subj = pd.read_csv(Path(args.label_data) / "index.csv").set_index("caseid").subjectid
    zs = [np.load(p) for p in args.dumps]
    rng = np.random.default_rng(args.seed)
    out: dict = {}
    for h in [str(x) for x in zs[0]["labels"]]:
        y = np.concatenate([z[f"y__{h}"] for z in zs])
        cid = np.concatenate([z[f"caseid__{h}"] for z in zs])
        t = np.concatenate([z[f"t__{h}"] for z in zs])
        pres = np.array([present[(int(c), int(tt))]
                         for c, tt in zip(cid, t, strict=True)]).astype(bool)

        def score(m: str) -> np.ndarray:
            return np.concatenate([z[f"score__{h}__{m}"] for z in zs])  # noqa: B023

        g = subj.loc[cid].to_numpy()
        groups = np.unique(g)
        rows = {u: np.where(g == u)[0] for u in groups}
        boots = [np.concatenate([rows[u] for u in rng.choice(groups, len(groups))])
                 for _ in range(args.n_boot)]
        out[h] = {}
        for c in args.conds:
            keep = pres[:, [SIGNAL_INDEX[k] for k in c]].all(1)
            yy = y[keep]
            m_s, p_s = score(f"{args.method}:{c}")[keep], score(f"{args.pair}:{c}")[keep]
            r_s = score(args.reference)[keep]
            pos = {v: i for i, v in enumerate(np.where(keep)[0])}

            def metrics(s: np.ndarray, idx=slice(None)) -> list[float]:
                return [roc_auc_score(yy[idx], s[idx]), average_precision_score(yy[idx], s[idx])]  # noqa: B023

            r = {"n": int(keep.sum()), "pos": int(yy.sum()), args.method: metrics(m_s),
                 args.pair: metrics(p_s), args.reference: metrics(r_s)}
            d = []
            for b in boots:
                k = np.array([pos[i] for i in b if i in pos])
                if yy[k].min() == yy[k].max():
                    continue
                d.append(np.subtract(metrics(m_s, k), metrics(p_s, k)))
            r["d"] = list(np.subtract(r[args.method], r[args.pair]))
            r["d_ci"] = np.percentile(np.array(d), [2.5, 97.5], axis=0).T.tolist()
            out[h][c] = r
            print(f"{h:9s} {c:3s} n={r['n']:5d} pos={r['pos']:4d} {args.pair} "
                  f"{r[args.pair][0]:.3f}/{r[args.pair][1]:.3f} {args.method} "
                  f"{r[args.method][0]:.3f}/{r[args.method][1]:.3f}  dAUROC {r['d'][0]:+.3f} "
                  f"[{r['d_ci'][0][0]:+.3f},{r['d_ci'][0][1]:+.3f}] dAUPRC {r['d'][1]:+.3f} "
                  f"[{r['d_ci'][1][0]:+.3f},{r['d_ci'][1][1]:+.3f}]", flush=True)
    Path(args.out).write_text(json.dumps({"method": args.method, "pair": args.pair,
                                          "labels": out}, indent=1))


if __name__ == "__main__":
    main()
