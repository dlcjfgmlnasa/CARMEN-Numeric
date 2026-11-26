"""Hand-made waveform features for the arrhythmia task, and a cross-validated GBM comparison of
waveform-only vs waveform + numeric features (no foundation model, CPU only).

    # 1) features of every labelled minute (one pass over all cases; run where the data is)
    python scripts/wave_features.py extract --arr-dir $ARR --label-data $SIG --cases-csv $CL \
        --vital-dir $BASE/vitaldb/raw --split $R/folds_arr/all.json --out $R/runs/arr_wfeat.npz
    # 2) GBMs per CV fold (writes <out-prefix><k>.npz in the compare_ioh.py --dump format)
    python scripts/wave_features.py cv --feats $R/runs/arr_wfeat.npz --folds $R/folds_arr \
        --out-prefix $R/runs/arr_cmp_wf_f
    python scripts/cv_folds.py pool $R/runs/arr_cmp_wf_f{0,1,2,3,4}.npz --reference hand_gbm \
        --pair-prefix wf --label-data $SIG --out $R/runs/arr_wf_cv.json

The waveform window of a sample is the 60 s of its minute row, read and preprocessed exactly
as for the CARMEN tokens (``extract_carmen_tokens.case_windows``: 100 Hz, ECG band-pass,
ABP / PPG low-pass; a signal is absent when > 30 % of the window is missing or flat, and ABP
also when its 5-95 % range is < 10 mmHg — an unconnected line reads ~20-30 mmHg). Beats
are detected without the annotation: ECG by the peaks of a 150 ms energy envelope of the
derivative (Pan-Tompkins style), ABP / PPG by systolic peaks. Per signal: interval statistics
(rate, CV, RMSSD, pNN50 / pNN20, lag-1 autocorrelation of intervals, fractions of short /
long intervals, longest pause, histogram entropy) and beat morphology (correlation with the
median beat, residual energy, amplitude CV); ECG adds QRS-envelope width and the
beat-to-beat consistency / energy of the P-wave region; pulse deficits compare ECG beats with
ABP / PPG pulses. An absent signal gives NaN features (the GBM handles missing values).

Methods per waveform condition ``c`` (e, p, a, ep, eap): ``wf:c`` = features of the signals
in ``c``; ``wfnum:c`` = the same plus the numeric features of ``hand_gbm``
(``downstream_tasks.ARR_HAND``: up to 15 min of monitor history). All use the GBM of
``hand_gbm`` (``probe_ioh.gbm_probe``) fitted on the fold's train part, so ``hand_gbm`` here
equals the one of ``compare_ioh.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import downstream_tasks as T  # noqa: E402
import extract_carmen_tokens as E  # noqa: E402
import probe_ioh as P  # noqa: E402

FS = E.TARGET_SR
SIG = ("e", "a", "p")  # slot order of extract_carmen_tokens.SIGNALS: ECG, ABP, PPG
INTERVAL = ["n", "rate", "cv", "rmssd_n", "pnn50", "pnn20", "ac1", "short", "long", "pause",
            "maxmin", "sd_med", "entropy"]
MORPH = ["corr_mean", "corr_p10", "corr_low", "resid", "amp_cv"]
ECG_EXTRA = ["qrs_w_med", "qrs_w_p90", "qrs_wide", "p_corr", "p_energy"]
CROSS = ["deficit_ea", "deficit_ep"]
NAMES = ([f"{s}_{n}" for s in SIG for n in INTERVAL + MORPH]
         + [f"e_{n}" for n in ECG_EXTRA] + CROSS)
CONDS = ("eap", "ep", "e", "p", "a")
SEG = {"e": (25, 40), "a": (20, 30), "p": (20, 30)}  # samples before / after a beat
ABP_MIN_PULSE = 10.0  # mmHg (95th - 5th percentile of the window); below: no arterial line


def ecg_beats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beat indices and QRS-envelope widths (s) of a 100 Hz band-passed ECG."""
    from scipy.signal import find_peaks

    d = np.diff(x, prepend=x[0])
    env = np.convolve(d * d, np.ones(15) / 15, mode="same")  # 150 ms integration
    pk, prop = find_peaks(env, height=0.3 * np.percentile(env, 98),
                          distance=int(0.27 * FS), width=1)
    return pk, prop["widths"] / FS


def pulse_beats(x: np.ndarray) -> np.ndarray:
    from scipy.signal import find_peaks

    lo, hi = np.percentile(x, [5, 95])
    pk, _ = find_peaks(x, distance=int(0.3 * FS), prominence=0.3 * max(hi - lo, 1e-6))
    return pk


def interval_feats(pk: np.ndarray, n_win: int) -> list[float]:
    out = [float(len(pk))] + [np.nan] * (len(INTERVAL) - 1)
    if len(pk) < 5:
        return out
    b = pk / FS
    r = np.diff(b)
    med, dr = np.median(r), np.diff(r)
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r[:-1].std() > 0 and r[1:].std() > 0 \
        else np.nan
    h, _ = np.histogram(r / med, bins=np.linspace(0.3, 2.0, 18))
    p = h[h > 0] / max(h.sum(), 1)
    return out[:1] + [
        60.0 / med, r.std() / r.mean(), np.sqrt(np.mean(dr ** 2)) / r.mean(),
        float(np.mean(np.abs(dr) > 0.05)), float(np.mean(np.abs(dr) > 0.02)), ac1,
        float(np.mean(r < 0.85 * med)), float(np.mean(r > 1.15 * med)),
        max(r.max(), b[0], n_win / FS - b[-1]), r.max() / r.min(),
        float(np.median(np.abs(dr))) / med, float(-(p * np.log(p)).sum()),
    ]


def _segments(x: np.ndarray, pk: np.ndarray, before: int, after: int) -> np.ndarray:
    pk = pk[(pk >= before) & (pk + after <= len(x))]
    if len(pk) < 5:
        return np.zeros((0, before + after))
    idx = pk[:, None] + np.arange(-before, after)[None]
    seg = x[idx]
    return seg - seg.mean(1, keepdims=True)


def _template_corr(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tpl = np.median(seg, 0)
    c = seg @ tpl / (np.linalg.norm(seg, axis=1) * np.linalg.norm(tpl) + 1e-12)
    return c, tpl


def morph_feats(x: np.ndarray, pk: np.ndarray, before: int, after: int) -> list[float]:
    seg = _segments(x, pk, before, after)
    if len(seg) == 0:
        return [np.nan] * len(MORPH)
    c, tpl = _template_corr(seg)
    amp = seg.max(1) - seg.min(1)
    return [float(c.mean()), float(np.percentile(c, 10)), float(np.mean(c < 0.8)),
            float(((seg - tpl) ** 2).mean() / ((tpl ** 2).mean() + 1e-12)),
            float(amp.std() / (amp.mean() + 1e-12))]


def ecg_extra(x: np.ndarray, pk: np.ndarray, width: np.ndarray) -> list[float]:
    if len(pk) < 5:
        return [np.nan] * len(ECG_EXTRA)
    wmed = float(np.median(width))
    p_seg = _segments(x, pk, 25, -8)  # P-wave region: 250 .. 80 ms before the QRS
    q_seg = _segments(x, pk, 8, 8)
    p_corr = p_en = np.nan
    if len(p_seg) and len(q_seg):
        c, tpl = _template_corr(p_seg)
        p_corr = float(c.mean())
        p_en = float((tpl ** 2).mean() / ((np.median(q_seg, 0) ** 2).mean() + 1e-12))
    return [wmed, float(np.percentile(width, 90)), float(np.mean(width > 1.5 * wmed)),
            p_corr, p_en]


def window_feats(wave: np.ndarray, present: np.ndarray) -> np.ndarray:
    """(3, N_WIN) 100 Hz window of ECG / ABP / PPG -> (len(NAMES),) features."""
    feats: dict[str, list[float]] = {}
    n_beats: dict[str, float] = {}
    for slot, s in enumerate(SIG):
        x = wave[slot].astype(np.float64)
        # an unconnected / zeroed arterial line passes the range check as ~20-30 mmHg noise
        flat_abp = s == "a" and np.subtract(*np.percentile(x, [95, 5])) < ABP_MIN_PULSE
        if not present[slot] or flat_abp:
            feats[s] = [np.nan] * (len(INTERVAL) + len(MORPH))
            continue
        if s == "e":
            pk, width = ecg_beats(x)
            feats["e_extra"] = ecg_extra(x, pk, width)
        else:
            pk = pulse_beats(x)
        feats[s] = interval_feats(pk, len(x)) + morph_feats(x, pk, *SEG[s])
        n_beats[s] = float(len(pk))
    cross = [(n_beats["e"] - n_beats[o]) / n_beats["e"]
             if "e" in n_beats and o in n_beats and n_beats["e"] > 0 else np.nan
             for o in ("a", "p")]
    out = feats["e"] + feats["a"] + feats["p"] + feats.get("e_extra", [np.nan] * 5) + cross
    return np.asarray(out, np.float32)


def _case_feats(job: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    caseid, ends, vital_dir = job
    _, wave, present = E.case_windows((caseid, ends, vital_dir, (0, 1, 2)))
    f = np.full((len(ends), len(NAMES)), np.nan, np.float32)
    for i in range(len(ends)):
        if present[i].any():
            f[i] = window_feats(wave[i], present[i])
    return caseid, f, present


def cond_columns(cond: str) -> np.ndarray:
    keep = [k for k, n in enumerate(NAMES)
            if (n.split("_", 1)[0] in cond if not n.startswith("deficit_")
                else set(n[-2:]) <= set(cond))]
    return np.asarray(keep)


def extract(args) -> None:
    import pandas as pd

    from carmen_numeric import TableSchema
    from carmen_numeric.data.dataset import load_cases

    label_data = Path(args.label_data)
    schema = TableSchema.load(label_data / "schema.json")
    ids = sorted(json.loads(Path(args.split).read_text())["train"])[: args.max_cases]
    s = T.build_task("arrhythmia", load_cases(label_data, ids), schema, args.cases_csv,
                     arr_dir=args.arr_dir)
    clin = pd.read_csv(args.cases_csv, usecols=["caseid", "anestart"]).set_index("caseid")
    n = len(s["t"])
    jobs, offsets = [], []
    for c in np.unique(s["caseid"]):  # samples are ordered by caseid
        m = np.flatnonzero(s["caseid"] == c)
        a = clin.anestart.get(c, np.nan)
        start = int(max(0.0, a)) if np.isfinite(a) else 0
        jobs.append((int(c), start + 60 * (np.floor(s["t"][m]).astype(int) + 1), args.vital_dir))
        offsets.append(m[0])
    feats = np.full((n, len(NAMES)), np.nan, np.float32)
    present = np.zeros((n, 3), bool)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, ((_, f, p), off) in enumerate(
                zip(pool.imap(_case_feats, jobs, chunksize=1), offsets, strict=True), start=1):
            feats[off: off + len(f)] = f
            present[off: off + len(f)] = p
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} cases ({time.time() - t0:.0f}s)", flush=True)
    hr = s["hand"][:, T.ARR_HAND.index("hr_last")]
    for sig in SIG:  # detector check: beat rate vs the monitor's HR
        rate = feats[:, NAMES.index(f"{sig}_rate")]
        ok = np.isfinite(rate) & np.isfinite(hr)
        if ok.any():
            err = np.abs(rate[ok] - hr[ok])
            print(f"{sig}: present {present[:, SIG.index(sig)].mean():.3f}; |rate - monitor HR| "
                  f"median {np.median(err):.1f} bpm, <= 5 bpm {np.mean(err <= 5):.3f} "
                  f"(n={ok.sum()})", flush=True)
    np.savez(args.out, caseid=s["caseid"], t=s["t"], y=s["y"], hand=s["hand"],
             labels=np.array(s["labels"]), hand_names=np.array(T.ARR_HAND), feats=feats,
             names=np.array(NAMES), present=present)
    print(f"wrote {args.out}: {n} samples, {len(NAMES)} features ({time.time() - t0:.0f}s)")


def cv(args) -> None:
    z = np.load(args.feats)
    caseid, t, y, hand, feats = z["caseid"], z["t"], z["y"], z["hand"], z["feats"]
    labels = [str(h) for h in z["labels"]]
    hr = list(z["hand_names"]).index("hr_last")
    folds = sorted(Path(args.folds).glob("fold*.json"))
    for fp in folds:
        split = json.loads(fp.read_text())
        tr, te = np.isin(caseid, split["train"]), np.isin(caseid, split["test"])
        dump: dict[str, np.ndarray] = {}
        for j, h in enumerate(labels):
            ok = ~np.isnan(y[:, j])
            a, b = tr & ok, te & ok
            scores = {"hr_last": hand[b, hr], "hand_gbm": P.gbm_probe(hand[a], y[a, j], hand[b])}
            for c in CONDS:
                xw = feats[:, cond_columns(c)]
                scores[f"wf:{c}"] = P.gbm_probe(xw[a], y[a, j], xw[b])
                xc = np.hstack([xw, hand])
                scores[f"wfnum:{c}"] = P.gbm_probe(xc[a], y[a, j], xc[b])
            dump[f"y__{h}"] = y[b, j]
            dump[f"caseid__{h}"] = caseid[b]
            dump[f"t__{h}"] = t[b]
            dump.update({f"score__{h}__{m}": np.asarray(v, np.float64) for m, v in scores.items()})
            line = "  ".join(f"{m} {P.auroc(y[b, j], v):.3f}" for m, v in scores.items())
            print(f"{fp.stem} {h} (test n={b.sum()}, pos {int(y[b, j].sum())}): {line}",
                  flush=True)
        out = f"{args.out_prefix}{fp.stem.removeprefix('fold')}.npz"
        np.savez(out, labels=np.array(labels), **dump)
        print(f"wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="waveform features of every labelled minute")
    ex.add_argument("--arr-dir", required=True)
    ex.add_argument("--label-data", required=True)
    ex.add_argument("--cases-csv", required=True)
    ex.add_argument("--vital-dir", required=True)
    ex.add_argument("--split", required=True, help="split json; its train part is used")
    ex.add_argument("--workers", type=int, default=4)
    ex.add_argument("--max-cases", type=int, default=None)
    ex.add_argument("--out", required=True)
    c = sub.add_parser("cv", help="GBMs per fold (compare_ioh.py --dump format)")
    c.add_argument("--feats", required=True)
    c.add_argument("--folds", required=True, help="dir of fold<k>.json (cv_folds.py make)")
    c.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else cv(args)


if __name__ == "__main__":
    main()
