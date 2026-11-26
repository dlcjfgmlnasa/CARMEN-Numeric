"""Downstream task definitions shared by the probe / fine-tune / fusion / token scripts.

A task turns the cases of a split into samples: ``caseid``, prediction time ``t`` (minutes
since anesthesia start, a monitor-row time; the numeric window ends there), labels ``y``
``(n, L)`` (NaN = no label), ``labels`` (names of the L outputs), hand-made features ``hand``
(signal-only, for the GBM baseline), a one-feature reference score ``simple`` and the
waveform window ends ``wave_t`` ``(n, K)`` in minutes:

* ``ioh``      — intra-operative hypotension within 5 / 10 / 15 min, a prediction every 5 min
  (``probe_ioh.build_samples``); one 60 s waveform window ending at ``t`` (``K = 1``, tokens
  are 5 s chunks of that window).
* ``aki``      — post-operative acute kidney injury (KDIGO creatinine: max creatinine within
  7 days after surgery >= 1.5 x pre-operative, or +0.3 mg/dL within 48 h); case level.
* ``mortality`` — in-hospital death; case level.
* ``icu``      — ICU admission after surgery (``icu_days > 0``); case level.
* ``arrhythmia`` — heart rhythm during a minute (VitalDB Arrhythmia Database, PhysioNet
  ``vitaldb-arrhythmia``: R-peak times with anesthesiologist-validated rhythm labels; 482
  cases, ``--arr-dir``). Rhythm classes as in CARMEN's downstream task: AF (AFIB/AFL), VA
  (VT, patterned ventricular ectopy), BradyCond (AVB, SND). One sample per monitor row
  ``k`` whose 60 s (``[start + 60 k, start + 60 (k + 1))`` of the .vital timeline, ``start``
  = anesthesia start) are fully annotated: >= 20 beats, no gap > 5 s (no noise /
  unannotated stretch), no unclassifiable beat. Four one-vs-rest labels: ``NSR`` = every
  beat of the minute is normal sinus rhythm; ``AF`` / ``VA`` / ``BradyCond`` = the minute
  contains that rhythm. (Presence rather than CARMEN's "pure 30 s segment": VT runs last a
  median 1.5 s and ventricular ectopy runs 13 s, so pure minutes of VA hardly exist.)
  Beats flagged bad-signal-quality are kept — their rhythm label stays valid (dataset
  README). The numeric window ends with that row and the waveform window is the same
  60 s, so neither sees anything after the minute.

Case-level samples predict at the end of the anesthesia record (last monitor row): the
numeric window is the last rows of the case and ``K = 12`` waveform windows of 60 s are
spread evenly over the case (one token per signal and window). Only signal-derived inputs
are used; the labels come from ``clinical_data.csv`` / ``lab_data.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import probe_ioh as P

CASE_TASKS = ("aki", "mortality", "icu")
TASKS = ("ioh", *CASE_TASKS, "arrhythmia")
ARR_CLASSES = ["NSR", "AF", "VA", "BradyCond"]
ARR_COLLAPSE = {"N": 0, "AFIB/AFL": 1, "VT": 2, "Patterned Ventricular Ectopy": 2,
                "AVB": 3, "SND": 3}
ARR_MIN_BEATS, ARR_MAX_GAP, ARR_MIN_HISTORY = 20, 5.0, 3.0
ARR_HAND = ["t", "hr_last", "hr_mean5", "hr_std5", "hr_min5", "hr_max5", "hr_std15",
            "hr_slope15", "spo2_last", "map_last", "map_std5", "pp_last", "etco2_last"]
N_SNAP = 12
CASE_HAND = ["duration", "map_mean", "map_min", "hypo_min", "hypo_frac", "map_auc65",
             "hr_mean", "hr_max", "spo2_min", "desat_min", "etco2_mean", "bt_min", "bt_mean",
             "tv_mean", "has_art"]


def _case_labels(task: str, caseids: np.ndarray, cases_csv: str | Path,
                 labs_csv: str | Path | None) -> np.ndarray:
    import pandas as pd

    clin = pd.read_csv(cases_csv).set_index("caseid")
    clin = clin.reindex(caseids)
    if task == "mortality":
        return clin.death_inhosp.to_numpy(dtype=float)
    if task == "icu":
        return (clin.icu_days.to_numpy(dtype=float) > 0).astype(float)
    if task == "aki":
        if labs_csv is None:
            raise ValueError("aki needs --labs-csv")
        lab = pd.read_csv(labs_csv)
        cr = lab[lab.name == "cr"].copy()
        cr["result"] = pd.to_numeric(cr.result, errors="coerce")
        cr = cr.join(clin[["opend"]], on="caseid", how="inner")
        after = cr.dt - cr.opend
        mx7 = cr[(after > 0) & (after <= 7 * 86400)].groupby("caseid").result.max()
        mx48 = cr[(after > 0) & (after <= 2 * 86400)].groupby("caseid").result.max()
        base = clin.preop_cr.to_numpy(dtype=float)
        m7 = mx7.reindex(caseids).to_numpy(dtype=float)
        m48 = mx48.reindex(caseids).to_numpy(dtype=float)
        aki = (m7 >= 1.5 * base) | ((m48 - base) >= 0.3)
        ok = np.isfinite(base) & np.isfinite(m7)
        return np.where(ok, aki.astype(float), np.nan)
    raise ValueError(task)


def _case_hand(v: np.ndarray, col: dict[str, int]) -> list[float]:
    """Signal-only summary of a whole case (monitor rows)."""
    def f(name):
        return v[:, col[name]].astype(np.float64) if name in col else np.full(len(v), np.nan)

    tm = f("t_min")
    art, nibp = f("art_mbp"), f("nibp_mbp")
    has_art = np.isfinite(art).sum() >= 10
    mp = art if has_art else nibp
    obs = np.isfinite(mp)
    hypo = (mp < P.HYPO) & obs
    spo2 = f("spo2")
    nan = float("nan")

    def stat(fn, x):
        return float(fn(x[np.isfinite(x)])) if np.isfinite(x).any() else nan

    return [
        float(tm[-1] - tm[0]) if len(tm) else nan,
        stat(np.mean, mp), stat(np.min, mp), float(hypo.sum()),
        float(hypo.sum() / max(obs.sum(), 1)),
        float(np.clip(P.HYPO - mp[obs], 0, None).sum()),
        stat(np.mean, f("hr")), stat(np.max, f("hr")), stat(np.min, spo2),
        float(((spo2 < 90) & np.isfinite(spo2)).sum()),
        stat(np.mean, f("etco2")), stat(np.min, f("bt")), stat(np.mean, f("bt")),
        stat(np.mean, f("tv")), float(has_art),
    ]


def _arr_windows(ann_csv: Path, starts: np.ndarray) -> np.ndarray:
    """Labels ``(len(starts), 4)`` of the 60 s windows starting at ``starts`` (s); NaN rows =
    window not fully annotated."""
    import pandas as pd

    a = pd.read_csv(ann_csv, usecols=["time_second", "rhythm_label"])
    tb = a.time_second.to_numpy(dtype=np.float64)
    order = np.argsort(tb)
    tb = tb[order]
    lab = a.rhythm_label.fillna("").to_numpy()[order]
    cls = a.rhythm_label.map(ARR_COLLAPSE).to_numpy(dtype=np.float64)[order]  # NaN = other
    out = np.full((len(starts), len(ARR_CLASSES)), np.nan)
    for j, s0 in enumerate(starts):
        lo, hi = np.searchsorted(tb, s0), np.searchsorted(tb, s0 + 60.0)
        if hi - lo < ARR_MIN_BEATS:
            continue
        gaps = np.diff(np.concatenate([[s0], tb[lo:hi], [s0 + 60.0]]))
        if gaps.max() > ARR_MAX_GAP or (lab[lo:hi] == "Unclassifiable").any():
            continue
        c = cls[lo:hi]
        out[j] = [float((lab[lo:hi] == "N").all()), float((c == 1).any()),
                  float((c == 2).any()), float((c == 3).any())]
    return out


def _arr_hand(tm: np.ndarray, get: dict[str, np.ndarray], i: int) -> list[float]:
    """Signal-only numeric summary up to monitor row ``i`` (no waveform)."""
    def w(f: str, minutes: float) -> np.ndarray:
        lo = np.searchsorted(tm, tm[i] - minutes, side="right")
        return get[f][lo: i + 1] if f in get else np.full(i + 1 - lo, np.nan)

    def stat(fn, x):
        return float(fn(x[~np.isnan(x)])) if (~np.isnan(x)).any() else np.nan

    hr5, hr15 = w("hr", 5), w("hr", 15)
    art = get.get("art_mbp")
    use_art = art is not None and (~np.isnan(art[: i + 1])).sum() >= 3
    mp = "art_mbp" if use_art else "nibp_mbp"
    sbp, dbp = ("art_sbp", "art_dbp") if use_art else ("nibp_sbp", "nibp_dbp")
    return [
        tm[i], P._last(hr5), P._mean(hr5), stat(np.std, hr5), stat(np.min, hr5),
        stat(np.max, hr5), stat(np.std, hr15), P._slope(tm[i + 1 - len(hr15): i + 1], hr15),
        P._last(w("spo2", 5)), P._last(w(mp, 15)), stat(np.std, w(mp, 5)),
        P._last(w(sbp, 15)) - P._last(w(dbp, 15)), P._last(w("etco2", 5)),
    ]


def _build_arrhythmia(cases: dict[int, dict], schema, cases_csv: str | Path,
                      arr_dir: str | Path) -> dict:
    import pandas as pd

    ann = Path(arr_dir)
    ann = ann / "Annotation_Files" if (ann / "Annotation_Files").is_dir() else ann
    clin = pd.read_csv(cases_csv, usecols=["caseid", "anestart"]).set_index("caseid").anestart
    mon, col = P.monitor_columns(schema)
    fields = ("hr", "spo2", "art_sbp", "art_dbp", "art_mbp", "nibp_sbp", "nibp_dbp",
              "nibp_mbp", "etco2")
    cid_l, t_l, y_l, hand_l = [], [], [], []
    for cid in sorted(cases):
        f = ann / f"Annotation_file_{cid}.csv"
        if not f.exists():
            continue
        c = cases[cid]
        v = c["values"][c["row_type"] == mon]
        if v.size == 0:
            continue
        a = clin.get(cid, np.nan)
        start = int(max(0.0, a)) if np.isfinite(a) else 0  # as in vitaldb.anesthesia_window
        tm = v[:, col["t_min"]].astype(np.float64)
        rows = np.flatnonzero(tm >= ARR_MIN_HISTORY)
        if rows.size == 0:
            continue
        lab = _arr_windows(f, start + 60.0 * np.floor(tm[rows]))
        get = {g: v[:, col[g]].astype(np.float64) for g in fields if g in col}
        ok = ~np.isnan(lab[:, 0])
        for i, y in zip(rows[ok], lab[ok], strict=True):
            cid_l.append(cid)
            t_l.append(tm[i])
            y_l.append(y)
            hand_l.append(_arr_hand(tm, get, i))
    t = np.asarray(t_l, np.float64)
    return {"caseid": np.asarray(cid_l, np.int64), "t": t,
            "y": np.asarray(y_l, np.float64).reshape(-1, len(ARR_CLASSES)),
            "hand": np.asarray(hand_l, np.float64).reshape(-1, len(ARR_HAND)),
            "labels": list(ARR_CLASSES), "wave_t": t[:, None],
            "simple": ("hr_last", ARR_HAND.index("hr_last"), 1.0), "per_hour": 60.0}


def build_task(task: str, cases: dict[int, dict], schema, cases_csv: str | Path | None = None,
               labs_csv: str | Path | None = None, min_rows: int = 30,
               arr_dir: str | Path | None = None) -> dict:
    if task == "arrhythmia":
        if cases_csv is None or arr_dir is None:
            raise ValueError("arrhythmia needs --cases-csv and --arr-dir")
        return _build_arrhythmia(cases, schema, cases_csv, arr_dir)
    if task == "ioh":
        s = P.build_samples(cases, schema, [5, 10, 15], 5, 15.0)
        return {**s, "labels": ["5", "10", "15"], "wave_t": s["t"][:, None],
                "simple": ("map_only", 1, -1.0), "per_hour": 12.0}
    if task not in CASE_TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {TASKS}")
    if cases_csv is None:
        raise ValueError(f"{task} needs --cases-csv")
    mon, col = P.monitor_columns(schema)
    cid_l, t_l, hand_l, snap_l = [], [], [], []
    for cid in sorted(cases):
        c = cases[cid]
        v = c["values"][c["row_type"] == mon]
        if len(v) < min_rows:
            continue
        tm = v[:, col["t_min"]].astype(np.float64)
        cid_l.append(cid)
        t_l.append(tm[-1])
        hand_l.append(_case_hand(v, col))
        # K windows spread over the case; each must end at or before the last row
        snap_l.append(np.linspace(tm[0] + 1.0, tm[-1], N_SNAP))
    caseid = np.asarray(cid_l, np.int64)
    y = _case_labels(task, caseid, cases_csv, labs_csv)[:, None]
    keep = np.isfinite(y[:, 0])
    return {"caseid": caseid[keep], "t": np.asarray(t_l)[keep], "y": y[keep],
            "hand": np.asarray(hand_l, np.float64).reshape(-1, len(CASE_HAND))[keep],
            "labels": [task], "wave_t": np.asarray(snap_l).reshape(-1, N_SNAP)[keep],
            "simple": ("hypo_minutes", CASE_HAND.index("hypo_min"), 1.0), "per_hour": 1.0}
