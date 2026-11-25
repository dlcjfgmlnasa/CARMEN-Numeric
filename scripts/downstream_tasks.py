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
TASKS = ("ioh", *CASE_TASKS)
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


def build_task(task: str, cases: dict[int, dict], schema, cases_csv: str | Path | None = None,
               labs_csv: str | Path | None = None, min_rows: int = 30) -> dict:
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
