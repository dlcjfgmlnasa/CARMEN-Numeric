"""VitalDB numerical data -> heterogeneous UniTTab rows.

Three row types per case (anesthesia period only):

* ``static``  — one row: demographics, ASA, surgery context, pre-op labs
  (``clinical_data.csv`` / ``api.vitaldb.net/cases``).
* ``monitor`` — one row per minute: vital signs, ventilator, anesthetic depth and
  TCI effect-site concentrations, each the median of that minute's samples.
* ``lab``     — one row per intra-operative blood draw (ABGA / electrolyte panel,
  ``lab_data.csv`` / ``api.vitaldb.net/labs``; ``dt`` counts from case start).

Dynamic rows carry two time fields: ``t_min`` (minutes since anesthesia start) and
``dt_min`` (minutes since the previous dynamic row). They are masked jointly.

Values outside a field's ``valid_range`` are artifacts and become missing. The ranges
only remove physically impossible readings; the physiological tails (hypotension,
desaturation, …) are kept on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Field, RowType, TableSchema

# ── field specification ────────────────────────────────────────────

TIME_FIELDS = [
    Field("t_min", "num", is_time=True, norm_range=(0.0, 720.0), unit="min"),
    Field("dt_min", "num", is_time=True, norm_range=(0.0, 60.0), unit="min"),
]

# name -> (VitalDB track, valid range, unit)
MONITOR_TRACKS: dict[str, tuple[str, tuple[float, float], str]] = {
    "hr": ("Solar8000/HR", (20, 250), "bpm"),
    "spo2": ("Solar8000/PLETH_SPO2", (50, 100), "%"),
    "art_sbp": ("Solar8000/ART_SBP", (20, 300), "mmHg"),
    "art_dbp": ("Solar8000/ART_DBP", (5, 225), "mmHg"),
    "art_mbp": ("Solar8000/ART_MBP", (10, 250), "mmHg"),
    "nibp_sbp": ("Solar8000/NIBP_SBP", (20, 300), "mmHg"),
    "nibp_dbp": ("Solar8000/NIBP_DBP", (5, 225), "mmHg"),
    "nibp_mbp": ("Solar8000/NIBP_MBP", (10, 250), "mmHg"),
    "etco2": ("Solar8000/ETCO2", (5, 100), "mmHg"),
    "rr": ("Solar8000/RR_CO2", (2, 60), "/min"),
    "bt": ("Solar8000/BT", (30, 42), "degC"),
    "bis": ("BIS/BIS", (1, 100), ""),
    "tv": ("Primus/TV", (20, 2000), "mL"),
    "peep": ("Primus/PEEP_MBAR", (0, 30), "mbar"),
    "pip": ("Primus/PIP_MBAR", (1, 80), "mbar"),
    "fio2": ("Primus/FIO2", (18, 100), "%"),
    "mac": ("Primus/MAC", (0, 5), ""),
    "ppf_ce": ("Orchestra/PPF20_CE", (0, 20), "mcg/mL"),
    "rftn_ce": ("Orchestra/RFTN20_CE", (0, 30), "ng/mL"),
}
BIS_SQI_TRACK = "BIS/SQI"  # BIS is kept only where SQI >= 50
BIS_MIN_SQI = 50.0

# lab name in lab_data.csv -> (valid range, unit)
LAB_FIELDS: dict[str, tuple[tuple[float, float], str]] = {
    "ph": ((6.5, 8.0), ""),
    "pco2": ((5, 150), "mmHg"),
    "po2": ((10, 700), "mmHg"),
    "hco3": ((2, 60), "mmol/L"),
    "sao2": ((20, 100), "%"),
    "hct": ((5, 75), "%"),
    "hb": ((2, 25), "g/dL"),
    "na": ((100, 180), "mmol/L"),
    "k": ((1.5, 10), "mmol/L"),
    "ica": ((0.2, 3.5), "mmol/L"),
    "gluc": ((10, 1000), "mg/dL"),
    "lac": ((0, 30), "mmol/L"),
}

# static numerical: column -> (valid range, unit)
STATIC_NUM: dict[str, tuple[tuple[float, float], str]] = {
    "age": ((0, 120), "y"),
    "height": ((40, 220), "cm"),
    "weight": ((2, 250), "kg"),
    "bmi": ((8, 80), "kg/m2"),
    "preop_hb": ((2, 25), "g/dL"),
    "preop_plt": ((1, 1500), "10^3/uL"),
    "preop_na": ((100, 180), "mmol/L"),
    "preop_k": ((1.5, 10), "mmol/L"),
    "preop_gluc": ((10, 1000), "mg/dL"),
    "preop_alb": ((0.5, 7), "g/dL"),
    "preop_cr": ((0.1, 20), "mg/dL"),
}
# static categorical: column -> ordered vocabulary (codes 0..V-1)
STATIC_CAT: dict[str, list] = {
    "sex": ["F", "M"],
    "asa": [1, 2, 3, 4, 5, 6],
    "emop": [0, 1],
    "department": ["General surgery", "Thoracic surgery", "Gynecology", "Urology"],
    "approach": ["Open", "Videoscopic", "Robotic"],
    "ane_type": ["General", "Spinal", "Sedationalgesia"],
    "preop_htn": [0, 1],
    "preop_dm": [0, 1],
}

LAB_PREFIX = "lab_"  # lab field names are prefixed so they never clash with monitor names


def vitaldb_schema() -> TableSchema:
    fields: list[Field] = list(TIME_FIELDS)
    fields += [Field(k, "num", valid_range=v[1], unit=v[2]) for k, v in MONITOR_TRACKS.items()]
    fields += [
        Field(LAB_PREFIX + k, "num", valid_range=v[0], unit=v[1]) for k, v in LAB_FIELDS.items()
    ]
    fields += [Field(k, "num", valid_range=v[0], unit=v[1]) for k, v in STATIC_NUM.items()]
    fields += [Field(k, "cat") for k in STATIC_CAT]
    row_types = [
        RowType("static", tuple(STATIC_NUM) + tuple(STATIC_CAT)),
        RowType("monitor", ("t_min", "dt_min") + tuple(MONITOR_TRACKS)),
        RowType("lab", ("t_min", "dt_min") + tuple(LAB_PREFIX + k for k in LAB_FIELDS)),
    ]
    return TableSchema(fields=fields, row_types=row_types)


def vitaldb_cat_sizes() -> dict[str, int]:
    return {k: len(v) for k, v in STATIC_CAT.items()}


# ── case builder ───────────────────────────────────────────────────


@dataclass
class CaseRows:
    caseid: int
    subjectid: int
    values: np.ndarray  # (R, K) float32, NaN = missing
    row_type: np.ndarray  # (R,) int8


def _clip_valid(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = x.astype(np.float64, copy=True)
    x[(x < lo) | (x > hi)] = np.nan
    return x


def anesthesia_window(case: pd.Series, n_sec: int) -> tuple[int, int]:
    """[start, end) in seconds of the .vital timeline, clipped to the recording."""
    start, end = case.get("anestart"), case.get("aneend")
    if not np.isfinite(start):
        start = 0
    if not np.isfinite(end) or end <= start:
        end = case.get("caseend", n_sec)
    start = int(max(0, start))
    end = int(min(n_sec, end))
    return start, end


def build_case(
    case: pd.Series,
    labs: pd.DataFrame,
    track_data: np.ndarray,  # (n_sec, len(MONITOR_TRACKS) + 1) at 1 s, last col = BIS SQI
    schema: TableSchema,
    min_monitor_rows: int = 10,
) -> CaseRows | None:
    """One case -> rows (static, then monitor/lab rows in time order)."""
    k_max = schema.max_fields_per_row
    rt_static = schema.row_type_id["static"]
    rt_mon = schema.row_type_id["monitor"]
    rt_lab = schema.row_type_id["lab"]

    n_sec = track_data.shape[0]
    start, end = anesthesia_window(case, n_sec)
    n_min = (end - start) // 60
    if n_min < min_monitor_rows:
        return None

    # ── monitor rows: per-minute median of 1 s samples ──
    names = list(MONITOR_TRACKS)
    seg = track_data[start : start + n_min * 60].astype(np.float64)
    sqi = seg[:, -1]
    mon = np.empty((n_min, len(names)))
    for j, name in enumerate(names):
        lo, hi = MONITOR_TRACKS[name][1]
        x = _clip_valid(seg[:, j], lo, hi)
        if name == "bis":
            x[~(sqi >= BIS_MIN_SQI)] = np.nan
        x = x.reshape(n_min, 60)
        with np.errstate(all="ignore"):
            cnt = np.isfinite(x).sum(1)
            med = np.nanmedian(np.where(cnt[:, None] > 0, x, 0.0), axis=1)
        mon[:, j] = np.where(cnt > 0, med, np.nan)
    keep = np.isfinite(mon).any(1)
    if keep.sum() < min_monitor_rows:
        return None
    mon_t = (np.arange(n_min) + 0.5)[keep]  # minute centers, minutes since start
    mon = mon[keep]

    # ── lab rows: one per draw time inside the window ──
    lab_names = list(LAB_FIELDS)
    lab_rows, lab_t = [], []
    if labs is not None and len(labs):
        lb = labs[(labs.dt >= start) & (labs.dt < end) & labs.name.isin(lab_names)]
        for dt, grp in lb.groupby("dt"):
            row = np.full(len(lab_names), np.nan)
            for name, val in zip(
                grp.name, pd.to_numeric(grp.result, errors="coerce"), strict=True
            ):
                lo, hi = LAB_FIELDS[name][0]
                if np.isfinite(val) and lo <= val <= hi:
                    row[lab_names.index(name)] = val
            if np.isfinite(row).any():
                lab_rows.append(row)
                lab_t.append((dt - start) / 60.0)

    # ── merge dynamic rows by time (lab after monitor on ties) ──
    t_all = np.concatenate([mon_t, np.asarray(lab_t, dtype=np.float64)])
    kind = np.concatenate([np.zeros(len(mon_t), int), np.ones(len(lab_t), int)])
    order = np.lexsort((kind, t_all))
    t_sorted = t_all[order]
    dt_sorted = np.diff(t_sorted, prepend=t_sorted[0])
    n_dyn = len(order)

    values = np.full((1 + n_dyn, k_max), np.nan, dtype=np.float32)
    row_type = np.empty(1 + n_dyn, dtype=np.int8)

    # static row (slot order = schema row type)
    row_type[0] = rt_static
    for j, fname in enumerate(schema.row_types[rt_static - 1].fields):
        raw = case.get(fname)
        if fname in STATIC_CAT:
            vocab = STATIC_CAT[fname]
            if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
                key = int(raw) if isinstance(vocab[0], int) else raw
                if key in vocab:
                    values[0, j] = vocab.index(key)
        else:
            lo, hi = STATIC_NUM[fname][0]
            v = pd.to_numeric(raw, errors="coerce")
            if np.isfinite(v) and lo <= v <= hi:
                values[0, j] = v

    for r, idx in enumerate(order, start=1):
        if kind[idx] == 0:
            row_type[r] = rt_mon
            values[r, 2 : 2 + len(names)] = mon[idx]
        else:
            row_type[r] = rt_lab
            values[r, 2 : 2 + len(lab_names)] = lab_rows[idx - len(mon_t)]
        values[r, 0] = t_sorted[r - 1]
        values[r, 1] = dt_sorted[r - 1]

    return CaseRows(
        caseid=int(case["caseid"]),
        subjectid=int(case.get("subjectid", case["caseid"])),
        values=values,
        row_type=row_type,
    )


def load_tracks(source: str | Path | int, n_extra_tracks: bool = True) -> np.ndarray | None:
    """Read the monitor tracks (+ BIS SQI) of one case at 1 s.

    ``source`` is a local ``.vital`` path or a VitalDB case id (downloaded).
    """
    import vitaldb

    tracks = [v[0] for v in MONITOR_TRACKS.values()] + [BIS_SQI_TRACK]
    if isinstance(source, (int, np.integer)):
        return vitaldb.load_case(int(source), tracks, 1)
    vf = vitaldb.VitalFile(str(source), tracks)
    return vf.to_numpy(tracks, 1)


def save_case(rows: CaseRows, out_dir: Path) -> None:
    np.savez_compressed(
        out_dir / f"{rows.caseid}.npz",
        values=rows.values,
        row_type=rows.row_type,
        caseid=rows.caseid,
        subjectid=rows.subjectid,
    )
