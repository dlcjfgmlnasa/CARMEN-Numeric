"""Restrict a prepared row cache to a subset of fields (e.g. signal-derived values only).

The subset keeps the original bin edges and scaling of every kept field, so MAE of a
model trained on the subset is directly comparable with one trained on all fields.
Row types left without a non-time field (``static``, ``lab`` for the signal profile)
are dropped, rows whose kept values are all missing are dropped, and ``dt_min`` is
recomputed from ``t_min`` over the remaining rows.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from ..normalize import Normalizer
from ..schema import RowType, TableSchema

# Values measured from the patient by a sensor whose waveform CARMEN sees (or a cuff /
# temperature probe). Left out: anesthetic drugs (TCI Ce, MAC), BIS (EEG, not in CARMEN,
# drug-driven), clinician settings (FiO2, PEEP), labs and pre-operative data.
FIELD_PROFILES: dict[str, list[str]] = {
    "signal": [
        "hr",  # ECG
        "spo2",  # PPG
        "art_sbp", "art_dbp", "art_mbp",  # ABP
        "nibp_sbp", "nibp_dbp", "nibp_mbp",  # cuff
        "etco2", "rr",  # capnogram
        "bt",  # temperature probe
        "tv", "pip",  # airway pressure / flow
    ],
}


def subset_schema(schema: TableSchema, keep: list[str]) -> TableSchema:
    unknown = [k for k in keep if k not in schema.field_id]
    if unknown:
        raise ValueError(f"unknown fields: {unknown}")
    keep_set = set(keep) | {f.name for f in schema.fields if f.is_time}
    row_types = []
    for rt in schema.row_types:
        fields = tuple(f for f in rt.fields if f in keep_set)
        if any(not schema.get_field(f).is_time for f in fields):
            row_types.append(RowType(rt.name, fields))
    used = {f for rt in row_types for f in rt.fields}
    fields = [f for f in schema.fields if f.name in used]
    return TableSchema(fields=fields, row_types=row_types)


def subset_normalizer(norm: Normalizer, new: TableSchema) -> Normalizer:
    old = norm.schema
    out = Normalizer(schema=new)
    for f in new.fields:
        o, n = old.field_id[f.name], new.field_id[f.name]
        if o in norm.numeric:
            out.numeric[n] = norm.numeric[o]
        if o in norm.cat_sizes:
            out.cat_sizes[n] = norm.cat_sizes[o]
    return out


def subset_case(
    values: np.ndarray,  # (R, K_old) raw values, NaN = missing
    row_type: np.ndarray,  # (R,) old row-type ids
    old: TableSchema,
    new: TableSchema,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the kept row types, kept columns only; ``dt_min`` recomputed."""
    k_new = new.max_fields_per_row
    out_v, out_t, times = [], [], []
    for rt in new.row_types:
        o_id = old.row_type_id[rt.name]
        o_fields = old.row_types[o_id - 1].fields
        cols = [o_fields.index(f) for f in rt.fields]
        sel = row_type == o_id
        v = np.full((int(sel.sum()), k_new), np.nan, dtype=values.dtype)
        v[:, : len(cols)] = values[sel][:, cols]
        data = [j for j, f in enumerate(rt.fields) if not new.get_field(f).is_time]
        keep = ~np.isnan(v[:, data]).all(1)
        v = v[keep]
        out_v.append(v)
        out_t.append(np.full(len(v), new.row_type_id[rt.name], dtype=row_type.dtype))
        times.append(v[:, rt.fields.index("t_min")] if "t_min" in rt.fields
                     else np.full(len(v), -np.inf))
    v, t, tm = np.concatenate(out_v), np.concatenate(out_t), np.concatenate(times)
    order = np.argsort(tm, kind="stable")  # time-less (context) rows first
    v, t, tm = v[order], t[order], tm[order]
    dyn = np.isfinite(tm)
    if dyn.any():
        dt = np.diff(tm[dyn], prepend=tm[dyn][0])
        for rt in new.row_types:
            if "dt_min" in rt.fields:
                rows = np.flatnonzero(dyn & (t == new.row_type_id[rt.name]))
                pos = np.searchsorted(np.flatnonzero(dyn), rows)
                v[rows, rt.fields.index("dt_min")] = dt[pos]
    return v, t


def derive_subset(src: str | Path, dst: str | Path, keep: list[str]) -> dict:
    """Write a new cache under ``dst`` with only ``keep`` (plus time) fields."""
    src, dst = Path(src), Path(dst)
    old = TableSchema.load(src / "schema.json")
    new = subset_schema(old, keep)
    norm = subset_normalizer(Normalizer.load(old, src / "normalizer.json"), new)
    (dst / "cases").mkdir(parents=True, exist_ok=True)
    new.save(dst / "schema.json")
    norm.save(dst / "normalizer.json")
    for name in ("split.json", "index.csv"):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    n_rows = [0, 0]
    files = sorted((src / "cases").glob("*.npz"), key=lambda p: int(p.stem))
    for f in files:
        d = np.load(f)
        v, t = subset_case(d["values"], d["row_type"], old, new)
        n_rows[0] += len(d["row_type"])
        n_rows[1] += len(t)
        extra = {k: d[k] for k in d.files if k not in ("values", "row_type")}
        np.savez_compressed(dst / "cases" / f.name, values=v, row_type=t, **extra)
    info = {"src": str(src), "fields": keep, "cases": len(files),
            "rows_before": n_rows[0], "rows_after": n_rows[1]}
    (dst / "subset.json").write_text(json.dumps(info, indent=2))
    return info
