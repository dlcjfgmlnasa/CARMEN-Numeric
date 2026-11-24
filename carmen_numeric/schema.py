"""Table schema for time series of heterogeneous rows (UniTTab, Luetto et al. 2023).

A sequence is a list of rows. Each row has a *row type* ``h`` and the type fixes which
fields (attributes) the row carries and in which order (``A_h = {a_1..a_{k_h}}``).
A field is either numerical or categorical, and the same field may appear in several
row types (e.g. the time fields), in which case it shares its embedding and its
prediction head across those types.

Global field ids are 1-based: id 0 is reserved for an empty (padding) field slot.
Row type ids are 1-based as well: id 0 is a padding row.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Field:
    """One attribute.

    Parameters
    ----------
    name:
        Unique field name.
    kind:
        ``"num"`` (numerical) or ``"cat"`` (categorical).
    is_time:
        Time-stamp field. All time fields of a row are masked jointly during
        pre-training ("time stamp masking").
    valid_range:
        Numerical fields only: physiologically plausible ``(lo, hi)`` in the
        field's unit. Values outside are treated as artifacts (missing).
    norm_range:
        Numerical fields only: fixed ``(lo, hi)`` used to scale the value to
        [0, 1] before the frequency encoding. ``None`` means it is estimated from
        the training data (robust percentiles).
    unit:
        Free-text unit, for documentation only.
    """

    name: str
    kind: Literal["num", "cat"]
    is_time: bool = False
    valid_range: tuple[float, float] | None = None
    norm_range: tuple[float, float] | None = None
    unit: str = ""


@dataclass(frozen=True)
class RowType:
    """One row type: an ordered tuple of field names."""

    name: str
    fields: tuple[str, ...]


@dataclass
class TableSchema:
    """Fields and row types of a dataset."""

    fields: list[Field]
    row_types: list[RowType]
    # derived (not serialized)
    field_id: dict[str, int] = field(init=False, repr=False)
    row_type_id: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("field names must be unique")
        self.field_id = {f.name: i + 1 for i, f in enumerate(self.fields)}
        self.row_type_id = {rt.name: i + 1 for i, rt in enumerate(self.row_types)}
        for rt in self.row_types:
            for fname in rt.fields:
                if fname not in self.field_id:
                    raise ValueError(f"row type {rt.name!r} uses unknown field {fname!r}")
            if len(set(rt.fields)) != len(rt.fields):
                raise ValueError(f"row type {rt.name!r} repeats a field")

    # ── convenience ────────────────────────────────────────────────

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    @property
    def num_row_types(self) -> int:
        return len(self.row_types)

    @property
    def max_fields_per_row(self) -> int:
        return max(len(rt.fields) for rt in self.row_types)

    def get_field(self, name: str) -> Field:
        return self.fields[self.field_id[name] - 1]

    def row_field_ids(self, row_type: str | int) -> list[int]:
        """Global field ids (1-based) of the slots of a row type, in slot order."""
        rt = self.row_types[row_type - 1] if isinstance(row_type, int) else self.row_types[
            self.row_type_id[row_type] - 1
        ]
        return [self.field_id[f] for f in rt.fields]

    # ── (de)serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "fields": [asdict(f) for f in self.fields],
            "row_types": [{"name": rt.name, "fields": list(rt.fields)} for rt in self.row_types],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TableSchema:
        def _field(x: dict) -> Field:
            x = dict(x)
            for k in ("valid_range", "norm_range"):
                if x.get(k) is not None:
                    x[k] = tuple(x[k])
            return Field(**x)

        return cls(
            fields=[_field(x) for x in d["fields"]],
            row_types=[RowType(x["name"], tuple(x["fields"])) for x in d["row_types"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> TableSchema:
        return cls.from_dict(json.loads(Path(path).read_text()))
