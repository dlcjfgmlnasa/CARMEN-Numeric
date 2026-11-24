"""Value scaling and quantization targets for numerical fields.

Input side (what the network sees): a numerical value ``v`` is scaled to ``[0, 1]`` with
a *global* (dataset-level, not per-patient) range and then frequency-encoded
(``encoding.FrequencyEncoding``). A global range keeps the absolute physiological level
in the representation, the same design choice as CARMEN's loc/scale conditioning.

Output side (what the network predicts for a masked value): following UniTTab, ``v`` is
quantized into an ordered vocabulary of bins and the bin index is the pseudo label of
a cross-entropy loss with neighborhood label smoothing. The quantized value is never fed
to the network.

The paper does not say how the input range or the bins are chosen. Here: the input
range is the [0.5, 99.5] percentile of the training data unless the field fixes
``norm_range``, and the bins are training-data quantiles (``num_bins``, duplicates
merged, so integer-valued fields can end up with fewer bins).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .schema import TableSchema


@dataclass
class NumericStats:
    norm_lo: float
    norm_hi: float
    edges: np.ndarray  # (n_bins + 1,) increasing
    centers: np.ndarray  # (n_bins,) mean training value inside each bin
    median: float = float("nan")  # training median (simple imputation baseline)

    @property
    def n_bins(self) -> int:
        return len(self.edges) - 1


@dataclass
class Normalizer:
    """Per-field statistics. Indexed by global field id (1-based)."""

    schema: TableSchema
    numeric: dict[int, NumericStats] = field(default_factory=dict)
    cat_sizes: dict[int, int] = field(default_factory=dict)

    # ── fitting ────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        schema: TableSchema,
        values_per_field: dict[int, np.ndarray],
        cat_sizes: dict[str, int],
        num_bins: int = 100,
        pct: tuple[float, float] = (0.5, 99.5),
    ) -> Normalizer:
        """Fit from training values.

        values_per_field:
            global field id -> 1-D array of observed (finite) raw values.
        cat_sizes:
            categorical field name -> vocabulary size (codes are 0..size-1).
        """
        norm = cls(schema=schema)
        for f in schema.fields:
            fid = schema.field_id[f.name]
            if f.kind == "cat":
                if f.name not in cat_sizes:
                    raise ValueError(f"missing vocabulary size for categorical {f.name!r}")
                norm.cat_sizes[fid] = int(cat_sizes[f.name])
                continue
            v = np.asarray(values_per_field.get(fid, np.empty(0)), dtype=np.float64)
            v = v[np.isfinite(v)]
            if v.size == 0:
                raise ValueError(f"no training values for numerical field {f.name!r}")
            if f.norm_range is not None:
                lo, hi = f.norm_range
            else:
                lo, hi = np.percentile(v, pct)
            if hi <= lo:
                hi = lo + 1.0
            edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, num_bins + 1)))
            if edges.size < 2:
                edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
            idx = _bin_index(v, edges)
            counts = np.bincount(idx, minlength=edges.size - 1)
            sums = np.bincount(idx, weights=v, minlength=edges.size - 1)
            mids = 0.5 * (edges[:-1] + edges[1:])
            centers = np.where(counts > 0, sums / np.maximum(counts, 1), mids)
            norm.numeric[fid] = NumericStats(
                float(lo), float(hi), edges, centers, float(np.median(v))
            )
        return norm

    # ── per-field output sizes (prediction vocabulary) ─────────────

    def vocab_sizes(self) -> list[int]:
        """Output vocabulary size per field, index 0 = padding (size 0)."""
        out = [0] * (self.schema.num_fields + 1)
        for fid, st in self.numeric.items():
            out[fid] = st.n_bins
        for fid, n in self.cat_sizes.items():
            out[fid] = n
        return out

    # ── transform ──────────────────────────────────────────────────

    def transform(self, values: np.ndarray, row_type: np.ndarray) -> dict[str, np.ndarray]:
        """Raw rows -> network inputs and pre-training targets.

        Parameters
        ----------
        values:
            ``(R, K)`` float, raw field values in slot order of each row's type;
            ``NaN`` = missing; slots beyond ``k_h`` are ignored. Categorical
            fields hold integer codes.
        row_type:
            ``(R,)`` int, 1-based row type ids.

        Returns
        -------
        dict of ``(R, K)`` arrays: ``field_id`` (int64, 0 = empty slot),
        ``num_value`` (float32 in [0, 1]), ``cat_value`` (int64), ``missing``
        (bool, real field without a value), ``target`` (int64, -1 where there is
        nothing to predict).
        """
        r, k = values.shape
        field_id = np.zeros((r, k), dtype=np.int64)
        num_value = np.zeros((r, k), dtype=np.float32)
        cat_value = np.zeros((r, k), dtype=np.int64)
        missing = np.zeros((r, k), dtype=bool)
        target = np.full((r, k), -1, dtype=np.int64)

        for h in np.unique(row_type):
            rows = np.nonzero(row_type == h)[0]
            for j, fid in enumerate(self.schema.row_field_ids(int(h))):
                v = values[rows, j].astype(np.float64)
                finite = np.isfinite(v)
                field_id[rows, j] = fid
                missing[rows, j] = ~finite
                if fid in self.numeric:
                    st = self.numeric[fid]
                    scaled = np.clip((v - st.norm_lo) / (st.norm_hi - st.norm_lo), 0.0, 1.0)
                    num_value[rows, j] = np.where(finite, scaled, 0.0)
                    b = _bin_index(np.where(finite, v, st.edges[0]), st.edges)
                    target[rows, j] = np.where(finite, b, -1)
                else:
                    n = self.cat_sizes[fid]
                    code = np.where(finite, v, 0).astype(np.int64)
                    bad = finite & ((code < 0) | (code >= n))
                    if bad.any():
                        raise ValueError(f"categorical code out of range for field id {fid}")
                    cat_value[rows, j] = code
                    target[rows, j] = np.where(finite, code, -1)
        return {
            "field_id": field_id,
            "num_value": num_value,
            "cat_value": cat_value,
            "missing": missing,
            "target": target,
        }

    def decode_bins(self, field_ids: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Bin index -> representative raw value (numerical fields; NaN otherwise)."""
        out = np.full(bins.shape, np.nan, dtype=np.float64)
        for fid, st in self.numeric.items():
            sel = field_ids == fid
            if sel.any():
                out[sel] = st.centers[np.clip(bins[sel], 0, st.n_bins - 1)]
        return out

    # ── (de)serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "numeric": {
                str(fid): {
                    "norm_lo": st.norm_lo,
                    "norm_hi": st.norm_hi,
                    "edges": st.edges.tolist(),
                    "centers": st.centers.tolist(),
                    "median": st.median,
                }
                for fid, st in self.numeric.items()
            },
            "cat_sizes": {str(fid): n for fid, n in self.cat_sizes.items()},
        }

    @classmethod
    def from_dict(cls, schema: TableSchema, d: dict) -> Normalizer:
        norm = cls(schema=schema)
        for fid, x in d["numeric"].items():
            norm.numeric[int(fid)] = NumericStats(
                x["norm_lo"], x["norm_hi"], np.asarray(x["edges"]), np.asarray(x["centers"]),
                x.get("median", float("nan")),
            )
        norm.cat_sizes = {int(fid): int(n) for fid, n in d["cat_sizes"].items()}
        return norm

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, schema: TableSchema, path: str | Path) -> Normalizer:
        return cls.from_dict(schema, json.loads(Path(path).read_text()))


def _bin_index(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the bin ``[edges[i], edges[i+1])`` holding ``v`` (clipped to the range)."""
    return np.searchsorted(edges[1:-1], v, side="right").astype(np.int64)
