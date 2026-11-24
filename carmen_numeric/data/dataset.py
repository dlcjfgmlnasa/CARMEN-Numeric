"""Windowed sequences of heterogeneous rows + collate into a padded batch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..normalize import Normalizer
from ..schema import TableSchema


def load_cases(cache_dir: str | Path, caseids: list[int] | None = None) -> dict[int, dict]:
    """Load per-case ``.npz`` files (``values`` (R, K), ``row_type`` (R,), ``subjectid``)."""
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("cases/*.npz"), key=lambda p: int(p.stem))
    keep = None if caseids is None else set(int(c) for c in caseids)
    out = {}
    for f in files:
        cid = int(f.stem)
        if keep is not None and cid not in keep:
            continue
        d = np.load(f)
        out[cid] = {
            "values": d["values"],
            "row_type": d["row_type"].astype(np.int64),
            "subjectid": int(d["subjectid"]),
        }
    return out


class WindowDataset(Dataset):
    """One item = the static row(s) of a case + a window of its dynamic rows.

    Rows whose type is in ``context_types`` (e.g. ``static``) are always prepended;
    the remaining rows are cut into windows of up to ``max_rows - n_context`` rows.

    train (``random_crop=True``): one random window per case per epoch
    (``windows_per_case`` draws), as in UniTTab's random sub-sequence sampling.
    eval  (``random_crop=False``): every non-overlapping window, deterministic.
    """

    def __init__(
        self,
        cases: dict[int, dict],
        normalizer: Normalizer,
        max_rows: int = 128,
        context_types: tuple[int, ...] = (1,),
        random_crop: bool = True,
        windows_per_case: int = 1,
        seed: int = 0,
    ) -> None:
        self.cases = cases
        self.caseids = sorted(cases)
        self.norm = normalizer
        self.max_rows = max_rows
        self.context_types = np.asarray(context_types)
        self.random_crop = random_crop
        self.windows_per_case = windows_per_case
        self.rng = np.random.default_rng(seed)
        self._split = {}
        for cid in self.caseids:
            rt = cases[cid]["row_type"]
            ctx = np.isin(rt, self.context_types)
            self._split[cid] = (np.nonzero(ctx)[0], np.nonzero(~ctx)[0])
        self.span = max_rows - max(len(c) for c, _ in self._split.values())
        if self.span < 1:
            raise ValueError("max_rows too small for the context rows")
        if random_crop:
            self.index = [(cid, -1) for cid in self.caseids for _ in range(windows_per_case)]
        else:
            self.index = []
            for cid in self.caseids:
                n = len(self._split[cid][1])
                for s in range(0, max(n, 1), self.span):
                    self.index.append((cid, s))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        cid, start = self.index[i]
        ctx, dyn = self._split[cid]
        n = len(dyn)
        if start < 0:
            start = int(self.rng.integers(0, max(n - self.span, 0) + 1))
        rows = np.concatenate([ctx, dyn[start : start + self.span]])
        c = self.cases[cid]
        item = self.norm.transform(c["values"][rows], c["row_type"][rows])
        item["row_type"] = c["row_type"][rows]
        item["raw"] = c["values"][rows].astype(np.float32)
        item["caseid"] = np.int64(cid)
        return item


def collate(items: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Pad rows to the longest item. Padding rows get ``row_type`` 0, ``field_id`` 0."""
    t = max(len(it["row_type"]) for it in items)
    k = items[0]["field_id"].shape[1]
    b = len(items)
    out = {
        "row_type": torch.zeros(b, t, dtype=torch.long),
        "field_id": torch.zeros(b, t, k, dtype=torch.long),
        "num_value": torch.zeros(b, t, k, dtype=torch.float32),
        "cat_value": torch.zeros(b, t, k, dtype=torch.long),
        "missing": torch.zeros(b, t, k, dtype=torch.bool),
        "target": torch.full((b, t, k), -1, dtype=torch.long),
        "raw": torch.full((b, t, k), float("nan"), dtype=torch.float32),
        "caseid": torch.tensor([int(it["caseid"]) for it in items]),
    }
    for i, it in enumerate(items):
        n = len(it["row_type"])
        out["row_type"][i, :n] = torch.from_numpy(it["row_type"])
        for key in ("field_id", "num_value", "cat_value", "missing", "target", "raw"):
            out[key][i, :n] = torch.from_numpy(it[key])
    return out


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def patient_split(
    cases: dict[int, dict], fractions: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> dict[str, list[int]]:
    """Split case ids by patient (``subjectid``) so no patient is on two sides."""
    subj = sorted({c["subjectid"] for c in cases.values()})
    rng = np.random.default_rng(seed)
    rng.shuffle(subj)
    n = len(subj)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    side = {s: ("train" if i < a else "val" if i < b else "test") for i, s in enumerate(subj)}
    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for cid, c in sorted(cases.items()):
        out[side[c["subjectid"]]].append(cid)
    return out


def collect_field_values(
    cases: dict[int, dict], schema: TableSchema, max_per_field: int = 2_000_000, seed: int = 0
) -> dict[int, np.ndarray]:
    """Observed raw values per numerical field (for ``Normalizer.fit``), subsampled."""
    buckets: dict[int, list[np.ndarray]] = {}
    for c in cases.values():
        v, rt = c["values"], c["row_type"]
        for h in np.unique(rt):
            rows = rt == h
            for j, fid in enumerate(schema.row_field_ids(int(h))):
                if schema.fields[fid - 1].kind != "num":
                    continue
                x = v[rows, j]
                x = x[np.isfinite(x)]
                if x.size:
                    buckets.setdefault(fid, []).append(x)
    rng = np.random.default_rng(seed)
    out = {}
    for fid, parts in buckets.items():
        x = np.concatenate(parts)
        if x.size > max_per_field:
            x = rng.choice(x, max_per_field, replace=False)
        out[fid] = x
    return out


class GPUWindowBatcher:
    """Batches built entirely on the device (no DataLoader workers).

    All rows of all cases are transformed once and kept on ``device`` (a few hundred MB
    for the whole of VitalDB); windows are cut by index arithmetic. ``sample`` draws random
    windows for training; ``eval_batches`` walks every case in non-overlapping windows
    (the same windows as ``WindowDataset(random_crop=False)``). Use it when the host CPU
    is the bottleneck (e.g. a shared node). Context rows (``context_types``) must be the
    first rows of each case, as ``data.vitaldb.build_case`` writes them.

    ``keep_raw`` also keeps the raw values (needed for MAE in physical units).
    """

    def __init__(
        self,
        cases: dict[int, dict],
        normalizer: Normalizer,
        max_rows: int,
        context_types: tuple[int, ...],
        device: torch.device,
        keep_raw: bool = False,
    ) -> None:
        keys = ["field_id", "num_value", "cat_value", "missing", "target", "row_type"]
        if keep_raw:
            keys.append("raw")
        parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        n_ctx, dyn_start, dyn_len, ctx_start = [], [], [], []
        offset = 0
        self.caseids = sorted(cases)
        for cid in self.caseids:
            c = cases[cid]
            rt = c["row_type"]
            ctx = np.isin(rt, np.asarray(context_types))
            k = int(ctx.sum())
            if k and not ctx[:k].all():
                raise ValueError(f"case {cid}: context rows must come first")
            tr = normalizer.transform(c["values"], rt)
            for key in ("field_id", "num_value", "cat_value", "missing", "target"):
                parts[key].append(tr[key])
            parts["row_type"].append(rt)
            if keep_raw:
                parts["raw"].append(c["values"].astype(np.float32))
            n_ctx.append(k)
            ctx_start.append(offset)
            dyn_start.append(offset + k)
            dyn_len.append(len(rt) - k)
            offset += len(rt)
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        dt = {"field_id": torch.int16, "cat_value": torch.int16, "target": torch.int16,
              "row_type": torch.int8, "num_value": torch.float32, "missing": torch.bool,
              "raw": torch.float32}
        self.rows = {k: torch.as_tensor(v).to(device=device, dtype=dt[k]) for k, v in cat.items()}
        self.n_ctx = torch.tensor(n_ctx, device=device)
        self.ctx_start = torch.tensor(ctx_start, device=device)
        self.dyn_start = torch.tensor(dyn_start, device=device)
        self.dyn_len = torch.tensor(dyn_len, device=device)
        self.max_ctx = int(max(n_ctx))
        self.span = max_rows - self.max_ctx
        if self.span < 1:
            raise ValueError("max_rows too small for the context rows")
        self.device = device
        self.num_cases = len(n_ctx)
        self._dyn_len_host = dyn_len

    def _gather(self, case: torch.Tensor, start: torch.Tensor) -> dict[str, torch.Tensor]:
        """Window = context rows of ``case`` + up to ``span`` dynamic rows from ``start``."""
        dev = self.device
        take = (self.dyn_len[case] - start).clamp(min=0, max=self.span)
        t_dyn = max(int(take.max()), 1)
        ar = torch.arange(t_dyn, device=dev)
        dyn_idx = self.dyn_start[case, None] + start[:, None] + ar  # (B, t_dyn)
        dyn_ok = ar < take[:, None]
        if self.max_ctx:
            ac = torch.arange(self.max_ctx, device=dev)
            ctx_idx = self.ctx_start[case, None] + ac
            ctx_ok = ac < self.n_ctx[case, None]
            idx = torch.cat([ctx_idx, dyn_idx], 1)
            ok = torch.cat([ctx_ok, dyn_ok], 1)
        else:
            idx, ok = dyn_idx, dyn_ok
        idx = torch.where(ok, idx, 0)
        okk = ok.unsqueeze(-1)
        r = self.rows
        out = {
            "row_type": torch.where(ok, r["row_type"][idx].long(), 0),
            "field_id": torch.where(okk, r["field_id"][idx].long(), 0),
            "num_value": torch.where(okk, r["num_value"][idx], 0.0),
            "cat_value": torch.where(okk, r["cat_value"][idx].long(), 0),
            "missing": r["missing"][idx] & okk,
            "target": torch.where(okk, r["target"][idx].long(), -1),
        }
        if "raw" in r:
            out["raw"] = torch.where(okk, r["raw"][idx], float("nan"))
        return out

    def sample(self, batch_size: int, generator: torch.Generator) -> dict[str, torch.Tensor]:
        """Random cases, each with a random window of its dynamic rows (training)."""
        dev = self.device
        case = torch.randint(self.num_cases, (batch_size,), device=dev, generator=generator)
        length = self.dyn_len[case]
        room = (length - length.clamp(max=self.span) + 1).float()
        start = (torch.rand(batch_size, device=dev, generator=generator) * room).long()
        return self._gather(case, start)

    def eval_windows(self) -> list[tuple[int, int]]:
        """Every (case index, start) of the non-overlapping evaluation windows."""
        out = []
        for i, n in enumerate(self._dyn_len_host):
            for s in range(0, max(n, 1), self.span):
                out.append((i, s))
        return out

    def eval_batches(self, batch_size: int) -> EvalBatches:
        return EvalBatches(self, batch_size)


class EvalBatches:
    """Re-iterable deterministic evaluation batches of a ``GPUWindowBatcher``."""

    def __init__(self, batcher: GPUWindowBatcher, batch_size: int) -> None:
        self.batcher = batcher
        self.batch_size = batch_size
        w = torch.tensor(batcher.eval_windows(), dtype=torch.long)
        self.case, self.start = w[:, 0].to(batcher.device), w[:, 1].to(batcher.device)

    def __len__(self) -> int:
        return -(-len(self.case) // self.batch_size)

    @property
    def num_windows(self) -> int:
        return len(self.case)

    def __iter__(self):
        for i in range(0, len(self.case), self.batch_size):
            sl = slice(i, i + self.batch_size)
            yield self.batcher._gather(self.case[sl], self.start[sl])
