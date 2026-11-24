"""Masked-value reconstruction metrics in physical units, against simple baselines.

For every masked numerical value the model's prediction is decoded back to the field's
unit (bin mode, and the expectation over bin centers). Two baselines see the same
masked batch:

* ``median``  — the training median of the field;
* ``nearest`` — the field's value in the nearest row of the same sequence where it is
  observed and not masked (forward/backward fill). For dense vitals this is a strong
  baseline: a model that does not beat it has not learned temporal structure.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .data.dataset import to_device
from .model import SpanMaskConfig, UniTTab, masked_token_loss, sample_mask, sample_span_mask
from .normalize import Normalizer


def _nearest_fill(
    raw: np.ndarray,  # (B, T, K)
    field_id: np.ndarray,  # (B, T, K)
    visible: np.ndarray,  # (B, T, K) observed and not masked
    fid: int,
    b: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    has = (field_id == fid) & visible
    row_has = has.any(-1)  # (B, T) — a field sits in at most one slot per row
    row_val = np.where(has, raw, 0.0).sum(-1)
    out = np.full(len(b), np.nan)
    for bi in np.unique(b):
        rows = np.nonzero(row_has[bi])[0]
        if rows.size == 0:
            continue
        sel = np.nonzero(b == bi)[0]
        pos = np.searchsorted(rows, t[sel])
        lo = rows[np.clip(pos - 1, 0, rows.size - 1)]
        hi = rows[np.clip(pos, 0, rows.size - 1)]
        pick = np.where(np.abs(t[sel] - lo) <= np.abs(hi - t[sel]), lo, hi)
        out[sel] = row_val[bi, pick]
    return out


@torch.no_grad()
def evaluate(
    model: UniTTab,
    loader,
    normalizer: Normalizer,
    device: torch.device,
    p_field: float = 0.15,
    p_row: float = 0.1,
    eps: float = 0.1,
    radius: int = 5,
    seed: int = 1234,
    max_batches: int | None = None,
    amp_dtype: torch.dtype | None = None,
    span: SpanMaskConfig | None = None,
    context_row_types: tuple[int, ...] = (),
) -> dict:
    """Loss, accuracies and per-field MAE (model vs. baselines) on a fixed masking.

    ``span=None`` evaluates the UniTTab masks (``p_field``, ``p_row``); with a
    ``SpanMaskConfig`` only span masks are used — the gap-filling regime, where the
    nearest-value baseline has to bridge whole stretches.
    """
    model.eval()
    schema = normalizer.schema
    is_time = torch.tensor([False] + [f.is_time for f in schema.fields])
    gen = torch.Generator().manual_seed(seed)
    loss_sum, n_sum = 0.0, 0
    acc = defaultdict(float)
    err: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        if span is None:
            masked = sample_mask(batch, is_time, p_field, p_row, generator=gen)
        else:
            masked = sample_span_mask(batch, is_time, span, torch.tensor(context_row_types),
                                      generator=gen)
        if not masked.any():
            continue
        dev_batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(dev_batch, masked.to(device))
        loss, st = masked_token_loss(out["predictions"], model.cfg.field_kinds, eps, radius)
        n = st["n_num"] + st["n_cat"]
        loss_sum += loss.item() * n
        n_sum += n
        for k in ("acc_num", "acc_num_pm2"):
            acc[k] += st[k] * st["n_num"]
        acc["acc_cat"] += st["acc_cat"] * st["n_cat"]
        acc["n_num"] += st["n_num"]
        acc["n_cat"] += st["n_cat"]

        raw = batch["raw"].numpy()
        fids = batch["field_id"].numpy()
        visible = (~batch["missing"] & ~masked).numpy() & (fids > 0)
        for pr in out["predictions"]:
            fid = pr.field_id
            if fid not in normalizer.numeric:
                continue
            st_f = normalizer.numeric[fid]
            b = pr.batch_idx.cpu().numpy()
            t = pr.row_idx.cpu().numpy()
            truth = raw[b, t, pr.slot].astype(np.float64)
            prob = torch.softmax(pr.logits.float(), -1).cpu().numpy()
            centers = st_f.centers
            mode = centers[prob.argmax(-1)]
            mean = prob @ centers
            near = _nearest_fill(raw, fids, visible, fid, b, t)
            med = np.full_like(truth, st_f.median)
            near = np.where(np.isfinite(near), near, med)
            err[fid]["mode"].append(np.abs(mode - truth))
            err[fid]["mean"].append(np.abs(mean - truth))
            err[fid]["median"].append(np.abs(med - truth))
            err[fid]["nearest"].append(np.abs(near - truth))

    per_field = {}
    for fid, d in err.items():
        name = schema.fields[fid - 1].name
        per_field[name] = {k: float(np.mean(np.concatenate(v))) for k, v in d.items()}
        per_field[name]["n"] = int(sum(len(x) for x in d["mode"]))
    model.train()
    return {
        "loss": loss_sum / max(n_sum, 1),
        "acc_num": acc["acc_num"] / max(acc["n_num"], 1),
        "acc_num_pm2": acc["acc_num_pm2"] / max(acc["n_num"], 1),
        "acc_cat": acc["acc_cat"] / max(acc["n_cat"], 1),
        "n_masked": n_sum,
        "mae": per_field,
    }


def summarize_mae(per_field: dict, fields: list[str] | None = None) -> str:
    rows = [f"{'field':<12}{'n':>7}{'model':>9}{'(mean)':>9}{'nearest':>9}{'median':>9}"]
    for name, d in per_field.items():
        if fields and name not in fields:
            continue
        rows.append(
            f"{name:<12}{d['n']:>7}{d['mode']:>9.3f}{d['mean']:>9.3f}"
            f"{d['nearest']:>9.3f}{d['median']:>9.3f}"
        )
    return "\n".join(rows)
