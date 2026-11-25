"""Masked-value reconstruction metrics in physical units, against simple baselines.

For every masked numerical value the predicted bin distribution is turned back into a
value in the field's unit, three ways — the most likely bin (``mode``), the expectation
over bin centers (``mean``) and the distribution's median bin (``pmedian``) — and the
point estimate reported as ``model`` uses, per field, the decode given in
``point_decode`` (default ``mean``). The median is the MAE-optimal choice for fields whose
values pile up at one end (SpO2 at 100 %, MAC at 0), where the mean is pulled away from
the bulk.

Three baselines see the same masked batch:

* ``median``  — the training median of the field;
* ``nearest`` — the field's value in the nearest row of the same sequence where it is
  observed and not masked (forward/backward fill, ties to the earlier row). For dense
  vitals this is a strong baseline: a model that does not beat it has not learned
  temporal structure.
* ``linear``  — linear interpolation in time between the previous and the next visible
  value (``nearest`` where only one side exists). Minute-level vitals are smooth, so
  this is the bar for "more than interpolation". Under forecast masks nothing after the
  hidden rows is visible, and both baselines become persistence (last value).

Everything runs on the batches' device; the host only sees the final sums.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch

from .data.dataset import to_device
from .model import (
    ForecastMaskConfig,
    SpanMaskConfig,
    UniTTab,
    masked_token_loss,
    sample_forecast_mask,
    sample_mask,
    sample_span_mask,
)
from .normalize import Normalizer

DECODES = ("mode", "mean", "pmedian")
ERR_KEYS = ("model", "mode", "mean", "pmedian", "nearest", "linear", "median")


def neighbor_values(
    raw: torch.Tensor,  # (B, T, K) raw values (NaN = missing)
    field_id: torch.Tensor,  # (B, T, K)
    visible: torch.Tensor,  # (B, T, K) observed and not masked
    field: int,
    b_idx: torch.Tensor,  # (n,)
    t_idx: torch.Tensor,  # (n,)
    row_time: torch.Tensor | None = None,  # (B, T) time of each row (default: row index)
) -> tuple[torch.Tensor, torch.Tensor]:  # (n,), (n,) NaN where never visible
    """Nearest visible value of ``field`` and linear interpolation between neighbors."""
    has = (field_id == field) & visible
    row_has = has.any(-1)  # (B, T) — a field sits in at most one slot per row
    row_val = torch.where(has, raw, torch.zeros_like(raw)).sum(-1)
    _, t = row_has.shape
    ar = torch.arange(t, device=raw.device).expand_as(row_has)
    prev = torch.where(row_has, ar, -1).cummax(1).values  # last visible row <= t
    nxt = torch.where(row_has, ar, t).flip(1).cummin(1).values.flip(1)  # first >= t
    p, n = prev[b_idx, t_idx], nxt[b_idx, t_idx]
    has_p, has_n = p >= 0, n < t
    pc, nc = p.clamp(0, t - 1), n.clamp(0, t - 1)
    vp, vn = row_val[b_idx, pc], row_val[b_idx, nc]
    use_p = has_p & (~has_n | ((t_idx - p) <= (n - t_idx)))
    near = torch.where(use_p, vp, vn)
    if row_time is None:
        tt, tp, tn = t_idx.float(), p.float(), n.float()
    else:
        tt, tp, tn = row_time[b_idx, t_idx], row_time[b_idx, pc], row_time[b_idx, nc]
    w = ((tt - tp) / (tn - tp).clamp(min=1e-6)).clamp(0, 1)
    lin = torch.where(has_p & has_n, vp + (vn - vp) * w, near)
    nan = torch.full_like(near, float("nan"))
    ok = has_p | has_n
    return torch.where(ok, near, nan), torch.where(ok, lin, nan)


def nearest_values(
    raw: torch.Tensor,  # (B, T, K)
    field_id: torch.Tensor,  # (B, T, K)
    visible: torch.Tensor,  # (B, T, K)
    field: int,
    b_idx: torch.Tensor,  # (n,)
    t_idx: torch.Tensor,  # (n,)
) -> torch.Tensor:  # (n,)
    """Value of ``field`` in the nearest visible row of the same sequence."""
    return neighbor_values(raw, field_id, visible, field, b_idx, t_idx)[0]


def decode(prob: torch.Tensor, centers: torch.Tensor) -> dict[str, torch.Tensor]:
    """(n, V) bin probabilities -> point estimates (n,) per decode."""
    cdf = prob.cumsum(-1)
    med_bin = (cdf < 0.5).sum(-1).clamp(max=prob.shape[-1] - 1)
    return {
        "mode": centers[prob.argmax(-1)],
        "mean": prob @ centers,
        "pmedian": centers[med_bin],
    }


@torch.no_grad()
def evaluate(
    model: UniTTab,
    batches: Iterable[dict[str, torch.Tensor]],
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
    point_decode: dict[str, str] | None = None,
    forecast: ForecastMaskConfig | None = None,
) -> dict:
    """Loss, accuracies and per-field MAE (model vs. baselines) on a fixed masking.

    ``batches`` yields collated batches (with ``raw``), on the CPU or already on
    ``device``. By default the UniTTab masks (``p_field``, ``p_row``) are evaluated; with
    a ``SpanMaskConfig`` only span masks (gap filling), with a ``ForecastMaskConfig``
    only forecast masks (the last rows of each window).
    """
    model.eval()
    schema = normalizer.schema
    point_decode = point_decode or {}
    is_time = torch.tensor([False] + [f.is_time for f in schema.fields], device=device)
    ctx = torch.tensor(context_row_types, dtype=torch.long, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    centers = {fid: torch.as_tensor(st.centers, dtype=torch.float32, device=device)
               for fid, st in normalizer.numeric.items()}
    medians = {fid: float(st.median) for fid, st in normalizer.numeric.items()}
    choice = {schema.field_id[k]: v for k, v in point_decode.items() if k in schema.field_id}
    t_fid = schema.field_id.get("t_min")
    zero = torch.zeros((), device=device, dtype=torch.float64)
    err: dict[int, dict[str, torch.Tensor]] = defaultdict(lambda: {k: zero for k in ERR_KEYS})
    cnt: dict[int, int] = defaultdict(int)
    tot = {"loss": 0.0, "n": 0, "n_num": 0, "n_cat": 0, "acc_num": 0.0, "acc_num_pm2": 0.0,
           "acc_cat": 0.0, "windows": 0}
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        b = to_device(batch, device)
        if forecast is not None:
            masked = sample_forecast_mask(b, is_time, forecast, ctx, generator=gen)
        elif span is not None:
            masked = sample_span_mask(b, is_time, span, ctx, generator=gen)
        else:
            masked = sample_mask(b, is_time, p_field, p_row, generator=gen)
        tot["windows"] += b["row_type"].shape[0]
        if not masked.any():
            continue
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(b, masked)
        loss, st = masked_token_loss(out["predictions"], model.cfg.field_kinds, eps, radius)
        n = st["n_num"] + st["n_cat"]
        tot["loss"] += loss.item() * n
        tot["n"] += n
        tot["n_num"] += st["n_num"]
        tot["n_cat"] += st["n_cat"]
        tot["acc_num"] += st["acc_num"] * st["n_num"]
        tot["acc_num_pm2"] += st["acc_num_pm2"] * st["n_num"]
        tot["acc_cat"] += st["acc_cat"] * st["n_cat"]

        raw = b["raw"]
        visible = (b["field_id"] > 0) & ~b["missing"] & ~masked
        row_time = None
        if t_fid is not None:
            is_t = b["field_id"] == t_fid
            row_time = torch.where(is_t, raw, torch.zeros_like(raw)).sum(-1)
        for pr in out["predictions"]:
            fid = pr.field_id
            if fid not in centers:
                continue
            truth = raw[pr.batch_idx, pr.row_idx, pr.slot].double()
            est = decode(torch.softmax(pr.logits.float(), -1), centers[fid])
            est = {k: v.double() for k, v in est.items()}
            est["model"] = est[choice.get(fid, "mean")]
            est["median"] = torch.full_like(truth, medians[fid])
            near, lin = neighbor_values(raw, b["field_id"], visible, fid, pr.batch_idx,
                                        pr.row_idx, row_time)
            est["nearest"] = torch.where(torch.isnan(near), est["median"], near.double())
            est["linear"] = torch.where(torch.isnan(lin), est["median"], lin.double())
            e = err[fid]
            for k in ERR_KEYS:
                e[k] = e[k] + (est[k] - truth).abs().sum()
            cnt[fid] += truth.numel()

    per_field = {}
    for fid in sorted(err):
        name = schema.fields[fid - 1].name
        sums = torch.stack([err[fid][k] for k in ERR_KEYS]).tolist()
        per_field[name] = {k: s / cnt[fid] for k, s in zip(ERR_KEYS, sums, strict=True)}
        per_field[name]["n"] = cnt[fid]
        per_field[name]["decode"] = choice.get(fid, "mean")
    model.train()
    return {
        "loss": tot["loss"] / max(tot["n"], 1),
        "acc_num": tot["acc_num"] / max(tot["n_num"], 1),
        "acc_num_pm2": tot["acc_num_pm2"] / max(tot["n_num"], 1),
        "acc_cat": tot["acc_cat"] / max(tot["n_cat"], 1),
        "n_masked": tot["n"],
        "n_num": tot["n_num"],
        "n_cat": tot["n_cat"],
        "windows": tot["windows"],
        "mae": per_field,
    }


def combine(results: list[dict]) -> dict:
    """Merge ``evaluate`` outputs of disjoint parts (e.g. splits), weighting by counts."""
    out = {"n_masked": sum(r["n_masked"] for r in results),
           "n_num": sum(r["n_num"] for r in results),
           "n_cat": sum(r["n_cat"] for r in results),
           "windows": sum(r["windows"] for r in results)}
    out["loss"] = sum(r["loss"] * r["n_masked"] for r in results) / max(out["n_masked"], 1)
    for k, n in (("acc_num", "n_num"), ("acc_num_pm2", "n_num"), ("acc_cat", "n_cat")):
        out[k] = sum(r[k] * r[n] for r in results) / max(out[n], 1)
    mae: dict[str, dict] = {}
    for r in results:
        for name, d in r["mae"].items():
            m = mae.setdefault(name, {k: 0.0 for k in ERR_KEYS} | {"n": 0, "decode": d["decode"]})
            for k in ERR_KEYS:
                m[k] += d[k] * d["n"]
            m["n"] += d["n"]
    for m in mae.values():
        for k in ERR_KEYS:
            m[k] /= max(m["n"], 1)
    out["mae"] = mae
    return out


def summarize_mae(per_field: dict, fields: list[str] | None = None) -> str:
    rows = [f"{'field':<12}{'n':>8}{'model':>9} {'decode':<8}{'nearest':>9}{'linear':>9}"
            f"{'median':>9}   {'mode':>8}{'mean':>8}{'pmedian':>8}"]
    names = fields if fields else list(per_field)
    for name in names:
        if name not in per_field:
            continue
        d = per_field[name]
        rows.append(
            f"{name:<12}{d['n']:>8}{d['model']:>9.3f} {d['decode']:<8}{d['nearest']:>9.3f}"
            f"{d.get('linear', float('nan')):>9.3f}{d['median']:>9.3f}   "
            f"{d['mode']:>8.3f}{d['mean']:>8.3f}{d['pmedian']:>8.3f}"
        )
    return "\n".join(rows)
