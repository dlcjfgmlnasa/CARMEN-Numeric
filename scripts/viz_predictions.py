"""Numbers behind the prediction figures of a signal-only checkpoint (run where the data is).

    python scripts/viz_predictions.py --data $SIG --ckpt runs/unittab_signal_v1/best.pt \
        --out runs/viz_signal.json

* hypotension probe exactly as ``probe_ioh.py`` (current MAP, hand-made features + GBM,
  frozen encoder + logistic regression): ROC / PR curves, calibration and operating
  points (sensitivity, PPV, alerts per monitored hour) on test patients;
* forecast error by lead minute (1-15) on test windows: model vs. persistence;
* a few test cases drawn at random (seeded, not hand-picked): minute trends, the
  encoder's hypotension risk over time, and 15-minute forecasts (10/50/90 % of the
  predictive distribution) 10 minutes before hypotension onsets and at a stable time.

Writes aggregate curves plus the example cases' trends (public VitalDB cases).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_ioh as P  # noqa: E402

from carmen_numeric import (  # noqa: E402
    ForecastMaskConfig,
    Normalizer,
    TableSchema,
    sample_forecast_mask,
)
from carmen_numeric.data.dataset import GPUWindowBatcher, load_cases  # noqa: E402
from carmen_numeric.data.vitaldb import POINT_DECODE  # noqa: E402
from carmen_numeric.evaluate import decode, neighbor_values  # noqa: E402

LEAD = 15
FORE_FIELDS = ["art_mbp", "hr", "spo2", "nibp_mbp", "etco2"]
EXAMPLE_FIELDS = ["art_mbp", "hr", "spo2"]


def _r(x, nd: int = 4) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(x)]


def curve_points(y: np.ndarray, s: np.ndarray, n: int = 200) -> dict:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    i = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    j = np.unique(np.linspace(0, len(rec) - 1, n).astype(int))
    return {"fpr": _r(fpr[i]), "tpr": _r(tpr[i]), "recall": _r(rec[j]), "precision": _r(prec[j])}


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, len(edges) - 2)
    out = {"pred": [], "obs": [], "n": []}
    for k in range(len(edges) - 1):
        sel = idx == k
        if sel.any():
            out["pred"].append(round(float(p[sel].mean()), 4))
            out["obs"].append(round(float(y[sel].mean()), 4))
            out["n"].append(int(sel.sum()))
    return out


def operating_points(y: np.ndarray, s: np.ndarray, per_hour: float,
                     targets=(0.5, 0.7, 0.8, 0.9)) -> list[dict]:
    pos = np.sort(s[y == 1])
    out = []
    for target in targets:
        thr = pos[int(np.floor((1 - target) * len(pos)))]
        pred = s >= thr
        tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        out.append({"sensitivity": round(tp / max((y == 1).sum(), 1), 4),
                    "specificity": round(1 - fp / max((y == 0).sum(), 1), 4),
                    "ppv": round(tp / max(tp + fp, 1), 4),
                    "alerts_per_hour": round(float(pred.mean()) * per_hour, 3),
                    "false_alerts_per_hour": round(fp / len(y) * per_hour, 3)})
    return out


@torch.no_grad()
def forecast(model, batcher, case_idx, end, norm, schema, device, fields):
    """Predictive 10/50/90 % and truth of ``fields`` on the last LEAD rows ending at ``end``."""
    b = batcher.windows_ending_at(case_idx, end)
    is_time = torch.tensor([False] + [f.is_time for f in schema.fields], device=device)
    masked = sample_forecast_mask(b, is_time, ForecastMaskConfig(1.0, LEAD, LEAD))
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(b, masked)
    visible = (b["field_id"] > 0) & ~b["missing"] & ~masked
    t_fid = schema.field_id["t_min"]
    row_time = torch.where(b["field_id"] == t_fid, b["raw"], torch.zeros_like(b["raw"])).sum(-1)
    want = {schema.field_id[f]: f for f in fields}
    res: list[dict] = [{} for _ in range(len(end))]
    for pr in out["predictions"]:
        if pr.field_id not in want:
            continue
        name = want[pr.field_id]
        prob = torch.softmax(pr.logits.float(), -1)
        centers = torch.as_tensor(norm.numeric[pr.field_id].centers, dtype=torch.float32,
                                  device=device)
        cdf = prob.cumsum(-1)
        q = {f"q{int(qq * 100)}": centers[(cdf < qq).sum(-1).clamp(max=len(centers) - 1)]
             for qq in (0.1, 0.5, 0.9)}
        point = decode(prob, centers)[POINT_DECODE.get(name, "mean")]
        near, _ = neighbor_values(b["raw"], b["field_id"], visible, pr.field_id, pr.batch_idx,
                                  pr.row_idx, row_time)
        truth = b["raw"][pr.batch_idx, pr.row_idx, pr.slot]
        tt = row_time[pr.batch_idx, pr.row_idx]
        for k in range(len(end)):
            sel = pr.batch_idx == k
            if not sel.any():
                continue
            o = torch.argsort(tt[sel])
            res[k][name] = {"t": _r(tt[sel][o].cpu(), 1), "truth": _r(truth[sel][o].cpu(), 1),
                            "point": _r(point[sel][o].cpu(), 1),
                            "persist": _r(near[sel][o].cpu(), 1),
                            **{kk: _r(v[sel][o].cpu(), 1) for kk, v in q.items()}}
    return res


@torch.no_grad()
def lead_errors(model, cases, norm, schema, device, batch_size: int) -> dict:
    """MAE by lead minute of the last-LEAD-rows forecast: model vs. persistence."""
    batcher = GPUWindowBatcher(cases, norm, model.cfg.max_rows, (), device, keep_raw=True)
    is_time = torch.tensor([False] + [f.is_time for f in schema.fields], device=device)
    cfg = ForecastMaskConfig(1.0, LEAD, LEAD)
    want = {schema.field_id[f]: f for f in FORE_FIELDS if f in schema.field_id}
    z = lambda: torch.zeros(LEAD, dtype=torch.float64, device=device)  # noqa: E731
    acc = {f: {"model": z(), "persist": z(), "n": z()} for f in want.values()}
    model.eval()
    for b in batcher.eval_batches(batch_size):
        masked = sample_forecast_mask(b, is_time, cfg)
        if not masked.any():
            continue
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(b, masked)
        hidden = masked.any(-1)
        n_hidden = hidden.sum(1)
        rank = (b["row_type"] > 0).flip(1).long().cumsum(1).flip(1)
        lead = n_hidden[:, None] + 1 - rank  # 1 = first hidden row
        visible = (b["field_id"] > 0) & ~b["missing"] & ~masked
        for pr in out["predictions"]:
            if pr.field_id not in want:
                continue
            name = want[pr.field_id]
            centers = torch.as_tensor(norm.numeric[pr.field_id].centers, dtype=torch.float32,
                                      device=device)
            point = decode(torch.softmax(pr.logits.float(), -1), centers)[
                POINT_DECODE.get(name, "mean")].double()
            near, _ = neighbor_values(b["raw"], b["field_id"], visible, pr.field_id,
                                      pr.batch_idx, pr.row_idx)
            truth = b["raw"][pr.batch_idx, pr.row_idx, pr.slot].double()
            ok = ~torch.isnan(near)
            ld = (lead[pr.batch_idx, pr.row_idx] - 1).clamp(0, LEAD - 1)[ok]
            a = acc[name]
            a["model"].index_add_(0, ld, (point - truth).abs()[ok])
            a["persist"].index_add_(0, ld, (near.double() - truth).abs()[ok])
            a["n"].index_add_(0, ld, torch.ones_like(truth)[ok])
    return {f: {"lead": list(range(1, LEAD + 1)),
                "model": _r((a["model"] / a["n"].clamp(min=1)).cpu()),
                "persist": _r((a["persist"] / a["n"].clamp(min=1)).cpu()),
                "n": [int(x) for x in a["n"].cpu()]} for f, a in acc.items()}


def onsets(tm: np.ndarray, mp: np.ndarray) -> list[int]:
    """Rows where MAP first drops below 65 after >= 15 min of observed MAP >= 65."""
    out = []
    for i in np.flatnonzero(mp < P.HYPO):
        lo = np.searchsorted(tm, tm[i] - 15)
        prev = mp[lo:i]
        prev = prev[~np.isnan(prev)]
        if prev.size >= 8 and (prev >= P.HYPO).all():
            out.append(int(i))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="signal-only cache")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-event-cases", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    schema = TableSchema.load(data / "schema.json")
    norm = Normalizer.load(schema, data / "normalizer.json")
    split = json.loads((data / "split.json").read_text())
    ids = {p: sorted(split[p])[: args.max_cases] for p in P.PARTS}
    model = P.load_model(args.ckpt, device)
    t0 = time.time()

    # ── hypotension probe with curves ──
    horizons = [5, 10, 15]
    samples = {p: P.build_samples(load_cases(data, ids[p]), schema, horizons, 5, 15.0)
               for p in P.PARTS}
    feats = {p: P.encoder_features(model, data, ids[p], samples[p], device, args.batch_size)
             for p in P.PARTS}
    print(f"probe inputs ready in {time.time() - t0:.0f}s", flush=True)
    probe: dict = {}
    probes: dict[int, dict] = {}
    for k, h in enumerate(horizons):
        ok = {p: ~np.isnan(samples[p]["y"][:, k]) for p in P.PARTS}
        y = {p: samples[p]["y"][ok[p], k] for p in P.PARTS}
        hand = {p: samples[p]["hand"][ok[p]] for p in P.PARTS}
        logit, info = P.linear_probe(feats["train"][ok["train"]], y["train"],
                                     feats["val"][ok["val"]], y["val"],
                                     feats["test"][ok["test"]], device, return_model=True)
        probes[h] = info
        scores = {"map_only": -hand["test"][:, 1],
                  "hand_gbm": P.gbm_probe(hand["train"], y["train"], hand["test"]),
                  "encoder": 1 / (1 + np.exp(-logit))}
        yt = y["test"]
        probe[str(h)] = {
            "n": int(len(yt)), "prevalence": round(float(yt.mean()), 4),
            "methods": {m: {"auroc": round(P.auroc(yt, s), 4), "auprc": round(P.auprc(yt, s), 4),
                            **curve_points(yt, s),
                            "operating": operating_points(yt, s, per_hour=60 / 5)}
                        for m, s in scores.items()},
            "calibration": {m: calibration(yt, scores[m]) for m in ("hand_gbm", "encoder")},
        }
        print(f"[probe {h} min] " + " ".join(
            f"{m}={v['auroc']:.3f}" for m, v in probe[str(h)]["methods"].items()), flush=True)

    # ── forecast error by lead minute (test) ──
    test_cases = load_cases(data, ids["test"])
    leads = lead_errors(model, test_cases, norm, schema, device, args.batch_size)
    print(f"lead errors done ({time.time() - t0:.0f}s)", flush=True)

    # ── example cases, drawn at random ──
    mon, col = P.monitor_columns(schema)
    rng = np.random.default_rng(args.seed)
    event, stable = [], []
    for cid in sorted(test_cases):
        c = test_cases[cid]
        v = c["values"][c["row_type"] == mon]
        mp = v[:, col["art_mbp"]]
        if (~np.isnan(mp)).sum() < 120:
            continue
        on = onsets(v[:, col["t_min"]].astype(np.float64), mp)
        if on:
            event.append(cid)
        elif np.nanmin(mp) >= P.HYPO:
            stable.append(cid)
    picks = list(rng.choice(event, min(args.n_event_cases, len(event)), replace=False))
    picks += list(rng.choice(stable, min(1, len(stable)), replace=False))
    picks = [int(c) for c in picks]
    ex_cases = {c: test_cases[c] for c in picks}
    batcher = GPUWindowBatcher(ex_cases, norm, model.cfg.max_rows, (), device, keep_raw=True)
    examples = []
    for ci, cid in enumerate(batcher.caseids):
        c = ex_cases[cid]
        v = c["values"][c["row_type"] == mon]
        tm = v[:, col["t_min"]].astype(np.float64)
        mp = v[:, col["art_mbp"]].astype(np.float64)
        # risk at every minute with an arterial MAP in the last 5 min
        risk_rows = [i for i in range(len(tm)) if tm[i] >= 15 and
                     (~np.isnan(mp[np.searchsorted(tm, tm[i] - 5, side="right"): i + 1])).any()]
        smp = {"caseid": np.full(len(risk_rows), cid), "t": tm[risk_rows]}
        x = P.encoder_features(model, data, [cid], smp, device, args.batch_size)
        risk = {}
        for h, pr in probes.items():
            z = ((x - pr["mu"]) / pr["sd"]) @ pr["weight"] + pr["bias"]
            risk[str(h)] = _r(1 / (1 + np.exp(-z)))
        on = onsets(tm, mp)
        anchors = []
        for i in on[:2]:
            a = int(np.searchsorted(tm, tm[i] - 10, side="right") - 1)
            if a >= 30 and a + LEAD < len(tm):
                anchors.append(("before_onset", a))
        if not on:
            a = int(rng.integers(30, max(31, len(tm) - LEAD - 1)))
            anchors.append(("stable", a))
        fc = []
        if anchors:
            ends = torch.tensor([a + LEAD for _, a in anchors], device=device)
            fres = forecast(model, batcher, torch.full_like(ends, ci), ends, norm, schema, device,
                            EXAMPLE_FIELDS)
            fc = [{"kind": kind, "anchor_t": round(float(tm[a]), 1), **r}
                  for (kind, a), r in zip(anchors, fres, strict=True)]
        examples.append({
            "caseid": cid, "has_onset": bool(on),
            "series": {"t": _r(tm, 1), **{f: _r(v[:, col[f]], 1) for f in EXAMPLE_FIELDS}},
            "risk_t": _r(tm[risk_rows], 1), "risk": risk,
            "onsets_t": [round(float(tm[i]), 1) for i in on], "forecasts": fc,
        })
    out = {"ckpt": args.ckpt, "probe": probe, "leads": leads, "examples": examples,
           "probe_pick": {str(h): {k: v for k, v in pr.items() if k in ("auroc", "wd", "epoch")}
                          for h, pr in probes.items()}}
    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
