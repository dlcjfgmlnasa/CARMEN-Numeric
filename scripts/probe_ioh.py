"""Frozen-encoder probe: intra-operative hypotension (arterial MAP < 65 mmHg) within H min.

    python scripts/probe_ioh.py --label-data $SIG \
        --model signal  $SIG  runs/unittab_signal_v1/best.pt \
        --model full_sf $FULL runs/unittab_full_sf_v1/best.pt \
        --random-control --out runs/probe_ioh.json

Samples come from the label cache (any cache with ``art_mbp``; the signal-only one):

* a prediction time ``t`` every ``--stride`` minutes after ``--min-history`` minutes;
* eligible when the arterial MAP was recorded in the last 5 minutes and all of those
  values are >= 65 (not already hypotensive);
* label for horizon H: any arterial MAP < 65 in ``(t, t + H]``; dropped for that H when no
  arterial value exists in the horizon.

Each model sees the window of its own cache that ends at ``t`` (rows with
``t_min <= t``, nothing masked, no future rows) and gives ``[CLS, last row, mean of
rows]``; a logistic regression on those features (weight decay and epoch picked on val
patients) is scored on test patients. Baselines on the same samples: the current MAP
alone, and hand-made trend features of the signal fields (HistGradientBoosting).
``--random-control`` also probes an untrained encoder of the first model's architecture.
Patient-level bootstrap gives 95% intervals.
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

from carmen_numeric import Normalizer, TableSchema, UniTTab, UniTTabConfig  # noqa: E402
from carmen_numeric.data.dataset import GPUWindowBatcher, load_cases  # noqa: E402

PARTS = ("train", "val", "test")
HYPO = 65.0


# ── samples and labels ─────────────────────────────────────────────


def monitor_columns(schema: TableSchema) -> tuple[int, dict[str, int]]:
    rt = schema.row_types[schema.row_type_id["monitor"] - 1]
    return schema.row_type_id["monitor"], {f: j for j, f in enumerate(rt.fields)}


def _slope(t: np.ndarray, v: np.ndarray) -> float:
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return np.nan
    tt, vv = t[ok], v[ok]
    tt = tt - tt.mean()
    return float((tt * (vv - vv.mean())).sum() / max((tt * tt).sum(), 1e-6))


def _last(v: np.ndarray) -> float:
    ok = np.flatnonzero(~np.isnan(v))
    return float(v[ok[-1]]) if ok.size else np.nan


def _mean(v: np.ndarray) -> float:
    return float(np.nanmean(v)) if (~np.isnan(v)).any() else np.nan


def _col(get: dict[str, np.ndarray], f: str, w: slice) -> np.ndarray:
    return get[f][w] if f in get else np.full(1, np.nan)


HAND = ["t", "map_last", "map_mean5", "map_mean15", "map_min15", "map_std15", "map_slope15",
        "map_mean30", "map_slope30", "sbp_last", "dbp_last", "hr_last", "hr_mean5",
        "hr_slope15", "spo2_last", "etco2_last", "rr_last", "tv_last", "pip_last", "bt_last"]


def build_samples(cases: dict[int, dict], schema: TableSchema, horizons: list[int],
                  stride: int, min_history: float) -> dict:
    mon, col = monitor_columns(schema)
    need = {"t_min", "art_mbp"}
    if not need <= set(col):
        raise ValueError(f"label cache needs {need} in the monitor row")
    cid_l, t_l, y_l, hand_l = [], [], [], []
    for cid in sorted(cases):
        c = cases[cid]
        v = c["values"][c["row_type"] == mon]
        if v.size == 0:
            continue
        tm = v[:, col["t_min"]].astype(np.float64)
        mp = v[:, col["art_mbp"]].astype(np.float64)
        if np.isnan(mp).all():
            continue
        get = {f: v[:, col[f]].astype(np.float64) for f in
               ("art_sbp", "art_dbp", "hr", "spo2", "etco2", "rr", "tv", "pip", "bt") if f in col}
        anchors = np.flatnonzero((tm >= min_history) & (np.floor(tm) % stride == 0))
        for i in anchors:
            t = tm[i]
            lo5 = np.searchsorted(tm, t - 5, side="right")
            recent = mp[lo5 : i + 1]
            recent = recent[~np.isnan(recent)]
            if recent.size == 0 or (recent < HYPO).any():
                continue
            ys = []
            for h in horizons:
                hi = np.searchsorted(tm, t + h, side="right")
                fut = mp[i + 1 : hi]
                fut = fut[~np.isnan(fut)]
                ys.append(np.nan if fut.size == 0 else float((fut < HYPO).any()))
            if np.isnan(ys).all():
                continue
            w15 = slice(np.searchsorted(tm, t - 15, side="right"), i + 1)
            w30 = slice(np.searchsorted(tm, t - 30, side="right"), i + 1)
            w5 = slice(lo5, i + 1)
            def g(f: str, w: slice, get=get) -> np.ndarray:
                return _col(get, f, w)

            hand_l.append([
                t, _last(mp[: i + 1]), _mean(mp[w5]), _mean(mp[w15]),
                float(np.nanmin(mp[w15])), float(np.nanstd(mp[w15])), _slope(tm[w15], mp[w15]),
                _mean(mp[w30]), _slope(tm[w30], mp[w30]),
                _last(g("art_sbp", w15)), _last(g("art_dbp", w15)), _last(g("hr", w15)),
                _mean(g("hr", w5)), _slope(tm[w15], g("hr", w15)) if "hr" in get else np.nan,
                _last(g("spo2", w15)), _last(g("etco2", w15)), _last(g("rr", w15)),
                _last(g("tv", w15)), _last(g("pip", w15)), _last(g("bt", w30)),
            ])
            cid_l.append(cid)
            t_l.append(t)
            y_l.append(ys)
    return {"caseid": np.asarray(cid_l, np.int64), "t": np.asarray(t_l),
            "y": np.asarray(y_l, np.float64).reshape(-1, len(horizons)),
            "hand": np.asarray(hand_l, np.float64).reshape(-1, len(HAND))}


# ── encoder features ───────────────────────────────────────────────


@torch.no_grad()
def encoder_features(model: UniTTab, data: Path, caseids: list[int], samples: dict,
                     device: torch.device, batch_size: int) -> np.ndarray:
    schema = TableSchema.load(data / "schema.json")
    norm = Normalizer.load(schema, data / "normalizer.json")
    ctx = (schema.row_type_id["static"],) if "static" in schema.row_type_id else ()
    for rt in schema.row_types:
        if rt.name not in ("static",) and rt.fields[0] != "t_min":
            raise ValueError(f"row type {rt.name}: t_min must be the first slot")
    cases = load_cases(data, caseids)
    batcher = GPUWindowBatcher(cases, norm, model.cfg.max_rows, ctx, device)
    index = {cid: i for i, cid in enumerate(batcher.caseids)}
    dyn_t = {}
    for cid, c in cases.items():
        dyn = ~np.isin(c["row_type"], np.asarray(ctx))
        dyn_t[cid] = c["values"][dyn][:, 0].astype(np.float64)
    case_idx = np.array([index[c] for c in samples["caseid"]])
    end = np.array([np.searchsorted(dyn_t[c], t, side="right") - 1
                    for c, t in zip(samples["caseid"], samples["t"], strict=True)])
    if (end < 0).any():
        raise ValueError("a prediction time precedes every row of its case")
    model.eval()
    out = []
    amp = device.type == "cuda"
    for s in range(0, len(end), batch_size):
        ci = torch.as_tensor(case_idx[s : s + batch_size], device=device)
        ei = torch.as_tensor(end[s : s + batch_size], device=device)
        b = batcher.windows_ending_at(ci, ei)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            z = model.encode(b)
        rows, mask = z["rows"].float(), z["row_mask"]
        n = mask.sum(1).clamp(min=1)
        last = rows[torch.arange(len(n), device=device), n - 1]
        mean = (rows * mask[..., None]).sum(1) / n[:, None]
        out.append(torch.cat([z["cls"].float(), last, mean], 1).cpu())
    del batcher
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out).numpy()


# ── probes and metrics ─────────────────────────────────────────────


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s)) if y.sum() > 0 else np.nan


def bootstrap(y: np.ndarray, s: np.ndarray, groups: np.ndarray, n: int = 200,
              seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    roc, pr = [], []
    for _ in range(n):
        pick = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        roc.append(auroc(y[pick], s[pick]))
        pr.append(auprc(y[pick], s[pick]))
    q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731
    return {"auroc": auroc(y, s), "auroc_ci": q(roc), "auprc": auprc(y, s), "auprc_ci": q(pr)}


def linear_probe(xtr, ytr, xva, yva, xte, device, epochs: int = 30) -> tuple[np.ndarray, dict]:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    t = lambda x: torch.as_tensor((x - mu) / sd, dtype=torch.float32, device=device)  # noqa: E731
    xtr_t, xva_t, xte_t = t(xtr), t(xva), t(xte)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    best = {"auroc": -1.0}
    gen = torch.Generator(device=device).manual_seed(0)
    for wd in (1e-4, 1e-3, 1e-2, 1e-1):
        lin = torch.nn.Linear(xtr.shape[1], 1).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        for ep in range(epochs):
            perm = torch.randperm(len(ytr_t), device=device, generator=gen)
            for s in range(0, len(perm), 2048):
                i = perm[s : s + 2048]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    lin(xtr_t[i]).squeeze(-1), ytr_t[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                a = auroc(yva, lin(xva_t).squeeze(-1).cpu().numpy())
            if a > best["auroc"]:
                with torch.no_grad():
                    best = {"auroc": a, "wd": wd, "epoch": ep + 1,
                            "test": lin(xte_t).squeeze(-1).cpu().numpy()}
    return best.pop("test"), best


def gbm_probe(xtr, ytr, xte) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingClassifier

    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                         early_stopping=True, random_state=0)
    clf.fit(xtr, ytr)
    return clf.predict_proba(xte)[:, 1]


# ── main ───────────────────────────────────────────────────────────


def load_model(ckpt: str, device: torch.device, random_init: bool = False) -> UniTTab:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = UniTTab(UniTTabConfig(**ck["config"]))
    if not random_init:
        model.load_state_dict(ck["model"])
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label-data", required=True, help="cache with art_mbp (signal cache)")
    ap.add_argument("--model", nargs=3, action="append", default=[],
                    metavar=("NAME", "DATA", "CKPT"), help="encoder to probe (repeatable)")
    ap.add_argument("--random-control", action="store_true")
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 15])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-history", type=float, default=15.0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-cases", type=int, default=None, help="per split (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_data = Path(args.label_data)
    schema = TableSchema.load(label_data / "schema.json")
    split = json.loads((label_data / "split.json").read_text())
    ids = {p: sorted(split[p])[: args.max_cases] for p in PARTS}

    t0 = time.time()
    samples = {}
    for p in PARTS:
        samples[p] = build_samples(load_cases(label_data, ids[p]), schema, args.horizons,
                                   args.stride, args.min_history)
    counts = {p: {"samples": len(samples[p]["t"]),
                  "cases": int(np.unique(samples[p]["caseid"]).size),
                  **{f"prev_{h}": float(np.nanmean(samples[p]["y"][:, k]))
                     for k, h in enumerate(args.horizons)}} for p in PARTS}
    print(f"samples built in {time.time() - t0:.0f}s: {json.dumps(counts)}", flush=True)

    # score[name][part] = (n, D) features ; baselines are handled separately
    feats: dict[str, dict[str, np.ndarray]] = {}
    specs = [(n, Path(d), c, False) for n, d, c in args.model]
    if args.random_control and specs:
        n, d, c, _ = specs[0]
        specs.append((f"random_{n}", d, c, True))
    for name, data, ckpt, rnd in specs:
        t1 = time.time()
        model = load_model(ckpt, device, random_init=rnd)
        feats[name] = {p: encoder_features(model, data, ids[p], samples[p], device,
                                           args.batch_size) for p in PARTS}
        del model
        print(f"[{name}] features {feats[name]['train'].shape} in {time.time() - t1:.0f}s",
              flush=True)

    results: dict = {"counts": counts, "horizons": args.horizons, "methods": {}}
    te_groups = samples["test"]["caseid"]
    for k, h in enumerate(args.horizons):
        ok = {p: ~np.isnan(samples[p]["y"][:, k]) for p in PARTS}
        y = {p: samples[p]["y"][ok[p], k] for p in PARTS}
        g = te_groups[ok["test"]]
        res = {}
        # current MAP alone (lower MAP = higher risk; a monotone score needs no fitting)
        res["map_only"] = bootstrap(y["test"], -samples["test"]["hand"][ok["test"], 1], g)
        hand = {p: samples[p]["hand"][ok[p]] for p in PARTS}
        res["hand_gbm"] = bootstrap(y["test"], gbm_probe(hand["train"], y["train"],
                                                         hand["test"]), g)
        for name, f in feats.items():
            s, info = linear_probe(f["train"][ok["train"]], y["train"], f["val"][ok["val"]],
                                   y["val"], f["test"][ok["test"]], device)
            res[name] = {**bootstrap(y["test"], s, g), "val": info}
        results["methods"][str(h)] = res
        print(f"\n--- hypotension within {h} min: test n={len(y['test'])} "
              f"prevalence={y['test'].mean():.3f}")
        for m, r in res.items():
            print(f"  {m:<18} AUROC {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, "
                  f"{r['auroc_ci'][1]:.3f}]  AUPRC {r['auprc']:.3f} "
                  f"[{r['auprc_ci'][0]:.3f}, {r['auprc_ci'][1]:.3f}]", flush=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
