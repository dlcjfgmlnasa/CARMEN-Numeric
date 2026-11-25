"""Hypotension prediction from minute-level numbers (UniTTab) + CARMEN waveform tokens.

    python scripts/finetune_fusion.py --mode fusion --data $SIG --label-data $SIG \
        --ckpt runs/unittab_signal_v1/best.pt --tokens runs/../carmen_tokens --out runs/fusion
    python scripts/finetune_fusion.py --mode wave ...   # waveform tokens only
    python scripts/finetune_fusion.py --mode late ...   # concatenate pooled numbers + waveform

Samples, labels and splits are those of ``probe_ioh.py``; waveform tokens come from
``extract_carmen_tokens.py`` (frozen CARMEN, the 60 s ending at the prediction time,
``3 signals x 12 five-second tokens``) in the same sample order.

``fusion``: the numeric encoder (pre-trained, fine-tuned here) yields three query tokens
``[CLS, last row, mean of rows]``; two cross-attention blocks (self-attention among the
queries, then attention to the projected waveform tokens + a learned "no waveform" token)
update them, and an MLP gives one logit per horizon. ``wave``: the same blocks with learned
queries and no numeric encoder. ``late``: the numeric features concatenated with mean-pooled
waveform tokens. CARMEN stays frozen. The step with the best mean val AUROC is kept; test
metrics, curves and logits (``test_scores.npz``) are written.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import finetune_ioh as F  # noqa: E402
import probe_ioh as P  # noqa: E402
import viz_predictions as V  # noqa: E402

from carmen_numeric import TableSchema  # noqa: E402
from carmen_numeric.data.dataset import load_cases  # noqa: E402

N_SIG, N_CHUNK = 3, 12


class WaveTokens(nn.Module):
    """CARMEN tokens (B, 36, d_c) -> (B, 37, m) with signal / time embeddings + a null token."""

    def __init__(self, d_carmen: int, m: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_carmen), nn.Linear(d_carmen, m))
        self.sig = nn.Embedding(N_SIG, m)
        self.pos = nn.Embedding(N_CHUNK, m)
        self.null = nn.Parameter(torch.zeros(1, 1, m))
        idx = torch.arange(N_SIG * N_CHUNK)
        self.register_buffer("sig_idx", idx // N_CHUNK, persistent=False)
        self.register_buffer("pos_idx", idx % N_CHUNK, persistent=False)

    def forward(self, tok: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(tok.float()) + self.sig(self.sig_idx) + self.pos(self.pos_idx)
        b = x.shape[0]
        x = torch.cat([self.null.expand(b, -1, -1), x], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~valid], 1)
        return x, pad  # pad: True = ignore


class CrossBlocks(nn.Module):
    def __init__(self, m: int, layers: int = 2, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(m, heads, 4 * m, dropout, batch_first=True,
                                       norm_first=True, activation="gelu")
            for _ in range(layers)])
        self.norm = nn.LayerNorm(m)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            q = blk(q, mem, memory_key_padding_mask=pad)
        return self.norm(q)


class FusionModel(nn.Module):
    def __init__(self, mode: str, encoder, d_carmen: int, m: int, n_out: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.mode = mode
        self.encoder = encoder if mode != "wave" else None
        self.wave = WaveTokens(d_carmen, m)
        if mode in ("fusion", "wave"):
            self.cross = CrossBlocks(m, dropout=dropout)
        if mode == "wave":
            self.queries = nn.Parameter(torch.randn(1, 3, m) * 0.02)
        width = {"fusion": 3 * m, "wave": 3 * m, "late": 4 * m}[mode]
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Dropout(dropout),
                                  nn.Linear(width, 256), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(256, n_out))

    def numeric_queries(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # (B, 3, m)
        z = self.encoder.encode(batch)
        rows, mask = z["rows"].float(), z["row_mask"]
        n = mask.sum(1).clamp(min=1)
        last = rows[torch.arange(len(n), device=rows.device), n - 1]
        mean = (rows * mask[..., None]).sum(1) / n[:, None]
        return torch.stack([z["cls"].float(), last, mean], 1)

    def forward(self, batch, tok: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mem, pad = self.wave(tok, valid)
        if self.mode == "late":
            q = self.numeric_queries(batch)
            keep = (~pad).float()[..., None]
            w = (mem * keep).sum(1) / keep.sum(1)  # null token is always kept
            return self.head(torch.cat([q.flatten(1), w], 1))
        q = self.numeric_queries(batch) if self.mode == "fusion" \
            else self.queries.expand(tok.shape[0], -1, -1)
        return self.head(self.cross(q, mem, pad).flatten(1))


def load_tokens(path: Path, part: str, samples: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    meta = np.load(path / f"{part}_meta.npz")
    if not (np.array_equal(meta["caseid"], samples["caseid"])
            and np.allclose(meta["t"], samples["t"])):
        raise ValueError(f"{part}: token order does not match the samples")
    tok = torch.from_numpy(np.load(path / f"{part}_tokens.npy")).to(device)
    valid = torch.from_numpy(np.load(path / f"{part}_valid.npy")).to(device)
    return tok, valid


@torch.no_grad()
def predict(model, win, tok, valid, n, device, batch_size=512) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n), device=device)
        b = win.batch(idx) if win is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(b, tok[idx], valid[idx]).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=["fusion", "wave", "late"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--label-data", required=True)
    ap.add_argument("--ckpt", required=True, help="pre-trained UniTTab (also sets d_model)")
    ap.add_argument("--tokens", required=True, help="extract_carmen_tokens.py output dir")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--example-cases", type=int, nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, label_data, tok_dir = Path(args.data), Path(args.label_data), Path(args.tokens)
    lschema = TableSchema.load(label_data / "schema.json")
    split = json.loads((label_data / "split.json").read_text())
    ids = {p: sorted(split[p])[: args.max_cases] for p in P.PARTS}
    t0 = time.time()
    samples = {p: P.build_samples(load_cases(label_data, ids[p]), lschema, F.HORIZONS, 5, 15.0)
               for p in P.PARTS}
    counts = {p: len(samples[p]["t"]) for p in P.PARTS}
    toks = {p: load_tokens(tok_dir, p, samples[p], device) for p in P.PARTS}
    d_carmen = toks["train"][0].shape[-1]
    wave_rate = {p: float(toks[p][1].any(1).float().mean()) for p in P.PARTS}
    print(f"samples {counts}; with waveform {wave_rate} ({time.time() - t0:.0f}s)", flush=True)

    encoder = P.load_model(args.ckpt, device)
    m = encoder.cfg.d_model
    win = {p: F.Windows(data, list(samples[p]["caseid"]), samples[p]["caseid"],
                        samples[p]["t"], encoder.cfg.max_rows, device)
           for p in P.PARTS} if args.mode != "wave" else {p: None for p in P.PARTS}
    model = FusionModel(args.mode, encoder, d_carmen, m, len(F.HORIZONS)).to(device)
    enc_params = list(model.encoder.parameters()) if model.encoder is not None else []
    enc_ids = {id(p) for p in enc_params}
    other = [p for p in model.parameters() if id(p) not in enc_ids]
    groups = [{"params": other, "lr": args.lr_head}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    base_lr = [g["lr"] for g in opt.param_groups]
    y = {p: torch.as_tensor(samples[p]["y"], dtype=torch.float32, device=device)
         for p in P.PARTS}
    gen = torch.Generator(device=device).manual_seed(args.seed)
    best, best_state, best_step, bad, log = -1.0, None, 0, 0, []
    model.train()
    for step in range(1, args.steps + 1):
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g, lr in zip(opt.param_groups, base_lr, strict=True):
            g["lr"] = lr * max(f, 0.05)
        idx = torch.randint(counts["train"], (args.batch_size,), device=device, generator=gen)
        yy = y["train"][idx]
        ok = ~torch.isnan(yy)
        b = win["train"].batch(idx) if win["train"] is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(b, toks["train"][0][idx], toks["train"][1][idx]).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logit[ok],
                                                              torch.nan_to_num(yy)[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            va = F.mean_auroc(samples["val"]["y"], predict(model, win["val"], *toks["val"],
                                                           counts["val"], device))
            log.append({"step": step, "loss": round(loss.item(), 4), "val_auroc": round(va, 4)})
            print(json.dumps(log[-1]) + f" ({time.time() - t0:.0f}s)", flush=True)
            if va > best:
                best, best_step, bad = va, step, 0
                best_state = copy.deepcopy({k: v.detach().cpu()
                                            for k, v in model.state_dict().items()})
            else:
                bad += 1
                if bad >= args.patience:
                    print(f"early stop at {step}", flush=True)
                    break
    model.load_state_dict(best_state)
    logit = predict(model, win["test"], *toks["test"], counts["test"], device)
    te = samples["test"]
    np.savez(out / "test_scores.npz", logit=logit, caseid=te["caseid"], t=te["t"],
             has_wave=toks["test"][1].any(1).cpu().numpy())

    res: dict = {"mode": args.mode, "best_step": best_step, "best_val_auroc": round(best, 4),
                 "log": log, "counts": counts, "with_waveform": wave_rate, "args": vars(args),
                 "horizons": {}}
    for k, h in enumerate(F.HORIZONS):
        ok = {p: ~np.isnan(samples[p]["y"][:, k]) for p in P.PARTS}
        yt = te["y"][ok["test"], k]
        g = te["caseid"][ok["test"]]
        hand = {p: samples[p]["hand"][ok[p]] for p in P.PARTS}
        scores = {
            args.mode: 1 / (1 + np.exp(-logit[ok["test"], k])),
            "hand_gbm": P.gbm_probe(hand["train"], samples["train"]["y"][ok["train"], k],
                                    hand["test"]),
            "map_only": -hand["test"][:, 1],
        }
        res["horizons"][str(h)] = {
            "n": int(len(yt)), "prevalence": round(float(yt.mean()), 4),
            "methods": {mm: {**P.bootstrap(yt, s, g), **V.curve_points(yt, s),
                             "operating": V.operating_points(yt, s, per_hour=12)}
                        for mm, s in scores.items()},
            "calibration": {mm: V.calibration(yt, scores[mm]) for mm in (args.mode, "hand_gbm")},
        }
        r = res["horizons"][str(h)]["methods"]
        print(f"--- {h} min: " + "  ".join(
            f"{mm} {v['auroc']:.3f} [{v['auroc_ci'][0]:.3f},{v['auroc_ci'][1]:.3f}] "
            f"AUPRC {v['auprc']:.3f}" for mm, v in r.items()), flush=True)
    res["examples"] = [
        {"caseid": int(c), "risk_t": V._r(te["t"][te["caseid"] == c], 1),
         "risk": {str(h): V._r(1 / (1 + np.exp(-logit[te["caseid"] == c, k])))
                  for k, h in enumerate(F.HORIZONS)}}
        for c in args.example_cases if (te["caseid"] == c).any()]
    (out / "result.json").write_text(json.dumps(res))
    torch.save({"model": best_state, "mode": args.mode, "step": best_step, "args": vars(args)},
               out / "best.pt")
    print(f"wrote {out}/result.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
