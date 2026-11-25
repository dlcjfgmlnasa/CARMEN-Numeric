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

Missing modalities: ``--drop-wave`` / ``--drop-signal`` hide all waveform tokens of a sample
/ the tokens of one signal at random during training (modality dropout); ``--subset-train``
draws a signal subset per sample instead. Test scores are computed for every signal subset
(``eap``, ``ea``, ``ep`` = no arterial line, ``ap``, ``e``, ``a``, ``p``, ``none``). With
``--token-mode joint`` subsets are made by masking tokens of a joint encoding (leaky: the kept
tokens saw the other signals); ``subsets`` uses a separate CARMEN encoding per subset and
``independent`` single-signal encodings (both exact). ``--eval-only CKPT`` scores an existing
``best.pt`` without training.
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
SLOT = {"ecg": 0, "abp": 1, "ppg": 2}  # token slots = CARMEN signal types
# signal subsets as bitmasks (ECG 1, ABP 2, PPG 4) and their token-file names
SUBSET_NAME = {1: "e", 2: "a", 3: "ea", 4: "p", 5: "ep", 6: "ap", 7: "eap"}
COND_MASK = {"eap": 7, "ea": 3, "ep": 5, "ap": 6, "e": 1, "a": 2, "p": 4, "none": 0}
CONDITIONS = tuple(COND_MASK)
ALIASES = {"full": "eap", "no_wave": "none", "no_abp": "ep", "abp_only": "a"}


def slot_mask(bits: torch.Tensor) -> torch.Tensor:  # (B,) bitmask -> (B, 36) allowed slots
    sig = torch.stack([(bits >> k) & 1 for k in range(N_SIG)], -1).bool()  # (B, 3)
    return sig[..., None].expand(-1, -1, N_CHUNK).reshape(len(bits), N_SIG * N_CHUNK)


def apply_condition(valid: torch.Tensor, cond: str) -> torch.Tensor:  # (B, 36) -> (B, 36)
    """Mask-based condition (on jointly encoded tokens this leaks — see TokenSource)."""
    bits = torch.full((len(valid),), COND_MASK[ALIASES.get(cond, cond)], device=valid.device)
    return valid & slot_mask(bits)


class TokenSource:
    """CARMEN tokens of one split; ``get(idx, bits)`` gives tokens of a signal subset.

    ``joint``: one joint encoding of all signals; subsets by masking tokens (other signals'
    information stays in the kept tokens). ``subsets``: a separate encoding per subset
    (``<split>_<subset>_tokens.npy`` from ``extract_carmen_tokens.py --subsets all``) —
    exact. ``independent``: the single-signal encodings side by side; subsets by masking,
    exact because no signal saw another.
    """

    def __init__(self, path: Path, part: str, samples: dict, device, mode: str) -> None:
        meta = np.load(path / f"{part}_meta.npz")
        if not (np.array_equal(meta["caseid"], samples["caseid"])
                and np.allclose(meta["t"], samples["t"])):
            raise ValueError(f"{part}: token order does not match the samples")
        load = lambda name: (  # noqa: E731
            torch.from_numpy(np.load(path / f"{name}_tokens.npy")).to(device),
            torch.from_numpy(np.load(path / f"{name}_valid.npy")).to(device))
        self.mode = mode
        if mode == "joint":
            self.tok, self.valid = load(part)
        elif mode == "independent":
            parts = [load(f"{part}_{k}") for k in ("e", "a", "p")]
            self.tok = parts[0][0] + parts[1][0] + parts[2][0]  # slot-disjoint, zeros elsewhere
            self.valid = parts[0][1] | parts[1][1] | parts[2][1]
        elif mode == "subsets":
            loaded = [load(f"{part}_{SUBSET_NAME[b]}") for b in range(1, 8)]
            self.tok_all = torch.stack([t for t, _ in loaded])  # (7, n, 36, d)
            self.valid_all = torch.stack([v for _, v in loaded])
            self.tok, self.valid = self.tok_all[6], self.valid_all[6]  # all signals
        else:
            raise ValueError(mode)
        sig = self.valid.view(-1, N_SIG, N_CHUNK).any(-1)  # (n, 3) signal present
        self.present_bits = (sig.long() * torch.tensor([1, 2, 4], device=device)).sum(-1)

    def get(self, idx: torch.Tensor, bits: torch.Tensor | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        if bits is None:
            return self.tok[idx], self.valid[idx]
        bits = bits & self.present_bits[idx]
        if self.mode != "subsets":
            return self.tok[idx], self.valid[idx] & slot_mask(bits)
        k = (bits - 1).clamp(min=0)
        tok, valid = self.tok_all[k, idx], self.valid_all[k, idx]
        return tok, valid & (bits > 0)[:, None]


def sample_subsets(present: torch.Tensor, p_full: float, p_none: float,
                   gen: torch.Generator) -> torch.Tensor:
    """Per-sample signal subset for training: all available signals with ``p_full``, none
    with ``p_none``, otherwise a random non-empty subset of the available ones."""
    b = len(present)
    u = torch.rand(b, generator=gen, device=present.device)
    r = torch.randint(1, 8, (b,), generator=gen, device=present.device) & present
    r = torch.where(r == 0, present, r)
    return torch.where(u < p_full, present, torch.where(u > 1 - p_none, torch.zeros_like(r), r))


def modality_dropout(valid: torch.Tensor, p_wave: float, p_signal: float,
                     gen: torch.Generator) -> torch.Tensor:
    v = valid.clone().view(-1, N_SIG, N_CHUNK)
    b = v.shape[0]
    if p_signal > 0:
        drop = torch.rand(b, N_SIG, generator=gen, device=v.device) < p_signal
        v &= ~drop[..., None]
    if p_wave > 0:
        drop = torch.rand(b, generator=gen, device=v.device) < p_wave
        v &= ~drop[:, None, None]
    return v.view(-1, N_SIG * N_CHUNK)


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


@torch.no_grad()
def predict_cond(model, win, src: TokenSource, cond: str, n: int, device,
                 batch_size: int = 512) -> np.ndarray:
    """Logits with the tokens of one signal subset (condition name or alias)."""
    model.eval()
    out = []
    bits_all = COND_MASK[ALIASES.get(cond, cond)]
    for s in range(0, n, batch_size):
        idx = torch.arange(s, min(s + batch_size, n), device=device)
        bits = torch.full((len(idx),), bits_all, device=device, dtype=torch.long)
        tok, valid = src.get(idx, bits)
        b = win.batch(idx) if win is not None else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model(b, tok, valid).float().cpu())
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
    ap.add_argument("--drop-wave", type=float, default=0.0,
                    help="training: probability of hiding all waveform tokens of a sample")
    ap.add_argument("--drop-signal", type=float, default=0.0,
                    help="training: probability of hiding one signal's tokens")
    ap.add_argument("--eval-only", default=None,
                    help="best.pt of an earlier run: skip training, score test conditions")
    ap.add_argument("--token-mode", default="joint", choices=["joint", "subsets", "independent"])
    ap.add_argument("--subset-train", nargs=2, type=float, default=None,
                    metavar=("P_FULL", "P_NONE"),
                    help="training: per-sample random signal subsets (instead of --drop-*)")
    ap.add_argument("--select-conds", nargs="+", default=["eap"],
                    help="val conditions whose mean AUROC picks the checkpoint")
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
    toks = {p: TokenSource(tok_dir, p, samples[p], device, args.token_mode) for p in P.PARTS}
    d_carmen = toks["train"].tok.shape[-1]
    wave_rate = {p: float(toks[p].valid.any(1).float().mean()) for p in P.PARTS}
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
    if args.eval_only:
        ck = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        if ck["mode"] != args.mode:
            raise ValueError(f"{args.eval_only} is a {ck['mode']} model")
        best_state, best_step = ck["model"], ck["step"]
    for step in range(1, 0 if args.eval_only else args.steps + 1):
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g, lr in zip(opt.param_groups, base_lr, strict=True):
            g["lr"] = lr * max(f, 0.05)
        idx = torch.randint(counts["train"], (args.batch_size,), device=device, generator=gen)
        yy = y["train"][idx]
        ok = ~torch.isnan(yy)
        b = win["train"].batch(idx) if win["train"] is not None else None
        if args.subset_train:
            bits = sample_subsets(toks["train"].present_bits[idx], *args.subset_train, gen)
            tt, tv = toks["train"].get(idx, bits)
        else:
            tt, tv = toks["train"].get(idx)
            tv = modality_dropout(tv, args.drop_wave, args.drop_signal, gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logit = model(b, tt, tv).float()
        loss = nn.functional.binary_cross_entropy_with_logits(logit[ok],
                                                              torch.nan_to_num(yy)[ok])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            va = float(np.mean([
                F.mean_auroc(samples["val"]["y"], predict_cond(model, win["val"], toks["val"],
                                                               c, counts["val"], device))
                for c in args.select_conds]))
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
    te = samples["test"]
    cond_logits = {}
    for cond in CONDITIONS:
        cond_logits[cond] = predict_cond(model, win["test"], toks["test"], cond,
                                         counts["test"], device)
        np.savez(out / f"test_scores_{cond}.npz", logit=cond_logits[cond], caseid=te["caseid"],
                 t=te["t"])
    logit = cond_logits["eap"]
    np.savez(out / "test_scores.npz", logit=logit, caseid=te["caseid"], t=te["t"],
             has_wave=toks["test"].valid.any(1).cpu().numpy())

    res: dict = {"mode": args.mode, "best_step": best_step, "best_val_auroc": round(best, 4),
                 "log": log, "counts": counts, "with_waveform": wave_rate, "args": vars(args),
                 "horizons": {}, "conditions": {}}
    for cond, lg in cond_logits.items():
        res["conditions"][cond] = {}
        for k, h in enumerate(F.HORIZONS):
            ok = ~np.isnan(te["y"][:, k])
            res["conditions"][cond][str(h)] = P.bootstrap(
                te["y"][ok, k], lg[ok, k], te["caseid"][ok], n=500)
        print(f"[{cond}] " + "  ".join(
            f"{h}m AUROC {r['auroc']:.3f} AUPRC {r['auprc']:.3f}"
            for h, r in res["conditions"][cond].items()), flush=True)
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
    if not args.eval_only:
        torch.save({"model": best_state, "mode": args.mode, "step": best_step,
                    "args": vars(args)}, out / "best.pt")
    print(f"wrote {out}/result.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
