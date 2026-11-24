"""UniTTab masked-token pre-training on a prepared numerical-row cache.

    # CPU smoke test
    python scripts/pretrain.py --data data/vitaldb_numeric --preset tiny --steps 300 \
        --batch-size 32 --lr 3e-4 --out runs/smoke

    # one GPU
    python scripts/pretrain.py --data $DATA --preset base --steps 50000 --amp bf16 --out runs/base

    # all GPUs of a node (DDP)
    torchrun --nproc_per_node 8 scripts/pretrain.py --data $DATA --preset paper --amp bf16 ...

Paper defaults (UniTTab Tab. 16): AdamW, lr 5e-5, dropout 0.1, label smoothing ε 0.1,
p_f 0.15, p_r 0.1, neighborhood ±5 bins. A small model trained from scratch usually wants a
higher lr (1e-4 – 3e-4) with warmup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carmen_numeric import (  # noqa: E402
    PRESETS,
    Normalizer,
    TableSchema,
    UniTTab,
    UniTTabConfig,
    masked_token_loss,
    sample_mask,
)
from carmen_numeric.data.dataset import WindowDataset, collate, load_cases, to_device  # noqa: E402
from carmen_numeric.evaluate import evaluate, summarize_mae  # noqa: E402

KEY_FIELDS = ["hr", "spo2", "art_mbp", "nibp_mbp", "etco2", "bis", "bt", "ppf_ce", "lab_k",
              "lab_hct", "lab_gluc", "lab_lac", "age", "weight", "preop_cr"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="cache dir from prepare_vitaldb.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--max-rows", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64, help="per GPU")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--p-field", type=float, default=0.15)
    ap.add_argument("--p-row", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--windows-per-case", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"])
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def setup_dist() -> tuple[int, int, torch.device]:
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return rank, world, torch.device("cuda", local)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, 1, dev


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    prog = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    cos = 0.5 * (1 + math.cos(math.pi * prog))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cos)


def worker_init(_: int) -> None:
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def main() -> None:
    args = parse_args()
    rank, world, device = setup_dist()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    out = Path(args.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    data = Path(args.data)
    schema = TableSchema.load(data / "schema.json")
    norm = Normalizer.load(schema, data / "normalizer.json")
    split = json.loads((data / "split.json").read_text())
    ctx = (schema.row_type_id["static"],) if "static" in schema.row_type_id else ()

    train_cases = load_cases(data, split["train"])
    val_cases = load_cases(data, split["val"])
    train_ds = WindowDataset(train_cases, norm, args.max_rows, ctx, random_crop=True,
                             windows_per_case=args.windows_per_case, seed=args.seed + rank)
    val_ds = WindowDataset(val_cases, norm, args.max_rows, ctx, random_crop=False)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if world > 1 else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=sampler is None,
                          sampler=sampler, num_workers=args.workers, collate_fn=collate,
                          worker_init_fn=worker_init if args.workers else None, drop_last=True,
                          pin_memory=device.type == "cuda", persistent_workers=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    cfg = UniTTabConfig.from_schema(schema, norm, max_rows=args.max_rows, dropout=args.dropout,
                                    **PRESETS[args.preset])
    model = UniTTab(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98),
    )
    step = 0
    best = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best = ck["step"], ck.get("best", best)
    # heads of fields that are not masked in a batch get no gradient -> find_unused_parameters
    ddp = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device.index], find_unused_parameters=True) \
        if world > 1 else model
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")
    is_time = torch.tensor([False] + [f.is_time for f in schema.fields])
    gen = torch.Generator().manual_seed(args.seed * 1000 + rank)

    if is_main:
        print(f"device={device} world={world} params={n_params / 1e6:.2f}M "
              f"train windows/epoch={len(train_ds)} val windows={len(val_ds)} "
              f"cases train/val={len(train_cases)}/{len(val_cases)}", flush=True)
    log_f = open(out / "log.jsonl", "a") if is_main else None

    def save(tag: str, extra: dict) -> None:
        torch.save({
            "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
            "best": best, "config": cfg.to_dict(), "schema": schema.to_dict(),
            "normalizer": norm.to_dict(), "args": vars(args), **extra,
        }, out / f"{tag}.pt")

    model.train()
    t0 = time.time()
    run = {"loss": 0.0, "acc_num": 0.0, "acc_cat": 0.0, "n": 0}
    epoch = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in train_dl:
            if step >= args.steps:
                break
            masked = sample_mask(batch, is_time, args.p_field, args.p_row, generator=gen)
            if not masked.any():
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args)
            b = to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                res = ddp(b, masked.to(device))
            loss, st = masked_token_loss(res["predictions"], cfg.field_kinds, args.eps, args.radius)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            run["loss"] += loss.item()
            run["acc_num"] += st["acc_num"]
            run["acc_cat"] += st["acc_cat"]
            run["n"] += 1

            if is_main and step % args.log_every == 0:
                rec = {"step": step, "lr": lr_at(step, args), "gnorm": float(gnorm),
                       "sec": round(time.time() - t0, 1),
                       **{k: run[k] / run["n"] for k in ("loss", "acc_num", "acc_cat")}}
                print(json.dumps(rec), flush=True)
                log_f.write(json.dumps({"split": "train", **rec}) + "\n")
                log_f.flush()
                run = {"loss": 0.0, "acc_num": 0.0, "acc_cat": 0.0, "n": 0}

            if step % args.eval_every == 0 or step == args.steps:
                if is_main:
                    ev = evaluate(model, val_dl, norm, device, args.p_field, args.p_row,
                                  args.eps, args.radius, max_batches=args.eval_batches,
                                  amp_dtype=amp_dtype)
                    head = {k: ev[k] for k in ("loss", "acc_num", "acc_num_pm2", "acc_cat")}
                    print(f"[eval step {step}] {json.dumps(head)}")
                    print(summarize_mae(ev["mae"], KEY_FIELDS), flush=True)
                    log_f.write(json.dumps({"split": "val", "step": step, **ev}) + "\n")
                    log_f.flush()
                    if ev["loss"] < best:
                        best = ev["loss"]
                        save("best", {"val": head})
                    save("last", {"val": head})
                if world > 1:
                    dist.barrier()
        epoch += 1

    if is_main:
        log_f.close()
        print(f"finished {step} steps in {time.time() - t0:.0f}s; best val loss {best:.4f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
