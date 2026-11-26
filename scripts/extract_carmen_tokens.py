"""Frozen CARMEN waveform tokens for every hypotension-probe sample (run where the data is).

    python scripts/extract_carmen_tokens.py --label-data $SIG --vital-dir $BASE/vitaldb/raw \
        --cases-csv $BASE/aki/csv/clinical_data.csv --carmen-src $R/carmen_src \
        --weights $R/carmen_weights/carmen-base.pt --out $R/carmen_tokens

For each sample of ``probe_ioh.py`` (prediction time ``t``, minute row ``k = floor(t)``) the
waveform window is the 60 s that end where the minute row ends — ``[start + 60 k,
start + 60 (k + 1))`` seconds of the .vital timeline, ``start`` = anesthesia start — so it
holds nothing after ``t``. Arterial pressure, ECG lead II and PPG (``SNUADC/ART``,
``SNUADC/ECG_II``, ``SNUADC/PLETH``, 500 Hz) are preprocessed as in CARMEN pre-training
(range check, median filter for ABP/PPG, 60 Hz notch for ECG/PPG, Butterworth band/low-pass,
``resample_poly`` to 100 Hz) on the window plus 2 s of past padding. A signal is absent for
a sample when > 30 % of its window is missing or flat; small gaps are interpolated.
``--signals`` restricts what CARMEN sees: since the signals of a sample are encoded jointly,
masking one signal's tokens afterwards still leaves its information in the others' tokens —
to emulate a missing arterial line, re-encode with ``--signals ecg ppg``. ``--subsets all``
encodes every non-empty subset of the three signals on its own in one pass (the waveforms
are read and filtered once); the single-signal subsets are independent encodings.

CARMEN (frozen, ``task="masked"``, any-variate packing: the three signals of a sample attend
to each other) encodes 240 patches (0.25 s) per signal; patches are mean-pooled into 5 s
tokens (``--chunk-s``), giving ``(3 signals x 12 chunks, d_model)`` per sample. Writes per split
``<split>_tokens.npy`` (float16, ``(n, 36, d)``), ``<split>_valid.npy`` (``(n, 36)``) and
``<split>_meta.npz`` (caseid, t, presence) in the sample order of the task
(``downstream_tasks.build_task``; ``--task``). Case-level tasks give 12 windows spread over the
case per sample; each window becomes one token per signal (slot = signal x window).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import downstream_tasks as T  # noqa: E402
import probe_ioh as P  # noqa: E402

from carmen_numeric import TableSchema  # noqa: E402
from carmen_numeric.data.dataset import load_cases  # noqa: E402

NATIVE_SR, TARGET_SR = 500, 100
WIN_S, PAD_S = 60, 2
CHUNK_S = 5
# signal_type of CARMEN (0 ECG, 1 ABP, 2 PPG) doubles as the token slot
SIGNALS = [  # (slot / CARMEN signal_type, track, valid range, median k, notch, filter)
    (0, "SNUADC/ECG_II", (-5.0, 5.0), 0, 60.0, ("band", 0.5, 40.0)),
    (1, "SNUADC/ART", (20.0, 300.0), 5, None, ("low", 15.0)),
    (2, "SNUADC/PLETH", (0.0, 2000.0), 5, 60.0, ("low", 8.0)),
]
N_SIG = len(SIGNALS)
N_CHUNK = WIN_S // CHUNK_S
N_WIN = WIN_S * TARGET_SR


def _interp_nan(x: np.ndarray) -> np.ndarray:
    bad = np.isnan(x)
    if not bad.any():
        return x
    idx = np.flatnonzero(~bad)
    out = x.copy()
    out[bad] = np.interp(np.flatnonzero(bad), idx, x[idx])
    return out


def preprocess(x: np.ndarray, rng: tuple, median_k: int, notch: float | None,
               filt: tuple) -> np.ndarray | None:
    """Native-rate window (with past padding) -> 100 Hz window of N_WIN samples, or None."""
    from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample_poly, sosfiltfilt

    x = x.astype(np.float64)
    x[(x < rng[0]) | (x > rng[1])] = np.nan
    core = x[-WIN_S * NATIVE_SR:]
    if np.isnan(core).mean() > 0.3:
        return None
    if np.nanstd(core) < 1e-6:  # flat line (disconnected, flushed)
        return None
    x = _interp_nan(x)
    if median_k:
        x = medfilt(x, median_k)
    nyq = NATIVE_SR / 2
    if notch is not None:
        b, a = iirnotch(notch / nyq, 30.0)
        x = filtfilt(b, a, x)
    if filt[0] == "band":
        sos = butter(4, [filt[1] / nyq, filt[2] / nyq], btype="band", output="sos")
    else:
        sos = butter(4, filt[1] / nyq, btype="low", output="sos")
    x = sosfiltfilt(sos, x)
    y = resample_poly(x, TARGET_SR, NATIVE_SR)
    y = y[-N_WIN:]
    return y.astype(np.float32) if len(y) == N_WIN and np.isfinite(y).all() else None


def case_windows(args: tuple) -> tuple[int, np.ndarray, np.ndarray]:
    """(caseid, window ends in s, vital dir, kept slots) -> (caseid, (n, 3, N_WIN), (n, 3))."""
    caseid, ends, vital_dir, keep = args
    import vitaldb

    n = len(ends)
    wave = np.zeros((n, N_SIG, N_WIN), np.float32)
    present = np.zeros((n, N_SIG), bool)
    path = Path(vital_dir) / f"{caseid}.vital"
    if not path.exists():
        return caseid, wave, present
    try:
        vf = vitaldb.VitalFile(str(path), [s[1] for s in SIGNALS])
    except Exception:
        return caseid, wave, present
    names = set(vf.get_track_names())
    for slot, track, rng, mk, notch, filt in SIGNALS:
        if track not in names or slot not in keep:
            continue
        x = vf.to_numpy([track], 1 / NATIVE_SR)
        if x is None or x.size == 0:
            continue
        x = x[:, 0]
        for i, e in enumerate(ends):
            hi = int(e * NATIVE_SR)
            lo = hi - (WIN_S + PAD_S) * NATIVE_SR
            if lo < 0 or hi > len(x):
                continue
            y = preprocess(x[lo:hi], rng, mk, notch, filt)
            if y is not None:
                wave[i, slot] = y
                present[i, slot] = True
    return caseid, wave, present


class CarmenEncoder:
    def __init__(self, src: str, weights: str, device) -> None:
        import torch

        sys.path.insert(0, src)
        from carmen import CARMEN, make_batch
        from carmen.batch import to_device

        self.torch = torch
        self.make_batch = make_batch
        self.to_device = to_device
        self.model = CARMEN.from_pretrained(weights).to(device).eval()
        self.device = device
        self.patch = int(getattr(self.model, "patch_size", 25))
        self.d = None
        probe = np.zeros((1, N_SIG, N_WIN), np.float32)
        probe[0, 1] = 80 + 20 * np.sin(np.linspace(0, 120 * np.pi, N_WIN))
        self(probe, np.array([[False, True, False]]))  # sets self.d

    def __call__(self, wave: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(B, 3, N_WIN), (B, 3) -> tokens (B, 36, d) float16, valid (B, 36)."""
        torch = self.torch
        rows, keep = [], []
        for b in range(len(wave)):
            sig = [(slot, torch.from_numpy(wave[b, slot])) for slot, *_ in SIGNALS
                   if present[b, slot]]
            if sig:
                rows.append(self.make_batch(sig, patch_size=self.patch, max_length=N_SIG * N_WIN
                                            + self.patch))
                keep.append(b)
        n_tok = N_SIG * N_CHUNK
        if not rows:
            return None, np.zeros((len(wave), n_tok), bool)
        batch = _stack_batches(rows)
        batch = self.to_device(batch, self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=self.device.type == "cuda"):
            out = self.model(batch, task="masked")
        enc = out["encoded"].float()  # (R, N, d)
        st = out["patch_signal_types"].long()
        tid = out["time_id"].long()
        pm = out["patch_mask"].bool()
        d = enc.shape[-1]
        self.d = d
        patches_per_chunk = CHUNK_S * TARGET_SR // self.patch
        slot = (st * N_CHUNK + (tid // patches_per_chunk).clamp(max=N_CHUNK - 1))
        slot = torch.where(pm & (st < N_SIG), slot, torch.full_like(slot, n_tok))
        r = enc.shape[0]
        sums = torch.zeros(r, n_tok + 1, d, device=enc.device)
        cnt = torch.zeros(r, n_tok + 1, device=enc.device)
        sums.scatter_add_(1, slot[..., None].expand(-1, -1, d), enc)
        cnt.scatter_add_(1, slot, torch.ones_like(slot, dtype=torch.float32))
        tok = (sums[:, :n_tok] / cnt[:, :n_tok, None].clamp(min=1)).half().cpu().numpy()
        val = (cnt[:, :n_tok] > 0).cpu().numpy()
        full = np.zeros((len(wave), n_tok, d), np.float16)
        fval = np.zeros((len(wave), n_tok), bool)
        full[keep], fval[keep] = tok, val
        return full, fval


def _stack_batches(rows: list):
    """Stack single-row PackedBatches into one batch (rows padded to a common width)."""
    import torch
    from carmen.data.collate import PackedBatch

    width = max(r.values.shape[1] for r in rows)

    def pad(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(t, (0, width - t.shape[1]))

    cat = lambda name: torch.cat([getattr(r, name) for r in rows])  # noqa: E731
    return PackedBatch(
        values=torch.cat([pad(r.values) for r in rows]),
        sample_id=torch.cat([pad(r.sample_id) for r in rows]),
        variate_id=torch.cat([pad(r.variate_id) for r in rows]),
        lengths=cat("lengths"), sampling_rates=cat("sampling_rates"),
        signal_types=cat("signal_types"),
        padded_lengths=None if rows[0].padded_lengths is None else cat("padded_lengths"),
        start_samples=None if rows[0].start_samples is None else cat("start_samples"),
    )


# signal subsets by name (slot 0 = ECG, 1 = ABP, 2 = PPG)
SUBSETS = {"e": (0,), "a": (1,), "p": (2,), "ea": (0, 1), "ep": (0, 2), "ap": (1, 2),
           "eap": (0, 1, 2)}


class TokenWriter:
    """Buffers per-case waveforms, runs CARMEN in batches, writes .npy memmaps.

    ``outputs`` maps a file prefix to the signal slots CARMEN may see for it: one entry
    (``"<split>"``) for the default joint encoding, or one per signal subset
    (``"<split>_<subset>"``) so that each subset is encoded on its own.
    """

    def __init__(self, enc: CarmenEncoder, out: Path, outputs: dict[str, tuple[int, ...]],
                 n: int, batch_size: int, snap: int = 1) -> None:
        """``snap`` > 1: every sample has ``snap`` consecutive windows (case-level tasks); each
        window is pooled to one token per signal and the windows fill the chunk axis."""
        self.enc, self.out, self.outputs, self.n = enc, out, outputs, n
        self.snap = snap
        self.bs = max(snap, batch_size // snap * snap)
        self.tokens: dict[str, np.ndarray] = {}
        self.valid = {k: np.zeros((n, N_SIG * N_CHUNK), bool) for k in outputs}
        self.buf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, wave: np.ndarray, present: np.ndarray, idx: np.ndarray) -> None:
        self.buf.append((wave, present, idx))
        if sum(len(b[2]) for b in self.buf) >= 8 * self.bs:
            self.flush()

    def _memmap(self, prefix: str) -> np.ndarray:
        if prefix not in self.tokens:
            self.tokens[prefix] = np.lib.format.open_memmap(
                self.out / f"{prefix}_tokens.npy", mode="w+", dtype=np.float16,
                shape=(self.n, N_SIG * N_CHUNK, self.enc.d))
        return self.tokens[prefix]

    def flush(self) -> None:
        if not self.buf:
            return
        w = np.concatenate([b[0] for b in self.buf])
        p = np.concatenate([b[1] for b in self.buf])
        ii = np.concatenate([b[2] for b in self.buf])
        self.buf.clear()
        for prefix, slots in self.outputs.items():
            allow = np.zeros(N_SIG, bool)
            allow[list(slots)] = True
            for s0 in range(0, len(w), self.bs):
                sl = slice(s0, s0 + self.bs)
                tk, vl = self.enc(w[sl], p[sl] & allow)
                if tk is None:
                    continue
                rows = ii[sl]
                if self.snap > 1:
                    tk, vl = self._pool_snapshots(tk, vl)
                    rows = rows[:: self.snap]
                self._memmap(prefix)[rows] = tk
                self.valid[prefix][rows] = vl

    def _pool_snapshots(self, tk: np.ndarray, vl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(n*K, 36, d), (n*K, 36) -> (n, 36, d), (n, 36): slot = signal * K + window."""
        k, d = self.snap, tk.shape[-1]
        t = tk.astype(np.float32).reshape(-1, k, N_SIG, N_CHUNK, d)
        w = vl.reshape(-1, k, N_SIG, N_CHUNK).astype(np.float32)
        pooled = (t * w[..., None]).sum(3) / np.maximum(w.sum(3), 1)[..., None]  # (n, K, 3, d)
        valid = w.sum(3) > 0  # (n, K, 3)
        return (pooled.transpose(0, 2, 1, 3).reshape(-1, N_SIG * k, d).astype(np.float16),
                valid.transpose(0, 2, 1).reshape(-1, N_SIG * k))

    def close(self) -> None:
        self.flush()
        for prefix in self.outputs:
            self._memmap(prefix).flush()  # creates an all-invalid file if nothing was written
            np.save(self.out / f"{prefix}_valid.npy", self.valid[prefix])


def run_split(part: str, ids: list[int], args, schema, clin, enc, out: Path, t0: float) -> None:
    label_data = Path(args.label_data)
    names = {"ecg": 0, "abp": 1, "ppg": 2}
    keep = tuple(names[k] for k in args.signals)
    s = T.build_task(args.task, load_cases(label_data, ids), schema, args.cases_csv,
                     args.labs_csv, arr_dir=args.arr_dir)
    n = len(s["t"])
    snap = s["wave_t"].shape[1]
    if snap not in (1, N_CHUNK):
        raise ValueError(f"{snap} windows per sample; expected 1 or {N_CHUNK} (case-level "
                         "tasks need the default --chunk-s)")
    start = {}
    for c in np.unique(s["caseid"]):
        a = clin.anestart.get(c, np.nan)
        start[c] = int(max(0.0, a)) if np.isfinite(a) else 0
    # window k of a sample ends where the minute row of wave_t[:, k] ends
    ends = (np.array([start[c] for c in s["caseid"]])[:, None]
            + 60 * (np.floor(s["wave_t"]).astype(int) + 1))  # (n, K) seconds
    jobs, offsets, pos = [], [], 0
    for c in np.unique(s["caseid"]):  # build_samples orders samples by caseid
        m = np.flatnonzero(s["caseid"] == c)
        assert m[0] == pos and m[-1] == pos + len(m) - 1
        jobs.append((int(c), ends[m].reshape(-1), args.vital_dir, keep))
        offsets.append(pos)
        pos += len(m)
    if args.subsets:
        subsets = list(SUBSETS) if args.subsets == ["all"] else args.subsets
        outputs = {f"{part}_{k}": SUBSETS[k] for k in subsets}
    else:
        outputs = {part: keep}
    writer = TokenWriter(enc, out, outputs, n, args.batch_size, snap)
    present = np.zeros((n, N_SIG), bool)
    with Pool(args.workers) as pool:
        results = pool.imap(case_windows, jobs, chunksize=2)
        for k, ((_, w, p), off) in enumerate(zip(results, offsets, strict=True), start=1):
            m = len(w) // snap
            present[off: off + m] = p.reshape(m, snap, N_SIG).any(1)
            writer.add(w, p, np.repeat(np.arange(off, off + m), snap))
            if k % 200 == 0:
                print(f"[{part}] {k}/{len(jobs)} cases ({time.time() - t0:.0f}s)", flush=True)
    writer.close()
    np.savez(out / f"{part}_meta.npz", caseid=s["caseid"], t=s["t"], present=present,
             n_chunk=N_CHUNK)
    print(f"[{part}] {n} samples; signals present ECG/ABP/PPG = "
          f"{present.mean(0).round(3).tolist()}; any = {present.any(1).mean():.3f}; "
          f"outputs {list(outputs)} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    global CHUNK_S, N_CHUNK
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label-data", required=True)
    ap.add_argument("--vital-dir", required=True)
    ap.add_argument("--cases-csv", required=True)
    ap.add_argument("--labs-csv", default=None, help="lab_data.csv (task aki)")
    ap.add_argument("--arr-dir", default=None,
                    help="VitalDB Arrhythmia Database dir (task arrhythmia)")
    ap.add_argument("--split", default=None,
                    help="split json (default: <label-data>/split.json), e.g. a CV fold")
    ap.add_argument("--task", default="ioh", choices=list(T.TASKS))
    ap.add_argument("--carmen-src", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--splits", nargs="+", default=list(P.PARTS))
    ap.add_argument("--subsets", nargs="+", default=None,
                    help="encode each signal subset separately: names from "
                         f"{list(SUBSETS)} or 'all'; writes <split>_<subset>_tokens.npy")
    ap.add_argument("--signals", nargs="+", default=["ecg", "abp", "ppg"],
                    choices=["ecg", "abp", "ppg"],
                    help="signals given to CARMEN (others are treated as absent), e.g. "
                         "'ecg ppg' to encode as if there were no arterial line")
    ap.add_argument("--chunk-s", type=int, default=CHUNK_S,
                    help="seconds per token (60 s window -> 60 / chunk-s tokens per signal); "
                         "single-window tasks only, e.g. 2 for beat-level tasks (arrhythmia)")
    args = ap.parse_args()
    if WIN_S % args.chunk_s or (args.chunk_s * TARGET_SR) % 25:
        raise ValueError("--chunk-s must divide 60 s into whole 0.25 s patches")
    CHUNK_S, N_CHUNK = args.chunk_s, WIN_S // args.chunk_s

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    label_data = Path(args.label_data)
    schema = TableSchema.load(label_data / "schema.json")
    split = json.loads(Path(args.split or label_data / "split.json").read_text())
    clin = pd.read_csv(args.cases_csv, usecols=["caseid", "anestart"]).set_index("caseid")
    enc = CarmenEncoder(args.carmen_src, args.weights, device)
    t0 = time.time()
    for part in args.splits:
        run_split(part, sorted(split[part])[: args.max_cases], args, schema, clin, enc, out, t0)


if __name__ == "__main__":
    main()
