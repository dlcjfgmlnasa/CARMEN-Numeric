"""Build the VitalDB numerical-row cache for UniTTab pre-training.

    # open data, downloaded from api.vitaldb.net (small test: --max-cases 200)
    python scripts/prepare_vitaldb.py --out data/vitaldb_numeric --max-cases 200

    # on the GPU node, from local files
    python scripts/prepare_vitaldb.py --out $ORCH_HOME/carmen_numeric/vitaldb \
        --vital-dir $ORCH_HOME/vitaldb/raw \
        --cases-csv $ORCH_HOME/aki/csv/clinical_data.csv --labs-csv <lab_data.csv>

Writes ``<out>/cases/<caseid>.npz`` (rows of one case), ``schema.json``, ``index.csv``,
``split.json`` (patient-level 80/10/10 by ``subjectid``) and ``normalizer.json``
(input ranges + quantile bins, fitted on the training split only).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carmen_numeric.data.dataset import (  # noqa: E402
    collect_field_values,
    load_cases,
    patient_split,
)
from carmen_numeric.data.vitaldb import (  # noqa: E402
    build_case,
    load_tracks,
    save_case,
    vitaldb_cat_sizes,
    vitaldb_schema,
)
from carmen_numeric.normalize import Normalizer  # noqa: E402

API = "https://api.vitaldb.net"


def read_table(path: str | None, endpoint: str) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    with urllib.request.urlopen(f"{API}/{endpoint}", timeout=300) as r:
        raw = r.read()
    try:
        import gzip

        raw = gzip.decompress(raw)
    except OSError:
        pass
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")


def _work(args: tuple) -> tuple[int, str]:
    caseid, case, labs, vital_dir, out_dir = args
    schema = vitaldb_schema()
    try:
        src: str | int = caseid
        if vital_dir:
            p = Path(vital_dir) / f"{caseid}.vital"
            if not p.exists():
                return caseid, "no_vital_file"
            src = str(p)
        data = None
        for attempt in range(3):
            try:
                data = load_tracks(src)
                break
            except Exception:  # network hiccups when downloading
                if attempt == 2 or vital_dir:
                    raise
                time.sleep(2 * (attempt + 1))
        if data is None or data.ndim != 2 or data.shape[0] == 0:
            return caseid, "no_tracks"
        rows = build_case(pd.Series(case), labs, data, schema)
        if rows is None:
            return caseid, "too_short"
        save_case(rows, Path(out_dir) / "cases")
        return caseid, "ok"
    except Exception as exc:  # keep going; report at the end
        return caseid, f"error:{type(exc).__name__}:{exc}"[:200]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases-csv", default=None, help="clinical_data.csv (default: VitalDB API)")
    ap.add_argument("--labs-csv", default=None, help="lab_data.csv (default: VitalDB API)")
    ap.add_argument("--vital-dir", default=None, help="local <caseid>.vital files (default: API)")
    ap.add_argument("--max-cases", type=int, default=None, help="random subset for quick tests")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--num-bins", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    schema = vitaldb_schema()
    schema.save(out / "schema.json")

    cases = read_table(args.cases_csv, "cases")
    labs = read_table(args.labs_csv, "labs")
    cases = cases.sort_values("caseid")
    if args.max_cases:
        cases = cases.sample(n=min(args.max_cases, len(cases)), random_state=args.seed)
    labs_by_case = {cid: g for cid, g in labs.groupby("caseid")}
    print(f"cases: {len(cases)}  lab rows: {len(labs)}", flush=True)

    jobs = []
    for case in cases.to_dict("records"):
        cid = int(case["caseid"])
        if args.skip_existing and (out / "cases" / f"{cid}.npz").exists():
            continue
        jobs.append((cid, case, labs_by_case.get(cid), args.vital_dir, str(out)))

    status: dict[str, int] = {}
    errors = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_work, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            cid, st = f.result()
            key = st.split(":")[0]
            status[key] = status.get(key, 0) + 1
            if key == "error":
                errors.append((cid, st))
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  {status}  {time.time() - t0:.0f}s", flush=True)
    for cid, st in errors[:20]:
        print(f"  case {cid}: {st}")

    # ── index + patient-level split + normalizer (train only) ──
    all_cases = load_cases(out)
    index = pd.DataFrame(
        [
            {
                "caseid": cid,
                "subjectid": c["subjectid"],
                "n_rows": len(c["row_type"]),
                "n_monitor": int((c["row_type"] == schema.row_type_id["monitor"]).sum()),
                "n_lab": int((c["row_type"] == schema.row_type_id["lab"]).sum()),
            }
            for cid, c in all_cases.items()
        ]
    )
    index.to_csv(out / "index.csv", index=False)
    split = patient_split(all_cases, seed=args.seed)
    (out / "split.json").write_text(json.dumps(split))
    train = {cid: all_cases[cid] for cid in split["train"]}
    values = collect_field_values(train, schema, seed=args.seed)
    norm = Normalizer.fit(schema, values, vitaldb_cat_sizes(), num_bins=args.num_bins)
    norm.save(out / "normalizer.json")
    print(
        f"done: {len(all_cases)} cases, {int(index.n_rows.sum())} rows "
        f"(monitor {int(index.n_monitor.sum())}, lab {int(index.n_lab.sum())}); "
        f"split train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}"
    )


if __name__ == "__main__":
    main()
