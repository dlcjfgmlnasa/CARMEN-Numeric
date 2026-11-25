"""Derive a field-subset cache from a prepared one (e.g. signal-derived values only).

    python scripts/derive_subset.py --src data/vitaldb --dst data/vitaldb_signal --profile signal
    python scripts/derive_subset.py --src ... --dst ... --fields hr spo2 art_mbp

Bin edges and scaling are copied from the source, so reconstruction MAE stays comparable
with models trained on all fields. See ``carmen_numeric.data.subset``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carmen_numeric.data.subset import FIELD_PROFILES, derive_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--profile", choices=sorted(FIELD_PROFILES))
    g.add_argument("--fields", nargs="+")
    args = ap.parse_args()
    keep = FIELD_PROFILES[args.profile] if args.profile else args.fields
    print(json.dumps(derive_subset(args.src, args.dst, keep), indent=2), flush=True)


if __name__ == "__main__":
    main()
