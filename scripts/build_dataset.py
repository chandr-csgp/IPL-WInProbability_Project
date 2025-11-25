#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure local src/ is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from ipl_winprob.parse_cricsheet import parse_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build match-level and in-play datasets from Cricsheet IPL JSON")
    parser.add_argument("--ipl-dir", type=Path, default=None, help="Directory containing *.json from Cricsheet (default: DSP/ipl_json)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for CSVs (default: DSP/data)")
    args = parser.parse_args()

    # Resolve defaults relative to this script location
    repo_root = Path(__file__).resolve().parents[1]
    ipl_dir = args.ipl_dir or (repo_root / "ipl_json")
    out_dir = args.out_dir or (repo_root / "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    match_df, snaps_df = parse_directory(ipl_dir)

    match_csv = out_dir / "match_features.csv"
    snaps_csv = out_dir / "inplay_snapshots.csv"

    match_df.to_csv(match_csv, index=False)
    snaps_df.to_csv(snaps_csv, index=False)

    print(f"Wrote {len(match_df)} match rows -> {match_csv}")
    print(f"Wrote {len(snaps_df)} in-play rows -> {snaps_csv}")


if __name__ == "__main__":
    main()
