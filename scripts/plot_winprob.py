#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Ensure local src/ is on sys.path for any future imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

DEF_MODEL = "logreg_inplay.joblib"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot win probability curve for a given match id")
    parser.add_argument("--match-id", required=True, help="Match id (file stem, e.g., 336034)")
    parser.add_argument("--data-dir", type=Path, default=None, help="Data directory (default: DSP/data)")
    parser.add_argument("--models-dir", type=Path, default=None, help="Models directory (default: DSP/models)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or (repo_root / "data")
    models_dir = args.models_dir or (repo_root / "models")

    snaps = pd.read_csv(data_dir / "inplay_snapshots.csv")
    m = snaps[snaps["match_id"] == args.match_id].copy()
    if m.empty:
        raise SystemExit(f"No in-play snapshots found for match {args.match_id}")

    # Load model
    model_path = models_dir / DEF_MODEL
    clf = joblib.load(model_path)

    # Predict
    feat_cols = [
        "runs_to_go",
        "balls_remaining",
        "wickets_in_hand",
        "current_run_rate",
        "required_run_rate",
        "target",
    ]
    m["win_prob"] = clf.predict_proba(m[feat_cols])[:, 1]

    # Save CSV
    out_csv = data_dir / f"winprob_{args.match_id}.csv"
    m.to_csv(out_csv, index=False)

    # Plot curve vs delivery index
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m.index, m["win_prob"], label="Chasing Win Probability")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Delivery index (legal)")
    ax.set_ylabel("Win probability")
    ax.set_title(f"Win Probability – Match {args.match_id}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_png = data_dir / f"winprob_{args.match_id}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
