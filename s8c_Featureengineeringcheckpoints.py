from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"

FILES = {
    5: DATA / "checkpoint_5.csv",
    10: DATA / "checkpoint_10.csv",
    15: DATA / "checkpoint_15.csv",
    19: DATA / "checkpoint_19.csv",
}

print("[Info] Script: S8C - Checkpoint feature engineering")

for checkpoint, file_path in FILES.items():
    df = pd.read_csv(file_path, low_memory=False)

    # Focused engineered features
    df["rrr_minus_crr"] = df["required_run_rate"] - df["current_run_rate"]
    df["rrr_x_wickets_in_hand"] = df["required_run_rate"] * df["wickets_in_hand"]
    df["runs_to_go_x_wickets_in_hand"] = df["runs_to_go"] * df["wickets_in_hand"]

    df["runs_per_ball_needed"] = np.where(
        df["balls_remaining"] > 0,
        df["runs_to_go"] / df["balls_remaining"],
        0.0
    )

    df["target_completed_pct"] = np.where(
        df["target_runs"] > 0,
        df["current_score"] / df["target_runs"],
        0.0
    )

    out_path = DATA / f"checkpoint_{checkpoint}_fe.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[OK] Saved {out_path}")
    print(f"Rows: {len(df)}")
    print("New columns added:")
    print([
        "rrr_minus_crr",
        "rrr_x_wickets_in_hand",
        "runs_to_go_x_wickets_in_hand",
        "runs_per_ball_needed",
        "target_completed_pct",
    ])

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))