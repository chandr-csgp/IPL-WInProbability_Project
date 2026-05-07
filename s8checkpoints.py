# STEP 8B: CHECKPOINT DATASET BUILDER
# Creates one row per match at selected over checkpoints

from pathlib import Path
import pandas as pd

# Paths
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
IN_SNAP = DATA / "inplay_snapshots_v2.csv"

OUT_5  = DATA / "checkpoint_5.csv"
OUT_10 = DATA / "checkpoint_10.csv"
OUT_15 = DATA / "checkpoint_15.csv"
OUT_19 = DATA / "checkpoint_19.csv"

print("[Info] Script: S8B - Checkpoint dataset builder")

# Load snapshot data
snap = pd.read_csv(IN_SNAP, low_memory=False)
print(f"[Data] Loaded {len(snap)} snapshot rows from {IN_SNAP}")

# Required columns
required_cols = [
    "match_id", "innings", "over", "ball_in_over",
    "batting_team", "bowling_team",
    "current_score", "wickets_lost", "wickets_in_hand",
    "runs_to_go", "balls_remaining", "overs_remaining",
    "current_run_rate", "required_run_rate",
    "target_runs", "win_flag"
]
missing = [c for c in required_cols if c not in snap.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only second innings
snap = snap[snap["innings"] == 2].copy()

# Sort properly
snap = snap.sort_values(["match_id", "over", "ball_in_over"]).reset_index(drop=True)

def extract_checkpoint(df: pd.DataFrame, checkpoint_over: int) -> pd.DataFrame:
    """
    Keep one row per match at the selected checkpoint over.
    We take the latest ball available in that over for each match.
    """
    df_cp = df[df["over"] == checkpoint_over].copy()

    if df_cp.empty:
        print(f"[Warn] No rows found for checkpoint over {checkpoint_over}")
        return df_cp

    # Keep latest ball in that over for each match
    df_cp = (
        df_cp.sort_values(["match_id", "ball_in_over"])
             .groupby("match_id", as_index=False)
             .tail(1)
             .sort_values("match_id")
             .reset_index(drop=True)
    )

    # Label checkpoint
    df_cp["checkpoint_over"] = checkpoint_over

    # Add interaction / derived features
    df_cp["rrr_x_wickets_in_hand"] = df_cp["required_run_rate"] * df_cp["wickets_in_hand"]
    df_cp["runs_to_go_x_wickets_in_hand"] = df_cp["runs_to_go"] * df_cp["wickets_in_hand"]
    df_cp["rrr_minus_crr"] = df_cp["required_run_rate"] - df_cp["current_run_rate"]

    return df_cp

# Build checkpoint datasets
cp5  = extract_checkpoint(snap, 5)
cp10 = extract_checkpoint(snap, 10)
cp15 = extract_checkpoint(snap, 15)
cp19 = extract_checkpoint(snap, 19)

# Save
cp5.to_csv(OUT_5, index=False)
cp10.to_csv(OUT_10, index=False)
cp15.to_csv(OUT_15, index=False)
cp19.to_csv(OUT_19, index=False)

print(f"[OK] Saved {len(cp5)} rows  -> {OUT_5}")
print(f"[OK] Saved {len(cp10)} rows -> {OUT_10}")
print(f"[OK] Saved {len(cp15)} rows -> {OUT_15}")
print(f"[OK] Saved {len(cp19)} rows -> {OUT_19}")

def summary(df: pd.DataFrame, name: str):
    print(f"\n=== {name} Summary ===")
    if df.empty:
        print("No rows found.")
        return
    print(f"Rows: {len(df)}")
    print(f"Unique matches: {df['match_id'].nunique()}")
    print(f"Chasing wins: {int(df['win_flag'].sum())} / {len(df)}")
    print("Columns:")
    print(list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))

summary(cp5, "Checkpoint 5 overs")
summary(cp10, "Checkpoint 10 overs")
summary(cp15, "Checkpoint 15 overs")
summary(cp19, "Checkpoint 19 overs")