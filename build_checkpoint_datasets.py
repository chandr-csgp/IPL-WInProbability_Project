# Rebuild checkpoint datasets (5/10/15/19 overs) from the data committed in
# this repo, with season attached for temporal validation.
#
# Mirrors the extraction logic in s8checkpoints.py (one row per match: the
# last ball recorded in the checkpoint over, second innings only), but reads
# data/inplay_snapshots.csv and data/matches_clean.csv from this repo instead
# of the local DSP project folder, and joins on season/date so a temporal
# train/test split is possible.

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

IN_SNAP = DATA / "inplay_snapshots.csv"
IN_MATCHES = DATA / "matches_clean.csv"

OUT_FILES = {
    5: DATA / "checkpoint_5.csv",
    10: DATA / "checkpoint_10.csv",
    15: DATA / "checkpoint_15.csv",
    19: DATA / "checkpoint_19.csv",
}

print("[Info] Script: build_checkpoint_datasets.py")

snap = pd.read_csv(IN_SNAP, low_memory=False)
matches = pd.read_csv(IN_MATCHES, low_memory=False, usecols=["match_id", "date", "season"])
matches["date"] = pd.to_datetime(matches["date"])
print(f"[Data] Loaded {len(snap)} snapshot rows, {len(matches)} match records")

# Keep only second innings (the chase)
snap = snap[snap["innings"] == 2].copy()
snap = snap.sort_values(["match_id", "over", "ball_in_over"]).reset_index(drop=True)


def extract_checkpoint(df: pd.DataFrame, checkpoint_over: int) -> pd.DataFrame:
    """One row per match: the last ball recorded in the checkpoint over."""
    df_cp = df[df["over"] == checkpoint_over].copy()
    if df_cp.empty:
        print(f"[Warn] No rows found for checkpoint over {checkpoint_over}")
        return df_cp

    df_cp = (
        df_cp.sort_values(["match_id", "ball_in_over"])
        .groupby("match_id", as_index=False)
        .tail(1)
        .sort_values("match_id")
        .reset_index(drop=True)
    )

    df_cp["checkpoint_over"] = checkpoint_over
    df_cp["rrr_x_wickets_in_hand"] = df_cp["required_run_rate"] * df_cp["wickets_in_hand"]
    df_cp["runs_to_go_x_wickets_in_hand"] = df_cp["runs_to_go"] * df_cp["wickets_in_hand"]
    df_cp["rrr_minus_crr"] = df_cp["required_run_rate"] - df_cp["current_run_rate"]

    return df_cp


for checkpoint, out_path in OUT_FILES.items():
    cp = extract_checkpoint(snap, checkpoint)
    cp = cp.merge(matches, on="match_id", how="left", validate="many_to_one")

    missing_season = cp["season"].isna().sum()
    if missing_season:
        print(f"[Warn] Checkpoint {checkpoint}: {missing_season} rows have no season match")

    cp.to_csv(out_path, index=False)
    print(
        f"[OK] Checkpoint {checkpoint}: {len(cp)} rows, "
        f"{cp['match_id'].nunique()} matches, seasons {sorted(cp['season'].dropna().unique())} "
        f"-> {out_path}"
    )
