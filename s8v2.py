# STEP 8: IN-PLAY SNAPSHOT BUILDER

from pathlib import Path
import pandas as pd
import numpy as np

# Paths
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
IN_MATCHES = DATA / "matches_clean.csv"
IN_BALLS   = DATA / "balls_clean.csv"
OUT_SNAP   = DATA / "inplay_snapshots_v2.csv"

print("[Info] Script: S8.v2.0 - In-play snapshot builder (corrected target + features)")

# Load data
matches = pd.read_csv(IN_MATCHES, low_memory=False)
balls   = pd.read_csv(IN_BALLS, low_memory=False)

print(f"[Data] Loaded {len(matches)} matches and {len(balls)} ball records")

# Keep only matches with a winner
matches = matches[matches["winner"].notna()].reset_index(drop=True)

# Keep only balls with required fields
balls = balls.dropna(subset=["match_id", "innings", "batting_team"]).reset_index(drop=True)

# Scheduled overs per match
sched_overs_map = matches.set_index("match_id")["overs"].to_dict()
balls["scheduled_overs"] = balls["match_id"].map(sched_overs_map).fillna(20)

# Legal balls bowled and balls remaining
# ball_in_over may go beyond 6 when wides/no-balls are present, so cap at 6 for legal-ball count
ball_idx_capped = np.minimum(balls["ball_in_over"].astype(float), 6.0)
legal_balls_bowled = (balls["over"].astype(float) - 1.0) * 6.0 + ball_idx_capped

balls["balls_bowled_legal"] = legal_balls_bowled
balls["innings_total_balls"] = balls["scheduled_overs"].astype(float) * 6.0
balls["balls_remaining"] = (balls["innings_total_balls"] - balls["balls_bowled_legal"]).clip(lower=0)

# Cumulative stats within each innings
balls["cum_runs"] = balls.groupby(["match_id", "innings"])["runs_total"].cumsum()
balls["cum_wickets"] = balls.groupby(["match_id", "innings"])["wicket"].cumsum()

# First innings total = SUM of runs_total, not max()
first_innings_totals = (
    balls[balls["innings"] == 1]
    .groupby("match_id")["runs_total"]
    .sum()
    .rename("first_innings_total")
)

# Target for chase
target_map = (first_innings_totals + 1).to_dict()
balls["target_runs"] = balls["match_id"].map(target_map)

# Keep only second innings
balls["is_second_innings"] = (balls["innings"] == 2).astype(int)
chase = balls[balls["is_second_innings"] == 1].copy()

# Feature engineering
chase["current_score"] = chase["cum_runs"]
chase["wickets_lost"] = chase["cum_wickets"]
chase["runs_to_go"] = (chase["target_runs"] - chase["cum_runs"]).clip(lower=0)
chase["wickets_in_hand"] = 10 - chase["cum_wickets"]
chase["overs_remaining"] = chase["balls_remaining"] / 6.0

# Current run rate
chase["current_run_rate"] = (
    chase["cum_runs"] / (chase["balls_bowled_legal"] / 6.0).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Required run rate
chase["required_run_rate"] = (
    chase["runs_to_go"] / (chase["balls_remaining"] / 6.0).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Correct win label:
# win_flag = 1 if the second-innings batting team eventually won
winner_map = matches.set_index("match_id")["winner"].to_dict()
chase["win_flag"] = (chase["match_id"].map(winner_map) == chase["batting_team"]).astype(int)

# Final columns
cols = [
    "match_id", "innings", "over", "ball_in_over",
    "batting_team", "bowling_team",
    "current_score", "wickets_lost", "wickets_in_hand",
    "runs_to_go", "balls_remaining", "overs_remaining",
    "current_run_rate", "required_run_rate",
    "target_runs", "win_flag",
    "venue_canon", "home_team"
]

snap = chase[cols].dropna(subset=[
    "current_score", "wickets_lost", "runs_to_go",
    "balls_remaining", "overs_remaining", "win_flag"
]).copy()

# Sanity checks
assert (snap["current_score"] >= 0).all(), "Negative current_score found."
assert (snap["wickets_lost"].between(0, 10)).all(), "wickets_lost out of [0,10]."
assert (snap["wickets_in_hand"].between(0, 10)).all(), "wickets_in_hand out of [0,10]."
assert (snap["runs_to_go"] >= 0).all(), "Negative runs_to_go found."
assert (snap["balls_remaining"] >= 0).all(), "Negative balls_remaining found."
assert (snap["overs_remaining"] >= 0).all(), "Negative overs_remaining found."

# Save cleaned snapshot dataset
snap.to_csv(OUT_SNAP, index=False)
print(f"[OK] Saved {len(snap)} snapshots -> {OUT_SNAP}")

# Summary
n_matches = snap["match_id"].nunique()
wins_per_match = snap.groupby("match_id")["win_flag"].max().sum()

print("\n=== Step 8 Sanity Summary ===")
print(f"Total snapshots: {len(snap)}")
print(f"Unique matches with snapshots: {n_matches}")
print(f"Average snapshots per match: {len(snap) / max(n_matches, 1):.2f}")
print(f"Chasing team wins (matches): {int(wins_per_match)} / {n_matches}")
print(f"Columns: {list(snap.columns)}")

# Show first 10 rows
print("\n=== First 10 rows of snapshot dataset ===")
print(snap.head(10).to_string(index=False))