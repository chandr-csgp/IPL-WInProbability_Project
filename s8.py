
# STEP 8: IN-PLAY SNAPSHOT BUILDER 


from pathlib import Path
import pandas as pd
import numpy as np

# Paths 
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
IN_MATCHES = DATA / "matches_clean.csv"
IN_BALLS   = DATA / "balls_clean.csv"
OUT_SNAP   = DATA / "inplay_snapshots.csv"

print("[Info] Script: S8.v1.4 - In-play snapshot builder (legal-ball fix)")

# Load 
matches = pd.read_csv(IN_MATCHES, low_memory=False)
balls   = pd.read_csv(IN_BALLS,   low_memory=False)
print(f"[Data] Loaded {len(matches)} matches and {len(balls)} ball records")

# Keep only matches with a winner to label win_flag cleanly
matches = matches[matches["winner"].notna()].reset_index(drop=True)

# Balls must have basic keys (note: we use batting_team not 'team')
balls = balls.dropna(subset=["match_id", "innings", "batting_team"]).reset_index(drop=True)

# Scheduled overs per match (from Step 5 matches file) 
# For IPL T20, this is usually 20; if reduced-overs, info['overs'] should reflect it.
sched_overs_map = matches.set_index("match_id")["overs"].to_dict()
balls["scheduled_overs"] = balls["match_id"].map(sched_overs_map).fillna(20)

#LEGAL balls bowled & balls remaining 
# Cap ball_in_over at 6 to count only legal deliveries
ball_idx_capped = np.minimum(balls["ball_in_over"].astype(float), 6.0)
legal_balls_bowled = (balls["over"].astype(float) - 1.0) * 6.0 + ball_idx_capped

balls["balls_bowled_legal"] = legal_balls_bowled
balls["innings_total_balls"] = balls["scheduled_overs"].astype(float) * 6.0
balls["balls_remaining"] = (balls["innings_total_balls"] - balls["balls_bowled_legal"]).clip(lower=0)

# Cumulative stats within an innings 
balls["cum_runs"]    = balls.groupby(["match_id", "innings"])["runs_total"].cumsum()
balls["cum_wickets"] = balls.groupby(["match_id", "innings"])["wicket"].cumsum()

# Target = first-innings total + 1 
first_innings_totals = (
    balls[balls["innings"] == 1]
    .groupby("match_id")["runs_total"]
    .max()
    .rename("first_innings_total")
)
target_map = (first_innings_totals + 1).to_dict()
balls["target_runs"] = balls["match_id"].map(target_map)

# Keep only second innings (the chase) 
balls["is_second_innings"] = (balls["innings"] == 2).astype(int)
chase = balls[balls["is_second_innings"] == 1].copy()

# Feature engineering 
chase["runs_to_go"]      = (chase["target_runs"] - chase["cum_runs"]).clip(lower=0)
chase["wickets_in_hand"] = 10 - chase["cum_wickets"]

# Run-rate metrics (guard against div-by-zero / NaN)
chase["current_run_rate"] = (
    chase["cum_runs"] / (chase["balls_bowled_legal"] / 6.0).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0.0)

chase["required_run_rate"] = (
    chase["runs_to_go"] / (chase["balls_remaining"] / 6.0).replace(0, np.nan)
).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# Label: win_flag = 1 if chasing team (team2) eventually won 
winner_map = matches.set_index("match_id")["winner"].to_dict()
team2_map  = matches.set_index("match_id")["team2"].to_dict()

def win_flag_row(row):
    winner = winner_map.get(row["match_id"])
    team2  = team2_map.get(row["match_id"])
    if winner is None or team2 is None:
        return np.nan
    return 1 if winner == team2 else 0

chase["win_flag"] = chase.apply(win_flag_row, axis=1)

#  Select output columns 
cols = [
    "match_id", "innings", "over", "ball_in_over",
    "runs_to_go", "balls_remaining", "wickets_in_hand",
    "current_run_rate", "required_run_rate",
    "target_runs", "win_flag",
    "batting_team", "bowling_team", "venue_canon", "home_team"
]
snap = chase[cols].dropna(subset=[
    "runs_to_go", "balls_remaining", "wickets_in_hand", "win_flag"
]).copy()

# Sanity checks 
assert (snap["runs_to_go"] >= 0).all(), "Negative runs_to_go found."
assert (snap["balls_remaining"] >= 0).all(), "Negative balls_remaining found."
assert (snap["wickets_in_hand"].between(0, 10)).all(), "wickets_in_hand out of [0,10]."

# Save 
snap.to_csv(OUT_SNAP, index=False)
print(f"[OK] Saved {len(snap)} snapshots → {OUT_SNAP}")

#  Summary 
n_matches = snap["match_id"].nunique()
wins_per_match = snap.groupby("match_id")["win_flag"].max().sum()
print("\n=== Step 8 Sanity Summary ===")
print(f"Total snapshots: {len(snap)}")
print(f"Unique matches with snapshots: {n_matches}")
print(f"Average snapshots per match: {len(snap) / max(n_matches,1):.2f}")
print(f"Chasing team wins (matches): {int(wins_per_match)} / {n_matches}")
print(f"Columns: {list(snap.columns)}")
