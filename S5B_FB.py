
# STEP 5B: FEATURE ENGINEERING  
# Builds match-level binary indicators for regression models

from pathlib import Path
import pandas as pd
import numpy as np
import platform, sys, re

VERSION = "S5B.v1.5"
print(f"[Info] Script: {VERSION}")
print(f"[Env] Python {sys.version.split()[0]} | Pandas {pd.__version__} | Platform {platform.system()}")

#  dtype handling 
def get_safe_int_dtype():
    print("[Fix] Using float64")
    return "float64"

INTD = get_safe_int_dtype()

# Paths 
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
IN_MATCHES = DATA / "matches_clean.csv"
IN_BALLS   = DATA / "balls_clean.csv"
OUT_FEATS  = DATA / "matches_features.csv"

assert IN_MATCHES.exists(), f"Missing {IN_MATCHES}"
assert IN_BALLS.exists(), f"Missing {IN_BALLS}"

# Load data 
m = pd.read_csv(IN_MATCHES)
b = pd.read_csv(IN_BALLS, usecols=["match_id","innings","over","ball_in_over","batting_team"])

#  Batting-first inference 
first_ball = (
    b.sort_values(["match_id","innings","over","ball_in_over"])
     .groupby("match_id", as_index=False)
     .first()[["match_id","batting_team"]]
     .rename(columns={"batting_team":"batting_first_team"})
)
df = m.merge(first_ball, on="match_id", how="left")

def other_team(row):
    t1, t2, bf = row["team1"], row["team2"], row["batting_first_team"]
    if pd.isna(bf) or not isinstance(t1,str) or not isinstance(t2,str):
        return np.nan
    return t2 if bf == t1 else (t1 if bf == t2 else np.nan)

df["batting_second_team"] = df.apply(other_team, axis=1)

# Binary indicators 
df["toss_won_flag_team1"] = (df["toss_winner"] == df["team1"]).astype(INTD)
df["bat_first_flag_team1"] = (df["batting_first_team"] == df["team1"]).astype(INTD)
df["home_flag_team1"]      = (df["home_team"] == df["team1"]).astype(INTD)
df["neutral_venue_flag"]   = df["home_team"].isna().astype(INTD)
df["toss_bat_flag"]        = (df["toss_decision"].str.lower() == "bat").astype(INTD)

# Win flag with proper NaN handling for float64
win_flag = np.where(
    df["winner"].isna(), np.nan,
    np.where(df["winner"] == df["team1"], 1.0,
             np.where(df["winner"] == df["team2"], 0.0, np.nan))
)
df["win_flag_team1"] = win_flag.astype(INTD)

df["margin_normalized"] = df["by_runs"].fillna(df["by_wickets"])

#  Season normalization 
def season_to_int(s):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.match(r"^\s*(\d{4})\s*/\s*(\d{2})\s*$", s)
    if m: return float(int(m.group(1)[:2] + m.group(2)))
    m2 = re.match(r"^\s*(\d{4})\s*$", s)
    return float(int(m2.group(1))) if m2 else np.nan

df["season_num"] = df["season"].apply(season_to_int).astype(INTD)

# Toss vs batting-first consistency 
mask_known = df["toss_winner"].notna() & df["toss_decision"].notna()

def toss_consistent(row):
    if not mask_known.loc[row.name]: return np.nan
    tw, dec, bf, t1, t2 = row["toss_winner"], str(row["toss_decision"]).lower(), row["batting_first_team"], row["team1"], row["team2"]
    if dec == "bat": return 1.0 if bf == tw else 0.0
    if dec == "field":
        other = t1 if tw == t2 else t2
        return 1.0 if bf == other else 0.0
    return np.nan

df["toss_batfirst_consistent"] = df.apply(toss_consistent, axis=1).astype(INTD)

# Export clean features 
cols = [
    "match_id","date","season","season_num","team1","team2","home_team",
    "neutral_venue_flag","venue_canon","city_from_venue","winner","result",
    "by_runs","by_wickets","margin_normalized","toss_winner","toss_decision",
    "toss_bat_flag","toss_won_flag_team1","batting_first_team",
    "batting_second_team","bat_first_flag_team1","win_flag_team1","toss_batfirst_consistent"
]
df[cols].to_csv(OUT_FEATS, index=False)

# Sanity summary 
print("\n=== Step 5B sanity Checks ===")
print(f"Rows total: {len(df)}")
print(f"Missing batting_first_team: {df['batting_first_team'].isna().sum()}")
print(f"Toss vs batting-first consistent: {df['toss_batfirst_consistent'].eq(1.0).sum()} / {df['toss_batfirst_consistent'].notna().sum()}")
print(f"Output saved → {OUT_FEATS}")