
# STEP 5: BUILD MATCH & BALL-BY-BALL TABLES
# Uses canonical teams & venues; infers home_team

from pathlib import Path
import json
import os
import pandas as pd

VERSION = "S5.v1.0"

# Paths
ROOT = Path.home() / "Documents" / "DSP"
RAW  = ROOT / "ipl_json"          # raw Cricsheet JSONs (1169 files)
DATA = ROOT / "data"

MAP_TEAMS_JSON = DATA / "_team_alias_mapping.json"     # from Step 3
MAP_VENUE_JSON = DATA / "_venue_fix_mapping.json"      # from Step 4

OUT_MATCHES = DATA / "matches_clean.csv"
OUT_BALLS   = DATA / "balls_clean.csv"

print(f"[Info] Script: {VERSION}")
assert RAW.exists() and RAW.is_dir(), f"Missing raw folder: {RAW}"
assert MAP_TEAMS_JSON.exists(), f"Missing mapping: {MAP_TEAMS_JSON} (run Step 3)"
assert MAP_VENUE_JSON.exists(), f"Missing mapping: {MAP_VENUE_JSON} (run Step 4)"

# Load mappings 
team_map  = json.loads(MAP_TEAMS_JSON.read_text(encoding="utf-8"))
venue_map = json.loads(MAP_VENUE_JSON.read_text(encoding="utf-8"))

def norm_team(name: str):
    if not isinstance(name, str) or not name.strip():
        return name
    return team_map.get(name, name)

def fix_venue(v: str):
    if not isinstance(v, str) or not v.strip():
        return v
    return venue_map.get(v, v)

def extract_city_from_canon(venue_canon: str):
    if not isinstance(venue_canon, str) or "," not in venue_canon:
        return None
    parts = [p.strip() for p in venue_canon.split(",")]
    return parts[-1] if parts else None

# Conservative city→team map 
CITY_TEAM_MAP = {
    "Chennai": ["Chennai Super Kings"],
    "Mumbai": ["Mumbai Indians"],
    "Navi Mumbai": ["Mumbai Indians"],
    "Bengaluru": ["Royal Challengers Bangalore"],
    "Bangalore": ["Royal Challengers Bangalore"],
    "Kolkata": ["Kolkata Knight Riders"],
    "Delhi": ["Delhi Capitals", "Delhi Daredevils"],
    "Jaipur": ["Rajasthan Royals"],
    "Hyderabad": ["Sunrisers Hyderabad", "Deccan Chargers"],
    "Mohali": ["Punjab Kings", "Kings XI Punjab"],
    "Dharamsala": ["Punjab Kings", "Kings XI Punjab"],
    "Indore": ["Punjab Kings", "Kings XI Punjab"],
    "Ahmedabad": ["Gujarat Titans", "Gujarat Lions"],
    "Lucknow": ["Lucknow Super Giants"],
    "Pune": ["Rising Pune Supergiant", "Pune Warriors"],
    "Raipur": ["Delhi Capitals", "Delhi Daredevils"],
}

def infer_home(city: str, t1: str, t2: str):
    if not isinstance(city, str):
        return None
    for candidate in CITY_TEAM_MAP.get(city, []):
        # use canonical teams only
        cand = norm_team(candidate)
        if cand in (t1, t2):
            return cand
    return None

#Templates 
match_cols = [
    "match_id","date","season","match_type","competition",
    "team1","team2","winner","result","by_runs","by_wickets",
    "toss_winner","toss_decision","method",
    "venue_raw","venue_canon","city_raw","city_from_venue",
    "home_team","player_of_match","umpires","referee",
    "balls_per_over","overs"
]
ball_cols = [
    "match_id","date","season","innings","over","ball_in_over",
    "batting_team","bowling_team",
    "batter","non_striker","bowler",
    "runs_batter","runs_extras","runs_total",
    "wicket","wicket_type","wicket_player_out",
    "is_powerplay","powerplay_type",
    "venue_canon","home_team","toss_decision"
]

matches_out = []
balls_out   = []

# Iterate JSONs 
files = [f for f in os.listdir(RAW) if f.endswith(".json")]
files.sort()
print(f"[Scan] Found {len(files)} json files in {RAW}")

for idx, fname in enumerate(files, 1):
    fpath = RAW / fname
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as e:
        print(f"[ERR] Could not read {fname}: {e}")
        continue

    info = data.get("info", {}) or {}

    # Teams (canonical)
    raw_teams = info.get("teams", []) or []
    team1 = norm_team(raw_teams[0]) if len(raw_teams) > 0 else None
    team2 = norm_team(raw_teams[1]) if len(raw_teams) > 1 else None

    # Outcome
    outcome = info.get("outcome", {}) or {}
    by      = outcome.get("by", {}) or {}

    winner  = norm_team(outcome.get("winner"))
    result  = outcome.get("result")          # tie / no result / None
    method  = outcome.get("method")          # D/L, super over, etc.

    by_runs     = by.get("runs")
    by_wickets  = by.get("wickets")

    # Toss
    toss_winner   = norm_team(((info.get("toss") or {}).get("winner")))
    toss_decision = ((info.get("toss") or {}).get("decision"))

    # Venue & city
    venue_raw   = info.get("venue")
    venue_canon = fix_venue(venue_raw)
    city_raw    = info.get("city")
    city_from_v = extract_city_from_canon(venue_canon) or city_raw

    # Home team (conservative)
    home_team = infer_home(city_from_v, team1, team2)

    # Misc
    match_id        = fname.split(".")[0]
    date            = (info.get("dates") or [None])[0]
    season          = info.get("season")
    match_type      = info.get("match_type")
    competition     = info.get("competition")
    player_of_match = (info.get("player_of_match") or [None])[0] if isinstance(info.get("player_of_match"), list) else info.get("player_of_match")
    officials       = info.get("officials") or {}
    umpires         = ", ".join((officials.get("umpires") or []))
    referee         = ", ".join((officials.get("match_referees") or []))
    balls_per_over  = info.get("balls_per_over")
    overs           = info.get("overs")

    matches_out.append({
        "match_id": match_id, "date": date, "season": season,
        "match_type": match_type, "competition": competition,
        "team1": team1, "team2": team2, "winner": winner, "result": result,
        "by_runs": by_runs, "by_wickets": by_wickets,
        "toss_winner": toss_winner, "toss_decision": toss_decision, "method": method,
        "venue_raw": venue_raw, "venue_canon": venue_canon, "city_raw": city_raw, "city_from_venue": city_from_v,
        "home_team": home_team, "player_of_match": player_of_match,
        "umpires": umpires, "referee": referee, "balls_per_over": balls_per_over, "overs": overs
    })

    # Balls
    innings_list = data.get("innings", []) or []
    for inn_idx, innings in enumerate(innings_list, start=1):
        batting = norm_team(innings.get("team"))
        # bowling is the other team (safe for IPL 2-team matches)
        bowling = team2 if batting == team1 else team1

        powerplays = innings.get("powerplays", []) or []
        def in_pp(over_number: int):
            for pp in powerplays:
                f, t = pp.get("from"), pp.get("to")
                if isinstance(f, int) and isinstance(t, int):
                    if f <= over_number <= t:
                        return True, pp.get("type")
            return False, None

        for over_i, over in enumerate(innings.get("overs", []) or [], start=1):
            for ball_j, delivery in enumerate(over.get("deliveries", []) or [], start=1):
                runs = delivery.get("runs", {}) or {}
                wcks = delivery.get("wickets", []) or []

                is_pp, pp_type = in_pp(over_i)

                wicket_flag = bool(wcks)
                w_kind = wcks[0].get("kind") if wicket_flag else None
                w_out  = wcks[0].get("player_out") if wicket_flag else None

                balls_out.append({
                    "match_id": match_id, "date": date, "season": season,
                    "innings": inn_idx, "over": over_i, "ball_in_over": ball_j,
                    "batting_team": batting, "bowling_team": bowling,
                    "batter": delivery.get("batter"),
                    "non_striker": delivery.get("non_striker"),
                    "bowler": delivery.get("bowler"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_extras": runs.get("extras", 0),
                    "runs_total":  runs.get("total", 0),
                    "wicket": wicket_flag, "wicket_type": w_kind, "wicket_player_out": w_out,
                    "is_powerplay": is_pp, "powerplay_type": pp_type,
                    "venue_canon": venue_canon, "home_team": home_team,
                    "toss_decision": toss_decision
                })

# To DataFrames 
matches_df = pd.DataFrame(matches_out, columns=match_cols)
balls_df   = pd.DataFrame(balls_out,   columns=ball_cols)

# Sort for readability
if not matches_df.empty:
    matches_df = matches_df.sort_values(["date","match_id"]).reset_index(drop=True)
if not balls_df.empty:
    balls_df = balls_df.sort_values(["date","match_id","innings","over","ball_in_over"]).reset_index(drop=True)

# Save 
DATA.mkdir(parents=True, exist_ok=True)
matches_df.to_csv(OUT_MATCHES, index=False)
balls_df.to_csv(OUT_BALLS, index=False)

# Sanity checks 
n_files = len(files)
n_matches = matches_df["match_id"].nunique() if not matches_df.empty else 0
print("\n=== Sanity checks ===")
print(f"Files scanned: {n_files}")
print(f"Unique match_id in matches_clean: {n_matches}")
if n_matches != n_files:
    print("[WARN] Unique match_id != files scanned (some files may be skipped or duplicated).")

null_teams = int(matches_df["team1"].isna().sum() + matches_df["team2"].isna().sum()) if not matches_df.empty else 0
print(f"Null team names in matches_clean: {null_teams}")

pp_rows = int(balls_df["is_powerplay"].sum()) if not balls_df.empty else 0
print(f"Ball rows total: {len(balls_df)}   Powerplay-tagged deliveries: {pp_rows}")

# Winner must be either team1/team2 or NaN (for ties/no result)
if not matches_df.empty:
    bad_winner = matches_df[
        matches_df["winner"].notna() &
        ~matches_df["winner"].isin(matches_df["team1"]) &
        ~matches_df["winner"].isin(matches_df["team2"])
    ]
    print(f"Winner outside team1/team2 rows: {len(bad_winner)}")

print("\n[OK] Wrote:")
print(" ", OUT_MATCHES)
print(" ", OUT_BALLS)
