
# STEP 4 (B): APPLY VENUE FIX + INFER HOME TEAM 

from pathlib import Path
import pandas as pd
import json

VERSION = "S4.apply.v1.1"

DATA = Path.home() / "Documents" / "DSP" / "data"
IN_OVERVIEW = DATA / "_teams_alias_applied.csv"      # Step 3 output
IN_VMAP_JSON = DATA / "_venue_fix_mapping.json"

OUT_APPLIED  = DATA / "_venues_applied_and_home_inferred.csv"
OUT_UNMAPPED = DATA / "_venue_unmapped_names.csv"
OUT_COUNTS   = DATA / "_venue_canonical_counts.csv"
OUT_HOME_SUM = DATA / "_home_team_summary.csv"

print(f"[Info] Script: {VERSION}")
assert IN_OVERVIEW.exists(), f"Missing: {IN_OVERVIEW} (run Step 3 apply)"
assert IN_VMAP_JSON.exists(), f"Missing: {IN_VMAP_JSON} (run S4 detect/export)"

df = pd.read_csv(IN_OVERVIEW)
venue_map = json.loads(IN_VMAP_JSON.read_text(encoding="utf-8"))

#  Apply venue mapping
raw_venues = df["venue"].fillna("").astype(str)
unmapped = sorted({v for v in raw_venues.unique() if v not in venue_map})
if unmapped:
    pd.DataFrame({"unmapped_venue": unmapped}).to_csv(OUT_UNMAPPED, index=False)
    print(f"[WARN] {len(unmapped)} venues not in mapping → {OUT_UNMAPPED} (they'll remain as-is)")
else:
    print("[OK] All raw venues exist in mapping (good).")

def fix_venue(v):
    return venue_map.get(v, v) if isinstance(v, str) and v.strip() else v

df["venue_canon"] = raw_venues.apply(fix_venue)

# City extraction from canonical venue 
def extract_city(venue_canon: str):
    if not isinstance(venue_canon, str) or "," not in venue_canon:
        return None
    parts = [p.strip() for p in venue_canon.split(",")]
    return parts[-1] if parts else None

df["city_from_venue"] = df["venue_canon"].apply(extract_city).fillna(df.get("city"))

# Infer home team (city --> likely franchise but only if that team is in the match)
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

def infer_home(row):
    city = row["city_from_venue"]
    t1, t2 = row["team1"], row["team2"]
    if not isinstance(city, str) or not isinstance(t1, str) or not isinstance(t2, str):
        return None
    for candidate in CITY_TEAM_MAP.get(city, []):
        if candidate in (t1, t2):
            return candidate
    return None

df["home_team"] = df.apply(infer_home, axis=1)

# Checks & summaries
raw_distinct = df["venue"].nunique(dropna=False)
canon_distinct = df["venue_canon"].nunique(dropna=False)
print(f"[Check] distinct venues RAW={raw_distinct}  CANON={canon_distinct}  Δ={canon_distinct - raw_distinct}")

vc = df["venue_canon"].value_counts().rename_axis("venue_canon").reset_index(name="matches")
vc.to_csv(OUT_COUNTS, index=False)

home_sum = df["home_team"].fillna("None").value_counts().rename_axis("home_team").reset_index(name="matches")
home_sum.to_csv(OUT_HOME_SUM, index=False)

# Save final with key cols
cols = ["match_id","date","season","team1","team2","city","venue","venue_canon","city_from_venue","home_team"]
df = df.reindex(columns=cols)
df.to_csv(OUT_APPLIED, index=False)

print("\n[OK] Wrote:")
print(" ", OUT_APPLIED)
print(" ", OUT_COUNTS)
print(" ", OUT_HOME_SUM)
if unmapped:
    print(" ", OUT_UNMAPPED)
