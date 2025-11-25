
# STEP 4: VENUE DETECT + MAPPING 

from pathlib import Path
import pandas as pd
import json

VERSION = "S4.detect_export.v1.1"

DATA = Path.home() / "Documents" / "DSP" / "data"
IN_VENUES = DATA / "_raw_venues_counts.csv"
IN_OVERVIEW = DATA / "_raw_matches_overview.csv"

OUT_SUGG = DATA / "_venue_variants_suspects.csv"
OUT_MAP_JSON = DATA / "_venue_fix_mapping.json"
OUT_MAP_CSV  = DATA / "_venue_fix_mapping.csv"

print(f"[Info] Script: {VERSION}")
assert IN_VENUES.exists(), f"Missing {IN_VENUES} (run Step 2)"
assert IN_OVERVIEW.exists(), f"Missing {IN_OVERVIEW} (run Step 2)"

# Load unique venues + counts
dfv = pd.read_csv(IN_VENUES)
dfv["venue"] = dfv["venue"].astype(str)
venues = sorted(dfv["venue"].unique())
print(f"[OK] Loaded {len(venues)} unique venue labels")

# Heuristic: flag likely variants by lowering & removing punctuation / common words
def norm_loose(s: str) -> str:
    s = s.lower().replace("stadium", "").replace("ground", "")
    keep = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
    key = "".join(keep)
    key = " ".join(key.split())
    return key.strip()

bucket = {}
for v in venues:
    key = norm_loose(v)
    bucket.setdefault(key, []).append(v)

# Build suspects: only buckets with >1 unique labels
suspects = []
for k, group in bucket.items():
    if len(group) > 1:
        total = int(dfv[dfv["venue"].isin(group)]["count"].sum())
        suspects.append({
            "loose_key": k,
            "variants": " | ".join(sorted(group)),
            "labels": len(group),
            "matches_involved": total
        })

# Robust: create with explicit columns and guard sorting when empty
cols = ["loose_key", "variants", "labels", "matches_involved"]
sugg = pd.DataFrame(suspects, columns=cols)
if not sugg.empty:
    sugg = sugg.sort_values(["labels", "matches_involved"], ascending=[False, False]).reset_index(drop=True)
    print(f"[OK] Suspect variant buckets: {len(sugg)}")
else:
    print("[OK] No variant buckets detected by heuristic (venues already consistent).")

sugg.to_csv(OUT_SUGG, index=False)
print(f"[OK] Wrote suspects → {OUT_SUGG}")

# ---------------- Venue mapping (identity + curated fixes) ----------------
venue_map = {v: v for v in venues}

# India
venue_map.update({
    "Wankhede Stadium": "Wankhede Stadium, Mumbai",
    "Brabourne Stadium": "Brabourne Stadium, Mumbai",
    "Dr DY Patil Sports Academy": "Dr DY Patil Sports Academy, Navi Mumbai",
    "DY Patil Stadium": "Dr DY Patil Sports Academy, Navi Mumbai",

    "M Chinnaswamy Stadium": "M Chinnaswamy Stadium, Bengaluru",

    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium, Chennai",
    "MA Chidambaram Stadium": "MA Chidambaram Stadium, Chennai",

    "Feroz Shah Kotla": "Arun Jaitley Stadium, Delhi",
    "Arun Jaitley Stadium": "Arun Jaitley Stadium, Delhi",

    "Eden Gardens": "Eden Gardens, Kolkata",
    "Sawai Mansingh Stadium": "Sawai Mansingh Stadium, Jaipur",

    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium, Hyderabad",

    "Punjab Cricket Association Stadium, Mohali": "IS Bindra Stadium, Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "IS Bindra Stadium, Mohali",
    "IS Bindra Stadium": "IS Bindra Stadium, Mohali",

    "Himachal Pradesh Cricket Association Stadium": "HPCA Stadium, Dharamsala",

    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    "Narendra Modi Stadium": "Narendra Modi Stadium, Ahmedabad",

    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Ekana Cricket Stadium, Lucknow",
    "Maharashtra Cricket Association Stadium": "MCA Stadium, Pune",
    "Holkar Cricket Stadium": "Holkar Cricket Stadium, Indore",
    "Shaheed Veer Narayan Singh International Stadium": "SVNS Intl Stadium, Raipur",
})

# UAE (neutral)
venue_map.update({
    "Dubai International Cricket Stadium": "Dubai International Cricket Stadium, Dubai",
    "Sheikh Zayed Stadium": "Sheikh Zayed Stadium, Abu Dhabi",
    "Sharjah Cricket Stadium": "Sharjah Cricket Stadium, Sharjah",
})

# South Africa (2009)
venue_map.update({
    "Newlands": "Newlands, Cape Town",
    "SuperSport Park": "SuperSport Park, Centurion",
    "Kingsmead": "Kingsmead, Durban",
    "New Wanderers Stadium": "Wanderers Stadium, Johannesburg",
    "St George's Park": "St George's Park, Gqeberha",
    "Buffalo Park": "Buffalo Park, East London",
    "De Beers Diamond Oval": "De Beers Diamond Oval, Kimberley",
    "OUTsurance Oval": "OUTsurance Oval, Bloemfontein",
})

# Save mapping
OUT_MAP_JSON.write_text(json.dumps(venue_map, indent=2), encoding="utf-8")
pd.DataFrame([{"original": k, "canonical": v} for k, v in sorted(venue_map.items())]).to_csv(OUT_MAP_CSV, index=False)
print(f"[OK] Venue mapping saved:\n  {OUT_MAP_JSON}\n  {OUT_MAP_CSV}")

# Preview (safe when empty)
print("\n[Preview] Variant buckets (top 10):")
print(sugg.head(10).to_string(index=False) if not sugg.empty else "(none)")
