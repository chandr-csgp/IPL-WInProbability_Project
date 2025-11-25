
# STEP 3: TEAM ALIAS DETECTION + MAPPING 

from pathlib import Path
import pandas as pd
from rapidfuzz import fuzz
import json
import sys

VERSION = "S3.v1.2"
# Paths
DATA = Path.home() / "Documents" / "DSP" / "data"
TEAMS_FILE = DATA / "_raw_teams_counts.csv"
SUGG_CSV   = DATA / "_team_alias_suggestions.csv"
MAP_JSON   = DATA / "_team_alias_mapping.json"
MAP_CSV    = DATA / "_team_alias_mapping.csv"

print(f"[Info] Script: {VERSION}")
assert TEAMS_FILE.exists(), f"Run Step 2 first — missing {TEAMS_FILE}"

# Load & prep 
df = pd.read_csv(TEAMS_FILE)
teams = sorted(set(t for t in df["team"].dropna() if isinstance(t, str) and t.strip()))
print(f"[OK] Loaded {len(teams)} unique team names from {TEAMS_FILE.name}")

# Fuzzy suggestions 
pairs = []
for i, t1 in enumerate(teams):
    for t2 in teams[i+1:]:
        score = fuzz.token_sort_ratio(t1, t2)
        if score >= 80:
            pairs.append((t1, t2, score))

sugg = pd.DataFrame(pairs, columns=["name1", "name2", "similarity"])

def token_overlap(a: str, b: str) -> int:
    A = set(a.lower().split())
    B = set(b.lower().split())
    return len(A & B)

if not sugg.empty:
    sugg["token_overlap"] = sugg.apply(lambda r: token_overlap(r["name1"], r["name2"]), axis=1)
    sugg = sugg.sort_values(["similarity", "token_overlap"], ascending=[False, False]).reset_index(drop=True)
sugg.to_csv(SUGG_CSV, index=False)
print(f"[OK] Suggestions written → {SUGG_CSV} (rows: {len(sugg)})")
if not sugg.empty:
    print("\n[Preview] Top suggestions (score ≥ 80):")
    print(sugg.head(12).to_string(index=False))

# - Build mapping (identity + known renames) 
identity_map = {t: t for t in teams}
identity_map.update({
    # Confirmed historical / variants → canonical
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Deccan Chargers": "Sunrisers Hyderabad",
    # NOTE: Keep Gujarat Lions ≠ Gujarat Titans (different franchises)
})

# Checks on mapping 
missing_keys = sorted(set(teams) - set(identity_map.keys()))
if missing_keys:
    print("\n[WARN] Teams missing from mapping keys (will default to identity if applied separately):")
    for t in missing_keys: print("  -", t)

# Targets sanity: every canonical should be a known team label (present in set) or a sensible canonical we accept.
# Here, all targets are either themselves or real team labels we saw (e.g., Sunrisers Hyderabad).
unknown_targets = sorted(set(identity_map.values()) - set(teams))
if unknown_targets:
    print("\n[INFO] Canonical targets not in raw team list (acceptable if they exist elsewhere in data):")
    for t in unknown_targets: print("  -", t)

# One-to-many check print summary:
from collections import defaultdict
rev = defaultdict(list)
for k, v in identity_map.items():
    rev[v].append(k)

print("\n[Map] Canonical → originals (groups with more than 1):")
multi = {canon: srcs for canon, srcs in rev.items() if len(srcs) > 1}
if multi:
    for canon, srcs in sorted(multi.items()):
        print(f"  {canon}: {', '.join(sorted(srcs))}")
else:
    print("  (none)")

# Save mapping
MAP_JSON.write_text(json.dumps(identity_map, indent=2), encoding="utf-8")
pd.DataFrame([{"original": k, "canonical": v} for k, v in identity_map.items()]).to_csv(MAP_CSV, index=False)
print(f"\n[OK] Mapping saved:\n  {MAP_JSON}\n  {MAP_CSV}")

print("\n[Done] Review suggestions & mapping. Next: run S3_apply_aliases.py")
