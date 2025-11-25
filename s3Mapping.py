# ===========================================
# STEP 3 (B): APPLY TEAM ALIAS MAPPING + AUDITS
# ===========================================
from pathlib import Path
import pandas as pd
import json

VERSION = "S3.apply.v1.2"

# ---------- Paths ----------
DATA = Path.home() / "Documents" / "DSP" / "data"
IN_OVERVIEW = DATA / "_raw_matches_overview.csv"
IN_MAP_JSON = DATA / "_team_alias_mapping.json"

OUT_CANON   = DATA / "_teams_alias_applied.csv"
OUT_COUNTS  = DATA / "_team_canonical_counts.csv"
OUT_CHANGES = DATA / "_team_alias_before_after.csv"
OUT_UNMAPPED= DATA / "_team_unmapped_names.csv"

print(f"[Info] Script: {VERSION}")
assert IN_OVERVIEW.exists(), f"Missing: {IN_OVERVIEW} (run Step 2)"
assert IN_MAP_JSON.exists(), f"Missing: {IN_MAP_JSON} (run S3_detect_export.py)"

# ---------- Load ----------
df = pd.read_csv(IN_OVERVIEW)
alias_map = json.loads(IN_MAP_JSON.read_text(encoding="utf-8"))

# ---------- Split teams "A | B" ----------
def split_teams(s):
    if not isinstance(s, str):
        return (None, None)
    parts = [p.strip() for p in s.split("|")]
    if len(parts) == 1:
        return (parts[0], None)
    return (parts[0], parts[1])

df[["team1_raw", "team2_raw"]] = df["teams"].apply(split_teams).apply(pd.Series)

# ---------- Pre-check: unmapped names (should be zero after identity_map) ----------
raw_names = pd.Series(pd.unique(pd.concat([df["team1_raw"], df["team2_raw"]], ignore_index=True))).dropna()
unmapped = sorted([t for t in raw_names if t not in alias_map])
if unmapped:
    pd.DataFrame({"unmapped": unmapped}).to_csv(OUT_UNMAPPED, index=False)
    print(f"[WARN] Found {len(unmapped)} raw team names missing from mapping → {OUT_UNMAPPED}")
else:
    print("[OK] All raw team names exist in mapping (good).")

# ---------- Apply mapping ----------
def norm_team(name):
    if not isinstance(name, str) or not name.strip():
        return name
    return alias_map.get(name, name)  # identity for any name not overridden

df["team1"] = df["team1_raw"].apply(norm_team)
df["team2"] = df["team2_raw"].apply(norm_team)

# ---------- Before/After distincts ----------
before = pd.Series(pd.unique(pd.concat([df["team1_raw"], df["team2_raw"]], ignore_index=True))).dropna()
after  = pd.Series(pd.unique(pd.concat([df["team1"],   df["team2"]],   ignore_index=True))).dropna()

print(f"[Teams] distinct BEFORE aliasing: {len(before)}")
print(f"[Teams] distinct AFTER  aliasing: {len(after)}")
if len(after) > len(before):
    print("[ERROR] After > Before — mapping is wrong. Abort.")
    raise SystemExit(1)

# ---------- What changed (audit) ----------
changes = []
for name in sorted(before):
    mapped = alias_map.get(name, name)
    if mapped != name:
        changes.append({"original": name, "canonical": mapped})
changes_df = pd.DataFrame(changes).sort_values(["canonical", "original"])
if not changes_df.empty:
    print("\n[Alias changes] (up to 20 shown)")
    print(changes_df.head(20).to_string(index=False))
else:
    print("\n[Alias changes] none")

# ---------- Canonical usage counts ----------
stacked = pd.concat([df["team1"], df["team2"]]).dropna()
canon_counts = stacked.value_counts().rename_axis("team").reset_index(name="matches_appeared")
canon_counts.to_csv(OUT_COUNTS, index=False)

# ---------- Save outputs ----------
cols = ["match_id","date","season","city","venue","match_type","teams",
        "team1_raw","team2_raw","team1","team2"]
df = df.reindex(columns=cols)
df.to_csv(OUT_CANON, index=False)
changes_df.to_csv(OUT_CHANGES, index=False)

print("\n[OK] Wrote:")
print(" ", OUT_CANON)
print(" ", OUT_COUNTS)
print(" ", OUT_CHANGES if not changes_df.empty else "(no alias changes file written if empty)")
if unmapped:
    print(" ", OUT_UNMAPPED)

print("\n[Done]")