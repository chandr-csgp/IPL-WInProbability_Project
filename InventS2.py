from pathlib import Path
import json
import pandas as pd
from collections import Counter


DATA_ROOT = Path.home() / "Documents" / "DSP" / "ipl_json"
OUT = Path.home() / "Documents" / "DSP" / "data"
OUT.mkdir(parents=True, exist_ok=True)

def safe_print_df(df: pd.DataFrame, n: int = 12, title: str = ""):
    if title:
        print(title)
    try:
        print(df.head(n).to_string(index=False))
    except Exception as e:
        print(f"(could not pretty-print due to: {e})")
        print(df.head(n))


files = sorted(DATA_ROOT.glob("*.json"))
print("[Inventory] files:", len(files))
assert files, "No JSON files found."

teams_counter = Counter()
venues_counter = Counter()
cities_counter = Counter()
seasons_counter = Counter()
comp_counter = Counter()
mtype_counter = Counter()

rows = []  # one row per match for overview

for fp in files:
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    info = d.get("info", {}) or {}

    # Normalize season to string for counting; keep None as None
    season = info.get("season")
    season_str = str(season) if season is not None else None
    seasons_counter[season_str] += 1

    comp = info.get("competition")
    mtype = info.get("match_type")
    if comp:  comp_counter[comp] += 1
    if mtype: mtype_counter[mtype] += 1

    for t in info.get("teams", []) or []:
        teams_counter[t] += 1
    if info.get("venue"): venues_counter[info.get("venue")] += 1
    if info.get("city"):  cities_counter[info.get("city")] += 1

    rows.append({
        "match_id": fp.stem,
        "date": (info.get("dates") or [None])[0],
        "season": season_str,
        "teams": " | ".join(info.get("teams", []) or []),
        "venue": info.get("venue"),
        "city": info.get("city"),
        "competition": comp,
        "match_type": mtype,
    })

# ---------- Build DataFrames ----------
df_matches_overview = pd.DataFrame(rows).sort_values("date", na_position="last")

df_teams  = (
    pd.DataFrame({"team": list(teams_counter.keys()), "count": list(teams_counter.values())})
    .sort_values("count", ascending=False)
)

df_venues = (
    pd.DataFrame({"venue": list(venues_counter.keys()), "count": list(venues_counter.values())})
    .sort_values("count", ascending=False)
)

df_cities = (
    pd.DataFrame({"city": list(cities_counter.keys()), "count": list(cities_counter.values())})
    .sort_values("count", ascending=False)
)

# FIX: seasons may be mixed types → coerce helper column for sorting
df_seasons = (
    pd.DataFrame({"season": list(seasons_counter.keys()), "count": list(seasons_counter.values())})
    .assign(season_num=lambda d: pd.to_numeric(d["season"], errors="coerce"))
    .sort_values(["season_num", "season"], na_position="last")
    .drop(columns="season_num")
)

df_comp = (
    pd.DataFrame({"competition": list(comp_counter.keys()), "count": list(comp_counter.values())})
    .sort_values("count", ascending=False)
)

df_mtype = (
    pd.DataFrame({"match_type": list(mtype_counter.keys()), "count": list(mtype_counter.values())})
    .sort_values("count", ascending=False)
)


df_matches_overview.to_csv(OUT / "_raw_matches_overview.csv", index=False)
df_teams.to_csv(OUT / "_raw_teams_counts.csv", index=False)
df_venues.to_csv(OUT / "_raw_venues_counts.csv", index=False)
df_cities.to_csv(OUT / "_raw_cities_counts.csv", index=False)
df_seasons.to_csv(OUT / "_raw_seasons_counts.csv", index=False)
df_comp.to_csv(OUT / "_raw_competition_counts.csv", index=False)
df_mtype.to_csv(OUT / "_raw_matchtype_counts.csv", index=False)


safe_print_df(df_teams, title="\n[Teams] distinct: {}  Top:".format(len(df_teams)))
safe_print_df(df_venues, title="\n[Venues] distinct: {}  Top:".format(len(df_venues)))
safe_print_df(df_cities, title="\n[Cities] distinct: {}  Top:".format(len(df_cities)))

print("\n[Seasons] coverage (sorted numerically where possible):")
try:
    print(df_seasons.to_string(index=False))
except Exception as e:
    print(f"(could not pretty-print seasons due to: {e})")
    print(df_seasons.head(30))

print("\n[Competition] values:")
safe_print_df(df_comp, n=30)

print("\n[Match types] values:")
safe_print_df(df_mtype, n=30)

print(f"\n[OK] wrote reports to: {OUT}")

print("\n[OK] done.")