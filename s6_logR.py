
# STEP 6: LOGISTIC REGRESSION +  TESTING

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

VERSION = "S6.v1.2"
print(f"[Info] Script: {VERSION}")

# Paths
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
IN_FEATS = DATA / "matches_features.csv"
OUT_SUMMARY = DATA / "logit_results_summary.txt"

assert IN_FEATS.exists(), f"Missing {IN_FEATS}"

# Load dataset
df = pd.read_csv(IN_FEATS)
print(f"[Data] Loaded {len(df)} matches")
print(f"[Data] Columns: {df.columns.tolist()}")

# Check which columns actually exist
required_cols = ["win_flag_team1", "toss_won_flag_team1", "bat_first_flag_team1"]
optional_cols = ["home_flag_team1", "neutral_venue_flag"]

# Only use columns that exist
subset_cols = [col for col in required_cols if col in df.columns]
available_optional = [col for col in optional_cols if col in df.columns]

print(f"[Data] Using columns: {subset_cols + available_optional}")

# Keep valid rows
df = df.dropna(subset=subset_cols)
print(f"[Data] Using {len(df)} valid rows after dropping NaN")

# Hypothesis tests
def prop_test(successes, total, label):
    if total == 0:
        print(f"{label}: No data available")
        return np.nan, np.nan
    p_hat = successes / total
    se = np.sqrt(0.25 / total)
    z = (p_hat - 0.5) / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"{label}: {successes}/{total} = {p_hat:.3f}, p={p_val:.4f}")
    return p_hat, p_val

print("\n=== Hypothesis Tests (p, q, h) ===")

# Toss advantage: Among matches where team1 won toss, what % did team1 win?
toss_won_mask = df["toss_won_flag_team1"] == 1.0
toss_won_and_match_won = ((df["toss_won_flag_team1"] == 1.0) & (df["win_flag_team1"] == 1.0)).sum()
toss_won_total = toss_won_mask.sum()
p_toss, pval_toss = prop_test(toss_won_and_match_won, toss_won_total, "Toss advantage")

# Bat-first advantage: Among matches where team1 batted first, what % did team1 win?
bat_first_mask = df["bat_first_flag_team1"] == 1.0
bat_first_and_won = ((df["bat_first_flag_team1"] == 1.0) & (df["win_flag_team1"] == 1.0)).sum()
bat_first_total = bat_first_mask.sum()
p_bat, pval_bat = prop_test(bat_first_and_won, bat_first_total, "Bat-first advantage")

# Home advantage: Only if column exists
if "home_flag_team1" in df.columns:
    home_mask = df["home_flag_team1"] == 1.0
    home_and_won = ((df["home_flag_team1"] == 1.0) & (df["win_flag_team1"] == 1.0)).sum()
    home_total = home_mask.sum()
    p_home, pval_home = prop_test(home_and_won, home_total, "Home advantage")
else:
    print("Home advantage: Column not available")
    p_home, pval_home = np.nan, np.nan
    home_and_won, home_total = 0, 0

#  Logistic Regression 
# Build feature list based on available columns
feature_cols = ["toss_won_flag_team1", "bat_first_flag_team1"]
if "home_flag_team1" in df.columns:
    feature_cols.append("home_flag_team1")

X = df[feature_cols]
X = sm.add_constant(X)
y = df["win_flag_team1"]

model = sm.Logit(y, X).fit(disp=False)

print("\n=== Logistic Regression Summary ===")
print(model.summary())

# Odds ratios
odds_ratios = np.exp(model.params)
print("\n=== Odds Ratios ===")
print(odds_ratios)

# Save report 
with open(OUT_SUMMARY, "w") as f:
    f.write(f"IPL Match Win Probability Analysis - {VERSION}\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("HYPOTHESIS TESTS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Toss advantage: {toss_won_and_match_won}/{toss_won_total} = {p_toss:.3f}, p={pval_toss:.4f}\n")
    f.write(f"Bat-first advantage: {bat_first_and_won}/{bat_first_total} = {p_bat:.3f}, p={pval_bat:.4f}\n")
    if "home_flag_team1" in df.columns:
        f.write(f"Home advantage: {home_and_won}/{home_total} = {p_home:.3f}, p={pval_home:.4f}\n\n")
    else:
        f.write("Home advantage: Not available\n\n")
    
    f.write("LOGISTIC REGRESSION RESULTS\n")
    f.write("-" * 60 + "\n")
    f.write(model.summary().as_text())
    f.write("\n\nOdds Ratios:\n")
    f.write(odds_ratios.to_string())

print(f"\n[OK] Results saved → {OUT_SUMMARY}")