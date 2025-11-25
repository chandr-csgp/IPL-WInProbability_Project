

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

# sklearn imports
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    brier_score_loss
)

from sklearn.calibration import calibration_curve

# XGBoost optional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except:
    HAS_XGB = False

ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
FIGS = ROOT / "figures"
MODELS = ROOT / "models"

FIGS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SNAPSHOTS = DATA / "inplay_snapshots.csv"
assert SNAPSHOTS.exists(), f"Missing snapshot file: {SNAPSHOTS}"

print("[Info] Script: S10.v1.0 - Feature enrichment + multi-model comparison")


df = pd.read_csv(SNAPSHOTS)
print(f"[Data] Loaded snapshots: {len(df)} rows")

# Remove negative values if any
df = df[df["runs_to_go"] >= 0]
df = df[df["balls_remaining"] >= 0]



df["rr_diff"] = df["current_run_rate"] - df["required_run_rate"]
df["overs_done"] = (120 - df["balls_remaining"]) / 6
df["run_rate_ratio"] = df["current_run_rate"] / (df["required_run_rate"] + 1e-6)
df["pressure_index"] = df["runs_to_go"] / (df["wickets_in_hand"] + 1)
df["balls_per_wicket"] = (120 - df["balls_remaining"]) / (10 - df["wickets_in_hand"] + 1)
df["runs_per_ball"] = (df["target_runs"] - df["runs_to_go"]) / (120 - df["balls_remaining"] + 1)
df["is_death_over"] = (df["overs_done"] >= 15).astype(int)



features = [
    "runs_to_go", "balls_remaining", "wickets_in_hand",
    "current_run_rate", "required_run_rate",
    "rr_diff", "overs_done", "run_rate_ratio",
    "pressure_index", "balls_per_wicket", "runs_per_ball",
    "is_death_over"
]

X = df[features]
y = df["win_flag"]
groups = df["match_id"]

gkf = GroupKFold(n_splits=5)

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    break

print(f"[Split] Train: {len(X_train)} | Test: {len(X_test)}")



def evaluate(model_name, model, X_train, y_train, X_test, y_test, fig_name):
    model.fit(X_train, y_train)

    p_tr = model.predict_proba(X_train)[:, 1]
    p_te = model.predict_proba(X_test)[:, 1]

    auc_tr = roc_auc_score(y_train, p_tr)
    auc_te = roc_auc_score(y_test, p_te)

    ll_tr = log_loss(y_train, p_tr)
    ll_te = log_loss(y_test, p_te)

    br_tr = brier_score_loss(y_train, p_tr)
    br_te = brier_score_loss(y_test, p_te)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, p_te, n_bins=15)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Probability")
    plt.title(f"Calibration Curve - {model_name}")
    plt.grid(True)
    plt.savefig(FIGS / f"{fig_name}_calibration.png")
    plt.close()

    # Save model
    joblib.dump(model, MODELS / f"{fig_name}.pkl")

    return {
        "model": model_name,
        "auc_train": auc_tr, "auc_test": auc_te,
        "logloss_train": ll_tr, "logloss_test": ll_te,
        "brier_train": br_tr, "brier_test": br_te
    }



results = []

# Logistic Regression
logit = LogisticRegression(max_iter=2000)
results.append(evaluate("Logistic Regression", logit, X_train, y_train, X_test, y_test, "S10_logit"))

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)
results.append(evaluate("Random Forest", rf, X_train, y_train, X_test, y_test, "S10_rf"))

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3
)
results.append(evaluate("Gradient Boosting", gb, X_train, y_train, X_test, y_test, "S10_gb"))

# XGBoost (if installed)
if HAS_XGB:
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        max_depth=4,
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9
    )
    results.append(evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test, "S10_xgb"))



df_results = pd.DataFrame(results)
df_results.to_csv(DATA / "S10_model_comparison.csv", index=False)

print("\n=== Model Comparison ===")
print(df_results)

print("\n[OK] Saved outputs to:")
print(f" - {DATA/'S10_model_comparison.csv'}")
print(f" - Plots in: {FIGS}")
print(f" - Models in: {MODELS}")
