
# STEP 9: DYNAMIC MODELLING (v1.0)


from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try XGBoost if available
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Paths 
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
FIGS = ROOT / "figures"
MODELS = ROOT / "models"
IN_SNAP = DATA / "inplay_snapshots.csv"

FIGS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

print("[Info] Script: S9.v1.0 - Dynamic modelling")

#  Load 
df = pd.read_csv(IN_SNAP)
print(f"[Data] Loaded snapshots: {len(df)} rows")

# Features & label
FEATURES = [
    "runs_to_go",
    "balls_remaining",
    "wickets_in_hand",
    "current_run_rate",
    "required_run_rate",
]
TARGET = "win_flag"
GROUP = "match_id"

# Basic cleaning
df = df.dropna(subset=FEATURES + [TARGET, GROUP]).copy()
X = df[FEATURES].astype(float).values
y = df[TARGET].astype(int).values
groups = df[GROUP].astype(str).values

#  Group-aware train/test split (by match) 
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"[Split] Train rows: {len(y_train)} | Test rows: {len(y_test)} | Matches (unique): "
      f"{df.loc[train_idx, GROUP].nunique()} train / {df.loc[test_idx, GROUP].nunique()} test")

#  Helper: evaluation & calibration 
def evaluate_model(name, clf, X_tr, y_tr, X_te, y_te, save_prefix):
    """Compute metrics and draw calibration curve; return dict of metrics."""
    # Predict probabilities
    p_tr = clf.predict_proba(X_tr)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]

    # Clip to avoid log(0) without using deprecated eps arg
    p_tr = np.clip(p_tr, 1e-15, 1 - 1e-15)
    p_te = np.clip(p_te, 1e-15, 1 - 1e-15)

    # Metrics
    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_te = roc_auc_score(y_te, p_te)
    ll_tr  = log_loss(y_tr, p_tr, labels=[0, 1])
    ll_te  = log_loss(y_te, p_te, labels=[0, 1])
    br_tr  = brier_score_loss(y_tr, p_tr)
    br_te  = brier_score_loss(y_te, p_te)

    print(f"\n=== {name} ===")
    print(f"AUC      : train {auc_tr:.3f} | test {auc_te:.3f}")
    print(f"Log Loss : train {ll_tr:.3f} | test {ll_te:.3f}")
    print(f"Brier    : train {br_tr:.3f} | test {br_te:.3f}")

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=20, strategy="quantile")

    plt.figure(figsize=(5.2, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration – {name}")
    plt.tight_layout()
    out_png = FIGS / f"{save_prefix}_calibration.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[OK] Saved calibration curve → {out_png}")

    return {
        "auc_train": auc_tr, "auc_test": auc_te,
        "logloss_train": ll_tr, "logloss_test": ll_te,
        "brier_train": br_tr, "brier_test": br_te,
    }


# Model 1: Logistic Regression 
logit = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=200, n_jobs=None,
    )),
])
logit.fit(X_train, y_train)
metrics_logit = evaluate_model("Logistic Regression", logit, X_train, y_train, X_test, y_test, "S9_logit")
joblib.dump(logit, MODELS / "S9_logit.pkl")
print(f"[OK] Saved model → {MODELS / 'S9_logit.pkl'}")

# Export coefficients
coef = logit.named_steps["clf"].coef_[0]
coef_df = pd.DataFrame({"feature": FEATURES, "coef": coef}).sort_values("coef", ascending=False)
coef_csv = DATA / "S9_logit_coefficients.csv"
coef_df.to_csv(coef_csv, index=False)
print(f"[OK] Saved coefficients → {coef_csv}")

plt.figure(figsize=(7, 4))
plt.barh(coef_df["feature"], coef_df["coef"])
plt.gca().invert_yaxis()
plt.title("Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig(FIGS / "S9_logit_coefficients.png", dpi=180)
plt.close()

#  Model 2: Random Forest 
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight=None,
)
rf.fit(X_train, y_train)
metrics_rf = evaluate_model("Random Forest", rf, X_train, y_train, X_test, y_test, "S9_rf")
joblib.dump(rf, MODELS / "S9_rf.pkl")
print(f"[OK] Saved model → {MODELS / 'S9_rf.pkl'}")

# RF feature importance
imp = rf.feature_importances_
imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp}).sort_values("importance", ascending=True)
plt.figure(figsize=(7, 4))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(FIGS / "S9_rf_feature_importance.png", dpi=180)
plt.close()

#  Model 3: XGBoost 
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model_name = "XGBoost"
    fname = "S9_xgb.pkl"
else:
    xgb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
    )
    model_name = "GradientBoosting (fallback)"
    fname = "S9_gb.pkl"

xgb.fit(X_train, y_train)
metrics_xgb = evaluate_model(model_name, xgb, X_train, y_train, X_test, y_test, "S9_xgb")
joblib.dump(xgb, MODELS / fname)
print(f"[OK] Saved model → {MODELS / fname}")

# Tree importance
try:
    if hasattr(xgb, "feature_importances_"):
        imp2 = xgb.feature_importances_
        imp2_df = pd.DataFrame({"feature": FEATURES, "importance": imp2}).sort_values("importance", ascending=True)
        plt.figure(figsize=(7, 4))
        plt.barh(imp2_df["feature"], imp2_df["importance"])
        plt.title(f"{model_name} Feature Importance")
        plt.tight_layout()
        plt.savefig(FIGS / "S9_xgb_feature_importance.png", dpi=180)
        plt.close()
except Exception:
    pass

#  Save metrics summary 
summary = pd.DataFrame({
    "model": ["logit", "rf", "xgb/gb"],
    "auc_train": [metrics_logit["auc_train"], metrics_rf["auc_train"], metrics_xgb["auc_train"]],
    "auc_test":  [metrics_logit["auc_test"],  metrics_rf["auc_test"],  metrics_xgb["auc_test"]],
    "logloss_train": [metrics_logit["logloss_train"], metrics_rf["logloss_train"], metrics_xgb["logloss_train"]],
    "logloss_test":  [metrics_logit["logloss_test"],  metrics_rf["logloss_test"],  metrics_xgb["logloss_test"]],
    "brier_train":   [metrics_logit["brier_train"],   metrics_rf["brier_train"],   metrics_xgb["brier_train"]],
    "brier_test":    [metrics_logit["brier_test"],    metrics_rf["brier_test"],    metrics_xgb["brier_test"]],
})
out_csv = DATA / "S9_metrics_summary.csv"
summary.to_csv(out_csv, index=False)
print(f"\n[OK] Saved metrics summary → {out_csv}")

print("\n=== Done: Step 9 ===")
