# Temporal validation of the checkpoint win-probability models.
#
# Trains on seasons up to 2022 and tests on 2023-2024 (season 2025, present in
# the data, is excluded from both — out of scope for this comparison). Also
# reruns the original random stratified 75/25 split on the same data so both
# validation strategies can be compared side by side.
#
# For each checkpoint (5/10/15/19 overs) and model (logistic regression,
# random forest, gradient boosting): AUC, log loss, Brier score with a
# 1,000-sample bootstrap 95% CI, plus a calibration plot per checkpoint
# (temporal test set, all three models).

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILES = {
    5: DATA / "checkpoint_5.csv",
    10: DATA / "checkpoint_10.csv",
    15: DATA / "checkpoint_15.csv",
    19: DATA / "checkpoint_19.csv",
}

FEATURES = [
    "current_score",
    "wickets_lost",
    "wickets_in_hand",
    "runs_to_go",
    "balls_remaining",
    "overs_remaining",
    "current_run_rate",
    "required_run_rate",
]
TARGET = "win_flag"

TRAIN_SEASONS = {
    "2007/08", "2009", "2009/10", "2011", "2012", "2013", "2014", "2015",
    "2016", "2017", "2018", "2019", "2020/21", "2021", "2022",
}
TEST_SEASONS = {"2023", "2024"}
# season "2025" exists in the data but is excluded from both sets

N_BOOT = 1000
RANDOM_STATE = 42


def build_models():
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def bootstrap_ci(y_true, y_prob, n_boot=N_BOOT, seed=RANDOM_STATE):
    """Nonparametric bootstrap over the (fixed) test predictions."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)

    aucs, lls, briers = [], [], []
    skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            skipped += 1
            continue
        aucs.append(roc_auc_score(yt, yp))
        lls.append(log_loss(yt, yp, labels=[0, 1]))
        briers.append(brier_score_loss(yt, yp))

    def ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return lo, hi

    auc_lo, auc_hi = ci(aucs)
    ll_lo, ll_hi = ci(lls)
    br_lo, br_hi = ci(briers)

    if skipped:
        print(f"    [Warn] {skipped}/{n_boot} bootstrap resamples had a single class and were skipped")

    return {
        "auc_ci_lo": auc_lo, "auc_ci_hi": auc_hi,
        "log_loss_ci_lo": ll_lo, "log_loss_ci_hi": ll_hi,
        "brier_ci_lo": br_lo, "brier_ci_hi": br_hi,
        "n_boot_valid": len(aucs),
    }


def fit_and_score(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "auc": roc_auc_score(y_test, probs),
        "log_loss": log_loss(y_test, probs, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test, probs),
    }
    metrics.update(bootstrap_ci(y_test.values, probs))
    return metrics, probs


print("[Info] Script: validate_temporal.py")

all_results = []

for checkpoint, path in CHECKPOINT_FILES.items():
    print(f"\n{'='*70}\n[Checkpoint {checkpoint}] Loading {path.name}")
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    df["season"] = df["season"].astype(str)

    # ---- Temporal split ----
    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    test_df = df[df["season"].isin(TEST_SEASONS)]
    excluded_n = len(df) - len(train_df) - len(test_df)
    print(
        f"[Temporal] train={len(train_df)} rows, test={len(test_df)} rows, "
        f"excluded (season 2025)={excluded_n} rows"
    )

    X_train_t, y_train_t = train_df[FEATURES], train_df[TARGET].astype(int)
    X_test_t, y_test_t = test_df[FEATURES], test_df[TARGET].astype(int)

    calib_curves = {}
    for name, model in build_models().items():
        metrics, probs = fit_and_score(X_train_t, y_train_t, X_test_t, y_test_t, model)
        all_results.append({
            "checkpoint_over": checkpoint, "split": "temporal", "model": name,
            "n_train": len(X_train_t), "n_test": len(X_test_t),
            **metrics,
        })
        print(
            f"  [temporal] {name:20s} | AUC={metrics['auc']:.3f} "
            f"[{metrics['auc_ci_lo']:.3f}, {metrics['auc_ci_hi']:.3f}] | "
            f"LogLoss={metrics['log_loss']:.3f} | Brier={metrics['brier_score']:.3f}"
        )
        frac_pos, mean_pred = calibration_curve(y_test_t, probs, n_bins=8, strategy="quantile")
        calib_curves[name] = (mean_pred, frac_pos)

    # Calibration plot for this checkpoint (temporal test set, all 3 models)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for name, (mp, fp) in calib_curves.items():
        plt.plot(mp, fp, marker="o", label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration (temporal test set) — checkpoint over {checkpoint}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    fig_path = FIGS / f"calibration_temporal_checkpoint_{checkpoint}.png"
    plt.savefig(fig_path, dpi=180)
    plt.close()
    print(f"  [OK] Saved {fig_path}")

    # ---- Random stratified split (kept for comparison) ----
    X_all, y_all = df[FEATURES], df[TARGET].astype(int)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_all, y_all, test_size=0.25, random_state=RANDOM_STATE, stratify=y_all
    )
    for name, model in build_models().items():
        metrics, _ = fit_and_score(X_train_r, y_train_r, X_test_r, y_test_r, model)
        all_results.append({
            "checkpoint_over": checkpoint, "split": "random", "model": name,
            "n_train": len(X_train_r), "n_test": len(X_test_r),
            **metrics,
        })
        print(
            f"  [random]   {name:20s} | AUC={metrics['auc']:.3f} "
            f"[{metrics['auc_ci_lo']:.3f}, {metrics['auc_ci_hi']:.3f}] | "
            f"LogLoss={metrics['log_loss']:.3f} | Brier={metrics['brier_score']:.3f}"
        )

results_df = pd.DataFrame(all_results)
out_csv = RESULTS / "validate_temporal_results.csv"
results_df.to_csv(out_csv, index=False)
print(f"\n[OK] Saved full results -> {out_csv}")

print("\n=== Full Results (temporal vs random) ===")
cols = [
    "checkpoint_over", "split", "model", "n_train", "n_test",
    "auc", "auc_ci_lo", "auc_ci_hi",
    "log_loss", "log_loss_ci_lo", "log_loss_ci_hi",
    "brier_score", "brier_ci_lo", "brier_ci_hi",
]
print(results_df[cols].to_string(index=False))
