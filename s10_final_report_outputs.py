from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

# =========================================================
# Paths
# =========================================================
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"

BASELINE_RESULTS = ROOT / "results" / "checkpoint_models" / "checkpoint_model_results.csv"
FOCUSED_RESULTS = ROOT / "results" / "checkpoint_models_focused_v3" / "checkpoint_model_results_focused_v3.csv"

OUT = ROOT / "results" / "final_report_outputs"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILES = {
    5: DATA / "checkpoint_5.csv",
    10: DATA / "checkpoint_10.csv",
    15: DATA / "checkpoint_15.csv",
    19: DATA / "checkpoint_19.csv",
}

CHECKPOINT_FE_FILES = {
    5: DATA / "checkpoint_5_fe.csv",
    10: DATA / "checkpoint_10_fe.csv",
    15: DATA / "checkpoint_15_fe.csv",
    19: DATA / "checkpoint_19_fe.csv",
}

TARGET = "win_flag"

BASELINE_FEATURES = [
    "current_score",
    "wickets_lost",
    "wickets_in_hand",
    "runs_to_go",
    "balls_remaining",
    "overs_remaining",
    "current_run_rate",
    "required_run_rate",
]

FOCUSED_FEATURES = BASELINE_FEATURES + [
    "rrr_minus_crr",
    "rrr_x_wickets_in_hand",
    "runs_to_go_x_wickets_in_hand",
    "runs_per_ball_needed",
    "target_completed_pct",
]

# =========================================================
# Helpers
# =========================================================
def save_checkpoint_sizes():
    rows = []
    for cp, fp in CHECKPOINT_FILES.items():
        df = pd.read_csv(fp)
        rows.append({
            "checkpoint_over": cp,
            "rows": len(df),
            "unique_matches": df["match_id"].nunique(),
            "chasing_win_rate": round(df["win_flag"].mean(), 4)
        })
    out_df = pd.DataFrame(rows).sort_values("checkpoint_over")
    out_df.to_csv(OUT / "table_checkpoint_dataset_sizes.csv", index=False)
    print("[OK] Saved table_checkpoint_dataset_sizes.csv")
    print(out_df)
    return out_df


def save_best_baseline_models():
    df = pd.read_csv(BASELINE_RESULTS)

    best_rows = []
    for cp in sorted(df["checkpoint_over"].unique()):
        sub = df[df["checkpoint_over"] == cp].copy()
        sub = sub.sort_values(["auc", "log_loss", "brier_score"], ascending=[False, True, True])
        best = sub.iloc[0]

        best_rows.append({
            "checkpoint_over": int(best["checkpoint_over"]),
            "model": best["model"],
            "auc": round(best["auc"], 4),
            "log_loss": round(best["log_loss"], 4),
            "brier_score": round(best["brier_score"], 4),
            "n_rows": int(best["n_rows"])
        })

    out_df = pd.DataFrame(best_rows).sort_values("checkpoint_over")
    out_df.to_csv(OUT / "table_best_baseline_models.csv", index=False)
    print("[OK] Saved table_best_baseline_models.csv")
    print(out_df)
    return out_df


def save_baseline_vs_focused_logistic():
    base = pd.read_csv(BASELINE_RESULTS)
    foc = pd.read_csv(FOCUSED_RESULTS)

    # Keep only logistic regression rows
    base_log = base[base["model"] == "logistic_regression"].copy()
    foc_log = foc[
        (foc["model"] == "logistic_regression") &
        (foc["feature_variant"] == "focused")
    ].copy()

    # Keep only ONE best baseline logistic row per checkpoint
    base_log = (
        base_log.sort_values(
            ["checkpoint_over", "auc", "log_loss", "brier_score"],
            ascending=[True, False, True, True]
        )
        .groupby("checkpoint_over", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    # Keep only ONE best focused logistic row per checkpoint
    foc_log = (
        foc_log.sort_values(
            ["checkpoint_over", "auc", "log_loss", "brier_score"],
            ascending=[True, False, True, True]
        )
        .groupby("checkpoint_over", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    merged = (
        base_log[["checkpoint_over", "auc", "log_loss", "brier_score"]]
        .rename(columns={
            "auc": "baseline_auc",
            "log_loss": "baseline_log_loss",
            "brier_score": "baseline_brier"
        })
        .merge(
            foc_log[["checkpoint_over", "auc", "log_loss", "brier_score"]]
            .rename(columns={
                "auc": "focused_auc",
                "log_loss": "focused_log_loss",
                "brier_score": "focused_brier"
            }),
            on="checkpoint_over",
            how="inner"
        )
        .sort_values("checkpoint_over")
        .reset_index(drop=True)
    )

    merged["auc_diff_focused_minus_baseline"] = merged["focused_auc"] - merged["baseline_auc"]
    merged["logloss_diff_focused_minus_baseline"] = merged["focused_log_loss"] - merged["baseline_log_loss"]
    merged["brier_diff_focused_minus_baseline"] = merged["focused_brier"] - merged["baseline_brier"]

    merged.to_csv(OUT / "table_logistic_baseline_vs_focused.csv", index=False)
    print("[OK] Saved table_logistic_baseline_vs_focused.csv")
    print(merged)
    return merged


def plot_best_baseline_metric(best_df, metric, ylabel, filename):
    plt.figure(figsize=(7, 4.5))
    plt.plot(best_df["checkpoint_over"], best_df[metric], marker="o")
    plt.xlabel("Checkpoint over")
    plt.ylabel(ylabel)
    plt.title(f"Best baseline {ylabel} by checkpoint")
    plt.xticks(best_df["checkpoint_over"])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {filename}")


def plot_logistic_baseline_vs_focused(comp_df, metric_base, metric_foc, ylabel, filename):
    plt.figure(figsize=(7, 4.5))
    plt.plot(comp_df["checkpoint_over"], comp_df[metric_base], marker="o", label="Baseline")
    plt.plot(comp_df["checkpoint_over"], comp_df[metric_foc], marker="o", label="Focused")
    plt.xlabel("Checkpoint over")
    plt.ylabel(ylabel)
    plt.title(f"Logistic regression: baseline vs focused ({ylabel})")
    plt.xticks(comp_df["checkpoint_over"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {filename}")


def fit_lr_15_feature_importance():
    df = pd.read_csv(CHECKPOINT_FE_FILES[15])

    X = df[FOCUSED_FEATURES].copy()
    y = df[TARGET].astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

    pipe.fit(X, y)

    coefs = pipe.named_steps["model"].coef_[0]
    imp_df = pd.DataFrame({
        "feature": FOCUSED_FEATURES,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs)
    }).sort_values("abs_coefficient", ascending=False)

    imp_df.to_csv(OUT / "table_lr15_feature_importance.csv", index=False)

    top = imp_df.head(10).sort_values("coefficient")
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["coefficient"])
    plt.xlabel("Standardised logistic coefficient")
    plt.ylabel("Feature")
    plt.title("Checkpoint 15: Logistic regression coefficients")
    plt.tight_layout()
    plt.savefig(OUT / "figure_lr15_coefficients.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("[OK] Saved table_lr15_feature_importance.csv")
    print("[OK] Saved figure_lr15_coefficients.png")
    print(imp_df.head(10))
    return imp_df


def fit_gb_19_feature_importance():
    df = pd.read_csv(CHECKPOINT_FILES[19])

    X = df[BASELINE_FEATURES].copy()
    y = df[TARGET].astype(int)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    pipe.fit(X, y)

    importances = pipe.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({
        "feature": BASELINE_FEATURES,
        "importance": importances
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(OUT / "table_gb19_feature_importance.csv", index=False)

    top = imp_df.head(10).sort_values("importance")
    plt.figure(figsize=(8, 5))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Checkpoint 19: Gradient boosting feature importance")
    plt.tight_layout()
    plt.savefig(OUT / "figure_gb19_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("[OK] Saved table_gb19_feature_importance.csv")
    print("[OK] Saved figure_gb19_feature_importance.png")
    print(imp_df)
    return imp_df


# =========================================================
# Main
# =========================================================
print("[Info] Script: S10 Final Report Outputs")

sizes_df = save_checkpoint_sizes()
best_base_df = save_best_baseline_models()
comp_df = save_baseline_vs_focused_logistic()

plot_best_baseline_metric(best_base_df, "auc", "AUC", "figure_auc_by_checkpoint.png")
plot_best_baseline_metric(best_base_df, "log_loss", "Log Loss", "figure_logloss_by_checkpoint.png")
plot_best_baseline_metric(best_base_df, "brier_score", "Brier Score", "figure_brier_by_checkpoint.png")

plot_logistic_baseline_vs_focused(
    comp_df,
    "baseline_auc",
    "focused_auc",
    "AUC",
    "figure_logistic_baseline_vs_focused_auc.png"
)

lr_imp = fit_lr_15_feature_importance()
gb_imp = fit_gb_19_feature_importance()

print("\n[Done] Final report outputs saved to:")
print(OUT)