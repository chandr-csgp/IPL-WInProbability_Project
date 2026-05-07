# STEP 9B: CHECKPOINT MODELS
# Train and compare baseline vs improved feature sets
# across 5, 10, 15, and 19 over checkpoints

from pathlib import Path
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

# Paths

ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
OUT_DIR = ROOT / "results" / "checkpoint_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILES = {
    5: DATA / "checkpoint_5.csv",
    10: DATA / "checkpoint_10.csv",
    15: DATA / "checkpoint_15.csv",
    19: DATA / "checkpoint_19.csv",
}

# Feature sets

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

IMPROVED_FEATURES = BASELINE_FEATURES + [
    "rrr_x_wickets_in_hand",
    "runs_to_go_x_wickets_in_hand",
    "rrr_minus_crr",
]

TARGET = "win_flag"


# Helper functions

def add_extra_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Over/phase features
    df["overs_done"] = 20 - df["overs_remaining"]
    df["is_powerplay"] = (df["checkpoint_over"] <= 6).astype(int)
    df["is_middle_overs"] = ((df["checkpoint_over"] > 6) & (df["checkpoint_over"] <= 15)).astype(int)
    df["is_death_overs"] = (df["checkpoint_over"] > 15).astype(int)

    # Pressure features
    df["runs_per_ball_needed"] = np.where(
        df["balls_remaining"] > 0,
        df["runs_to_go"] / df["balls_remaining"],
        0.0
    )
    df["target_completed_pct"] = np.where(
        df["target_runs"] > 0,
        df["current_score"] / df["target_runs"],
        0.0
    )

    # Extra interactions
    df["crr_x_overs_remaining"] = df["current_run_rate"] * df["overs_remaining"]
    df["wickets_lost_x_crr"] = df["wickets_lost"] * df["current_run_rate"]

    return df


def get_feature_set(df: pd.DataFrame, variant: str) -> list[str]:
    if variant == "baseline":
        return BASELINE_FEATURES

    if variant == "improved":
        features = IMPROVED_FEATURES + [
            "overs_done",
            "is_powerplay",
            "is_middle_overs",
            "is_death_overs",
            "runs_per_ball_needed",
            "target_completed_pct",
            "crr_x_overs_remaining",
            "wickets_lost_x_crr",
        ]
        return features

    raise ValueError(f"Unknown feature variant: {variant}")


def build_models():
    models = {}

    # Logistic Regression
    models["logistic_regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

    # Random Forest
    models["random_forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Gradient Boosting
    models["gradient_boosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42
            ))
        ])
        print("[Info] XGBoost available and included.")
    except Exception:
        print("[Info] XGBoost not available. Skipping.")

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc": roc_auc_score(y_test, probs),
        "log_loss": log_loss(y_test, probs, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test, probs),
        "test_rows": len(y_test),
        "positive_rate_test": float(np.mean(y_test)),
    }
    return metrics, probs



# Main

print("[Info] Script: S9B - Checkpoint model comparison")

all_results = []
models = build_models()

for checkpoint, file_path in CHECKPOINT_FILES.items():
    print(f"\n{'='*70}")
    print(f"[Checkpoint {checkpoint}] Loading {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"[Data] Rows: {len(df)}, Unique matches: {df['match_id'].nunique()}")

    df = add_extra_engineered_features(df)

    for variant in ["baseline", "improved"]:
        features = get_feature_set(df, variant)

        # Keep only required columns
        use_cols = features + [TARGET]
        temp = df[use_cols].copy()

        # Drop missing target rows only
        temp = temp.dropna(subset=[TARGET]).reset_index(drop=True)

        X = temp[features]
        y = temp[TARGET].astype(int)

        print(f"\n[Checkpoint {checkpoint} | {variant}]")
        print(f"Features used: {features}")
        print(f"Rows available: {len(temp)}")
        print(f"Win rate: {y.mean():.4f}")

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

        for model_name, model in models.items():
            metrics, probs = evaluate_model(model, X_train, X_test, y_train, y_test)

            row = {
                "checkpoint_over": checkpoint,
                "feature_variant": variant,
                "model": model_name,
                "n_rows": len(temp),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "win_rate": y.mean(),
                **metrics
            }
            all_results.append(row)

            print(
                f"  {model_name:20s} | "
                f"AUC={metrics['auc']:.4f} | "
                f"LogLoss={metrics['log_loss']:.4f} | "
                f"Brier={metrics['brier_score']:.4f}"
            )

results_df = pd.DataFrame(all_results)
results_path = OUT_DIR / "checkpoint_model_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\n[OK] Saved full results -> {results_path}")

# Best model per checkpoint/feature set by AUC
best_auc = (
    results_df.sort_values(["checkpoint_over", "feature_variant", "auc"], ascending=[True, True, False])
              .groupby(["checkpoint_over", "feature_variant"], as_index=False)
              .head(1)
              .reset_index(drop=True)
)

best_auc_path = OUT_DIR / "checkpoint_best_by_auc.csv"
best_auc.to_csv(best_auc_path, index=False)

print(f"[OK] Saved best-per-checkpoint summary -> {best_auc_path}")

print("\n=== Full Results ===")
print(results_df.to_string(index=False))

print("\n=== Best by AUC ===")
print(best_auc.to_string(index=False))