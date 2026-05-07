from pathlib import Path
import warnings
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

warnings.filterwarnings("ignore")

# =========================================================
# Paths
# =========================================================
ROOT = Path.home() / "Documents" / "DSP"
DATA = ROOT / "data"
OUT_DIR = ROOT / "results" / "checkpoint_models_focused_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILES = {
    5: DATA / "checkpoint_5_fe.csv",
    10: DATA / "checkpoint_10_fe.csv",
    15: DATA / "checkpoint_15_fe.csv",
    19: DATA / "checkpoint_19_fe.csv",
}

TARGET = "win_flag"

# =========================================================
# Feature sets
# =========================================================
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
# Model builder
# =========================================================
def build_models():
    models = {}

    models["logistic_regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])

    models["random_forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    models["gradient_boosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    return models

# =========================================================
# Evaluation helper
# =========================================================
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    return {
        "auc": roc_auc_score(y_test, probs),
        "log_loss": log_loss(y_test, probs, labels=[0, 1]),
        "brier_score": brier_score_loss(y_test, probs),
        "test_rows": len(y_test),
        "positive_rate_test": float(y_test.mean()),
    }

# =========================================================
# Main
# =========================================================
print("[Info] Script: S9 Focused V3 - Checkpoint model comparison")

all_results = []
models = build_models()

for checkpoint, file_path in CHECKPOINT_FILES.items():
    print(f"\n{'='*70}")
    print(f"[Checkpoint {checkpoint}] Loading {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)
    print(f"[Data] Rows: {len(df)}, Unique matches: {df['match_id'].nunique()}")

    for variant, features in {
        "baseline": BASELINE_FEATURES,
        "focused": FOCUSED_FEATURES
    }.items():

        temp = df[features + [TARGET]].copy()
        temp = temp.dropna(subset=[TARGET]).reset_index(drop=True)

        X = temp[features]
        y = temp[TARGET].astype(int)

        print(f"\n[Checkpoint {checkpoint} | {variant}]")
        print(f"Rows available: {len(temp)}")
        print(f"Win rate: {y.mean():.4f}")
        print(f"Features used: {features}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )

        print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

        for model_name, model in models.items():
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

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
results_path = OUT_DIR / "checkpoint_model_results_focused_v3.csv"
results_df.to_csv(results_path, index=False)

print(f"\n[OK] Saved full results -> {results_path}")

best_auc = (
    results_df.sort_values(
        ["checkpoint_over", "feature_variant", "auc", "log_loss", "brier_score"],
        ascending=[True, True, False, True, True]
    )
    .groupby(["checkpoint_over", "feature_variant"], as_index=False)
    .head(1)
    .reset_index(drop=True)
)

best_auc_path = OUT_DIR / "checkpoint_best_by_auc_focused_v3.csv"
best_auc.to_csv(best_auc_path, index=False)

print(f"[OK] Saved best summary -> {best_auc_path}")

print("\n=== Full Results ===")
print(results_df.to_string(index=False))

print("\n=== Best by AUC ===")
print(best_auc.to_string(index=False))