#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Ensure local src/ is on sys.path for any future imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))


DEF_MATCH_FEATURES_NUM = [
    "first_innings_runs",
    "first_innings_wkts",
    "pp_runs",
    "pp_wkts",
]
DEF_MATCH_FEATURES_CAT = [
    "toss_decision",
    "venue",
]

INPLAY_NUM_FEATURES = [
    "runs_to_go",
    "balls_remaining",
    "wickets_in_hand",
    "current_run_rate",
    "required_run_rate",
    "target",
]


def train_match_level(match_csv: Path, models_dir: Path) -> None:
    df = pd.read_csv(match_csv)
    # Label: 1 if batting-first team won
    df = df.dropna(subset=["bat_first", "winner"])  # ensure label available
    df["bat_first_wins"] = (df["bat_first"] == df["winner"]).astype(int)
    # Optional engineered feature
    df["toss_bat_first"] = (df["toss_winner"] == df["bat_first"]).astype(int)

    X = df[DEF_MATCH_FEATURES_NUM + DEF_MATCH_FEATURES_CAT + ["toss_bat_first"]].copy()
    y = df["bat_first_wins"].values

    # Preprocess: scale numeric, one-hot categorical
    numeric_features = DEF_MATCH_FEATURES_NUM + ["toss_bat_first"]
    categorical_features = DEF_MATCH_FEATURES_CAT

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logreg_match": LogisticRegression(max_iter=500, n_jobs=None),
        "rf_match": RandomForestClassifier(n_estimators=300, random_state=42),
        "gbdt_match": GradientBoostingClassifier(random_state=42),
    }

    models_dir.mkdir(parents=True, exist_ok=True)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
        print(f"{name}: AUC={auc:.3f} ACC={acc:.3f} (n={len(y_test)})")
        joblib.dump(pipe, models_dir / f"{name}.joblib")


def train_inplay(snaps_csv: Path, models_dir: Path) -> None:
    df = pd.read_csv(snaps_csv)
    df = df.dropna(subset=["chasing_won"])  # ensure label available
    # Filter obviously invalid rows
    df = df[(df["balls_remaining"] > 0) & (df["target"] > 0)]

    X = df[INPLAY_NUM_FEATURES].copy()
    y = df["chasing_won"].astype(int).values

    pre = ColumnTransformer(transformers=[("num", StandardScaler(), INPLAY_NUM_FEATURES)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logreg = LogisticRegression(max_iter=500)
    pipe = Pipeline(steps=[("pre", pre), ("clf", logreg)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    print(f"logreg_inplay: AUC={auc:.3f} ACC={acc:.3f} (n={len(y_test)})")

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, models_dir / "logreg_inplay.joblib")



def main() -> None:
    parser = argparse.ArgumentParser(description="Train match-level and in-play models")
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Directory with match_features.csv and inplay_snapshots.csv"
    )
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory to save trained models")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or (repo_root / "data")
    models_dir = args.models_dir or (repo_root / "models")

    train_match_level(data_dir / "match_features.csv", models_dir)
    train_inplay(data_dir / "inplay_snapshots.csv", models_dir)


if __name__ == "__main__":
    main()
