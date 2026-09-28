# IPL Win Probability - checkpoint model dashboard.
#
# Reads only data/checkpoint_{5,10,15,19}.csv and
# results/validate_temporal_results.csv - no new model tuning, no new
# validation. See README.md for what these files are and their limitations.

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"

CHECKPOINTS = [5, 10, 15, 19]
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
TARGET_COL = "win_flag"
TRAIN_SEASONS = {
    "2007/08", "2009", "2009/10", "2011", "2012", "2013", "2014", "2015",
    "2016", "2017", "2018", "2019", "2020/21", "2021", "2022",
}

st.set_page_config(page_title="IPL Win Probability", layout="wide")


@st.cache_data
def load_checkpoint(checkpoint: int) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"checkpoint_{checkpoint}.csv")
    df["season"] = df["season"].astype(str)
    return df


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "validate_temporal_results.csv")


@st.cache_resource
def train_logreg(checkpoint: int):
    """Logistic regression trained on seasons <=2022, same pipeline as validate_temporal.py."""
    df = load_checkpoint(checkpoint).dropna(subset=FEATURES + [TARGET_COL])
    train = df[df["season"].isin(TRAIN_SEASONS)]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(train[FEATURES], train[TARGET_COL].astype(int))
    return model, len(train)


st.title("IPL Win Probability — Checkpoint Models")
st.caption(
    "Predicts the win probability of the team batting second, at fixed checkpoints "
    "(overs 5/10/15/19) in the chase. Not a per-ball live model. "
    "See README.md for data, methodology and full limitations."
)

st.sidebar.header("Controls")
checkpoint = st.sidebar.selectbox(
    "Checkpoint (over)", CHECKPOINTS, index=1,
    help="Which checkpoint's model and data to use throughout the page.",
)
model_choice = st.sidebar.selectbox(
    "Model (highlights the results table below)",
    ["logistic_regression", "random_forest", "gradient_boosting"],
    index=0,
    help=(
        "The win-probability calculator always uses Logistic Regression "
        "(trained on seasons <=2022) regardless of this choice — this only "
        "highlights a row in the validation results table."
    ),
)

logreg, n_train = train_logreg(checkpoint)

# ---------------------------------------------------------------------------
# Panel 1: win probability calculator
# ---------------------------------------------------------------------------
st.header("1. Win probability calculator")
st.caption(
    f"Logistic regression trained on {n_train} matches from seasons up to 2022, "
    f"at checkpoint over {checkpoint}. Uses the checkpoint's exact feature "
    "definitions (assumes a standard 20-over innings)."
)

col1, col2 = st.columns(2)
with col1:
    target = st.number_input("Target (runs to chase)", min_value=1, max_value=300, value=165)
    score = st.number_input(
        "Current score", min_value=0, max_value=int(target), value=min(80, max(target - 1, 0))
    )
    wickets = st.slider("Wickets lost", 0, 9, 3)
with col2:
    overs = st.number_input(
        f"Overs completed (defaults to checkpoint {checkpoint})",
        min_value=0.1, max_value=19.5, value=float(checkpoint), step=0.1,
    )
    st.caption(
        f"The model was trained on rows captured at exactly over {checkpoint}. "
        "Predictions are most trustworthy when this is close to that value."
    )

if score >= target:
    st.success("Chase already completed at this input (score ≥ target) — no prediction to make.")
else:
    runs_to_go = max(target - score, 0)
    wickets_in_hand = 10 - wickets
    balls_bowled = overs * 6.0
    balls_remaining = max(120.0 - balls_bowled, 0.0)
    overs_remaining = balls_remaining / 6.0
    current_run_rate = score / (balls_bowled / 6.0) if balls_bowled > 0 else 0.0
    required_run_rate = runs_to_go / overs_remaining if overs_remaining > 0 else 0.0

    row = pd.DataFrame([{
        "current_score": score,
        "wickets_lost": wickets,
        "wickets_in_hand": wickets_in_hand,
        "runs_to_go": runs_to_go,
        "balls_remaining": balls_remaining,
        "overs_remaining": overs_remaining,
        "current_run_rate": current_run_rate,
        "required_run_rate": required_run_rate,
    }])

    prob = logreg.predict_proba(row[FEATURES])[0, 1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Win probability (batting second)", f"{prob:.1%}")
    m2.metric("Required run rate", f"{required_run_rate:.2f}")
    m3.metric("Wickets in hand", f"{wickets_in_hand}")

    if abs(overs - checkpoint) > 1:
        st.warning(
            f"'Overs completed' ({overs:.1f}) is more than 1 over away from the "
            f"selected checkpoint ({checkpoint}) — the model never saw training "
            "rows this far from its checkpoint, so treat this prediction with extra caution."
        )

# ---------------------------------------------------------------------------
# Panel 2: temporal vs random validation results
# ---------------------------------------------------------------------------
st.header("2. Validation results: temporal vs random split")
st.caption(
    "From results/validate_temporal_results.csv — bootstrap 95% CIs, 1,000 resamples. "
    f"Row for the selected model ({model_choice}) is highlighted."
)

results = load_results()
sub = results[results["checkpoint_over"] == checkpoint].copy()
sub["AUC [95% CI]"] = sub.apply(
    lambda r: f"{r['auc']:.3f} [{r['auc_ci_lo']:.3f}, {r['auc_ci_hi']:.3f}]", axis=1
)
sub["Log loss"] = sub["log_loss"].round(3)
sub["Brier"] = sub["brier_score"].round(3)
display_cols = ["split", "model", "n_train", "n_test", "AUC [95% CI]", "Log loss", "Brier"]
sub_display = sub[display_cols].sort_values(["split", "model"]).reset_index(drop=True)


def highlight_model(row):
    color = "background-color: #fff3b0" if row["model"] == model_choice else ""
    return [color] * len(row)


st.dataframe(sub_display.style.apply(highlight_model, axis=1), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Panel 3: calibration plot
# ---------------------------------------------------------------------------
st.header("3. Model reliability — calibration (temporal test set)")
calib_path = FIGS / f"calibration_temporal_checkpoint_{checkpoint}.png"
if calib_path.exists():
    st.image(str(calib_path), caption=f"Checkpoint over {checkpoint} — all three models, temporal test set")
else:
    st.warning(f"Missing figure: {calib_path.name}. Run validate_temporal.py to generate it.")

# ---------------------------------------------------------------------------
# Panel 4: limitations
# ---------------------------------------------------------------------------
st.header("4. Limitations")
st.info(
    "- **Small test sets**: 110-144 matches per checkpoint under the temporal split — "
    "bootstrap 95% CIs on AUC span roughly ±0.03 to ±0.08 (see the table above).\n"
    "- **Base-rate drift**: the chasing team's win rate is lower in the 2023-2024 test "
    "seasons than in the 2008-2022 training window at every checkpoint, so logistic "
    "regression under-predicts slightly early in the chase and is well calibrated by "
    "the death overs.\n"
    "- No player-level features, no hyperparameter tuning, and several baseline "
    "features are near-duplicates of each other at a fixed checkpoint "
    "(e.g. `wickets_in_hand` = 10 - `wickets_lost` exactly). See README.md for detail."
)
