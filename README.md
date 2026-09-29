# IPL Win Probability — Checkpoint Models

Predicts the win probability of the team batting second in an IPL T20 match,
at four fixed points in the chase (end of overs 5, 10, 15 and 19), from
match-state features available at that point. Built on Cricsheet ball-by-ball
data.

This is not a per-ball live model. It predicts at four discrete checkpoints
per match, not continuously after every delivery — see [Limitations](#limitations).

**Run order:** `build_checkpoint_datasets.py` → `validate_temporal.py` → `app.py`.

---

## Data

Source: [Cricsheet](https://cricsheet.org/) IPL ball-by-ball data.

- `data/matches_clean.csv` — 1,169 matches, seasons 2007/08 through 2025
  (2008-04-18 to 2025-06-03).
- `data/inplay_snapshots.csv` — one row per legal delivery of the second
  innings (the chase) for 1,164 of those matches: 133,903 rows. Built from
  raw ball-by-ball data outside this repo. This file replaces an earlier,
  buggy version — see [Archive](#archive).
- `data/checkpoint_{5,10,15,19}.csv` — one row per match at each checkpoint
  over (the last ball recorded in that over), built from the two files above
  by `build_checkpoint_datasets.py`, with `season` attached for temporal
  splitting. Row counts fall at later checkpoints because not every chase
  reaches that over — the match may already be won or lost:

  | Checkpoint over | Matches |
  |---|---|
  | 5 | 1,161 |
  | 10 | 1,145 |
  | 15 | 1,088 |
  | 19 | 880 |

The raw Cricsheet JSON → `matches_clean.csv` / `inplay_snapshots.csv`
pipeline is **not** reproducible from this repo as committed — those
ingestion scripts (`S1.py`, `S3.py`, `s3Mapping.py`, `s4Venue.py`,
`s5BuildMatch & ball-by-ball.py`, `s8v2.py`) hardcode a path outside version
control and the ~1,150 raw JSON files aren't checked in. Everything from
`data/matches_clean.csv` and `data/inplay_snapshots.csv` onward — checkpoint
construction, modeling, validation — is reproducible with the scripts in
this repo; see [How to reproduce](#how-to-reproduce).

---

## Features

Eight match-state features, available at each checkpoint:

`current_score`, `wickets_lost`, `wickets_in_hand`, `runs_to_go`,
`balls_remaining`, `overs_remaining`, `current_run_rate`, `required_run_rate`

**These are heavily correlated with each other, not eight independent
signals.** At a fixed checkpoint (checked at over 15):

- `wickets_in_hand` = `10 − wickets_lost` exactly (r = −1.00)
- `overs_remaining` = `balls_remaining / 6` exactly (r = 1.00)
- `current_score` and `current_run_rate` (r = 1.00), `runs_to_go` and
  `required_run_rate` (r = 1.00) — both pairs are near-identical because the
  number of balls bowled is fixed at a given checkpoint, so run rate is just
  score rescaled by a near-constant.

In practice the model has closer to four independent inputs per checkpoint.
This is fine for a tree model's predictions but means logistic regression
coefficients should not be read as independent feature importances — see
[Limitations](#limitations). A set of interaction terms
(`rrr_x_wickets_in_hand`, `runs_to_go_x_wickets_in_hand`, `rrr_minus_crr`,
`runs_per_ball_needed`, `target_completed_pct`) was tested and made no
material difference to AUC (≤0.003 at every checkpoint), so the model below
uses the plain eight-feature set.

---

## Models

Logistic regression, random forest, and gradient boosting (scikit-learn),
trained separately per checkpoint. No hyperparameter tuning was performed —
all three use fixed, untuned settings.

---

## Validation

Two validation strategies, both computed by `validate_temporal.py` on the
same checkpoint data:

- **Temporal split**: train on seasons up to 2022, test on 2023-2024.
  Season 2025 (74 matches) exists in the data but is excluded from both
  sets. Confirmed zero shared `match_id`s between train and test at every
  checkpoint.
- **Random split**: stratified 75/25 split (`random_state=42`), kept
  alongside the temporal split for comparison — this is the split an
  earlier version of this pipeline used exclusively (see
  [Archive](#archive)).

Metrics include a 1,000-sample bootstrap 95% CI. Test sets are small
(110–144 matches per checkpoint), so treat CI width as a genuine
statement of uncertainty, not a formality.

| Checkpoint | Split | Model | n train | n test | AUC [95% CI] | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| 5 | temporal | Logistic Regression | 946 | 144 | 0.804 [0.730, 0.871] | 0.565 | 0.189 |
| 5 | temporal | Random Forest | 946 | 144 | 0.752 [0.669, 0.825] | 0.666 | 0.215 |
| 5 | temporal | Gradient Boosting | 946 | 144 | 0.764 [0.685, 0.835] | 0.684 | 0.212 |
| 5 | random | Logistic Regression | 870 | 291 | 0.803 [0.749, 0.849] | 0.543 | 0.183 |
| 5 | random | Random Forest | 870 | 291 | 0.784 [0.726, 0.835] | 0.581 | 0.194 |
| 5 | random | Gradient Boosting | 870 | 291 | 0.784 [0.727, 0.834] | 0.599 | 0.197 |
| 10 | temporal | Logistic Regression | 931 | 143 | 0.859 [0.794, 0.910] | 0.474 | 0.158 |
| 10 | temporal | Random Forest | 931 | 143 | 0.839 [0.765, 0.897] | 0.515 | 0.164 |
| 10 | temporal | Gradient Boosting | 931 | 143 | 0.849 [0.783, 0.902] | 0.531 | 0.164 |
| 10 | random | Logistic Regression | 858 | 287 | 0.854 [0.812, 0.892] | 0.474 | 0.159 |
| 10 | random | Random Forest | 858 | 287 | 0.821 [0.773, 0.861] | 0.539 | 0.177 |
| 10 | random | Gradient Boosting | 858 | 287 | 0.828 [0.778, 0.868] | 0.532 | 0.177 |
| 15 | temporal | Logistic Regression | 885 | 136 | 0.927 [0.878, 0.965] | 0.342 | 0.109 |
| 15 | temporal | Random Forest | 885 | 136 | 0.908 [0.853, 0.952] | 0.388 | 0.123 |
| 15 | temporal | Gradient Boosting | 885 | 136 | 0.903 [0.844, 0.952] | 0.422 | 0.126 |
| 15 | random | Logistic Regression | 816 | 272 | 0.941 [0.913, 0.964] | 0.312 | 0.099 |
| 15 | random | Random Forest | 816 | 272 | 0.921 [0.889, 0.949] | 0.357 | 0.115 |
| 15 | random | Gradient Boosting | 816 | 272 | 0.913 [0.879, 0.944] | 0.390 | 0.125 |
| 19 | temporal | Logistic Regression | 720 | 110 | 0.974 [0.944, 0.994] | 0.204 | 0.060 |
| 19 | temporal | Random Forest | 720 | 110 | 0.971 [0.933, 0.996] | 0.208 | 0.057 |
| 19 | temporal | Gradient Boosting | 720 | 110 | 0.978 [0.950, 0.996] | 0.197 | 0.056 |
| 19 | random | Logistic Regression | 660 | 220 | 0.970 [0.950, 0.985] | 0.236 | 0.077 |
| 19 | random | Random Forest | 660 | 220 | 0.964 [0.943, 0.982] | 0.267 | 0.080 |
| 19 | random | Gradient Boosting | 660 | 220 | 0.972 [0.955, 0.987] | 0.264 | 0.071 |

Full table: `results/validate_temporal_results.csv`.

Temporal AUCs track the random-split AUCs closely at every checkpoint
(within ~0.01–0.03, well inside the CIs above), so the model's discrimination
is not an artifact of the random split holding out rows from matches the
model effectively already "knows." The largest gap is at checkpoint 5,
where the temporal test set is smallest and early-innings signal is
noisiest.

**Calibration drift under the temporal split:** the chasing team's win rate
is lower in 2023-2024 (test) than in the 2008-2022 training window at every
checkpoint (e.g. 0.479 vs 0.544 at over 5). Logistic regression's mean
predicted probability sits between the two, so it under-predicts slightly
early in the chase (mean predicted 0.418 vs actual 0.479 at over 5, a 0.061
drift) and is well calibrated by the death overs (0.007 drift at over 19) —
match-state features dominate the base rate by then. Per-checkpoint
calibration plots: `figures/calibration_temporal_checkpoint_{5,10,15,19}.png`.

---

## Limitations

- **Small test sets.** 110–144 matches per checkpoint under the temporal
  split; bootstrap CIs on AUC span roughly ±0.03 to ±0.08. Don't read small
  differences between models or between splits as meaningful.
- **No player-level features.** No batter/bowler form, matchup history, or
  team strength — only match-state (score, wickets, run rates). Two
  identical match situations with different players at the crease get the
  same prediction.
- **Correlated features** (see [Features](#features)): logistic regression
  coefficients in this pipeline should not be interpreted as independent
  feature importances.
- **Four checkpoints, not continuous.** No model exists for overs between
  checkpoints; a live, ball-by-ball graphic would need either more
  checkpoints or an interpolation scheme, neither implemented here.
- **Base-rate drift across seasons** (see calibration note above) — a model
  retrained periodically would likely need this accounted for explicitly
  rather than relying on match-state features alone to compensate.
- **Venue and home-team fields exist in the data but aren't used as model
  features.**
- **No hyperparameter tuning.**
- Season 2025 (74 matches) is present in the data but wasn't used in this
  validation.

---

## Archive

Scripts and figures that were superseded, broken, or fake have been moved to
`archive/` (with git history preserved via `git mv`) rather than deleted or
silently left in place. See [`archive/README.md`](archive/README.md) for
exactly what's there and why each file was archived — including the
`target_runs` bug that the old `data/inplay_snapshots.csv` had, and the
hardcoded-fake `win_probability_curve.png`.

Not everything broken has been archived: `s9_models_Checkpointsv3_focused.py`
and `s9_models_Checkpointsv2.py` still exist at the repo root but are not
part of the active pipeline — the former doesn't currently run (missing
input files), and both hardcode a path to the author's local machine, so
neither works from a fresh clone. `build_checkpoint_datasets.py` and
`validate_temporal.py` are the reproducible replacements.

---

## Repository structure

```
IPL-WInProbability_Project/
├── data/
│   ├── matches_clean.csv           # match-level data, 1,169 matches
│   ├── inplay_snapshots.csv        # second-innings ball-by-ball, corrected
│   └── checkpoint_{5,10,15,19}.csv # one row per match per checkpoint, with season
├── figures/
│   └── calibration_temporal_checkpoint_{5,10,15,19}.png
├── results/
│   └── validate_temporal_results.csv
├── build_checkpoint_datasets.py    # data/inplay_snapshots.csv + matches_clean.csv -> checkpoint CSVs
├── validate_temporal.py            # temporal + random split validation, this README's results table
├── app.py                          # Streamlit dashboard
├── requirements.txt
├── archive/                        # superseded, broken, or fake — see archive/README.md
├── S1.py, S3.py, s3Mapping.py, s4Venue.py, s4venuedetect.py,
│   S5B_FB.py, s5BuildMatch & ball-by-ball.py, s6_logR.py,
│   s8.py, s8v2.py, s8checkpoints.py, s8c_Featureengineeringcheckpoints.py,
│   s9_models_Checkpointsv2.py, s9_models_Checkpointsv3_focused.py,
│   s10_final_report_outputs.py, InventS2.py
│                                    # original exploratory pipeline; several
│                                    # hardcode local paths or read stale data
│                                    # — see Archive
└── scripts/                        # early scaffold scripts, superseded by the above
```

---

## How to reproduce

Requires Python 3.12+ with `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

```bash
git clone https://github.com/chandr-csgp/IPL-WInProbability_Project.git
cd IPL-WInProbability_Project
pip install pandas numpy scikit-learn matplotlib

python build_checkpoint_datasets.py   # rebuilds data/checkpoint_{5,10,15,19}.csv
python validate_temporal.py           # reproduces the results table and figures above
```

This regenerates `data/checkpoint_*.csv`, `results/validate_temporal_results.csv`,
and `figures/calibration_temporal_checkpoint_*.png` entirely from the two
data files committed in this repo — no external paths involved.

---

## Author

**Chandra Sekar Putta**
MDS — University of Adelaide (2024–2026)
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
