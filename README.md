# 🏏 IPL Win Probability Prediction

**Dynamic, ball-by-ball win probability modelling for IPL T20 cricket**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

---

## 📌 Overview

This project builds a **real-time win probability model** for IPL T20 matches using ball-by-ball data. Unlike static pre-match predictions, this model updates after every delivery — just like a live broadcast graphic.

The model treats each ball as an independent inference point and uses **in-play context features** to estimate the batting team's probability of winning at any moment in the match.

---

## 📊 Dataset

- **Source:** [Cricsheet](https://cricsheet.org/) open dataset (YAML/CSV ball-by-ball data)
- **Matches:** 1,146 IPL matches (2008–2023)
- **Snapshots:** 133,000+ individual ball-by-ball records
- **Split:** Temporal — 2008–2020 train / 2021–2023 test (prevents data leakage)

---

## ⚙️ Feature Engineering

Real-time features engineered per delivery:

| Feature | Description |
|---|---|
| Required Run Rate | Runs needed per over to win |
| Current Run Rate | Runs scored per over so far |
| Wickets in Hand | Wickets remaining for batting team |
| Momentum Window | Run rate over last 3 overs |
| Dot Ball Rate | % of dot balls in last 5 overs |
| Balls Remaining | Deliveries left in innings |
| Target | Total runs set by first innings |
| Phase | Powerplay / Middle / Death overs |

---

## 🤖 Models Compared

| Model | AUC | Brier Score |
|---|---|---|
| Logistic Regression | 0.61 | 0.24 |
| Random Forest | ~0.59 | ~0.26 |
| Gradient Boosting | ~0.60 | ~0.25 |

**Selected model:** Logistic Regression (best calibration for probability outputs)

---

## 🗂️ Project Structure

```
IPL-WInProbability_Project/
├── data/               # Raw and processed match data
├── figures/            # Output plots and calibration curves
├── scripts/            # Pipeline scripts (see below)
├── S1.py               # Data ingestion from Cricsheet
├── S3.py               # Match filtering and cleaning
├── s3Mapping.py        # Team/player name normalisation
├── s4Venue.py          # Venue feature extraction
├── s5BuildMatch...py   # Ball-by-ball snapshot builder
├── s6_logR.py          # Logistic Regression model
├── s8.py / s9models.py # Model comparison and evaluation
├── s10_final_report_outputs.py  # Final charts and metrics
└── InventS2.py         # Data inventory/audit
```

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/chandr-csgp/IPL-WInProbability_Project.git
cd IPL-WInProbability_Project

# 2. Install dependencies
pip install pandas scikit-learn matplotlib seaborn

# 3. Run the pipeline in order
python S1.py          # ingest data
python S3.py          # clean and filter
python s5BuildMatch\ \&\ ball-by-ball.py   # build snapshots
python s6_logR.py     # train model
python s10_final_report_outputs.py         # generate report
```

---

## 📈 Key Results

- **AUC 0.61** — model discriminates winners from losers better than chance in live match conditions
- **Brier Score 0.24** — well-calibrated probabilities (0 = perfect, 0.25 = random)
- **Temporal split** ensures no future data leaks into training — a production-standard ML practice

---

## 🔧 Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

---

## 👤 Author

**Chandra Sekar Putta**  
MDS — University of Adelaide (2024–2026)  
[LinkedIn](https://linkedin.com/in/chandra-sekar-p-b4b033402) · [GitHub](https://github.com/chandr-csgp)
