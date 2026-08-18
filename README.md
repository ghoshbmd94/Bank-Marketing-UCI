# Bank Marketing — Term Deposit Subscription Prediction

BITS Pilani, WILP — M.Tech (AIML/DSE) — Machine Learning — Assignment 2

## a. Problem Statement

A Portuguese bank ran a series of direct phone-call marketing campaigns to
sell term deposits. This project builds and compares classification models
that predict, from a client's profile and campaign-contact attributes,
whether that client will subscribe to a term deposit (`yes` / `no`). This is
a **binary classification** problem framed to support a call-center's
targeting decisions — i.e., which clients to prioritize for a call.

## b. Dataset Description

- **Source**: [UCI Machine Learning Repository — Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing) (`bank-additional-full.csv`), originally from Moro, S., Rita, P., & Cortez, P. (2014).
- **Instances**: 41,188
- **Raw input features**: 20 (client demographics — age, job, marital status,
  education; financial attributes — default, housing loan, personal loan;
  campaign-contact attributes — contact type, month, day of week, number of
  contacts, days since previous contact, previous outcome; and 5
  macroeconomic indicators — employment variation rate, consumer price
  index, consumer confidence index, euribor 3-month rate, number of
  employees)
- **Target**: `y` — has the client subscribed to a term deposit (`yes`/`no`)
- **Class balance**: highly imbalanced — only ~11.3% of clients subscribed
- **Missing values**: none
- **Preprocessing**: the `duration` column (last call duration in seconds)
  was **dropped** before modeling. The dataset documentation itself notes
  that `duration` is unknown before a call is placed and is highly
  correlated with the outcome (a duration of 0 always means `no`) — keeping
  it would leak the target and produce an unrealistic, undeployable model.
  Numeric features were standardized; categorical features were one-hot
  encoded. Both are wrapped into each model's saved pipeline so raw CSVs
  can be scored directly.

## c. GitHub Repository Link

> _Add your repository link here after pushing, e.g._
> `https://github.com/<your-username>/bank-marketing-classification`

## d. Models Used

All 5 models were trained on the same 80/20 stratified train/test split
(same random seed) of the dataset described above.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9009 | 0.8008 | 0.6905 | 0.2188 | 0.3322 | 0.3516 |
| Decision Tree | 0.9007 | 0.7893 | 0.6440 | 0.2651 | 0.3756 | 0.3706 |
| kNN | 0.9003 | 0.7756 | 0.6458 | 0.2554 | 0.3660 | 0.3641 |
| Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| Random Forest (Ensemble) | 0.9014 | 0.8115 | 0.7014 | 0.2177 | 0.3322 | 0.3544 |

_(Regenerate with `python train_models.py`; exact values may shift slightly with different environments/library versions.)_

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-calibrated baseline (AUC 0.80). Highest precision (0.69) among the linear/simple models but the lowest recall — it only flags clients it's fairly confident about, missing many actual subscribers. Fast to train and easy to interpret via coefficients. |
| Decision Tree | Comparable accuracy to Logistic Regression but a noticeably lower AUC, suggesting weaker probability ranking despite similar hard-label accuracy. Best MCC of all 5 models, meaning its predictions correlate most strongly with the true labels once class imbalance is accounted for. Capped at `max_depth=8` to limit overfitting. |
| kNN | Middling performer across the board — no clear strength. Sensitive to the curse of dimensionality after one-hot encoding (many sparse categorical columns), and prediction is slower at inference time since it must compare against the full training set. |
| Naive Bayes | Clearly the outlier: much lower accuracy (0.80) but by far the highest recall (0.63) and F1 (0.42). Its independence assumption is violated by correlated macroeconomic features, hurting precision — but for this problem, where **missing a genuine subscriber is more costly than an extra phone call**, higher recall can matter more than raw accuracy. |
| Random Forest (Ensemble) | Best Accuracy (0.901) and best AUC (0.812) of all 5 models, with the highest precision (0.70) too — when it predicts "yes" it's usually right. Its MCC (0.354) trails the Decision Tree's (0.371) slightly, since it's also conservative about recall. As an ensemble, it's the most robust to overfitting, at the cost of being the least interpretable and heaviest model to ship. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble).** It gives the best AUC and Accuracy, and the best precision by a clear margin, making it the most trustworthy model when the bank acts on a "yes" prediction (e.g., prioritizing a call list). The **Decision Tree** is the closest runner-up and actually edges it out on MCC (0.371 vs 0.354) — a reasonable alternative if a single interpretable model is preferred. If the bank's priority shifted toward *maximizing subscribers caught* even at the cost of more wasted calls, **Naive Bayes** would be the more defensible choice instead, given its much higher recall (0.635). |

## Repository Structure

```
project-folder/
├── app.py                       # Streamlit app
├── train_models.py              # Orchestrator: executes all 5 notebooks below, builds comparison table
├── make_notebooks.py            # Generates the 5 notebooks (only needed if you want to regenerate them)
├── requirements.txt
├── README.md
├── bank-additional-full.csv     # Full raw dataset (source data)
├── test_data.csv                # Held-out test sample used in the Streamlit app
└── model/
    ├── data_utils.py                    # Shared data loading / split helper (used by train_models.py)
    ├── logistic_regression.ipynb        # Self-contained: loads data, trains, evaluates, saves
    ├── decision_tree.ipynb              # Self-contained: loads data, trains, evaluates, saves
    ├── knn.ipynb                        # Self-contained: loads data, trains, evaluates, saves
    ├── naive_bayes.ipynb                # Self-contained: loads data, trains, evaluates, saves
    ├── random_forest.ipynb              # Self-contained: loads data, trains, evaluates, saves
    ├── logistic_regression.joblib       # Saved fitted pipeline (produced by the notebook above)
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── *_metrics.json                   # Per-model metrics, written by each notebook
    ├── comparison_metrics.csv
    └── feature_schema.json
```

Each `.ipynb` in `model/` is fully self-contained — open it directly in
Jupyter/Colab and "Run All" to reproduce that model from scratch, no
external helper module required. `train_models.py` at the project root is
a convenience orchestrator that executes all five notebooks headlessly
(via `jupyter nbconvert --execute`) and assembles the comparison table.

## Running Locally

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to regenerate models via train_models.py

# Regenerate all 5 models + test_data.csv + comparison table
python train_models.py

streamlit run app.py
```

Alternatively, open any notebook in `model/` directly in Jupyter/Colab and
run it top to bottom to retrain just that one model.

## Streamlit App Features

- **Dataset upload (CSV)**: upload `test_data.csv` (or any CSV with the same
  raw columns) to score it against the selected model.
- **Model selection dropdown**: switch between all 5 trained models.
- **Evaluation metrics display**: Accuracy, AUC, Precision, Recall, F1, MCC
  shown live for the uploaded data (when it includes the true `y` label).
- **Confusion matrix & classification report**: rendered for the uploaded
  data, plus a full comparison table of all 5 models on the original
  held-out test set.

## Live Streamlit App Link

> _Add your deployed Streamlit Community Cloud link here._
