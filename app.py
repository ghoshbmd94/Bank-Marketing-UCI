"""Streamlit app for BITS Pilani ML Assignment 2 Bank Marketing (UCI) - Term

Deposit Subscription Prediction.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")

# Resolves the absolute path directory where app.py lives
BASE_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "Logistic Regression": BASE_DIR / "model" / "logistic_regression.joblib",
    "Decision Tree": BASE_DIR / "model" / "decision_tree.joblib",
    "kNN": BASE_DIR / "model" / "knn.joblib",
    "Naive Bayes": BASE_DIR / "model" / "naive_bayes.joblib",
    "Random Forest (Ensemble)": BASE_DIR / "model" / "random_forest.joblib",
}


@st.cache_resource
def load_model(path):
  return joblib.load(path)


# ---------------------------------------------------------------------------
# Header & Sidebar
# ---------------------------------------------------------------------------
st.title("🏦 Bank Marketing — Term Deposit Subscription Predictor")
st.caption(
    "BITS Pilani M.Tech (AIML/DSE) — Machine Learning, Assignment 2 · "
    "UCI Bank Marketing dataset · 5 classifiers compared"
)

st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))

# Load chosen model safely
try:
  model = load_model(MODEL_FILES[model_name])
except FileNotFoundError:
  st.error(
      f"Model file not found at `{MODEL_FILES[model_name]}`. Please ensure the"
      " `.joblib` file exists in your `model` directory."
  )
  st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About**: All 5 models were trained on the same 80/20 split of the "
    "UCI Bank Marketing dataset (41,188 rows, 19 features). The leaky "
    "`duration` column was dropped before training."
)

# ---------------------------------------------------------------------------
# 1. File Upload
# ---------------------------------------------------------------------------
st.subheader("1. Upload test data (CSV)")
st.write(
    "Upload `test_data.csv` from the repo (or any CSV with the same raw "
    "columns). Including the true `y` column lets the app also show "
    "accuracy metrics and a confusion matrix."
)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

# STOP HERE if user has not uploaded a file yet (prevents NoneType error)
if uploaded is None:
  st.info("👆 Please upload a CSV file above to generate predictions.")
  st.stop()

# Safe to read once uploaded is confirmed
data = pd.read_csv(uploaded)
st.write(f"Loaded **{len(data)}** rows, **{data.shape[1]}** columns.")
st.dataframe(data.head(10), use_container_width=True)

# Separate target 'y' if present
has_labels = "y" in data.columns
if has_labels:
  y_true = data["y"].map({"yes": 1, "no": 0})
  X = data.drop(columns=["y"])
else:
  X = data

# ---------------------------------------------------------------------------
# 2. Predictions
# ---------------------------------------------------------------------------
st.subheader("2. Predictions")
try:
  y_pred = model.predict(X)
  y_proba = model.predict_proba(X)[:, 1]
except Exception as e:
  st.error(
      f"Could not run predictions — the uploaded CSV's columns must match "
      f"the original raw feature columns. Details: {e}"
  )
  st.stop()

pred_display = data.copy()
pred_display["prediction"] = pd.Series(y_pred).map({1: "yes", 0: "no"})
pred_display["subscription_probability"] = y_proba.round(3)
st.dataframe(pred_display.head(20), use_container_width=True)

st.download_button(
    "Download predictions as CSV",
    pred_display.to_csv(index=False).encode("utf-8"),
    file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# 3. Metrics & Confusion Matrix
# ---------------------------------------------------------------------------
st.subheader("3. Evaluation metrics")

if not has_labels:
  st.warning(
      "No `y` column found in the uploaded file, so accuracy metrics and "
      "the confusion matrix can't be computed — predictions above are "
      "still valid."
  )
else:
  col1, col2, col3, col4, col5, col6 = st.columns(6)
  col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
  col2.metric("AUC", f"{roc_auc_score(y_true, y_proba):.3f}")
  col3.metric(
      "Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}"
  )
  col4.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
  col5.metric("F1", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
  col6.metric("MCC", f"{matthews_corrcoef(y_true, y_pred):.3f}")

  left, right = st.columns(2)

  with left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["no", "yes"],
        yticklabels=["no", "yes"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

  with right:
    st.markdown("**Classification Report**")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["no", "yes"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(
        pd.DataFrame(report).transpose().round(3), use_container_width=True
    )

# ---------------------------------------------------------------------------
# 4. Model Comparison Table
# ---------------------------------------------------------------------------
st.subheader("4. All-model comparison (full held-out test set)")
comp_path = BASE_DIR / "model" / "comparison_metrics.csv"
try:
  comp = pd.read_csv(comp_path, index_col="Model")
  st.dataframe(comp, use_container_width=True)
except FileNotFoundError:
  st.info("Run `train_models.py` to generate `model/comparison_metrics.csv`.")