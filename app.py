"""Streamlit app for Bank Marketing (UCI) - Term Deposit Subscription Prediction."""

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
import numpy as np

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Bank Marketing Classifier", layout="wide", initial_sidebar_state="expanded")

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
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏦 Bank Marketing Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'><i>Term Deposit Subscription Prediction using Machine Learning</i></p>", unsafe_allow_html=True)

st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    model_name = st.selectbox("📊 Choose a Model", list(MODEL_FILES.keys()), help="Select the classifier to use for predictions")
    
    st.markdown("---")
    
    st.markdown("### 📋 Model Information")
    model_info = {
        "Logistic Regression": "Fast, interpretable linear model",
        "Decision Tree": "Easy-to-understand tree-based classifier",
        "kNN": "Distance-based non-parametric classifier",
        "Naive Bayes": "Probabilistic classifier using Bayes' theorem",
        "Random Forest (Ensemble)": "Robust ensemble method combining multiple trees",
    }
    
    st.info(f"**{model_name}**\n\n{model_info[model_name]}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    with st.expander("Dataset & Training Details"):
        st.markdown("""
        - **Dataset**: UCI Bank Marketing (41,188 rows)
        - **Features**: 19 preprocessed features
        - **Target**: Term Deposit Subscription (yes/no)
        - **Train/Test Split**: 80/20
        - **Excluded**: Duration column (data leakage)
        """)
    
    st.markdown("---")
    st.caption("Bank Marketing · 5 Classifiers Compared")

# Load chosen model safely
try:
  model = load_model(MODEL_FILES[model_name])
except FileNotFoundError:
  st.error(
      f"Model file not found at `{MODEL_FILES[model_name]}`. Please ensure the"
      " `.joblib` file exists in your `model` directory."
  )
  st.stop()

# ---------------------------------------------------------------------------
# 1. File Upload Section
# ---------------------------------------------------------------------------
st.markdown("## 📁 Step 1: Upload Test Data")

col1, col2 = st.columns([2, 1])
with col1:
    st.write("Upload a CSV file with bank customer data (e.g., `test_data.csv`)")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader")

with col2:
    st.info("💡 Include the `y` column to see detailed metrics and confusion matrix")

# STOP HERE if user has not uploaded a file yet (prevents NoneType error)
if uploaded is None:
  st.warning("👆 Please upload a CSV file to proceed")
  st.stop()

# Safe to read once uploaded is confirmed
data = pd.read_csv(uploaded)

# Display data overview
with st.expander("📊 Data Overview", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Rows", f"{len(data):,}")
    col2.metric("📊 Columns", data.shape[1])
    col3.metric("📋 Features", data.shape[1] - (1 if "y" in data.columns else 0))
    
    st.dataframe(data.head(10), use_container_width=True, key="data_preview")

# Separate target 'y' if present
has_labels = "y" in data.columns
if has_labels:
  y_true = data["y"].map({"yes": 1, "no": 0})
  X = data.drop(columns=["y"])
else:
  X = data

# ---------------------------------------------------------------------------
# Run Model Button
# ---------------------------------------------------------------------------
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_model_button = st.button(
        "🚀 RUN MODEL & GENERATE PREDICTIONS",
        key="run_model_btn",
        use_container_width=True,
        type="primary"
    )

# Only generate predictions if button is clicked
if run_model_button or "predictions_generated" in st.session_state:
    st.session_state.predictions_generated = True
    
    # ---------------------------------------------------------------------------
    # 2. Predictions Section
    # ---------------------------------------------------------------------------
    st.markdown("## 🔮 Step 2: Generate Predictions")
    
    try:
      y_pred = model.predict(X)
      y_proba = model.predict_proba(X)[:, 1]
    except Exception as e:
      st.error(
          f"❌ Could not run predictions. The CSV columns must match the original feature set. Details: {e}"
      )
      st.stop()

    # Create predictions dataframe
    pred_display = data.copy()
    pred_display["Prediction"] = pd.Series(y_pred).map({1: "✅ Yes", 0: "❌ No"})
    pred_display["Subscription Probability"] = y_proba.round(4)

    # Display predictions with color coding
    with st.expander("📋 Detailed Predictions", expanded=True):
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            prob_threshold = st.slider("Filter by probability threshold", 0.0, 1.0, 0.0, 0.05)
        with col2:
            prediction_filter = st.selectbox("Filter by prediction", ["All", "✅ Yes", "❌ No"])
        
        # Apply filters
        filtered_pred = pred_display.copy()
        if prediction_filter != "All":
            filtered_pred = filtered_pred[filtered_pred["Prediction"] == prediction_filter]
        filtered_pred = filtered_pred[filtered_pred["Subscription Probability"] >= prob_threshold]
        
        st.dataframe(filtered_pred.head(20), use_container_width=True)
        
        # Display stats
        yes_count = (y_pred == 1).sum()
        no_count = (y_pred == 0).sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Predicted Yes", yes_count, f"{yes_count/len(y_pred)*100:.1f}%")
        col2.metric("❌ Predicted No", no_count, f"{no_count/len(y_pred)*100:.1f}%")
        col3.metric("Avg Probability", f"{y_proba.mean():.3f}")

    # Download button
    st.download_button(
        "📥 Download Predictions as CSV",
        pred_display.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{model_name.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

    # ---------------------------------------------------------------------------
    # 3. Metrics & Confusion Matrix
    # ---------------------------------------------------------------------------
    st.markdown("## 📊 Step 3: Evaluation Metrics")

    if not has_labels:
      st.warning(
          "⚠️ No `y` column found in the uploaded file. Accuracy metrics and "
          "confusion matrix cannot be computed — predictions above are still valid."
      )
    else:
      # Create tabs for different visualizations
      tab1, tab2, tab3, tab4 = st.tabs(["📈 Key Metrics", "🔥 Confusion Matrix", "📋 Classification Report", "📊 Probability Distribution"])
      
      with tab1:
          col1, col2, col3, col4, col5, col6 = st.columns(6)
          
          accuracy = accuracy_score(y_true, y_pred)
          auc = roc_auc_score(y_true, y_proba)
          precision = precision_score(y_true, y_pred, zero_division=0)
          recall = recall_score(y_true, y_pred, zero_division=0)
          f1 = f1_score(y_true, y_pred, zero_division=0)
          mcc = matthews_corrcoef(y_true, y_pred)
          
          col1.metric("🎯 Accuracy", f"{accuracy:.3f}", f"{accuracy*100:.1f}%")
          col2.metric("📊 AUC-ROC", f"{auc:.3f}")
          col3.metric("🎪 Precision", f"{precision:.3f}")
          col4.metric("🔍 Recall", f"{recall:.3f}")
          col5.metric("🎯 F1 Score", f"{f1:.3f}")
          col6.metric("🧮 MCC", f"{mcc:.3f}")
          
          st.markdown("---")
          st.markdown("#### 📌 Metric Interpretation:")
          interpretation = f"""
          - **Accuracy**: {accuracy*100:.1f}% of predictions are correct
          - **AUC-ROC**: {auc:.3f} (0.5=random, 1.0=perfect)
          - **Precision**: {precision*100:.1f}% of predicted subscriptions are correct
          - **Recall**: {recall*100:.1f}% of actual subscriptions are detected
          - **F1 Score**: Harmonic mean of precision and recall = {f1:.3f}
          """
          st.info(interpretation)
      
      with tab2:
          col1, col2 = st.columns(2)
          
          with col1:
              st.markdown("#### Confusion Matrix")
              cm = confusion_matrix(y_true, y_pred)
              
              fig, ax = plt.subplots(figsize=(6, 5))
              sns.heatmap(
                  cm,
                  annot=True,
                  fmt="d",
                  cmap="RdYlGn",
                  xticklabels=["❌ No", "✅ Yes"],
                  yticklabels=["❌ No", "✅ Yes"],
                  ax=ax,
                  cbar_kws={"label": "Count"},
                  annot_kws={"size": 14}
              )
              ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
              ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
              ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
              st.pyplot(fig)
          
          with col2:
              st.markdown("#### Matrix Interpretation")
              tn, fp, fn, tp = cm.ravel()
              st.write(f"""
              - **True Negatives (TN)**: {tn} - Correctly predicted No
              - **False Positives (FP)**: {fp} - Incorrectly predicted Yes
              - **False Negatives (FN)**: {fn} - Incorrectly predicted No
              - **True Positives (TP)**: {tp} - Correctly predicted Yes
              """)
      
      with tab3:
          st.markdown("#### Classification Report")
          report = classification_report(
              y_true,
              y_pred,
              target_names=["❌ No", "✅ Yes"],
              output_dict=True,
              zero_division=0,
          )
          
          report_df = pd.DataFrame(report).transpose().round(3)
          st.dataframe(report_df, use_container_width=True)
          
          st.markdown("""
          **Column Meanings:**
          - **Precision**: Of all positive predictions, how many were correct?
          - **Recall**: Of all actual positives, how many did we find?
          - **F1-Score**: Balanced measure of precision and recall
          - **Support**: Number of samples in each class
          """)
      
      with tab4:
          st.markdown("#### Prediction Probability Distribution")
          
          fig, axes = plt.subplots(1, 2, figsize=(12, 4))
          
          # Histogram
          axes[0].hist(y_proba, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
          axes[0].axvline(0.5, color="red", linestyle="--", linewidth=2, label="Decision Threshold")
          axes[0].set_xlabel("Probability", fontsize=12)
          axes[0].set_ylabel("Frequency", fontsize=12)
          axes[0].set_title("Distribution of Subscription Probabilities", fontsize=12, fontweight="bold")
          axes[0].legend()
          axes[0].grid(alpha=0.3)
          
          # Box plot by actual label
          prob_df = pd.DataFrame({
              "Probability": y_proba,
              "Actual": y_true.map({0: "❌ No", 1: "✅ Yes"})
          })
          
          for label in ["❌ No", "✅ Yes"]:
              data_to_plot = prob_df[prob_df["Actual"] == label]["Probability"]
              axes[1].scatter([label] * len(data_to_plot), data_to_plot, alpha=0.5, s=30)
          
          axes[1].set_ylabel("Probability", fontsize=12)
          axes[1].set_title("Probabilities by Actual Label", fontsize=12, fontweight="bold")
          axes[1].grid(alpha=0.3, axis="y")
          
          st.pyplot(fig)

    # ---------------------------------------------------------------------------
    # 4. Model Comparison Table
    # ---------------------------------------------------------------------------
    st.markdown("## 🏆 Step 4: All Models Comparison")

    comp_path = BASE_DIR / "model" / "comparison_metrics.csv"
    try:
      comp = pd.read_csv(comp_path, index_col="Model")
      
      # Display with better styling
      st.markdown("#### Performance Comparison Across All 5 Classifiers")
      
      # Highlight best values
      styled_comp = comp.style.highlight_max(color='lightgreen', axis=0).highlight_min(color='lightcoral', axis=0)
      st.dataframe(styled_comp, use_container_width=True)
      
      st.markdown("""
      #### 📊 Model Comparison Insights:
      - **Green**: Best performance in this metric
      - **Red**: Lowest performance in this metric
      - Use these metrics to understand trade-offs between models
      """)
      
    except FileNotFoundError:
      st.warning("⚠️ Comparison metrics file not found. Run `train_models.py` to generate `model/comparison_metrics.csv`.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
      <p><b>Bank Marketing UCI - Machine Learning Prediction</b></p>
      <p>5 Classifiers Evaluated</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show placeholder message when button not clicked
    st.markdown("---")
    st.info("👆 Click the **RUN MODEL & GENERATE PREDICTIONS** button above to start predictions and see detailed results!")