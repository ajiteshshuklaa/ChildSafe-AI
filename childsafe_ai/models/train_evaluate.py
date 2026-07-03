"""
ChildSafe AI – Model Training & Evaluation
==========================================
Trains 6 model combinations:
    - Logistic Regression  + TF-IDF
    - Logistic Regression  + CountVectorizer
    - Random Forest        + TF-IDF
    - Random Forest        + CountVectorizer
    - Decision Tree        + TF-IDF
    - Decision Tree        + CountVectorizer

Outputs saved to outputs/:
    - model_comparison.csv
    - confusion_matrices.png
    - feature_importance.png
    - accuracy_comparison.png

Trained models saved to models/saved/*.pkl
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")

# ── Add project root to path so we can import dataset_generator ───────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset_generator import load_dataset, get_train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LABEL_NAMES  = ["Safe", "Suspicious", "Dangerous"]
OUTPUT_DIR   = "outputs"
MODEL_DIR    = "models/saved"
DATASET_PATH = "data/cyberbullying_tweets.csv"
COLORS       = ["#2196F3", "#FF9800", "#F44336"]   # blue / orange / red

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_pipelines() -> dict:
    """Return dict of {model_name: sklearn Pipeline}."""
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    count = CountVectorizer(max_features=10000, ngram_range=(1, 2))

    return {
        "LR_TF-IDF": Pipeline([
            ("vec", TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "LR_Count": Pipeline([
            ("vec", CountVectorizer(max_features=10000, ngram_range=(1,2))),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "RF_TF-IDF": Pipeline([
            ("vec", TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        "RF_Count": Pipeline([
            ("vec", CountVectorizer(max_features=10000, ngram_range=(1,2))),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        "DT_TF-IDF": Pipeline([
            ("vec", TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)),
            ("clf", DecisionTreeClassifier(max_depth=20, random_state=42)),
        ]),
        "DT_Count": Pipeline([
            ("vec", CountVectorizer(max_features=10000, ngram_range=(1,2))),
            ("clf", DecisionTreeClassifier(max_depth=20, random_state=42)),
        ]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate(pipelines, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train all pipelines, save models, return comparison DataFrame."""
    results = []

    for name, pipe in pipelines.items():
        print(f"\n  Training : {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)  * 100
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)    * 100
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)        * 100

        print(f"    Accuracy  : {acc:.2f}%")
        print(f"    Precision : {prec:.2f}%")
        print(f"    Recall    : {rec:.2f}%")
        print(f"    F1 Score  : {f1:.2f}%")
        print(f"\n    Classification Report:")
        print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

        # Save model
        save_path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump({"pipeline": pipe, "name": name}, save_path)
        print(f"    Saved → {save_path}")

        results.append({
            "Model":     name,
            "Accuracy":  round(acc,  2),
            "Precision": round(prec, 2),
            "Recall":    round(rec,  2),
            "F1 Score":  round(f1,   2),
            "y_pred":    y_pred,   # kept for plotting, removed before CSV export
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRICES PLOT
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrices(results_df: pd.DataFrame, y_test):
    n = len(results_df)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    fig.patch.set_facecolor("#0d1117")
    axes = axes.flatten()

    for i, row in results_df.iterrows():
        cm = confusion_matrix(y_test, row["y_pred"])
        ax = axes[i]
        ax.set_facecolor("#0d1117")

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f'{row["Model"]}', color="white", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", color="#90caf9", fontsize=10)
        ax.set_ylabel("Actual",    color="#90caf9", fontsize=10)
        ax.set_xticks(range(len(LABEL_NAMES)))
        ax.set_yticks(range(len(LABEL_NAMES)))
        ax.set_xticklabels(LABEL_NAMES, color="white", fontsize=9)
        ax.set_yticklabels(LABEL_NAMES, color="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")

        thresh = cm.max() / 2
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, format(cm[r, c], "d"),
                        ha="center", va="center",
                        color="white" if cm[r, c] < thresh else "#0d1117",
                        fontsize=12, fontweight="bold")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices – All Models", color="white",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy_comparison(results_df: pd.DataFrame):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    bar_colors = ["#2196F3", "#00BCD4", "#4CAF50", "#FF9800"]
    x = np.arange(len(results_df))
    width = 0.20

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    for i, (metric, color) in enumerate(zip(metrics, bar_colors)):
        bars = ax.bar(x + i * width, results_df[metric], width,
                      label=metric, color=color, alpha=0.85, edgecolor="#0d1117")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom",
                    color="white", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right",
                       color="white", fontsize=10)
    ax.set_ylim(70, 105)
    ax.set_ylabel("Score (%)", color="#90caf9", fontsize=11)
    ax.set_title("Model Performance Comparison", color="white",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(facecolor="#1e3a5f", labelcolor="white", fontsize=9)
    ax.tick_params(colors="white")
    ax.yaxis.grid(True, color="#1e3a5f", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE (Random Forest only)
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(results_df: pd.DataFrame, pipelines: dict):
    rf_name = "RF_TF-IDF"
    if rf_name not in pipelines:
        print("  [SKIP] Random Forest TF-IDF not found — skipping feature importance.")
        return

    pipe = pipelines[rf_name]
    vectorizer  = pipe.named_steps["vec"]
    classifier  = pipe.named_steps["clf"]
    feature_names = vectorizer.get_feature_names_out()
    importances   = classifier.feature_importances_

    # Top 20 features
    top_idx   = np.argsort(importances)[-20:][::-1]
    top_feats = feature_names[top_idx]
    top_imps  = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars = ax.barh(range(len(top_feats)), top_imps[::-1],
                   color="#2196F3", alpha=0.85, edgecolor="#0d1117")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats[::-1], color="white", fontsize=10)
    ax.set_xlabel("Feature Importance", color="#90caf9", fontsize=11)
    ax.set_title("Top 20 Important Features – Random Forest (TF-IDF)",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    ax.xaxis.grid(True, color="#1e3a5f", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ChildSafe AI — Model Training & Evaluation")
    print("=" * 60)

    # 1. Load dataset
    print("\n[STEP 1] Loading dataset...")
    df = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    # 2. Build pipelines
    print("\n[STEP 2] Building model pipelines...")
    pipelines = build_pipelines()
    print(f"  {len(pipelines)} models to train: {list(pipelines.keys())}")

    # 3. Train & evaluate
    print("\n[STEP 3] Training all models (this may take 3–5 minutes)...")
    results_df = train_and_evaluate(pipelines, X_train, X_test, y_train, y_test)

    # 4. Save comparison CSV (drop y_pred column first)
    csv_df = results_df.drop(columns=["y_pred"])
    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    csv_df.to_csv(csv_path, index=False)
    print(f"\n[STEP 4] Saved comparison table → {csv_path}")

    # 5. Print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(csv_df.to_string(index=False))

    # 6. Plot visualisations
    print("\n[STEP 5] Generating visualisations...")
    plot_confusion_matrices(results_df, y_test)
    plot_accuracy_comparison(csv_df)
    plot_feature_importance(results_df, pipelines)

    # 7. Best model recommendation
    best = csv_df.loc[csv_df["F1 Score"].idxmax()]
    print("\n" + "=" * 60)
    print("  BEST MODEL (by F1 Score)")
    print("=" * 60)
    print(f"  Model     : {best['Model']}")
    print(f"  Accuracy  : {best['Accuracy']}%")
    print(f"  Precision : {best['Precision']}%")
    print(f"  Recall    : {best['Recall']}%")
    print(f"  F1 Score  : {best['F1 Score']}%")
    print("\n  Deployment recommendation: RF_TF-IDF")
    print("  (Best generalisation + interpretable feature importances)")
    print("\n[SUCCESS] All models trained and saved!")
    print("          Next step: streamlit run app.py")