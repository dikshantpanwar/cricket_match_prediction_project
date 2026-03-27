# -*- coding: utf-8 -*-
import os
import sys

# Fix Windows console encoding so all print() calls work without crashing
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")           # MUST be before pyplot import -- no GUI needed
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import MODELS_DIR, load_encoders, TARGET_COL

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)
os.makedirs(REPORTS_DIR, exist_ok=True)


def evaluate():
    # -- Load everything saved by train.py ----------------------
    all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    test_data  = joblib.load(os.path.join(MODELS_DIR, "test_data.pkl"))
    load_encoders()

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print("\n" + "="*60)
    print("  MODEL EVALUATION REPORT")
    print("="*60)

    summary = {}

    for name, model in all_models.items():
        preds = model.predict(X_test)

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1   = f1_score(y_test, preds, average="weighted", zero_division=0)

        summary[name] = {"Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1-Score": f1}

        print(f"\n-- {name} --")
        print(f"  Accuracy  : {acc*100:.2f}%")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1-Score  : {f1*100:.2f}%")

    # -- Side-by-side bar chart of all metrics -------------------
    print("\n[INFO] Saving metrics chart...")
    _plot_metrics(summary)

    # -- Confusion matrix for best model only --------------------
    best_name = max(summary, key=lambda n: summary[n]["Accuracy"])
    best_model = all_models[best_name]
    _plot_confusion(best_model, X_test, y_test, best_name)

    print(f"\n[BEST] {best_name}  ->  {summary[best_name]['Accuracy']*100:.2f}% accuracy")
    print(f"[INFO] Charts saved in  ->  {REPORTS_DIR}")
    print("\n[OK] evaluate.py")


def _plot_metrics(summary):
    """Bar chart comparing Accuracy, Precision, Recall, F1 across all models."""
    import numpy as np

    metrics  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models   = list(summary.keys())
    x        = np.arange(len(metrics))
    width    = 0.22
    colors   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (model_name, color) in enumerate(zip(models, colors)):
        vals = [summary[model_name][m] * 100 for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=model_name,
                      color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison -- All Metrics")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out = os.path.join(REPORTS_DIR, "model_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Metrics chart   ->  {out}")


def _plot_confusion(model, X_test, y_test, model_name):
    """Confusion matrix for the best model (top-10 teams for readability)."""
    from preprocess import encoders, TARGET_COL
    import numpy as np

    preds  = model.predict(X_test)
    le     = encoders[TARGET_COL]

    # Keep only top 10 most-frequent teams in test set for readable matrix
    unique, counts = np.unique(y_test, return_counts=True)
    top10 = unique[np.argsort(counts)[::-1][:10]]

    mask   = np.isin(y_test, top10)
    yt_top = y_test[mask]
    yp_top = preds[mask]

    labels      = top10
    label_names = le.inverse_transform(labels)

    cm   = confusion_matrix(yt_top, yp_top, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(f"Confusion Matrix -- {model_name} (top 10 teams)")
    plt.tight_layout()

    out = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix ->  {out}")


# -------------------------------------------------------------
#  Run  ->  python src/evaluate.py
# -------------------------------------------------------------
if __name__ == "__main__":
    evaluate()