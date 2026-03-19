"""
NephroScreen Model Training Pipeline
======================================
Trains, evaluates, and selects the best ML classifier for nephrotoxicity prediction.
Models compared: Random Forest, XGBoost, Logistic Regression.
Handles class imbalance with SMOTE and class weighting.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load feature matrix and labels."""
    feature_matrix = np.load(PROCESSED_DIR / "feature_matrix.npy")
    df = pd.read_csv(PROCESSED_DIR / "nephroscreen_dataset_valid.csv")
    labels = df["label"].values
    return feature_matrix, labels, df


def train_and_evaluate():
    """Full training pipeline with model comparison."""
    print("=" * 60)
    print("NEPHROSCREEN MODEL TRAINING")
    print("=" * 60)

    # Load data
    X, y, df = load_data()
    print(f"\nDataset: {X.shape[0]} compounds, {X.shape[1]} features")
    print(f"Class 0 (nephroprotective): {(y == 0).sum()}")
    print(f"Class 1 (nephrotoxic): {(y == 1).sum()}")

    # Split data: 80/20 stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Train class dist: {(y_train == 0).sum()} protective, {(y_train == 1).sum()} toxic")
    print(f"Test class dist: {(y_test == 0).sum()} protective, {(y_test == 1).sum()} toxic")

    # Replace NaN/Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Replace any remaining NaN from scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Handle class imbalance with SMOTE on training set only
    print("\nApplying SMOTE to balance training data...")
    smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 0).sum() - 1))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"After SMOTE: {(y_train_resampled == 0).sum()} protective, "
          f"{(y_train_resampled == 1).sum()} toxic")

    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        ),
    }

    # Train and evaluate each model
    results = {}
    best_model_name = None
    best_auc = 0

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"Training: {name}")
        print(f"{'─' * 40}")

        # Train on SMOTE-resampled data for RF and XGB, scaled for LR
        if name == "Logistic Regression":
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        elif name == "XGBoost":
            # XGBoost: use original training data with scale_pos_weight
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            # Random Forest: use SMOTE-resampled, no scaling needed
            model.fit(X_train_resampled, y_train_resampled)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        print(f"  5-Fold CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Nephroprotective', 'Nephrotoxic'])}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    # Select best model
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_model_name} (ROC-AUC = {best_auc:.4f})")
    print(f"{'=' * 60}")

    best = results[best_model_name]
    best_model = best["model"]

    # Save model artifacts
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # Save metrics
    metrics = {
        "best_model": best_model_name,
        "dataset_size": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
        "class_distribution": {
            "nephroprotective": int((y == 0).sum()),
            "nephrotoxic": int((y == 1).sum()),
        },
        "all_models": {},
    }
    for name, res in results.items():
        metrics["all_models"][name] = {
            "accuracy": round(res["accuracy"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1_score": round(res["f1_score"], 4),
            "roc_auc": round(res["roc_auc"], 4),
            "cv_auc_mean": round(res["cv_auc_mean"], 4),
            "cv_auc_std": round(res["cv_auc_std"], 4),
        }
    metrics["best_metrics"] = metrics["all_models"][best_model_name]

    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate plots
    _plot_confusion_matrices(results, y_test)
    _plot_roc_curves(results, y_test)
    _plot_feature_importance(best_model, best_model_name, X.shape[1])
    _plot_model_comparison(results)

    print(f"\nAll artifacts saved to {MODELS_DIR}/")
    return results, best_model_name


def _plot_confusion_matrices(results: dict, y_test: np.ndarray):
    """Generate confusion matrix plots for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Protective", "Toxic"],
            yticklabels=["Protective", "Toxic"],
        )
        ax.set_title(f"{name}\nAcc={res['accuracy']:.3f} | AUC={res['roc_auc']:.3f}",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.suptitle("NephroScreen — Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrices.png")


def _plot_roc_curves(results: dict, y_test: np.ndarray):
    """Generate ROC curve comparison plot."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {res['roc_auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("NephroScreen — ROC Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: roc_curves.png")


def _plot_feature_importance(model, model_name: str, n_features: int):
    """Plot top 20 most important features."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            print("Model has no feature importance attribute, skipping plot.")
            return

        # Load feature names
        with open(MODELS_DIR / "feature_names.json") as f:
            feature_names = json.load(f)

        # Get top 20
        top_indices = np.argsort(importances)[-20:][::-1]
        top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
                     for i in top_indices]
        top_values = importances[top_indices]

        fig, ax = plt.subplots(figsize=(8, 7))
        colors = ["#27AE60" if "morgan" not in n else "#2196F3" for n in top_names]
        bars = ax.barh(range(len(top_names)), top_values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Feature Importance", fontsize=11)
        ax.set_title(f"NephroScreen — Top 20 Features ({model_name})",
                     fontsize=12, fontweight="bold")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2196F3", alpha=0.8, label="Morgan FP bits"),
            Patch(facecolor="#27AE60", alpha=0.8, label="Molecular descriptors"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(MODELS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: feature_importance.png")
    except Exception as e:
        print(f"Feature importance plot failed: {e}")


def _plot_model_comparison(results: dict):
    """Bar chart comparing all models across metrics."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    model_names = list(results.keys())
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics_to_plot))
    width = 0.25

    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [results[name][m] for m in metrics_to_plot]
        bars = ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("NephroScreen — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_comparison.png")


if __name__ == "__main__":
    results, best_name = train_and_evaluate()
