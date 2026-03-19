"""
NephroScreen External Validation
==================================
Implements source-based splitting for external validation.
Training data = literature_curated + chembl + tox21
External validation = faers + review_papers (independent provenance)
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "models"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def split_by_source(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by data source for external validation.

    Training sources: literature_curated, chembl_*, tox21_*
    External sources: faers, review_papers

    Returns (train_df, external_df)
    """
    train_sources = ["literature_curated"]
    external_sources = ["faers", "review_papers"]

    train_mask = df["source"].apply(
        lambda s: any(s.startswith(ts) for ts in train_sources) or s.startswith("chembl") or s.startswith("tox21")
    )
    external_mask = df["source"].apply(
        lambda s: any(s == es for es in external_sources)
    )

    train_df = df[train_mask].copy()
    external_df = df[external_mask].copy()

    logger.info(f"Training set: {len(train_df)} compounds "
                f"({(train_df['label']==0).sum()} protective, {(train_df['label']==1).sum()} toxic)")
    logger.info(f"External set: {len(external_df)} compounds "
                f"({(external_df['label']==0).sum()} protective, {(external_df['label']==1).sum()} toxic)")

    return train_df, external_df


def evaluate_external(
    model, scaler, external_df: pd.DataFrame, feature_matrix_external: np.ndarray
) -> dict:
    """
    Evaluate a trained model on the external validation set.
    Returns a dictionary of metrics.
    """
    X_ext = np.nan_to_num(feature_matrix_external, nan=0.0, posinf=0.0, neginf=0.0)
    X_ext_scaled = scaler.transform(X_ext)
    X_ext_scaled = np.nan_to_num(X_ext_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    y_ext = external_df["label"].values

    y_pred = model.predict(X_ext_scaled)
    y_prob = model.predict_proba(X_ext_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_ext, y_pred), 4),
        "precision": round(precision_score(y_ext, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_ext, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_ext, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_ext, y_prob), 4) if len(np.unique(y_ext)) > 1 else None,
        "n_samples": int(len(y_ext)),
        "n_protective": int((y_ext == 0).sum()),
        "n_toxic": int((y_ext == 1).sum()),
    }

    logger.info(f"External Validation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    report = classification_report(
        y_ext, y_pred,
        target_names=["Nephroprotective", "Nephrotoxic"],
        zero_division=0,
    )
    logger.info(f"\n{report}")

    return metrics, y_pred, y_prob, y_ext


def plot_external_validation(y_true, y_pred, y_prob, output_dir: Path = None):
    """Generate external validation plots."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Protective", "Toxic"],
                yticklabels=["Protective", "Toxic"])
    axes[0].set_title("External Validation — Confusion Matrix", fontweight="bold")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ROC Curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[1].plot(fpr, tpr, color="#E74C3C", lw=2, label=f"External (AUC = {auc:.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("External Validation — ROC Curve", fontweight="bold")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "external_validation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'external_validation.png'}")
