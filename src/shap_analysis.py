"""
NephroScreen SHAP Explainability
==================================
SHAP (SHapley Additive exPlanations) analysis for the XGBoost model.
Maps important Morgan fingerprint bits back to chemical substructures.
"""

import io
import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def compute_shap_values(model, X_test_scaled: np.ndarray, feature_names: list):
    """
    Compute SHAP values for the test set using TreeExplainer.
    Returns SHAP explainer and values.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Save for reuse
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    np.save(MODELS_DIR / "shap_values.npy", shap_values)

    logger.info(f"SHAP values computed: shape {np.array(shap_values).shape}")
    return explainer, shap_values


def plot_shap_summary(shap_values, X_test, feature_names: list, output_dir: Path = None):
    """Generate SHAP summary plots (beeswarm and bar)."""
    import shap

    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar plot — mean |SHAP| values
    fig, ax = plt.subplots(figsize=(8, 7))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-20:][::-1]
    top_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in top_indices]
    top_values = mean_abs_shap[top_indices]

    # Color by type
    colors = ["#2196F3" if "morgan" in n else "#27AE60" for n in top_names]
    ax.barh(range(len(top_names)), top_values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("NephroScreen — Top 20 Features by SHAP Importance", fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", alpha=0.85, label="Morgan FP bits"),
        Patch(facecolor="#27AE60", alpha=0.85, label="Molecular descriptors"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'shap_bar.png'}")

    # Beeswarm plot
    try:
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_names,
            max_display=20,
            show=False,
        )
        plt.title("NephroScreen — SHAP Beeswarm Plot", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_dir / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {output_dir / 'shap_beeswarm.png'}")
    except Exception as e:
        logger.warning(f"Beeswarm plot failed: {e}")


def explain_single_prediction(
    smiles: str, model, scaler, feature_names: list
) -> dict:
    """
    Generate SHAP explanation for a single compound.
    Returns top contributing features with direction and magnitude.
    """
    import shap
    from .mol_utils import compute_descriptors, compute_morgan_fingerprint, mol_from_smiles

    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    # Build feature vector
    morgan_fp = compute_morgan_fingerprint(mol)
    descriptors = compute_descriptors(mol)
    desc_values = np.array(list(descriptors.values()))
    feature_vector = np.concatenate([morgan_fp, desc_values]).reshape(1, -1)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    feature_scaled = scaler.transform(feature_vector)
    feature_scaled = np.nan_to_num(feature_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(feature_scaled)[0]

    # Get top features by |SHAP|
    abs_shap = np.abs(shap_vals)
    top_indices = np.argsort(abs_shap)[-15:][::-1]

    contributions = []
    for idx in top_indices:
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        contributions.append({
            "feature": name,
            "shap_value": float(shap_vals[idx]),
            "direction": "toxic" if shap_vals[idx] > 0 else "protective",
            "magnitude": float(abs_shap[idx]),
            "feature_value": float(feature_scaled[0, idx]),
        })

    # Generate waterfall plot as image bytes
    waterfall_img = _generate_waterfall_image(explainer, feature_scaled, feature_names)

    return {
        "contributions": contributions,
        "base_value": float(explainer.expected_value),
        "waterfall_image": waterfall_img,
    }


def _generate_waterfall_image(explainer, feature_scaled, feature_names) -> bytes:
    """Generate a SHAP waterfall plot and return as PNG bytes."""
    import shap

    try:
        shap_vals = explainer(feature_scaled)
        # Assign feature names
        if hasattr(shap_vals, 'feature_names'):
            shap_vals.feature_names = feature_names[:feature_scaled.shape[1]]

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
        plt.title("Feature Contributions (SHAP)", fontsize=11, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Waterfall plot generation failed: {e}")
        return None
