"""
NephroScreen Prediction Module
===============================
Loads the trained model and makes nephrotoxicity predictions
for new compounds given their SMILES strings.
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from rdkit import Chem

from .mol_utils import (
    compute_descriptors,
    compute_morgan_fingerprint,
    mol_from_smiles,
)

MODELS_DIR = Path(__file__).parent.parent / "models"

# Cache loaded model artifacts
_model = None
_scaler = None
_metrics = None


def _load_artifacts():
    """Load model, scaler, and metrics (cached after first call)."""
    global _model, _scaler, _metrics
    if _model is None:
        _model = joblib.load(MODELS_DIR / "best_model.joblib")
        _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        with open(MODELS_DIR / "model_metrics.json") as f:
            _metrics = json.load(f)
    return _model, _scaler, _metrics


def predict_nephrotoxicity(smiles: str) -> Optional[dict]:
    """
    Predict nephrotoxicity for a single compound.

    Args:
        smiles: SMILES string of the compound

    Returns:
        Dictionary with prediction results, or None if SMILES is invalid.
        Keys: prediction, label, confidence, probability_protective,
              probability_toxic, descriptors, lipinski
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    model, scaler, metrics = _load_artifacts()

    # Compute features (same pipeline as training)
    morgan_fp = compute_morgan_fingerprint(mol)
    descriptors = compute_descriptors(mol)

    # Build feature vector: Morgan FP + descriptors
    desc_values = np.array(list(descriptors.values()))
    feature_vector = np.concatenate([morgan_fp, desc_values]).reshape(1, -1)

    # Handle NaN/Inf
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale
    feature_vector_scaled = scaler.transform(feature_vector)
    feature_vector_scaled = np.nan_to_num(feature_vector_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Predict
    prediction = model.predict(feature_vector_scaled)[0]
    probabilities = model.predict_proba(feature_vector_scaled)[0]

    # Lipinski Rule of 5
    from .mol_utils import lipinski_rule_of_5
    lipinski = lipinski_rule_of_5(descriptors)

    return {
        "prediction": int(prediction),
        "label": "Nephrotoxic" if prediction == 1 else "Nephroprotective",
        "confidence": float(max(probabilities)),
        "probability_protective": float(probabilities[0]),
        "probability_toxic": float(probabilities[1]),
        "descriptors": descriptors,
        "lipinski": lipinski,
    }


def get_model_metrics() -> dict:
    """Return the saved model performance metrics."""
    _, _, metrics = _load_artifacts()
    return metrics
