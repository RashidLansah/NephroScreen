"""
NephroScreen Similarity Search Module
=======================================
Computes Tanimoto similarity between a query compound and the training
dataset using Morgan fingerprints. Provides interpretability by showing
the most similar known compounds and their labels.
"""

import pickle
from pathlib import Path
from typing import Optional

from rdkit.DataStructs import TanimotoSimilarity

from .mol_utils import compute_morgan_fp_obj, mol_from_smiles

MODELS_DIR = Path(__file__).parent.parent / "models"

_fp_data = None


def _load_fp_data():
    """Load precomputed fingerprint data (cached)."""
    global _fp_data
    if _fp_data is None:
        with open(MODELS_DIR / "fingerprint_data.pkl", "rb") as f:
            _fp_data = pickle.load(f)
    return _fp_data


def find_similar_compounds(
    smiles: str, top_n: int = 5
) -> Optional[list[dict]]:
    """
    Find the most similar compounds in the training set.

    Args:
        smiles: SMILES string of the query compound
        top_n: Number of similar compounds to return

    Returns:
        List of dicts with keys: compound_name, smiles, label_description,
        similarity_score. Sorted by similarity (descending).
        Returns None if SMILES is invalid.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    query_fp = compute_morgan_fp_obj(mol)
    fp_data = _load_fp_data()

    similarities = []
    for entry in fp_data:
        sim = TanimotoSimilarity(query_fp, entry["morgan_fp"])
        similarities.append({
            "compound_name": entry["compound_name"],
            "smiles": entry["smiles"],
            "label": entry["label"],
            "label_description": entry["label_description"],
            "similarity_score": round(sim, 4),
        })

    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similarities[:top_n]


def compute_applicability_domain(smiles: str, threshold: float = 0.3) -> dict:
    """
    Check if a compound is within the applicability domain of the model.

    Uses the maximum Tanimoto similarity to any training compound.
    If the max similarity is below the threshold, the compound is flagged
    as being outside the training domain (predictions less reliable).

    Args:
        smiles: SMILES string
        threshold: Minimum similarity threshold (default 0.3)

    Returns:
        Dictionary with max_similarity, in_domain (bool), nearest_compound
    """
    similar = find_similar_compounds(smiles, top_n=1)
    if similar is None or len(similar) == 0:
        return {
            "max_similarity": 0.0,
            "in_domain": False,
            "nearest_compound": None,
            "message": "Could not compute similarity (invalid molecule).",
        }

    nearest = similar[0]
    in_domain = nearest["similarity_score"] >= threshold

    if in_domain:
        message = (
            f"This compound has a Tanimoto similarity of {nearest['similarity_score']:.3f} "
            f"to the nearest training compound ({nearest['compound_name']}). "
            f"The prediction is within the applicability domain."
        )
    else:
        message = (
            f"Warning: This compound has a maximum Tanimoto similarity of "
            f"{nearest['similarity_score']:.3f} to the nearest training compound "
            f"({nearest['compound_name']}). This is below the threshold of {threshold:.2f}, "
            f"indicating the compound is structurally dissimilar to the training data. "
            f"The prediction may be less reliable."
        )

    return {
        "max_similarity": nearest["similarity_score"],
        "in_domain": in_domain,
        "nearest_compound": nearest["compound_name"],
        "message": message,
    }
