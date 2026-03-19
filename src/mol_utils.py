"""
Molecular Feature Engineering for NephroScreen
================================================
Computes molecular descriptors, Morgan fingerprints, and MACCS keys
for use in ML classification and similarity searching.
"""

from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Safely parse a SMILES string into an RDKit Mol object."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def compute_descriptors(mol: Chem.Mol) -> dict[str, float]:
    """
    Compute key molecular descriptors for a molecule.
    Returns a dictionary of descriptor name -> value.
    """
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "NumRings": Descriptors.RingCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "MolRefractivity": Descriptors.MolMR(mol),
        "NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol),
        "MaxPartialCharge": Descriptors.MaxPartialCharge(mol),
        "MinPartialCharge": Descriptors.MinPartialCharge(mol),
        "BalabanJ": Descriptors.BalabanJ(mol) if Descriptors.RingCount(mol) > 0 else 0.0,
        "BertzCT": Descriptors.BertzCT(mol),
    }


def compute_morgan_fingerprint(
    mol: Chem.Mol, radius: int = 2, n_bits: int = 2048
) -> np.ndarray:
    """Compute Morgan (circular) fingerprint as a bit vector."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def compute_morgan_fp_obj(
    mol: Chem.Mol, radius: int = 2, n_bits: int = 2048
):
    """Return RDKit fingerprint object (for Tanimoto similarity)."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_maccs_keys(mol: Chem.Mol) -> np.ndarray:
    """Compute MACCS fingerprint keys (166 bits)."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def compute_maccs_keys_obj(mol: Chem.Mol):
    """Return RDKit MACCS keys object (for Tanimoto similarity)."""
    return MACCSkeys.GenMACCSKeys(mol)


def lipinski_rule_of_5(descriptors: dict[str, float]) -> dict[str, bool]:
    """
    Evaluate Lipinski's Rule of 5 for oral bioavailability.
    Returns individual checks and overall pass/fail.
    """
    checks = {
        "MW ≤ 500": descriptors["MolWt"] <= 500,
        "LogP ≤ 5": descriptors["LogP"] <= 5,
        "HBD ≤ 5": descriptors["HBD"] <= 5,
        "HBA ≤ 10": descriptors["HBA"] <= 10,
    }
    violations = sum(1 for v in checks.values() if not v)
    checks["Passes Ro5"] = violations <= 1
    checks["Violations"] = violations
    return checks


def tanimoto_similarity(fp1, fp2) -> float:
    """Compute Tanimoto similarity between two RDKit fingerprint objects."""
    return TanimotoSimilarity(fp1, fp2)


def featurize_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Compute all features for a dataset of compounds.

    Args:
        df: DataFrame with a 'smiles' column

    Returns:
        morgan_fps: numpy array of Morgan fingerprints (n_compounds x 2048)
        maccs_fps: numpy array of MACCS keys (n_compounds x 167)
        desc_df: DataFrame of molecular descriptors
    """
    morgan_fps = []
    maccs_fps = []
    descriptor_records = []
    valid_indices = []

    for idx, row in df.iterrows():
        mol = mol_from_smiles(row["smiles"])
        if mol is None:
            continue

        try:
            morgan_fps.append(compute_morgan_fingerprint(mol))
            maccs_fps.append(compute_maccs_keys(mol))
            descriptor_records.append(compute_descriptors(mol))
            valid_indices.append(idx)
        except Exception:
            continue

    morgan_array = np.array(morgan_fps)
    maccs_array = np.array(maccs_fps)
    desc_df = pd.DataFrame(descriptor_records, index=valid_indices)

    return morgan_array, maccs_array, desc_df


def resolve_compound_name(name: str) -> Optional[str]:
    """
    Resolve a compound name to SMILES using PubChem PUG REST API.
    Returns canonical SMILES or None if resolution fails.
    """
    import requests

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                smiles = props[0].get("CanonicalSMILES") or props[0].get("ConnectivitySMILES")
                if smiles:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return None
