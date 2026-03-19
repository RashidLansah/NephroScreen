"""
Ligand preparation for docking: SMILES → 3D conformer → PDBQT.
Uses RDKit for 3D generation and Meeko for PDBQT conversion.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smiles_to_3d_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES to a 3D-optimized RDKit molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result == -1:
        # Fallback: use random coordinates
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            logger.warning(f"Could not generate 3D coords for {smiles}")
            return None

    # Optimize with UFF force field
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass  # Use unoptimized coords

    return mol


def mol_to_pdbqt_string(mol: Chem.Mol) -> Optional[str]:
    """Convert an RDKit Mol with 3D coords to PDBQT string via Meeko."""
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        for setup in mol_setups:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                return pdbqt_string
            else:
                logger.warning(f"Meeko PDBQT error: {error_msg}")
    except Exception as e:
        logger.warning(f"Meeko conversion failed: {e}")

    # Fallback: write PDB and do simple conversion
    return _fallback_pdbqt(mol)


def _fallback_pdbqt(mol: Chem.Mol) -> Optional[str]:
    """Fallback PDBQT generation using RDKit PDB output."""
    try:
        pdb_block = Chem.MolToPDBBlock(mol)
        if not pdb_block:
            return None

        lines = []
        for line in pdb_block.split("\n"):
            if line.startswith("HETATM") or line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                element = atom_name[0] if atom_name else "C"

                ad_type_map = {"C": "C", "N": "NA", "O": "OA", "S": "SA", "H": "HD",
                               "F": "F", "Cl": "Cl", "Br": "Br", "I": "I", "P": "P"}
                ad_type = ad_type_map.get(element, "C")

                pdbqt_line = line[:54].ljust(54) + f"  0.00  0.00    {0.000:+8.3f} {ad_type:>2s}"
                lines.append(pdbqt_line)
            elif line.startswith("END"):
                lines.append("END")

        return "\n".join(lines) + "\n"
    except Exception as e:
        logger.warning(f"Fallback PDBQT failed: {e}")
        return None


def prepare_ligand(smiles: str) -> Optional[str]:
    """
    Full pipeline: SMILES → 3D molecule → PDBQT string.
    Returns PDBQT string ready for Vina docking.
    """
    mol = smiles_to_3d_mol(smiles)
    if mol is None:
        return None
    return mol_to_pdbqt_string(mol)


def save_ligand_pdbqt(smiles: str, output_path: Path) -> bool:
    """Save a ligand PDBQT file from SMILES."""
    pdbqt = prepare_ligand(smiles)
    if pdbqt is None:
        return False
    output_path.write_text(pdbqt)
    return True
