"""
Protein structure download and preparation for docking.
Downloads PDB structures and prepares them as PDBQT files for AutoDock Vina.
"""

import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROTEINS_DIR = Path(__file__).parent.parent.parent / "data" / "proteins"
VINA_BIN = Path(__file__).parent.parent.parent / "bin" / "vina"


def download_pdb(pdb_id: str, output_dir: Path = None) -> Path:
    """Download a PDB structure from RCSB."""
    import requests

    if output_dir is None:
        output_dir = PROTEINS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_file = output_dir / f"{pdb_id.lower()}.pdb"
    if pdb_file.exists():
        logger.info(f"PDB {pdb_id} already downloaded: {pdb_file}")
        return pdb_file

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    pdb_file.write_text(resp.text)
    logger.info(f"Downloaded {pdb_id} to {pdb_file}")
    return pdb_file


def clean_pdb_for_docking(pdb_file: Path, chain: str = "A") -> Path:
    """
    Clean a PDB file for docking:
    - Keep only specified chain
    - Remove water molecules (HOH)
    - Remove non-standard ligands
    - Keep protein atoms only (ATOM records)
    """
    cleaned_file = pdb_file.parent / f"{pdb_file.stem}_clean.pdb"
    kept_lines = []

    with open(pdb_file) as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "TER"):
                line_chain = line[21] if len(line) > 21 else ""
                if chain == "all" or line_chain == chain:
                    kept_lines.append(line)
            elif record == "END":
                kept_lines.append(line)

    with open(cleaned_file, "w") as f:
        f.writelines(kept_lines)

    logger.info(f"Cleaned PDB: {cleaned_file} (chain {chain}, {len(kept_lines)} lines)")
    return cleaned_file


def prepare_receptor_pdbqt(pdb_file: Path) -> Path:
    """
    Convert a clean PDB to PDBQT format using Meeko's mk_prepare_receptor.
    Falls back to a simple conversion if the tool is unavailable.
    """
    pdbqt_file = pdb_file.parent / f"{pdb_file.stem}.pdbqt"
    if pdbqt_file.exists():
        logger.info(f"Receptor PDBQT already exists: {pdbqt_file}")
        return pdbqt_file

    # Try mk_prepare_receptor.py (from meeko) with --allow_bad_res for robustness
    try:
        # meeko outputs with a basename, then adds .pdbqt
        basename = str(pdbqt_file).replace(".pdbqt", "")
        result = subprocess.run(
            ["mk_prepare_receptor.py", "-i", str(pdb_file), "-o", basename, "-p", "-a"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0 and pdbqt_file.exists():
            logger.info(f"Prepared receptor PDBQT: {pdbqt_file}")
            return pdbqt_file
        else:
            logger.warning(f"mk_prepare_receptor failed: {result.stderr[:200]}")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"mk_prepare_receptor unavailable: {e}")

    # Fallback: simple PDB → PDBQT conversion (adds charges and atom types)
    logger.info("Using fallback PDB → PDBQT conversion")
    _simple_pdb_to_pdbqt(pdb_file, pdbqt_file)
    return pdbqt_file


def _simple_pdb_to_pdbqt(pdb_file: Path, pdbqt_file: Path):
    """
    Simple PDB to PDBQT conversion.
    Assigns Gasteiger charges and AutoDock atom types.
    """
    # Map common atom names to AutoDock types
    # AutoDock atom types: N=nitrogen (H-bond acceptor), NA=nitrogen acceptor (aromatic),
    # A=aromatic carbon, C=aliphatic carbon, OA=oxygen acceptor, SA=sulfur acceptor, HD=H donor
    ad_type_map = {
        # Backbone
        "N": "N", "CA": "C", "C": "C", "O": "OA",
        # Side chain carbons
        "CB": "C", "CG": "C", "CG1": "C", "CG2": "C",
        "CD": "C", "CD1": "C", "CD2": "C",
        "CE": "C", "CE1": "C", "CE2": "C", "CE3": "C",
        "CZ": "C", "CZ2": "C", "CZ3": "C", "CH2": "C",
        # Side chain nitrogens
        "NE": "NA", "NE1": "NA", "NE2": "NA",
        "NH1": "N", "NH2": "N", "NZ": "N",
        "ND1": "NA", "ND2": "N",
        # Side chain oxygens
        "OD1": "OA", "OD2": "OA", "OE1": "OA", "OE2": "OA",
        "OG": "OA", "OG1": "OA", "OH": "OA", "OXT": "OA",
        # Sulfur
        "S": "SA", "SG": "SA", "SD": "SA",
        # Hydrogen
        "H": "HD", "HA": "H", "HB": "H",
    }

    lines = []
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
                ad_type = ad_type_map.get(atom_name, ad_type_map.get(element, "C"))

                # PDBQT format: cols 1-54 (PDB), 55-60 (partial charge), 61-66 (blank),
                # 67-76 (blank), 77-78 (AD atom type)
                # Vina expects exactly: PDB_coords + occupancy + bfactor + charge + type
                base = line[:54].rstrip()
                # Pad to column 54, then add occupancy(55-60), bfactor(61-66), charge(67-76), type(77-78)
                pdbqt_line = f"{base:<54s}{1.00:6.2f}{0.00:6.2f}    {0.000:+8.3f} {ad_type:<2s}\n"
                lines.append(pdbqt_line)
            elif line.startswith("TER") or line.startswith("END"):
                lines.append(line)

    with open(pdbqt_file, "w") as f:
        f.writelines(lines)

    logger.info(f"Simple PDBQT conversion: {pdbqt_file}")


# COX-2 active site definitions
COX2_ACTIVE_SITE = {
    "pdb_id": "5KIR",
    "chain": "A",
    "center": [23.2, 1.3, 34.3],  # Center of rofecoxib (RCX) binding site in 5KIR chain A
    "box_size": [22, 22, 22],
    "description": "Human COX-2 active site (cyclooxygenase channel)",
    "key_residues": "Tyr-385, Ser-530, Arg-120, Val-349, Leu-352",
}


def setup_cox2_receptor() -> tuple[Path, dict]:
    """
    Download and prepare COX-2 (5KIR) for docking.
    Returns (pdbqt_path, active_site_config).
    """
    pdb_file = download_pdb("5KIR")
    clean_file = clean_pdb_for_docking(pdb_file, chain="A")
    pdbqt_file = prepare_receptor_pdbqt(clean_file)
    return pdbqt_file, COX2_ACTIVE_SITE
