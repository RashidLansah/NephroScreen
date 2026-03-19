"""
AutoDock Vina wrapper for molecular docking via CLI.
Uses the Vina binary at bin/vina and temporary files for I/O.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VINA_BIN = Path(__file__).parent.parent.parent / "bin" / "vina"


def dock_compound(
    ligand_pdbqt: str,
    receptor_pdbqt_path: Path,
    center: list[float],
    box_size: list[float],
    exhaustiveness: int = 16,
    n_poses: int = 9,
) -> Optional[dict]:
    """
    Dock a single compound using AutoDock Vina CLI.

    Args:
        ligand_pdbqt: PDBQT string of the ligand
        receptor_pdbqt_path: Path to receptor PDBQT file
        center: [x, y, z] center of search box
        box_size: [sx, sy, sz] dimensions of search box
        exhaustiveness: Search thoroughness (default 16)
        n_poses: Number of binding modes to generate

    Returns:
        Dict with affinity, poses, raw output. None on failure.
    """
    if not VINA_BIN.exists():
        logger.error(f"Vina binary not found at {VINA_BIN}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        ligand_file = tmpdir / "ligand.pdbqt"
        output_file = tmpdir / "output.pdbqt"

        # Write ligand PDBQT
        ligand_file.write_text(ligand_pdbqt)

        # Run Vina
        cmd = [
            str(VINA_BIN),
            "--receptor", str(receptor_pdbqt_path),
            "--ligand", str(ligand_file),
            "--out", str(output_file),
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(n_poses),
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0:
                logger.warning(f"Vina failed: {result.stderr[:200]}")
                return None

            # Parse results from stdout
            affinities = _parse_vina_output(result.stdout)

            output_pdbqt = ""
            if output_file.exists():
                output_pdbqt = output_file.read_text()

            if affinities:
                return {
                    "best_affinity": affinities[0]["affinity"],
                    "all_affinities": affinities,
                    "n_poses": len(affinities),
                    "output_pdbqt": output_pdbqt,
                    "raw_stdout": result.stdout,
                }
            else:
                logger.warning("No affinities parsed from Vina output")
                return None

        except subprocess.TimeoutExpired:
            logger.warning("Vina timed out (300s)")
            return None
        except Exception as e:
            logger.warning(f"Vina error: {e}")
            return None


def _parse_vina_output(stdout: str) -> list[dict]:
    """Parse binding affinities from Vina stdout."""
    affinities = []
    in_results = False

    for line in stdout.split("\n"):
        line = line.strip()
        if "-----+------------+----------+----------" in line:
            in_results = True
            continue
        if in_results and line:
            parts = line.split()
            if len(parts) >= 4:
                try:
                    mode = int(parts[0])
                    affinity = float(parts[1])
                    rmsd_lb = float(parts[2])
                    rmsd_ub = float(parts[3])
                    affinities.append({
                        "mode": mode,
                        "affinity": affinity,
                        "rmsd_lb": rmsd_lb,
                        "rmsd_ub": rmsd_ub,
                    })
                except (ValueError, IndexError):
                    if affinities:  # We already got some, end of table
                        break

    return affinities


def batch_dock(
    compounds: list[dict],
    receptor_pdbqt_path: Path,
    center: list[float],
    box_size: list[float],
    exhaustiveness: int = 16,
) -> list[dict]:
    """
    Dock multiple compounds and return results.

    Args:
        compounds: List of dicts with 'name', 'smiles', and optionally 'pdbqt'
        receptor_pdbqt_path: Path to prepared receptor
        center, box_size: Docking box parameters
        exhaustiveness: Search thoroughness

    Returns:
        List of result dicts with compound info + docking results
    """
    from .ligand_prep import prepare_ligand

    results = []
    total = len(compounds)

    for i, compound in enumerate(compounds):
        name = compound.get("name", f"compound_{i}")
        smiles = compound.get("smiles", "")
        logger.info(f"Docking {i+1}/{total}: {name}")

        # Prepare ligand
        pdbqt = compound.get("pdbqt") or prepare_ligand(smiles)
        if pdbqt is None:
            results.append({
                "name": name,
                "smiles": smiles,
                "docking_status": "ligand_prep_failed",
                "best_affinity": None,
                "n_poses": 0,
            })
            continue

        # Dock
        dock_result = dock_compound(
            pdbqt, receptor_pdbqt_path, center, box_size, exhaustiveness
        )

        if dock_result is None:
            results.append({
                "name": name,
                "smiles": smiles,
                "docking_status": "docking_failed",
                "best_affinity": None,
                "n_poses": 0,
            })
        else:
            results.append({
                "name": name,
                "smiles": smiles,
                "docking_status": "success",
                "best_affinity": dock_result["best_affinity"],
                "n_poses": dock_result["n_poses"],
                "all_affinities": dock_result["all_affinities"],
            })

    return results
