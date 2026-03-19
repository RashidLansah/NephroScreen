"""
NephroScreen — A. conyzoides Compound Screening
=================================================
Screens specific compounds identified from Ageratum conyzoides
in published GC-MS, HPLC, and phytochemical studies.

IMPORTANT: These compounds are from published LITERATURE studies of
A. conyzoides (Kotta et al. 2020, Okunade 2002, Bosi et al. 2013),
NOT from Lansah's thesis experimental results. The thesis performed
qualitative phytochemical screening only (class-level, not compound-level).
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Compounds identified from A. conyzoides in published literature
# Each entry: name, SMILES, compound_class, literature_source
AGERATUM_COMPOUNDS = [
    # Chromenes (characteristic of Ageratum genus)
    {"name": "Precocene I", "smiles": "COc1cc2OC(C)(C)C=Cc2cc1", "class": "Chromene",
     "source": "Kotta et al. 2020; Okunade 2002"},
    {"name": "Precocene II", "smiles": "COc1cc2OC(C)(C)C=Cc2c(OC)c1", "class": "Chromene",
     "source": "Kotta et al. 2020; Okunade 2002"},
    {"name": "Ageratochromene", "smiles": "COc1cc2OC(C)(C)C=Cc2cc1OC", "class": "Chromene",
     "source": "Okunade 2002"},
    {"name": "6-Demethylageratochromene", "smiles": "COc1cc2OC(C)(C)C=Cc2cc1O", "class": "Chromene",
     "source": "Okunade 2002"},

    # Polymethoxyflavones (major bioactive flavonoids)
    {"name": "Sinensetin", "smiles": "COc1ccc(-c2cc(=O)c3c(OC)c(OC)c(OC)cc3o2)cc1OC", "class": "Flavonoid",
     "source": "Kotta et al. 2020"},
    {"name": "Nobiletin", "smiles": "COc1ccc(-c2cc(=O)c3c(OC)c(OC)c(OC)c(OC)c3o2)cc1OC", "class": "Flavonoid",
     "source": "Kotta et al. 2020"},
    {"name": "5'-Methoxynobiletin", "smiles": "COc1cc(-c2cc(=O)c3c(OC)c(OC)c(OC)c(OC)c3o2)cc(OC)c1OC", "class": "Flavonoid",
     "source": "Kotta et al. 2020"},
    {"name": "Eupalestin", "smiles": "COc1ccc(-c2cc(=O)c3c(OC)cc(OC)c(OC)c3o2)cc1OC", "class": "Flavonoid",
     "source": "Kotta et al. 2020"},
    {"name": "Linderoflavone B", "smiles": "COc1ccc(-c2cc(=O)c3c(OC)c(OC)c4c(c3o2)OCO4)cc1OC", "class": "Flavonoid",
     "source": "Okunade 2002"},
    {"name": "Kaempferol 3-glucoside", "smiles": "O=c1c(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12", "class": "Flavonoid",
     "source": "Kotta et al. 2020"},
    {"name": "Quercetin 3-rhamnoside", "smiles": "C[C@@H]1OC(Oc2c(-c3ccc(O)c(O)c3)oc4cc(O)cc(O)c4c2=O)[C@H](O)[C@H](O)[C@H]1O", "class": "Flavonoid",
     "source": "Kotta et al. 2020"},

    # Pyrrolizidine alkaloids
    {"name": "Lycopsamine", "smiles": "C[C@@H](O)[C@@](O)(CO)C(=O)OC[C@@H]1CCN2CC=C[C@H]12", "class": "Alkaloid",
     "source": "Wiedenfeld & Roder 1991; Okunade 2002"},
    {"name": "Echinatine", "smiles": "C[C@@H](O)[C@@](O)(CO)C(=O)OC[C@@H]1CCN2CC=C[C@H]12", "class": "Alkaloid",
     "source": "Okunade 2002"},

    # Terpenoids (from essential oil GC-MS studies)
    {"name": "beta-Caryophyllene", "smiles": "C=C1CC/C=C(\\C)CC[C@@H]2C[C@H]1C2(C)C", "class": "Sesquiterpene",
     "source": "Bosi et al. 2013; Kotta et al. 2020"},
    {"name": "alpha-Humulene", "smiles": "C/C1=C\\CC(/C=C/CC(/C)=C\\CC1)C", "class": "Sesquiterpene",
     "source": "Bosi et al. 2013"},
    {"name": "Germacrene D", "smiles": "C/C1=C\\C/C(=C(/C)\\CC/C(=C\\C1)/C)C(C)C", "class": "Sesquiterpene",
     "source": "Bosi et al. 2013"},
    {"name": "beta-Elemene", "smiles": "C=C(C)[C@@H]1CC=C(C)CC1C(=C)C", "class": "Sesquiterpene",
     "source": "Bosi et al. 2013"},
    {"name": "alpha-Pinene", "smiles": "CC1=CCC2CC1C2(C)C", "class": "Monoterpene",
     "source": "Bosi et al. 2013"},
    {"name": "beta-Pinene", "smiles": "C=C1CCC2CC1C2(C)C", "class": "Monoterpene",
     "source": "Bosi et al. 2013"},
    {"name": "Limonene", "smiles": "C=C(C)[C@@H]1CC=C(C)CC1", "class": "Monoterpene",
     "source": "Bosi et al. 2013"},
    {"name": "Linalool", "smiles": "C=CC(C)(O)CCC=C(C)C", "class": "Monoterpene",
     "source": "Bosi et al. 2013"},
    {"name": "Eugenol", "smiles": "COc1cc(CC=C)ccc1O", "class": "Phenylpropanoid",
     "source": "Bosi et al. 2013"},

    # Phenolic acids
    {"name": "p-Coumaric acid", "smiles": "OC(=O)/C=C/c1ccc(O)cc1", "class": "Phenolic acid",
     "source": "Kotta et al. 2020"},
    {"name": "Caffeic acid", "smiles": "OC(=O)/C=C/c1ccc(O)c(O)c1", "class": "Phenolic acid",
     "source": "Kotta et al. 2020"},
    {"name": "Ferulic acid", "smiles": "COc1cc(/C=C/C(O)=O)ccc1O", "class": "Phenolic acid",
     "source": "Kotta et al. 2020"},
    {"name": "Gallic acid", "smiles": "OC(=O)c1cc(O)c(O)c(O)c1", "class": "Phenolic acid",
     "source": "Kotta et al. 2020"},
    {"name": "Protocatechuic acid", "smiles": "OC(=O)c1ccc(O)c(O)c1", "class": "Phenolic acid",
     "source": "Kotta et al. 2020"},

    # Steroids / triterpenoids
    {"name": "Stigmasterol", "smiles": "CC[C@H](/C=C/[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(C)CC=C1[C@H]3CC(C)(C)CCC3=CC[C@@]12C)C(C)C", "class": "Steroid",
     "source": "Kotta et al. 2020"},
    {"name": "beta-Sitosterol", "smiles": "CC[C@H](CC[C@@H](C)[C@H]1CC[C@@H]2[C@@]1(C)CC=C1[C@H]3CC(C)(C)CCC3=CC[C@@]12C)C(C)C", "class": "Steroid",
     "source": "Kotta et al. 2020"},
    {"name": "Friedelin", "smiles": "CC1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(=O)C(C)(C)C5CCC43C)C2C1C", "class": "Triterpenoid",
     "source": "Okunade 2002"},

    # Coumarins
    {"name": "Coumarin", "smiles": "O=c1ccc2ccccc2o1", "class": "Coumarin",
     "source": "Okunade 2002"},
    {"name": "Agerarin", "smiles": "COc1cc2OC(=O)C=Cc2c(OC)c1OC", "class": "Coumarin",
     "source": "Okunade 2002"},
]


def screen_ageratum_compounds() -> pd.DataFrame:
    """
    Screen all curated A. conyzoides compounds through the NephroScreen model.
    Returns DataFrame with predictions and applicability domain assessment.
    """
    from .predict import predict_nephrotoxicity
    from .similarity import compute_applicability_domain

    results = []
    for compound in AGERATUM_COMPOUNDS:
        smiles = compound["smiles"]
        pred = predict_nephrotoxicity(smiles)

        if pred is None:
            results.append({
                "name": compound["name"],
                "smiles": smiles,
                "compound_class": compound["class"],
                "literature_source": compound["source"],
                "prediction": "Invalid SMILES",
                "probability_protective": None,
                "probability_toxic": None,
                "confidence": None,
                "in_domain": None,
                "max_similarity": None,
                "nearest_compound": None,
                "in_training_set": False,
            })
            continue

        domain = compute_applicability_domain(smiles)

        results.append({
            "name": compound["name"],
            "smiles": smiles,
            "compound_class": compound["class"],
            "literature_source": compound["source"],
            "prediction": pred["label"],
            "probability_protective": pred["probability_protective"],
            "probability_toxic": pred["probability_toxic"],
            "confidence": pred["confidence"],
            "in_domain": domain["in_domain"],
            "max_similarity": domain["max_similarity"],
            "nearest_compound": domain["nearest_compound"],
            "in_training_set": domain["max_similarity"] >= 0.99,
            "mw": pred["descriptors"]["MolWt"],
            "logp": pred["descriptors"]["LogP"],
        })

    df = pd.DataFrame(results)
    return df


def generate_ageratum_report(results_df: pd.DataFrame, output_dir: Path = None):
    """Generate screening report and publication-quality figure."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(PROCESSED_DIR / "ageratum_screening_results.csv", index=False)

    valid = results_df[results_df["prediction"] != "Invalid SMILES"].copy()
    if len(valid) == 0:
        logger.warning("No valid predictions for A. conyzoides compounds")
        return

    # Summary
    n_prot = (valid["prediction"] == "Nephroprotective").sum()
    n_tox = (valid["prediction"] == "Nephrotoxic").sum()
    n_domain = valid["in_domain"].sum()
    n_training = valid["in_training_set"].sum()

    logger.info(f"\nA. conyzoides Screening Results:")
    logger.info(f"  Total compounds: {len(valid)}")
    logger.info(f"  Nephroprotective: {n_prot}")
    logger.info(f"  Nephrotoxic: {n_tox}")
    logger.info(f"  In applicability domain: {n_domain}")
    logger.info(f"  Already in training set: {n_training}")

    # Publication figure: horizontal bar chart
    valid_sorted = valid.sort_values("probability_toxic", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(valid_sorted) * 0.35)))
    colors = ["#27AE60" if p == "Nephroprotective" else "#E74C3C"
              for p in valid_sorted["prediction"]]
    bars = ax.barh(
        range(len(valid_sorted)),
        valid_sorted["probability_toxic"],
        color=colors, alpha=0.85, height=0.7,
    )

    # Add compound class labels
    for i, (_, row) in enumerate(valid_sorted.iterrows()):
        label = f"{row['name']} ({row['compound_class']})"
        ax.text(-0.02, i, label, ha="right", va="center", fontsize=8)
        # Mark if outside applicability domain
        if not row.get("in_domain", True):
            ax.text(row["probability_toxic"] + 0.01, i, "*", fontsize=10, color="orange")

    ax.set_yticks([])
    ax.set_xlim(-0.01, 1.05)
    ax.set_xlabel("Probability of Nephrotoxicity", fontsize=11)
    ax.set_title(
        "NephroScreen Predictions for A. conyzoides Compounds\n"
        "(from published GC-MS/HPLC literature, not from thesis experimental data)",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Decision boundary")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27AE60", alpha=0.85, label=f"Nephroprotective (n={n_prot})"),
        Patch(facecolor="#E74C3C", alpha=0.85, label=f"Nephrotoxic (n={n_tox})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ageratum_screening.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'ageratum_screening.png'}")

    # Also save a summary by compound class
    class_summary = valid.groupby("compound_class").agg(
        count=("prediction", "count"),
        protective=("prediction", lambda x: (x == "Nephroprotective").sum()),
        toxic=("prediction", lambda x: (x == "Nephrotoxic").sum()),
        mean_prob_toxic=("probability_toxic", "mean"),
    ).round(3)
    logger.info(f"\nBy compound class:\n{class_summary}")


if __name__ == "__main__":
    results = screen_ageratum_compounds()
    generate_ageratum_report(results)
    print(f"\nScreened {len(results)} compounds")
    print(results[["name", "compound_class", "prediction", "confidence", "in_domain"]].to_string())
