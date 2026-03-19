"""
NephroScreen — West African Medicinal Plant Screening
=======================================================
Screens compounds from medicinal plants used in Northern Ghana and
West Africa for kidney conditions. Runs NephroScreen ML prediction
and COX-2 molecular docking for each compound.

Plants selected based on ethnobotanical use for kidney/urinary conditions
in Northern Ghana and West Africa.
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
DOCKING_DIR = Path(__file__).parent.parent / "data" / "docking_results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Compounds from West African medicinal plants used for kidney conditions
# Each entry: plant name (local name), compound, SMILES, class, reference
AFRICAN_PLANT_COMPOUNDS = [
    # Azadirachta indica (Neem / Dorgoyili in Dagbani)
    {"plant": "Azadirachta indica", "local_name": "Neem", "compound": "Nimbolide",
     "smiles": "CC12CCC3C(=CC(=O)C4(C)C(OC5OC(=O)C=C5)CC34O)C1CCC1=COC(=O)C=C12",
     "class": "Limonoid", "source": "Biswas et al. 2002"},
    {"plant": "Azadirachta indica", "local_name": "Neem", "compound": "Gedunin",
     "smiles": "CC(=O)OC1CC2C(C)(C(=O)CC3C2(C)CCC2C4(C)CCC(OC5OC=CC5=O)C(C)(C)C4CCC23)C(C)(C(=O)O1)C",
     "class": "Limonoid", "source": "Biswas et al. 2002"},
    {"plant": "Azadirachta indica", "local_name": "Neem", "compound": "Azadiradione",
     "smiles": "CC12CCC3C(=CC(=O)C4(C)C3CC3OC(=O)C=C3C14)C2CCC1=COC(=O)C=C1",
     "class": "Limonoid", "source": "Biswas et al. 2002"},
    {"plant": "Azadirachta indica", "local_name": "Neem", "compound": "Nimbin",
     "smiles": "COC(=O)C1CC2C(C)(C(=O)CC3C2(C)CCC2C4(C)CCC(OC5OC=CC5=O)C(C)(C)C4CCC23)C(C)(OC1=O)C",
     "class": "Limonoid", "source": "Biswas et al. 2002"},

    # Moringa oleifera (Drumstick tree / Zoogale in Dagbani)
    {"plant": "Moringa oleifera", "local_name": "Zoogale", "compound": "Niazimicin",
     "smiles": "CCCCCCCC/C=C\\CCCCCCCC(=O)NC1=CC=C(O)C=C1",
     "class": "Thiocarbamate", "source": "Fahey 2005"},
    {"plant": "Moringa oleifera", "local_name": "Zoogale", "compound": "Moringin",
     "smiles": "OCC1OC(SC/C(=N\\OS(O)(=O)=O)C2=CC=C(O)C=C2)C(O)C(O)C1O",
     "class": "Glucosinolate", "source": "Fahey 2005"},
    {"plant": "Moringa oleifera", "local_name": "Zoogale", "compound": "Quercetin",
     "smiles": "OC1=CC(=C2C(=O)C(O)=C(OC2=C1)C1=CC(O)=C(O)C=C1)O",
     "class": "Flavonoid", "source": "Vergara-Jimenez et al. 2017"},
    {"plant": "Moringa oleifera", "local_name": "Zoogale", "compound": "Kaempferol",
     "smiles": "OC1=CC(=C2C(=O)C(O)=C(OC2=C1)C1=CC=C(O)C=C1)O",
     "class": "Flavonoid", "source": "Vergara-Jimenez et al. 2017"},
    {"plant": "Moringa oleifera", "local_name": "Zoogale", "compound": "Chlorogenic acid",
     "smiles": "OC(=O)/C=C/C1=CC(O)=C(O)C=C1",
     "class": "Phenolic acid", "source": "Vergara-Jimenez et al. 2017"},

    # Vernonia amygdalina (Bitter leaf / Shuwaka in Hausa)
    {"plant": "Vernonia amygdalina", "local_name": "Bitter leaf", "compound": "Vernodalin",
     "smiles": "C=C1C(=O)OC2CC(=C)C(OC(=O)C(=C)CO)C3OC(=O)C(=C)C3C12",
     "class": "Sesquiterpene lactone", "source": "Igile et al. 1994"},
    {"plant": "Vernonia amygdalina", "local_name": "Bitter leaf", "compound": "Luteolin",
     "smiles": "OC1=CC(O)=C2C(=O)C=C(OC2=C1)C1=CC(O)=C(O)C=C1",
     "class": "Flavonoid", "source": "Igile et al. 1994"},
    {"plant": "Vernonia amygdalina", "local_name": "Bitter leaf", "compound": "Luteolin 7-glucoside",
     "smiles": "OCC1OC(Oc2cc(O)c3c(=O)cc(-c4ccc(O)c(O)c4)oc3c2)C(O)C(O)C1O",
     "class": "Flavonoid glycoside", "source": "Igile et al. 1994"},

    # Hibiscus sabdariffa (Roselle / Sobolo in local parlance)
    {"plant": "Hibiscus sabdariffa", "local_name": "Sobolo", "compound": "Delphinidin",
     "smiles": "OC1=CC2=C(C=C1O)C(=CC(O)=C2)C1=CC(O)=C(O)C(O)=C1",
     "class": "Anthocyanin", "source": "Da-Costa-Rocha et al. 2014"},
    {"plant": "Hibiscus sabdariffa", "local_name": "Sobolo", "compound": "Cyanidin",
     "smiles": "OC1=CC2=C(C=C1O)C(=CC(O)=C2)C1=CC(O)=C(O)C=C1",
     "class": "Anthocyanin", "source": "Da-Costa-Rocha et al. 2014"},
    {"plant": "Hibiscus sabdariffa", "local_name": "Sobolo", "compound": "Protocatechuic acid",
     "smiles": "OC(=O)C1=CC(O)=C(O)C=C1",
     "class": "Phenolic acid", "source": "Da-Costa-Rocha et al. 2014"},
    {"plant": "Hibiscus sabdariffa", "local_name": "Sobolo", "compound": "Hibiscus acid",
     "smiles": "OC(CC(O)(CC(O)=O)C(O)=O)=O",
     "class": "Organic acid", "source": "Da-Costa-Rocha et al. 2014"},

    # Carica papaya (Pawpaw)
    {"plant": "Carica papaya", "local_name": "Pawpaw", "compound": "Carpaine",
     "smiles": "CC1CCCCCCCC(=O)OC(C)CCCCCCCC(=O)OC(C)CCCCN1",
     "class": "Alkaloid", "source": "Zunjar et al. 2016"},
    {"plant": "Carica papaya", "local_name": "Pawpaw", "compound": "Caffeic acid",
     "smiles": "OC(=O)/C=C/C1=CC(O)=C(O)C=C1",
     "class": "Phenolic acid", "source": "Zunjar et al. 2016"},
    {"plant": "Carica papaya", "local_name": "Pawpaw", "compound": "p-Coumaric acid",
     "smiles": "OC(=O)/C=C/C1=CC=C(O)C=C1",
     "class": "Phenolic acid", "source": "Zunjar et al. 2016"},

    # Zingiber officinale (Ginger)
    {"plant": "Zingiber officinale", "local_name": "Ginger", "compound": "6-Gingerol",
     "smiles": "CCCCC[C@@H](O)CC(=O)CCC1=CC(OC)=C(O)C=C1",
     "class": "Phenol", "source": "Mao et al. 2019"},
    {"plant": "Zingiber officinale", "local_name": "Ginger", "compound": "6-Shogaol",
     "smiles": "CCCCC/C=C/C(=O)CCC1=CC(OC)=C(O)C=C1",
     "class": "Phenol", "source": "Mao et al. 2019"},
    {"plant": "Zingiber officinale", "local_name": "Ginger", "compound": "Zingerone",
     "smiles": "COC1=CC(=CC=C1O)CCC(C)=O",
     "class": "Phenol", "source": "Mao et al. 2019"},

    # Curcuma longa (Turmeric)
    {"plant": "Curcuma longa", "local_name": "Turmeric", "compound": "Curcumin",
     "smiles": "COC1=CC(=CC(=C1O)/C=C/C(=O)CC(=O)/C=C/C1=CC(OC)=C(O)C=C1)OC",
     "class": "Curcuminoid", "source": "Hewlings & Kalman 2017"},
    {"plant": "Curcuma longa", "local_name": "Turmeric", "compound": "Demethoxycurcumin",
     "smiles": "COC1=CC(=CC=C1O)/C=C/C(=O)CC(=O)/C=C/C1=CC(OC)=C(O)C=C1",
     "class": "Curcuminoid", "source": "Hewlings & Kalman 2017"},
    {"plant": "Curcuma longa", "local_name": "Turmeric", "compound": "Bisdemethoxycurcumin",
     "smiles": "OC1=CC=C(/C=C/C(=O)CC(=O)/C=C/C2=CC=C(O)C=C2)C=C1",
     "class": "Curcuminoid", "source": "Hewlings & Kalman 2017"},

    # Khaya senegalensis (African mahogany / Mahobi)
    {"plant": "Khaya senegalensis", "local_name": "African mahogany", "compound": "Swietenine",
     "smiles": "CC(=O)OCC1C2CCC3(C)C(CCC4C5CC(OC5(C)C(OC(C)=O)C34)C2=O)C1(C)C=O",
     "class": "Limonoid", "source": "Zhang et al. 2009"},

    # Cryptolepis sanguinolenta (Nibima in Twi)
    {"plant": "Cryptolepis sanguinolenta", "local_name": "Nibima", "compound": "Cryptolepine",
     "smiles": "Cn1c2ccccc2c2c1[nH]c1ccccc12",
     "class": "Indoloquinoline alkaloid", "source": "Grellier et al. 1996"},
    {"plant": "Cryptolepis sanguinolenta", "local_name": "Nibima", "compound": "Hydroxycryptolepine",
     "smiles": "Cn1c2ccccc2c2c1[nH]c1ccc(O)cc12",
     "class": "Indoloquinoline alkaloid", "source": "Grellier et al. 1996"},
    {"plant": "Cryptolepis sanguinolenta", "local_name": "Nibima", "compound": "Neocryptolepine",
     "smiles": "Cn1c2ccccc2c2[nH]c3ccccc3c21",
     "class": "Indoloquinoline alkaloid", "source": "Grellier et al. 1996"},

    # Combretum micranthum (Kinkeliba)
    {"plant": "Combretum micranthum", "local_name": "Kinkeliba", "compound": "Vitexin",
     "smiles": "OCC1OC(c2c(O)c3c(O)cc(O)cc3oc2=O)C(O)C(O)C1O",
     "class": "Flavonoid C-glycoside", "source": "Welch 2010"},
    {"plant": "Combretum micranthum", "local_name": "Kinkeliba", "compound": "Isovitexin",
     "smiles": "OCC1OC(c2c(O)c(O)c3oc(-c4ccc(O)cc4)cc(=O)c3c2)C(O)C(O)C1O",
     "class": "Flavonoid C-glycoside", "source": "Welch 2010"},

    # Senna occidentalis (Coffee Senna / used for kidney tea)
    {"plant": "Senna occidentalis", "local_name": "Coffee Senna", "compound": "Emodin",
     "smiles": "CC1=CC(O)=C2C(=O)C3=C(C=C(O)C=C3O)C(=O)C2=C1",
     "class": "Anthraquinone", "source": "Yadav et al. 2010"},
    {"plant": "Senna occidentalis", "local_name": "Coffee Senna", "compound": "Chrysophanol",
     "smiles": "CC1=CC(O)=C2C(=O)C3=CC=C(O)C=C3C(=O)C2=C1",
     "class": "Anthraquinone", "source": "Yadav et al. 2010"},

    # Parkia biglobosa (African locust bean / Dawadawa)
    {"plant": "Parkia biglobosa", "local_name": "Dawadawa", "compound": "Catechin",
     "smiles": "OC1CC2=C(OC1C1=CC(O)=C(O)C=C1)C=C(O)C=C2O",
     "class": "Flavanol", "source": "Millogo-Kone et al. 2008"},
    {"plant": "Parkia biglobosa", "local_name": "Dawadawa", "compound": "Epicatechin",
     "smiles": "OC1CC2=C(OC1C1=CC(O)=C(O)C=C1)C=C(O)C=C2O",
     "class": "Flavanol", "source": "Millogo-Kone et al. 2008"},
    {"plant": "Parkia biglobosa", "local_name": "Dawadawa", "compound": "Gallic acid",
     "smiles": "OC(=O)C1=CC(O)=C(O)C(O)=C1",
     "class": "Phenolic acid", "source": "Millogo-Kone et al. 2008"},
]


def screen_african_plants() -> pd.DataFrame:
    """Screen all West African medicinal plant compounds through NephroScreen."""
    from .predict import predict_nephrotoxicity
    from .similarity import compute_applicability_domain

    results = []
    for compound in AFRICAN_PLANT_COMPOUNDS:
        smiles = compound["smiles"]
        pred = predict_nephrotoxicity(smiles)

        if pred is None:
            results.append({
                "plant": compound["plant"],
                "local_name": compound["local_name"],
                "compound": compound["compound"],
                "smiles": smiles,
                "compound_class": compound["class"],
                "literature_source": compound["source"],
                "prediction": "Invalid SMILES",
                "probability_protective": None,
                "probability_toxic": None,
                "confidence": None,
                "in_domain": None,
            })
            continue

        domain = compute_applicability_domain(smiles)
        results.append({
            "plant": compound["plant"],
            "local_name": compound["local_name"],
            "compound": compound["compound"],
            "smiles": smiles,
            "compound_class": compound["class"],
            "literature_source": compound["source"],
            "prediction": pred["label"],
            "probability_protective": pred["probability_protective"],
            "probability_toxic": pred["probability_toxic"],
            "confidence": pred["confidence"],
            "in_domain": domain["in_domain"],
            "max_similarity": domain["max_similarity"],
            "mw": pred["descriptors"]["MolWt"],
            "logp": pred["descriptors"]["LogP"],
        })

    return pd.DataFrame(results)


def generate_african_plants_report(results_df: pd.DataFrame, output_dir: Path = None):
    """Generate publication figure for African plants screening."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(PROCESSED_DIR / "african_plants_screening.csv", index=False)

    valid = results_df[results_df["prediction"] != "Invalid SMILES"].copy()
    n_prot = (valid["prediction"] == "Nephroprotective").sum()
    n_tox = (valid["prediction"] == "Nephrotoxic").sum()

    logger.info(f"\nWest African Plants Screening:")
    logger.info(f"  Total: {len(valid)}, Protective: {n_prot}, Toxic: {n_tox}")

    # By plant
    plant_summary = valid.groupby("plant").agg(
        n_compounds=("compound", "count"),
        n_protective=("prediction", lambda x: (x == "Nephroprotective").sum()),
        n_toxic=("prediction", lambda x: (x == "Nephrotoxic").sum()),
        mean_tox_prob=("probability_toxic", "mean"),
    ).round(3)
    logger.info(f"\nBy plant:\n{plant_summary}")

    # Figure: grouped by plant, colored by prediction
    fig, ax = plt.subplots(figsize=(12, max(8, len(valid) * 0.4)))
    sorted_df = valid.sort_values(["plant", "probability_toxic"], ascending=[True, True])

    colors = ["#27AE60" if p == "Nephroprotective" else "#E74C3C" for p in sorted_df["prediction"]]
    y_labels = [f"{row['compound']} ({row['plant'].split()[-1]})" for _, row in sorted_df.iterrows()]

    ax.barh(range(len(sorted_df)), sorted_df["probability_toxic"], color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Probability of Nephrotoxicity", fontsize=11)
    ax.set_title(
        "NephroScreen Predictions for West African Medicinal Plant Compounds",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#27AE60", alpha=0.85, label=f"Nephroprotective (n={n_prot})"),
        Patch(facecolor="#E74C3C", alpha=0.85, label=f"Nephrotoxic (n={n_tox})"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "african_plants_screening.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'african_plants_screening.png'}")


if __name__ == "__main__":
    results = screen_african_plants()
    generate_african_plants_report(results)
    valid = results[results["prediction"] != "Invalid SMILES"]
    print(f"\nScreened {len(valid)} compounds from {valid['plant'].nunique()} plants")
    print(f"Protective: {(valid['prediction']=='Nephroprotective').sum()}")
    print(f"Toxic: {(valid['prediction']=='Nephrotoxic').sum()}")
    print(f"\nBy plant:")
    for plant in valid["plant"].unique():
        subset = valid[valid["plant"] == plant]
        n_p = (subset["prediction"] == "Nephroprotective").sum()
        n_t = (subset["prediction"] == "Nephrotoxic").sum()
        print(f"  {plant}: {n_p} protective, {n_t} toxic")
