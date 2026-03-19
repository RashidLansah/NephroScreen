"""
NephroScreen Data Collection Pipeline
======================================
Collects nephrotoxicity/nephroprotection bioactivity data from:
1. ChEMBL database (kidney cell viability assays, renal biomarker assays)
2. Curated literature lists of known nephrotoxicants and nephroprotectants
3. DrugBank-annotated nephrotoxic compounds

All SMILES are validated and canonicalized via RDKit before inclusion.
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ---------------------------------------------------------------------------
# 1. Curated Literature Compounds
# ---------------------------------------------------------------------------

# Known nephroprotective compounds from peer-reviewed literature
KNOWN_NEPHROPROTECTIVE = {
    # Reference drugs / well-established nephroprotectants
    "silymarin": "COc1cc(ccc1O)[C@@H]1Oc2cc([C@@H]3Oc4cc(O)cc(O)c4C(=O)[C@H]3O)ccc2O[C@H]1CO",
    "N-acetylcysteine": "CC(=O)N[C@@H](CS)C(O)=O",
    "quercetin": "OC1=CC(=C2C(=O)C(O)=C(OC2=C1)C1=CC(O)=C(O)C=C1)O",
    "kaempferol": "OC1=CC(=C2C(=O)C(O)=C(OC2=C1)C1=CC=C(O)C=C1)O",
    "curcumin": "COC1=CC(=CC(=C1O)/C=C/C(=O)CC(=O)/C=C/C1=CC(OC)=C(O)C=C1)OC",
    "epigallocatechin gallate": "OC1=CC(=C2C[C@@H](OC(=O)C3=CC(O)=C(O)C(O)=C3)[C@@H](O)OC2=C1)O",
    "resveratrol": "OC1=CC=C(C=C1)/C=C/C1=CC(O)=CC(O)=C1",
    "alpha-lipoic acid": "OC(=O)CCCCC1CCSS1",
    "astaxanthin": "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)C(=O)[C@@H](O)CC2(C)C)C(C)(C)C[C@H](O)C1=O",
    "berberine": "COc1ccc2CC3c4cc5OCOc5cc4CC=N3Cc2c1OC",
    "luteolin": "OC1=CC(O)=C2C(=O)C=C(OC2=C1)C1=CC(O)=C(O)C=C1",
    "apigenin": "OC1=CC(O)=C2C(=O)C=C(OC2=C1)C1=CC=C(O)C=C1",
    "catechin": "OC1CC2=C(O[C@H]1C1=CC(O)=C(O)C=C1)C=C(O)C=C2O",
    "rutin": "O[C@@H]1[C@@H](O)[C@H](OC[C@H]1O)OC[C@H]1OC(OC2=C(OC3=CC(O)=CC(O)=C3C2=O)C2=CC(O)=C(O)C=C2)[C@@H](O)[C@@H](O)[C@@H]1O",
    "ellagic acid": "OC1=CC2=C3C(=C1O)OC(=O)C1=CC(O)=C(O)C=C1C3=C1C=C(O)C(O)=CC1=C2",
    "naringenin": "OC1=CC=C(C=C1)[C@H]1CC(=O)C2=C(O1)C=C(O)C=C2O",
    "hesperidin": "COC1=CC=C(C=C1O)[C@H]1CC(=O)C2=C(O)C=C(O[C@@H]3O[C@H](CO[C@@H]4O[C@@H](C)[C@H](O)[C@@H](O)[C@H]4O)[C@@H](O)[C@H](O)[C@H]3O)C=C2O1",
    "gallic acid": "OC(=O)C1=CC(O)=C(O)C(O)=C1",
    "ferulic acid": "COC1=CC(=CC=C1O)/C=C/C(O)=O",
    "caffeic acid": "OC(=O)/C=C/C1=CC(O)=C(O)C=C1",
    "carnosic acid": "CC(C)C1=CC2=C(C(O)=C1O)[C@@]1(CCCC(C)(C)[C@@H]1CC2)C(O)=O",
    "thymoquinone": "CC1=CC(=O)C(C(C)C)=CC1=O",
    "lycopene": "CC(=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C=C(/C)\\C=C\\C=C(/C)\\C=C\\C=C(\\C)C)C",
    "melatonin": "COC1=CC2=C(NC=C2CCNC(C)=O)C=C1",
    "taurine": "NCCS(O)(=O)=O",
    "vitamin_e_alpha_tocopherol": "CC1=C(C)C2=C(CC[C@@](C)(CCC[C@H](C)CCC[C@H](C)CCCC(C)C)O2)C(C)=C1O",
    "coenzyme_q10_ubiquinone": "COC1=C(OC)C(=O)C(C/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)CC/C=C(\\C)C)=C(C)C1=O",
    "sulforaphane": "CS(=O)CCCCN=C=S",
    "bardoxolone_methyl": "COC(=O)[C@@]12CC[C@]3(C)[C@@H](CC(=O)[C@H]3[C@@H]1CC(=O)C1=CC(=O)C=C[C@@]12C)C1(C)CCC(=O)C(C)(C)C1C#N",
    "diosmin": "COC1=CC(=CC=C1O)/C=C/C1=CC(=CC2=C1C(=O)C=C(O2)C1=CC(O[C@@H]2O[C@H](CO[C@@H]3O[C@@H](C)[C@H](O)[C@@H](O)[C@H]3O)[C@@H](O)[C@H](O)[C@H]2O)=C(O)C=C1)O",
    "zingerone": "COC1=CC(=CC=C1O)CCC(C)=O",
    "allicin": "C=CCSS(=O)CC=C",
    "l_carnitine": "C[N+](C)(C)C[C@H](O)CC([O-])=O",
    "erythropoietin_mimetic_fgf23_pathway": None,  # Protein, skip
    "pentoxifylline": "Cn1c(=O)c2c(ncn2CCCCC(C)=O)n(C)c1=O",
    "glycyrrhizin": "O[C@@H]1[C@@H](O)[C@H](O[C@@H]([C@@H]1O)C(O)=O)OC1CC[C@@]2(C)[C@H](CC[C@]3(C)[C@@H]2C(=O)C=C2[C@@H]4C[C@](C)(CC[C@]4(C)CC[C@]23C)C(O)=O)C1(C)C",
    "silibinin": "COc1cc(ccc1O)[C@@H]1Oc2cc([C@@H]3Oc4cc(O)cc(O)c4C(=O)[C@H]3O)ccc2O[C@H]1CO",
    "oleanolic acid": "C[C@@H]1CC[C@@]2(CC[C@@]3(C)[C@H](CC=C4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@@]43C)[C@@H]2[C@H]1C)C(O)=O",
    "ursolic acid": "C[C@H]1CC[C@@]2(CC[C@@]3(C)[C@H](CC=C4[C@@]5(C)CC[C@H](O)C(C)(C)[C@@H]5CC[C@@]43C)[C@@H]2[C@@H]1C)C(O)=O",
    "beta_carotene": "CC1=C(/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)CCCC2(C)C)C(C)(C)CCC1",
    "selenium_selenomethionine": "C[Se]CC[C@H](N)C(O)=O",
    "baicalein": "OC1=C(O)C(O)=C2C(=O)C=C(OC2=C1)C1=CC=CC=C1",
    "wogonin": "COC1=C(O)C=C(O)C2=C1OC(=CC2=O)C1=CC=CC=C1",
    "chrysin": "OC1=CC(O)=C2C(=O)C=C(OC2=C1)C1=CC=CC=C1",
    "genistein": "OC1=CC=C(C=C1)C1=COC2=CC(O)=CC(O)=C2C1=O",
    "daidzein": "OC1=CC=C(C=C1)C1=COC2=CC(O)=CC=C2C1=O",
    "myricetin": "OC1=CC(O)=C2C(=O)C(O)=C(OC2=C1)C1=CC(O)=C(O)C(O)=C1",
    "chlorogenic acid": "OC(=O)/C=C/C1=CC(O)=C(O)C=C1",
    "rosmarinic acid": "OC(=O)[C@@H](CC1=CC(O)=C(O)C=C1)OC(=O)/C=C/C1=CC(O)=C(O)C=C1",
    "sinapic acid": "COC1=CC(=CC(OC)=C1O)/C=C/C(O)=O",
    "vanillic acid": "COC1=CC(=CC=C1O)C(O)=O",
    "p_coumaric acid": "OC(=O)/C=C/C1=CC=C(O)C=C1",
    "carvacrol": "CC1=CC=C(C(C)C)C(O)=C1",
    "thymol": "CC1=CC(O)=C(C(C)C)C=C1",
    "eugenol": "COC1=CC(CC=C)=CC=C1O",
    "mangiferin": "OC1=CC2=C(C(=C1)O)C(=O)C1=C(O2)[C@H]2OC(CO)[C@@H](O)[C@@H](O)[C@@H]2OC1=C1C=C(O)C(O)=C(O)C=C1",
}

# Known nephrotoxic compounds from literature
KNOWN_NEPHROTOXIC = {
    # Aminoglycoside antibiotics
    "gentamicin": "CC(C(=O)N[C@H]1[C@@H](O[C@@H]([C@@H]([C@H]1O)O)O[C@H]1[C@H](O[C@H]([C@@H]([C@H]1N)O)O[C@@H]1[C@@H](CC(N)=O)OC([C@@H]1O)O)N)N)NC",
    "tobramycin": "NC[C@@H]1OC(O[C@@H]2[C@@H](N)C[C@@H](N)[C@H](O[C@H]3OC(CO)[C@@H](O)[C@H](N)[C@H]3O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
    "amikacin": "NCC[C@@H](O)C(=O)N[C@@H]1C[C@H](N)[C@@H](O[C@H]2OC(CO)[C@@H](O)[C@H](N)[C@H]2O)[C@H](O)[C@@H]1O[C@@H]1OC[C@H](O)[C@@H](O)[C@H]1O",
    "neomycin": "NC[C@@H]1OC(O[C@@H]2[C@@H](N)C[C@@H](N)[C@H](O[C@H]3OC(CO)[C@@H](O[C@H]4OC(CO)[C@@H](O)[C@H](N)[C@H]4O)[C@H](O)[C@H]3N)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O",
    "streptomycin": "NC1=NC(NCCN1)NC1OC(O)[C@H](O)[C@@H](O)[C@@H]1OC1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1NC",
    "kanamycin": "NC[C@@H]1OC(O[C@@H]2[C@@H](O)[C@H](O[C@H]3OC(CO)[C@@H](O)[C@H](N)[C@H]3O)[C@@H](N)C[C@@H]2N)[C@H](O)[C@@H](O)[C@@H]1O",
    # Platinum-based chemotherapy
    "cisplatin": "[NH3][Pt]([NH3])(Cl)Cl",
    "carboplatin": "O=C1O[Pt](OC1=O)([NH3])[NH3]",
    "oxaliplatin": "O=C1O[Pt]2(OC1=O)[NH2][C@@H]1CCCC[C@H]1[NH2]2",
    # Calcineurin inhibitors
    "cyclosporine": "CC[C@@H]1NC(=O)[C@H](CC)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@@H](C)NC(=O)[C@@H](C)NC(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](CC(C)C)N(C)C(=O)[C@H](/C=C/C)N(C)C(=O)C(C)(C)N(C)C1=O",
    "tacrolimus": "CO[C@H]1C[C@@H](CC(=O)[C@@H](CC=C)\\C=C(/C)[C@@H](O[C@H]2OC(C)[C@@H](O)[C@](O2)(O)C)[C@@H](C)C(=O)C(=O)N2CCCC[C@@H]2C(=O)O1)OC",
    # NSAIDs
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(O)=O",
    "indomethacin": "COC1=CC2=C(C=C1)C(CC(O)=O)=C(C)N2C(=O)C1=CC=C(Cl)C=C1",
    "diclofenac": "OC(=O)CC1=CC=CC=C1NC1=CC=CC(Cl)=C1Cl",
    "piroxicam": "OC1=C2C=CC=CC2=S(=O)(=O)N(C)C1=C(O)NC1=CC=CC=N1",
    "celecoxib": "CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F",
    "naproxen": "COC1=CC2=CC(=CC=C2C=C1)C(C)C(O)=O",
    "meloxicam": "CN1C(=C(O)C2=CC=CC=C2S1(=O)=O)C(=O)NC1=NC=CS1",
    "ketorolac": "OC(=O)C1CCN2C1=CC=C2C(=O)C1=CC=CC=C1",
    # Antifungals
    "amphotericin_b": "O[C@H]1C=CC=CC=CC=CC=CC=CC(=O)O[C@@H](C[C@H](O)C[C@H](O)CC(=O)C[C@H](O)[C@H](O)C[C@H](O)C[C@H](O[C@H]2O[C@H](C)[C@@H](O)[C@H](N)[C@@H]2O)CC(O)=C1C)CC",
    # ACE inhibitors / ARBs (at high doses or specific conditions)
    "vancomycin": "O[C@@H]1[C@@H](O)[C@H](OC1CO)OC1=CC=C(C=C1Cl)C1OC2=CC3=CC(OC4=CC=C(C=C4Cl)[C@H](O)[C@@H]4NC(=O)[C@H](NC(=O)[C@@H]5CC(O)=CC=C5C5=C(O)C=C(O)C(=C5)[C@@H](NC4=O)C(O)=O)C(=O)N[C@@H](CC(N)=O)C(=O)N[C@H]4C(=O)N[C@H](C(=O)N[C@@H]1C(=O)N4)C(C)(C)C)=CC(O)=C3C(=C2)O",
    # Radiocontrast agents
    "iohexol": "OCC(O)CNC(=O)C1=C(I)C(=C(I)C(=C1I)C(=O)NCC(O)CO)N(C)C(C)=O",
    "iopamidol": "OCC(O)CNC(=O)C1=C(I)C(NC(=O)C(O)C)=C(I)C(=C1I)C(=O)NCC(O)CO",
    # Aristolochic acid (herbal nephrotoxin)
    "aristolochic_acid_I": "COC1=CC2=C(C=C1)C1=CC3=C(OCO3)C(=C1C(=N2)C(O)=O)[N+]([O-])=O",
    "aristolochic_acid_II": "OC(=O)C1=NC2=CC3=C(C=C2C2=C1C=C1OCO1C=C2[N+]([O-])=O)C=CC=C3",
    # Heavy metals (as organic compounds or common salts)
    "mercuric_chloride": "[Hg](Cl)Cl",
    "cadmium_chloride": "[Cl-].[Cl-].[Cd+2]",
    "lead_acetate": "CC(=O)[O-].CC(=O)[O-].[Pb+2]",
    # Other nephrotoxicants
    "lithium_carbonate": "[Li+].[Li+].[O-]C([O-])=O",
    "methotrexate": "CN(CC1=CN=C2N=C(N)N=C(N2)C1=C1C=CC(=CC=1)C(=O)N[C@@H](CCC(O)=O)C(O)=O)C1=CC=C(N)C=C1",
    "tenofovir": "NC1=NC=NC2=C1N=CN2[C@@H](CO)COCP(O)(O)=O",
    "cidofovir": "NC1=NC(=O)N(C=C1)C[C@H](CO)OCP(O)(O)=O",
    "adefovir": "NC1=C2N=CN(CCOCP(O)(O)=O)C2=NC=N1",
    "foscarnet": "OP(=O)(O)C(O)=O",
    "pentamidine": "N=C(N)C1=CC=C(OCCCCCOC2=CC=C(C=C2)C(N)=N)C=C1",
    "rifampin": "COC1=C2C3=C(C(=C1/C=N/N1CCN(CC1)C)O)C(O)=C(NC(=O)/C(=C/C=C/[C@H](OC(C)=O)[C@@H](C)/C=C/C(C)[C@H](O)/C(=C\\C1=CC(=C(O)C2=C1O3)C)C)C)C",
    "colistin": "CC(C)CCCC(=O)N[C@@H](CCN)C(=O)N[C@H]1CCNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CCN)NC(=O)[C@@H](CCN)NC(=O)[C@@H](CC(C)C)NC1=O)CC(C)C",
    "polymyxin_b": "CC(C)CCCC(=O)N[C@@H](CCN)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)N[C@H]1CCNC(=O)[C@@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](CCN)NC(=O)[C@@H](CCN)NC(=O)[C@@H](CC(C)C)NC1=O)CC(C)C",
    "paraquat": "C[N+]1=CC=C(C=C1)C1=CC=[N+](C)C=C1",
    "carbon_tetrachloride": "ClC(Cl)(Cl)Cl",
    "ochratoxin_a": "OC(=O)[C@@H](CC1=CC=CC=C1)NC(=O)C1=CC(=CC2=C1C(=O)OC2C)Cl",
    "zoledronic_acid": "OC(P(O)(O)=O)(P(O)(O)=O)CN1C=CN=C1",
    "pamidronate": "NCCC(O)(P(O)(O)=O)P(O)(O)=O",
    "acetaminophen_overdose": "CC(=O)NC1=CC=C(O)C=C1",  # Toxic at high doses
    "phenacetin": "CCOC1=CC=C(NC(C)=O)C=C1",
    "5_aminosalicylic_acid": "NC1=CC=C(O)C(=C1)C(O)=O",
    "gold_sodium_thiomalate": None,  # Complex salt, skip
    "penicillamine": "CC(C)(S)[C@@H](N)C(O)=O",
    "mitomycin_c": "CO[C@@]12[C@H](COC(N)=O)C3=C(C(=C(N)C=C3)C)N1C[C@H]1N[C@@H]12",
    "doxorubicin": "COC1=CC=CC2=C1C(=O)C1=C(O)C3=C(C[C@](O)(C[C@@H]3O[C@H]3C[C@H](N)[C@H](O)[C@H](C)O3)C(=O)CO)C(O)=C1C2=O",
    "ifosfamide": "O=P1(NCCCl)OCCCN1CCCl",
    "streptozotocin": "O=NN(C)C(=O)N[C@H]1[C@@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O",
    "carmustine": "O=NN(CCCl)C(=O)NCCCl",
    "lomustine": "O=NN(CCCl)C(=O)NC1CCCCC1",
    "sirolimus": "CO[C@H]1C[C@@H](CC(=O)[C@@H](/C=C(/C)[C@@H](O[C@@H]2OC(C)[C@@H](O)[C@@](O2)(O)C)[C@H](C)C[C@H]2CC[C@@H](O)[C@@H](O2)CC(=O)[C@@H](/C=C/C=C/C=C/C)[C@@H](OC)C(=O)[C@H](C)C[C@@H](C)C(=O)N2CCCC[C@@H]2C(=O)O1)C)OC",
    "acyclovir_high_dose": "NC1=NC2=C(N=C1)N(COCCO)C=N2",
    "indinavir": "CC(C)(C)NC(=O)[C@@H]1CN(CC2=CC=CC=C2)CCN1C[C@@H](O)C[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H]1C(=O)N2CCC[C@H]2C(=O)NC(C)(C)C",
    "ethylene_glycol": "OCCO",
    "melamine": "NC1=NC(N)=NC(N)=N1",
    "puromycin": "CN(C)C1=NC=NC2=C1N=CN2[C@@H]1O[C@H](CO[C@H]2OC[C@H](N)[C@@H]2O)[C@@H](O)[C@H]1O",
}

# Additional nephroprotective compounds from more literature sources
ADDITIONAL_NEPHROPROTECTIVE = {
    "lisinopril": "NCCCC[C@H](N[C@@H](CCC(O)=O)C(O)=O)C(=O)N1CCC[C@H]1C(O)=O",
    "losartan": "CCCCC1=NC(Cl)=C(N1CC1=CC=C(C=C1)C1=CC=CC=C1C1=NN=N[NH]1)CO",
    "enalapril": "CCOC(=O)[C@H](CCC1=CC=CC=C1)N[C@@H](C)C(=O)N1CCC[C@H]1C(O)=O",
    "telmisartan": "CCCC1=NC2=C(C=CC(=C2)C2=CC=CC=C2C2=NN=N[NH]2)N1CC1=CC=C(C=C1)C(O)=O",
    "irbesartan": "CCCCC1=NC2=CC=CC=C2N1CC1=CC=C(C=C1)C1=CC=CC=C1C1=NCCN1",
    "pioglitazone": "CCC1=CC=C(CCOC2=CC=C(CC3SC(=O)NC3=O)C=C2)NC=1",
    "erythropoietin": None,  # Protein, skip
    "allopurinol": "O=C1N=CN=C2NN=CC1=2",
    "febuxostat": "CC1=C(OCC(C)C)C(=CC(=C1)C#N)C1=CC=C(C=C1)C(O)=O",
    "nifedipine": "COC(=O)C1=C(C)NC(C)=C(C1C1=CC=CC=C1[N+]([O-])=O)C(=O)OC",
    "amlodipine": "CCOC(=O)C1=C(COCCN)NC(C)=C(C1C1=CC=CC=C1Cl)C(=O)OC",
    "atorvastatin": "CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CCC(O)C[C@@H](O)CC(O)=O)C1=CC=C(F)C=C1)C1=CC=CC=C1",
    "rosuvastatin": "CC(C1=NC(N(C)S(=O)(=O)C)=NC(=C1/C=C/[C@@H](O)C[C@@H](O)CC(O)=O)C1=CC=C(F)C=C1)C",
    "carvedilol": "COC1=CC=CC=C1OCCNC[C@H](O)COC1=CC=CC2=C1[NH]C1=CC=CC=C21",
    "edaravone": "CC1=NN(C(=O)C1)C1=CC=CC=C1",
    "tempol": "CC1(C)CC(CC(C)(C)N1[O])O",
    "bilirubin": "CC1=C(/C=C\\2/NC(=O)C(C=C)=C2C)C(=O)NC1=C\\C1=NC(=C(CC)C1C)/C=C1\\NC(=O)C(C)=C1CCC(O)=O",
    "hydrogen_sulfide_donor_nahs": "[SH-].[Na+]",
    "sodium_bicarbonate": "[Na+].OC([O-])=O",
    "dimethyl_fumarate": "COC(=O)/C=C/C(=O)OC",
    "spironolactone": "C[C@@H]1C[C@H]2[C@@H]3CC[C@]4(C)[C@H](CC[C@]4(OC(C)=O)C(=O)SCc4ccccc4)[C@@H]3CC[C@]2(C)C[C@@H]1O",
}

# Additional nephrotoxic from more lit
ADDITIONAL_NEPHROTOXIC = {
    "mercury_methylmercury": "C[Hg]Cl",
    "uranium_uranyl_acetate": None,  # Skip
    "chromium_vi_potassium_dichromate": None,  # Inorganic
    "sodium_fluoride": "[Na+].[F-]",
    "ferric_nitrilotriacetate": None,  # Complex
    "d_serine_high_dose": "N[C@H](CO)C(O)=O",
    "cephaloridine": "CC1=CS[C@@H]2[C@H](NC(=O)CC3=CC=CS3)C(=O)N12",
    "cephalothin": "CC(=O)OCC1=C(N2[C@@H](SC1)[C@@H](NC(=O)CC1=CC=CS1)C2=O)C(O)=O",
    "polymixin_e_colistin_methanesulfonate": None,  # Complex
    "mannitol_high_dose": "OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
    "dextran_high_dose": None,  # Polymer
    "hydroxyethyl_starch": None,  # Polymer
    "lithium_chloride": "[Li+].[Cl-]",
    "sevoflurane": "OC(F)(F)C(F)OC(F)(F)F",
    "methoxyflurane": "COC(F)(F)C(Cl)Cl",
    "dapsone": "NC1=CC=C(C=C1)S(=O)(=O)C1=CC=C(N)C=C1",
    "sulfonamide_sulfadiazine": "NC1=CC=C(C=C1)S(=O)(=O)NC1=CC=NC=N1",
    "trimethoprim": "COC1=CC(=CC(OC)=C1OC)CC1=CN=C(N)N=C1N",
    "ciprofloxacin_crystalluria": "OC(=O)C1=CN(C2CC2)C2=CC(N3CCNCC3)=C(F)C=C2C1=O",
    "metformin_lactic_acidosis": "CN(C)C(=N)NC(N)=N",
    "orlistat": "CCCCCCCCCCCCC[C@@H](OC(=O)[C@H]1C(=O)O[C@@H]1CCCCCC)C[C@H](O)CCCCCCC",
    "mesalazine_5asa": "NC1=CC=C(O)C(=C1)C(O)=O",
    "valacyclovir": "CC(C)[C@@H](N)C(=O)OCCOCN1C=NC2=C1N=C(N)NC2=O",
    "atazanavir": "COC(=O)N[C@@H](C(=O)N[C@H](C[C@@H](O)[C@H](CC1=CC=CC=C1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)CC1=CC=CC=C1)C(C)(C)C",
    "ritonavir": "CC(C)[C@H](NC(=O)N(C)CC1=CSC(=N1)C(C)C)C(=O)N[C@H](C[C@@H](O)[C@H](CC1=CC=CC=C1)NC(=O)OCC1=CN=CS1)CC1=CC=CC=C1",
    "gadolinium_contrast": None,  # Complex chelate
    "hydrochlorothiazide": "NS(=O)(=O)C1=CC2=C(NCNS2(=O)=O)C=C1Cl",
    "furosemide": "NS(=O)(=O)C1=CC(C(O)=O)=CC(NCC2=CC=CO2)=C1Cl",
    "triamterene": "NC1=NC2=NC(N)=NC(N)=C2N=C1C1=CC=CC=C1",
    "topiramate": "OC[C@@]1(OS(N)(=O)=O)OC2OC3(CCCCC3C)O[C@@H]2[C@@H]1O",
}


def validate_smiles(smiles: str) -> Optional[str]:
    """Validate and canonicalize a SMILES string using RDKit."""
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def compile_literature_compounds() -> pd.DataFrame:
    """
    Compile all curated literature compounds into a labeled DataFrame.
    Each compound is validated via RDKit before inclusion.
    """
    records = []

    # Process nephroprotective compounds
    all_protective = {**KNOWN_NEPHROPROTECTIVE, **ADDITIONAL_NEPHROPROTECTIVE}
    for name, smiles in all_protective.items():
        canonical = validate_smiles(smiles)
        if canonical:
            records.append({
                "compound_name": name.replace("_", " ").title(),
                "smiles": canonical,
                "label": 0,  # 0 = nephroprotective
                "source": "literature_curated",
                "label_description": "nephroprotective",
            })

    # Process nephrotoxic compounds
    all_toxic = {**KNOWN_NEPHROTOXIC, **ADDITIONAL_NEPHROTOXIC}
    for name, smiles in all_toxic.items():
        canonical = validate_smiles(smiles)
        if canonical:
            records.append({
                "compound_name": name.replace("_", " ").title(),
                "label": 1,  # 1 = nephrotoxic
                "smiles": canonical,
                "source": "literature_curated",
                "label_description": "nephrotoxic",
            })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["smiles"], keep="first")
    logger.info(
        f"Literature compounds: {len(df)} total "
        f"({(df['label'] == 0).sum()} protective, {(df['label'] == 1).sum()} toxic)"
    )
    return df


# ---------------------------------------------------------------------------
# 2. ChEMBL Data Collection
# ---------------------------------------------------------------------------

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def query_chembl_assays(search_term: str, limit: int = 100) -> list[dict]:
    """Query ChEMBL for assays matching a search term."""
    url = f"{CHEMBL_API_BASE}/assay/search.json"
    params = {"q": search_term, "limit": limit, "format": "json"}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("assays", [])
    except Exception as e:
        logger.warning(f"ChEMBL assay search failed for '{search_term}': {e}")
        return []


def query_chembl_activities(assay_id: str, limit: int = 500) -> list[dict]:
    """Get bioactivity data for a specific ChEMBL assay."""
    url = f"{CHEMBL_API_BASE}/activity.json"
    params = {
        "assay_chembl_id": assay_id,
        "limit": limit,
        "format": "json",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("activities", [])
    except Exception as e:
        logger.warning(f"ChEMBL activity fetch failed for {assay_id}: {e}")
        return []


def get_compound_smiles(chembl_id: str) -> Optional[str]:
    """Fetch canonical SMILES for a ChEMBL compound ID."""
    url = f"{CHEMBL_API_BASE}/molecule/{chembl_id}.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        structs = data.get("molecule_structures")
        if structs and structs.get("canonical_smiles"):
            return validate_smiles(structs["canonical_smiles"])
    except Exception:
        pass
    return None


def collect_chembl_kidney_data() -> pd.DataFrame:
    """
    Query ChEMBL for kidney-related bioactivity data.
    Searches for assays involving kidney cell lines and renal biomarkers.
    """
    search_terms = [
        "nephrotoxicity",
        "nephrotoxic",
        "renal toxicity",
        "kidney toxicity",
        "HEK293 cell viability",
        "renal tubular",
        "RPTEC",
        "kidney cell",
        "renal cell cytotoxicity",
        "nephroprotective",
        "kidney injury",
        "creatinine clearance",
        "BUN",
        "KIM-1",
        "NGAL",
    ]

    all_assay_ids = set()
    for term in search_terms:
        logger.info(f"Searching ChEMBL for: {term}")
        assays = query_chembl_assays(term, limit=50)
        for assay in assays:
            aid = assay.get("assay_chembl_id")
            if aid:
                all_assay_ids.add(aid)
        time.sleep(0.5)  # Rate limiting

    logger.info(f"Found {len(all_assay_ids)} unique assays from ChEMBL")

    records = []
    for i, assay_id in enumerate(all_assay_ids):
        if i % 10 == 0:
            logger.info(f"Processing assay {i + 1}/{len(all_assay_ids)}")
        activities = query_chembl_activities(assay_id, limit=200)
        for act in activities:
            mol_id = act.get("molecule_chembl_id")
            smiles = act.get("canonical_smiles")
            if not smiles:
                continue
            canonical = validate_smiles(smiles)
            if not canonical:
                continue

            # Determine label from activity type and value
            standard_type = act.get("standard_type", "")
            standard_value = act.get("standard_value")
            standard_units = act.get("standard_units", "")
            activity_comment = act.get("activity_comment", "") or ""

            label = classify_chembl_activity(
                standard_type, standard_value, standard_units, activity_comment
            )
            if label is not None:
                records.append({
                    "compound_name": act.get("molecule_pref_name", mol_id),
                    "smiles": canonical,
                    "label": label,
                    "source": f"chembl_{assay_id}",
                    "label_description": "nephroprotective" if label == 0 else "nephrotoxic",
                    "chembl_id": mol_id,
                    "assay_id": assay_id,
                    "activity_type": standard_type,
                    "activity_value": standard_value,
                    "activity_units": standard_units,
                })
        time.sleep(0.3)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["smiles"], keep="first")
    logger.info(
        f"ChEMBL compounds: {len(df)} total "
        f"({(df['label'] == 0).sum() if len(df) > 0 else 0} protective, "
        f"{(df['label'] == 1).sum() if len(df) > 0 else 0} toxic)"
    )
    return df


def classify_chembl_activity(
    standard_type: str,
    standard_value: Optional[str],
    standard_units: str,
    activity_comment: str,
) -> Optional[int]:
    """
    Classify a ChEMBL activity record as nephroprotective (0) or nephrotoxic (1).

    Classification heuristics:
    - IC50/EC50 for cell viability: low values = toxic, high values = protective/inactive
    - Inhibition %: high inhibition of kidney cells = toxic
    - Activity comments mentioning toxicity/protection
    """
    comment_lower = activity_comment.lower() if activity_comment else ""

    # Check activity comments first
    toxic_keywords = ["toxic", "nephrotoxic", "cytotoxic", "cell death", "apoptosis", "necrosis"]
    protective_keywords = ["protective", "nephroprotective", "cytoprotective", "antioxidant"]

    if any(kw in comment_lower for kw in toxic_keywords):
        return 1
    if any(kw in comment_lower for kw in protective_keywords):
        return 0

    # Classify by activity value
    if standard_value is None:
        return None

    try:
        value = float(standard_value)
    except (ValueError, TypeError):
        return None

    upper_type = standard_type.upper() if standard_type else ""

    # IC50 for cytotoxicity: < 10 uM is toxic, > 100 uM may be non-toxic
    if upper_type in ("IC50", "EC50", "GI50", "CC50") and standard_units in ("nM", "uM"):
        if standard_units == "nM":
            value_um = value / 1000
        else:
            value_um = value

        if value_um < 10:
            return 1  # Toxic at low concentration
        elif value_um > 100:
            return 0  # Not toxic at reasonable concentrations

    # Inhibition percentage
    if upper_type == "INHIBITION" and standard_units == "%":
        if value > 50:
            return 1  # >50% inhibition = toxic
        elif value < 20:
            return 0  # <20% inhibition = not significantly toxic

    return None


# ---------------------------------------------------------------------------
# 3. PubChem Supplement
# ---------------------------------------------------------------------------

def query_pubchem_nephrotoxic_list() -> pd.DataFrame:
    """
    Fetch compounds from PubChem classified in nephrotoxicity-related assays.
    Uses PubChem PUG REST API to search for compounds tested in kidney toxicity assays.
    """
    records = []

    # Search PubChem for compounds annotated with nephrotoxicity
    search_terms = [
        "nephrotoxicity",
        "renal toxicity",
        "kidney damage",
    ]

    for term in search_terms:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{term}/property/CanonicalSMILES,IUPACName/JSON"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                props = data.get("PropertyTable", {}).get("Properties", [])
                for p in props:
                    smiles = validate_smiles(p.get("CanonicalSMILES", ""))
                    if smiles:
                        records.append({
                            "compound_name": p.get("IUPACName", "Unknown"),
                            "smiles": smiles,
                            "label": 1,
                            "source": "pubchem",
                            "label_description": "nephrotoxic",
                        })
        except Exception as e:
            logger.warning(f"PubChem query failed for '{term}': {e}")
        time.sleep(0.5)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["smiles"], keep="first")
    logger.info(f"PubChem compounds: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# 4. Main Pipeline
# ---------------------------------------------------------------------------

def run_data_collection(skip_chembl: bool = False) -> pd.DataFrame:
    """
    Run the full data collection pipeline:
    1. Compile curated literature compounds
    2. Query ChEMBL for kidney-related bioactivity data
    3. Supplement from PubChem
    4. Merge, deduplicate, and save

    Args:
        skip_chembl: If True, skip the slow ChEMBL queries and use only
                     literature-curated compounds. Useful for fast iteration.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Literature compounds
    logger.info("=" * 60)
    logger.info("Step 1: Compiling literature-curated compounds")
    logger.info("=" * 60)
    lit_df = compile_literature_compounds()
    lit_df.to_csv(RAW_DIR / "literature_compounds.csv", index=False)

    # Step 2: ChEMBL data
    if not skip_chembl:
        logger.info("=" * 60)
        logger.info("Step 2: Querying ChEMBL database")
        logger.info("=" * 60)
        chembl_df = collect_chembl_kidney_data()
        chembl_df.to_csv(RAW_DIR / "chembl_compounds.csv", index=False)
    else:
        logger.info("Skipping ChEMBL queries (skip_chembl=True)")
        chembl_df = pd.DataFrame()

    # Step 3: PubChem supplement
    logger.info("=" * 60)
    logger.info("Step 3: Querying PubChem")
    logger.info("=" * 60)
    pubchem_df = query_pubchem_nephrotoxic_list()
    if len(pubchem_df) > 0:
        pubchem_df.to_csv(RAW_DIR / "pubchem_compounds.csv", index=False)

    # Step 4: Merge and deduplicate
    logger.info("=" * 60)
    logger.info("Step 4: Merging and deduplicating")
    logger.info("=" * 60)
    dfs = [lit_df]
    if len(chembl_df) > 0:
        dfs.append(chembl_df[["compound_name", "smiles", "label", "source", "label_description"]])
    if len(pubchem_df) > 0:
        dfs.append(pubchem_df)

    combined = pd.concat(dfs, ignore_index=True)

    # Deduplicate by SMILES, keeping the first occurrence (literature takes priority)
    combined = combined.drop_duplicates(subset=["smiles"], keep="first")

    # Remove any rows with missing SMILES
    combined = combined.dropna(subset=["smiles"])

    # Filter out very small molecules (MW < 50) and very large ones (MW > 2000)
    # These are unlikely to be relevant drug-like compounds
    valid_rows = []
    for _, row in combined.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if 50 < mw < 2000:
                valid_rows.append(row)
    combined = pd.DataFrame(valid_rows)

    # Save final dataset
    combined.to_csv(PROCESSED_DIR / "nephroscreen_dataset.csv", index=False)

    # Print summary
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total compounds: {len(combined)}")
    logger.info(f"Nephroprotective (label=0): {(combined['label'] == 0).sum()}")
    logger.info(f"Nephrotoxic (label=1): {(combined['label'] == 1).sum()}")
    logger.info(f"Class ratio: {(combined['label'] == 1).sum() / max((combined['label'] == 0).sum(), 1):.2f}")
    logger.info(f"Sources: {combined['source'].value_counts().to_dict()}")
    logger.info(f"Saved to: {PROCESSED_DIR / 'nephroscreen_dataset.csv'}")

    return combined


if __name__ == "__main__":
    # For initial build, start with literature compounds + ChEMBL
    # Set skip_chembl=True for faster testing
    df = run_data_collection(skip_chembl=False)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"\nClass distribution:\n{df['label_description'].value_counts()}")
    print(f"\nSource distribution:\n{df['source'].value_counts()}")
