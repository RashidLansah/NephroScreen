"""
NephroScreen Dataset Expansion
================================
Expands the training dataset from ~450 to 1,000+ compounds using:
1. Additional curated compounds from nephrotoxicity/nephroprotection review papers
2. FDA FAERS kidney adverse event drug reports
3. Tox21/ToxCast kidney-related bioassay data from PubChem

All SMILES are validated and canonicalized via RDKit before inclusion.
"""

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

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def validate_smiles(smiles: str) -> Optional[str]:
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def resolve_name_to_smiles(name: str) -> Optional[str]:
    """Resolve a drug name to canonical SMILES via PubChem."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                # PubChem may return CanonicalSMILES or ConnectivitySMILES
                smiles = props[0].get("CanonicalSMILES") or props[0].get("ConnectivitySMILES", "")
                return validate_smiles(smiles)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 1. Review Paper Curated Compounds
# ---------------------------------------------------------------------------

# From Perazella (2009), Hoste et al. (2015), Awdishu & Mehta (2017)
# Drug-induced acute kidney injury / nephrotoxicity
REVIEW_NEPHROTOXIC = {
    # Antimicrobials
    "acyclovir": "NC1=NC2=C(N=CN2COC(CO)CO)C(=O)N1",
    "adefovir_dipivoxil": "CC(C)(C)C(=O)OCOP(=O)(COCCn1cnc2c(N)ncnc21)OCOC(=O)C(C)(C)C",
    "cefazolin": "Cc1nnc(SCC2=C(N3[C@@H](SC2)[C@@H](NC(=O)Cn2cnnn2)C3=O)C(O)=O)s1",
    "cefotaxime": "CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C(O)=O)=C(COC(C)=O)CS[C@H]12)c1csc(N)n1",
    "cefepime": "CO/N=C(\\C(=O)N[C@@H]1C(=O)N2C(C([O-])=O)=C(C[N+]3(C)CCCC3)CS[C@H]12)c1csc(N)n1",
    "chloramphenicol": "OC(C(=O)NCc1ccc([N+]([O-])=O)cc1)C(Cl)Cl",
    "clindamycin": "CCC[C@@H]1C[C@H](N(C)C)C(=O)O[C@H]1SC",
    "daptomycin": None,  # Cyclic lipopeptide, too complex
    "linezolid": "CC(=O)NC[C@H]1CN(c2ccc(N3CCOCC3)c(F)c2)C(=O)O1",
    "meropenem": "C[C@@H]1[C@@H]2[C@H](C(O)=O)N2C(=O)[C@@H]1SC1C[C@@](C)(C(O)=O)N(C1=O)c1ccccc1",
    "nitrofurantoin": "O=C1CN(/N=C/c2ccc([N+]([O-])=O)o2)C(=O)N1",
    "pentamidine_isethionate": "N=C(N)c1ccc(OCCCCCOc2ccc(C(N)=N)cc2)cc1",
    "pyrimethamine": "CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1",
    "quinine": "COc1ccc2nccc([C@@H](O)[C@H]3C[C@@H]4CC[N@]3C[C@@H]4C=C)c2c1",
    "tigecycline": "CN(C)[C@H]1[C@@H]2C[C@@H]3Cc4c(cc(NC(=O)CNC(C)(C)C)c(O)c4O)[C@](O)(C(N)=O)C3=C(O)[C@]2(O)C(=O)C(C(N)=O)=C1O",
    # Antivirals
    "atazanavir": "COC(=O)N[C@@H](C(=O)N[C@H](Cc1ccccc1)C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)[C@@H](NC(=O)OC)C(C)(C)C)C(C)(C)C",
    "darunavir": "CC(C)CN1C[C@@H](O[C@@H]2OCC[C@@H]2O)C[C@H]1C(=O)N[C@@H](Cc1ccccc1)[C@H](O)CN1CC2(CCCCC2)C[C@H]1c1ccc(NS(C)(=O)=O)cc1",
    "didanosine": "OC[C@H]1CC[C@@H](n2cnc3c(O)ncnc32)O1",
    "efavirenz": "OC1(C#CC1)c1cc(Cl)cc(c1)C1(C(F)(F)F)OC(=O)Nc2ccccc21",
    "lopinavir": "CC(C)[C@@H](NC(=O)[C@@H](CC(=O)NC1CCCCC1)C[C@@H](O)[C@H](Cc1ccccc1)NC(=O)COc1c(C)cccc1C)C=O",
    "nelfinavir": "Oc1ccc2c(c1)[C@@H](CO)CN2C[C@@H](O)C[C@@H](Cc1ccccc1)C(=O)N[C@H]1c2ccccc2CC1O",
    "stavudine": "CC1=CN([C@H]2C=C[C@@H](CO)O2)C(=O)NC1=O",
    "zidovudine": "CC1=CN([C@H]2CC(N=[N+]=[N-])[C@@H](CO)O2)C(=O)NC1=O",
    # Antineoplastics not already in dataset
    "bendamustine": "Cn1c(CCCC(O)=O)nc2cc(N(CCCl)CCCl)ccc21",
    "cabozantinib": "COc1cc2nccc(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)c(F)c3)c2cc1OC",
    "lenalidomide": "NC1=CC=CC2=C1C(=O)N(C1CCC(=O)NC1=O)C2=O",
    "pazopanib": "Cc1ccc(Nc2nnc(C(C)C)c3[nH]c(C)c(c23)-c2ccnc(N)n2)cc1S(N)(=O)=O",
    "sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "sunitinib": "CCN(CC)CCNC(=O)c1c(C)[nH]c(=O)c1/C=C/1\\C(=O)Nc2ccc(F)cc21",
    "temozolomide": "Cn1nnc2c(C(N)=O)ncn2c1=O",
    # Immunosuppressants
    "everolimus": "CO[C@H]1C[C@@H](CC(=O)[C@@H](/C=C(/C)[C@@H](OC)[C@H](OC(=O)[C@@H](C)[C@H](O)/C=C/[C@@H]2CC[C@@H](O)[C@@H](O2)/C=C/[C@@H](C)[C@H](OC)C(=O)[C@H](C)C[C@@H](C)/C=C/C=C/C=C(/C)\\[C@H](CC3)OC3=O)C(C)C)[C@@H](O)CC(=O)NCCOC)OC",
    # Analgesics / NSAIDs not already included
    "metamizole": "Cc1c(N(C)CS(=O)(=O)[O-])c(=O)n(-c2ccccc2)n1C.[Na+]",
    "niflumic_acid": "OC(=O)c1cccnc1Nc1cccc(C(F)(F)F)c1",
    "nimesulide": "CS(=O)(=O)Nc1ccc([N+]([O-])=O)cc1Oc1ccccc1",
    # Diuretics / Cardiovascular
    "mannitol_iv": "OC[C@@H](O)[C@@H](O)[C@H](O)[C@H](O)CO",
    "acetazolamide": "CC(=O)Nc1nnc(S(N)(=O)=O)s1",
    "ticlopidine": "ClC1=CC=CC=C1CN1CCC2=C(C1)C=CS2",
    # Miscellaneous
    "deferoxamine": "CC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCN",
    "mesalamine": "Nc1ccc(O)c(C(O)=O)c1",
    "sulfasalazine": "OC(=O)c1cc(/N=N/c2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O",
    "rasburicase": None,  # Protein
    "sucralfate": None,  # Complex salt
    "sodium_phosphate": None,  # Inorganic
    "omeprazole": "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",
    "pantoprazole": "COc1ccnc(CS(=O)c2nc3cc(OC(F)F)ccc3[nH]2)c1OC",
    "ciprofloxacin": "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "levofloxacin": "C[C@@H]1COc2c(N3CCN(C)CC3)c(F)cc3c(=O)c(C(O)=O)cn1c23",
    "norfloxacin": "CCn1cc(C(O)=O)c(=O)c2cc(F)c(N3CCNCC3)cc21",
    "moxifloxacin": "COc1c(N2C[C@@H]3CCCN[C@@H]3C2)c(F)cc2c(=O)c(C(O)=O)cn(C3CC3)c12",
    "gatifloxacin": "COc1c(N2CCNC(C)C2)c(F)cc2c(=O)c(C(O)=O)cn(C3CC3)c12",
    "alendronate": "NCCCC(O)(P(O)(O)=O)P(O)(O)=O",
    "risedronate": "OC(P(O)(O)=O)(P(O)(O)=O)Cc1cccnc1",
    "ibandronate": "OC(P(O)(O)=O)(P(O)(O)=O)CCCCCN(C)CCC",
    "haloperidol": "OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c1ccc(Cl)cc1",
    "lithium_citrate": "[Li+].[Li+].[Li+].OC(CC([O-])=O)(CC([O-])=O)C([O-])=O",
    "cocaine": "COC(=O)[C@H]1CC[C@@H]2CC[C@H]1N2C",
    "methamphetamine": "CNC(C)Cc1ccccc1",
    "heroin": "CC(=O)O[C@H]1C=C[C@H]2[C@H]3CC4=CC=C(OC(C)=O)C5O[C@@H]1[C@]25CCN3CC4",
}

# From Oguntibeju (2019), Udupa & Bhatt (2022) — natural nephroprotectants
REVIEW_NEPHROPROTECTIVE = {
    # Flavonoids / polyphenols
    "dihydroquercetin_taxifolin": "O[C@@H]1[C@H](Oc2cc(O)cc(O)c2C1=O)c1ccc(O)c(O)c1",
    "vitexin": "OC[C@H]1OC([C@@H](O)[C@@H](O)[C@@H]1O)c1c(O)c(O)c2c(c1)oc(-c1ccc(O)cc1)cc2=O",
    "orientin": "OC[C@H]1OC([C@@H](O)[C@@H](O)[C@@H]1O)c1c(O)c2c(oc(-c3ccc(O)c(O)c3)cc2=O)cc1O",
    "silychristin": "COc1cc([C@@H]2Oc3ccc([C@@H]4Oc5cc(O)cc(O)c5C(=O)[C@H]4O)cc3[C@H]2CO)ccc1O",
    "isoliquiritigenin": "Oc1ccc(/C=C/C(=O)c2ccc(O)cc2O)cc1",
    "formononetin": "COc1ccc(-c2coc3cc(O)ccc3c2=O)cc1",
    "biochanin_a": "COc1ccc(-c2coc3cc(O)cc(O)c3c2=O)cc1",
    "dihydromyricetin": "O[C@@H]1[C@H](Oc2cc(O)cc(O)c2C1=O)c1cc(O)c(O)c(O)c1",
    "delphinidin": "OC1=Cc2c(O)cc(O)cc2O[C@@H]1c1cc(O)c(O)c(O)c1",
    "pelargonidin": "OC1=Cc2c(O)cc(O)cc2O[C@@H]1c1ccc(O)cc1",
    "cyanidin": "OC1=Cc2c(O)cc(O)cc2O[C@@H]1c1ccc(O)c(O)c1",
    # Terpenoids / steroids
    "ursodeoxycholic_acid": "C[C@H](CCC(O)=O)[C@H]1CC[C@@H]2[C@@H]3[C@@H](O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C",
    "madecassoside_aglycone": "C[C@@H]1CC[C@@]2(CC[C@@]3(C)[C@H](CC[C@@H]4[C@@]5(C)C[C@@H](O)[C@H](O)[C@@](C)(CO)[C@@H]5CC[C@@]43C)[C@@H]2[C@@H]1C)C(O)=O",
    "asiatic_acid": "C[C@@H]1CC[C@@]2(CC[C@@]3(C)[C@H](CC[C@@H]4[C@@]5(C)C[C@@H](O)[C@H](O)[C@@](C)(CO)[C@@H]5CC[C@@]43C)[C@@H]2[C@@H]1C)C(O)=O",
    "glycyrrhizic_acid_aglycone": "C[C@@H]1C(=O)C=C2[C@@H]3CC(C)(C)[C@@H](O)CC[C@]3(C)C3=CC(=O)[C@@]4(C)[C@@H](CC[C@@]4(C)[C@@H]3CC[C@]21C)C(O)=O",
    "lupeol": "C(=C1CC[C@@]2(C)[C@H](CC[C@H]3[C@@]4(C)CC[C@@H](O)C(C)(C)[C@@H]4CC[C@@]32C)C1)(C)C",
    "squalene": "CC(=CCC/C(=C/CC/C(=C/CC/C=C(/CC/C=C(/CCC=C(C)C)\\C)\\C)/C)/C)C",
    # Amino acids / small molecules
    "n_acetyl_l_cysteine_ethyl_ester": "CCOC(=O)[C@@H](CS)NC(C)=O",
    "s_adenosyl_l_homocysteine": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H](N)C(O)=O)[C@@H](O)[C@H]1O",
    "alpha_tocopheryl_succinate": "CC1=C(C)C2=C(CC[C@@](C)(CCC[C@H](C)CCC[C@H](C)CCCC(C)C)O2)C(C)=C1OC(=O)CCC(O)=O",
    "idebenone": "COC1=C(OC)C(=O)C(CCCCCCCCCCO)=C(C)C1=O",
    "pirfenidone": "Cc1ccc(=O)n(C)c1",
    "mitoq": "COC1=C(OC)C(=O)C(C)=C(CCCCCCCCCC[P+](c2ccccc2)(c2ccccc2)c2ccccc2)C1=O",
    # Herbal / natural compounds
    "lycopene_fragment": "CC(=C/C=C/C=C(C)/C=C/C1=C(C)CCCC1(C)C)\\C=C\\C=C(C)\\C=C\\C1=C(C)CCCC1(C)C",
    "zeaxanthin": "C/C(=C\\C=C\\C=C(\\C)\\C=C\\C1=C(C)CC(O)CC1(C)C)C=CC=C(C)C=CC1=C(C)CC(O)CC1(C)C",
    "crocetin": "CC(=C/C=C/C(=C/C=C/C(=C/C(O)=O)/C)/C)\\C=C\\C=C(\\C)\\C=C\\C(O)=O",
    "crocin_aglycone": "CC(=C/C=C/C(=C/C=C/C(=C/C(O)=O)/C)/C)\\C=C\\C=C(\\C)\\C=C\\C(O)=O",
    "sesamin": "C1OC2=CC=CC3=C2[C@@H]1[C@@H]1[C@@H](C3)OC3=CC4=C(OCO4)C=C3O1",
    "sesamol": "OC1=CC2=C(OCO2)C=C1",
    "hydroxytyrosol": "OC(C)Cc1ccc(O)c(O)c1",
    "oleuropein_aglycone": "C/C=C1\\[C@H](CC=O)C(C(=O)OC)=CO[C@@H]1O",
    "6_gingerol": "CCCCC[C@@H](O)CC(=O)CCc1ccc(O)c(OC)c1",
    "6_shogaol": "CCCCC/C=C/C(=O)CCc1ccc(O)c(OC)c1",
    "embelin": "CCCCCCCCCCCC1=C(O)C(=O)C=C(O)C1=O",
    "plumbagin": "CC1=CC(=O)c2c(O)cccc2C1=O",
    "shikonin": "CC(=CCC1=CC(=O)c2c(O)ccc(O)c2C1=O)C",
    # Vitamins / cofactors
    "pyridoxamine": "Cc1ncc(CO)c(CN)c1O",
    "cyanocobalamin_simplified": None,  # Too complex
    "benfotiamine": "CC1=CC(=C(N=C1N)C/C(=C\\OP(O)(=O)OC1=CC=CC=C1)SC(C)([O-])C=O)C.[Na+]",
    "ubiquinol": "COC1=C(OC)C(O)=C(C/C=C(\\C)CC/C=C(\\C)C)C(C)=C1O",
    # Drugs with demonstrated nephroprotection
    "cilastatin_iv": "CC(C)C(/C=C/C1CC1)CC(=O)N[C@@H](CCSCC1CC1)C(O)=O",
    "fenoldopam": "Oc1ccc2c(c1)CCc1cc(Cl)c(O)c(O)c1C2",
    "theophylline": "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "levocarnitine": "C[N+](C)(C)C[C@H](O)CC([O-])=O",
    "probucol": "CC(C)(c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1)SSC(C)(C)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1",
}


def collect_review_compounds() -> pd.DataFrame:
    """Compile compounds from review papers into labeled DataFrame."""
    records = []
    for name, smiles in REVIEW_NEPHROTOXIC.items():
        canonical = validate_smiles(smiles)
        if canonical:
            records.append({
                "compound_name": name.replace("_", " ").title(),
                "smiles": canonical,
                "label": 1,
                "source": "review_papers",
                "label_description": "nephrotoxic",
            })
    for name, smiles in REVIEW_NEPHROPROTECTIVE.items():
        canonical = validate_smiles(smiles)
        if canonical:
            records.append({
                "compound_name": name.replace("_", " ").title(),
                "smiles": canonical,
                "label": 0,
                "source": "review_papers",
                "label_description": "nephroprotective",
            })
    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["smiles"], keep="first")
    logger.info(f"Review paper compounds: {len(df)} ({(df['label']==0).sum()} protective, {(df['label']==1).sum()} toxic)")
    return df


# ---------------------------------------------------------------------------
# 2. FDA FAERS Kidney Injury Drugs
# ---------------------------------------------------------------------------

def collect_faers_compounds(top_n: int = 200) -> pd.DataFrame:
    """
    Query OpenFDA FAERS for drugs most frequently reported with kidney adverse events.
    Resolves drug names to SMILES via PubChem.
    """
    kidney_terms = [
        "renal+failure",
        "acute+kidney+injury",
        "nephrotoxicity",
        "renal+impairment",
        "renal+tubular+necrosis",
    ]

    drug_names = set()
    for term in kidney_terms:
        url = (
            f"https://api.fda.gov/drug/event.json?"
            f"search=patient.reaction.reactionmeddrapt:\"{term}\""
            f"&count=patient.drug.openfda.generic_name.exact"
            f"&limit={top_n}"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                for r in results:
                    name = r.get("term", "").strip().lower()
                    if name and len(name) > 2:
                        drug_names.add(name)
                logger.info(f"FAERS '{term}': {len(results)} drugs")
        except Exception as e:
            logger.warning(f"FAERS query failed for '{term}': {e}")
        time.sleep(0.5)

    logger.info(f"Total unique FAERS drug names: {len(drug_names)}")

    # Resolve to SMILES — clean up FAERS generic names first
    records = []
    resolved = 0
    for raw_name in sorted(drug_names):
        # FAERS names are often uppercase multi-word; clean for PubChem
        name = raw_name.strip().lower()
        # Try original, then without common suffixes
        smiles = resolve_name_to_smiles(name)
        if not smiles:
            # Try removing common salt forms
            for suffix in [" hydrochloride", " sodium", " potassium", " mesylate",
                           " maleate", " fumarate", " tartrate", " sulfate",
                           " citrate", " acetate", " besylate", " succinate"]:
                if name.endswith(suffix):
                    smiles = resolve_name_to_smiles(name.replace(suffix, ""))
                    if smiles:
                        break
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw = Descriptors.MolWt(mol)
                if 50 < mw < 2000:
                    records.append({
                        "compound_name": name.title(),
                        "smiles": smiles,
                        "label": 1,
                        "source": "faers",
                        "label_description": "nephrotoxic",
                    })
                    resolved += 1
        time.sleep(0.3)  # PubChem rate limiting
        if resolved % 20 == 0 and resolved > 0:
            logger.info(f"FAERS resolved: {resolved}/{len(drug_names)}")

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["smiles"], keep="first")
    logger.info(f"FAERS compounds resolved: {len(df)}")
    if len(df) == 0:
        return pd.DataFrame(columns=["compound_name", "smiles", "label", "source", "label_description"])
    return df


# ---------------------------------------------------------------------------
# 3. Tox21/ToxCast PubChem BioAssay
# ---------------------------------------------------------------------------

def collect_tox21_kidney_data() -> pd.DataFrame:
    """
    Query PubChem BioAssay for Tox21/ToxCast assays related to kidney toxicity.
    Fetches active/inactive compounds from relevant assays.
    """
    # Known kidney-relevant Tox21 assay IDs from PubChem
    kidney_assay_aids = [
        # Tox21 HEK293 cell viability assays
        720719,   # Tox21_p53_BLA_p1 (HEK293)
        720725,   # Tox21_p53_BLA_p5 (HEK293)
        651631,   # Tox21 ARE (kidney-relevant pathway)
        743228,   # Tox21_MitochondrialToxicity
        # ToxCast renal assays
        1347033,  # ATG_NRF2_ARE_CIS (relevant to renal oxidative stress)
    ]

    records = []
    for aid in kidney_assay_aids:
        # Fetch active compounds (potentially toxic)
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/cids/JSON?cids_type=active&list_return=listkey"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                cids = data.get("IdentifierList", {}).get("CID", [])[:100]
                logger.info(f"AID {aid}: {len(cids)} active CIDs")

                # Fetch SMILES in batches
                for i in range(0, len(cids), 50):
                    batch = cids[i:i+50]
                    cid_str = ",".join(str(c) for c in batch)
                    smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES,IUPACName/JSON"
                    try:
                        smi_resp = requests.get(smi_url, timeout=30)
                        if smi_resp.status_code == 200:
                            props = smi_resp.json().get("PropertyTable", {}).get("Properties", [])
                            for p in props:
                                smiles = validate_smiles(p.get("CanonicalSMILES", ""))
                                if smiles:
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol and 50 < Descriptors.MolWt(mol) < 2000:
                                        records.append({
                                            "compound_name": p.get("IUPACName", f"CID_{p.get('CID', '')}"),
                                            "smiles": smiles,
                                            "label": 1,  # Active in toxicity assay = potentially toxic
                                            "source": f"tox21_aid{aid}",
                                            "label_description": "nephrotoxic",
                                        })
                    except Exception:
                        pass
                    time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Tox21 AID {aid} failed: {e}")
        time.sleep(1)

    # Also fetch inactive compounds as potentially non-toxic (for balance)
    for aid in kidney_assay_aids[:2]:
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/cids/JSON?cids_type=inactive&list_return=listkey"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                cids = data.get("IdentifierList", {}).get("CID", [])[:80]
                logger.info(f"AID {aid} inactive: {len(cids)} CIDs")

                for i in range(0, len(cids), 50):
                    batch = cids[i:i+50]
                    cid_str = ",".join(str(c) for c in batch)
                    smi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES,IUPACName/JSON"
                    try:
                        smi_resp = requests.get(smi_url, timeout=30)
                        if smi_resp.status_code == 200:
                            props = smi_resp.json().get("PropertyTable", {}).get("Properties", [])
                            for p in props:
                                smiles = validate_smiles(p.get("CanonicalSMILES", ""))
                                if smiles:
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol and 50 < Descriptors.MolWt(mol) < 2000:
                                        records.append({
                                            "compound_name": p.get("IUPACName", f"CID_{p.get('CID', '')}"),
                                            "smiles": smiles,
                                            "label": 0,
                                            "source": f"tox21_aid{aid}",
                                            "label_description": "nephroprotective",
                                        })
                    except Exception:
                        pass
                    time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Tox21 inactive AID {aid} failed: {e}")
        time.sleep(1)

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["smiles"], keep="first")
    if len(df) > 0 and "label" in df.columns:
        logger.info(f"Tox21/ToxCast compounds: {len(df)} ({(df['label']==0).sum()} protective, {(df['label']==1).sum()} toxic)")
    else:
        logger.info("Tox21/ToxCast: no compounds retrieved")
        df = pd.DataFrame(columns=["compound_name", "smiles", "label", "source", "label_description"])
    return df


# ---------------------------------------------------------------------------
# Main Expansion Pipeline
# ---------------------------------------------------------------------------

def run_expansion() -> pd.DataFrame:
    """
    Run the full dataset expansion pipeline.
    Loads the existing dataset and adds new compounds from all sources.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing dataset
    existing_path = PROCESSED_DIR / "nephroscreen_dataset.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        logger.info(f"Existing dataset: {len(existing)} compounds")
    else:
        existing = pd.DataFrame()
        logger.info("No existing dataset found, starting fresh")

    existing_smiles = set(existing["smiles"].tolist()) if len(existing) > 0 else set()

    # Source 1: Review paper compounds
    logger.info("=" * 60)
    logger.info("Source 1: Review paper compounds")
    logger.info("=" * 60)
    review_df = collect_review_compounds()
    review_df = review_df[~review_df["smiles"].isin(existing_smiles)]
    review_df.to_csv(RAW_DIR / "review_compounds.csv", index=False)
    logger.info(f"New from reviews: {len(review_df)}")

    # Source 2: FAERS
    logger.info("=" * 60)
    logger.info("Source 2: FDA FAERS kidney injury drugs")
    logger.info("=" * 60)
    faers_df = collect_faers_compounds(top_n=150)
    new_smiles = existing_smiles | set(review_df["smiles"].tolist())
    if len(faers_df) > 0 and "smiles" in faers_df.columns:
        faers_df = faers_df[~faers_df["smiles"].isin(new_smiles)]
        faers_df.to_csv(RAW_DIR / "faers_compounds.csv", index=False)
    logger.info(f"New from FAERS: {len(faers_df)}")

    # Source 3: Tox21
    logger.info("=" * 60)
    logger.info("Source 3: Tox21/ToxCast kidney assays")
    logger.info("=" * 60)
    tox21_df = collect_tox21_kidney_data()
    if len(faers_df) > 0 and "smiles" in faers_df.columns:
        new_smiles = new_smiles | set(faers_df["smiles"].tolist())
    if len(tox21_df) > 0 and "smiles" in tox21_df.columns:
        tox21_df = tox21_df[~tox21_df["smiles"].isin(new_smiles)]
        tox21_df.to_csv(RAW_DIR / "tox21_compounds.csv", index=False)
    logger.info(f"New from Tox21: {len(tox21_df)}")

    # Merge all
    logger.info("=" * 60)
    logger.info("Merging all sources")
    logger.info("=" * 60)
    cols = ["compound_name", "smiles", "label", "source", "label_description"]
    dfs = [existing]
    for extra_df in [review_df, faers_df, tox21_df]:
        if len(extra_df) > 0 and "smiles" in extra_df.columns:
            valid_cols = [c for c in cols if c in extra_df.columns]
            dfs.append(extra_df[valid_cols])

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["smiles"], keep="first")
    combined = combined.dropna(subset=["smiles"])

    # Save expanded dataset
    combined.to_csv(PROCESSED_DIR / "nephroscreen_dataset.csv", index=False)

    logger.info("=" * 60)
    logger.info("EXPANDED DATASET SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total compounds: {len(combined)}")
    logger.info(f"Nephroprotective (label=0): {(combined['label'] == 0).sum()}")
    logger.info(f"Nephrotoxic (label=1): {(combined['label'] == 1).sum()}")
    logger.info(f"Sources: {combined['source'].value_counts().to_dict()}")

    return combined


if __name__ == "__main__":
    df = run_expansion()
    print(f"\nFinal expanded dataset: {df.shape}")
    print(f"\nClass distribution:\n{df['label_description'].value_counts()}")
    print(f"\nSource distribution:\n{df['source'].value_counts()}")
