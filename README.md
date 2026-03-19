# NephroScreen — AI-Powered Nephrotoxicity Prediction

**NephroScreen** is an interactive web application that predicts whether a given chemical compound is likely **nephroprotective** or **nephrotoxic**, using machine learning trained on real bioactivity data from ChEMBL and curated pharmacological literature.

> **Live Demo**: [nephroscreen.streamlit.app](https://nephroscreen.streamlit.app) *(link active after deployment)*

---

## Why This Matters

Chronic kidney disease (CKD) affects roughly **13% of the population in Northern Ghana**, where I grew up. Communities rely heavily on herbal medicines whose nephrotoxic or nephroprotective potential is almost entirely uncharacterised computationally. My MPhil thesis studied one plant extract through bench methods. **NephroScreen extends that work** by asking: *what if we could computationally screen thousands of compounds for nephrotoxic/nephroprotective potential?*

---

## Thesis Context

This project is a computational extension of my MPhil thesis at the **University for Development Studies** (Tamale, Northern Ghana).

### What I Studied
I investigated the **nephroprotective effects of the ethanol extract of *Ageratum conyzoides* leaves (ESE)** against carbon tetrachloride (CCl₄)-induced renal toxicity in a rat model.

### Biomarkers Measured
| Marker | Category | What It Measures |
|--------|----------|------------------|
| **SOD** | Antioxidant enzyme | First-line defence; converts superoxide radicals to H₂O₂ |
| **GSH** | Antioxidant | Major non-enzymatic antioxidant; conjugates reactive metabolites |
| **CAT** | Antioxidant enzyme | Converts H₂O₂ to water and oxygen |
| **MDA** | Oxidative stress | Lipid peroxidation marker; indicates membrane damage |
| **TNF-α** | Pro-inflammatory | Drives acute inflammation and tissue injury |
| **TGF-β1** | Pro-fibrotic | Elevated in chronic kidney injury |
| **NF-κB** | Transcription factor | Master regulator of inflammatory gene expression |
| **COX-2** | Inflammatory enzyme | Produces prostaglandins; drives inflammation |

### Key Results
- ESE at **500 mg/kg** achieved **96% kidney protection**, surpassing silymarin (reference drug) at **93%**
- **Dose-dependent restoration** of antioxidant enzymes (SOD, GSH, CAT) to near-normal levels
- **Dose-dependent reduction** of MDA (lipid peroxidation)
- **Significant downregulation** of all four pro-inflammatory cytokines (TNF-α, TGF-β1, NF-κB, COX-2)
- Histopathological confirmation: CCl₄ caused diffuse tubular ectasia with coagulation necrosis; ESE at 500 mg/kg restored near-normal kidney architecture
- ESE alone showed **no toxicity** across haematological, biochemical, or histological parameters

### Mechanistic Conclusion
The nephroprotective effect is mediated through **dual antioxidant and anti-inflammatory mechanisms**. The phytochemical classes found in the extract (flavonoids, phenolics, triterpenes, alkaloids, tannins) synergistically enhance antioxidant defences, stabilise membranes, and suppress inflammatory pathways (NF-κB and MAPK signalling).

**Important note**: My thesis performed qualitative phytochemical screening (presence/absence with relative concentration), not compound-level identification via GC-MS or HPLC. Specific named compounds from *A. conyzoides* (e.g., precocene I, polymethoxyflavones) are from general literature (Kotta et al., 2020), not from my experimental results.

---

## Methodology

### Data Sources
1. **ChEMBL Database**: Bioactivity data from kidney cell viability assays (HEK293, renal tubular cells, RPTEC), classified by IC50/EC50 thresholds
2. **Curated Literature**: Known nephrotoxic and nephroprotective compounds from published reviews (Tiong et al., 2014; Hoover et al., 2023), drug databases, and pharmacology literature

### Dataset
| Metric | Value |
|--------|-------|
| Total compounds | **1,005** |
| Nephroprotective | 353 |
| Nephrotoxic | 652 |
| Training set | 779 (literature-curated + ChEMBL + Tox21) |
| External validation set | 226 (FAERS + review papers) |

Data sources: ChEMBL bioactivity assays, FDA FAERS kidney adverse events, curated literature reviews (Hoste et al. 2015, Perazella 2009, Oguntibeju 2019), and PubChem-resolved compounds. Class imbalance handled with SMOTE oversampling and balanced class weights.

### Feature Engineering
- **Morgan Fingerprints** (radius=2, 2048 bits): Circular fingerprints encoding molecular substructure — primary features for classification
- **18 Molecular Descriptors**: MW, LogP, TPSA, HBD, HBA, rotatable bonds, aromatic rings, Fraction sp³, molar refractivity, and others
- **MACCS Keys** (166 bits): Used for similarity searching

### Model Selection
Three classifiers were compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | 5-Fold CV AUC |
|-------|----------|-----------|--------|-----|---------|---------------|
| Random Forest | 0.878 | 0.881 | 0.952 | 0.915 | 0.945 | 0.940 ± 0.031 |
| **XGBoost** | **0.856** | **0.902** | **0.887** | **0.894** | **0.948** | **0.919 ± 0.031** |
| Logistic Regression | 0.878 | 0.870 | 0.968 | 0.916 | 0.907 | 0.961 ± 0.023 |

**Selected model**: XGBoost (highest test ROC-AUC = 0.948)

### External Validation
The model was independently validated on 226 compounds from FAERS and review paper sources (completely separate provenance from training data):

| Metric | Value |
|--------|-------|
| External ROC-AUC | **0.898** |
| External Accuracy | 0.757 |
| External F1 | 0.830 |

The ~5% AUC drop from internal (0.948) to external (0.898) validation is expected and demonstrates the model generalises beyond its training distribution.

### SHAP Explainability
SHAP (SHapley Additive exPlanations) analysis identifies which molecular features drive individual predictions. Key discriminating features include MinPartialCharge, specific Morgan fingerprint bits corresponding to hydroxyl groups and aromatic ring patterns, and BalabanJ topological index.

### A. conyzoides Compound Screening
32 compounds identified from published GC-MS/HPLC studies of *A. conyzoides* (Kotta et al. 2020, Okunade 2002, Bosi et al. 2013) were screened through NephroScreen. **31 of 32 compounds (97%) were predicted as nephroprotective**, computationally validating the thesis finding that the plant extract has nephroprotective properties. Only coumarin was borderline (66% nephrotoxic probability).

### Applicability Domain
The model includes an applicability domain check using Tanimoto similarity. If a query compound's maximum similarity to any training compound falls below 0.30, a warning is displayed indicating that the prediction may be less reliable because the compound is structurally distant from the training data.

---

## Features

### Single Compound Analysis
- Enter SMILES directly, look up by compound name (PubChem API), or select from example compounds
- 2D molecular structure visualisation (RDKit)
- Key molecular descriptors (MW, LogP, TPSA, HBD, HBA, etc.)
- Lipinski Rule of 5 compliance check
- Nephrotoxicity/nephroprotection prediction with confidence gauge
- Applicability domain assessment
- Top 5 most similar training compounds (Tanimoto similarity)

### Batch Screening
- Upload CSV with SMILES column
- Run predictions on all valid compounds
- Interactive results table with color-coded predictions
- Downloadable CSV results

### SHAP Explainability Panel
- Per-prediction waterfall plot showing feature contributions
- Top contributing features table with direction (protective/toxic)

### A. conyzoides Screening Tab
- Pre-computed screening of 32 compounds from published literature
- Interactive results table with predictions and confidence scores
- Bar chart ranked by nephrotoxicity probability

### Thesis Connection Panel
- Biomarker descriptions and key results from my MPhil research
- Explanation of how computational screening extends bench work

---

## Limitations

This project has several important limitations that should be considered when interpreting predictions:

1. **Dataset size**: 449 compounds is moderate for this task. Larger datasets (thousands of compounds) would improve generalisability.

2. **Structure-only predictions**: The model predicts based on molecular structure alone. It **cannot** account for:
   - Dose and exposure duration
   - Metabolic conversion (prodrugs, reactive metabolites)
   - Drug formulation and bioavailability
   - Individual patient factors (genetics, co-morbidities, co-medications)

3. **Crude extract vs. individual compounds**: My thesis studied a crude ethanol extract containing hundreds of compounds acting synergistically. This tool predicts activity for individual compounds, which is a fundamentally different question.

4. **Class imbalance**: The training data is skewed toward nephrotoxic compounds (2.18:1). While this was addressed with SMOTE and class weighting, it may affect performance on edge cases.

5. **Applicability domain**: Predictions for compounds structurally dissimilar to the training data (low Tanimoto similarity) should be treated with caution.

6. **Binary classification**: Nephrotoxicity is a spectrum, not binary. A compound may be protective at one dose and toxic at another (e.g., acetaminophen).

7. **No temporal validation**: The model was validated with random splits, not temporal splits. Future work should include prospective validation.

---

## Future Work

1. **Larger datasets**: Integrate data from ToxCast, Tox21, and additional ChEMBL assays to expand to thousands of compounds
2. **Deep learning**: Graph neural networks (GNNs) that operate directly on molecular graphs, avoiding handcrafted fingerprints
3. **Molecular docking**: Integrate with docking simulations against key renal targets (e.g., OAT transporters, megalin/cubilin)
4. **Multi-organ toxicity**: Expand to predict hepatotoxicity, cardiotoxicity, and other organ-specific toxicities
5. **African traditional medicine**: Systematically screen compounds from medicinal plants used in Northern Ghana and West Africa
6. **Dose-response modelling**: Move from binary classification to quantitative toxicity prediction
7. **Explainable AI**: SHAP or attention-based explanations highlighting which molecular substructures drive predictions

---

## Personal Context

I grew up in **Northern Ghana**, where chronic kidney disease is prevalent and communities depend on herbal medicines with poorly characterised safety profiles. My MPhil research at the University for Development Studies was the first step — showing through bench experiments that *Ageratum conyzoides* extract protects against kidney damage. NephroScreen is the second step — building computational tools to screen compounds at scale. My goal is to pursue a PhD in **computational drug discovery** to continue developing AI tools that bridge traditional medicine and modern drug safety assessment.

---

## How to Run

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/nephroscreen.git
cd nephroscreen

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Project Structure
```
nephroscreen/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/
│   ├── raw/                        # Raw data from ChEMBL and literature
│   ├── processed/                  # Cleaned dataset with features
│   └── example_compounds.csv       # Example compounds for the UI
├── models/
│   ├── best_model.joblib           # Trained XGBoost classifier
│   ├── scaler.joblib               # Feature scaler
│   ├── fingerprint_data.pkl        # Pre-computed fingerprints for similarity
│   ├── model_metrics.json          # Performance metrics
│   ├── feature_names.json          # Feature name mapping
│   ├── confusion_matrices.png      # Confusion matrix plots
│   ├── roc_curves.png              # ROC curve comparison
│   ├── feature_importance.png      # Top feature importance plot
│   └── model_comparison.png        # Model comparison bar chart
├── src/
│   ├── __init__.py
│   ├── data_collection.py          # Data collection from ChEMBL + literature
│   ├── mol_utils.py                # Molecular feature engineering
│   ├── predict.py                  # Prediction module
│   ├── similarity.py               # Tanimoto similarity search
│   └── train_model.py              # Model training pipeline
└── .streamlit/
    └── config.toml                 # Streamlit theme configuration
```

---

## Deploying to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository, branch (`main`), and main file (`app.py`)
5. Click "Deploy"

The app will be live at `https://your-app-name.streamlit.app`.

**Note**: Streamlit Cloud provides free hosting for public apps. The `rdkit-pypi` package installs cleanly on Streamlit Cloud without additional system dependencies.

---

## Citation

If you use or reference this tool, please cite:

> Sarfo-Antwi F, Adam ARL, Larbie C, Emikpe BO, Suurbaar J. "Anti-Inflammatory and Antioxidant Potential of *Ageratum conyzoides* Ethanol Extract Against CCl₄-Induced Renal Toxicity in Rats." *Biomedical and Pharmacology Journal*, 2025;18(1).

---

## License

This project is open-source and available for academic and research purposes.

## Contact

- **Abdul-Rashid Lansah Adam**
- Portfolio: [uxlansah.com](https://uxlansah.com)
