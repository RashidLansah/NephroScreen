"""
NephroScreen — AI-Powered Nephrotoxicity Prediction Tool
==========================================================
A Streamlit web application that predicts whether a given chemical
compound is likely nephroprotective or nephrotoxic using machine
learning trained on real bioactivity data.

Author: Abdul-Rashid Lansah Adam
Publication: Sarfo-Antwi F, Adam ARL, Larbie C, Emikpe BO, Suurbaar J.
    Biomed Pharmacol J, 2025;18(1).
"""

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from src.mol_utils import (
    compute_descriptors,
    lipinski_rule_of_5,
    mol_from_smiles,
    resolve_compound_name,
)
from src.predict import get_model_metrics, predict_nephrotoxicity
from src.similarity import compute_applicability_domain, find_similar_compounds

# ---------------------------------------------------------------------------
# Page Config & Custom CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NephroScreen — Nephrotoxicity Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Global */
    .stApp { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Header area */
    .main-header {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: #E8F5E9;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .main-header .subtitle {
        color: #81C784;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .main-header .thesis-link {
        color: #B0BEC5;
        font-size: 0.85rem;
        line-height: 1.5;
    }

    /* Prediction result cards */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-protective {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border: 2px solid #27AE60;
    }
    .prediction-toxic {
        background: linear-gradient(135deg, #FFEBEE, #FFCDD2);
        border: 2px solid #E74C3C;
    }
    .prediction-label {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .prediction-confidence {
        font-size: 1rem;
        opacity: 0.85;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .info-card h4 {
        color: #1A1A2E;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }

    /* Domain warning */
    .domain-warning {
        background: #FFF3E0;
        border: 1px solid #FF9800;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .domain-ok {
        background: #E8F5E9;
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    /* Thesis panel */
    .thesis-panel {
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        border: 1px solid #64B5F6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .thesis-panel h3 {
        color: #1565C0;
        margin-bottom: 0.8rem;
    }

    /* Similarity table */
    .sim-protective { color: #27AE60; font-weight: 600; }
    .sim-toxic { color: #E74C3C; font-weight: 600; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem 0;
        margin-top: 2rem;
        border-top: 1px solid #E0E0E0;
        color: #78909C;
        font-size: 0.8rem;
    }
    .footer a { color: #1565C0; text-decoration: none; }

    /* Sidebar refinements */
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem;
        color: #1A1A2E;
    }

    /* Metric cards in sidebar */
    .metric-box {
        background: #F5F5F5;
        border-radius: 6px;
        padding: 0.6rem 0.8rem;
        margin: 0.3rem 0;
        border-left: 3px solid #27AE60;
    }
    .metric-box .metric-label {
        font-size: 0.75rem;
        color: #78909C;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-box .metric-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1A1A2E;
    }

    /* Disclaimer */
    .disclaimer {
        background: #ECEFF1;
        border-radius: 6px;
        padding: 0.8rem;
        font-size: 0.78rem;
        color: #546E7A;
        line-height: 1.5;
        margin-top: 0.5rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>NephroScreen</h1>
    <div class="subtitle">AI-Powered Nephrotoxicity Prediction</div>
    <div class="thesis-link">
        Extending bench research into computational screening — from the
        nephroprotective effects of <em>Ageratum conyzoides</em> extract
        (96% kidney protection at 500 mg/kg) to AI-driven compound analysis.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### About NephroScreen")
    st.markdown(
        "NephroScreen predicts whether a chemical compound is likely "
        "**nephroprotective** or **nephrotoxic** using machine learning "
        "trained on real bioactivity data from ChEMBL and curated literature."
    )

    # Model metrics
    st.markdown("---")
    st.markdown("### Model Performance")
    metrics = get_model_metrics()
    best = metrics.get("best_metrics", {})
    model_name = metrics.get("best_model", "Unknown")

    st.markdown(f"**Model:** {model_name}")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-value">{best.get('roc_auc', 0):.3f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{best.get('accuracy', 0):.3f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">{best.get('f1_score', 0):.3f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Precision</div>
        <div class="metric-value">{best.get('precision', 0):.3f}</div>
    </div>
    <div class="metric-box">
        <div class="metric-label">Recall</div>
        <div class="metric-value">{best.get('recall', 0):.3f}</div>
    </div>
    """, unsafe_allow_html=True)

    # External validation
    ext_val = metrics.get("external_validation", {})
    if ext_val:
        st.markdown("---")
        st.markdown("### External Validation")
        st.markdown(f"""
        <div class="metric-box" style="border-left-color: #FF5722;">
            <div class="metric-label">External ROC-AUC</div>
            <div class="metric-value">{ext_val.get('roc_auc', 'N/A')}</div>
        </div>
        <div class="metric-box" style="border-left-color: #FF5722;">
            <div class="metric-label">External Accuracy</div>
            <div class="metric-value">{ext_val.get('accuracy', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Tested on {ext_val.get('n_samples', '?')} independent compounds (FAERS + review papers)")

    # Dataset stats
    st.markdown("---")
    st.markdown("### Dataset")
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Total Compounds</div>
        <div class="metric-value">{metrics.get('dataset_size', 0)}</div>
    </div>
    """, unsafe_allow_html=True)
    dist = metrics.get("class_distribution", {})
    st.markdown(
        f"- Nephroprotective: **{dist.get('nephroprotective', 0)}**\n"
        f"- Nephrotoxic: **{dist.get('nephrotoxic', 0)}**"
    )
    st.markdown(f"- Features: **{metrics.get('n_features', 0)}**")
    st.markdown("- Sources: ChEMBL, PubChem, curated literature")

    # Methodology
    st.markdown("---")
    st.markdown("### Methodology")
    st.markdown(
        "1. **Data**: Curated compounds from ChEMBL bioactivity assays "
        "and published nephrotoxicity/nephroprotection literature\n"
        "2. **Features**: 2048-bit Morgan fingerprints (radius=2) + "
        "18 molecular descriptors (MW, LogP, TPSA, etc.)\n"
        "3. **Model**: Compared Random Forest, XGBoost, and Logistic "
        "Regression. Best model selected by ROC-AUC on held-out test set\n"
        "4. **Validation**: 80/20 stratified split + 5-fold cross-validation. "
        "Class imbalance handled with SMOTE and balanced class weights"
    )

    st.markdown("---")
    st.markdown("### All Models")
    all_models = metrics.get("all_models", {})
    model_df = pd.DataFrame(all_models).T
    if len(model_df) > 0:
        model_df = model_df[["accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_auc_mean"]]
        model_df.columns = ["Acc", "Prec", "Rec", "F1", "AUC", "CV AUC"]
        st.dataframe(model_df.style.format("{:.3f}"), use_container_width=True)


# ---------------------------------------------------------------------------
# How to Use
# ---------------------------------------------------------------------------
with st.expander("📖 **How to Use NephroScreen** — Click to expand", expanded=False):
    st.markdown("""
    <div style="line-height: 1.8; font-size: 0.92rem;">

    **NephroScreen predicts whether a chemical compound is likely to protect or damage the kidneys.**

    ---

    #### 🔬 Quick Start (3 steps)
    1. **Enter a compound** using any of the input methods below
    2. **View the prediction** — green = nephroprotective, red = nephrotoxic
    3. **Explore the details** — molecular properties, similar compounds, and confidence score

    ---

    #### 📥 Input Methods

    | Tab | What it does | Best for |
    |-----|-------------|----------|
    | **SMILES Input** | Paste a SMILES string directly | Chemists who know SMILES notation |
    | **Name Lookup** | Type a compound name (e.g. "aspirin") and we resolve it via PubChem | Everyone — just type a drug name |
    | **Example Compounds** | Pick from pre-loaded nephroprotective and nephrotoxic compounds | Quick demo and exploration |
    | **A. conyzoides Screening** | Pre-computed results for 32 compounds from *Ageratum conyzoides* literature | Seeing the thesis connection |
    | **West African Plants** | Pre-computed results for 36 compounds from 12 medicinal plants used in Northern Ghana | Ethnopharmacology context |
    | **Molecular Docking** | Pre-computed AutoDock Vina docking results against COX-2 (PDB: 5KIR) | Understanding binding mechanisms |
    | **Batch Screening** | Upload a CSV of SMILES to screen many compounds at once | Researchers with compound libraries |

    ---

    #### 🎯 Understanding the Output

    - **Prediction**: "Likely Nephroprotective" or "Likely Nephrotoxic" with a confidence percentage
    - **Confidence gauge**: Visual indicator — higher confidence = model is more certain
    - **Applicability domain**: ⚠️ warnings appear if the compound is structurally distant from training data
    - **Similar compounds**: Top 5 most structurally similar compounds from the training set, showing what the model is basing its prediction on
    - **Molecular descriptors**: Drug-likeness properties (Lipinski's Rule of 5, LogP, molecular weight, etc.)

    ---

    #### ⚠️ Important Disclaimer
    This is a **research tool for exploratory screening**. It is not validated for clinical decision-making.
    Predictions are based on structural similarity to known nephrotoxic and nephroprotective compounds.
    Always validate computationally flagged compounds through experimental methods.

    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Compound Input
# ---------------------------------------------------------------------------
st.markdown("## Compound Input")

input_tab1, input_tab2, input_tab3, input_tab5, input_tab6, input_tab7, input_tab4 = st.tabs([
    "SMILES Input", "Name Lookup", "Example Compounds",
    "A. conyzoides Screening", "West African Plants", "Molecular Docking", "Batch Screening"
])

smiles_input = None

with input_tab1:
    st.markdown("Enter a SMILES string directly.")
    smiles_text = st.text_input(
        "SMILES string",
        placeholder="e.g., CC(=O)NC1=CC=C(O)C=C1 (acetaminophen)",
        key="smiles_direct",
    )
    if smiles_text:
        smiles_input = smiles_text.strip()

with input_tab2:
    st.markdown("Type a compound name — resolved via PubChem API.")
    compound_name = st.text_input(
        "Compound name",
        placeholder="e.g., aspirin, quercetin, gentamicin",
        key="name_lookup",
    )
    if compound_name:
        with st.spinner(f"Resolving '{compound_name}' via PubChem..."):
            resolved = resolve_compound_name(compound_name)
            if resolved:
                st.success(f"Resolved to: `{resolved}`")
                smiles_input = resolved
            else:
                st.error(
                    f"Could not resolve '{compound_name}'. "
                    "Check the spelling or try entering SMILES directly."
                )

with input_tab3:
    st.markdown("Select from known nephroprotective and nephrotoxic compounds.")
    examples = pd.read_csv(Path(__file__).parent / "data" / "example_compounds.csv")

    col_prot, col_tox = st.columns(2)
    with col_prot:
        st.markdown("**Nephroprotective**")
        prot = examples[examples["category"] == "nephroprotective"]
        for _, row in prot.iterrows():
            st.markdown(f"- {row['name']}: *{row['description']}*")

    with col_tox:
        st.markdown("**Nephrotoxic**")
        tox = examples[examples["category"] == "nephrotoxic"]
        for _, row in tox.iterrows():
            st.markdown(f"- {row['name']}: *{row['description']}*")

    selected_example = st.selectbox(
        "Select a compound",
        options=[""] + examples["name"].tolist(),
        key="example_select",
    )
    if selected_example:
        row = examples[examples["name"] == selected_example].iloc[0]
        smiles_input = row["smiles"]
        st.info(f"**{row['name']}** ({row['category']}): {row['description']}")

    # Thesis context
    st.markdown("---")
    st.markdown("""
    <div class="thesis-panel">
        <h4 style="color: #1565C0; margin-top: 0;">Thesis Context</h4>
        <p style="font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
            My MPhil thesis investigated the nephroprotective effects of the
            ethanol extract of <em>Ageratum conyzoides</em> leaves against
            CCl<sub>4</sub>-induced renal toxicity in rats. The extract at
            500 mg/kg achieved <strong>96% kidney protection</strong>,
            surpassing the reference drug silymarin (93%). Qualitative
            phytochemical screening revealed the presence of alkaloids, tannins,
            phenols, flavonoids, triterpenes, and saponins. This tool extends
            that work computationally — screening individual compounds for
            nephrotoxic/nephroprotective potential rather than crude extracts.
        </p>
    </div>
    """, unsafe_allow_html=True)

with input_tab5:
    st.markdown(
        "Compounds identified from *A. conyzoides* in published GC-MS/HPLC studies "
        "(Kotta et al. 2020, Okunade 2002, Bosi et al. 2013), screened through NephroScreen."
    )
    ageratum_results_path = Path(__file__).parent / "data" / "processed" / "ageratum_screening_results.csv"
    if ageratum_results_path.exists():
        ag_df = pd.read_csv(ageratum_results_path)
        valid_ag = ag_df[ag_df["prediction"] != "Invalid SMILES"]
        n_prot_ag = (valid_ag["prediction"] == "Nephroprotective").sum()
        n_tox_ag = (valid_ag["prediction"] == "Nephrotoxic").sum()

        st.success(
            f"**{len(valid_ag)}** compounds screened: "
            f"**{n_prot_ag}** nephroprotective, **{n_tox_ag}** nephrotoxic"
        )

        # Color-code results
        def color_ag_pred(val):
            if val == "Nephroprotective":
                return "color: #27AE60; font-weight: 600"
            elif val == "Nephrotoxic":
                return "color: #E74C3C; font-weight: 600"
            return ""

        display_cols = ["name", "compound_class", "prediction", "confidence", "in_domain", "literature_source"]
        available_cols = [c for c in display_cols if c in valid_ag.columns]
        styled_ag = valid_ag[available_cols].style.applymap(
            color_ag_pred, subset=["prediction"]
        ).format({"confidence": "{:.3f}"}, na_rep="—")
        st.dataframe(styled_ag, use_container_width=True, hide_index=True, height=400)

        # Bar chart
        fig_ag = go.Figure()
        sorted_ag = valid_ag.sort_values("probability_toxic", ascending=True)
        colors_ag = ["#27AE60" if p == "Nephroprotective" else "#E74C3C" for p in sorted_ag["prediction"]]
        fig_ag.add_trace(go.Bar(
            y=sorted_ag["name"],
            x=sorted_ag["probability_toxic"],
            orientation="h",
            marker_color=colors_ag,
            text=[f"{v:.1%}" for v in sorted_ag["probability_toxic"]],
            textposition="outside",
        ))
        fig_ag.update_layout(
            title="Nephrotoxicity Probability for A. conyzoides Compounds",
            xaxis_title="P(Nephrotoxic)",
            xaxis_range=[0, 1.1],
            height=max(400, len(sorted_ag) * 25),
            margin=dict(l=200),
            showlegend=False,
        )
        fig_ag.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        st.plotly_chart(fig_ag, use_container_width=True)

        st.caption(
            "These compounds are from published literature on A. conyzoides "
            "(NOT from the thesis experimental results). Each compound is attributed "
            "to its source publication."
        )
    else:
        st.info("A. conyzoides screening results not yet generated. Run `python -m src.ageratum_screening` first.")

with input_tab6:
    st.markdown(
        "Compounds from **12 medicinal plants** used in Northern Ghana and West Africa "
        "for kidney/urinary conditions, screened through NephroScreen."
    )
    african_results_path = Path(__file__).parent / "data" / "processed" / "african_plants_screening.csv"
    if african_results_path.exists():
        af_df = pd.read_csv(african_results_path)
        valid_af = af_df[af_df["prediction"] != "Invalid SMILES"]
        n_prot_af = (valid_af["prediction"] == "Nephroprotective").sum()
        n_tox_af = (valid_af["prediction"] == "Nephrotoxic").sum()
        n_plants = valid_af["plant"].nunique()

        st.success(
            f"**{len(valid_af)}** compounds from **{n_plants}** plants: "
            f"**{n_prot_af}** nephroprotective, **{n_tox_af}** nephrotoxic"
        )

        # Group by plant
        for plant in valid_af["plant"].unique():
            subset = valid_af[valid_af["plant"] == plant]
            local = subset["local_name"].iloc[0]
            n_p = (subset["prediction"] == "Nephroprotective").sum()
            n_t = (subset["prediction"] == "Nephrotoxic").sum()
            color = "#27AE60" if n_t == 0 else "#E74C3C" if n_p == 0 else "#FF9800"
            st.markdown(
                f"**{plant}** ({local}) — "
                f"<span style='color:{color}'>{n_p} protective, {n_t} toxic</span>",
                unsafe_allow_html=True,
            )

        # Table
        display_af = valid_af[["plant", "compound", "compound_class", "prediction", "confidence"]].copy()

        def color_af_pred(val):
            if val == "Nephroprotective":
                return "color: #27AE60; font-weight: 600"
            elif val == "Nephrotoxic":
                return "color: #E74C3C; font-weight: 600"
            return ""

        st.dataframe(
            display_af.style.applymap(color_af_pred, subset=["prediction"])
            .format({"confidence": "{:.3f}"}, na_rep="—"),
            use_container_width=True, hide_index=True, height=400,
        )
    else:
        st.info("Run `python -m src.african_plants_screening` first.")

with input_tab7:
    st.markdown(
        "**Molecular docking** of *A. conyzoides* compounds against **COX-2** (PDB: 5KIR) "
        "using AutoDock Vina. COX-2 was chosen because the thesis showed dose-dependent "
        "COX-2 downregulation by the plant extract."
    )
    docking_path = Path(__file__).parent / "data" / "docking_results" / "ageratum_cox2_docking.csv"
    if docking_path.exists():
        dk_df = pd.read_csv(docking_path)
        valid_dk = dk_df[dk_df["best_affinity"].notna()]

        strong = valid_dk[valid_dk["best_affinity"] <= -8.0]
        moderate = valid_dk[(valid_dk["best_affinity"] > -8.0) & (valid_dk["best_affinity"] <= -6.0)]

        st.success(
            f"**{len(valid_dk)}** compounds docked: "
            f"**{len(strong)}** strong binders (< -8.0 kcal/mol), "
            f"**{len(moderate)}** moderate binders (-8.0 to -6.0)"
        )

        # Table
        dk_display = valid_dk[["name", "compound_class", "best_affinity", "n_poses", "docking_status"]].copy()
        dk_display = dk_display.sort_values("best_affinity")

        def color_affinity(val):
            try:
                v = float(val)
                if v <= -8.0:
                    return "color: #1B5E20; font-weight: 700"
                elif v <= -6.0:
                    return "color: #27AE60; font-weight: 600"
                elif v <= -4.0:
                    return "color: #FF9800"
                else:
                    return "color: #E74C3C"
            except (ValueError, TypeError):
                return ""

        st.dataframe(
            dk_display.style.applymap(color_affinity, subset=["best_affinity"])
            .format({"best_affinity": "{:.2f}"}, na_rep="—"),
            use_container_width=True, hide_index=True,
        )

        # Show docking affinity chart
        dock_fig_path = Path(__file__).parent / "figures" / "docking_affinity_chart.png"
        if dock_fig_path.exists():
            st.image(str(dock_fig_path), use_container_width=True)

        st.caption(
            "Binding affinity in kcal/mol (more negative = stronger binding). "
            "Strong binders (< -8.0) are comparable to known COX-2 inhibitors like indomethacin. "
            "Docking performed with AutoDock Vina 1.2.7 against human COX-2 (PDB: 5KIR)."
        )
    else:
        st.info("Docking results not yet generated.")

with input_tab4:
    st.markdown(
        "Upload a CSV file with a `smiles` column. "
        "Predictions will be generated for all valid compounds."
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        if "smiles" not in batch_df.columns:
            st.error("CSV must contain a `smiles` column.")
        else:
            st.markdown(f"Found **{len(batch_df)}** rows. Running predictions...")
            progress = st.progress(0)
            results = []
            for i, row in batch_df.iterrows():
                smi = str(row["smiles"]).strip()
                pred = predict_nephrotoxicity(smi)
                if pred:
                    results.append({
                        "smiles": smi,
                        "name": row.get("name", ""),
                        "prediction": pred["label"],
                        "confidence": pred["confidence"],
                        "prob_protective": pred["probability_protective"],
                        "prob_toxic": pred["probability_toxic"],
                        "MW": pred["descriptors"]["MolWt"],
                        "LogP": pred["descriptors"]["LogP"],
                    })
                else:
                    results.append({
                        "smiles": smi,
                        "name": row.get("name", ""),
                        "prediction": "Invalid SMILES",
                        "confidence": None,
                        "prob_protective": None,
                        "prob_toxic": None,
                        "MW": None,
                        "LogP": None,
                    })
                progress.progress((i + 1) / len(batch_df))

            results_df = pd.DataFrame(results)
            st.markdown(f"### Batch Results ({len(results_df)} compounds)")

            # Color-code the prediction column
            def highlight_prediction(val):
                if val == "Nephroprotective":
                    return "background-color: #E8F5E9; color: #27AE60; font-weight: 600"
                elif val == "Nephrotoxic":
                    return "background-color: #FFEBEE; color: #E74C3C; font-weight: 600"
                return ""

            styled = results_df.style.applymap(
                highlight_prediction, subset=["prediction"]
            ).format({
                "confidence": "{:.3f}",
                "prob_protective": "{:.3f}",
                "prob_toxic": "{:.3f}",
                "MW": "{:.1f}",
                "LogP": "{:.2f}",
            }, na_rep="—")

            st.dataframe(styled, use_container_width=True, height=400)

            # Download button
            csv_buffer = results_df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv_buffer,
                file_name="nephroscreen_batch_results.csv",
                mime="text/csv",
            )

            # Summary stats
            valid = results_df[results_df["prediction"] != "Invalid SMILES"]
            if len(valid) > 0:
                n_prot = (valid["prediction"] == "Nephroprotective").sum()
                n_tox = (valid["prediction"] == "Nephrotoxic").sum()
                st.markdown(
                    f"**Summary**: {n_prot} nephroprotective, {n_tox} nephrotoxic, "
                    f"{len(batch_df) - len(valid)} invalid SMILES"
                )


# ---------------------------------------------------------------------------
# Single Compound Analysis
# ---------------------------------------------------------------------------
if smiles_input and uploaded_file is None:
    st.markdown("---")

    # Validate
    mol = mol_from_smiles(smiles_input)
    if mol is None:
        st.error(
            f"Invalid SMILES: `{smiles_input}`. "
            "Please check the structure and try again."
        )
    else:
        canonical_smiles = Chem.MolToSmiles(mol)

        # Run prediction
        with st.spinner("Analyzing compound..."):
            result = predict_nephrotoxicity(canonical_smiles)
            similar = find_similar_compounds(canonical_smiles, top_n=5)
            domain = compute_applicability_domain(canonical_smiles)

        if result is None:
            st.error("Prediction failed. Please try a different compound.")
        else:
            # ─── ROW 1: Structure + Prediction ───────────────────
            col_struct, col_pred = st.columns([1, 1])

            with col_struct:
                st.markdown("### Molecular Structure")
                img = Draw.MolToImage(mol, size=(400, 350))
                st.image(img, use_container_width=True)
                st.code(canonical_smiles, language=None)

            with col_pred:
                st.markdown("### Prediction Result")

                # Main prediction card
                if result["prediction"] == 0:
                    card_class = "prediction-protective"
                    label_color = "#27AE60"
                    icon = "shield"
                else:
                    card_class = "prediction-toxic"
                    label_color = "#E74C3C"
                    icon = "warning"

                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <div class="prediction-label" style="color: {label_color};">
                        {result['label']}
                    </div>
                    <div class="prediction-confidence">
                        Confidence: {result['confidence']:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["probability_toxic"] * 100,
                    title={"text": "Nephrotoxicity Probability", "font": {"size": 14}},
                    number={"suffix": "%", "font": {"size": 24}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#E74C3C" if result["probability_toxic"] > 0.5 else "#27AE60"},
                        "steps": [
                            {"range": [0, 30], "color": "#E8F5E9"},
                            {"range": [30, 70], "color": "#FFF3E0"},
                            {"range": [70, 100], "color": "#FFEBEE"},
                        ],
                        "threshold": {
                            "line": {"color": "#1A1A2E", "width": 2},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Applicability domain
                if domain["in_domain"]:
                    st.markdown(f"""
                    <div class="domain-ok">
                        <strong>Within applicability domain</strong><br>
                        Nearest training compound: {domain['nearest_compound']}
                        (similarity: {domain['max_similarity']:.3f})
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="domain-warning">
                        <strong>Outside applicability domain</strong><br>
                        {domain['message']}
                    </div>
                    """, unsafe_allow_html=True)

            # ─── ROW 2: Descriptors + Lipinski ────────────────────
            st.markdown("---")
            st.markdown("### Molecular Analysis")

            col_desc, col_lip = st.columns([1.2, 0.8])

            with col_desc:
                st.markdown("#### Key Descriptors")
                desc = result["descriptors"]
                desc_display = {
                    "Molecular Weight": f"{desc['MolWt']:.2f} Da",
                    "LogP (Lipophilicity)": f"{desc['LogP']:.2f}",
                    "H-Bond Donors": f"{int(desc['HBD'])}",
                    "H-Bond Acceptors": f"{int(desc['HBA'])}",
                    "TPSA": f"{desc['TPSA']:.2f} A\u00b2",
                    "Rotatable Bonds": f"{int(desc['RotatableBonds'])}",
                    "Aromatic Rings": f"{int(desc['AromaticRings'])}",
                    "Heavy Atom Count": f"{int(desc['HeavyAtomCount'])}",
                    "Fraction sp3": f"{desc['FractionCSP3']:.3f}",
                    "Molar Refractivity": f"{desc['MolRefractivity']:.2f}",
                }
                desc_df = pd.DataFrame(
                    list(desc_display.items()),
                    columns=["Property", "Value"],
                )
                st.dataframe(desc_df, use_container_width=True, hide_index=True)

            with col_lip:
                st.markdown("#### Lipinski Rule of 5")
                lip = result["lipinski"]
                for rule, passes in lip.items():
                    if rule == "Violations":
                        continue
                    if rule == "Passes Ro5":
                        if passes:
                            st.success(f"Overall: PASSES ({lip['Violations']} violations)")
                        else:
                            st.warning(f"Overall: FAILS ({lip['Violations']} violations)")
                    else:
                        if passes:
                            st.markdown(f"- {rule}: PASS")
                        else:
                            st.markdown(f"- {rule}: **FAIL**")

            # ─── ROW 3: Similarity Panel ──────────────────────────
            st.markdown("---")
            st.markdown("### Structural Similarity to Training Data")
            st.markdown(
                "Top 5 most similar compounds from the training dataset "
                "(Tanimoto similarity on Morgan fingerprints, radius=2)."
            )

            if similar:
                sim_data = []
                for s in similar:
                    sim_data.append({
                        "Compound": s["compound_name"],
                        "Classification": s["label_description"].title(),
                        "Tanimoto Similarity": s["similarity_score"],
                        "SMILES": s["smiles"][:60] + ("..." if len(s["smiles"]) > 60 else ""),
                    })
                sim_df = pd.DataFrame(sim_data)

                def color_classification(val):
                    if val == "Nephroprotective":
                        return "color: #27AE60; font-weight: 600"
                    elif val == "Nephrotoxic":
                        return "color: #E74C3C; font-weight: 600"
                    return ""

                styled_sim = sim_df.style.applymap(
                    color_classification, subset=["Classification"]
                ).format({"Tanimoto Similarity": "{:.4f}"})

                st.dataframe(styled_sim, use_container_width=True, hide_index=True)

                # Similarity bar chart
                fig_sim = go.Figure()
                colors = ["#27AE60" if s["label"] == 0 else "#E74C3C" for s in similar]
                fig_sim.add_trace(go.Bar(
                    x=[s["compound_name"] for s in similar],
                    y=[s["similarity_score"] for s in similar],
                    marker_color=colors,
                    text=[f"{s['similarity_score']:.3f}" for s in similar],
                    textposition="outside",
                ))
                fig_sim.update_layout(
                    title="Tanimoto Similarity to Top 5 Nearest Compounds",
                    yaxis_title="Tanimoto Similarity",
                    yaxis_range=[0, 1.1],
                    height=350,
                    margin=dict(l=40, r=40, t=50, b=80),
                    showlegend=False,
                )
                st.plotly_chart(fig_sim, use_container_width=True)

            # ─── ROW 4: SHAP Explanation ─────────────────────
            st.markdown("---")
            st.markdown("### Feature Contributions (SHAP)")
            st.markdown(
                "SHAP (SHapley Additive exPlanations) shows which molecular features "
                "push the prediction toward nephroprotective or nephrotoxic."
            )
            try:
                from src.shap_analysis import explain_single_prediction
                import joblib as _jl
                _model = _jl.load(Path(__file__).parent / "models" / "best_model.joblib")
                _scaler = _jl.load(Path(__file__).parent / "models" / "scaler.joblib")
                import json as _json
                with open(Path(__file__).parent / "models" / "feature_names.json") as _f:
                    _feat_names = _json.load(_f)

                shap_result = explain_single_prediction(
                    canonical_smiles, _model, _scaler, _feat_names
                )
                if shap_result and shap_result.get("waterfall_image"):
                    st.image(shap_result["waterfall_image"], use_container_width=True)

                if shap_result and shap_result.get("contributions"):
                    st.markdown("**Top contributing features:**")
                    contrib_data = []
                    for c in shap_result["contributions"][:10]:
                        contrib_data.append({
                            "Feature": c["feature"],
                            "SHAP Value": c["shap_value"],
                            "Direction": c["direction"].title(),
                        })
                    contrib_df = pd.DataFrame(contrib_data)

                    def color_direction(val):
                        if val == "Toxic":
                            return "color: #E74C3C; font-weight: 600"
                        elif val == "Protective":
                            return "color: #27AE60; font-weight: 600"
                        return ""

                    st.dataframe(
                        contrib_df.style.applymap(color_direction, subset=["Direction"])
                        .format({"SHAP Value": "{:.4f}"}),
                        use_container_width=True, hide_index=True,
                    )
            except Exception as e:
                st.caption(f"SHAP analysis unavailable: {e}")

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This is a research tool for exploratory screening.
                It is not validated for clinical decision-making. Predictions are based on
                structural similarity to known nephrotoxic and nephroprotective compounds in
                the training dataset. The model cannot account for dose, metabolic conversion,
                formulation, or individual patient factors.
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Thesis Connection Panel
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("## Connection to Bench Research")

thesis_col1, thesis_col2 = st.columns([1.1, 0.9])

with thesis_col1:
    st.markdown("""
    <div class="thesis-panel">
        <h3 style="margin-top:0;">From Bench to In Silico</h3>
        <p style="font-size: 0.9rem; line-height: 1.7;">
            My MPhil thesis at the University for Development Studies (Tamale, Northern Ghana)
            demonstrated that the ethanol extract of <em>Ageratum conyzoides</em> leaves protects
            against CCl<sub>4</sub>-induced kidney damage through <strong>dual antioxidant and
            anti-inflammatory mechanisms</strong>.
        </p>
        <p style="font-size: 0.9rem; line-height: 1.7;">
            The extract at 500 mg/kg body weight achieved <strong>96% kidney protection</strong>,
            surpassing the reference drug silymarin at 93%. This was evidenced by:
        </p>
        <ul style="font-size: 0.88rem; line-height: 1.8;">
            <li>Dose-dependent restoration of antioxidant enzymes (SOD, GSH, CAT)</li>
            <li>Reduction of lipid peroxidation marker (MDA)</li>
            <li>Significant downregulation of pro-inflammatory cytokines
                (TNF-&alpha;, TGF-&beta;1, NF-&kappa;B, COX-2)</li>
            <li>Histopathological confirmation of near-normal kidney architecture</li>
        </ul>
        <p style="font-size: 0.88rem; line-height: 1.7; margin-bottom:0;">
            <strong>This tool asks the next question:</strong> which structural features make
            individual compounds nephroprotective? My thesis showed the crude extract works.
            NephroScreen extends this by computationally screening compounds based on their
            molecular fingerprints and properties.
        </p>
    </div>
    """, unsafe_allow_html=True)

with thesis_col2:
    st.markdown("#### Biomarkers Measured in Thesis")

    biomarkers = {
        "SOD (Superoxide Dismutase)": "First-line antioxidant enzyme; converts superoxide radicals to H\u2082O\u2082",
        "GSH (Reduced Glutathione)": "Major non-enzymatic antioxidant; conjugates reactive metabolites",
        "CAT (Catalase)": "Converts H\u2082O\u2082 to water and oxygen; prevents oxidative damage",
        "MDA (Malondialdehyde)": "Lipid peroxidation marker; elevated levels indicate oxidative membrane damage",
        "TNF-\u03b1": "Pro-inflammatory cytokine; drives acute inflammation and tissue injury",
        "TGF-\u03b21": "Fibrosis mediator; elevated in chronic kidney injury and tissue remodelling",
        "NF-\u03baB": "Master transcription factor for inflammatory genes; key signalling node",
        "COX-2": "Inducible enzyme producing prostaglandins; drives inflammation and pain",
    }

    for marker, desc in biomarkers.items():
        st.markdown(f"**{marker}**")
        st.caption(desc)

    st.markdown("---")
    st.markdown("#### Key Result")
    st.markdown("""
    | Group | Kidney Protection |
    |-------|:-:|
    | CCl\u2084 only | 0% (severe damage) |
    | ESE 250 mg/kg | ~72% |
    | ESE 500 mg/kg | **96%** |
    | Silymarin (ref) | 93% |
    | ESE alone | No toxicity observed |
    """)

    st.markdown("#### Phytochemical Classes in Extract")
    st.markdown(
        "Qualitative screening revealed: alkaloids (high), tannins (high), "
        "phenols (moderate), flavonoids (moderate), triterpenes (moderate), "
        "saponins (low). These classes are known from literature (Kotta et al., 2020) "
        "to contain compounds with antioxidant and anti-inflammatory properties."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <strong>NephroScreen</strong> — Built by
    <a href="https://uxlansah.com" target="_blank">Abdul-Rashid Lansah Adam</a>
    <br>
    Publication:
    <a href="https://biomedpharmajournal.org/" target="_blank">
        Sarfo-Antwi F, Adam ARL, Larbie C, Emikpe BO, Suurbaar J.
        <em>Biomed Pharmacol J</em>, 2025;18(1).
    </a>
    <br><br>
    <em>This is a research tool for exploratory screening. Not validated for clinical use.</em>
</div>
""", unsafe_allow_html=True)
