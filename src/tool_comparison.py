"""
NephroScreen — Tool Comparison
================================
Prepares a compound set for comparison against existing toxicity
prediction tools (pkCSM, admetSAR) and generates comparison metrics.

Workflow:
1. Run prepare_comparison_set() to get a SMILES list
2. Manually submit to pkCSM / admetSAR web tools
3. Save external predictions as CSV
4. Run compare_tools() to generate comparison
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
COMPARISON_DIR = Path(__file__).parent.parent / "data" / "comparison"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def prepare_comparison_set(n: int = 50) -> pd.DataFrame:
    """
    Select a stratified random subset of test compounds for external tool comparison.
    Saves SMILES list for manual submission to pkCSM/admetSAR.
    """
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROCESSED_DIR / "nephroscreen_dataset_valid.csv")

    # Stratified sample
    protective = df[df["label"] == 0].sample(n=min(n // 3, len(df[df["label"] == 0])), random_state=42)
    toxic = df[df["label"] == 1].sample(n=min(2 * n // 3, len(df[df["label"] == 1])), random_state=42)
    subset = pd.concat([protective, toxic]).sample(frac=1, random_state=42)

    # Save for external tools
    subset[["compound_name", "smiles", "label", "label_description"]].to_csv(
        COMPARISON_DIR / "comparison_set.csv", index=False
    )

    # Save just SMILES for easy pasting
    with open(COMPARISON_DIR / "comparison_smiles.txt", "w") as f:
        for smi in subset["smiles"]:
            f.write(smi + "\n")

    logger.info(f"Comparison set: {len(subset)} compounds saved to {COMPARISON_DIR}")
    logger.info(f"  Protective: {len(protective)}, Toxic: {len(toxic)}")
    logger.info(f"\nManual steps:")
    logger.info(f"  1. Open {COMPARISON_DIR / 'comparison_smiles.txt'}")
    logger.info(f"  2. Submit to pkCSM (http://biosig.unimelb.edu.au/pkcsm/)")
    logger.info(f"  3. Save results as {COMPARISON_DIR / 'pkcsm_results.csv'}")

    return subset


def compare_tools(
    nephroscreen_preds: np.ndarray,
    y_true: np.ndarray,
    external_results: dict[str, np.ndarray] = None,
) -> pd.DataFrame:
    """
    Compare NephroScreen predictions against external tool predictions.

    Args:
        nephroscreen_preds: NephroScreen probability predictions
        y_true: True labels
        external_results: Dict of {tool_name: prediction_array}

    Returns:
        DataFrame with comparison metrics
    """
    results = {}

    # NephroScreen
    ns_binary = (nephroscreen_preds > 0.5).astype(int)
    results["NephroScreen"] = {
        "Accuracy": accuracy_score(y_true, ns_binary),
        "ROC-AUC": roc_auc_score(y_true, nephroscreen_preds),
        "Sensitivity": np.sum((ns_binary == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1),
        "Specificity": np.sum((ns_binary == 0) & (y_true == 0)) / max(np.sum(y_true == 0), 1),
    }

    if external_results:
        for tool_name, preds in external_results.items():
            binary = (preds > 0.5).astype(int) if preds.max() <= 1 else preds
            results[tool_name] = {
                "Accuracy": accuracy_score(y_true, binary),
                "ROC-AUC": roc_auc_score(y_true, preds) if preds.max() <= 1 else None,
                "Sensitivity": np.sum((binary == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1),
                "Specificity": np.sum((binary == 0) & (y_true == 0)) / max(np.sum(y_true == 0), 1),
            }

    comparison_df = pd.DataFrame(results).T
    logger.info(f"\nTool Comparison:\n{comparison_df.round(3)}")
    return comparison_df


def plot_tool_comparison(comparison_df: pd.DataFrame, output_dir: Path = None):
    """Generate tool comparison bar chart."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [c for c in comparison_df.columns if comparison_df[c].notna().any()]
    tools = comparison_df.index.tolist()
    colors = ["#27AE60", "#2196F3", "#FF5722", "#9C27B0"][:len(tools)]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.8 / len(tools)

    for i, (tool, color) in enumerate(zip(tools, colors)):
        values = [comparison_df.loc[tool, m] if pd.notna(comparison_df.loc[tool, m]) else 0 for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=tool, color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x + width * (len(tools) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("NephroScreen vs. Existing Tools", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tool_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'tool_comparison.png'}")


if __name__ == "__main__":
    subset = prepare_comparison_set(n=50)
    print(f"Comparison set prepared: {len(subset)} compounds")
