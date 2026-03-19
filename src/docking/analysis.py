"""
Docking results analysis and visualization.
Generates publication-quality plots for the paper.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
DOCKING_DIR = Path(__file__).parent.parent.parent / "data" / "docking_results"


def plot_docking_affinities(results_df: pd.DataFrame, output_dir: Path = None):
    """Bar chart of docking affinities for all compounds."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = results_df[results_df["best_affinity"].notna()].copy()
    valid = valid.sort_values("best_affinity", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(8, len(valid) * 0.35)))

    # Color by affinity strength
    colors = []
    for aff in valid["best_affinity"]:
        if aff <= -8.0:
            colors.append("#1B5E20")  # Strong binder (dark green)
        elif aff <= -6.0:
            colors.append("#27AE60")  # Moderate binder (green)
        elif aff <= -4.0:
            colors.append("#FFC107")  # Weak binder (yellow)
        else:
            colors.append("#E74C3C")  # Very weak (red)

    bars = ax.barh(range(len(valid)), valid["best_affinity"], color=colors, alpha=0.85)

    # Labels
    labels = []
    for _, row in valid.iterrows():
        cls = f" ({row['compound_class']})" if "compound_class" in row and pd.notna(row.get("compound_class")) else ""
        labels.append(f"{row['name']}{cls}")

    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Binding Affinity (kcal/mol)", fontsize=11)
    ax.set_title(
        "AutoDock Vina Docking: A. conyzoides Compounds vs. COX-2 (PDB: 5KIR)",
        fontsize=11, fontweight="bold",
    )
    ax.axvline(x=-6.0, color="gray", linestyle="--", alpha=0.5, label="Moderate binding (-6.0)")
    ax.axvline(x=-8.0, color="gray", linestyle=":", alpha=0.5, label="Strong binding (-8.0)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "docking_affinity_chart.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'docking_affinity_chart.png'}")


def plot_docking_vs_ml(results_df: pd.DataFrame, output_dir: Path = None):
    """Scatter plot: docking affinity vs ML nephroprotective probability."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    valid = results_df[
        results_df["best_affinity"].notna() & results_df["probability_protective"].notna()
    ].copy()

    if len(valid) < 3:
        logger.warning("Not enough data for docking vs ML plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by compound class
    classes = valid["compound_class"].unique() if "compound_class" in valid.columns else ["Unknown"]
    color_map = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    class_colors = dict(zip(classes, color_map))

    for cls in classes:
        mask = valid["compound_class"] == cls if "compound_class" in valid.columns else [True] * len(valid)
        subset = valid[mask]
        ax.scatter(
            subset["best_affinity"],
            subset["probability_protective"],
            c=[class_colors.get(cls, "gray")],
            s=80, alpha=0.7, edgecolors="white", linewidth=0.5,
            label=cls,
        )
        # Label points
        for _, row in subset.iterrows():
            ax.annotate(
                row["name"], (row["best_affinity"], row["probability_protective"]),
                fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlabel("Docking Affinity (kcal/mol) — more negative = stronger binding", fontsize=10)
    ax.set_ylabel("ML-Predicted Nephroprotective Probability", fontsize=10)
    ax.set_title(
        "Docking Affinity vs. ML Prediction\nA. conyzoides Compounds",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8, title="Compound Class", title_fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=-6.0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "docking_vs_ml_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_dir / 'docking_vs_ml_correlation.png'}")


def generate_docking_summary(results_df: pd.DataFrame) -> str:
    """Generate a text summary of docking results."""
    valid = results_df[results_df["best_affinity"].notna()]
    failed = results_df[results_df["best_affinity"].isna()]

    summary = []
    summary.append(f"Docking Summary: {len(valid)} successful, {len(failed)} failed")
    summary.append(f"  Mean affinity: {valid['best_affinity'].mean():.2f} kcal/mol")
    summary.append(f"  Best binder: {valid.loc[valid['best_affinity'].idxmin(), 'name']} "
                   f"({valid['best_affinity'].min():.2f} kcal/mol)")
    summary.append(f"  Weakest binder: {valid.loc[valid['best_affinity'].idxmax(), 'name']} "
                   f"({valid['best_affinity'].max():.2f} kcal/mol)")

    strong = valid[valid["best_affinity"] <= -8.0]
    moderate = valid[(valid["best_affinity"] > -8.0) & (valid["best_affinity"] <= -6.0)]
    weak = valid[valid["best_affinity"] > -6.0]
    summary.append(f"  Strong binders (< -8.0): {len(strong)}")
    summary.append(f"  Moderate binders (-8.0 to -6.0): {len(moderate)}")
    summary.append(f"  Weak binders (> -6.0): {len(weak)}")

    if "compound_class" in valid.columns:
        summary.append("\n  By compound class:")
        class_stats = valid.groupby("compound_class")["best_affinity"].agg(["mean", "min", "count"])
        for cls, row in class_stats.iterrows():
            summary.append(f"    {cls}: mean={row['mean']:.2f}, best={row['min']:.2f}, n={int(row['count'])}")

    return "\n".join(summary)
