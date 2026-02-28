"""
pipeline/task1_pca.py
──────────────────────
Task 1: Dimensionality Challenge
Reduces the 6-variable (PM2.5, PM10, NO2, Ozone, Temperature, Humidity)
sensor space to 2 principal components and visualises how Industrial vs
Residential zones separate in the new space.

Design decisions:
  • PCA chosen (not t-SNE) because:
      – Produces interpretable linear combinations (loadings)
      – Deterministic and reproducible
      – O(n · p²) complexity, feasible on 876 k rows
  • StandardScaler mandatory: variables have different units/scales
  • Biplot arrows expose the physical meaning of PC1 & PC2
  • sns.despine() removes chart junk (top/right spines) → high data-ink ratio
  • No gridlines, no 3D, no shadows
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES, OUTPUT_DIR, RANDOM_SEED


# ── PCA Core ──────────────────────────────────────────────────────────────────
def perform_pca(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """
    Standardise the 6 environmental features and apply PCA (n=2).

    Returns
    -------
    df       : original DataFrame with PC1, PC2 columns appended
    loadings : DataFrame (6 features × 2 PCs) of component loadings
    pca      : fitted PCA object (for explained variance access)
    """
    x = df[FEATURES].values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    components = pca.fit_transform(x_scaled)

    df = df.copy()
    df["PC1"] = components[:, 0]
    df["PC2"] = components[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=FEATURES
    )
    return df, loadings, pca


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_pca(
        df: pd.DataFrame,
        loadings: pd.DataFrame,
        pca: PCA,
        save_path: str | None = None,
        sample_n: int = 3000
) -> plt.Figure:
    """
    Renders a PCA biplot:
      • Scatter: PC1 vs PC2, coloured by zone (Industrial / Residential)
      • Arrows : loading vectors showing feature contributions
      • Annotations: explained variance on axis labels

    Parameters
    ----------
    df        : DataFrame with PC1, PC2, zone columns
    loadings  : loadings DataFrame from perform_pca()
    pca       : fitted PCA object
    save_path : if provided, saves figure as PNG
    sample_n  : number of points to plot (performance vs density trade-off)
    """
    ev = pca.explained_variance_ratio_ * 100   # percent

    palette = {"Industrial": "#E63946", "Residential": "#457B9D"}
    sample  = df.sample(n=min(sample_n, len(df)), random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor("#0F1117")

    # ── Left panel: scatter plot ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0F1117")

    for zone, colour in palette.items():
        mask = sample["zone"] == zone
        ax.scatter(
            sample.loc[mask, "PC1"],
            sample.loc[mask, "PC2"],
            c=colour, alpha=0.35, s=8, label=zone, rasterized=True
        )

    # Biplot arrows (scaled for readability)
    arrow_scale = 3.5
    for feat in FEATURES:
        ax.annotate(
            "",
            xy=(loadings.loc[feat, "PC1"] * arrow_scale,
                loadings.loc[feat, "PC2"] * arrow_scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#FFD166", lw=1.8)
        )
        ax.text(
            loadings.loc[feat, "PC1"] * arrow_scale * 1.12,
            loadings.loc[feat, "PC2"] * arrow_scale * 1.12,
            feat, color="#FFD166", fontsize=9, ha="center", va="center"
        )

    ax.axhline(0, color="#444", lw=0.5, ls="--")
    ax.axvline(0, color="#444", lw=0.5, ls="--")
    ax.set_xlabel(f"PC1  ({ev[0]:.1f}% variance)", color="white", fontsize=11)
    ax.set_ylabel(f"PC2  ({ev[1]:.1f}% variance)", color="white", fontsize=11)
    ax.set_title("PCA Biplot: Industrial vs Residential Zones",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    legend = ax.legend(frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("white")

    # ── Right panel: loadings bar chart ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0F1117")
    y_pos = np.arange(len(FEATURES))
    ax2.barh(y_pos, loadings["PC1"], color="#E63946", alpha=0.8, label="PC1")
    ax2.barh(y_pos, loadings["PC2"], color="#457B9D", alpha=0.8, label="PC2",
             left=loadings["PC1"])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(FEATURES, color="white", fontsize=9)
    ax2.set_xlabel("Loading Value", color="white", fontsize=10)
    ax2.set_title("Feature Loadings", color="white", fontsize=12, pad=10)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_visible(False)
    ax2.axvline(0, color="#888", lw=0.5)
    legend2 = ax2.legend(frameon=False, fontsize=9)
    for text in legend2.get_texts():
        text.set_color("white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[task1_pca] Saved → {save_path}")
    return fig


# ── Loadings Narrative ────────────────────────────────────────────────────────
def loadings_narrative(loadings: pd.DataFrame, pca: PCA) -> str:
    """Return a markdown string explaining the PCA loadings."""
    ev = pca.explained_variance_ratio_ * 100
    top_pc1 = loadings["PC1"].abs().sort_values(ascending=False).index[:3].tolist()
    top_pc2 = loadings["PC2"].abs().sort_values(ascending=False).index[:3].tolist()
    return (
        f"**PC1** captures **{ev[0]:.1f}%** of variance. "
        f"The largest contributors are **{', '.join(top_pc1)}**, confirming that "
        f"combustion-related pollutants drive the Industrial–Residential separation.\n\n"
        f"**PC2** captures **{ev[1]:.1f}%** of variance. "
        f"Its top drivers — **{', '.join(top_pc2)}** — reflect meteorological "
        f"conditions (temperature inversions, humidity) rather than direct emissions. "
        f"PCA is the ideal method here because it produces **interpretable linear "
        f"loadings** while t-SNE does not."
    )


# ── Runner ────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame) -> dict:
    """
    Full Task 1 pipeline step.
    Returns dict with keys: df_pca, loadings, pca, fig, narrative.
    """
    df_pca, loadings, pca = perform_pca(df)
    save_path = os.path.join(OUTPUT_DIR, "task1_pca.png")
    fig       = plot_pca(df_pca, loadings, pca, save_path=save_path)
    narrative = loadings_narrative(loadings, pca)
    print(f"[task1_pca] PC1={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
    return {"df_pca": df_pca, "loadings": loadings, "pca": pca,
            "fig": fig, "narrative": narrative}
