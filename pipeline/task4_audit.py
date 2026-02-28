"""
pipeline/task4_audit.py
────────────────────────
Task 4: Visual Integrity Audit
Evaluates the proposed 3D bar chart for Pollution vs Population Density
vs Region, then replaces it with a Small Multiples (FacetGrid) approach.

Rejection Justification:
  • Lie Factor (Tufte):  Perspective in 3D charts physically distorts bar
    heights — a bar "behind" another appears shorter even if identical in
    value. This creates a non-zero Lie Factor where LF = 1.0 is honest.
  • Data-Ink Ratio:  3D surfaces, perspective axes, floor shadows, and
    grid walls consume ink without encoding any additional data dimension.
    All that ink violates Tufte's maximise-data-ink rule.

Alternative: Small Multiples (FacetGrid)
  • One scatter panel per Region  (4 regions → 4 panels)
  • x = Population Density, y = PM2.5  (2 real data dimensions)
  • Colour = PM2.5 value with Viridis sequential colormap
  • No additional visual encoding is wasted; every element carries data.

Colour Scale Justification:
  • Viridis is a perceptually uniform sequential colormap.
    Its luminance increases monotonically from dark (low) to light (high).
  • Rainbow/Jet is NOT perceptually uniform — it has artificial banding
    at cyan/green causing false visual "edges" in the data that do not
    correspond to real data discontinuities.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR


# ── Small Multiples Visualisation ─────────────────────────────────────────────
def plot_small_multiples(
        df: pd.DataFrame,
        save_path: str | None = None,
        sample_n: int = 4000
) -> plt.Figure:
    """
    Renders a 1×4 FacetGrid of scatter plots (one per Region).
    Encodes three variables cleanly:
      x  → Population Density (people/km²)
      y  → Daily Mean PM2.5   (µg/m³)
      c  → PM2.5 value via Viridis sequential colormap

    No 3D, no gradients, no shadow, no unnecessary grid.
    """
    # Aggregate to daily mean per sensor (avoids 876k-point overplotting)
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    daily = (df.groupby(["sensor_id", "zone", "region", "population_density", "date"])
               ["PM2.5"].mean()
               .reset_index())
    sample = daily.sample(n=min(sample_n, len(daily)), random_state=42)

    regions = sorted(sample["region"].unique())
    n_cols  = len(regions)

    # Shared colour normalisation across all panels
    vmin = sample["PM2.5"].quantile(0.01)
    vmax = sample["PM2.5"].quantile(0.99)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.viridis

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                             sharey=True)
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle(
        "Small Multiples: PM2.5 Pollution vs Population Density by Region\n"
        "(Sequential Viridis Colourmap — perceptually uniform luminance)",
        color="white", fontsize=13, y=1.03
    )

    for ax, region in zip(axes, regions):
        ax.set_facecolor("#161B22")
        subset = sample[sample["region"] == region]

        sc = ax.scatter(
            subset["population_density"],
            subset["PM2.5"],
            c=subset["PM2.5"],
            cmap=cmap, norm=norm,
            alpha=0.55, s=15, rasterized=True
        )

        # Add a light trend line (industrial vs residential distinction)
        for zone, ls in [("Industrial", "--"), ("Residential", "-")]:
            z_sub = subset[subset["zone"] == zone]
            if len(z_sub) > 10:
                z = np.polyfit(z_sub["population_density"], z_sub["PM2.5"], 1)
                p = np.poly1d(z)
                xs = np.linspace(z_sub["population_density"].min(),
                                 z_sub["population_density"].max(), 100)
                colour = "#E63946" if zone == "Industrial" else "#A8DADC"
                ax.plot(xs, p(xs), lw=1.5, ls=ls, color=colour,
                        label=zone, alpha=0.85)

        ax.set_title(f"Region: {region}", color="white", fontsize=11, pad=6)
        ax.set_xlabel("Population Density (per km²)", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#333")
        if ax == axes[0]:
            ax.set_ylabel("Daily Mean PM2.5 (µg/m³)", color="white", fontsize=9)
        leg = ax.legend(frameon=False, fontsize=7)
        for t in leg.get_texts():
            t.set_color("white")

    # Shared colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.02, shrink=0.8)
    cbar.set_label("PM2.5 (µg/m³)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[task4_audit] Small Multiples saved → {save_path}")
    return fig


# ── Rainbow vs Viridis Demonstration ─────────────────────────────────────────
def plot_colormap_comparison(save_path: str | None = None) -> plt.Figure:
    """
    Side-by-side demonstration of why Viridis outperforms Rainbow/Jet
    for displaying quantitative pollution data.
    """
    x = np.linspace(0, 1, 500)
    gradient = np.vstack([x, x])

    fig, axes = plt.subplots(2, 1, figsize=(10, 3))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle("Colourmap Comparison: Sequential vs Rainbow",
                 color="white", fontsize=12)

    cmaps = [("viridis", "[OK] Viridis (Sequential) -- perceptually uniform luminance"),
             ("jet",     "[AVOID] Jet/Rainbow -- false banding artefacts (cyan, green jumps)")]

    for ax, (cmap, label) in zip(axes, cmaps):
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.set_yticks([])
        ax.set_xticks([0, 125, 250, 375, 499])
        ax.set_xticklabels(["0", "25", "50", "75", "100 µg/m³"],
                           color="white", fontsize=9)
        ax.set_ylabel(label, color="white", fontsize=8, rotation=0,
                      labelpad=5, ha="left", va="center")
        ax.tick_params(colors="white")
        ax.spines[:].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[task4_audit] Colourmap comparison saved → {save_path}")
    return fig


# ── Runner ────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame) -> dict:
    """Full Task 4 pipeline step."""
    sm_path       = os.path.join(OUTPUT_DIR, "task4_small_multiples.png")
    cmap_path     = os.path.join(OUTPUT_DIR, "task4_colormap_comparison.png")
    fig_sm        = plot_small_multiples(df, save_path=sm_path)
    fig_cmap      = plot_colormap_comparison(save_path=cmap_path)
    return {"fig_small_multiples": fig_sm, "fig_colormap": fig_cmap}
