"""
pipeline/task3_distribution.py
────────────────────────────────
Task 3: Distribution Modeling & Tail Integrity
Reports extreme PM2.5 events (> 200 µg/m³) for Industrial sensors.

Two complementary plots:
  • Plot A — Standard KDE + Histogram (linear axes)
      Optimised to reveal the **peak** (mode) of the distribution.
      Problem: rare extreme values produce near-invisible bars at > 200.

  • Plot B — Log-Y Axis Histogram + ECDF overlay (log scale)
      Optimised to reveal the **tail** (extreme events).
      Log scale prevents tall bulk bars from hiding the sparse tail.
      An ECDF line shows cumulative probability without histogram bin-size bias.

Key statistic derived: 99th percentile of Industrial PM2.5.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, EXTREME_THRESHOLD


# ── Statistics ────────────────────────────────────────────────────────────────
def compute_tail_stats(series: pd.Series) -> dict:
    """Compute key tail statistics for the target variable."""
    p99      = float(np.percentile(series, 99))
    p999     = float(np.percentile(series, 99.9))
    extreme  = float((series > EXTREME_THRESHOLD).mean() * 100)   # %
    skewness = float(stats.skew(series.dropna()))
    return {
        "p99"              : p99,
        "p999"             : p999,
        "pct_extreme_hazard": extreme,
        "skewness"         : skewness,
    }


# ── Visualisations ────────────────────────────────────────────────────────────
def plot_distributions(
        series: pd.Series,
        tail_stats: dict,
        save_path: str | None = None
) -> plt.Figure:
    """
    Side-by-side distribution figure:
      Left  — KDE + histogram, linear axes (peaks)
      Right — Log-Y histogram + ECDF overlay    (tails)
    """
    p99  = tail_stats["p99"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0F1117")
    fig.suptitle(
        "Industrial Zone PM2.5 Distribution — Peaks vs. Tail Integrity",
        color="white", fontsize=14, y=1.02
    )

    # ── LEFT: Standard Histogram + KDE ───────────────────────────────────────
    ax1.set_facecolor("#0F1117")
    # Use 60 bins; clip extreme outliers at 300 for readability of the bulk
    data_clipped = series.clip(upper=300)
    sns.histplot(data_clipped, bins=60, kde=True,
                 color="#457B9D", alpha=0.7, ax=ax1,
                 line_kws={"lw": 2, "color": "#A8DADC"})

    ax1.axvline(p99, color="#FFD166", lw=2, ls="--",
                label=f"99th Pctl: {p99:.1f} µg/m³")
    ax1.axvline(EXTREME_THRESHOLD, color="#E63946", lw=2, ls="--",
                label=f"Extreme Hazard: {EXTREME_THRESHOLD} µg/m³")

    ax1.set_xlabel("PM2.5 (µg/m³)", color="white", fontsize=11)
    ax1.set_ylabel("Count", color="white", fontsize=11)
    ax1.set_title("Plot A: Peaks — Standard Histogram\n(Linear Axes)",
                  color="white", fontsize=12, pad=8)
    ax1.tick_params(colors="white")
    ax1.spines[:].set_color("#333")
    leg1 = ax1.legend(frameon=False, fontsize=9)
    for t in leg1.get_texts():
        t.set_color("white")

    ax1.annotate(
        "Extreme tail\nvisually lost →",
        xy=(EXTREME_THRESHOLD, 50),
        xytext=(EXTREME_THRESHOLD - 60, 500),
        arrowprops=dict(arrowstyle="->", color="#E63946"),
        color="#E63946", fontsize=8, ha="right"
    )

    # ── RIGHT: Log-Y Histogram + ECDF ────────────────────────────────────────
    ax2.set_facecolor("#0F1117")

    # Histogram with log Y to amplify the tail
    _, bin_edges, patches = ax2.hist(
        series, bins=80,
        color="#E63946", alpha=0.75, log=True, label="Log-Y Histogram"
    )

    # ECDF overlay on twin axis
    ax2_r = ax2.twinx()
    ax2_r.set_facecolor("#0F1117")
    sorted_vals = np.sort(series.dropna().values)
    ecdf        = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax2_r.plot(sorted_vals, ecdf, color="#FFD166", lw=1.5,
               label="ECDF", alpha=0.9)
    ax2_r.set_ylabel("Cumulative Probability", color="#FFD166", fontsize=10)
    ax2_r.tick_params(axis="y", colors="#FFD166", labelsize=8)
    ax2_r.spines[:].set_color("#333")

    # Threshold lines
    ax2.axvline(p99, color="#FFD166", lw=2, ls="--",
                label=f"99th Pctl: {p99:.1f} µg/m³")
    ax2.axvline(EXTREME_THRESHOLD, color="white", lw=2, ls="--",
                label=f"Extreme Hazard: {EXTREME_THRESHOLD} µg/m³")

    ax2.set_xlabel("PM2.5 (µg/m³)", color="white", fontsize=11)
    ax2.set_ylabel("Count (log scale)", color="white", fontsize=11)
    ax2.set_title("Plot B: Tails — Log-Y Histogram + ECDF\n(Tail-Honest View)",
                  color="white", fontsize=12, pad=8)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")

    # Combined legend
    handles = [
        mlines.Line2D([], [], color="#FFD166", ls="--",
                      label=f"99th Pctl: {p99:.1f} µg/m³"),
        mlines.Line2D([], [], color="white",   ls="--",
                      label=f"Extreme Hazard: {EXTREME_THRESHOLD} µg/m³"),
        mlines.Line2D([], [], color="#FFD166",
                      label="ECDF"),
    ]
    leg2 = ax2.legend(handles=handles, frameon=False, fontsize=9)
    for t in leg2.get_texts():
        t.set_color("white")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[task3_distribution] Saved → {save_path}")
    return fig


# ── Runner ────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame) -> dict:
    """Full Task 3 pipeline step."""
    industrial = df[df["zone"] == "Industrial"]["PM2.5"].dropna()
    tail_stats = compute_tail_stats(industrial)

    print(f"[task3_distribution] 99th pctl  = {tail_stats['p99']:.2f} µg/m³")
    print(f"[task3_distribution] % extreme  = {tail_stats['pct_extreme_hazard']:.2f}%")
    print(f"[task3_distribution] skewness   = {tail_stats['skewness']:.3f}")

    save_path = os.path.join(OUTPUT_DIR, "task3_distribution.png")
    fig       = plot_distributions(industrial, tail_stats, save_path=save_path)
    return {"fig": fig, "tail_stats": tail_stats, "series": industrial}
