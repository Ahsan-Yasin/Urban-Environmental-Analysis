"""
pipeline/task2_temporal.py
───────────────────────────
Task 2: High-Density Temporal Analysis
Visualises PM2.5 readings from 100 sensors across all 365 days in a
compact, overplot-free heatmap and performs periodic signature analysis.

Why a heatmap?
  • 100 overlapping time-series → unreadable spaghetti chart
  • Heatmap encodes magnitude as colour (no overplotting by definition)
  • Preserves both sensor identity (rows) and time (columns)
  • High data-ink ratio: colour does the work, no redundant axes

Periodic Signature Analysis:
  • Daily (24 h) pattern: computed from hourly average across all sensors
  • Monthly (30-day) pattern: computed from daily average autocorrelation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, HEALTH_THRESHOLD


# ── Heatmap ───────────────────────────────────────────────────────────────────
def build_daily_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to daily mean PM2.5 per sensor,
    then pivot to (sensor × date) matrix for heatmap rendering.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    daily = (df.groupby(["sensor_id", "date"])["PM2.5"]
               .mean()
               .reset_index())
    pivot = daily.pivot(index="sensor_id", columns="date", values="PM2.5")
    # Sort sensors: Industrial first, then Residential (both alphabetically)
    pivot = pivot.sort_index()
    return pivot


def plot_heatmap(
        df: pd.DataFrame,
        save_path: str | None = None
) -> plt.Figure:
    """
    Renders the sensor × day PM2.5 heatmap.

    Design principles applied:
      • YlOrRd sequential colormap — perceptually uniform luminance
      • vmax=150 clips extreme outliers so the bulk is visible
      • One annotation line at Health Threshold (35 µg/m³) level via colorbar
      • Minimal ticks (every 30 days on x-axis, every 10 sensors on y-axis)
      • No gridlines, no 3D, no shadow
    """
    pivot = build_daily_pivot(df)

    fig, ax = plt.subplots(figsize=(20, 8))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")

    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        vmin=0, vmax=120,
        ax=ax,
        xticklabels=30,   # show one label every 30 days
        yticklabels=10,   # show one label every 10 sensors
        cbar_kws={"label": "Daily Mean PM2.5 (µg/m³)", "shrink": 0.6}
    )

    # Threshold annotation on colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(HEALTH_THRESHOLD, color="white", linewidth=1.5, linestyle="--")
    cbar.ax.text(2.2, HEALTH_THRESHOLD, f"  ← Health\n  Threshold\n  ({HEALTH_THRESHOLD})",
                 va="center", color="white", fontsize=7)
    cbar.set_label("Daily Mean PM2.5 (µg/m³)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("PM2.5 Temporal Heatmap — 100 Sensors × 365 Days (2025)",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("Date", color="white", fontsize=11)
    ax.set_ylabel("Sensor ID", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=8)

    # Horizontal divider line between Industrial (S001–S050) and Residential
    n_industrial = pivot.index.str.startswith("S0") & \
        (pivot.index.str[1:].astype(int) <= 50)
    n_ind_count = n_industrial.sum()
    ax.axhline(n_ind_count, color="#A8DADC", lw=1.5, ls="--")
    ax.text(5, n_ind_count + 0.5, "← Residential", color="#A8DADC", fontsize=8)
    ax.text(5, n_ind_count - 1.5, "← Industrial",  color="#E63946", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[task2_temporal] Heatmap saved → {save_path}")
    return fig


# ── Periodic Signature ────────────────────────────────────────────────────────
def plot_periodic_signature(df: pd.DataFrame, save_path: str | None = None) -> plt.Figure:
    """
    Two-panel figure revealing:
      Panel A — Daily (24 h) cycle: hourly mean PM2.5 averaged across all sensors
      Panel B — Monthly autocorrelation: 30-day rolling mean shows seasonal shifts
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"]     = df["datetime"].dt.hour
    df["date"]     = df["datetime"].dt.date

    # Panel A: hourly average
    hourly_mean = df.groupby("hour")["PM2.5"].mean()

    # Panel B: daily mean → 7-day rolling average to smooth noise
    daily_mean  = df.groupby("date")["PM2.5"].mean().reset_index()
    daily_mean["date"] = pd.to_datetime(daily_mean["date"])
    daily_mean = daily_mean.sort_values("date")
    daily_mean["rolling7"] = daily_mean["PM2.5"].rolling(7, center=True).mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#0F1117")

    # ── Panel A ──────────────────────────────────────────────────────────────
    ax1.set_facecolor("#0F1117")
    ax1.plot(hourly_mean.index, hourly_mean.values,
             color="#FFD166", lw=2.5, marker="o", markersize=4)
    ax1.fill_between(hourly_mean.index, hourly_mean.values,
                     alpha=0.25, color="#FFD166")
    ax1.set_xlabel("Hour of Day", color="white", fontsize=11)
    ax1.set_ylabel("Mean PM2.5 (µg/m³)", color="white", fontsize=11)
    ax1.set_title("Daily (24-Hour) Traffic Cycle Pattern",
                  color="white", fontsize=12, pad=10)
    ax1.tick_params(colors="white")
    ax1.set_xticks(range(0, 24, 2))
    ax1.spines[:].set_color("#333")
    ax1.axvline(8,  color="#E63946", lw=1.2, ls="--", label="Morning rush (08:00)")
    ax1.axvline(18, color="#E63946", lw=1.2, ls="--", label="Evening rush (18:00)")
    leg = ax1.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color("white")

    # ── Panel B ──────────────────────────────────────────────────────────────
    ax2.set_facecolor("#0F1117")
    ax2.plot(daily_mean["date"], daily_mean["PM2.5"],
             color="#457B9D", lw=0.8, alpha=0.5, label="Daily mean")
    ax2.plot(daily_mean["date"], daily_mean["rolling7"],
             color="#A8DADC", lw=2.5, label="7-day rolling mean")
    ax2.set_xlabel("Month (2025)", color="white", fontsize=11)
    ax2.set_ylabel("Mean PM2.5 (µg/m³)", color="white", fontsize=11)
    ax2.set_title("Monthly (Seasonal) Pollution Shift",
                  color="white", fontsize=12, pad=10)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#333")
    ax2.axhline(HEALTH_THRESHOLD, color="#FFD166", lw=1.2, ls="--",
                label=f"Health Threshold ({HEALTH_THRESHOLD}µg/m³)")
    leg2 = ax2.legend(frameon=False, fontsize=9)
    for t in leg2.get_texts():
        t.set_color("white")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[task2_temporal] Periodic signature saved → {save_path}")
    return fig


# ── Runner ────────────────────────────────────────────────────────────────────
def run(df: pd.DataFrame) -> dict:
    """Full Task 2 pipeline step."""
    heatmap_path  = os.path.join(OUTPUT_DIR, "task2_heatmap.png")
    periodic_path = os.path.join(OUTPUT_DIR, "task2_periodic.png")
    fig_heatmap   = plot_heatmap(df,  save_path=heatmap_path)
    fig_periodic  = plot_periodic_signature(df, save_path=periodic_path)
    return {"fig_heatmap": fig_heatmap, "fig_periodic": fig_periodic}
