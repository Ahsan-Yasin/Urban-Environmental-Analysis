"""
app.py
───────
Urban Environmental Intelligence — Interactive Streamlit Dashboard
Implements all 4 assignment tasks with interactive controls.
"""

import os
import sys
import streamlit as st
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import API_KEY, FEATURES, HEALTH_THRESHOLD, EXTREME_THRESHOLD
from pipeline.data_generator import generate_city_data
from pipeline import task1_pca, task2_temporal, task3_distribution, task4_audit

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark base */
    [data-testid="stAppViewContainer"] { background: #0F1117; }
    [data-testid="stSidebar"]          { background: #161B22; }
    [data-testid="stHeader"]           { background: transparent; }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #E8EAF0 !important; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(22, 27, 34, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        color: #FFD166 !important;
        font-weight: 700 !important;
    }

    /* Tab styling */
    [data-testid="stTabs"] button {
        color: #8B949E;
        font-weight: 600;
        font-size: 16px;
        padding: 12px 24px;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #FFD166;
        border-bottom: 3px solid #FFD166;
    }

    /* Divider */
    hr { border-color: #30363D; margin: 8px 0; }

    /* Info / success / error boxes */
    [data-testid="stAlert"] { border-radius: 8px; }

    /* Section header accent */
    .section-badge {
        display: inline-block;
        background: #1F6FEB;
        color: white;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌍 Dashboard")
    st.markdown("---")
    st.markdown("**Data Source**")
    st.code("OpenAQ Global Air Quality API", language=None)
    st.markdown(f"**API Key (masked)**\n`{API_KEY[:8]}…{API_KEY[-8:]}`")
    st.markdown("---")
    st.markdown("**Dataset Spec**")
    st.markdown("- 🏭 100 sensor nodes (50 Industrial / 50 Residential)\n"
                "- 📅 365 days × 24 hours = **876,000 rows**\n"
                "- 📊 6 environmental variables\n"
                "- 🗜️ Parquet (Snappy) columnar storage")
    st.markdown("---")
    st.markdown("**Thresholds**")
    st.markdown(f"- 🟡 Health Threshold: **{HEALTH_THRESHOLD} µg/m³**\n"
                f"- 🔴 Extreme Hazard: **{EXTREME_THRESHOLD} µg/m³**")
    st.markdown("---")
    sample_n = st.slider("Plot sample size", 1000, 876_000, 10_000, step=1000,
                         help="Number of data points to render in scatter plots (WARNING: Very large numbers may slow down rendering)")


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset… (first run only)")
def _load_data() -> pd.DataFrame:
    real_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_air_quality_merged.csv")
    real_parquet = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "real_air_quality_merged.parquet")
    
    # Priority 1: Real API data (CSV)
    if os.path.exists(real_csv):
        st.sidebar.success("✅ Connected to Real OpenAQ Data (CSV)!")
        return pd.read_csv(real_csv)
    # Priority 2: Real API data (Parquet)
    elif os.path.exists(real_parquet):
        st.sidebar.success("✅ Connected to Real OpenAQ Data (Parquet)!")
        return pd.read_parquet(real_parquet)
    
    # Priority 3: Fallback to synthetic if fetch hasn't run yet
    st.sidebar.warning("⚠️ Using synthetic test dataset. Run pipeline fetcher for real data.")
    return generate_city_data()

df = _load_data()

# Header
st.markdown("""
<div style="text-align:center;padding:16px 0 32px 0;">
  <h1 style="font-size:2.8rem; font-weight:800; color:#E8EAF0; margin:0; letter-spacing:-0.03em;">Urban Environmental Intelligence</h1>
  <p style="color:#8B949E; font-size:1.1rem; margin-top:8px; font-weight:400;">
    Smart City Diagnostic Engine
  </p>
</div>
""", unsafe_allow_html=True)

# KPI Row
col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
industrial = df[df["zone"] == "Industrial"]["PM2.5"]
with col_k1:
    st.metric("Total Records", f"{len(df):,}")
with col_k2:
    st.metric("Sensor Nodes", df["sensor_id"].nunique())
with col_k3:
    st.metric("Mean PM2.5 (Ind.)", f"{industrial.mean():.1f} µg/m³")
with col_k4:
    p99 = float(np.percentile(industrial, 99))
    st.metric("99th Pctl PM2.5", f"{p99:.1f} µg/m³")
with col_k5:
    pct_extreme = (industrial > EXTREME_THRESHOLD).mean() * 100
    st.metric("Extreme Events", f"{pct_extreme:.2f}%")

st.divider()

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 PCA",
    "🌡️ Temporal",
    "📊 Distributions",
    "🔍 Audit",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: PCA
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-badge">25 MARKS</span>', unsafe_allow_html=True)
    st.markdown("### The Dimensionality Challenge")
    st.markdown(
        "Standard scatter plots fail with high-dimensional data. "
        "**PCA** projects data into 2 dimensions while preserving maximum variance."
    )

    @st.cache_data(show_spinner="Running PCA …")
    def _run_pca(sample_size):
        return task1_pca.run(df)

    results1 = _run_pca(sample_n)
    df_pca   = results1["df_pca"]
    loadings = results1["loadings"]
    pca      = results1["pca"]

    # Plot
    fig1 = task1_pca.plot_pca(df_pca, loadings, pca, sample_n=sample_n)
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

    # Loadings table + narrative
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### Loadings")
        styled = loadings.style.background_gradient(cmap="RdBu_r", axis=None)\
                               .format("{:.3f}")
        st.dataframe(styled, use_container_width=True)

    with c2:
        st.markdown("#### Analysis")
        ev = pca.explained_variance_ratio_ * 100
        st.info(
            f"**PC1** explains **{ev[0]:.1f}%** of total variance.  \n"
            f"**PC2** explains **{ev[1]:.1f}%** of total variance.  \n"
            f"Combined: **{sum(ev):.1f}%** — sufficient for meaningful interpretation."
        )
        st.markdown(results1["narrative"])
        st.markdown("""
**Why PCA and not t-SNE?**
- PCA is **linear** and produces **loadings** — quantitative weights showing each 
  variable's contribution to each axis. This is mandatory for the assignment's 
  physical interpretation.
- t-SNE is non-linear; its axes have no interpretable meaning and cannot produce loadings.
- PCA is deterministic and scales to 876k rows in seconds.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: TEMPORAL HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-badge">25 MARKS</span>', unsafe_allow_html=True)
    st.markdown("### Temporal Analysis")
    st.markdown(
        "A **Sensor × Day heatmap** compactly encodes all 100 time-series, avoiding spaghetti charts."
    )

    @st.cache_data(show_spinner="Building heatmap …")
    def _run_task2():
        return task2_temporal.run(df)

    results2 = _run_task2()

    st.markdown("#### Heatmap (100 Sensors × 365 Days)")
    st.pyplot(results2["fig_heatmap"], use_container_width=True)
    plt.close(results2["fig_heatmap"])

    st.markdown("#### Periodic Signature")
    st.pyplot(results2["fig_periodic"], use_container_width=True)
    plt.close(results2["fig_periodic"])

    # ── Health Threshold Violation Ranking ───────────────────────────────────
    st.markdown("#### Health Violations (PM2.5 > 35 µg/m³)")
    st.markdown(
        "*Which neighborhoods consistently exceed safe limits?*"
    )

    @st.cache_data(show_spinner=False)
    def _violation_ranking():
        tmp = df.copy()
        tmp["date"] = pd.to_datetime(tmp["datetime"]).dt.date
        daily_max = tmp.groupby(["sensor_id", "zone", "region", "date"])["PM2.5"].mean().reset_index()
        violations = daily_max[daily_max["PM2.5"] > HEALTH_THRESHOLD]
        ranking = (
            violations.groupby(["sensor_id", "zone", "region"])
            .agg(
                days_violated   = ("date", "count"),
                mean_pm25       = ("PM2.5", "mean"),
                max_pm25        = ("PM2.5", "max")
            )
            .reset_index()
            .sort_values("days_violated", ascending=False)
            .head(20)
            .reset_index(drop=True)
        )
        ranking.index += 1
        ranking.columns = ["Sensor", "Zone", "Region",
                           "Days Violated", "Mean PM2.5 (µg/m³)", "Max PM2.5 (µg/m³)"]
        return ranking

    viol_df = _violation_ranking()
    st.dataframe(
        viol_df.style
               .background_gradient(subset=["Days Violated"], cmap="YlOrRd")
               .format({"Mean PM2.5 (µg/m³)": "{:.1f}", "Max PM2.5 (µg/m³)": "{:.1f}"}),
        use_container_width=True
    )

    total_sensors = df["sensor_id"].nunique()
    n_violators = viol_df.shape[0]
    st.caption(
        f"Showing top 20 of all sensors with ≥1 violation day. "
        f"Industrial sensors dominate the top ranks due to higher base PM2.5 levels."
    )

    st.markdown("""
**Analysis — Periodic Signature Conclusion:**

| Pattern | Evidence | Mechanism |
|---------|----------|-----------|
| **Daily (24-h)** | Two clear peaks at 08:00 and 18:00 in the hourly panel | Morning/evening traffic rush hours |
| **Monthly (Seasonal)** | Sinusoidal rolling-mean wave peaking Jan–Feb and dipping Jun–Aug | Winter thermal inversions trap pollutants near ground |

> **Conclusion:** Pollution events are driven by **BOTH** patterns simultaneously.
> The **dominant driver is seasonal (30-day shifts)** — winter inversion episodes multiply
> baseline concentrations across entire sensor networks. The **secondary driver is daily
> (24-hour traffic cycles)** — visible as within-day modulation on top of the seasonal envelope.
> The heatmap reveals this hierarchy clearly: broad horizontal intensity bands (seasonal) with
> fine vertical structure (daily rush hours), which 100 overlapping line charts cannot show.
    """)



# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: DISTRIBUTION TAILS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-badge">25 MARKS</span>', unsafe_allow_html=True)
    st.markdown("### Distribution Tails")

    zone_sel = st.selectbox("Select Zone", ["Industrial", "Residential"],
                            index=0, key="zone_t3")

    @st.cache_data(show_spinner="Computing distributions …")
    def _run_task3(zone):
        sub_df = df[df["zone"] == zone]
        series = sub_df["PM2.5"].dropna()
        tail   = task3_distribution.compute_tail_stats(series)
        fig    = task3_distribution.plot_distributions(series, tail)
        return fig, tail, series

    fig3, tail3, series3 = _run_task3(zone_sel)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("99th Percentile",  f"{tail3['p99']:.2f} µg/m³")
    s2.metric("99.9th Percentile", f"{tail3['p999']:.2f} µg/m³")
    s3.metric("% Extreme Hazard (>200)", f"{tail3['pct_extreme_hazard']:.3f}%")
    s4.metric("Distribution Skewness", f"{tail3['skewness']:.3f}")

    st.markdown(f"""
**Analysis:**

> **Probability of Extreme Hazard Event (PM2.5 > {EXTREME_THRESHOLD} µg/m³) =
> {tail3['pct_extreme_hazard']:.3f}%**  
> At the 99th percentile, PM2.5 = **{tail3['p99']:.1f} µg/m³** for the {zone_sel} zone.

| Plot | Axis Scale | Best For | Weakness |
|------|-----------|----------|---------|
| **Plot A** — Standard Histogram | Linear | Reveals **mode** (bulk peak) | Extreme values (>{EXTREME_THRESHOLD}) become invisible flat bars close to y=0 |
| **Plot B** — Log-Y + ECDF | Logarithmic Y | Reveals **tail** — hazard events > {EXTREME_THRESHOLD} µg/m³ are amplified | Bulk distribution compressed |

**Technical Justification — which plot is more "honest" for rare events?**  
Plot B (Log-Y + ECDF) is definitively more honest because:
1. **Log scale** gives rare events proportional visual weight. A bar at count=5 is not
   invisible next to a bar at count=60,000 — it occupies meaningful vertical space.
2. **ECDF** is bin-size agnostic — it doesn't require choosing bins that can hide or
   exaggerate the tail. It reads directly: P(PM2.5 > 200) = 1 − ECDF(200) ≈ {tail3['pct_extreme_hazard']:.3f}%.
3. Standard histograms with linear axes violate **data-ink ratio**: most ink goes to the
   bulk bars, while the critical tail region (the decision-relevant zone for public health
   policy) receives no ink at all.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4: VISUAL INTEGRITY AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-badge">25 MARKS</span>', unsafe_allow_html=True)
    st.markdown("### Visual Integrity Audit")

    st.error(
        "❌ **3D Bar Chart Proposal: REJECTED**\n\n"
        "**Lie Factor** (Tufte): Perspective distortion makes rear bars appear smaller "
        "than front bars of equal value. LF ≠ 1.0 → dishonest visualisation.\n\n"
        "**Data-Ink Ratio**: 3D perspective walls, floor shadows, and skewed axis grids "
        "consume ink that encodes zero additional data variables."
    )

    st.success(
        "✅ **Alternative Accepted: Small Multiples (FacetGrid)**\n\n"
        "One panel per Region. Each panel encodes Pollution (y), Population Density (x), "
        "and PM2.5 intensity (Viridis colour). Three variables, zero distortion."
    )

    @st.cache_data(show_spinner="Building small multiples …")
    def _run_task4():
        return task4_audit.run(df)

    results4 = _run_task4()

    st.markdown("#### Small Multiples by Region")
    st.pyplot(results4["fig_small_multiples"], use_container_width=True)
    plt.close(results4["fig_small_multiples"])

    st.markdown("#### Viridis vs Rainbow")
    st.pyplot(results4["fig_colormap"], use_container_width=True)
    plt.close(results4["fig_colormap"])

    st.markdown("""
**Colour Scale Justification (Sequential vs Rainbow):**

| Property | Viridis ✅ | Jet/Rainbow ❌ |
|----------|-----------|--------------|
| Luminance progression | Monotonically increasing (dark → light) | Non-monotonic; false banding at cyan/green |
| Colourblind-safe | Yes (blue–green–yellow) | No |
| Data-to-perception mapping | Linear | Distorted — equal data steps ≠ equal perceived steps |
| Artefacts | None | False "edges" appear where no data discontinuity exists |

**Conclusion:** The **Viridis sequential colormap** is chosen because human perception of 
luminance (lightness) is the most robust channel for encoding ordered quantitative data. 
Rainbow maps create perceptual artefacts that can mislead decision-makers about pollution boundaries.
    """)

st.divider()
st.markdown(
    "<p style='text-align:center;color:#555;font-size:12px;'>"
    "Urban Environmental Intelligence Dashboard • Data: OpenAQ API + Synthetic Simulation • "
    "Built with Streamlit, Matplotlib, Seaborn, Scikit-Learn"
    "</p>",
    unsafe_allow_html=True
)