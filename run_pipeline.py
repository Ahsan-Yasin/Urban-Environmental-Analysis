"""
run_pipeline.py
───────────────
Modular CLI Orchestrator for the Urban Environmental Intelligence Pipeline.
Executes all 4 analysis tasks sequentially and saves output figures.

Usage:
    python run_pipeline.py              # Use synthetic data (default)
    python run_pipeline.py --real-data  # Fetch real data from OpenAQ first
    python run_pipeline.py --force-regen  # Regenerate synthetic data from scratch
"""

import os
import sys
import time
import argparse

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, PARQUET_PATH
from pipeline.data_generator import generate_city_data
from pipeline import task1_pca, task2_temporal, task3_distribution, task4_audit


def banner(msg: str) -> None:
    sep = "─" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


def run_pipeline(use_real_data: bool = False, force_regen: bool = False) -> None:
    """
    Master pipeline runner.

    Steps:
      1. Load / generate data
      2. Task 1 — PCA dimensionality reduction
      3. Task 2 — High-density temporal heatmap + periodic signature
      4. Task 3 — Distribution peak & tail plots
      5. Task 4 — Visual integrity audit → small multiples
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    # ── Step 0: Data Loading ──────────────────────────────────────────────────
    banner("Step 0 | Loading Data")

    if use_real_data:
        from pipeline.data_fetcher import run_fetch
        df_real = run_fetch()
        if df_real.empty:
            print("  Real data fetch returned empty. Falling back to synthetic data.")
            df = generate_city_data(force=force_regen)
        else:
            df = df_real
    else:
        df = generate_city_data(force=force_regen)

    print(f"  Loaded  : {len(df):,} rows  ×  {df.shape[1]} columns")
    print(f"  Sensors : {df['sensor_id'].nunique()}")
    print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ── Step 1: PCA ──────────────────────────────────────────────────────────
    banner("Task 1 | Dimensionality Challenge — PCA")
    t1 = time.time()
    results1 = task1_pca.run(df)
    print(f"  Done in {time.time()-t1:.1f}s")
    print(f"  Narrative: {results1['narrative'][:120]}…")

    # Task 2 needs the full df (including PC columns is fine)
    df_pca = results1["df_pca"]

    # ── Step 2: Temporal Heatmap ─────────────────────────────────────────────
    banner("Task 2 | High-Density Temporal Analysis")
    t2 = time.time()
    results2 = task2_temporal.run(df)
    print(f"  Done in {time.time()-t2:.1f}s")

    # ── Step 3: Distribution Tails ───────────────────────────────────────────
    banner("Task 3 | Distribution Modeling & Tail Integrity")
    t3 = time.time()
    results3 = task3_distribution.run(df)
    ts        = results3["tail_stats"]
    print(f"  99th percentile   : {ts['p99']:.2f} µg/m³")
    print(f"  % Extreme Hazard  : {ts['pct_extreme_hazard']:.2f}%")
    print(f"  Done in {time.time()-t3:.1f}s")

    # ── Step 4: Visual Integrity Audit ───────────────────────────────────────
    banner("Task 4 | Visual Integrity Audit — 3D Rejected → Small Multiples")
    t4 = time.time()
    results4 = task4_audit.run(df)
    print(f"  Done in {time.time()-t4:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Pipeline Complete")
    print(f"  Total time : {time.time()-t_total:.1f}s")
    print(f"  Outputs    : {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        print(f"    • {fname}  ({os.path.getsize(fpath)/1024:.0f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urban Environmental Intelligence Pipeline"
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="Fetch real data from OpenAQ API (slow, requires network)"
    )
    parser.add_argument(
        "--force-regen", action="store_true",
        help="Force regeneration of synthetic data even if cached Parquet exists"
    )
    args = parser.parse_args()
    run_pipeline(use_real_data=args.real_data, force_regen=args.force_regen)
