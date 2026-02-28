"""
verify.py - Quick verification script for all pipeline tasks
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")

from pipeline.data_generator import generate_city_data
from pipeline import task1_pca, task2_temporal, task3_distribution, task4_audit
import numpy as np

print("=== PIPELINE VERIFICATION ===")
df = generate_city_data()
print(f"Data shape : {df.shape}")
print(f"Zones      : {df['zone'].value_counts().to_dict()}")
print(f"Sensors    : {df['sensor_id'].nunique()}")
print(f"Memory     : {df.memory_usage(deep=True).sum()/1e6:.1f} MB")
print()

r1 = task1_pca.run(df)
ev = r1["pca"].explained_variance_ratio_ * 100
print(f"Task 1 OK  - PC1={ev[0]:.1f}%  PC2={ev[1]:.1f}%  Total={sum(ev):.1f}%")

r2 = task2_temporal.run(df)
print("Task 2 OK  - heatmap + periodic figures")

r3 = task3_distribution.run(df)
ts = r3["tail_stats"]
print(f"Task 3 OK  - 99th pctl={ts['p99']:.2f} µg/m³  extreme={ts['pct_extreme_hazard']:.3f}%")

r4 = task4_audit.run(df)
print("Task 4 OK  - small multiples + colormap comparison")

print()
print("=== Output files ===")
from config import OUTPUT_DIR
for f in sorted(os.listdir(OUTPUT_DIR)):
    p = os.path.join(OUTPUT_DIR, f)
    print(f"  {f:40s}  {os.path.getsize(p)/1024:.0f} KB")

print()
print("ALL TASKS PASSED")
