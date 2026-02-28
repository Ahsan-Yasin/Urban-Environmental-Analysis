"""
config.py
─────────
Central configuration for the Urban Environmental Intelligence pipeline.
All constants and paths are defined here so every module stays DRY.
"""

import os

# ── API ──────────────────────────────────────────────────────────────────────
API_KEY = "24e40725296a25ee3924d78ffd9993ebdda1d8533292d548b9ed75bb8dbdce6b"
OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# Number of sensor stations to analyse
NUM_STATIONS = 100

# Date range for 2025 (full year)
START_DATE = "2025-01-01T00:00:00Z"
END_DATE   = "2025-12-31T23:59:59Z"

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PARQUET_PATH = os.path.join(DATA_DIR, "synthetic_air_quality.parquet")

for _dir in (DATA_DIR, OUTPUT_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── FEATURES ─────────────────────────────────────────────────────────────────
FEATURES = ["PM2.5", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]

# ── THRESHOLDS ───────────────────────────────────────────────────────────────
HEALTH_THRESHOLD   = 35.0    # PM2.5 µg/m³  (Health Threshold Violation)
EXTREME_THRESHOLD  = 200.0   # PM2.5 µg/m³  (Extreme Hazard)

# ── RANDOM SEED (reproducibility) ────────────────────────────────────────────
RANDOM_SEED = 42
