"""
pipeline/data_generator.py
──────────────────────────
Big-Data Simulation Engine
Generates 100 sensors × 365 days × 24 hours = 876,000 rows of realistic
air quality data. Uses chunked NumPy operations to simulate multi-gigabyte
handling. Outputs a Parquet file (columnar, compressed) for fast I/O.

Environmental Simulation Model:
  • Zone-specific base values  (Industrial >> Residential)
  • Annual sinusoidal seasonality (winter inversions → higher pollution)
  • Daily 24-hour traffic cycle  (rush hours at 08:00 & 18:00)
  • 2% extreme-hazard injection  (PM2.5 > 200) for Industrial sensors
  • Inter-variable correlations  (PM10 ≈ 1.5×PM2.5, NO2 ≈ 0.8×PM2.5)
"""

import numpy as np
import pandas as pd
import os
import sys

# Allow running as script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, PARQUET_PATH, NUM_STATIONS, RANDOM_SEED
)

# ── Constants ────────────────────────────────────────────────────────────────
ZONES    = ["Industrial", "Residential"]
REGIONS  = ["North", "South", "East", "West"]
DAYS     = pd.date_range("2025-01-01", "2025-12-31", freq="D")
HOURS    = list(range(24))
N_DAYS   = len(DAYS)          # 365
N_HOURS  = 24
CHUNK_SIZE = 10               # process 10 sensors at a time (big-data pattern)


def _daily_traffic_factor(hour: int) -> float:
    """Return a multiplier (1.0–2.0) peaking at rush hours 08:00 & 18:00."""
    # Two Gaussian peaks: morning (8h) and evening (18h)
    morning = np.exp(-0.5 * ((hour - 8)  / 2) ** 2)
    evening = np.exp(-0.5 * ((hour - 18) / 2) ** 2)
    return 1.0 + (morning + evening) * 0.5


def _seasonality(day_of_year: np.ndarray) -> np.ndarray:
    """
    Sinusoidal seasonal signal: peaks in winter (Jan/Dec → high pollution
    due to thermal inversions), trough in summer.
    """
    return np.sin((day_of_year / 365.0) * 2 * np.pi + np.pi) * 10  # +π → winter peak


def generate_city_data(force: bool = False) -> pd.DataFrame:
    """
    Generates or loads the city-wide air quality dataset.

    Parameters
    ----------
    force : bool
        If True, regenerate even if Parquet file already exists.

    Returns
    -------
    pd.DataFrame with columns:
        sensor_id, zone, region, population_density,
        datetime,  PM2.5, PM10, NO2, Ozone, Temperature, Humidity
    """
    if os.path.exists(PARQUET_PATH) and not force:
        print(f"[data_generator] Loading cached data from {PARQUET_PATH}")
        return pd.read_parquet(PARQUET_PATH)

    print(f"[data_generator] Generating {NUM_STATIONS} × {N_DAYS} × {N_HOURS} = "
          f"{NUM_STATIONS * N_DAYS * N_HOURS:,} rows …")

    rng = np.random.default_rng(RANDOM_SEED)
    chunks = []

    # Pre-compute time arrays (shared across sensors)
    day_of_year = np.tile(
        np.repeat(np.arange(1, N_DAYS + 1), N_HOURS),
        1
    )                                         # shape (N_DAYS*N_HOURS,)
    hour_arr    = np.tile(HOURS, N_DAYS)      # shape (N_DAYS*N_HOURS,)
    dt_index    = pd.date_range("2025-01-01", periods=N_DAYS * N_HOURS, freq="h")

    seasonality = _seasonality(day_of_year)   # (N_DAYS*N_HOURS,)
    traffic     = np.array([_daily_traffic_factor(h) for h in hour_arr])

    for chunk_start in range(0, NUM_STATIONS, CHUNK_SIZE):
        chunk_end    = min(chunk_start + CHUNK_SIZE, NUM_STATIONS)
        n_sensors    = chunk_end - chunk_start

        for i in range(n_sensors):
            sensor_num = chunk_start + i + 1            # 1-indexed
            zone       = ZONES[0] if sensor_num <= 50 else ZONES[1]
            region     = REGIONS[sensor_num % 4]
            pop_density = int(rng.integers(1_000, 20_001))

            # Base PM2.5 depends on zone
            base_pm25 = 40.0 if zone == "Industrial" else 12.0

            n_ts = N_DAYS * N_HOURS
            noise_pm25 = rng.normal(0, 5, n_ts)

            pm25 = np.maximum(0, base_pm25 + seasonality + noise_pm25) * traffic

            # Inject extreme hazards (Industrial only, ~2% of time steps)
            if zone == "Industrial":
                extreme_mask = rng.random(n_ts) < 0.02
                pm25[extreme_mask] = rng.uniform(200, 350, extreme_mask.sum())

            pm10        = np.maximum(0, pm25 * 1.5 + rng.normal(0, 3, n_ts))
            no2         = np.maximum(0, pm25 * 0.8 + rng.normal(0, 3, n_ts))
            ozone       = rng.uniform(10, 60, n_ts)
            temperature = 15 + seasonality * (-1) + rng.normal(0, 2, n_ts)  # inverse season
            humidity    = 55 + seasonality * 0.5  + rng.normal(0, 5, n_ts)

            sensor_df = pd.DataFrame({
                "sensor_id"        : f"S{sensor_num:03d}",
                "zone"             : zone,
                "region"           : region,
                "population_density": pop_density,
                "datetime"         : dt_index,
                "day_of_year"      : day_of_year,
                "hour"             : hour_arr,
                "PM2.5"            : pm25.round(2),
                "PM10"             : pm10.round(2),
                "NO2"              : no2.round(2),
                "Ozone"            : ozone.round(2),
                "Temperature"      : temperature.round(2),
                "Humidity"         : humidity.round(2),
            })
            chunks.append(sensor_df)

        print(f"  … sensors {chunk_start + 1}–{chunk_end} generated")

    df = pd.concat(chunks, ignore_index=True)

    # Save as Parquet (columnar, compressed) — big-data best practice
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False, compression="snappy")
    print(f"[data_generator] Saved {len(df):,} rows → {PARQUET_PATH}")
    return df


if __name__ == "__main__":
    df = generate_city_data(force=True)
    print(df.info())
    print(df.describe())
