"""
pipeline/data_fetcher.py
─────────────────────────
Real OpenAQ v3 API Data Fetcher
Fetches air quality data for 100 global stations for the full year 2025.
Designed for big-data robustness:
  • Chunked monthly date windows to avoid API response size limits
  • Exponential back-off retry on transient failures
  • Per-station CSV caching so interrupted runs can resume
  • Header-based auth (X-API-Key)
  • Rate-limited with time.sleep to respect API limits

Usage:
    python -m pipeline.data_fetcher
    python -m pipeline.data_fetcher --stations 10 --resume
"""

import os
import sys
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_KEY, OPENAQ_BASE_URL, DATA_DIR, NUM_STATIONS

# ── API Settings ─────────────────────────────────────────────────────────────
HEADERS       = {"X-API-Key": API_KEY, "Accept": "application/json"}
MAX_RETRIES   = 3
RETRY_BACKOFF = 2.0     # seconds (doubles on each retry)
RATE_LIMIT    = 0.5     # seconds between requests
PAGE_LIMIT    = 1000    # max results per page

PARAMETERS    = ["pm25", "pm10", "no2", "o3", "temperature", "humidity"]
PARAM_RENAME  = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2",
                 "o3": "Ozone", "temperature": "Temperature", "humidity": "Humidity"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get(url: str, params: dict) -> dict | None:
    """GET with retry + exponential back-off."""
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 429:          # rate limited
                print(f"    Rate-limited. Waiting {delay}s …")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {exc}")
            time.sleep(delay)
            delay *= 2
    return None


# ── Station Fetching ──────────────────────────────────────────────────────────
def fetch_location_ids(n: int = NUM_STATIONS) -> list[int]:
    """
    Fetch n location IDs from OpenAQ v3. 
    We filter by providers_id=113 (US EPA AirNow) to guarantee highly 
    active, currently transmitting stations with reliable 2025 data.
    """
    url    = f"{OPENAQ_BASE_URL}/locations"
    params = {
        "limit"       : n,
        "page"        : 1,
        "providers_id": 113
    }
    data = _get(url, params)
    if not data or "results" not in data:
        print("[data_fetcher] WARNING: Could not fetch location IDs. "
              "Check API key and network.")
        return []
    ids = [loc["id"] for loc in data["results"]]
    print(f"[data_fetcher] Fetched {len(ids)} location IDs.")
    return ids


# ── Measurement Fetching ──────────────────────────────────────────────────────
def fetch_sensor_ids_for_location(location_id: int) -> list[int]:
    """Get all sensor IDs under a given location."""
    url  = f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors"
    data = _get(url, {})
    if not data:
        return []
    return [s["id"] for s in data.get("results", [])]


def fetch_measurements_for_sensor(
        sensor_id: int,
        start: datetime,
        end: datetime
) -> list[dict]:
    """
    Fetch all measurements for one sensor between start and end (UTC),
    paginating through results automatically.
    """
    url      = f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements"
    results  = []
    page     = 1

    while True:
        params = {
            "datetime_from": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to"  : end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit"        : PAGE_LIMIT,
            "page"         : page,
        }
        data = _get(url, params)
        time.sleep(RATE_LIMIT)

        if not data or not data.get("results"):
            break
        results.extend(data["results"])
        if len(data["results"]) < PAGE_LIMIT:
            break            # last page
        page += 1

    return results


def fetch_station_yearly(
        location_id: int,
        year: int = 2024,
        chunk_months: int = 1
) -> pd.DataFrame:
    """
    Fetch a full year of data for one location in monthly chunks.
    Returns a tidy DataFrame (one row per measurement).
    """
    csv_path   = os.path.join(DATA_DIR, f"station_{location_id}_2025.csv")
    if os.path.exists(csv_path):
        print(f"  [cache hit] {csv_path}")
        return pd.read_csv(csv_path)

    sensor_ids = fetch_sensor_ids_for_location(location_id)
    if not sensor_ids:
        print(f"  [skip] No sensors for loca tion {location_id}")
        return pd.DataFrame()

    all_rows = []
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end   = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
  
    cur = start
    while cur < end:
        # Monthly chunk
        if cur.month == 12:
            chunk_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        else:
            next_month = cur.replace(day=1, month=cur.month + chunk_months)
            chunk_end  = min(next_month - timedelta(seconds=1), end)

        for sid in sensor_ids:
            rows = fetch_measurements_for_sensor(sid, cur, chunk_end)
            for r in rows:
                # Normalise OpenAQ v3 response structure
                value = r.get("value")
                param = r.get("parameter", {})
                dt    = r.get("period", {}).get("datetime_from", {}).get("utc", "")
                if value is not None and dt:
                    all_rows.append({
                        "location_id" : location_id,
                        "sensor_id"   : sid,
                        "datetime"    : dt,
                        "parameter"   : param.get("name", "unknown"),
                        "value"       : value,
                        "unit"        : param.get("units", ""),
                    })
        cur = chunk_end + timedelta(seconds=1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df):,} rows → {csv_path}")
    return df


# ── Pivot & Merge ─────────────────────────────────────────────────────────────
def pivot_and_merge(location_ids: list[int]) -> pd.DataFrame:
    """
    Load all per-station CSVs, pivot parameter rows to columns,
    and merge into a single DataFrame matching the synthetic schema.
    """
    frames = []
    for lid in location_ids:
        csv_path = os.path.join(DATA_DIR, f"station_{lid}_2025.csv")
        if not os.path.exists(csv_path):
            continue
        df     = pd.read_csv(csv_path)
        if df.empty:
            continue
        df_piv = (df.pivot_table(
                      index=["location_id", "sensor_id", "datetime"],
                      columns="parameter",
                      values="value",
                      aggfunc="mean")
                  .reset_index()
                  .rename(columns=PARAM_RENAME))
        frames.append(df_piv)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Main Entry ────────────────────────────────────────────────────────────────
def run_fetch(n_stations: int = NUM_STATIONS) -> pd.DataFrame:
    """Fetch data for n_stations and return merged DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[data_fetcher] Fetching {n_stations} stations from OpenAQ v3 …")
    location_ids = fetch_location_ids(n_stations)

    if not location_ids:
        print("[data_fetcher] No location IDs fetched. "
              "Using synthetic data instead.")
        return pd.DataFrame()

    for i, lid in enumerate(location_ids, start=1):
        print(f"[{i}/{len(location_ids)}] Location {lid}")
        fetch_station_yearly(lid)

    merged = pivot_and_merge(location_ids)
    merged_path_parquet = os.path.join(DATA_DIR, "real_air_quality_merged.parquet")
    merged_path_csv = os.path.join(DATA_DIR, "real_air_quality_merged.csv")
    if not merged.empty:
        merged.to_parquet(merged_path_parquet, index=False, compression="snappy")
        merged.to_csv(merged_path_csv, index=False)
        print(f"[data_fetcher] Merged real data saved → {merged_path_parquet} and {merged_path_csv}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAQ data fetcher")
    parser.add_argument("--stations", type=int, default=NUM_STATIONS,
                        help="Number of stations to fetch (default: 100)")
    args = parser.parse_args()
    run_fetch(n_stations=args.stations)
