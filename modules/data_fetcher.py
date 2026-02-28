"""
data_fetcher.py
Module to fetch air quality data from OpenAQ API for 100 stations for 2025.
Handles chunked downloads and saves data as CSVs for big data handling.
"""
import requests
import pandas as pd
import os
from datetime import datetime, timedelta

API_KEY = "24e40725296a25ee3924d78ffd9993ebdda1d8533292d548b9ed75bb8dbdce6b"
BASE_URL = "https://api.openaq.org/v2/measurements"
DATA_DIR = "data"

# List of 100 station IDs (to be filled with actual IDs after initial fetch)
STATION_IDS = []  # Placeholder, will be filled by fetch_station_ids()

PARAMETERS = ["pm25", "pm10", "no2", "o3", "temperature", "humidity"]
START_DATE = "2025-01-01T00:00:00+00:00"
END_DATE = "2025-12-31T23:59:59+00:00"


def fetch_station_ids(limit=100):
    """Fetch 100 unique station IDs from OpenAQ."""
    url = "https://api.openaq.org/v2/locations"
    params = {
        "limit": limit,
        "page": 1,
        "offset": 0,
        "sort": "desc",
        "order_by": "measurements",
        # "country_id": None,  # Global
    }
    headers = {"x-api-key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    ids = [loc["id"] for loc in data["results"]]
    return ids


def fetch_data_for_station(station_id, parameters, start_date, end_date, chunk_days=7):
    """Fetch data for a single station in weekly chunks for all parameters."""
    all_data = []
    start = datetime.fromisoformat(start_date.replace("+00:00", ""))
    end = datetime.fromisoformat(end_date.replace("+00:00", ""))
    while start < end:
        chunk_end = min(start + timedelta(days=chunk_days), end)
        for param in parameters:
            params = {
                "location_id": station_id,
                "parameter": param,
                "date_from": start.isoformat() + "+00:00",
                "date_to": chunk_end.isoformat() + "+00:00",
                "limit": 10000,
                "api_key": API_KEY
            }
            resp = requests.get(BASE_URL, params=params)
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    df["parameter"] = param
                    all_data.append(df)
        start = chunk_end
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def save_station_data(station_id, df):
    """Save station data as CSV in the data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    file_path = os.path.join(DATA_DIR, f"station_{station_id}_2025.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved data for station {station_id} to {file_path}")


def main():
    print("Fetching station IDs...")
    station_ids = fetch_station_ids()
    print(f"Fetched {len(station_ids)} station IDs.")
    for sid in station_ids:
        print(f"Fetching data for station {sid}...")
        df = fetch_data_for_station(sid, PARAMETERS, START_DATE, END_DATE)
        if not df.empty:
            save_station_data(sid, df)
        else:
            print(f"No data found for station {sid}.")

if __name__ == "__main__":
    main()
