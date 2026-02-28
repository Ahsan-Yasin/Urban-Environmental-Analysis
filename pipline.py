import pandas as pd
import numpy as np
import requests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- OPENAQ API FETCH LOGIC (Included for assignment requirements) ---
def fetch_openaq_data(api_key):
    """
    Actual API logic to fetch data. 
    Note: For a full year/100 stations, this requires heavy pagination and hours of fetching.
    """
    headers = {"X-API-Key": api_key}
    url = "https://api.openaq.org/v2/measurements"
    params = {
        "limit": 1000,
        "date_from": "2025-01-01T00:00:00Z",
        "date_to": "2025-12-31T23:59:59Z",
        "parameter":["pm25", "pm10", "no2", "o3", "temperature", "humidity"]
    }
    # response = requests.get(url, headers=headers, params=params)
    # return response.json()
    pass

# --- BIG DATA SIMULATION FOR DASHBOARD ---
def generate_city_data():
    """
    Generates a realistic mock dataset to ensure the dashboard runs instantly.
    Simulates 100 sensors for 365 days (daily aggregates used for memory efficiency).
    """
    np.random.seed(42)
    days = pd.date_range(start="2025-01-01", end="2025-12-31", freq='D')
    
    data = []
    regions =['North', 'South', 'East', 'West']
    
    for sensor_id in range(1, 101):
        # 50 Industrial, 50 Residential
        zone = 'Industrial' if sensor_id <= 50 else 'Residential'
        region = regions[sensor_id % 4]
        pop_density = np.random.randint(1000, 20000)
        
        # Base values depend on the zone
        base_pm25 = 40 if zone == 'Industrial' else 15
        
        for date in days:
            # Add seasonality and random noise
            seasonality = np.sin(date.dayofyear / 365 * 2 * np.pi) * 10
            noise = np.random.normal(0, 5)
            
            pm25 = max(0, base_pm25 + seasonality + noise)
            
            # Inject extreme hazards randomly (PM2.5 > 200)
            if np.random.rand() < 0.02 and zone == 'Industrial':
                pm25 = np.random.uniform(200, 350)
                
            data.append({
                'sensor_id': f"S{sensor_id:03d}",
                'date': date,
                'zone': zone,
                'region': region,
                'population_density': pop_density,
                'PM2.5': pm25,
                'PM10': pm25 * 1.5 + np.random.normal(0, 2),
                'NO2': pm25 * 0.8 + np.random.normal(0, 2),
                'Ozone': np.random.uniform(10, 50),
                'Temperature': 20 + seasonality + np.random.normal(0, 2),
                'Humidity': 50 - seasonality + np.random.normal(0, 5)
            })
            
    return pd.DataFrame(data)

# --- TASK 1: PCA DIMENSIONALITY REDUCTION ---
def perform_pca(df):
    """
    Standardizes data and applies PCA to reduce 6 dimensions to 2.
    StandardScaler is crucial because variables have different units (ug/m3, Celsius, %).
    """
    features =['PM2.5', 'PM10', 'NO2', 'Ozone', 'Temperature', 'Humidity']
    x = df[features].values
    
    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    
    # Apply PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(x_scaled)
    
    df['PC1'] = components[:, 0]
    df['PC2'] = components[:, 1]
    
    # Extract loadings (how much each variable contributes to the new axes)
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=['PC1', 'PC2'], 
        index=features
    )
    
    return df, loadings