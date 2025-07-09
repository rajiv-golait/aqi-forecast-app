import numpy as np
from sklearn.neighbors import BallTree
import pandas as pd

# Haversine distance is built into BallTree with metric='haversine'

class StationLocator:
    def __init__(self, stations):
        # stations: list of dicts with 'id', 'lat', 'lon', 'aqi'
        self.stations = stations
        self.coords = np.array([[s['lat'], s['lon']] for s in stations])
        self.tree = BallTree(np.radians(self.coords), metric='haversine')

    def query(self, lat, lon, k=4):
        dist, idx = self.tree.query(np.radians([[lat, lon]]), k=k)
        dist_km = dist[0] * 6371.0  # convert radians to km
        neighbors = [self.stations[i] for i in idx[0]]
        return neighbors, dist_km

def idw_interpolate(target_lat, target_lon, stations, power=2, k=4):
    locator = StationLocator(stations)
    neighbors, dists = locator.query(target_lat, target_lon, k=k)
    aqi_values = np.array([s['aqi'] for s in neighbors])
    dists = np.maximum(dists, 1e-3)  # avoid division by zero
    weights = 1 / (dists ** power)
    aqi_est = np.sum(weights * aqi_values) / np.sum(weights)
    return aqi_est, neighbors, dists

def get_weather_features(lat, lon, k=1):
    """
    Fetch or interpolate weather features for a given lat/lon.
    Merge station coordinates and weather data, then use BallTree on coordinates.
    Return weather features from the nearest station's weather row.
    """
    coords_df = pd.read_csv("data/merged_coords.csv")
    weather_df = pd.read_csv("data/latest_aqi_weather.csv")
    merged = pd.merge(coords_df, weather_df, left_on="StationName", right_on="station")
    coords = merged[["Latitude", "Longitude"]].values
    from sklearn.neighbors import BallTree
    tree = BallTree(np.radians(coords), metric='haversine')
    dist, idx = tree.query(np.radians([[lat, lon]]), k=k)
    nearest = merged.iloc[idx[0][0]]
    # Example weather features (adjust as per your CSV columns)
    features = {
        "temp": nearest.get("temperature", None),
        "humidity": nearest.get("humidity", None),
        "wind_speed": nearest.get("wind_speed", None)
    }
    return features 