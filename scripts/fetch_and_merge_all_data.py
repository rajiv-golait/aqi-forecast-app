import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import logging
import requests
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from src.data.fetchers import AQIFetcher, WeatherFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

POLLUTANT_COLS = ['pm25', 'pm10', 'aqi', 'no2', 'so2', 'o3', 'co']

FETCH_LOG_PATH = 'data/fetch_merge_log.csv'

class DataPipeline:
    def __init__(self):
        self.config = Config()
        self.aqi_fetcher = AQIFetcher()
        self.weather_fetcher = WeatherFetcher()
        # Load city coordinates
        self.city_coords = self._load_city_coords()

    def _load_city_coords(self):
        """Load city-level coordinates from address.csv: (city, state) -> (lat, lon)"""
        city_coords = {}
        try:
            df_city = pd.read_csv('data/address.csv')
            for _, row in df_city.iterrows():
                key = (str(row['City']).strip().lower(), str(row['State']).strip().lower())
                city_coords[key] = (row['Latitude'], row['Longitude'])
        except FileNotFoundError:
            logger.info("address.csv not found (optional, skipping city-level coordinates).")
        except Exception as e:
            logger.info(f"Could not load address.csv (optional): {e}")
        return city_coords

    def engineer_features(self, df):
        """Engineer features for ML model (only essential features with actual data)"""
        try:
            if df.empty:
                return df
            
            # Only create features that have actual data
            if 'temperature' in df.columns and 'humidity' in df.columns:
                # Check if temperature and humidity have actual values
                if not df['temperature'].isna().all() and not df['humidity'].isna().all():
                    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            
            if 'wind_speed' in df.columns:
                # Check if wind_speed has actual values
                if not df['wind_speed'].isna().all():
                    df['wind_speed_squared'] = df['wind_speed'] ** 2
            
            logger.info(f"Feature engineering completed. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            return df

    def geocode_missing_stations(self):
        """Find stations missing coordinates and fill them using Nominatim geocoding API."""
        logger.info("Checking for stations missing coordinates...")
        # Load current coords
        df_coords = pd.read_csv('data/stations_with_coords.csv')
        existing_keys = set((str(row['StationName']).strip().lower(), str(row['City']).strip().lower(), str(row['State']).strip().lower()) for _, row in df_coords.iterrows())
        # Load valid locations
        with open('data/valid_locations.json', 'r') as f:
            valid_locations = json.load(f)
        missing = []
        for state, cities in valid_locations.items():
            for city, stations in cities.items():
                for station in stations:
                    key = (str(station).strip().lower(), str(city).strip().lower(), str(state).strip().lower())
                    if key not in existing_keys:
                        missing.append({'StationName': station, 'City': city, 'State': state})
        if not missing:
            logger.info("No missing stations for coordinates.")
            return
        logger.info(f"Found {len(missing)} stations missing coordinates. Attempting to geocode...")
        # Geocode using Nominatim
        for entry in missing:
            query = f"{entry['StationName']}, {entry['City']}, {entry['State']}, India"
            try:
                url = f"https://nominatim.openstreetmap.org/search"
                params = {'q': query, 'format': 'json', 'limit': 1}
                resp = requests.get(url, params=params, headers={'User-Agent': 'aqi-forecast-app'})
                resp.raise_for_status()
                results = resp.json()
                if results:
                    entry['Latitude'] = float(results[0]['lat'])
                    entry['Longitude'] = float(results[0]['lon'])
                    logger.info(f"Geocoded {query} -> ({entry['Latitude']}, {entry['Longitude']})")
                else:
                    entry['Latitude'] = None
                    entry['Longitude'] = None
                    logger.warning(f"Could not geocode {query}")
            except Exception as e:
                entry['Latitude'] = None
                entry['Longitude'] = None
                logger.warning(f"Error geocoding {query}: {e}")
        # Append to CSV
        new_rows = [e for e in missing if e['Latitude'] is not None and e['Longitude'] is not None]
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_new['StationId'] = ''
            df_coords = pd.concat([df_coords, df_new[['StationId','StationName','City','State','Latitude','Longitude']]], ignore_index=True)
            df_coords.to_csv('data/stations_with_coords.csv', index=False)
            logger.info(f"Added {len(new_rows)} new stations with coordinates to stations_with_coords.csv")
        else:
            logger.warning("No new stations could be geocoded.")

    def print_missing_station_addresses(self):
        """Print all (station, city, state) addresses missing from stations_with_coords.csv for manual geocoding."""
        logger.info("Listing all stations missing coordinates...")
        df_coords = pd.read_csv('data/stations_with_coords.csv')
        existing_keys = set((str(row['StationName']).strip().lower(), str(row['City']).strip().lower(), str(row['State']).strip().lower()) for _, row in df_coords.iterrows())
        with open('data/valid_locations.json', 'r') as f:
            valid_locations = json.load(f)
        missing = []
        for state, cities in valid_locations.items():
            for city, stations in cities.items():
                for station in stations:
                    key = (str(station).strip().lower(), str(city).strip().lower(), str(state).strip().lower())
                    if key not in existing_keys:
                        missing.append(f"{station}, {city}, {state}, India")
        if not missing:
            print("No missing stations for coordinates.")
        else:
            print("\n".join(missing))
            logger.info(f"Total missing: {len(missing)}")

    def get_coords(self, station, city, state, coords, city_coords):
        """Get coordinates for a station, falling back to city-level if needed."""
        key = (str(station).strip().lower(), str(city).strip().lower(), str(state).strip().lower())
        city_key = (str(city).strip().lower(), str(state).strip().lower())
        if key in coords:
            return coords[key], 'station'
        elif city_key in city_coords:
            return city_coords[city_key], 'city'
        else:
            return (None, None), None

    def merge_coords_files(self):
        """Merge stations_with_coords.csv and address.csv into merged_coords.csv for all stations in valid_locations.json."""
        logger.info("Merging station and city coordinates into merged_coords.csv ...")
        # Load station-level coords
        station_coords = {}
        try:
            with open('data/stations_with_coords.csv', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['StationName'].strip().lower(), row['City'].strip().lower(), row['State'].strip().lower())
                    station_coords[key] = (row['Latitude'], row['Longitude'])
        except Exception as e:
            logger.warning(f"Could not load stations_with_coords.csv: {e}")
        # Load city-level coords
        city_coords = {}
        try:
            with open('data/address.csv', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row['City'].strip().lower(), row['State'].strip().lower())
                    city_coords[key] = (row['Latitude'], row['Longitude'])
        except FileNotFoundError:
            logger.info("address.csv not found (optional, skipping city-level coordinates).")
        except Exception as e:
            logger.info(f"Could not load address.csv (optional): {e}")
        # Load all stations from valid_locations.json
        with open('data/valid_locations.json', 'r', encoding='utf-8') as f:
            valid_locations = json.load(f)
        merged = []
        for state, cities in valid_locations.items():
            for city, stations in cities.items():
                for station in stations:
                    key = (station.strip().lower(), city.strip().lower(), state.strip().lower())
                    city_key = (city.strip().lower(), state.strip().lower())
                    if key in station_coords:
                        lat, lon = station_coords[key]
                    elif city_key in city_coords:
                        lat, lon = city_coords[city_key]
                    else:
                        lat, lon = '', ''
                    merged.append({
                                'StationName': station,
                                'City': city,
                                'State': state,
                        'Latitude': lat,
                        'Longitude': lon
                    })
        # Write merged file
        with open('data/merged_coords.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['StationName', 'City', 'State', 'Latitude', 'Longitude'])
            writer.writeheader()
            writer.writerows(merged)
        logger.info(f"Merged coordinates written to data/merged_coords.csv. Total: {len(merged)} stations.")

    def _load_merged_coords(self):
        """Load merged station/city coordinates from merged_coords.csv: (station, city, state) -> (lat, lon)"""
        coords = {}
        try:
            df = pd.read_csv('data/merged_coords.csv')
            for _, row in df.iterrows():
                key = (str(row['StationName']).strip().lower(), str(row['City']).strip().lower(), str(row['State']).strip().lower())
                coords[key] = (row['Latitude'], row['Longitude'])
        except Exception as e:
            logger.warning(f"Could not load merged_coords.csv: {e}")
        return coords

    def run_pipeline(self):
        """Run the data pipeline for all valid stations in India: only AQI and OpenWeatherMap weather data (no satellite)."""
        try:
            logger.info("Starting data pipeline for ALL valid India stations (AQI + weather only, no satellite)...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            # Load valid locations
            with open('data/valid_locations.json', 'r') as f:
                valid_locations = json.load(f)
            
            # Count total stations in valid_locations.json
            total_valid_stations = sum(len(stations) for cities in valid_locations.values() for stations in cities.values())
            logger.info(f"Total valid stations to check: {total_valid_stations}")
            
            all_data = []
            # Load coordinates from merged_coords.csv
            coords_df = pd.read_csv('data/merged_coords.csv')
            city_coords = {(str(row['City']).strip().lower(), str(row['State']).strip().lower()): (row['Latitude'], row['Longitude']) for _, row in coords_df.iterrows()}
            
            # Fetch all-India AQI data once
            aqi_df = self.aqi_fetcher.fetch_all_india_data(start_date, end_date)
            total_api_stations = len(aqi_df) if not aqi_df.empty else 0
            logger.info(f"Fetched {total_api_stations} AQI records for all India.")
            
            stations_with_data = 0
            stations_processed = 0
            
            for state, cities in valid_locations.items():
                for city, stations in cities.items():
                    key = (city.strip().lower(), state.strip().lower())
                    lat, lon = city_coords.get(key, (None, None))
                    if lat is None or lon is None:
                        logger.warning(f"No coordinates for {city}, {state}. Skipping.")
                        continue
                    for station in stations:
                        stations_processed += 1
                        # Filter AQI records for this station
                        station_aqi = aqi_df[(aqi_df['city'].str.strip().str.lower() == city.strip().lower()) & (aqi_df['state'].str.strip().str.lower() == state.strip().lower()) & (aqi_df['station'].str.strip().str.lower() == station.strip().lower())]
                        if station_aqi.empty:
                            logger.info(f"No AQI data for {station}, {city}, {state} in last 24h.")
                            continue
                        
                        stations_with_data += 1
                        
                        # Validate coordinates
                        if lat is None or lon is None or str(lat).strip() == '' or str(lon).strip() == '':
                            logger.warning(f"Missing coordinates for {city}, {state}. Skipping weather fetch.")
                            continue
                        try:
                            lat_f = float(lat)
                            lon_f = float(lon)
                        except Exception:
                            logger.warning(f"Invalid coordinates for {city}, {state}: lat={lat}, lon={lon}. Skipping weather fetch.")
                            continue
                        # Fetch current weather for this city
                        weather = self.weather_fetcher.fetch_weather_data(city, lat_f, lon_f)
                        if weather is None or weather.empty:
                            logger.warning(f"No weather data for {city}, {state}.")
                            continue
                        # Merge AQI and weather for each record
                        for _, row in station_aqi.iterrows():
                            merged = row.to_dict()
                            # Backfill pollutant columns if NaN: look back up to 1 hour for this station
                            for col in POLLUTANT_COLS:
                                if pd.isna(merged.get(col)):
                                    # Find most recent non-NaN value for this station within 1 hour before this datetime
                                    dt = merged['datetime'] if 'datetime' in merged else merged.get('last_update')
                                    if pd.isna(dt):
                                        continue
                                    dt = pd.to_datetime(dt)
                                    mask = (station_aqi['datetime'] < dt) & (station_aqi['datetime'] >= dt - pd.Timedelta(hours=1))
                                    prev = station_aqi.loc[mask, col].dropna()
                                    if not prev.empty:
                                        merged[col] = prev.iloc[-1]
                            merged.update(weather.iloc[0].to_dict())
                            all_data.append(merged)
            
            if not all_data:
                logger.error("No data collected from any station.")
                return {
                    'total_api_stations': total_api_stations,
                    'stations_with_data': 0,
                    'stations_appended': 0
                }
            
            df = pd.DataFrame(all_data)
            # Drop rows where all AQI pollutant columns are NaN
            pollutant_cols = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'co']
            df = df.dropna(subset=pollutant_cols, how='all')
            # Only drop columns that are completely empty (all NaN)
            empty_cols = [col for col in df.columns if df[col].isnull().all()]
            if len(empty_cols) > 0:
                logger.info(f"Dropping all-empty columns: {empty_cols}")
                df = df.drop(columns=empty_cols)
            logger.info(f"Final merged data shape: {df.shape}")
            print("\n--- DATA PREVIEW ---")
            print(df.head(10))
            
            # Save to CSV (append new data, avoid duplicates)
            output_path = 'data/latest_aqi_weather.csv'
            old_df = None
            if os.path.exists(output_path):
                old_df = pd.read_csv(output_path)
                combined = pd.concat([old_df, df], ignore_index=True)
                # Drop duplicates based on station, city, state, datetime
                if combined is not None and not combined.empty:
                    combined = combined.drop_duplicates(subset=['station', 'city', 'state', 'datetime'], keep='last')
                    # Keep only last 14 days of data
                    combined['datetime'] = pd.to_datetime(combined['datetime'])
                    cutoff = combined['datetime'].max() - pd.Timedelta(days=14)
                    combined = combined[combined['datetime'] >= cutoff]
                    combined.to_csv(output_path, index=False)
                    logger.info(f"Appended new data. Combined shape: {combined.shape}. Kept only last 14 days.")
                else:
                    logger.warning("No combined data to save")
            else:
                df.to_csv(output_path, index=False)
                logger.info("Saved merged AQI + weather data to data/latest_aqi_weather.csv")
            
            # Calculate stations appended (new unique stations in this run)
            stations_appended = len(df['station'].unique()) if not df.empty else 0
            
            # Log station metrics
            logger.info(f"📊 Station Metrics:")
            logger.info(f"   Total API stations: {total_api_stations}")
            logger.info(f"   Stations with data: {stations_with_data}")
            logger.info(f"   Stations appended: {stations_appended}")
            logger.info(f"   Inactive stations: {total_api_stations - stations_with_data}")
            logger.info(f"   Data coverage: {(stations_with_data/total_api_stations*100):.1f}%" if total_api_stations > 0 else "0%")
            
            return {
                'total_api_stations': total_api_stations,
                'stations_with_data': stations_with_data,
                'stations_appended': stations_appended
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return None

    def _get_station_coordinates(self, station, city, state):
        """Get coordinates for a station (you can expand this mapping)"""
        # Default coordinates for major cities
        city_coords = {
            'Delhi': (28.6139, 77.2090),
            'Mumbai': (19.0760, 72.8777),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
        }
        
        # Return city coordinates if available, otherwise use Delhi as default
        return city_coords.get(city, (28.6139, 77.2090))

    def debug_print_latest_aqi(self):
        """Fetch and print the latest AQI API response (raw and DataFrame head) for the last 24 hours."""
        logger.info("Fetching latest AQI data for debug...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        df = self.aqi_fetcher.fetch_all_india_data(start_date, end_date)
        print("\n--- RAW DATAFRAME SHAPE ---")
        print(df.shape)
        print("\n--- DATAFRAME HEAD ---")
        print(df.head(20))
        if df.empty:
            print("No AQI data returned by API for the last 24 hours.")
        else:
            print(f"Returned {len(df)} records from AQI API.")

def log_fetch_merge(run_time, latest_dt, new_records, total_records, status, error_msg=None, 
                   total_api_stations=0, stations_with_data=0, stations_appended=0):
    """Append a row to the fetch/merge log CSV."""
    log_exists = os.path.exists(FETCH_LOG_PATH)
    with open(FETCH_LOG_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow([
                'run_time', 'latest_data_datetime', 'new_records', 'total_records', 'status', 'error_msg',
                'total_api_stations', 'stations_with_data', 'stations_appended', 'inactive_stations', 'data_coverage_pct'
            ])
        
        inactive_stations = total_api_stations - stations_with_data if total_api_stations > 0 else 0
        data_coverage_pct = (stations_with_data / total_api_stations * 100) if total_api_stations > 0 else 0
        
        writer.writerow([
            run_time.strftime('%Y-%m-%d %H:%M:%S'),
            latest_dt.strftime('%Y-%m-%d %H:%M:%S') if latest_dt is not None else '',
            new_records,
            total_records,
            status,
            error_msg or '',
            total_api_stations,
            stations_with_data,
            stations_appended,
            inactive_stations,
            f"{data_coverage_pct:.1f}%"
        ])

def main():
    output_path = 'data/latest_aqi_weather.csv'
    now = datetime.now()
    run_time = now
    latest_dt = None
    new_records = 0
    total_records = 0
    status = 'skipped'
    error_msg = None
    total_api_stations = 0
    stations_with_data = 0
    stations_appended = 0
    
    if os.path.exists(output_path):
        try:
            df = pd.read_csv(output_path, parse_dates=['datetime'])
            if not df.empty:
                latest_dt = pd.to_datetime(df['datetime'].max())
                # Only run if at least 1 hour has passed since the latest data
                if latest_dt + timedelta(hours=1) > now:
                    print(f"Latest data is {latest_dt:%Y-%m-%d %H:%M}. Next fetch allowed after {(latest_dt + timedelta(hours=1)):%Y-%m-%d %H:%M}.")
                    log_fetch_merge(run_time, latest_dt, 0, len(df), 'skipped', 'Too soon since last data',
                                  total_api_stations, stations_with_data, stations_appended)
                    return
        except Exception as e:
            error_msg = f"Warning: Could not check latest timestamp in CSV: {e}"
            print(error_msg)
            log_fetch_merge(run_time, latest_dt, 0, 0, 'error', error_msg,
                          total_api_stations, stations_with_data, stations_appended)
            return
    
    try:
        pipeline = DataPipeline()
        before_count = 0
        if os.path.exists(output_path):
            try:
                before_count = pd.read_csv(output_path).shape[0]
            except Exception:
                before_count = 0
        
        # Run pipeline and get station metrics
        station_metrics = pipeline.run_pipeline()
        if station_metrics:
            total_api_stations = station_metrics.get('total_api_stations', 0)
            stations_with_data = station_metrics.get('stations_with_data', 0)
            stations_appended = station_metrics.get('stations_appended', 0)
        
        # After run, check new record count
        if os.path.exists(output_path):
            df = pd.read_csv(output_path, parse_dates=['datetime'])
            latest_dt = pd.to_datetime(df['datetime'].max()) if not df.empty else None
            total_records = len(df)
            new_records = total_records - before_count
            status = 'success'
        else:
            status = 'error'
            error_msg = 'Output file missing after pipeline run.'
        
        log_fetch_merge(run_time, latest_dt, new_records, total_records, status, error_msg,
                       total_api_stations, stations_with_data, stations_appended)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Pipeline failed: {e}")
        log_fetch_merge(run_time, latest_dt, 0, 0, 'error', error_msg,
                       total_api_stations, stations_with_data, stations_appended)

# Add a utility entry point for geocoding
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "geocode":
        DataPipeline().geocode_missing_stations()
    elif len(sys.argv) > 1 and sys.argv[1] == "missing_addresses":
        DataPipeline().print_missing_station_addresses()
    elif len(sys.argv) > 1 and sys.argv[1] == "merge_coords":
        DataPipeline().merge_coords_files()
    elif len(sys.argv) > 1 and sys.argv[1] == "debug_aqi":
        DataPipeline().debug_print_latest_aqi()
    else:
        main()