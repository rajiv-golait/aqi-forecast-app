# src/data/merger.py
"""Data merger for combining multiple AQI and weather data sources."""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from config.config import Config

logger = logging.getLogger(__name__)

# Create settings instance
settings = Config()

class HistoricalDataMerger:
    def __init__(self, station_file: Optional[Path] = None):
        self.station_file = station_file or (settings.DATA_DIR / 'stations.csv')
        self.station_mapping = {}
        self.all_data = []

    def load_station_mapping(self):
        """Load station ID to city mapping from stations.csv"""
        try:
            df = pd.read_csv(self.station_file)
            logger.info(f"Station file columns: {df.columns.tolist()}")
            station_col = next((col for col in df.columns if 'station' in col.lower() and 'id' in col.lower()), None)
            city_col = next((col for col in df.columns if 'city' in col.lower()), None)
            state_col = next((col for col in df.columns if 'state' in col.lower()), None)
            name_col = next((col for col in df.columns if 'name' in col.lower() and 'station' in col.lower()), None)
            if station_col and city_col:
                for _, row in df.iterrows():
                    self.station_mapping[row[station_col]] = {
                        'city': row[city_col],
                        'state': row[state_col] if state_col else 'Unknown',
                        'station_name': row[name_col] if name_col else row[station_col]
                    }
                logger.info(f"Loaded {len(self.station_mapping)} station mappings")
            else:
                logger.warning("Could not find station ID and city columns in stations.csv")
        except Exception as e:
            logger.error(f"Error loading station mapping: {e}")

    def process_station_day(self, file_path: Optional[Path] = None):
        file_path = file_path or (settings.DATA_DIR / 'station_day.csv')
        try:
            logger.info(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            column_mapping = {
                'stationid': 'station_id',
                'date': 'datetime',
                'pm2.5': 'pm25',
                'pm10': 'pm10',
                'aqi': 'aqi',
                'aqi_bucket': 'aqi_category'
            }
            df.rename(columns=column_mapping, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if 'station_id' in df.columns:
                df['city'] = df['station_id'].map(lambda x: self.station_mapping.get(x, {}).get('city', f'Station_{x}'))
            df['source'] = 'station_day'
            df['frequency'] = 'daily'
            return df
        except Exception as e:
            logger.error(f"Error processing station_day: {e}")
            return None

    def process_station_hour(self, file_path: Optional[Path] = None):
        file_path = file_path or (settings.DATA_DIR / 'station_hour.csv')
        try:
            logger.info(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            column_mapping = {
                'stationid': 'station_id',
                'datetime': 'datetime',
                'date': 'datetime',
                'pm2.5': 'pm25',
                'pm10': 'pm10',
                'aqi': 'aqi'
            }
            df.rename(columns=column_mapping, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            if 'station_id' in df.columns:
                df['city'] = df['station_id'].map(lambda x: self.station_mapping.get(x, {}).get('city', f'Station_{x}'))
            df['source'] = 'station_hour'
            df['frequency'] = 'hourly'
            return df
        except Exception as e:
            logger.error(f"Error processing station_hour: {e}")
            return None

    def process_city_hour(self, file_path: Optional[Path] = None):
        file_path = file_path or (settings.DATA_DIR / 'city_hour.csv')
        try:
            logger.info(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            column_mapping = {
                'datetime': 'datetime',
                'date': 'datetime',
                'pm2.5': 'pm25',
                'pm2_5': 'pm25',
                'pm10': 'pm10',
                'aqi': 'aqi'
            }
            df.rename(columns=column_mapping, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['source'] = 'city_hour'
            df['frequency'] = 'hourly'
            return df
        except Exception as e:
            logger.error(f"Error processing city_hour: {e}")
            return None

    def process_city_day(self, file_path: Optional[Path] = None):
        file_path = file_path or (settings.DATA_DIR / 'city_day.csv')
        try:
            logger.info(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            column_mapping = {
                'date': 'datetime',
                'pm2.5': 'pm25',
                'pm2_5': 'pm25',
                'pm10': 'pm10',
                'aqi': 'aqi'
            }
            df.rename(columns=column_mapping, inplace=True)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df['source'] = 'city_day'
            return df
        except Exception as e:
            logger.error(f"Error processing city_day: {e}")
            return None

    def process_air_pollution_data(self, file_path: Optional[Path] = None):
        file_path = file_path or (settings.DATA_DIR / 'air_pollution_data.csv')
        try:
            logger.info(f"Processing {file_path}")
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower().str.strip()
            # Adjust as needed for your data structure
            df['source'] = 'air_pollution_data'
            return df
        except Exception as e:
            logger.error(f"Error processing air_pollution_data: {e}")
            return None

    def merge_all(self):
        """Merge all processed data sources into a single DataFrame"""
        self.load_station_mapping()
        dfs = []
        for func in [self.process_station_day, self.process_station_hour, self.process_city_hour, self.process_city_day, self.process_air_pollution_data]:
            df = func()
            if df is not None:
                dfs.append(df)
        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            logger.info(f"Merged data shape: {merged.shape}")
            return merged
        else:
            logger.warning("No data sources found to merge.")
            return pd.DataFrame()

# Create alias for expected class name
DataMerger = HistoricalDataMerger 