"""Station service for managing AQI station data and live readings."""

import json
import pandas as pd
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass

from api.models import StationInfo, LiveAQIData, AQICategory

logger = logging.getLogger(__name__)

@dataclass
class Station:
    station_id: str
    station_name: str
    city: str
    state: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class StationService:
    def __init__(self):
        self.stations_cache = {}
        self.station_coords = {}
        self.load_station_data()
    
    def load_station_data(self):
        """Load station data from valid_locations.json and merged_coords.csv"""
        try:
            # Load station locations
            with open('data/valid_locations.json', 'r') as f:
                locations = json.load(f)
            
            # Load coordinates if available
            coords_df = None
            try:
                coords_df = pd.read_csv('data/merged_coords.csv')
            except FileNotFoundError:
                logger.warning("Coordinates file not found")
            
            # Build station cache
            for state, cities in locations.items():
                for city, stations in cities.items():
                    for station_name in stations:
                        station_id = f"{city}_{station_name}".replace(" ", "_").replace("-", "_")
                        
                        # Find coordinates
                        lat, lon = None, None
                        if coords_df is not None:
                            # Escape special regex characters in station name
                            import re
                            station_part = re.escape(station_name.split(' - ')[0])
                            match = coords_df[
                                (coords_df['City'].str.contains(city, case=False, na=False)) &
                                (coords_df['StationName'].str.contains(station_part, case=False, na=False))
                            ]
                            if not match.empty:
                                lat = match.iloc[0].get('latitude')
                                lon = match.iloc[0].get('longitude')
                        
                        station = Station(
                            station_id=station_id,
                            station_name=station_name,
                            city=city,
                            state=state,
                            latitude=lat,
                            longitude=lon
                        )
                        
                        self.stations_cache[station_id] = station
                        # Also index by city and state for easier lookup
                        self.stations_cache[f"{city}_{state}"] = station
            
            logger.info(f"Loaded {len(self.stations_cache)} stations")
            
        except Exception as e:
            logger.error(f"Error loading station data: {e}")
    
    def get_station_by_id(self, station_id: str) -> Optional[Station]:
        """Get station by ID."""
        return self.stations_cache.get(station_id)
    
    def search_stations(self, city: str = None, state: str = None, 
                       station_name: str = None, limit: int = 50) -> List[Station]:
        """Search stations by criteria."""
        results = []
        
        for station in self.stations_cache.values():
            if isinstance(station, Station):  # Skip non-Station entries
                match = True
                
                if city and city.lower() not in station.city.lower():
                    match = False
                if state and state.lower() not in station.state.lower():
                    match = False
                if station_name and station_name.lower() not in station.station_name.lower():
                    match = False
                
                if match:
                    results.append(station)
                    if len(results) >= limit:
                        break
        
        return results
    
    def get_aqi_category(self, aqi: int) -> AQICategory:
        """Convert AQI value to category."""
        if aqi <= 50:
            return AQICategory.GOOD
        elif aqi <= 100:
            return AQICategory.SATISFACTORY
        elif aqi <= 200:
            return AQICategory.MODERATE
        elif aqi <= 300:
            return AQICategory.POOR
        elif aqi <= 400:
            return AQICategory.VERY_POOR
        else:
            return AQICategory.SEVERE
    
    def fetch_live_aqi(self, station_id: str) -> Optional[LiveAQIData]:
        """Fetch live AQI data for a specific station."""
        try:
            station = self.get_station_by_id(station_id)
            if not station:
                return None
            
            # For now, we'll simulate live data from the latest CSV
            # In production, this would fetch from the real-time API
            latest_data = self.get_latest_aqi_data(station)
            
            if latest_data:
                return LiveAQIData(
                    station_id=station.station_id,
                    station_name=station.station_name,
                    city=station.city,
                    state=station.state,
                    timestamp=datetime.now(),
                    pm25=latest_data.get('pm25'),
                    pm10=latest_data.get('pm10'),
                    no2=latest_data.get('no2'),
                    so2=latest_data.get('so2'),
                    co=latest_data.get('co'),
                    o3=latest_data.get('o3'),
                    aqi=latest_data.get('aqi', 0),
                    aqi_category=self.get_aqi_category(latest_data.get('aqi', 0)),
                    latitude=station.latitude,
                    longitude=station.longitude
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching live AQI for {station_id}: {e}")
            return None
    
    def get_latest_aqi_data(self, station: Station) -> Optional[Dict]:
        """Get latest AQI data for a station from CSV files."""
        try:
            # Try to find data for this station in the latest CSV
            latest_csv = 'data/latest_aqi_weather.csv'
            if not os.path.exists(latest_csv):
                return None
            
            df = pd.read_csv(latest_csv)
            
            # Look for station data (this is a simplified approach)
            # In reality, you'd need to match station names properly
            station_match = df[
                df['city'].str.contains(station.city, case=False, na=False)
            ]
            
            if not station_match.empty:
                latest_row = station_match.iloc[-1]
                return {
                    'pm25': latest_row.get('pm25'),
                    'pm10': latest_row.get('pm10'),
                    'no2': latest_row.get('no2'),
                    'so2': latest_row.get('so2'),
                    'co': latest_row.get('co'),
                    'o3': latest_row.get('o3'),
                    'aqi': latest_row.get('aqi', 0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest AQI data: {e}")
            return None
    
    def get_station_info(self, station_id: str) -> Optional[StationInfo]:
        """Get station information."""
        station = self.get_station_by_id(station_id)
        if not station:
            return None
        
        return StationInfo(
            station_id=station.station_id,
            station_name=station.station_name,
            city=station.city,
            state=station.state,
            latitude=station.latitude,
            longitude=station.longitude,
            last_updated=datetime.now()
        )

# Global instance
station_service = StationService() 