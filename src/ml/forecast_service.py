"""Forecasting service for AQI predictions."""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from api.models import ForecastPoint, StationForecast, AQICategory
from src.data.station_service import station_service

logger = logging.getLogger(__name__)

class ForecastService:
    def __init__(self):
        self.model_version = "3day_ensemble_v1.0"
        self.forecast_cache = {}
        self.load_forecast_cache()
    
    def load_forecast_cache(self):
        """Load existing forecasts from JSON files."""
        try:
            # Try to load the fixed forecast file first
            fixed_file = Path('data/predictions_3day_20250706_024937_fixed.json')
            forecasts = None
            if fixed_file.exists():
                with open(fixed_file, 'r') as f:
                    forecasts = json.load(f)
                logger.info(f"Loaded fixed forecast file: {fixed_file}")
            else:
                # Fallback to original forecast files
                forecast_files = list(Path('data').glob('predictions_*.json'))
                if forecast_files:
                    latest_file = max(forecast_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        forecasts = json.load(f)
            # Always populate the cache, regardless of which file was loaded
            if forecasts is not None:
                if isinstance(forecasts, list):
                    for item in forecasts:
                        # Construct station_id from city and station if not present
                        if 'station_id' not in item:
                            city = item.get('city', '')
                            station = item.get('station', '')
                            # Match API format: preserve comma, replace spaces and hyphens
                            station_id = f"{city}_{station}".replace(" ", "_").replace("-", "_")
                            item['station_id'] = station_id
                        self.forecast_cache[item['station_id']] = item
                    logger.info(f"Loaded {len(self.forecast_cache)} station forecasts from list cache")
                elif isinstance(forecasts, dict):
                    for station_id, forecast_data in forecasts.items():
                        self.forecast_cache[station_id] = forecast_data
                    logger.info(f"Loaded {len(self.forecast_cache)} station forecasts from dict cache")
            # After populating self.forecast_cache, log the keys
            logger.info(f"Forecast cache keys: {list(self.forecast_cache.keys())[:10]}")
            if 'Delhi_Alipur,_Delhi___DPCC' in self.forecast_cache:
                logger.info("Delhi_Alipur,_Delhi___DPCC is in the forecast cache!")
            else:
                logger.warning("Delhi_Alipur,_Delhi___DPCC is NOT in the forecast cache!")
        except Exception as e:
            logger.error(f"Error loading forecast cache: {e}")
    
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
    
    def get_station_forecast(self, station_id: str) -> Optional[StationForecast]:
        """Get forecast for a specific station."""
        try:
            logger.info(f"Looking up forecast for station_id: {station_id}")
            logger.info(f"Cache contains {len(self.forecast_cache)} stations")
            logger.info(f"Cache keys: {list(self.forecast_cache.keys())[:5]}")
            
            # Get station info
            station = station_service.get_station_by_id(station_id)
            if not station:
                logger.warning(f"Station not found: {station_id}")
                return None
            logger.info(f"Found station: {station.station_name}")
            
            # Get forecast data from cache
            forecast_data = self.forecast_cache.get(station_id)
            if not forecast_data:
                logger.warning(f"No forecast data for station: {station_id}")
                logger.warning(f"Available station IDs in cache: {list(self.forecast_cache.keys())[:10]}")
                return None
            logger.info(f"Found forecast data for station: {station_id}")
            
            # PATCH: handle new forecast point structure
            forecast_points = []
            for point in forecast_data.get('forecasts', []):
                try:
                    # Map aqi_category to valid enum values
                    aqi_category_str = point.get('aqi_category', '')
                    aqi_category_enum = None
                    if 'Good' in aqi_category_str:
                        aqi_category_enum = AQICategory.GOOD
                    elif 'Moderate' in aqi_category_str:
                        aqi_category_enum = AQICategory.MODERATE
                    elif 'Poor' in aqi_category_str:
                        aqi_category_enum = AQICategory.POOR
                    elif 'Very Poor' in aqi_category_str:
                        aqi_category_enum = AQICategory.VERY_POOR
                    elif 'Severe' in aqi_category_str:
                        aqi_category_enum = AQICategory.SEVERE
                    else:
                        aqi_category_enum = AQICategory.SATISFACTORY  # Default
                    
                    forecast_point = ForecastPoint(
                        datetime=datetime.fromisoformat(point['timestamp']),
                        pm25=0,  # Set to 0 instead of None
                        pm10=0,  # Set to 0 instead of None
                        aqi=int(point['aqi_predicted']),
                        aqi_category=aqi_category_enum,
                        confidence=0.8  # Set default confidence
                    )
                    forecast_points.append(forecast_point)
                except Exception as e:
                    logger.error(f"Error parsing forecast point: {e}")
                    continue
            if not forecast_points:
                logger.warning(f"No valid forecast points for station: {station_id}")
                return None
            logger.info(f"Created {len(forecast_points)} forecast points")
            
            return StationForecast(
                station_id=station.station_id,
                station_name=station.station_name,
                city=station.city,
                state=station.state,
                forecast_generated_at=datetime.fromisoformat(forecast_data.get('last_known_time', datetime.now().isoformat())),
                model_version=self.model_version,
                forecasts=forecast_points
            )
        except Exception as e:
            logger.error(f"Error getting station forecast for {station_id}: {e}")
            return None
    
    def get_forecast_summary(self, station_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of forecast data for a station."""
        try:
            forecast = self.get_station_forecast(station_id)
            if not forecast:
                return None
            
            # Calculate summary statistics
            aqi_values = [f.aqi for f in forecast.forecasts]
            pm25_values = [f.pm25 for f in forecast.forecasts]
            pm10_values = [f.pm10 for f in forecast.forecasts]
            
            # Get category distribution
            category_counts = {}
            for f in forecast.forecasts:
                category = f.aqi_category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            return {
                'station_id': station_id,
                'station_name': forecast.station_name,
                'city': forecast.city,
                'state': forecast.state,
                'forecast_period': {
                    'start': forecast.forecasts[0].datetime.isoformat(),
                    'end': forecast.forecasts[-1].datetime.isoformat(),
                    'total_hours': len(forecast.forecasts)
                },
                'aqi_summary': {
                    'min': min(aqi_values),
                    'max': max(aqi_values),
                    'avg': round(sum(aqi_values) / len(aqi_values), 2),
                    'current': aqi_values[0] if aqi_values else None
                },
                'pm25_summary': {
                    'min': min(pm25_values),
                    'max': max(pm25_values),
                    'avg': round(sum(pm25_values) / len(pm25_values), 2)
                },
                'pm10_summary': {
                    'min': min(pm10_values),
                    'max': max(pm10_values),
                    'avg': round(sum(pm10_values) / len(pm10_values), 2)
                },
                'category_distribution': category_counts,
                'model_version': forecast.model_version,
                'generated_at': forecast.forecast_generated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting forecast summary for {station_id}: {e}")
            return None
    
    def get_available_stations(self) -> List[str]:
        """Get list of stations with available forecasts."""
        return list(self.forecast_cache.keys())
    
    def get_forecast_metadata(self) -> Dict[str, Any]:
        """Get metadata about available forecasts."""
        try:
            total_stations = len(self.forecast_cache)
            if total_stations == 0:
                return {
                    'total_stations': 0,
                    'model_version': self.model_version,
                    'last_updated': None,
                    'coverage': 'No forecasts available'
                }
            
            # Get latest generation time
            latest_time = None
            for station_id, forecast_data in self.forecast_cache.items():
                gen_time = forecast_data.get('generated_at')
                if gen_time:
                    try:
                        dt = datetime.fromisoformat(gen_time)
                        if latest_time is None or dt > latest_time:
                            latest_time = dt
                    except:
                        pass
            
            return {
                'total_stations': total_stations,
                'model_version': self.model_version,
                'last_updated': latest_time.isoformat() if latest_time else None,
                'coverage': f"{total_stations} stations across India"
            }
            
        except Exception as e:
            logger.error(f"Error getting forecast metadata: {e}")
            return {
                'total_stations': 0,
                'model_version': self.model_version,
                'last_updated': None,
                'coverage': 'Error loading metadata'
            }

# Global instance
forecast_service = ForecastService() 