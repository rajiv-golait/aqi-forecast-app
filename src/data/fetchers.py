# src/data/fetchers.py
"""Data fetchers for AQI and weather data sources."""

import requests
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from config.config import settings
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)

# Wrapper classes for compatibility
class AQIFetcher:
    def __init__(self):
        self.fetcher = IndiaGovAQIFetcher()
    
    def fetch_all_india_data(self, start_date, end_date):
        """Fetch AQI data for all India stations"""
        return self.fetcher.fetch_all_india_data(start_date, end_date)

class WeatherFetcher:
    def __init__(self):
        self.fetcher = OpenWeatherFetcher()
    
    def fetch_weather_data(self, city, lat, lon):
        """Fetch weather data for a city"""
        return self.fetcher.fetch_weather_data(city, lat, lon)

CITY_MAPPING = {
    "Delhi": ("Delhi", "Delhi", 28.6139, 77.2090),
    "Mumbai": ("Maharashtra", "Mumbai", 19.0760, 72.8777),
    "Bangalore": ("Karnataka", "Bengaluru", 12.9716, 77.5946),
    "Chennai": ("TamilNadu", "Chennai", 13.0827, 80.2707),
    "Kolkata": ("West_Bengal", "Kolkata", 22.5726, 88.3639),
    "Hyderabad": ("Telangana", "Hyderabad", 17.3850, 78.4867),
    "Pune": ("Maharashtra", "Pune", 18.5204, 73.8567),
    "Ahmedabad": ("Gujarat", "Ahmedabad", 23.0225, 72.5714),
}

class IndiaGovAQIFetcher:
    def __init__(self, api_key: str = None, api_url: str = None):
        from config.config import settings
        self.api_key = api_key or settings.DATA_GOV_API_KEY
        self.api_url = api_url or settings.DATA_GOV_BASE_URL

    def fetch_city_data(self, city: str, start_date, end_date):
        """Fetch AQI data for a specific city and date range (station-level, real API timestamps only)"""
        try:
            city_info = CITY_MAPPING.get(city)
            if not city_info:
                logger.error(f"City {city} not found in mapping")
                return pd.DataFrame()
            state, city_name, lat, lon = city_info
            data = self.fetch(state=state, city=city_name)
            if data is None or 'station_data' not in data:
                return pd.DataFrame()
            df_data = []
            for station_name, station_info in data['station_data'].items():
                ts = station_info.get('last_update')
                if not ts:
                    logger.debug(f"No last_update for station {station_name}")
                    continue
                try:
                    dt = pd.to_datetime(ts, dayfirst=True)
                except Exception as e:
                    logger.debug(f"Failed to parse last_update '{ts}' for station {station_name}: {e}")
                    continue
                logger.info(f"Station: {station_name}, last_update: {ts}, parsed: {dt}, start: {start_date}, end: {end_date}")
                if not (start_date <= dt <= end_date):
                    logger.info(f"Skipping station {station_name} at {dt} (outside range)")
                    continue
                df_data.append({
                    'datetime': dt,
                    'city': city,
                    'state': state,
                    'station': station_name,
                    'station_id': station_name,
                    'pm25': station_info.get('pm25_avg'),
                    'pm10': station_info.get('pm10_avg'),
                    'no2': station_info.get('other_pollutants', {}).get('NO2'),
                    'so2': station_info.get('other_pollutants', {}).get('SO2'),
                    'o3': station_info.get('other_pollutants', {}).get('O3'),
                    'co': station_info.get('other_pollutants', {}).get('CO'),
                    'aqi': station_info.get('aqi'),
                    'last_updated': ts
                })
            return pd.DataFrame(df_data)
        except Exception as e:
            logger.error(f"Error fetching city data for {city}: {e}")
            return pd.DataFrame()

    def fetch(self, state: Optional[str] = None, city: Optional[str] = None, station: Optional[str] = None, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Fetch AQI data from data.gov.in API with improved parsing (station-level)"""
        try:
            params = {
                "api-key": self.api_key,
                "format": "json",
                "limit": limit
            }
            if state:
                params["filters[state]"] = state
            if city:
                params["filters[city]"] = city
            if station:
                params["filters[station]"] = station
            logger.info(f"API Request: {self.api_url}")
            logger.info(f"Parameters: {params}")
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'records' not in data:
                logger.error(f"Invalid response structure: {data.keys()}")
                return None
            records = data.get('records', [])
            if not records:
                logger.warning(f"No AQI data found for {city}, {state}")
                return None
            # Parse records - group by station and pollutant
            station_data = {}
            for record in records:
                try:
                    station_name = record.get('station', 'Unknown')
                    pollutant_id = record.get('pollutant_id', '')
                    avg_value = record.get('avg_value') or record.get('pollutant_avg')
                    last_update = record.get('last_update', '')
                    if station_name not in station_data:
                        station_data[station_name] = {
                            'pm25': [],
                            'pm10': [],
                            'last_update': last_update,
                            'other_pollutants': {}
                        }
                    if avg_value and avg_value != 'NA':
                        try:
                            if isinstance(avg_value, str):
                                pollutant_value = float(avg_value.replace(',', ''))
                            else:
                                pollutant_value = float(avg_value)
                            if pollutant_id == 'PM2.5':
                                station_data[station_name]['pm25'].append(pollutant_value)
                            elif pollutant_id == 'PM10':
                                station_data[station_name]['pm10'].append(pollutant_value)
                            else:
                                station_data[station_name]['other_pollutants'][pollutant_id] = pollutant_value
                            if last_update > station_data[station_name]['last_update']:
                                station_data[station_name]['last_update'] = last_update
                        except (ValueError, TypeError):
                            continue
                except Exception as e:
                    logger.error(f"Error parsing record: {e}")
                    continue
            
            # Calculate averages for each station
            for station_name, data in station_data.items():
                if data['pm25']:
                    data['pm25_avg'] = sum(data['pm25']) / len(data['pm25'])
                if data['pm10']:
                    data['pm10_avg'] = sum(data['pm10']) / len(data['pm10'])
                
                # Calculate AQI based on PM2.5 and PM10 (simplified calculation)
                pm25_aqi = self._calculate_aqi(data.get('pm25_avg', 0), 'pm25')
                pm10_aqi = self._calculate_aqi(data.get('pm10_avg', 0), 'pm10')
                data['aqi'] = max(pm25_aqi, pm10_aqi) if pm25_aqi > 0 or pm10_aqi > 0 else None
            
            if station_data:
                result = {
                    'station_data': station_data,
                    'station_count': len(station_data),
                    'last_updated': max([data['last_update'] for data in station_data.values()]),
                    'stations_with_data': list(station_data.keys())
                }
                logger.info(f"Successfully parsed data for {len(station_data)} stations")
                return result
            else:
                logger.warning(f"No station data found in records")
                return None
        except Exception as e:
            logger.error(f"Error fetching AQI data: {e}")
            return None
    
    def _calculate_aqi(self, concentration, pollutant_type):
        """Calculate AQI based on pollutant concentration (simplified)"""
        try:
            if not concentration or concentration <= 0:
                return 0
            
            if pollutant_type == 'pm25':
                # PM2.5 breakpoints (μg/m³)
                if concentration <= 12:
                    return concentration * 50 / 12
                elif concentration <= 35.4:
                    return 51 + (concentration - 12) * 49 / 23.4
                elif concentration <= 55.4:
                    return 101 + (concentration - 35.4) * 49 / 20
                elif concentration <= 150.4:
                    return 151 + (concentration - 55.4) * 49 / 95
                elif concentration <= 250.4:
                    return 201 + (concentration - 150.4) * 49 / 100
                else:
                    return 301 + (concentration - 250.4) * 199 / 249.6
            elif pollutant_type == 'pm10':
                # PM10 breakpoints (μg/m³)
                if concentration <= 54:
                    return concentration * 50 / 54
                elif concentration <= 154:
                    return 51 + (concentration - 54) * 49 / 100
                elif concentration <= 254:
                    return 101 + (concentration - 154) * 49 / 100
                elif concentration <= 354:
                    return 151 + (concentration - 254) * 49 / 100
                elif concentration <= 424:
                    return 201 + (concentration - 354) * 49 / 70
                else:
                    return 301 + (concentration - 424) * 199 / 576
            else:
                return 0
        except:
            return 0

    def fetch_all_india_data(self, start_date, end_date):
        """Fetch AQI data for ALL active stations in India"""
        try:
            # Fetch data for all states/cities without filters
            url = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': 1000  # Get maximum records
            }
            
            logger.info(f"Fetching ALL India AQI data...")
            logger.info(f"API Request: {url}")
            logger.info(f"Parameters: {params}")
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # DEBUG: Print a sample of the raw API response
            if data and 'records' in data and len(data['records']) > 0:
                print("\n--- RAW AQI API SAMPLE ---")
                import json as _json
                print(_json.dumps(data['records'][:2], indent=2))
                print("--- END SAMPLE ---\n")
            
            if not data or 'records' not in data:
                logger.warning("No data found in API response")
                return pd.DataFrame()
            
            # Parse all stations and aggregate pollutants by (station, city, state, last_update)
            grouped = defaultdict(lambda: {
                'state': None, 'city': None, 'station': None, 'station_id': None, 'datetime': None,
                'pm25': None, 'pm10': None, 'aqi': None, 'no2': None, 'so2': None, 'o3': None, 'co': None
            })
            for record in data['records']:
                station_name = record.get('station', 'Unknown Station')
                state = record.get('state', 'Unknown State')
                city = record.get('city', 'Unknown City')
                ts = record.get('last_update')
                if not ts:
                    continue
                try:
                    parsed_dt = pd.to_datetime(ts, dayfirst=True)
                except Exception:
                    continue
                key = (station_name, city, state, parsed_dt)
                g = grouped[key]
                g['state'] = state
                g['city'] = city
                g['station'] = station_name
                g['station_id'] = station_name
                g['datetime'] = parsed_dt
                pollutant = record.get('pollutant_id', '').upper()
                avg_value = record.get('avg_value') or record.get('pollutant_avg')
                try:
                    val = float(avg_value) if avg_value not in [None, '', 'NA'] else None
                except Exception:
                    val = None
                if pollutant == 'PM2.5':
                    g['pm25'] = val
                elif pollutant == 'PM10':
                    g['pm10'] = val
                elif pollutant == 'NO2':
                    g['no2'] = val
                elif pollutant == 'SO2':
                    g['so2'] = val
                elif pollutant == 'O3':
                    g['o3'] = val
                elif pollutant == 'CO':
                    g['co'] = val
                # Optionally, add more pollutants here
            # Calculate AQI (max of PM2.5/PM10 AQI if available)
            df_data = []
            for row in grouped.values():
                pm25 = row['pm25']
                pm10 = row['pm10']
                aqi = None
                if pm25 is not None or pm10 is not None:
                    pm25_aqi = self._calculate_aqi(pm25, 'pm25') if pm25 is not None else 0
                    pm10_aqi = self._calculate_aqi(pm10, 'pm10') if pm10 is not None else 0
                    aqi = max(pm25_aqi, pm10_aqi)
                row['aqi'] = aqi
                df_data.append(row)
            if not df_data:
                logger.warning("No valid data found within date range")
                return pd.DataFrame()
            df = pd.DataFrame(df_data)
            logger.info(f"Successfully parsed data for {len(df)} stations across India")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching all India data: {e}")
            return pd.DataFrame()
    
    def _safe_float(self, value):
        """Safely convert value to float, return None if invalid"""
        try:
            if value is None or value == '':
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

class OpenWeatherFetcher:
    def __init__(self, api_key: str = None, base_url: str = "https://api.openweathermap.org/data/2.5"):
        from config.config import settings
        self.api_key = api_key or settings.OPENWEATHER_API_KEY
        self.base_url = base_url

    def fetch_weather_data(self, city: str, lat: float, lon: float):
        """Fetch weather data for a city and convert to DataFrame"""
        try:
            # Fetch data using the existing fetch method
            data = self.fetch(city, lat, lon)
            
            if data is None:
                return pd.DataFrame()
            
            # Convert to DataFrame format
            df_data = [{
                'datetime': datetime.now(),
                'city': city,
                'temperature': data.get('temperature'),
                'humidity': data.get('humidity'),
                'wind_speed': data.get('wind_speed'),
                'wind_deg': data.get('wind_deg'),
                'pressure': data.get('pressure'),
                'weather': data.get('weather'),
                'clouds': 0,  # Add default clouds value
                'station_id': None  # Ensure station_id column exists
            }]
            
            return pd.DataFrame(df_data)
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {city}: {e}")
            return pd.DataFrame()

    def fetch(self, city_name: str, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetch weather data from OpenWeatherMap"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind'].get('deg', 0),
                'pressure': data['main']['pressure'],
                'weather': data['weather'][0]['main']
            }
        except Exception as e:
            logger.error(f"Error fetching weather data for {city_name}: {e}")
            return None

    def fetch_weather_for_datetimes(self, city: str, lat: float, lon: float, datetimes):
        """Fetch weather data for a list of datetimes (returns same value for all, as OWM free API only gives current weather)."""
        try:
            data = self.fetch(city, lat, lon)
            if data is None:
                return pd.DataFrame()
            df_data = []
            for dt in datetimes:
                df_data.append({
                    'datetime': dt,
                    'city': city,
                    'temperature': data.get('temperature'),
                    'humidity': data.get('humidity'),
                    'wind_speed': data.get('wind_speed'),
                    'wind_deg': data.get('wind_deg'),
                    'pressure': data.get('pressure'),
                    'weather': data.get('weather'),
                    'clouds': 0,
                    'station_id': None
                })
            return pd.DataFrame(df_data)
        except Exception as e:
            logger.error(f"Error fetching weather data for {city}: {e}")
            return pd.DataFrame()

# Create aliases for expected class names
AQIFetcher = IndiaGovAQIFetcher
WeatherFetcher = OpenWeatherFetcher

# Implement fetchers here 