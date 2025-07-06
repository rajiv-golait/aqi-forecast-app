import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # API Keys
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY")
    
    # Supabase configuration (optional)
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "aqi-forecast")
    
    # Admin key for API key generation
    ADMIN_KEY = os.getenv("ADMIN_KEY", "admin_key_123")
    
    # Data retention
    DATA_RETENTION_DAYS = 7  # Keep only last 7 days
    
    # API endpoints
    DATA_GOV_BASE_URL = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    
    # Cities configuration
    CITIES = {
        "Delhi": {"lat": 28.6139, "lon": 77.2090, "radius": 50000},
        "Mumbai": {"lat": 19.0760, "lon": 72.8777, "radius": 40000},
        "Bangalore": {"lat": 12.9716, "lon": 77.5946, "radius": 35000},
        "Chennai": {"lat": 13.0827, "lon": 80.2707, "radius": 30000},
        "Kolkata": {"lat": 22.5726, "lon": 88.3639, "radius": 30000},
        "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "radius": 35000},
        "Pune": {"lat": 18.5204, "lon": 73.8567, "radius": 25000},
        "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "radius": 25000}
    }
    
    # ML Model parameters
    SEQUENCE_LENGTH = 168  # 7 days of hourly data
    FORECAST_HOURS = 72
    FEATURES = [
        'pm25', 'pm10', 'no', 'no2', 'nox', 'nh3', 'co', 'so2', 'o3',
        'temperature', 'humidity', 'wind_speed', 'wind_deg'
    ]
    
    # Model version
    MODEL_VERSION = "3day_ensemble_v1.0"
    
    FORECAST_HORIZON = FORECAST_HOURS

# Create a settings instance for easy importing
settings = Config()