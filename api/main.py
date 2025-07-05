"""Main FastAPI app for AQI Forecast API."""

from fastapi import FastAPI, HTTPException, Depends, Security, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import os
import glob
import logging
import secrets

from config.config import settings
from src.storage.storage_manager import StorageManager
from api.auth import api_key_auth
from src.storage.supabase_client import SupabaseManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AQI Forecast API",
    description="API for Air Quality Index forecasts for Indian cities",
    version="1.0.0"
)

security = HTTPBearer()

storage = StorageManager(use_cloud=False)  # Set to True to enable Supabase

ADMIN_KEY = getattr(settings, 'ADMIN_KEY', None)
print("Loaded ADMIN_KEY:", ADMIN_KEY)
supabase = SupabaseManager()

# Global variable to store forecasts
FORECASTS = {}

# Pydantic models for response
class ForecastPoint(BaseModel):
    datetime: str
    pm25: float
    pm10: float
    aqi: int

class ForecastResponse(BaseModel):
    city: str
    forecasts: List[dict]
    generated_at: datetime
    model_version: str

class CityListResponse(BaseModel):
    cities: List[str]
    total: int

class WelcomeResponse(BaseModel):
    message: str
    version: str
    endpoints: List[str]
    available_cities: int

class ErrorResponse(BaseModel):
    error: str
    detail: str

class APIKeyCreateRequest(BaseModel):
    user_email: str = None
    usage_limit: int = 1000
    expires_in_days: int = 30

class APIKeyCreateResponse(BaseModel):
    key: str
    usage_limit: int
    expires_at: datetime

def load_forecast_file(filepath: str) -> Dict:
    """Load a single forecast CSV file"""
    try:
        # Extract city name from filename
        filename = os.path.basename(filepath)
        city = filename.replace('forecast_', '').replace('.csv', '').replace('_', ' ')
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Convert to dictionary format
        forecast_data = {
            'city': city,
            'forecast_generated_at': df['forecast_generated_at'].iloc[0] if 'forecast_generated_at' in df.columns else None,
            'forecasts': []
        }
        
        # Process each row
        for _, row in df.iterrows():
            forecast_point = {
                'datetime': row['datetime'],
                'pm25': round(float(row['pm25']), 2),
                'pm10': round(float(row['pm10']), 2),
                'aqi': int(row['aqi']) if 'aqi' in row else 0
            }
            forecast_data['forecasts'].append(forecast_point)
        
        logger.info(f"Loaded forecast for {city}: {len(forecast_data['forecasts'])} points")
        return city, forecast_data
        
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None, None

def load_all_forecasts():
    """Load all forecast CSV files from data/forecasts/"""
    global FORECASTS
    FORECASTS = {}
    
    # Find all forecast CSV files
    forecast_dir = 'data/forecasts'
    if not os.path.exists(forecast_dir):
        logger.warning(f"Forecast directory {forecast_dir} not found")
        return
    
    forecast_files = glob.glob(os.path.join(forecast_dir, 'forecast_*.csv'))
    
    if not forecast_files:
        logger.warning("No forecast files found")
        return
    
    logger.info(f"Found {len(forecast_files)} forecast files")
    
    # Load each file
    for filepath in forecast_files:
        city, forecast_data = load_forecast_file(filepath)
        if city and forecast_data:
            # Store both original case and lowercase for flexible lookup
            FORECASTS[city.lower()] = forecast_data
            FORECASTS[city] = forecast_data
    
    logger.info(f"Loaded forecasts for {len(set(k.lower() for k in FORECASTS.keys()))} cities")

@app.on_event("startup")
async def startup_event():
    """Load forecasts when the API starts"""
    logger.info("Starting AQI Forecast API...")
    load_all_forecasts()
    logger.info("API startup complete")

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if settings.API_KEY and credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid authentication")
    return credentials.credentials

@app.get("/")
async def root():
    return {
        "message": "AQI Forecast API",
        "version": "1.0.0",
        "endpoints": ["/forecast/{city}", "/cities", "/health"]
    }

@app.get("/forecast/{city}", response_model=ForecastResponse)
async def get_forecast(city: str, token: str = Depends(api_key_auth)):
    city = city.title()
    if city not in settings.CITIES:
        raise HTTPException(status_code=404, detail=f"City {city} not found")
    forecast_df = storage.load_forecast(city)
    if forecast_df is None or forecast_df.empty:
        raise HTTPException(status_code=404, detail=f"No forecast available for {city}")
    forecasts = forecast_df.to_dict('records')
        return ForecastResponse(
            city=city,
        forecasts=forecasts,
        generated_at=datetime.now(),
        model_version=settings.MODEL_VERSION
    )

@app.get("/cities", response_model=CityListResponse)
async def list_cities(token: str = Depends(api_key_auth)):
    return CityListResponse(
        cities=settings.CITIES,
        total=len(settings.CITIES)
    )

@app.get("/health")
async def health_check():
    try:
        # Check storage (local or cloud)
        healthy = True
        for city in settings.CITIES[:1]:  # Just check one city for speed
            df = storage.load_forecast(city)
            if df is None:
                healthy = False
        status = "healthy" if healthy else "degraded"
    except Exception:
        status = "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now(),
        "model_version": settings.MODEL_VERSION
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

@app.post("/admin/generate_key", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def generate_api_key(request: APIKeyCreateRequest, admin_key: str = Header(..., alias="X-ADMIN-KEY")):
    if not ADMIN_KEY or admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Admin authentication failed")
    key = supabase.create_api_key(
        user_email=request.user_email,
        usage_limit=request.usage_limit,
        expires_in_days=request.expires_in_days
    )
    expires_at = (datetime.utcnow() + timedelta(days=request.expires_in_days))
    return APIKeyCreateResponse(key=key, usage_limit=request.usage_limit, expires_at=expires_at)

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )