"""Main FastAPI app for AQI Forecast API with authentication and station-specific endpoints."""

from fastapi import FastAPI, HTTPException, Depends, status, Header, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import logging
import pandas as pd
import os
import numpy as np

from config.config import settings
from api.auth import api_key_auth, get_api_usage_info, require_admin_key
from api.models import (
    LiveAQIData, StationForecast, StationInfo, StationSearchResponse,
    APIKeyCreateRequest, APIKeyCreateResponse, APIUsageResponse,
    SuccessResponse, ErrorResponse, HealthCheckResponse
)
from src.data.station_service import station_service
from src.ml.forecast_service import forecast_service
from src.storage.supabase_client import SupabaseManager
from src.utils.helpers import clean_json_response
from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.data.interpolation import idw_interpolate
from src.ml.forecaster import load_regression_model
from src.data.interpolation import get_weather_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AQI Forecast API",
    description="API for Air Quality Index forecasts and live data for Indian cities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
supabase = SupabaseManager()

# Load regression model at startup (global)
try:
    AQI_REGRESSION_MODEL = load_regression_model()
except Exception:
    AQI_REGRESSION_MODEL = None

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return clean_json_response({
        "message": "AQI Forecast API",
        "version": "1.0.0",
        "description": "API for Air Quality Index forecasts and live data",
        "endpoints": {
            "live_data": "/live/{station_id}",
            "forecast": "/forecast/{station_id}",
            "forecast_summary": "/forecast/{station_id}/summary",
            "search_stations": "/stations/search",
            "station_info": "/stations/{station_id}",
            "api_usage": "/usage",
            "health": "/health"
        },
        "authentication": "Requires X-API-KEY header",
        "documentation": "/docs"
    })

@app.get("/live/{station_id}", response_model=LiveAQIData)
async def get_live_aqi(
    station_id: str,
    api_key: str = Depends(api_key_auth)
):
    """Get live AQI data for a specific station."""
    try:
        live_data = station_service.fetch_live_aqi(station_id)
        if not live_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Live AQI data not available for station: {station_id}"
            )
        return clean_json_response(live_data.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching live AQI for {station_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching live AQI data"
        )

@app.get("/forecast/available_stations", response_model=List[str])
async def get_available_forecast_stations(api_key: str = Depends(api_key_auth)):
    """Get a list of all station IDs with available forecast data."""
    try:
        station_ids = forecast_service.get_available_stations()
        return station_ids
    except Exception as e:
        logger.error(f"Error getting available forecast stations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting available forecast stations"
        )

@app.get("/forecast/{station_id}", response_model=StationForecast)
async def get_station_forecast(
    station_id: str,
    api_key: str = Depends(api_key_auth)
):
    """Get AQI forecast for a specific station."""
    try:
        # Check if forecast cache is loaded and not empty
        if not forecast_service.forecast_cache or len(forecast_service.forecast_cache) == 0:
            logger.error("Forecast cache is empty! No forecast data available.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Forecast data is not available. Please generate forecasts first."
            )
        forecast = forecast_service.get_station_forecast(station_id)
        if not forecast:
            logger.error(f"Forecast not available for station: {station_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Forecast not available for station: {station_id}"
            )
        return clean_json_response(forecast.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forecast for {station_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching forecast data"
        )

@app.get("/forecast/{station_id}/summary")
async def get_forecast_summary(
    station_id: str,
    api_key: str = Depends(api_key_auth)
):
    """Get a summary of forecast data for a station."""
    try:
        summary = forecast_service.get_forecast_summary(station_id)
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Forecast summary not available for station: {station_id}"
            )
        return clean_json_response(summary)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching forecast summary for {station_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching forecast summary"
        )

@app.get("/stations/search", response_model=StationSearchResponse)
async def search_stations(
    city: Optional[str] = Query(None, description="Filter by city name"),
    state: Optional[str] = Query(None, description="Filter by state name"),
    station_name: Optional[str] = Query(None, description="Filter by station name"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of results"),
    api_key: str = Depends(api_key_auth)
):
    """Search for stations by various criteria."""
    try:
        stations = station_service.search_stations(
            city=city,
            state=state,
            station_name=station_name,
            limit=limit
        )
        
        station_infos = []
        for station in stations:
            station_info = station_service.get_station_info(station.station_id)
            if station_info:
                station_infos.append(station_info)
        
        return clean_json_response(StationSearchResponse(
            stations=station_infos,
            total=len(station_infos),
            limit=limit
        ).dict())
    except Exception as e:
        logger.error(f"Error searching stations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error searching stations"
        )

@app.get("/stations/{station_id}", response_model=StationInfo)
async def get_station_info(
    station_id: str,
    api_key: str = Depends(api_key_auth)
):
    """Get information about a specific station."""
    try:
        station_info = station_service.get_station_info(station_id)
        if not station_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Station not found: {station_id}"
            )
        return clean_json_response(station_info.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching station info for {station_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching station information"
        )

@app.get("/usage", response_model=APIUsageResponse)
async def get_api_usage(api_key: str = Depends(api_key_auth)):
    """Get current API usage information for the authenticated key."""
    try:
        usage_info = get_api_usage_info(api_key)
        if not usage_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API usage info not found"
            )
        return clean_json_response(usage_info.dict())
    except Exception as e:
        logger.error(f"Error fetching API usage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching API usage info"
        )

@app.post("/admin/generate_key", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def generate_api_key(
    request: APIKeyCreateRequest,
    admin_key: str = Depends(require_admin_key)
):
    """Generate a new API key (admin only)."""
    try:
        key = supabase.create_api_key(
            user_email=request.user_email,
            usage_limit=request.usage_limit,
            expires_in_days=request.expires_in_days
        )
        
        # Get key info for response
        key_info = supabase.get_api_key(key)
        expires_at = key_info.get('expires_at') if key_info else None
        
        return APIKeyCreateResponse(
            key=key,
            usage_limit=request.usage_limit,
            expires_at=expires_at,
            message="API key generated successfully"
        )
    except Exception as e:
        logger.error(f"Error generating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating API key"
        )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check services
        station_count = len(station_service.stations_cache)
        forecast_metadata = forecast_service.get_forecast_metadata()
        
        # Check database connection
        db_status = "healthy"
        try:
            # Simple test query
            test_result = supabase.get_api_key("test")
            db_status = "healthy"
        except:
            db_status = "degraded"
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            model_version=forecast_service.model_version,
            total_stations=station_count,
            database_status=db_status
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            model_version="unknown",
            total_stations=0,
            database_status="unhealthy"
        )

@app.get("/metadata")
async def get_api_metadata(api_key: str = Depends(api_key_auth)):
    """Get API metadata and available data information."""
    try:
        forecast_metadata = forecast_service.get_forecast_metadata()
        
        return {
            "api_version": "1.0.0",
            "forecast_coverage": forecast_metadata,
            "available_endpoints": [
                "/live/{station_id}",
                "/forecast/{station_id}",
                "/forecast/{station_id}/summary",
                "/stations/search",
                "/stations/{station_id}",
                "/usage"
            ],
            "authentication": "X-API-KEY header required",
            "rate_limits": "Based on API key usage limits"
        }
    except Exception as e:
        logger.error(f"Error fetching API metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching API metadata"
        )

@app.post("/register", response_model=APIKeyCreateResponse)
async def register_user(user_email: str = Body(..., embed=True)):
    """
    Register a new user and issue an API key.
    """
    try:
        # Check if user already has a key
        existing_key = supabase.get_api_key_by_email(user_email)
        if existing_key and existing_key.get('active', True):
            return APIKeyCreateResponse(
                key=existing_key['key'],
                usage_limit=existing_key['usage_limit'],
                expires_at=existing_key['expires_at'],
                message="API key already exists for this email"
            )
        # Generate a new key
        key = supabase.create_api_key(
            user_email=user_email,
            usage_limit=1000,  # default limit
            expires_in_days=30
        )
        key_info = supabase.get_api_key(key)
        return APIKeyCreateResponse(
            key=key,
            usage_limit=key_info['usage_limit'],
            expires_at=key_info['expires_at'],
            message="API key generated successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error registering user: {e}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# --- NEW: Frontend endpoints ---
@app.get("/history/{station_id}")
async def get_station_history(station_id: str, days: int = Query(30, ge=1, le=365), api_key: str = Depends(api_key_auth)):
    """Get historical AQI for a station from real data. Patch: reconstruct station_id from city and station if needed. Replace non-finite AQI with None."""
    csv_path = os.path.join("data", "latest_aqi_weather.csv")
    if not os.path.exists(csv_path):
        return {"station_id": station_id, "history": [], "error": "Historical data not found."}
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        # Try to match by station_id column first
        filtered = df[df["station_id"] == station_id]
        if filtered.empty:
            # If no match, reconstruct station_id from city and station columns
            def reconstruct_id(row):
                return f"{row['city']}_{row['station']}".replace(" ", "_").replace("-", "_").replace(",", "_")
            df["_recon_id"] = df.apply(reconstruct_id, axis=1)
            filtered = df[df["_recon_id"] == station_id]
        if filtered.empty:
            return {"station_id": station_id, "history": [], "error": "No data for this station_id (or city/station)."}
        # Convert datetime and filter by days
        filtered["datetime"] = pd.to_datetime(filtered["datetime"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        filtered = filtered[filtered["datetime"] >= cutoff]
        # Sort and format output, replace non-finite AQI with None
        history = [
            {"datetime": row["datetime"].isoformat(), "aqi": row["aqi"] if pd.notnull(row["aqi"]) and np.isfinite(row["aqi"]) else None}
            for _, row in filtered.sort_values("datetime").iterrows()
        ]
        return {"station_id": station_id, "history": history}
    except Exception as e:
        return {"station_id": station_id, "history": [], "error": str(e)}

@app.get("/advisory/{aqi}")
async def get_health_advisory(aqi: int):
    if aqi <= 50:
        return {"advice": "Air quality is good. No precautions needed."}
    elif aqi <= 100:
        return {"advice": "Air quality is satisfactory. Sensitive individuals should take care."}
    elif aqi <= 200:
        return {"advice": "Air quality is moderate. Consider reducing outdoor activity."}
    elif aqi <= 300:
        return {"advice": "Unhealthy. Wear a mask and avoid outdoor exercise."}
    elif aqi <= 400:
        return {"advice": "Very unhealthy. Stay indoors and use air purifiers."}
    else:
        return {"advice": "Hazardous! Avoid all outdoor activity and follow emergency instructions."}

@app.get("/advisory/auto/{station_id}")
async def get_auto_advisory(station_id: str, api_key: str = Depends(api_key_auth)):
    forecast = forecast_service.get_station_forecast(station_id)
    if forecast and forecast.forecasts:
        aqi = forecast.forecasts[0].aqi
        return await get_health_advisory(aqi)
    return {"advice": "No AQI data available for this station."}

@app.get("/stations/geojson")
async def get_stations_geojson(api_key: str = Depends(api_key_auth)):
    # Mocked from station_service
    features = []
    for s in station_service.stations_cache.values():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [s.lon, s.lat]},
            "properties": {"station_id": s.station_id, "name": s.station_name, "city": s.city, "state": s.state}
        })
    return {"type": "FeatureCollection", "features": features}

@app.post("/alerts/subscribe")
async def subscribe_alert(station_id: str = Body(...), user_email: str = Body(...), api_key: str = Depends(api_key_auth)):
    # Mock: just acknowledge
    return {"message": f"Subscribed {user_email} to alerts for {station_id}. (Mocked)"}

@app.post("/alerts/unsubscribe")
async def unsubscribe_alert(station_id: str = Body(...), user_email: str = Body(...), api_key: str = Depends(api_key_auth)):
    # Mock: just acknowledge
    return {"message": f"Unsubscribed {user_email} from alerts for {station_id}. (Mocked)"}

@app.get("/alerts/check/{station_id}")
async def check_alert(station_id: str, threshold: int = Query(150), api_key: str = Depends(api_key_auth)):
    forecast = forecast_service.get_station_forecast(station_id)
    if forecast and forecast.forecasts:
        aqi = forecast.forecasts[0].aqi
        alert = aqi >= threshold
        return {"station_id": station_id, "aqi": aqi, "alert": alert, "threshold": threshold}
    return {"station_id": station_id, "alert": False, "message": "No AQI data available."}

@app.get("/cities")
async def get_cities(api_key: str = Depends(api_key_auth)):
    cities = sorted(set(s.city for s in station_service.stations_cache.values()))
    return {"cities": cities}

@app.get("/categories")
async def get_aqi_categories():
    return {
        "categories": [
            {"range": "0-50", "label": "Good", "advice": "No precautions needed."},
            {"range": "51-100", "label": "Satisfactory", "advice": "Sensitive individuals should take care."},
            {"range": "101-200", "label": "Moderate", "advice": "Consider reducing outdoor activity."},
            {"range": "201-300", "label": "Poor", "advice": "Wear a mask and avoid outdoor exercise."},
            {"range": "301-400", "label": "Very Poor", "advice": "Stay indoors and use air purifiers."},
            {"range": "401+", "label": "Severe", "advice": "Avoid all outdoor activity and follow emergency instructions."}
        ]
    }
# --- END NEW ---

# Helper to load station metadata and latest AQI
STATION_COORDS_PATH = "data/merged_coords.csv"
LATEST_AQI_PATH = "data/latest_aqi_weather.csv"

def get_station_list():
    coords_df = pd.read_csv(STATION_COORDS_PATH)
    aqi_df = pd.read_csv(LATEST_AQI_PATH)
    merged = pd.merge(coords_df, aqi_df, left_on="StationName", right_on="station")
    stations = []
    for _, row in merged.iterrows():
        try:
            aqi_val = float(row["aqi"])
        except Exception:
            continue
        stations.append({
            "id": row["StationName"],
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "aqi": aqi_val
        })
    return stations

class AQIEstimateRequest(BaseModel):
    lat: float
    lon: float
    k: int = 4

class AQIEstimateResponse(BaseModel):
    predicted_aqi: float
    neighbors: list
    weather: dict
    method: str = "IDW"

from fastapi import APIRouter
router = APIRouter()

@router.post("/estimate/aqi_at_location", response_model=AQIEstimateResponse)
def estimate_aqi_at_location(req: AQIEstimateRequest):
    stations = get_station_list()
    aqi_est, neighbors, dists = idw_interpolate(req.lat, req.lon, stations, k=req.k)
    weather = get_weather_features(req.lat, req.lon)
    predicted_aqi = None
    method = "IDW"
    # Prepare features for regression model if available
    if AQI_REGRESSION_MODEL is not None and all(v is not None for v in weather.values()):
        # Example: model expects [idw_aqi, temp, humidity, wind_speed]
        features = np.array([[aqi_est, weather["temp"], weather["humidity"], weather["wind_speed"]]])
        try:
            predicted_aqi = float(AQI_REGRESSION_MODEL.predict(features)[0])
            method = "IDW+WeatherRegression"
        except Exception:
            predicted_aqi = None
    # Fallback to IDW if regression fails
    if predicted_aqi is None:
        predicted_aqi = aqi_est
    return {
        "predicted_aqi": predicted_aqi,
        "neighbors": [
            {"id": n["id"], "lat": n["lat"], "lon": n["lon"], "aqi": n["aqi"], "distance_km": float(d)}
            for n, d in zip(neighbors, dists)
        ],
        "weather": weather,
        "method": method
    }

# Register the router if not already
try:
    app.include_router(router)
except Exception:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Print all registered routes for debugging
    print("\nRegistered routes:")
    for route in app.routes:
        print(route.path)