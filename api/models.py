"""Pydantic models for API request and response schemas."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum

class AQICategory(str, Enum):
    GOOD = "Good"
    SATISFACTORY = "Satisfactory"
    MODERATE = "Moderate"
    POOR = "Poor"
    VERY_POOR = "Very Poor"
    SEVERE = "Severe"

class StationInfo(BaseModel):
    station_id: str
    station_name: str
    city: str
    state: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    last_updated: datetime

class LiveAQIData(BaseModel):
    station_id: str
    station_name: str
    city: str
    state: str
    timestamp: datetime
    pm25: Optional[float] = None
    pm10: Optional[float] = None
    no2: Optional[float] = None
    so2: Optional[float] = None
    co: Optional[float] = None
    o3: Optional[float] = None
    aqi: int
    aqi_category: AQICategory
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class ForecastPoint(BaseModel):
    datetime: datetime
    pm25: float
    pm10: float
    aqi: int
    aqi_category: AQICategory
    confidence: Optional[float] = None

class StationForecast(BaseModel):
    station_id: str
    station_name: str
    city: str
    state: str
    forecast_generated_at: datetime
    model_version: str
    forecasts: List[ForecastPoint]

class APIKeyInfo(BaseModel):
    key_id: str
    user_email: Optional[str] = None
    usage_limit: int
    current_usage: int
    expires_at: datetime
    is_active: bool
    created_at: datetime

class APIKeyCreateRequest(BaseModel):
    user_email: Optional[str] = Field(None, description="User email for tracking")
    usage_limit: int = Field(1000, ge=1, le=100000, description="Maximum API calls allowed")
    expires_in_days: int = Field(30, ge=1, le=365, description="Days until key expires")

class APIKeyCreateResponse(BaseModel):
    key: str
    usage_limit: int
    expires_at: datetime
    message: str

class StationSearchRequest(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    station_name: Optional[str] = None
    limit: int = Field(50, ge=1, le=100)

class StationSearchResponse(BaseModel):
    stations: List[StationInfo]
    total: int
    page: int = 1
    limit: int = 50

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now)

class SuccessResponse(BaseModel):
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class APIUsageResponse(BaseModel):
    key_id: str
    current_usage: int
    usage_limit: int
    remaining_calls: int
    expires_at: datetime
    is_active: bool

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    model_version: str
    total_stations: int
    database_status: str

# Implement API models here 