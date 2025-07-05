"""Authentication utilities for the AQI Forecast API."""

from fastapi import HTTPException, Header
from src.storage.supabase_client import SupabaseManager

supabase = SupabaseManager()

def api_key_auth(x_api_key: str = Header(..., alias="X-API-KEY")):
    key = x_api_key
    if not supabase.is_key_valid(key):
        raise HTTPException(status_code=403, detail="Invalid or expired API key")
    supabase.increment_usage(key)
    return key

# Implement authentication here 