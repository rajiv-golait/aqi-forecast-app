"""Authentication utilities for the AQI Forecast API."""

from fastapi import HTTPException, Header, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging
from datetime import datetime

from src.storage.supabase_client import SupabaseManager
from api.models import APIUsageResponse

logger = logging.getLogger(__name__)

security = HTTPBearer()
supabase = SupabaseManager()

class APIKeyAuth:
    def __init__(self):
        self.supabase = SupabaseManager()
    
    async def __call__(self, x_api_key: str = Header(..., alias="X-API-KEY")) -> str:
        """Validate API key and track usage."""
        try:
            # Validate the API key
            if not self.supabase.is_key_valid(x_api_key):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or expired API key"
                )
            
            # Get key info for usage tracking
            key_info = self.supabase.get_api_key(x_api_key)
            if not key_info:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key not found"
                )
            
            # Check if key is active
            if not key_info.get('active', True):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key is deactivated"
                )
            
            # Check usage limit
            current_usage = key_info.get('usage_count', 0)
            usage_limit = key_info.get('usage_limit', 0)
            
            if usage_limit > 0 and current_usage >= usage_limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"API usage limit exceeded. Limit: {usage_limit}, Used: {current_usage}"
                )
            
            # Increment usage count
            self.supabase.increment_usage(x_api_key)
            
            logger.info(f"API key validated successfully. Key: {x_api_key[:8]}..., Usage: {current_usage + 1}/{usage_limit}")
            
            return x_api_key
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )

# Create auth instance
api_key_auth = APIKeyAuth()

def get_api_usage_info(api_key: str) -> Optional[APIUsageResponse]:
    """Get API key usage information."""
    try:
        key_info = supabase.get_api_key(api_key)
        if not key_info:
            return None
        
        current_usage = key_info.get('usage_count', 0)
        usage_limit = key_info.get('usage_limit', 0)
        
        return APIUsageResponse(
            key_id=key_info.get('id', ''),
            current_usage=current_usage,
            usage_limit=usage_limit,
            remaining_calls=max(0, usage_limit - current_usage),
            expires_at=key_info.get('expires_at'),
            is_active=key_info.get('active', True)
        )
    except Exception as e:
        logger.error(f"Error getting API usage info: {e}")
        return None

def require_admin_key(admin_key: str = Header(..., alias="X-ADMIN-KEY")) -> str:
    """Validate admin key for administrative operations."""
    from config.config import settings
    
    if not settings.ADMIN_KEY or admin_key != settings.ADMIN_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key"
        )
    return admin_key 