"""Supabase client for cloud storage and database operations."""

from supabase import create_client, Client
from typing import Optional, Dict, Any
import pandas as pd
import os
import logging
from datetime import datetime, timedelta, timezone
from config.config import settings
import secrets
from dateutil.parser import isoparse

logger = logging.getLogger(__name__)

class SupabaseManager:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None, bucket: Optional[str] = None):
        self.url = url or settings.SUPABASE_URL
        self.key = key or settings.SUPABASE_KEY
        self.bucket_name = bucket or settings.SUPABASE_BUCKET
        
        # Check if Supabase is configured
        if not self.url or not self.key:
            logger.warning("Supabase not configured. Using local mode.")
            self.client = None
            self.enabled = False
        else:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.enabled = True
                self.ensure_bucket_exists()
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.client = None
                self.enabled = False

    def ensure_bucket_exists(self):
        """Ensure storage bucket exists (only if Supabase is enabled)."""
        if not self.enabled:
            return
            
        try:
            buckets = self.client.storage.list_buckets()
            if not any(b.name == self.bucket_name for b in buckets):
                self.client.storage.create_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {e}")

    def upload_forecast(self, city: str, forecast_df: pd.DataFrame) -> bool:
        """Upload forecast to Supabase (only if enabled)."""
        if not self.enabled:
            logger.warning("Supabase not enabled. Skipping upload.")
            return False
            
        try:
            # Always overwrite the same file for each city
            filename = f"forecasts/{city}/forecast.csv"
            csv_data = forecast_df.to_csv(index=False)
            self.client.storage.from_(self.bucket_name).upload(
                path=filename,
                file=csv_data.encode(),
                file_options={"content-type": "text/csv"}
            )
            logger.info(f"Uploaded forecast for {city} to Supabase: {filename}")
            return True
        except Exception as e:
            logger.error(f"Error uploading forecast: {e}")
            return False

    def download_latest_forecast(self, city: str) -> Optional[pd.DataFrame]:
        """Download latest forecast from Supabase (only if enabled)."""
        if not self.enabled:
            logger.warning("Supabase not enabled. Skipping download.")
            return None
            
        try:
            files = self.client.storage.from_(self.bucket_name).list(f"forecasts/{city}/")
            if not files:
                logger.warning(f"No forecast files found for {city}")
                return None
            latest_file = sorted(files, key=lambda x: x['name'])[-1]
            data = self.client.storage.from_(self.bucket_name).download(f"forecasts/{city}/{latest_file['name']}")
            df = pd.read_csv(pd.io.common.BytesIO(data))
            logger.info(f"Downloaded latest forecast for {city} from Supabase")
            return df
        except Exception as e:
            logger.error(f"Error downloading forecast: {e}")
            return None

    def upload_model(self, model_path: str, version: str) -> bool:
        """Upload model to Supabase (only if enabled)."""
        if not self.enabled:
            logger.warning("Supabase not enabled. Skipping model upload.")
            return False
            
        try:
            with open(model_path, 'rb') as f:
                self.client.storage.from_(self.bucket_name).upload(
                    path=f"models/{version}/model.h5",
                    file=f.read(),
                    file_options={"content-type": "application/octet-stream"}
                )
            logger.info(f"Uploaded model version {version} to Supabase")
            return True
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False

    def create_api_key(self, user_email: str = None, usage_limit: int = 1000, expires_in_days: int = 30) -> str:
        """Generate and store a new API key with usage limit and expiration."""
        if not self.enabled:
            # Generate a simple key for local mode
            key = secrets.token_urlsafe(32)
            logger.warning(f"Supabase not enabled. Generated local API key: {key[:8]}...")
            return key
            
        try:
            key = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
            record = {
                "key": key,
                "user_email": user_email,
                "usage_limit": usage_limit,
                "expires_at": expires_at.isoformat(),
                "active": True
            }
            self.client.table('api_keys').insert(record).execute()
            return key
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            # Fallback to local key
            key = secrets.token_urlsafe(32)
            logger.warning(f"Failed to create Supabase key. Generated local key: {key[:8]}...")
            return key

    def get_api_key(self, key: str):
        """Fetch API key record from Supabase."""
        if not self.enabled:
            # Return a mock record for local mode
            return {
                "key": key,
                "user_email": "local@example.com",
                "usage_limit": 1000,
                "usage_count": 0,
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "active": True
            }
            
        try:
            resp = self.client.table('api_keys').select('*').eq('key', key).single().execute()
            return resp.data if hasattr(resp, 'data') else None
        except Exception as e:
            logger.error(f"Error getting API key: {e}")
            return None

    def increment_usage(self, key: str):
        """Increment usage count for the API key."""
        if not self.enabled:
            logger.debug("Supabase not enabled. Skipping usage increment.")
            return
            
        try:
            # Fetch current usage_count
            record = self.get_api_key(key)
            if not record:
                return
            current_count = record.get("usage_count", 0)
            new_count = current_count + 1
            self.client.table('api_keys').update({"usage_count": new_count}).eq('key', key).execute()
        except Exception as e:
            logger.error(f"Error incrementing usage: {e}")

    def is_key_valid(self, key: str) -> bool:
        """Check if API key is valid, active, not expired, and within usage limit."""
        if not self.enabled:
            # For local mode, accept any key that looks like a valid format
            return len(key) >= 32
            
        try:
            record = self.get_api_key(key)
            if not record:
                return False
            if not record.get('active', True):
                return False
            if record.get('expires_at') and isoparse(record['expires_at']) < datetime.now(timezone.utc):
                return False
            if record.get('usage_limit') is not None and record.get('usage_count', 0) >= record['usage_limit']:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return False

    def get_api_key_by_email(self, user_email: str):
        """Fetch API key record by user email from Supabase."""
        if not self.enabled:
            # Return a mock record for local mode
            return {
                "key": "local-test-key",
                "user_email": user_email,
                "usage_limit": 1000,
                "usage_count": 0,
                "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "active": True
            }
        try:
            resp = self.client.table('api_keys').select('*').eq('user_email', user_email).single().execute()
            return resp.data if hasattr(resp, 'data') else None
        except Exception as e:
            logger.error(f"Error getting API key by email: {e}")
            return None 