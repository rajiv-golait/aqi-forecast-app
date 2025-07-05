"""Unified storage manager for local and cloud operations."""

import logging
import pandas as pd
from typing import Optional
from config.config import settings
from .local_storage import LocalStorageManager
from .supabase_client import SupabaseManager

logger = logging.getLogger(__name__)

class StorageManager:
    def __init__(self, use_cloud: bool = False):
        self.use_cloud = use_cloud
        self.local = LocalStorageManager()
        self.cloud = SupabaseManager() if use_cloud else None

    def save_forecast(self, forecast_df: pd.DataFrame, city: str) -> str:
        local_path = self.local.save_forecast(forecast_df, city)
        if self.use_cloud and self.cloud:
            self.cloud.upload_forecast(city, forecast_df)
        return local_path

    def load_forecast(self, city: str) -> Optional[pd.DataFrame]:
        if self.use_cloud and self.cloud:
            df = self.cloud.download_latest_forecast(city)
            if df is not None:
                return df
        return self.local.load_forecast(city)

    def save_model(self, model, scaler, feature_cols, model_name: str = 'aqi_forecast_model', version: Optional[str] = None):
        local_path = self.local.save_model(model, scaler, feature_cols, model_name)
        if self.use_cloud and self.cloud and version:
            self.cloud.upload_model(local_path, version)
        return local_path

    def load_model(self, model_name: str = 'aqi_forecast_model'):
        return self.local.load_model(model_name) 