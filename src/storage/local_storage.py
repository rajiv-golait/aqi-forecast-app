"""Local storage manager for saving/loading files locally."""

import os
import logging
import pandas as pd
from typing import Optional
from config.config import settings

logger = logging.getLogger(__name__)

class LocalStorageManager:
    def __init__(self, data_dir: Optional[str] = None, model_dir: Optional[str] = None):
        self.data_dir = data_dir or (settings.DATA_DIR / 'forecasts')
        self.model_dir = model_dir or (settings.MODEL_DIR / 'trained')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def save_forecast(self, forecast_df: pd.DataFrame, city: str) -> str:
        file_path = os.path.join(self.data_dir, f"{city}_forecast.csv")
        forecast_df.to_csv(file_path, index=False)
        logger.info(f"Saved forecast for {city} to {file_path}")
        return file_path

    def load_forecast(self, city: str) -> Optional[pd.DataFrame]:
        file_path = os.path.join(self.data_dir, f"{city}_forecast.csv")
        if not os.path.exists(file_path):
            logger.warning(f"Forecast file not found for {city}: {file_path}")
            return None
        df = pd.read_csv(file_path)
        logger.info(f"Loaded forecast for {city} from {file_path}")
        return df

    def save_model(self, model, scaler, feature_cols, model_name: str = 'aqi_forecast_model'):
        import joblib
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        features_path = os.path.join(self.model_dir, "feature_cols.pkl")
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_cols, features_path)
        logger.info(f"Saved model, scaler, and features to {self.model_dir}")
        return model_path

    def load_model(self, model_name: str = 'aqi_forecast_model'):
        import joblib
        from tensorflow.keras.models import load_model
        model_path = os.path.join(self.model_dir, f"{model_name}.h5")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        features_path = os.path.join(self.model_dir, "feature_cols.pkl")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None, None, None
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)
        logger.info(f"Loaded model, scaler, and features from {self.model_dir}")
        return model, scaler, feature_cols 