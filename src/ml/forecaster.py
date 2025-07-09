"""Forecast generator for AQI predictions."""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from tensorflow.keras.models import load_model
from config.config import settings
import pickle

logger = logging.getLogger(__name__)

class AQIForecaster:
    def __init__(self, model_dir: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.sequence_length = settings.SEQUENCE_LENGTH
        self.forecast_horizon = settings.FORECAST_HORIZON
        self.model_dir = model_dir or (settings.MODEL_DIR / 'trained')
        self._load_model_and_config()

    def _load_model_and_config(self):
        try:
            model_path = os.path.join(self.model_dir, 'aqi_forecast_model.h5')
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            features_path = os.path.join(self.model_dir, 'feature_cols.pkl')
            if not os.path.exists(model_path):
                logger.error(f"No trained model found at {model_path}")
                return False
            self.model = load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_cols = joblib.load(features_path)
            logger.info(f"Loaded model, scaler, and features from {self.model_dir}")
            return True
        except Exception as e:
            logger.error(f"Error loading model/config: {e}")
            return False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if 'hour' not in df.columns:
                df['hour'] = df['datetime'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['datetime'].dt.dayofweek
            if 'month' not in df.columns:
                df['month'] = df['datetime'].dt.month
            for col in self.feature_cols:
                if 'rolling' in col and col not in df.columns:
                    base_col = col.split('_rolling')[0]
                    window = int(col.split('_')[-1].replace('h', ''))
                    if base_col in df.columns:
                        df[col] = df.groupby('city')[base_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                elif 'lag' in col and col not in df.columns:
                    base_col = col.split('_lag')[0]
                    lag = int(col.split('_')[-1].replace('h', ''))
                    if base_col in df.columns:
                        df[col] = df.groupby('city')[base_col].shift(lag)
            df = df.fillna(method='ffill').fillna(method='bfill')
            for col in self.feature_cols:
                if col not in df.columns:
                    logger.warning(f"Missing feature {col}, filling with zeros")
                    df[col] = 0
            return df
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return df

    def generate_forecast_for_city(self, df: pd.DataFrame, city: str) -> Optional[pd.DataFrame]:
        try:
            city_data = df[df['city'] == city].sort_values('datetime')
            if len(city_data) < self.sequence_length:
                logger.warning(f"{city}: Insufficient data ({len(city_data)} < {self.sequence_length})")
                return None
            latest_data = city_data.tail(self.sequence_length).copy()
            features = latest_data[self.feature_cols].values
            features_scaled = self.scaler.transform(features)
            sequence = features_scaled.reshape(1, self.sequence_length, len(self.feature_cols))
            predictions = self.model.predict(sequence, verbose=0)[0]
            # PM2.5 and PM10 predictions
            forecast_times = [latest_data['datetime'].iloc[-1] + pd.Timedelta(hours=i+1) for i in range(self.forecast_horizon)]
            forecast_df = pd.DataFrame({
                'datetime': forecast_times,
                'city': city,
                'pm25_pred': predictions[:, 0],
                'pm10_pred': predictions[:, 1]
            })
            return forecast_df
        except Exception as e:
            logger.error(f"Error generating forecast for {city}: {e}")
            return None

    def generate_all_forecasts(self, df: pd.DataFrame, cities: List[str]) -> Dict[str, pd.DataFrame]:
        forecasts = {}
        for city in cities:
            forecast = self.generate_forecast_for_city(df, city)
            if forecast is not None:
                forecasts[city] = forecast
        logger.info(f"Generated forecasts for {len(forecasts)}/{len(cities)} cities.")
        return forecasts

    def save_forecast(self, forecast_df: pd.DataFrame, city: str, output_dir: Optional[str] = None):
        output_dir = output_dir or (settings.DATA_DIR / 'forecasts')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{city}_forecast.csv")
        forecast_df.to_csv(file_path, index=False)
        logger.info(f"Saved forecast for {city} to {file_path}")

def load_regression_model(model_path=None):
    if model_path is None:
        model_path = os.path.join("models", "trained", "aqi_regressor.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model 