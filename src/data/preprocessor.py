# src/data/preprocessor.py
"""Data preprocessor for feature engineering and cleaning."""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        pass

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataframe (forward fill, interpolate)."""
        if df is None or df.empty:
            logger.warning("Input DataFrame is empty or None.")
            return df
        df = df.sort_values(['city', 'datetime'])
        df = df.groupby('city').apply(lambda x: x.fillna(method='ffill', limit=3))
        df = df.reset_index(drop=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df.groupby('city')[numeric_cols].apply(
            lambda x: x.interpolate(method='linear', limit=2)
        ).reset_index(drop=True)
        logger.info("Missing values handled (forward fill, interpolate).")
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features (hour, day_of_week, month) for ML models."""
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            logger.info("Time features added: hour, day_of_week, month.")
        else:
            logger.warning("'datetime' column not found for time feature engineering.")
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full preprocessing pipeline: handle missing values, add features."""
        df = self.handle_missing_values(df)
        df = self.add_time_features(df)
        # Feature engineering: rolling means and lags for pm25 and pm10
        df = df.sort_values(['city', 'datetime'])
        df['pm25_rolling_6h'] = df.groupby('city')['pm25'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
        df['pm25_lag_1h'] = df.groupby('city')['pm25'].shift(1)
        df['pm10_rolling_6h'] = df.groupby('city')['pm10'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
        df['pm10_lag_1h'] = df.groupby('city')['pm10'].shift(1)
        return df 