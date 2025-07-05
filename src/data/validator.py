"""Data validator for checking data quality and consistency."""

import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, pollutants: Optional[List[str]] = None):
        self.pollutants = pollutants or [
            'pm25', 'pm10', 'no', 'no2', 'nox', 'nh3', 'co', 'so2', 'o3'
        ]

    def validate(self, df: pd.DataFrame) -> bool:
        """Run basic validation checks on the dataframe."""
        if df is None or df.empty:
            logger.error("DataFrame is empty or None.")
            return False
        logger.info(f"Loaded {len(df)} rows.")
        logger.info(f"Columns: {df.columns.tolist()}")
        # Missing values
        missing = df.isnull().sum()
        logger.info("Missing values per column:")
        logger.info(f"\n{missing}")
        # Date range
        if 'datetime' in df.columns:
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        if 'city' in df.columns:
            logger.info(f"Unique cities: {df['city'].nunique()}")
        # Negative values
        for col in self.pollutants:
            if col in df.columns:
                negatives = (df[col] < 0).sum()
                if negatives > 0:
                    logger.warning(f"{negatives} negative values in {col}")
        logger.info("Validation complete.")
        return True 