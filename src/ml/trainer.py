"""Model trainer for AQI forecasting models."""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from config.config import settings

logger = logging.getLogger(__name__)

class SimpleLSTMModel:
    def __init__(self, n_features, sequence_length=24, forecast_horizon=24):
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(self.forecast_horizon * 2)(x)
        outputs = layers.Reshape((self.forecast_horizon, 2))(outputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        return model

REQUIRED_COLS = [
    'datetime','city','station_id','pm25','pm10','no','no2','nox','nh3','co','so2','o3','benzene','toluene','xylene','aqi','aqi_category','source','frequency','aqi_bucket','pm2_5','pm25_rolling_6h','pm25_lag_1h','pm10_rolling_6h','pm10_lag_1h','temperature','humidity','wind_speed','wind_deg'
]

class AQIModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.sequence_length = settings.SEQUENCE_LENGTH
        self.forecast_horizon = settings.FORECAST_HORIZON

    def load_and_combine_data(self, csv_files=None):
        all_data = []
        try:
            df_collected = pd.read_csv(settings.DATA_DIR / 'historical_training_data.csv')
            df_collected['source'] = 'collected'
            all_data.append(df_collected)
            logger.info(f"Loaded {len(df_collected)} records from collected data")
        except Exception as e:
            logger.warning(f"No collected data found: {e}")
        if csv_files:
            for csv_file in csv_files:
                try:
                    df_csv = pd.read_csv(csv_file)
                    df_csv['source'] = os.path.basename(csv_file)
                    column_mapping = {
                        'PM2.5': 'pm25',
                        'PM10': 'pm10',
                        'Temperature': 'temperature',
                        'Humidity': 'humidity',
                        'City': 'city',
                        'Date': 'datetime',
                        'DateTime': 'datetime',
                        'Wind Speed': 'wind_speed',
                        'Wind Direction': 'wind_deg'
                    }
                    df_csv.rename(columns=column_mapping, inplace=True)
                    all_data.append(df_csv)
                    logger.info(f"Loaded {len(df_csv)} records from {csv_file}")
                except Exception as e:
                    logger.error(f"Error loading {csv_file}: {e}")
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            df_combined['datetime'] = pd.to_datetime(df_combined['datetime'])
            df_combined = df_combined.drop_duplicates(subset=['city', 'datetime'])
            logger.info(f"Combined data: {len(df_combined)} total records")
            return df_combined
        else:
            logger.error("No data available for training")
            return None

    def prepare_features(self, df):
        # Only keep required columns
        for col in REQUIRED_COLS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[REQUIRED_COLS]
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        # Rolling and lag features for pollutants and weather
        group_cols = ['city', 'station_id']
        for col in ['pm25', 'pm10', 'temperature', 'humidity']:
            if col in df.columns:
                df[f'{col}_rolling_6h'] = df.groupby(group_cols)[col].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
                df[f'{col}_lag_1h'] = df.groupby(group_cols)[col].shift(1)
        # Fill missing
        df = df.fillna(method='ffill').fillna(method='bfill')
        # Feature columns for model
        self.feature_cols = [
            'pm25','pm10','temperature','humidity',
            'pm25_rolling_6h','pm25_lag_1h','pm10_rolling_6h','pm10_lag_1h',
            'temperature_rolling_6h','temperature_lag_1h','humidity_rolling_6h','humidity_lag_1h',
            'hour','day_of_week','month'
        ]
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        return df

    def time_based_split(self, df, val_frac=0.2):
        # Split last 20% of data for each station for validation
        train_idx, val_idx = [], []
        for _, group in df.groupby(['city', 'station_id']):
            n = len(group)
            split = int(n * (1 - val_frac))
            idx = group.index.tolist()
            train_idx += idx[:split]
            val_idx += idx[split:]
        return df.loc[train_idx], df.loc[val_idx]

    def create_sequences(self, df):
        all_sequences, all_targets = [], []
        for (city, station_id), group in df.groupby(['city', 'station_id']):
            group = group.sort_values('datetime')
            features = group[self.feature_cols].values
            target_idx_pm25 = self.feature_cols.index('pm25') if 'pm25' in self.feature_cols else None
            target_idx_pm10 = self.feature_cols.index('pm10') if 'pm10' in self.feature_cols else None
            if target_idx_pm25 is None or target_idx_pm10 is None:
                continue
            for i in range(len(features) - self.sequence_length - self.forecast_horizon + 1):
                seq = features[i:i + self.sequence_length]
                target = features[
                    i + self.sequence_length:i + self.sequence_length + self.forecast_horizon,
                    [target_idx_pm25, target_idx_pm10]
                ]
                all_sequences.append(seq)
                all_targets.append(target)
        if not all_sequences:
            logger.error("No sequences created. Need more data!")
            return None, None
        X = np.array(all_sequences)
        y = np.array(all_targets)
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped).reshape(X.shape)
        return X_scaled, y

    def train(self, df):
        df = self.prepare_features(df)
        train_df, val_df = self.time_based_split(df)
        X_train, y_train = self.create_sequences(train_df)
        X_val, y_val = self.create_sequences(val_df)
        if X_train is None or y_train is None or X_val is None or y_val is None:
            logger.error("Training aborted: insufficient data.")
            return False
        self.model = SimpleLSTMModel(
            n_features=X_train.shape[2],
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon
        ).model
        history = self.model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        logger.info("Model training complete.")
        return history

    def save(self, model_dir=None):
        model_dir = model_dir or (settings.MODEL_DIR / 'trained')
        model_dir = os.path.abspath(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'aqi_forecast_model.h5')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        features_path = os.path.join(model_dir, 'feature_cols.pkl')
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_cols, features_path)
        logger.info(f"Model, scaler, and features saved to {model_dir}") 