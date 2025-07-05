import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import logging
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Sklearn imports
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_3day_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AQI3DayForecastTrainer:
    def __init__(self):
        self.config = Config()
        self.models_dir = Path('models/trained')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model parameters optimized for 3-day forecasting
        self.sequence_length = 48  # 2 days of hourly data for context
        self.forecast_horizon = 72  # 3 days ahead
        self.batch_size = 64  # Larger batch for efficiency
        self.epochs = 50  # Reduced epochs
        
        # Features to use (optimized for forecasting)
        self.pollutant_features = ['pm25', 'pm10', 'no2', 'so2', 'co', 'aqi']
        self.weather_features = ['temperature', 'humidity', 'wind_speed', 'wind_deg', 'pressure']
        self.temporal_features = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour']
        self.engineered_features = ['temp_humidity_interaction', 'wind_speed_squared']
        
        self.all_features = (self.pollutant_features + self.weather_features + 
                           self.temporal_features + self.engineered_features)
        
        # Scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Models
        self.models = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for training"""
        logger.info("Loading and preparing data...")
        
        try:
            # Load data
            data_path = Path('data/latest_aqi_weather.csv')
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path, parse_dates=['datetime'])
            logger.info(f"Loaded {len(df)} records")
            
            # Sort by datetime
            df = df.sort_values(['city', 'station', 'datetime'])
            
            # Add temporal features if not present
            if 'hour' not in df.columns:
                df['hour'] = df['datetime'].dt.hour
            if 'day_of_week' not in df.columns:
                df['day_of_week'] = df['datetime'].dt.dayofweek
            if 'is_weekend' not in df.columns:
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            if 'is_rush_hour' not in df.columns:
                df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
            
            # Add engineered features if not present
            if 'temp_humidity_interaction' not in df.columns and all(col in df.columns for col in ['temperature', 'humidity']):
                df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
            
            if 'wind_speed_squared' not in df.columns and 'wind_speed' in df.columns:
                df['wind_speed_squared'] = df['wind_speed'] ** 2
            
            # Handle missing features
            available_features = [f for f in self.all_features if f in df.columns]
            logger.info(f"Available features: {len(available_features)}/{len(self.all_features)}")
            logger.info(f"Available features: {available_features}")
            
            # Update feature list
            self.all_features = available_features
            
            # Fill missing values
            df = self._handle_missing_values(df)
            
            # Create sequences
            sequences, targets, metadata = self._create_sequences(df)
            
            logger.info(f"Created {len(sequences)} sequences")
            
            return sequences, targets, metadata
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _handle_missing_values(self, df):
        """Handle missing values intelligently"""
        # Forward fill for short gaps (up to 3 hours)
        for col in self.all_features:
            if col in df.columns:
                df[col] = df.groupby(['city', 'station'])[col].fillna(method='ffill', limit=3)
        
        # Interpolate for weather features
        weather_cols = [col for col in self.weather_features if col in df.columns]
        if weather_cols:
            df[weather_cols] = df.groupby(['city', 'station'])[weather_cols].transform(lambda group: group.interpolate(method='linear'))
        
        # Use city average for remaining missing values
        for col in self.all_features:
            if col in df.columns:
                df[col] = df.groupby(['city', 'datetime'])[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        return df
    
    def _create_sequences(self, df):
        """Create sequences for 3-day forecasting"""
        sequences = []
        targets = []
        metadata = []
        
        # Group by station
        total_groups = len(df.groupby(['city', 'station']))
        sequences_created = 0
        
        for (city, station), group in df.groupby(['city', 'station']):
            group = group.sort_values('datetime')
            
            # Skip if not enough data (need 48 + 72 = 120 hours minimum)
            if len(group) < self.sequence_length + self.forecast_horizon:
                logger.debug(f"Skipping {station}, {city}: insufficient data ({len(group)} < {self.sequence_length + self.forecast_horizon})")
                continue
            
            station_sequences = 0
            
            # Create sequences with sliding window (step by 12 hours for efficiency)
            for i in range(0, len(group) - self.sequence_length - self.forecast_horizon + 1, 12):
                # Input sequence
                seq_data = group.iloc[i:i + self.sequence_length][self.all_features].values
                
                # Target values (next 72 hours of AQI)
                target_data = group.iloc[
                    i + self.sequence_length:i + self.sequence_length + self.forecast_horizon
                ]['aqi'].values
                
                # Lenient NaN filtering - allow up to 80% NaN in features
                seq_nan_ratio = np.isnan(seq_data).sum() / seq_data.size
                if seq_nan_ratio > 0.8:
                    continue
                
                # Lenient NaN filtering for targets - allow up to 80% NaN
                target_nan_ratio = np.isnan(target_data).sum() / len(target_data)
                if target_nan_ratio > 0.8:
                    continue
                
                # Fill remaining NaN with 0 (after checks)
                seq_data = np.nan_to_num(seq_data, nan=0.0)
                target_data = np.nan_to_num(target_data, nan=0.0)
                
                sequences.append(seq_data)
                targets.append(target_data)
                metadata.append({
                    'city': city,
                    'station': station,
                    'start_time': group.iloc[i]['datetime'],
                    'end_time': group.iloc[i + self.sequence_length + self.forecast_horizon - 1]['datetime']
                })
                station_sequences += 1
            
            sequences_created += station_sequences
            if station_sequences > 0:
                logger.debug(f"Created {station_sequences} sequences for {station}, {city}")
        
        logger.info(f"Created {len(sequences)} sequences from {total_groups} groups (avg: {len(sequences)/total_groups:.1f} per group)")
        return np.array(sequences), np.array(targets), metadata
    
    def build_lstm_model(self):
        """Build LSTM model optimized for 3-day forecasting"""
        inputs = Input(shape=(self.sequence_length, len(self.all_features)))
        
        # LSTM layers with more capacity for longer sequences
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer for 72-hour forecast
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the 3-day forecasting model"""
        logger.info("Training 3-day forecasting model...")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train LSTM model
        logger.info("Training LSTM model for 3-day forecasting...")
        lstm_model = self.build_lstm_model()
        
        lstm_checkpoint = ModelCheckpoint(
            str(self.models_dir / 'lstm_3day_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        
        lstm_history = lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr, lstm_checkpoint],
            verbose=1
        )
        
        self.models['lstm'] = lstm_model
        
        return {
            'lstm': lstm_history
        }
    
    def train(self):
        """Main training function"""
        try:
            logger.info("="*60)
            logger.info("Starting AQI 3-Day Forecasting Model Training")
            logger.info("="*60)
            
            start_total = time.time()
            
            # Load and prepare data
            start = time.time()
            sequences, targets, metadata = self.load_and_prepare_data()
            logger.info(f"Data loading and preparation took {time.time() - start:.2f} seconds")
            
            if len(sequences) == 0:
                logger.error("No sequences created. Check data availability.")
                return
            
            # Split data (time-based split)
            split_idx = int(len(sequences) * 0.8)
            X_train, X_val = sequences[:split_idx], sequences[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            logger.info(f"Training samples: {len(X_train)}")
            logger.info(f"Validation samples: {len(X_val)}")
            
            # Scale features
            start = time.time()
            self.feature_scaler = RobustScaler()
            X_train_scaled = self.feature_scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            X_val_scaled = self.feature_scaler.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            logger.info(f"Feature scaling took {time.time() - start:.2f} seconds")
            
            # Scale targets
            start = time.time()
            self.target_scaler = RobustScaler()
            y_train_scaled = self.target_scaler.fit_transform(y_train)
            y_val_scaled = self.target_scaler.transform(y_val)
            logger.info(f"Target scaling took {time.time() - start:.2f} seconds")
            
            # Train model
            start = time.time()
            histories = self.train_model(
                X_train_scaled, y_train_scaled,
                X_val_scaled, y_val_scaled
            )
            logger.info(f"Model training took {time.time() - start:.2f} seconds")
            
            # Evaluate on validation set
            start = time.time()
            self.evaluate(X_val_scaled, y_val_scaled, y_val)
            logger.info(f"Evaluation took {time.time() - start:.2f} seconds")
            
            # Save models and scalers
            start = time.time()
            self.save_models()
            logger.info(f"Saving models took {time.time() - start:.2f} seconds")
            
            logger.info(f"Training completed successfully! Total time: {time.time() - start_total:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def evaluate(self, X_val_scaled, y_val_scaled, y_val_original):
        """Evaluate model performance"""
        logger.info("\nEvaluating 3-day forecasting model...")
        
        # Get predictions
        lstm_pred_scaled = self.models['lstm'].predict(X_val_scaled, batch_size=self.batch_size)
        
        # Inverse transform
        lstm_predictions = self.target_scaler.inverse_transform(lstm_pred_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val_original, lstm_predictions)
        rmse = np.sqrt(mean_squared_error(y_val_original, lstm_predictions))
        r2 = r2_score(y_val_original, lstm_predictions)
        
        # Calculate metrics by forecast hour (every 24 hours)
        hourly_mae = []
        for h in range(self.forecast_horizon):
            h_mae = mean_absolute_error(y_val_original[:, h], lstm_predictions[:, h])
            hourly_mae.append(h_mae)
        
        # Log results
        logger.info(f"\nOverall Performance:")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"R²: {r2:.4f}")
        
        logger.info(f"\nMAE by forecast day:")
        for day in range(3):
            start_hour = day * 24
            end_hour = (day + 1) * 24
            day_mae = np.mean(hourly_mae[start_hour:end_hour])
            logger.info(f"Day {day+1}: {day_mae:.2f}")
        
        # Save evaluation results
        eval_results = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'hourly_mae': hourly_mae,
            'forecast_horizon': self.forecast_horizon,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(self.models_dir / 'evaluation_results_3day.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
    
    def save_models(self):
        """Save all models and preprocessing objects"""
        logger.info("\nSaving models and scalers...")
        
        # Save scalers
        joblib.dump(self.feature_scaler, self.models_dir / 'feature_scaler_3day.pkl')
        joblib.dump(self.target_scaler, self.models_dir / 'target_scaler_3day.pkl')
        
        # Save metadata
        metadata = {
            'features': self.all_features,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'lstm_3day': '1.0'
            }
        }
        
        with open(self.models_dir / 'model_metadata_3day.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Models saved successfully!")


def main():
    """Main training entry point"""
    trainer = AQI3DayForecastTrainer()
    trainer.train()


if __name__ == "__main__":
    main() 