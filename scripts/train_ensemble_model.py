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
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Concatenate, Attention
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Sklearn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AQIModelTrainer:
    def __init__(self):
        self.config = Config()
        self.models_dir = Path('models/trained')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model parameters - reduced for available data
        self.sequence_length = 24  # 1 day of hourly data (reduced from 48)
        self.forecast_horizon = 12  # 12 hours ahead (reduced from 24)
        self.batch_size = 32
        self.epochs = 100
        
        # Features to use
        self.pollutant_features = ['pm25', 'pm10', 'no2', 'so2', 'co', 'aqi']  # Removed 'o3' as it's all NaN
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
        """Create sequences for time series prediction"""
        sequences = []
        targets = []
        metadata = []
        
        # Group by station
        total_groups = len(df.groupby(['city', 'station']))
        sequences_created = 0
        
        for (city, station), group in df.groupby(['city', 'station']):
            group = group.sort_values('datetime')
            
            # Skip if not enough data
            if len(group) < self.sequence_length + self.forecast_horizon:
                logger.debug(f"Skipping {station}, {city}: insufficient data ({len(group)} < {self.sequence_length + self.forecast_horizon})")
                continue
            
            station_sequences = 0
            
            # Create sequences with sliding window
            for i in range(len(group) - self.sequence_length - self.forecast_horizon + 1):
                # Input sequence
                seq_data = group.iloc[i:i + self.sequence_length][self.all_features].values
                
                # Target values (next forecast_horizon hours of AQI)
                target_data = group.iloc[
                    i + self.sequence_length:i + self.sequence_length + self.forecast_horizon
                ]['aqi'].values
                
                # Much more lenient NaN filtering - allow up to 80% NaN in features
                seq_nan_ratio = np.isnan(seq_data).sum() / seq_data.size
                if seq_nan_ratio > 0.8:
                    continue
                
                # More lenient NaN filtering for targets - allow up to 80% NaN
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
        """Build LSTM model with attention"""
        inputs = Input(shape=(self.sequence_length, len(self.all_features)))
        
        # LSTM layers
        x = LSTM(128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        
        # Attention mechanism
        attention = Attention()([x, x])
        x = Concatenate()([x, attention])
        
        # Final LSTM
        x = LSTM(32)(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='huber',  # Robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_gru_model(self):
        """Build GRU model"""
        inputs = Input(shape=(self.sequence_length, len(self.all_features)))
        
        # CNN for feature extraction
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = MaxPooling1D(2)(x)
        
        # GRU layers
        x = GRU(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = GRU(64, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = GRU(32)(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        
        # Output
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_transformer_model(self):
        """Build Transformer model for time series"""
        inputs = Input(shape=(self.sequence_length, len(self.all_features)))
        
        # Positional encoding
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        position_embeddings = layers.Embedding(
            input_dim=self.sequence_length, 
            output_dim=len(self.all_features)
        )(positions)
        
        x = inputs + position_embeddings
        
        # Transformer blocks
        for _ in range(2):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=4, 
                key_dim=32, 
                dropout=0.2
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed forward
            ff_output = Dense(128, activation='relu')(x)
            ff_output = Dense(len(self.all_features))(ff_output)
            ff_output = Dropout(0.2)(ff_output)
            x = LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Output
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """Train ensemble of models"""
        logger.info("Training ensemble models...")
        
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
        logger.info("Training LSTM model...")
        lstm_model = self.build_lstm_model()
        
        lstm_checkpoint = ModelCheckpoint(
            str(self.models_dir / 'lstm_model.h5'),
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
        
        # Train GRU model
        logger.info("Training GRU model...")
        gru_model = self.build_gru_model()
        
        gru_checkpoint = ModelCheckpoint(
            str(self.models_dir / 'gru_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        
        gru_history = gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr, gru_checkpoint],
            verbose=1
        )
        
        self.models['gru'] = gru_model
        
        # Train Transformer model
        logger.info("Training Transformer model...")
        transformer_model = self.build_transformer_model()
        
        transformer_checkpoint = ModelCheckpoint(
            str(self.models_dir / 'transformer_model.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        
        transformer_history = transformer_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
                        epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[early_stop, reduce_lr, transformer_checkpoint],
            verbose=1
        )
        
        self.models['transformer'] = transformer_model
        
        # Train gradient boosting models for residual correction
        logger.info("Training gradient boosting models...")
        
        # Get ensemble predictions on training data
        train_preds = self._get_ensemble_predictions(X_train)
        train_residuals = y_train - train_preds
        
        # Flatten sequences for tree-based models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # XGBoost for residual correction
        self.models['xgb_residual'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Train separate model for each forecast hour
        xgb_models = []
        for h in range(self.forecast_horizon):
            logger.info(f"Training XGBoost for hour {h+1}/{self.forecast_horizon}")
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_flat, train_residuals[:, h])
            xgb_models.append(model)
        
        self.models['xgb_models'] = xgb_models
        
        return {
            'lstm': lstm_history,
            'gru': gru_history,
            'transformer': transformer_history
        }
    
    def _get_ensemble_predictions(self, X):
        """Get weighted ensemble predictions"""
        # Weights based on validation performance (can be optimized)
        weights = {
            'lstm': 0.35,
            'gru': 0.35,
            'transformer': 0.30
        }
        
        predictions = np.zeros((X.shape[0], self.forecast_horizon))
        
        for model_name in ['lstm', 'gru', 'transformer']:
            if model_name in self.models:
                model_pred = self.models[model_name].predict(X, batch_size=self.batch_size)
                predictions += model_pred * weights[model_name]
        
        return predictions
    
    def train(self):
        """Main training function"""
        try:
            logger.info("="*60)
            logger.info("Starting AQI Model Training")
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
            
            # Train ensemble
            start = time.time()
            histories = self.train_ensemble(
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
        logger.info("\nEvaluating models...")
        
        # Get predictions
        ensemble_pred_scaled = self._get_ensemble_predictions(X_val_scaled)
        
        # Apply residual correction
        X_val_flat = X_val_scaled.reshape(X_val_scaled.shape[0], -1)
        residual_corrections = np.zeros_like(ensemble_pred_scaled)
        
        if 'xgb_models' in self.models:
            for h, model in enumerate(self.models['xgb_models']):
                residual_corrections[:, h] = model.predict(X_val_flat)
        
        # Final predictions
        final_pred_scaled = ensemble_pred_scaled + residual_corrections
        
        # Inverse transform
        final_predictions = self.target_scaler.inverse_transform(final_pred_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val_original, final_predictions)
        rmse = np.sqrt(mean_squared_error(y_val_original, final_predictions))
        r2 = r2_score(y_val_original, final_predictions)
        
        # Calculate metrics by forecast hour
        hourly_mae = []
        for h in range(self.forecast_horizon):
            h_mae = mean_absolute_error(y_val_original[:, h], final_predictions[:, h])
            hourly_mae.append(h_mae)
        
        # Log results
        logger.info(f"\nOverall Performance:")
        logger.info(f"MAE: {mae:.2f}")
        logger.info(f"RMSE: {rmse:.2f}")
        logger.info(f"R²: {r2:.4f}")
        
        logger.info(f"\nMAE by forecast hour:")
        for h in [0, 11, 23, 47, 71]:  # Sample hours
            logger.info(f"Hour {h+1}: {hourly_mae[h]:.2f}")
        
        # Save evaluation results
        eval_results = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'hourly_mae': hourly_mae,
            'evaluation_date': datetime.now().isoformat()
        }
        
        with open(self.models_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
    
    def save_models(self):
        """Save all models and preprocessing objects"""
        logger.info("\nSaving models and scalers...")
        
        # Save scalers
        joblib.dump(self.feature_scaler, self.models_dir / 'feature_scaler.pkl')
        joblib.dump(self.target_scaler, self.models_dir / 'target_scaler.pkl')
        
        # Save XGBoost models
        if 'xgb_models' in self.models:
            for i, model in enumerate(self.models['xgb_models']):
                model.save_model(str(self.models_dir / f'xgb_hour_{i}.json'))
        
        # Save metadata
        metadata = {
            'features': self.all_features,
            'sequence_length': self.sequence_length,
            'forecast_horizon': self.forecast_horizon,
            'training_date': datetime.now().isoformat(),
            'model_versions': {
                'lstm': '1.0',
                'gru': '1.0',
                'transformer': '1.0',
                'xgboost': '1.0'
            }
        }
        
        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Models saved successfully!")


def main():
    """Main training entry point"""
    trainer = AQIModelTrainer()
    trainer.train()


if __name__ == "__main__":
    main()