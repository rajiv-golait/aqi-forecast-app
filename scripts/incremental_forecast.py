#!/usr/bin/env python3
"""
Hackathon-Ready Incremental AQI Forecast System
Optimized for demo with proper pipeline and error handling
"""

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
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HackathonIncrementalForecaster:
    def __init__(self):
        """Initialize the hackathon-ready incremental forecaster"""
        self.models_dir = Path('models/trained')
        self.data_dir = Path('data')
        self.forecast_horizon = 1  # 1 hour forecast
        self.max_forecast_hours = 72  # Maximum forecast window
        
        # Load model metadata
        self.metadata_file = self.models_dir / 'model_metadata_3day.json'
        if not self.metadata_file.exists():
            logger.error("❌ Model metadata not found. Please train models first.")
            raise FileNotFoundError("Model metadata not found")
        
        with open(self.metadata_file, 'r') as f:
            self.model_metadata = json.load(f)
        
        # Load trained models
        self._load_models()
        
    def _load_models(self):
        """Load trained models with compatibility handling and robust metadata handling"""
        try:
            # Try to load model metadata if available
            metadata_file = self.models_dir / 'model_metadata_3day.json'
            model_file = 'lstm_3day_model.h5'
            feature_scaler_file = 'feature_scaler_3day.pkl'
            target_scaler_file = 'target_scaler_3day.pkl'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.model_metadata = json.load(f)
                # Try new-style metadata
                if 'model_files' in self.model_metadata:
                    mf = self.model_metadata['model_files']
                    model_file = mf.get('model', model_file)
                    feature_scaler_file = mf.get('feature_scaler', feature_scaler_file)
                    target_scaler_file = mf.get('target_scaler', target_scaler_file)
                    logger.info(f"ℹ️ Using model files from metadata: {model_file}, {feature_scaler_file}, {target_scaler_file}")
                else:
                    # Try legacy keys or fallback
                    model_file = self.model_metadata.get('model_file', model_file)
                    feature_scaler_file = self.model_metadata.get('feature_scaler_file', feature_scaler_file)
                    target_scaler_file = self.model_metadata.get('target_scaler_file', target_scaler_file)
                    logger.info(f"ℹ️ Using legacy or default model files: {model_file}, {feature_scaler_file}, {target_scaler_file}")
            else:
                logger.info(f"ℹ️ No metadata found, using default model files: {model_file}, {feature_scaler_file}, {target_scaler_file}")
            # Custom model loading with compatibility handling
            self.model = self._load_model_with_compatibility(str(self.models_dir / model_file))
            self.feature_scaler = joblib.load(self.models_dir / feature_scaler_file)
            self.target_scaler = joblib.load(self.models_dir / target_scaler_file)
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _load_model_with_compatibility(self, model_path):
        """Load model with TensorFlow compatibility handling"""
        try:
            # First try normal loading
            return load_model(model_path)
        except ValueError as e:
            if "time_major" in str(e):
                logger.info("⚠️ Detected TensorFlow compatibility issue, using custom loading...")
                return self._load_model_custom(model_path)
            else:
                raise e
    
    def _load_model_custom(self, model_path):
        """Custom model loading for TensorFlow compatibility"""
        import tensorflow as tf
        from tensorflow import keras
        
        # Load model config
        with tf.keras.utils.custom_object_scope({}):
            try:
                # Try loading with custom objects
                model = load_model(model_path, compile=False)
                return model
            except Exception as e:
                logger.warning(f"⚠️ Custom loading failed: {e}")
                # Try loading with custom objects and custom_objects
                custom_objects = {
                    'LSTM': keras.layers.LSTM,
                    'Dense': keras.layers.Dense,
                    'Dropout': keras.layers.Dropout,
                    'Input': keras.layers.Input,
                    'Model': keras.Model
                }
                return load_model(model_path, custom_objects=custom_objects, compile=False)
    
    def load_existing_forecasts(self):
        """Load existing forecasts if available"""
        try:
            forecast_files = list(self.data_dir.glob('predictions_3day_*.json'))
            if not forecast_files:
                logger.info("ℹ️ No existing forecasts found - will create new ones")
                return None
            
            # Get the latest forecast file
            latest_forecast = max(forecast_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"📁 Loading existing forecasts from: {latest_forecast.name}")
            
            with open(latest_forecast, 'r') as f:
                existing_forecasts = json.load(f)
            
            return existing_forecasts
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading existing forecasts: {e}")
            return None
    
    def prepare_latest_data(self):
        """Prepare the most recent data for prediction"""
        logger.info("📊 Loading and preparing latest data...")
        
        try:
            data_path = Path('data/latest_aqi_weather.csv')
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path, parse_dates=['datetime'])
            logger.info(f"📈 Loaded {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
            
            # Add temporal features efficiently
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
            
            # Add engineered features efficiently
            if all(col in df.columns for col in ['temperature', 'humidity']):
                df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
            
            if 'wind_speed' in df.columns:
                df['wind_speed_squared'] = df['wind_speed'] ** 2

            # Handle missing values efficiently - only for required features
            required_features = [f for f in self.model_metadata['features'] if f in df.columns]
            for col in required_features:
                # Forward fill first
                df[col] = df.groupby(['city', 'station'])[col].fillna(method='ffill', limit=3)
                # Then fill with city average
                df[col] = df.groupby(['city', 'datetime'])[col].transform(
                    lambda x: x.fillna(x.mean())
                )
            
            logger.info("✅ Data preparation completed")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error preparing data: {e}")
            raise
    
    def create_prediction_sequences(self, df):
        """Create sequences for 1-hour prediction"""
        logger.info("🔧 Creating prediction sequences...")
        
        sequences = []
        metadata = []
        
        # Process stations efficiently - remove demo limit for production
        station_groups = df.groupby(['city', 'station'])
        total_stations = len(station_groups)
        processed_stations = 0
        
        for (city, station), group in station_groups:
            group = group.sort_values('datetime')
            
            # Skip if not enough data (need at least 48 hours for prediction)
            if len(group) < 48:
                logger.debug(f"⏭️ Skipping {station}, {city}: insufficient data ({len(group)} < 48)")
                continue
            
            # Get the most recent 48-hour sequence
            latest_sequence = group.iloc[-48:][self.model_metadata['features']].values
            
            # Check for too many NaN values
            if np.isnan(latest_sequence).sum() > latest_sequence.size * 0.8:
                logger.debug(f"⏭️ Skipping {station}, {city}: too many NaN values")
                continue
            
            # Fill NaN values
            latest_sequence = np.nan_to_num(latest_sequence, nan=0.0)
            
            sequences.append(latest_sequence)
            metadata.append({
                'city': city,
                'station': station,
                'last_datetime': group.iloc[-1]['datetime']
            })
            
            processed_stations += 1
        
        logger.info(f"✅ Created {len(sequences)} prediction sequences from {processed_stations}/{total_stations} stations")
        return np.array(sequences), metadata
    
    def predict_next_hour(self):
        """Make 1-hour AQI predictions for all stations"""
        logger.info("🚀 Starting 1-hour AQI predictions...")
        
        start_time = time.time()
        
        try:
            # Prepare data
            df = self.prepare_latest_data()
            sequences, metadata = self.create_prediction_sequences(df)
            
            if len(sequences) == 0:
                logger.error("❌ No sequences created for prediction")
                return None
            
            # Scale features efficiently
            logger.info("⚡ Scaling features...")
            sequences_scaled = self.feature_scaler.transform(
                sequences.reshape(-1, sequences.shape[-1])
            ).reshape(sequences.shape)
            
            # Make predictions with optimized settings
            logger.info("🧠 Making predictions...")
            predictions_scaled = self.model.predict(sequences_scaled, batch_size=128, verbose=0)
            
            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(predictions_scaled)
            
            # Create results efficiently
            results = []
            for i, (pred, meta) in enumerate(zip(predictions, metadata)):
                # Create forecast timestamp (next hour)
                forecast_time = meta['last_datetime'] + timedelta(hours=1)
                
                # Compute station_id
                station_id = f"{meta['city']}_{meta['station']}".replace(" ", "_").replace("-", "_").replace(",", "_")
                
                # Create result for this station
                station_result = {
                    'station_id': station_id,
                    'city': meta['city'],
                    'station': meta['station'],
                    'last_known_time': meta['last_datetime'].isoformat(),
                    'forecast_time': forecast_time.isoformat(),
                    'aqi_predicted': float(pred[0]),  # Only 1-hour prediction
                    'aqi_category': self._get_aqi_category(pred[0])
                }
                
                results.append(station_result)
            
            end_time = time.time()
            logger.info(f"✅ Generated {len(results)} 1-hour forecasts in {end_time - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in prediction: {e}")
            raise
    
    def _get_aqi_category(self, aqi_value):
        """Convert AQI value to category"""
        if aqi_value <= 50:
            return "Good"
        elif aqi_value <= 100:
            return "Moderate"
        elif aqi_value <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi_value <= 200:
            return "Unhealthy"
        elif aqi_value <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def update_forecasts(self, new_predictions, existing_forecasts=None):
        """Update existing forecasts with new 1-hour predictions"""
        logger.info("🔄 Updating forecasts with new predictions...")
        
        current_time = datetime.now()
        
        if existing_forecasts is None:
            # Create new forecast structure
            logger.info("🆕 Creating new forecast structure...")
            updated_forecasts = []
            for pred in new_predictions:
                station_forecast = {
                    'station_id': pred['station_id'],
                    'city': pred['city'],
                    'station': pred['station'],
                    'last_known_time': pred['last_known_time'],
                    'forecasts': []
                }
                
                # Add the new 1-hour forecast
                forecast_point = {
                    'timestamp': pred['forecast_time'],
                    'hour_ahead': 1,
                    'aqi_predicted': pred['aqi_predicted'],
                    'aqi_category': pred['aqi_category']
                }
                station_forecast['forecasts'].append(forecast_point)
                
                updated_forecasts.append(station_forecast)
        else:
            # Update existing forecasts
            logger.info("📝 Updating existing forecasts...")
            updated_forecasts = []
            
            for pred in new_predictions:
                # Find existing forecast for this station
                existing_station = None
                for station in existing_forecasts:
                    if station['station_id'] == pred['station_id']:
                        existing_station = station
                        break
                
                if existing_station:
                    # Update existing station forecast
                    station_forecast = existing_station.copy()
                    
                    # Remove forecasts that are now in the past
                    current_forecasts = []
                    for forecast in station_forecast['forecasts']:
                        forecast_time = datetime.fromisoformat(forecast['timestamp'])
                        if forecast_time > current_time:
                            current_forecasts.append(forecast)
                    
                    # Add new 1-hour forecast
                    new_forecast = {
                        'timestamp': pred['forecast_time'],
                        'hour_ahead': len(current_forecasts) + 1,
                        'aqi_predicted': pred['aqi_predicted'],
                        'aqi_category': pred['aqi_category']
                    }
                    current_forecasts.append(new_forecast)
                    
                    # Update hour_ahead values
                    for i, forecast in enumerate(current_forecasts):
                        forecast['hour_ahead'] = i + 1
                    
                    station_forecast['forecasts'] = current_forecasts
                    station_forecast['last_known_time'] = pred['last_known_time']
                else:
                    # New station, create new forecast
                    station_forecast = {
                        'station_id': pred['station_id'],
                        'city': pred['city'],
                        'station': pred['station'],
                        'last_known_time': pred['last_known_time'],
                        'forecasts': [{
                            'timestamp': pred['forecast_time'],
                            'hour_ahead': 1,
                            'aqi_predicted': pred['aqi_predicted'],
                            'aqi_category': pred['aqi_category']
                        }]
                    }
                
                updated_forecasts.append(station_forecast)
        
        logger.info(f"✅ Updated forecasts for {len(updated_forecasts)} stations")
        return updated_forecasts
    
    def save_updated_forecasts(self, updated_forecasts):
        """Save updated forecasts to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed predictions
            output_file = f'data/predictions_3day_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(updated_forecasts, f, indent=2)
            
            # Create summary CSV
            summary_data = []
            for station in updated_forecasts:
                for forecast in station['forecasts']:
                    summary_data.append({
                        'city': station['city'],
                        'station': station['station'],
                        'timestamp': forecast['timestamp'],
                        'hour_ahead': forecast['hour_ahead'],
                        'aqi_predicted': forecast['aqi_predicted'],
                        'aqi_category': forecast['aqi_category']
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_file = f'data/predictions_3day_summary_{timestamp}.csv'
            summary_df.to_csv(summary_file, index=False)
            
            logger.info(f"💾 Forecasts saved to {output_file}")
            logger.info(f"📊 Summary saved to {summary_file}")
            
            return output_file, summary_file
            
        except Exception as e:
            logger.error(f"❌ Error saving forecasts: {e}")
            raise
    
    def print_summary(self, updated_forecasts):
        """Print a summary of updated forecasts"""
        logger.info("\n" + "="*60)
        logger.info("🎯 HACKATHON FORECAST SUMMARY")
        logger.info("="*60)
        
        total_stations = len(updated_forecasts)
        total_forecasts = sum(len(station['forecasts']) for station in updated_forecasts)
        
        logger.info(f"📊 Total stations: {total_stations}")
        logger.info(f"⏰ Total forecast points: {total_forecasts}")
        
        # Show forecast range
        all_timestamps = []
        for station in updated_forecasts:
            for forecast in station['forecasts']:
                all_timestamps.append(datetime.fromisoformat(forecast['timestamp']))
        
        if all_timestamps:
            min_time = min(all_timestamps)
            max_time = max(all_timestamps)
            forecast_hours = (max_time - min_time).total_seconds() / 3600
            
            logger.info(f"📅 Forecast range: {min_time} to {max_time}")
            logger.info(f"⏱️ Forecast window: {forecast_hours:.1f} hours")
        
        # Show sample predictions
        logger.info("\n🔍 Sample updated forecasts (first 3 stations):")
        for i, station in enumerate(updated_forecasts[:3]):
            logger.info(f"\n📍 {station['station']}, {station['city']}:")
            for forecast in station['forecasts'][:5]:  # Show first 5 forecasts
                logger.info(f"  Hour {forecast['hour_ahead']}: AQI {forecast['aqi_predicted']:.1f} ({forecast['aqi_category']})")


def main():
    """Main incremental forecast function for hackathon"""
    try:
        logger.info("🚀 Starting Hackathon Incremental AQI Forecasting System")
        logger.info("="*60)

        # --- NEW: Check if 72-hour forecast window is already full ---
        from glob import glob
        latest_data_path = 'data/latest_aqi_weather.csv'
        forecast_files = sorted(glob('data/predictions_3day_*.json'), reverse=True)
        if os.path.exists(latest_data_path) and forecast_files:
            import pandas as pd
            from datetime import datetime
            # Get latest data timestamp
            df = pd.read_csv(latest_data_path, parse_dates=['datetime'])
            latest_data_time = df['datetime'].max()
            # Get latest forecast timestamp
            with open(forecast_files[0], 'r') as f:
                forecasts = json.load(f)
            latest_forecast_time = None
            for station in forecasts:
                if station['forecasts']:
                    t = station['forecasts'][-1]['timestamp']
                    t = datetime.fromisoformat(t)
                    if latest_forecast_time is None or t > latest_forecast_time:
                        latest_forecast_time = t
            if latest_forecast_time is not None:
                hours_ahead = (latest_forecast_time - latest_data_time).total_seconds() / 3600
                if hours_ahead >= 72:
                    logger.info(f"⏩ Forecast window is already full ({hours_ahead:.1f}h ahead). Skipping incremental update.")
                    return
        # --- END NEW ---

        start_time = time.time()
        # Initialize forecaster
        forecaster = HackathonIncrementalForecaster()
        # Load existing forecasts
        existing_forecasts = forecaster.load_existing_forecasts()
        # Generate new 1-hour predictions
        new_predictions = forecaster.predict_next_hour()
        if new_predictions:
            # Update forecasts with new predictions
            updated_forecasts = forecaster.update_forecasts(new_predictions, existing_forecasts)
            # Save and display results
            forecaster.save_updated_forecasts(updated_forecasts)
            forecaster.print_summary(updated_forecasts)
            end_time = time.time()
            total_time = end_time - start_time
            logger.info("="*60)
            logger.info(f"🎉 HACKATHON FORECASTING COMPLETED!")
            logger.info(f"⏱️ Total time: {total_time:.2f} seconds")
            logger.info(f"📈 Performance: {len(new_predictions)} stations processed")
            logger.info(f"🚀 Ready for demo!")
            logger.info("="*60)
        else:
            logger.error("❌ No predictions generated")
    except Exception as e:
        logger.error(f"💥 Hackathon forecast failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 