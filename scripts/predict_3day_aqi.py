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
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQI3DayPredictor:
    def __init__(self):
        """Initialize the predictor with trained models."""
        self.models_dir = Path('models/trained')
        self.data_dir = Path('data')
        self.forecast_horizon = 72  # 3 days
        self.sequence_length = 48   # Match trained model
        
        # Load model metadata (create if missing)
        self.metadata_file = self.models_dir / 'model_metadata_3day.json'
        if not self.metadata_file.exists():
            logger.warning("model_metadata_3day.json not found, creating default metadata...")
            self._create_default_metadata()
        
        with open(self.metadata_file, 'r') as f:
            self.model_metadata = json.load(f)
        
        # Load trained models
        self._load_models()
        
    def _create_default_metadata(self):
        """Create default model metadata if missing."""
        default_metadata = {
            "model_version": "3day_ensemble_v1.0",
            "training_date": datetime.now().isoformat(),
            "features": ["pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed"],
            "target": "aqi",
            "forecast_horizon": 72,
            "model_files": {
                "lstm": "lstm_3day_model.h5",
                "feature_scaler": "feature_scaler_3day.pkl",
                "target_scaler": "target_scaler_3day.pkl"
            }
        }
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(default_metadata, f, indent=2)
        
        logger.info(f"Created default model metadata: {self.metadata_file}")
    
    def _load_models(self):
        """Load models and scalers"""
        # Handle both old and new metadata formats
        if 'model_files' in self.model_metadata:
            # Old format with model_files key
            model_file = self.model_metadata['model_files']['lstm']
            feature_scaler_file = self.model_metadata['model_files']['feature_scaler']
            target_scaler_file = self.model_metadata['model_files']['target_scaler']
        else:
            # New format - use default file names
            model_file = 'lstm_3day_model.h5'
            feature_scaler_file = 'feature_scaler_3day.pkl'
            target_scaler_file = 'target_scaler_3day.pkl'
        
        self.model = load_model(str(self.models_dir / model_file))
        self.feature_scaler = joblib.load(self.models_dir / feature_scaler_file)
        self.target_scaler = joblib.load(self.models_dir / target_scaler_file)
        
        logger.info(f"Loaded model for {self.forecast_horizon}-hour forecasting")
        logger.info(f"Features: {len(self.model_metadata['features'])}")
    
    def prepare_latest_data(self):
        """Prepare the most recent data for prediction"""
        logger.info("Loading latest data...")
        
        # Load latest data
        data_path = Path('data/latest_aqi_weather.csv')
        df = pd.read_csv(data_path, parse_dates=['datetime'])
        
        # Add temporal features
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['datetime'].dt.dayofweek
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        if 'is_rush_hour' not in df.columns:
            df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Add engineered features
        if 'temp_humidity_interaction' not in df.columns and all(col in df.columns for col in ['temperature', 'humidity']):
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        
        if 'wind_speed_squared' not in df.columns and 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
        
        # Handle missing values
        for col in self.model_metadata['features']:
            if col in df.columns:
                df[col] = df.groupby(['city', 'station'])[col].fillna(method='ffill', limit=3)
                df[col] = df.groupby(['city', 'datetime'])[col].transform(
                    lambda x: x.fillna(x.mean())
                )
        
        return df
    
    def create_prediction_sequences(self, df):
        """Create sequences for prediction from latest data"""
        sequences = []
        metadata = []
        for (city, station), group in df.groupby(['city', 'station']):
            group = group.sort_values('datetime')
            # Skip if not enough data
            if len(group) < self.sequence_length:
                logger.debug(f"Skipping {station}, {city}: insufficient data")
                continue
            # Get the most recent sequence
            latest_sequence = group.iloc[-self.sequence_length:][self.model_metadata['features']].values
            # Check for too many NaN values
            if np.isnan(latest_sequence).sum() > latest_sequence.size * 0.8:
                logger.debug(f"Skipping {station}, {city}: too many NaN values")
                continue
            # Fill NaN values
            latest_sequence = np.nan_to_num(latest_sequence, nan=0.0)
            sequences.append(latest_sequence)
            metadata.append({
                'city': city,
                'station': station,
                'last_datetime': group.iloc[-1]['datetime']
            })
        logger.info(f"Created {len(sequences)} prediction sequences")
        return np.array(sequences), metadata
    
    def predict_3day_aqi(self):
        """Make 3-day AQI predictions for all stations"""
        logger.info("Starting 3-day AQI predictions...")
        
        # Prepare data
        df = self.prepare_latest_data()
        sequences, metadata = self.create_prediction_sequences(df)
        
        if len(sequences) == 0:
            logger.error("No sequences created for prediction")
            return None
        
        # Scale features
        sequences_scaled = self.feature_scaler.transform(
            sequences.reshape(-1, sequences.shape[-1])
        ).reshape(sequences.shape)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions_scaled = self.model.predict(sequences_scaled, batch_size=64)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        # Create results
        results = []
        for i, (pred, meta) in enumerate(zip(predictions, metadata)):
            # Create forecast timestamps
            start_time = meta['last_datetime'] + timedelta(hours=1)
            forecast_times = [start_time + timedelta(hours=h) for h in range(self.forecast_horizon)]

            # Compute station_id as in API (matching station_service.py logic)
            station_id = f"{meta['city']}_{meta['station']}".replace(" ", "_").replace("-", "_").replace(",", "_")

            # Create result for this station
            station_result = {
                'station_id': station_id,
                'city': meta['city'],
                'station': meta['station'],
                'last_known_time': meta['last_datetime'].isoformat(),
                'forecasts': []
            }

            # Add hourly forecasts
            for j, (timestamp, aqi_pred) in enumerate(zip(forecast_times, pred)):
                # Determine AQI category
                aqi_category = self._get_aqi_category(aqi_pred)
                
                forecast_point = {
                    'timestamp': timestamp.isoformat(),
                    'hour_ahead': j + 1,
                    'aqi_predicted': float(aqi_pred),
                    'aqi_category': aqi_category
                }
                station_result['forecasts'].append(forecast_point)
            
            results.append(station_result)
        
        logger.info(f"Generated {len(results)} station forecasts")
        return results
    
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
    
    def save_predictions(self, predictions):
        """Save predictions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed predictions
        output_file = f'data/predictions_3day_{timestamp}.json'
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for station in predictions:
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
        
        logger.info(f"Predictions saved to {output_file}")
        logger.info(f"Summary saved to {summary_file}")
        
        return output_file, summary_file
    
    def print_summary(self, predictions):
        """Print a summary of predictions"""
        logger.info("\n" + "="*60)
        logger.info("3-DAY AQI FORECAST SUMMARY")
        logger.info("="*60)
        
        total_stations = len(predictions)
        logger.info(f"Total stations forecasted: {total_stations}")
        
        # Calculate average AQI for each day
        for day in range(3):
            day_start = day * 24
            day_end = (day + 1) * 24
            
            day_aqis = []
            for station in predictions:
                day_forecasts = station['forecasts'][day_start:day_end]
                day_aqis.extend([f['aqi_predicted'] for f in day_forecasts])
            
            avg_aqi = np.mean(day_aqis)
            max_aqi = np.max(day_aqis)
            min_aqi = np.min(day_aqis)
            
            logger.info(f"Day {day+1} - Avg AQI: {avg_aqi:.1f}, Range: {min_aqi:.1f}-{max_aqi:.1f}")
        
        # Show some sample predictions
        logger.info("\nSample predictions (first 3 stations):")
        for i, station in enumerate(predictions[:3]):
            logger.info(f"\n{station['station']}, {station['city']}:")
            for hour in [0, 23, 47, 71]:  # Show 1st, 24th, 48th, 72nd hour
                if hour < len(station['forecasts']):
                    forecast = station['forecasts'][hour]
                    logger.info(f"  Hour {forecast['hour_ahead']}: AQI {forecast['aqi_predicted']:.1f} ({forecast['aqi_category']})")


def main():
    """Main prediction function"""
    try:
        predictor = AQI3DayPredictor()
        predictions = predictor.predict_3day_aqi()
        
        if predictions:
            predictor.print_summary(predictions)
            predictor.save_predictions(predictions)
            logger.info("3-day AQI forecasting completed successfully!")
        else:
            logger.error("No predictions generated")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 