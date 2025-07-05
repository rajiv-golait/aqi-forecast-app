# 🌬️ AQI Forecast App - Air Quality Index Prediction System

A comprehensive machine learning system for predicting Air Quality Index (AQI) across India using real-time data from government APIs and weather information.

## 📊 Project Overview

This system provides **3-day AQI forecasts** for **356+ monitoring stations** across India, helping citizens and authorities make informed decisions about air quality.

### 🎯 Key Features

- **Real-time Data Collection**: Fetches AQI data from India government APIs
- **Weather Integration**: Combines weather data from OpenWeatherMap API
- **Advanced ML Models**: Ensemble of LSTM, GRU, and Transformer models
- **3-Day Forecasting**: Predicts AQI values up to 72 hours ahead
- **Comprehensive Coverage**: 356+ stations across all major Indian cities
- **Automated Pipeline**: End-to-end data collection, training, and prediction

## 🏗️ System Architecture

```
📁 aqi-forecast-app/
├── 📁 api/                    # FastAPI backend
├── 📁 automation/             # Cron jobs and automation
├── 📁 BIN/                   # Utility scripts
├── 📁 config/                # Configuration files
├── 📁 data/                  # Data storage
│   ├── latest_aqi_weather.csv    # Latest merged data
│   ├── predictions_3day_*.json   # Forecast results
│   └── merged_coords.csv         # Station coordinates
├── 📁 logs/                  # Training and system logs
├── 📁 models/                # Trained ML models
├── 📁 scripts/               # Main pipeline scripts
├── 📁 src/                   # Source code modules
└── 📁 tests/                 # Unit tests
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 1. Data Collection

```bash
# Fetch latest AQI and weather data
python scripts/fetch_and_merge_all_data.py
```

### 2. Model Training

```bash
# Train the ensemble model
python scripts/train_ensemble_model.py
```

### 3. Generate Forecasts

```bash
# Generate 3-day AQI predictions
python scripts/predict_3day_aqi.py
```

### 4. View Results

```bash
# Analyze prediction results
python analyze_predictions.py
```

## 📈 Current Results

### Latest Forecast Summary (July 6, 2025)
- **Total Stations**: 356 monitoring stations
- **Predictions Generated**: 25,632 individual AQI predictions
- **Forecast Period**: 3 days (72 hours) ahead
- **Coverage**: All major Indian cities

### Sample Predictions
```
Station: Bardowali, Agartala - Tripura SPCB
Hour 1: AQI = 119.65 (Unhealthy for Sensitive Groups)
Hour 2: AQI = 119.39 (Unhealthy for Sensitive Groups)
Hour 3: AQI = 116.79 (Unhealthy for Sensitive Groups)
```

## 🔧 Core Components

### 1. Data Pipeline (`scripts/fetch_and_merge_all_data.py`)

**Features:**
- Fetches AQI data from India government API
- Integrates weather data from OpenWeatherMap
- Handles missing coordinates with geocoding
- Merges and cleans data for ML training

**Usage:**
```bash
# Run full pipeline
python scripts/fetch_and_merge_all_data.py

# Debug AQI data
python scripts/fetch_and_merge_all_data.py debug_aqi

# Merge coordinates
python scripts/fetch_and_merge_all_data.py merge_coords
```

### 2. Model Training (`scripts/train_ensemble_model.py`)

**Models:**
- **LSTM with Attention**: Captures temporal dependencies
- **GRU with CNN**: Feature extraction + sequence modeling
- **Transformer**: Advanced attention mechanisms
- **XGBoost**: Residual correction for ensemble

**Features:**
- 17 input features (pollutants, weather, temporal)
- 24-hour input sequence
- 12-hour forecast horizon
- Robust scaling and missing value handling

### 3. Prediction System (`scripts/predict_3day_aqi.py`)

**Capabilities:**
- Loads trained ensemble models
- Generates 72-hour forecasts for all stations
- Outputs JSON and CSV formats
- Includes AQI categories and confidence scores

## 📊 Data Sources

### AQI Data
- **Source**: India Government API (data.gov.in)
- **Coverage**: 423+ monitoring stations
- **Pollutants**: PM2.5, PM10, NO2, SO2, CO, O3
- **Frequency**: Hourly updates

### Weather Data
- **Source**: OpenWeatherMap API
- **Parameters**: Temperature, humidity, wind speed, pressure
- **Coverage**: All station locations
- **Frequency**: Real-time updates

## 🎯 Model Performance

### Training Metrics
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination
- **Hourly Performance**: Individual forecast hour accuracy

### Model Architecture
```
Input: 24 hours × 17 features
├── LSTM (128 → 64 → 32) + Attention
├── GRU (128 → 64 → 32) + CNN
├── Transformer (4 heads, 2 layers)
└── XGBoost (residual correction)
Output: 12-hour AQI predictions
```

## 📁 File Structure

### Key Files
```
data/
├── latest_aqi_weather.csv          # Latest merged data
├── predictions_3day_*.json         # Detailed predictions
├── predictions_3day_summary_*.csv  # Summary results
├── merged_coords.csv              # Station coordinates
└── valid_locations.json           # Valid station list

models/trained/
├── lstm_model.h5                  # Trained LSTM model
├── gru_model.h5                   # Trained GRU model
├── transformer_model.h5           # Trained Transformer
├── feature_scaler.pkl             # Feature scaler
├── target_scaler.pkl              # Target scaler
└── evaluation_results.json        # Model performance
```

## 🔄 Automation

### Scheduled Tasks
```bash
# Hourly data collection
0 * * * * python scripts/fetch_and_merge_all_data.py

# Daily model retraining (if needed)
0 2 * * * python scripts/train_ensemble_model.py

# Daily forecast generation
0 6 * * * python scripts/predict_3day_aqi.py
```

## 🛠️ Configuration

### Environment Variables
```bash
# Weather API
OPENWEATHER_API_KEY=your_api_key_here

# AQI API
INDIA_GOV_API_KEY=your_api_key_here

# Model parameters
SEQUENCE_LENGTH=24
FORECAST_HORIZON=12
BATCH_SIZE=32
```

### Configuration Files
- `config/config.py`: Main configuration
- `config/logging_config.py`: Logging setup
- `config/gee-credentials.json`: Google Earth Engine (if used)

## 📊 Output Formats

### JSON Predictions
```json
[
  {
    "station": "Station Name",
    "city": "City Name",
    "state": "State Name",
    "predictions": [
      {
        "timestamp": "2025-07-06T01:00:00",
        "aqi": 119.65,
        "category": "Unhealthy for Sensitive Groups"
      }
    ]
  }
]
```

### CSV Summary
```csv
city,station,timestamp,hour_ahead,aqi_predicted,aqi_category
Agartala,Bardowali Station,2025-07-06T01:00:00,1,119.65,Unhealthy for Sensitive Groups
```

## 🚨 AQI Categories

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0-50 | Good | Minimal impact |
| 51-100 | Moderate | Acceptable |
| 101-150 | Unhealthy for Sensitive Groups | Caution for sensitive people |
| 151-200 | Unhealthy | Everyone may experience effects |
| 201-300 | Very Unhealthy | Health warnings |
| 301+ | Hazardous | Emergency conditions |

## 🔍 Monitoring & Logs

### Log Files
```
logs/
├── training_YYYYMMDD_HHMMSS.log    # Training logs
├── fetch_YYYYMMDD_HHMMSS.log       # Data collection logs
└── predict_YYYYMMDD_HHMMSS.log     # Prediction logs
```

### Key Metrics
- Data collection success rate
- Model training performance
- Prediction accuracy
- System uptime and errors

## 🛡️ Error Handling

### Data Collection
- API rate limiting
- Network timeouts
- Missing data handling
- Coordinate validation

### Model Training
- Memory management
- Early stopping
- Gradient clipping
- Validation monitoring

### Predictions
- Model loading errors
- Data preprocessing
- Output validation
- File I/O errors

## 🔧 Troubleshooting

### Common Issues

**1. API Rate Limits**
```bash
# Check API status
python scripts/fetch_and_merge_all_data.py debug_aqi
```

**2. Memory Issues**
```bash
# Reduce batch size in config
BATCH_SIZE=16
```

**3. Missing Coordinates**
```bash
# Regenerate coordinates
python scripts/fetch_and_merge_all_data.py merge_coords
```

**4. Model Loading Errors**
```bash
# Retrain models
python scripts/train_ensemble_model.py
```

## 📈 Future Enhancements

### Planned Features
- [ ] Web dashboard with real-time maps
- [ ] Email/SMS alerts for high AQI
- [ ] Mobile app integration
- [ ] Historical trend analysis
- [ ] Satellite data integration
- [ ] Multi-city comparison tools

### Technical Improvements
- [ ] GPU acceleration for training
- [ ] Real-time streaming predictions
- [ ] Advanced ensemble methods
- [ ] Automated hyperparameter tuning
- [ ] Model versioning and A/B testing

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd aqi-forecast-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Add type hints
- Include docstrings
- Write unit tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Data Sources**: India Government API, OpenWeatherMap
- **ML Frameworks**: TensorFlow, scikit-learn, XGBoost
- **Geospatial**: Google Earth Engine (optional)
- **Monitoring**: Custom logging and alerting system

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation

---

**Last Updated**: July 6, 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
