# 🌬️ AQI Forecast App

A production-ready machine learning system for forecasting Air Quality Index (AQI) across India, using real-time government and weather data, advanced ML models, and a robust API backend.

---

## 🚀 Hackathon Context
This project is designed for a hackathon challenge: **Developing an Algorithm for Air Quality Visualizer and Forecast App to Generate Granular, Real-time, and Predictive Air Quality Information**.

- **Goal:** Deliver hyperlocal, real-time, and predictive AQI insights for both large cities and underserved regions.
- **Features:**
  - Real-time AQI from ground stations
  - Historical AQI trends
  - 24–72h AQI forecasting using meteorological data
  - Health recommendations
  - (Optional) Push alerts, pollution source mapping

---

## 🛠️ Pipeline Workflow

1. **Data Fetching:** Download and merge AQI and weather data from government APIs.
2. **Model Training:** Train ensemble ML models on historical data.
3. **Forecast Generation:** Predict AQI for the next 3 days for all stations.
4. **Incremental Forecasting:** Update 1-hour-ahead AQI forecasts for all stations every hour.
5. **API Serving:** Serve real-time, historical, and forecast AQI via FastAPI.
6. **Automation:** Orchestrator script automates all steps and logs job status.

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
# Python 3.8+
git clone <your-repo-url>
cd aqi-forecast-app
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Also install the scheduler dependency:
pip install schedule
```

---

## 1️⃣ Data Fetching

Fetch and merge all available AQI and weather data:
```bash
python scripts/fetch_and_merge_all_data.py
```

---

## 2️⃣ Model Training

Train the ensemble ML model:
```bash
python scripts/train_ensemble_model.py
```

---

## 3️⃣ Forecast Generation

Generate 3-day AQI forecasts:
```bash
python scripts/predict_3day_aqi.py
```

---

## 4️⃣ Incremental Forecasting

Update 1-hour-ahead AQI forecasts for all stations (run hourly):
```bash
python scripts/incremental_forecast.py
```
- This script is automatically scheduled by the orchestrator to run every hour (at :05).

---

## 5️⃣ API Server

Start the FastAPI server to serve predictions:
```bash
python -m api.main
# or
uvicorn api.main:app --reload
```

---

## 6️⃣ Automation & Orchestration

Run the orchestrator to automate all backend jobs:
```bash
python run_backend_orchestrator.py
```
- Starts API server, schedules data fetch, **incremental forecast (hourly)**, 3-day forecast (daily), and model retraining (every 3 days).
- Logs job status to `logs/orchestrator_job_status.csv` (timestamp, job, status, error)

---

## 🔑 API Usage
- All endpoints require an API key (see `/register` endpoint).
- Pass your API key in the `X-API-KEY` header.

### **Main Endpoints**
| Method | Path                                 | Description |
|--------|--------------------------------------|-------------|
| GET    | `/`                                  | API info root |
| GET    | `/live/{station_id}`                 | Real-time AQI for a station |
| GET    | `/forecast/available_stations`       | List all station IDs with forecast data |
| GET    | `/forecast/{station_id}`             | 3-day AQI forecast for a station |
| GET    | `/forecast/{station_id}/summary`     | Forecast summary for a station |
| GET    | `/stations/search`                   | Search stations by city, state, name |
| GET    | `/stations/{station_id}`             | Station metadata |
| GET    | `/usage`                             | API key usage info |
| POST   | `/admin/generate_key`                | (Admin) Generate API key |
| GET    | `/health`                            | API health check |
| GET    | `/metadata`                          | API metadata and endpoints |
| POST   | `/register`                          | Register user and get API key |
| GET    | `/history/{station_id}`              | Historical AQI for a station |
| GET    | `/advisory/{aqi}`                    | Health advice for AQI value |
| GET    | `/advisory/auto/{station_id}`        | Health advice for a station's AQI |
| GET    | `/stations/geojson`                  | All stations as GeoJSON |
| POST   | `/alerts/subscribe`                  | Subscribe to AQI alerts (mocked) |
| POST   | `/alerts/unsubscribe`                | Unsubscribe from AQI alerts (mocked) |
| GET    | `/alerts/check/{station_id}`         | Check if AQI exceeds threshold |
| GET    | `/cities`                            | List all cities with stations |
| GET    | `/categories`                        | AQI categories and breakpoints |

---

### **Example: Get Forecast**
```python
import requests
headers = {"X-API-KEY": "your_api_key"}
resp = requests.get("http://localhost:8000/forecast/Delhi_ITO_Delhi_CPCB", headers=headers)
print(resp.json())
```

---

## ⚙️ Configuration
- Main config: `config/config.py`
- Environment: `.env` (optional, for secrets)
- Data, models, and logs are **not included** in the repo (see `.gitignore`).

---

## 🏗️ Folder Structure
```
api-forecast-app/
├── api/           # FastAPI backend
├── automation/    # Automation scripts (hourly/daily)
├── config/        # Configuration files
├── data/          # (Ignored) Data storage
├── logs/          # (Ignored) Logs
├── models/        # (Ignored) Trained models
├── scripts/       # Main pipeline scripts
├── src/           # Core source code
├── tests/         # Unit tests
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🤝 Contributing
1. Fork the repo and create a feature branch.
2. Make your changes with clear commits.
3. Ensure all tests pass (`pytest`).
4. Open a pull request with a clear description.

---

## 📄 License
MIT License. See `LICENSE` file.

---

## 📬 Contact
For questions or contributions, open an issue or contact the maintainer.
