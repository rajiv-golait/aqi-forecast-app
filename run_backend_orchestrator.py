import subprocess
import schedule
import time
import threading
import sys
import os
import csv
from datetime import datetime

STATUS_CSV = 'logs/orchestrator_job_status.csv'

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Helper to log job status to CSV
def log_job_status(job, status, error=None):
    with open(STATUS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(['timestamp', 'job', 'status', 'error'])
        writer.writerow([
            datetime.now().isoformat(timespec='seconds'),
            job,
            status,
            error or ''
        ])
    print(f"[status] {datetime.now().isoformat(timespec='seconds')} | {job} | {status} | {error or ''}")

# Helper to run a script and log status
def run_script_with_status(job_name, script_path):
    print(f"[orchestrator] Running: {script_path}")
    try:
        result = subprocess.run([sys.executable, script_path])
        if result.returncode == 0:
            log_job_status(job_name, 'success')
        else:
            log_job_status(job_name, 'fail', f"exit code {result.returncode}")
    except Exception as e:
        log_job_status(job_name, 'fail', str(e))

# 1. Fetch and merge hourly
def fetch_and_merge():
    run_script_with_status('fetch_and_merge', os.path.join('scripts', 'fetch_and_merge_all_data.py'))

# 2. Incremental forecast hourly
def incremental_forecast():
    run_script_with_status('incremental_forecast', os.path.join('scripts', 'incremental_forecast.py'))

# 3. Predict 3-day forecast once a day
def predict_3day():
    run_script_with_status('predict_3day', os.path.join('scripts', 'predict_3day_aqi.py'))

# 4. Train ML model once every 3 days
def train_model():
    run_script_with_status('train_model', os.path.join('scripts', 'train_ensemble_model.py'))

# 5. API server always running (in background)
def start_api_server():
    print("[orchestrator] Starting API server...")
    return subprocess.Popen([sys.executable, '-m', 'api.main'])

# Schedule jobs
def schedule_jobs():
    schedule.every().hour.at(":00").do(fetch_and_merge)
    schedule.every().hour.at(":05").do(incremental_forecast)
    schedule.every().day.at("01:00").do(predict_3day)
    schedule.every(3).days.at("02:00").do(train_model)

# Main loop
def main():
    api_proc = start_api_server()
    schedule_jobs()
    print("[orchestrator] Scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("[orchestrator] Shutting down...")
        api_proc.terminate()
        api_proc.wait()
        print("[orchestrator] API server stopped.")

if __name__ == "__main__":
    main() 