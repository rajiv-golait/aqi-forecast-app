@echo off
set PYTHONPATH=%~dp0
:loop
REM Run the data pipeline (fetch AQI and weather, merge, preprocess)
python scripts\run_data_pipeline.py
REM Wait for 1 hour (3600 seconds)
timeout /t 3600
REM Repeat
goto loop 