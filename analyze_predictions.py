import json
import pandas as pd
from datetime import datetime

# Load prediction results
print("=== AQI FORECAST RESULTS ANALYSIS ===")

# Load JSON predictions
with open('data/predictions_3day_20250706_024937.json', 'r') as f:
    predictions = json.load(f)

print(f"\n📊 PREDICTION SUMMARY:")
print(f"Total stations with predictions: {len(predictions)}")

# Show sample predictions
print(f"\n📈 SAMPLE PREDICTIONS:")
if len(predictions) > 0:
    first_station = predictions[0]
    print(f"Station: {first_station.get('station', 'Unknown')}")
    print(f"City: {first_station.get('city', 'Unknown')}")
    print(f"State: {first_station.get('state', 'Unknown')}")
    print(f"Number of predictions: {len(first_station.get('predictions', []))}")
    
    # Show first few predictions
    preds = first_station.get('predictions', [])
    if preds:
        print(f"\nFirst 5 predictions:")
        for i, pred in enumerate(preds[:5]):
            print(f"  Hour {i+1}: AQI = {pred.get('aqi', 'N/A')}")

# Load CSV summary
try:
    df = pd.read_csv('data/predictions_3day_summary_20250706_024937.csv')
    print(f"\n📋 CSV SUMMARY:")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"\nSample data:")
        print(df.head(3).to_string())
        
except Exception as e:
    print(f"Could not read CSV: {e}")

print(f"\n✅ Analysis complete!") 