import time
import json
import pandas as pd
import requests

# 1. Configuration
# Ensure your FastAPI server is running locally on port 8000
API_URL = "http://127.0.0.1:8000/predict-health"

# Load the original CSV dataset to simulate live hardware readings
try:
    df = pd.read_csv('data/ai4i2020.csv')
    print(f"Successfully loaded {len(df)} simulation records.")
except FileNotFoundError:
    print("Error: 'ai4i2020.csv' not found. Please ensure it is in this directory.")
    exit()

# 2. Extract a mix of healthy and failure rows for an interesting simulation
# We'll take the first 20 rows, plus a few rows where we know a failure happens
healthy_sample = df[df['Machine failure'] == 0].head(15)
failure_sample = df[df['Machine failure'] == 1].head(5)
simulation_data = pd.concat([healthy_sample, failure_sample]).sample(frac=1, random_state=42)

print("Starting factory floor simulation. Press Ctrl+C to stop...\n")
print(f"{'Step':<5} | {'Sensor Speed':<12} | {'Sensor Torque':<13} | {'Backend API Response'}")
print("-" * 70)

# 3. Loop through the data to stream payloads to the API
for index, row in simulation_data.iterrows():
    # Map raw CSV columns to your API Pydantic model structure
    # Parse the string 'Type' column back to the one-hot encoded variables
    payload = {
        "air_temp": float(row['Air temperature [K]']),
        "process_temp": float(row['Process temperature [K]']),
        "rotational_speed": float(row['Rotational speed [rpm]']),
        "torque": float(row['Torque [Nm]']),
        "tool_wear": float(row['Tool wear [min]']),
        "type_M": 1 if row['Type'] == 'M' else 0,
        "type_H": 1 if row['Type'] == 'H' else 0
    }

    try:
        # Fire the POST request to the backend
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            res_data = response.json()
            prob_pct = f"{res_data['failure_probability'] * 100:.1f}%"

            # Print the real-time telemetry log
            print(f"{index:<5} | {payload['rotational_speed']:<12.1f} | {payload['torque']:<13.1f} | {res_data['status']} ({prob_pct} prob)")
        else:
            print(f"API Error: Status code {response.status_code}")

    except requests.exceptions.ConnectionError:
        print("Connection Error: Is your FastAPI server running? Run 'uvicorn main:app --reload'")
        break

    # Wait 1 second before sending the next telemetry packet
    time.sleep(1)