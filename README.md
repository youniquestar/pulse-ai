# Pulse AI: Real-Time Predictive Maintenance & Root-Cause Diagnosis Engine 🚀

`Pulse AI` is an end-to-end, production-grade machine learning pipeline and backend API designed to monitor industrial machinery health, detect operational anomalies, and diagnose root-cause asset failures before catastrophic breakdowns occur.

Built around the **AI4I 2020 Predictive Maintenance Dataset** (simulating 10,000 milling machine records), this project bridges the gap between raw time-series sensor telemetry and robust, deployable backend architecture.

---

## 🛠️ Tech Stack & Architecture

* **Data Handling & Physics Engineering:** `Pandas`, `NumPy`
* **Machine Learning Frameworks:** `XGBoost`, `Scikit-Learn`, `Imbalanced-Learn` 
* **Production Deployment:** `FastAPI` (Inference API), `Uvicorn` (ASGI Server), `Pydantic` 
* **Model Serialization:** `Joblib`

### System Blueprint
1. **Data Ingestion & Engineering:** Telemetry ingestion $\rightarrow$ Mathematical feature extraction.
2. **ML Optimization Engine:** Scale-invariant parameter processing $\rightarrow$ Gradient boosting adjustment.
3. **Inference Pipeline:** Serialized pipeline assets embedded inside a production REST API handling live JSON streams.

---

## 📊 Phase 1: Feature Physics & Preprocessing

Raw industrial sensors are inherently noisy, and individual dimensions often mask failures. This pipeline introduces advanced domain-specific feature engineering to capture the physical interaction of multidimensional parameters:

* **Temperature Delta ($\Delta T$):** Calculated as $\text{Process Temperature} - \text{Air Temperature}$ to isolate thermal dissipation issues.
* **Mechanical Power Consumption ($P$):** Derived via $\text{Rotational Speed} \times \text{Torque}$ to reveal structural jamming or mechanical resistance.
* **Nominal Quality Encoding:** Categorical tool variants (Low, Medium, High quality) are transformed using **One-Hot Encoding** to maintain unbiased feature weights and avoid creating arbitrary mathematical hierarchies.
* **Feature Standardization:** Scaled via `StandardScaler` to ensure high-magnitude parameters (e.g., RPM in thousands) do not mathematically drown out critical lower-magnitude metrics like torque.

---

## 📈 Phase 2: The Optimization Journey (Combating Imbalance)

Industrial maintenance data features a severe class imbalance (~96% healthy, ~4% failure). Evaluating model performance strictly via strict **Confusion Matrix** auditing (Precision vs. Recall) rather than misleading global accuracy scores, the engine underwent an iterative optimization evolution:

| Iteration | Model Strategy                  | Precision |  Recall  | Key Insights & Architectural Takeaway                                                   |
|:---------:|:--------------------------------|:---------:|:--------:|:----------------------------------------------------------------------------------------|
|  **01**   | `Random Forest` (Baseline)      | **0.93**  | **0.79** | Excellent precision but missed 21% of critical failures due to data class bias.         |
|  **02**   | `Random Forest` (Class Weights) | **0.94**  | **0.71** | Custom weight scaling warped tree impurity math, dropping recall performance.           |
|  **03**   | `Random Forest` + `SMOTE`       | **0.61**  | **0.82** | Synthetic data generation increased recall but introduced high false-alarm paranoia.    |
|  **04**   | `XGBoost` (scale_pos_weight)    | **0.51**  | **0.91** | Massive recall surge; however, heavy gradient weighting introduced over-aggressiveness. |
|  **05**   | `Optimized XGBoost` (Final)     | **0.64**  | **0.82** | Systematic hyperparameter optimization achieved the optimal industry-ready balance.     |

### Hyperparameter Fine-Tuning
Using a 3-fold cross-validated Stratified `GridSearchCV` optimized against the **F1-score**, the final operational model bounds were unlocked:

```python
Optimal_Parameters = {
    'learning_rate': 0.1,      # Restricts aggressive tree correction steps
    'max_depth': 5,            # Controls structural complexity to prevent noise memorization
    'min_child_weight': 1      # Dictates strict partition minimums for rare anomalies
}
```

---

## 🚀 Phase 3: Production Deployment & Live Simulation

The finalized machine learning pipeline is decoupled from training environments, fully serialized, and embedded into a low-latency asynchronous application layer.

### 1. Production API Topology (`main.py`)

The backend service hosts a high-performance HTTP POST endpoint `/predict-health` that validates streaming industrial telemetry structures via `Pydantic`, calculates on-the-fly physics deltas, and scores operational risk instantaneously.

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize application
app = FastAPI(title="Vibe-Check: Predictive Maintenance Engine")

# Load serialized model assets
model = joblib.load('maint_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the expected JSON input schema using Pydantic
class SensorInput(BaseModel):
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type_M: int  # One-hot encoded column for Medium quality
    type_H: int  # One-hot encoded column for High quality

@app.post("/predict-health")
def predict_health(data: SensorInput):
    # 1. Extract domain features exactly as done during training
    temp_delta = data.process_temp - data.air_temp
    power = data.rotational_speed * data.torque
    
    # 2. Arrange features into the precise 9-column array your model expects
    raw_features = np.array([[
        data.air_temp, data.process_temp, data.rotational_speed, 
        data.torque, data.tool_wear, temp_delta, power, 
        data.type_M, data.type_H
    ]])
    
    # 3. Scale the input features using the loaded scaler
    scaled_features = scaler.transform(raw_features)
    
    # 4. Run inference
    prediction = int(model.predict(scaled_features)[0])
    probability = float(model.predict_proba(scaled_features)[0][1])
    
    # 5. Return live payload response
    return {
        "machine_failure_predicted": prediction,
        "failure_probability": round(probability, 4),
        "status": "ALERT: Maintenance Required" if prediction == 1 else "System Healthy"
    }
```

To launch the backend engine natively on your machine, execute the following command:

```Bash
uvicorn main:app --reload
```

### 2. Factory Floor Telemetry Stream Simulator (simulator.py)
To test backend stability and model performance under real-time conditions, a decoupled hardware client pulls asynchronous packets from raw telemetry logs and streams them over live network transport layers to the API.

```python
import time
import pandas as pd
import requests

# Configuration
API_URL = "[http://127.0.0.1:8000/predict-health](http://127.0.0.1:8000/predict-health)"

# Load the original CSV dataset to simulate live hardware readings
try:
    df = pd.read_csv('ai4i2020.csv')
    print(f"Successfully loaded {len(df)} simulation records.")
except FileNotFoundError:
    print("Error: 'ai4i2020.csv' not found. Please ensure it is in this directory.")
    exit()

# Extract a mix of healthy and failure rows for an interesting simulation
healthy_sample = df[df['Machine failure'] == 0].head(15)
failure_sample = df[df['Machine failure'] == 1].head(5)
simulation_data = pd.concat([healthy_sample, failure_sample]).sample(frac=1, random_state=42)

print("Starting factory floor simulation. Press Ctrl+C to stop...\n")
print(f"{'Step':<5} | {'Sensor Speed':<12} | {'Sensor Torque':<13} | {'Backend API Response'}")
print("-" * 70)

# Loop through the data to stream payloads to the API
for index, row in simulation_data.iterrows():
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
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            res_data = response.json()
            prob_pct = f"{res_data['failure_probability'] * 100:.1f}%"
            print(f"{index:<5} | {payload['rotational_speed']:<12.1f} | {payload['torque']:<13.1f} | {res_data['status']} ({prob_pct} prob)")
    except requests.exceptions.ConnectionError:
        print("Connection Error: Is your FastAPI server running? Run 'uvicorn main:app --reload'")
        break

    time.sleep(1)
```

To execute the live hardware telemetry stream simulation, open a secondary terminal and run:

```Bash
python simulator.py
```

### 3. Real-Time Live Telemetry Stream Dashboard Output
When both scripts run concurrently, your client terminal will log real-time API performance outputs as follows:

```Plaintext
Successfully loaded 10000 simulation records.
Starting factory floor simulation. Press Ctrl+C to stop...

Step  | Sensor Speed | Sensor Torque | Backend API Response 
----------------------------------------------------------------------
142   | 1512.0       | 38.4          | System Healthy (1.2% prob) 
143   | 1498.0       | 41.1          | System Healthy (2.8% prob) 
814   | 2642.0       | 12.3          | System Healthy (8.4% prob)
3205  | 1354.0       | 56.7          | ALERT: Maintenance Required (87.6% prob) 🔥
3206  | 1321.0       | 61.2          | ALERT: Maintenance Required (93.1% prob) 🔥
```

---

## 📂 Project Organization Structure
The repository workspace layout is contained within the following explicit file tree:

```Plaintext
vibe-check/
│
├── data # A folder for relevant data and models
    ├── main_model.pkl      # Production-serialized, optimized XGBoost model binary 
    ├── scaler.pkl           # Serialized StandardScaler preprocessing asset
    ├── ai4i2020.csv         # Local copy of the industrial manufacturing dataset
├── main.py              # FastAPI production deployment backend application 
├── simulator.py         # Live factory floor data-stream simulator client
├── xgboost_model.py     # The final standardized XGBoost model 
└── README.md            # Professional technical repository documentation
```

---

## 📈 Future Scalability Roadmaps

* **Explainable AI Integration:** Embed localized SHAP (SHapley Additive exPlanations) vector processing inside the API response payloads. This will provide field technicians with explicit root-cause breakdown justifications (e.g., explaining why a machine is flagged for failure due to cumulative tool wear vs. thermal deltas).
* **Streaming Windowing Systems:** Transition state ingestion architectures from static endpoint inferences to real-time windowing streams. This modification scales the framework to process rolling historical windows natively across production Kafka or MQTT data pipes.