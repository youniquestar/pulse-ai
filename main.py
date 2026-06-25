from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# initialize application
app = FastAPI(title="Pulse AI: Predictive Maintenance Engine")

# load the serialized model assets
model = joblib.load('data/main_model.pkl')
scaler = joblib.load('data/scaler.pkl')

# Define the expected JSON input schema using Pydantic
class SensorInput(BaseModel):
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type_M: int # one-hot encoded column for Medium Quality
    type_H: int # one-hot encoded column for High Quality

@app.post("/predict-health")
def predict_health(data: SensorInput):
    # extraxt domain features exactly as done during training
    temp_delta = data.process_temp - data.air_temp
    power = data.rotational_speed * data.torque

    # arrange features into a precise 9-column array that the model expects
    raw_features = np.array([[
        data.air_temp, data.process_temp, data.rotational_speed,
        data.torque, data.tool_wear, temp_delta, power,
        data.type_M, data.type_H
    ]])

    # scale the input features using the loaded scaler
    scaled_features = scaler.transform(raw_features)

    # run inference
    prediction = int(model.predict(scaled_features)[0])
    probability = float(model.predict_proba(scaled_features)[0][1])

    # return live payload responeses
    return {
        "machine_failure_predicted": prediction,
        "failure_probability": round(probability, 4),
        "status": "ALERT: Maintenance Required" if prediction == 1 else "System Healthy"
    }