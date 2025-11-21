
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from filterpy.kalman import KalmanFilter
import shap
import asyncio
import time
# backend/main.py
# ... other imports like FastAPI, joblib, etc. ...
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from collections import deque # <-- ADD THIS
import json
import keras
from typing import Dict, Optional
import numpy as np
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.models import load_model
from pydantic import Field # For defining lists in Pydantic
from typing import List # For defining lists in Pydantic
from tensorflow.keras.models import load_model, model_from_json# Add model_from_json
SEQUENCE_LEN = 10
# Initialize the FastAPI application
app = FastAPI(title="SmartPredict AI Platform API")
# ... after app = FastAPI(...) line ...

# Store recent readings (e.g., last 12 readings = 2 minutes of 10s intervals)
MAX_RECENT_READINGS = 20
recent_readings: Dict[str, deque] = {} # Key: machine_id, Value: deque of readings
# --- CORS Middleware Setup ---
# This allows your React frontend (running on localhost:3000) to communicate with this backend
origins = [
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 1. LOAD ALL MODELS AND EXPLAINERS ON STARTUP
# ==============================================================================

# --- Battery Model ---
try:
    battery_model = joblib.load('backend/rul_predictor_model.joblib')
    print("✅ Battery model loaded successfully.")
except FileNotFoundError:
    print("❌ Battery model file not found.")
    battery_model = None

# --- Motor Model ---
try:
    motor_model = joblib.load('backend/motor_model.joblib')
    print("✅ Motor model loaded successfully.")
except FileNotFoundError:
    print("❌ Motor model file not found.")
    motor_model = None

# --- Hydraulic Model ---
try:
    hydraulic_model = joblib.load('backend/hydraulic_model.joblib')
    hydraulic_status_labels = {
        3: 'Optimal Efficiency',
        20: 'Reduced Efficiency',
        100: 'Close to Total Failure'
    }
    print("✅ Hydraulic model and labels loaded successfully.")
except FileNotFoundError:
    hydraulic_model = None

# --- SHAP Explainer for Motor Model (Using the robust modern Explainer) ---
try:
    motor_model_explainer = shap.Explainer(motor_model)
    print("✅ SHAP explainer for motor model created successfully.")
except Exception as e:
    motor_model_explainer = None
    print(f"❌ Could not create SHAP explainer: {e}")

# In backend/main.py, section 1

# In backend/main.py, section 1

# --- LSTM Autoencoder Model & Scaler (Loading Separately with explicit imports) ---
try:
    # --- ADD IMPORTS INSIDE TRY BLOCK ---
    import keras
    from tensorflow.keras.models import model_from_json
    # --- END ADDED IMPORTS ---

    print(f"--- Keras version being used by backend: {keras.__version__} ---") # Add version check

    # Load architecture from JSON file
    with open('backend/motor_lstm_autoencoder_arch.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    motor_ae_model = model_from_json(loaded_model_json)

    # Load weights into the new model
    motor_ae_model.load_weights("backend/motor_lstm_autoencoder_weights.weights.h5")

    # Compile the model
    motor_ae_model.compile(optimizer='adam', loss='mae')

    motor_ae_scaler = joblib.load('backend/motor_ae_scaler.joblib')
    print("✅ LSTM Autoencoder model (arch+weights) and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Could not load LSTM Autoencoder model or scaler: {e}")
    motor_ae_model = None
    motor_ae_scaler = None
# ==============================================================================
# 2. DEFINE INPUT DATA MODELS (PYDANTIC)
# ==============================================================================

class BatteryFeatures(BaseModel):
    cycle: int
    capacity: float
    temp_mean: float
    voltage_mean: float
    current_mean: float
    degradation_anomaly_score: float

class MotorFeatures(BaseModel):
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float

class HydraulicFeatures(BaseModel):
    PS1: float
    PS2: float
    PS3: float
    PS4: float
    PS5: float
    PS6: float
    EPS1: float
    FS1: float
    TS1: float
    TS2: float
    TS3: float
    TS4: float
    VS1: float
    CE: float
    CP: float
    SE: float
# In backend/main.py, section 2

class LiveSensorData(BaseModel):
    machine_id: str
    temperature_c: float
    humidity_percent: float
    timestamp: Optional[float] = None # Make timestamp optional
# In backend/main.py, section 2

# In backend/main.py, section 2

class MotorSequenceData(BaseModel):
    sequence: List[List[float]] = Field(..., min_items=SEQUENCE_LEN, max_items=SEQUENCE_LEN) # This uses the global SEQUENCE_LEN
# ==============================================================================
# 3. DEFINE API ENDPOINTS
# ==============================================================================

@app.get("/")
def read_root():
    return {"message": "Welcome to the SmartPredict Predictive Maintenance API!"}

@app.post("/predict/battery")
def predict_rul(features: BatteryFeatures):
    if battery_model is None:
        return {"error": "Battery model not loaded."}
    input_df = pd.DataFrame([features.dict()])
    prediction = battery_model.predict(input_df)
    predicted_rul = prediction[0]
    return {"predicted_RUL": round(float(predicted_rul), 2)}

@app.post("/predict/motor")
def predict_motor_failure(features: MotorFeatures):
    if motor_model is None:
        return {"error": "Motor model not loaded."}
    
    feature_dict = features.dict()
    feature_dict['Air temperature [K]'] = feature_dict.pop('air_temperature_k')
    feature_dict['Process temperature [K]'] = feature_dict.pop('process_temperature_k')
    feature_dict['Rotational speed [rpm]'] = feature_dict.pop('rotational_speed_rpm')
    feature_dict['Torque [Nm]'] = feature_dict.pop('torque_nm')
    feature_dict['Tool wear [min]'] = feature_dict.pop('tool_wear_min')
    input_df = pd.DataFrame([feature_dict])

    prediction = motor_model.predict(input_df)
    probability = motor_model.predict_proba(input_df)
    status = "At Risk" if prediction[0] == 1 else "Normal"
    health_score = (1 - probability[0][1]) * 100
    return {"status": status, "health_score": f"{health_score:.2f}"}
