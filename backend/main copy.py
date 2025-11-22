# backend/main.py
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

@app.post("/predict/hydraulic")
def predict_hydraulic_failure(features: HydraulicFeatures):
    if hydraulic_model is None:
        return {"error": "Hydraulic model not loaded."}
    input_df = pd.DataFrame([features.dict()])
    prediction_code = hydraulic_model.predict(input_df)[0]
    status = hydraulic_status_labels.get(prediction_code, "Unknown Condition")
    health_score = 100 - prediction_code
    return {"status": status, "health_score": f"{health_score:.2f}"}

# --- THIS IS THE FULLY CORRECTED AND ROBUST EXPLAINABILITY ENDPOINT ---
@app.post("/explain/motor")
def explain_motor_prediction(features: MotorFeatures):
    if motor_model is None or motor_model_explainer is None:
        return {"error": "Model or explainer not loaded."}

    # Prepare the input DataFrame exactly as before
    feature_dict = features.dict()
    feature_dict['Air temperature [K]'] = feature_dict.pop('air_temperature_k')
    feature_dict['Process temperature [K]'] = feature_dict.pop('process_temperature_k')
    feature_dict['Rotational speed [rpm]'] = feature_dict.pop('rotational_speed_rpm')
    feature_dict['Torque [Nm]'] = feature_dict.pop('torque_nm')
    feature_dict['Tool wear [min]'] = feature_dict.pop('tool_wear_min')
    input_df = pd.DataFrame([feature_dict])

    # Use the explainer object directly. This returns a rich Explanation object.
    shap_explanation = motor_model_explainer(input_df)
    
    # For a binary classifier, the output has two sets of values.
    # We want the values for class 1 ("At Risk").
    # The .values attribute contains the SHAP values.
    # The [0, :, 1] slicing gets:
    # [0]   - The first (and only) prediction sample.
    # [:]   - All features for that sample.
    # [1]   - The values for class 1 ("At Risk").
    shap_values_for_failure = shap_explanation.values[0, :, 1]

    feature_importance = pd.DataFrame({
        'feature': input_df.columns,
        'importance': shap_values_for_failure
    }).sort_values('importance', ascending=False)
    
    return feature_importance.to_dict('records')

# In backend/main.py

# ... (keep all your existing imports and code)

# backend/main.py

# ==============================================================================
# 3.5 BUSINESS IMPACT ENDPOINT (NEW)
# ==============================================================================
@app.get("/business-impact/{machine_type}")
def get_business_impact(machine_type: str):
    """
    Calculates and returns the potential cost of failure and savings
    based on pre-defined costs for the machine type.
    """
    try:
        with open('backend/cost_config.json', 'r') as f:
            cost_data = json.load(f)

        if machine_type not in cost_data:
            return {"error": "Cost data not available for this machine type."}

        config = cost_data[machine_type]
        cost_per_hour = config.get("costPerHourDowntime", 0)
        hours_repair = config.get("hoursToRepairFailure", 0)
        cost_proactive = config.get("costOfProactiveMaintenance", 0)

        total_cost_failure = cost_per_hour * hours_repair
        predicted_savings = total_cost_failure - cost_proactive

        return {
            "totalCostOfFailure": total_cost_failure,
            "predictedSavings": predicted_savings,
            "costPerHourDowntime": cost_per_hour, # Send back for display/adjustment
            "hoursToRepairFailure": hours_repair,
            "costOfProactiveMaintenance": cost_proactive
        }
    except FileNotFoundError:
        return {"error": "Cost configuration file not found."}
    except Exception as e:
        return {"error": f"Error calculating business impact: {str(e)}"}
# ==============================================================================
# 4. UNIVERSAL DRIFT DETECTION ENDPOINT
# ==============================================================================
@app.get("/drift-analysis/{machine_type}")
def get_drift_analysis(machine_type: str):
    """
    Simulates and returns data drift for a specified machine type and its key feature.
    """
    try:
        # --- Configure parameters based on machine type ---
        if machine_type == 'battery':
            filepath = 'data/processed_battery_data.csv'
            feature_column = 'capacity'
            drift_simulation = -0.15  # Simulate capacity fade
        elif machine_type in ['motor', 'pump']: # Reuse motor data for pump
            filepath = 'data/ai4i2020.csv'
            feature_column = 'Torque [Nm]'
            drift_simulation = 6.5  # Simulate increased torque load
        elif machine_type == 'hydraulic':
            # NOTE: Using the motor dataset as a stand-in for hydraulic for demo purposes
            # In a full project, you'd load the actual hydraulic_data.csv
            filepath = 'data/ai4i2020.csv'
            feature_column = 'Process temperature [K]' # Using temperature as a proxy for PS1
            drift_simulation = 10.0 # Simulate system overheating
        else:
            return {"error": "Unknown machine type"}

        # 1. Load the original training data
        df_original = pd.read_csv(filepath)
        original_data = df_original[feature_column]

        # 2. Simulate "live" data with drift
        live_data = original_data + drift_simulation

        # 3. Calculate distributions for charting
        combined_min = min(original_data.min(), live_data.min())
        combined_max = max(original_data.max(), live_data.max())
        bins = np.linspace(combined_min, combined_max, 40)
        original_counts, _ = np.histogram(original_data, bins=bins)
        live_counts, _ = np.histogram(live_data, bins=bins)

        # 4. Format the data for the frontend
        chart_data = [
            {
                "bin_start": bins[i],
                "original_distribution": int(original_counts[i]),
                "live_distribution": int(live_counts[i]),
            }
            for i in range(len(original_counts))
        ]
        
        return {"status": "success", "data": chart_data}

    except FileNotFoundError:
        return {"error": f"Dataset for '{machine_type}' not found at {filepath}."}
    except Exception as e:
        return {"error": str(e)}
    

# In backend/main.py

# ==============================================================================
# 5. MODEL RETRAINING ENDPOINT (SIMULATION)
# ==============================================================================
@app.get("/retrain/{machine_type}")
def retrain_model(machine_type: str):
    """
    Simulates the initiation of a model retraining pipeline.
    In a real-world scenario, this would trigger a cloud-based training job.
    """
    print(f"✅ Retraining process initiated for machine type: {machine_type}")
    return {
        "status": "success",
        "message": f"Retraining pipeline for {machine_type} model has been successfully initiated."
    }

# In backend/main.py

# ==============================================================================
# 6. CLUSTERING INSIGHTS ENDPOINT
# ==============================================================================
# In backend/main.py, find the /fleet-clusters/{machine_type} endpoint
# In backend/main.py

@app.get("/fleet-clusters/{machine_type}")
def get_fleet_clusters(machine_type: str):
    """
    Returns the pre-computed clustering analysis for a specified machine type.
    """
    try:
        if machine_type == 'battery':
            filepath = 'data/clustered_fleet_data.csv'
            df_clusters = pd.read_csv(filepath)

        elif machine_type == 'motor':
            filepath = 'data/clustered_motor_data.csv'
            df_clusters = pd.read_csv(filepath)

            # --- THIS IS THE FIX ---
            # If the data is too large, take a random sample of 300 points
            if len(df_clusters) > 300:
                df_clusters = df_clusters.sample(n=300, random_state=42)
            # --- END OF FIX ---

        else:
            return {"error": "Clustering analysis not available for this machine type."}

        return df_clusters.to_dict('records')

    except FileNotFoundError:
        return {"error": f"Clustered data for '{machine_type}' not found."}

# In backend/main.py

# ==============================================================================
# 7. DATA PREPROCESSING VISUALIZATION
# ==============================================================================
@app.get("/visualize/noise-filter")
def get_noise_filter_visualization():
    """
    Generates a sample of a noisy signal and its smoothed version
    using a moving average to demonstrate data preprocessing.
    """
    # 1. Create a clean sine wave as our "true signal"
    x = np.linspace(0, 10, 100)
    true_signal = np.sin(x) * 10 + 50  # e.g., a temperature oscillating around 50°C

    # 2. Add random noise to simulate a real-world sensor
    noise = np.random.normal(0, 2.5, 100) # Add random fluctuations
    noisy_signal = true_signal + noise

    # 3. Apply a moving average filter to smooth the noise
    window_size = 5
    smoothed_signal = pd.Series(noisy_signal).rolling(window=window_size).mean()

    # 4. Format the data for the frontend chart
    chart_data = []
    for i in range(len(x)):
        chart_data.append({
            "time_step": i,
            "raw_noisy_data": noisy_signal[i],
            # Handle initial null values from rolling average
            "smoothed_data": smoothed_signal.iloc[i] if pd.notna(smoothed_signal.iloc[i]) else None,
        })

    return chart_data
# ==============================================================================
# 8. KALMAN FILTER VISUALIZATION
# ==============================================================================
# ...existing code...

@app.get("/visualize/kalman-filter")
def get_kalman_filter_visualization():
    """
    Generates a noisy signal and shows the superior smoothing
    of a Kalman Filter compared to a standard Moving Average.
    """
    # 1. Create a clean signal (e.g., rising temperature)
    true_signal = np.linspace(50, 80, 100)

    # 2. Add significant noise
    noise = np.random.normal(0, 4, 100)
    noisy_signal = true_signal + noise

    # 3. Apply a Moving Average (for comparison)
    smoothed_signal_ma = pd.Series(noisy_signal).rolling(window=10).mean()

    # 4. Apply a Kalman Filter
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[50.], [0.]])  # Initial state [position; velocity]
    kf.F = np.array([[1., 1.], [0., 1.]])   # State Transition Matrix
    kf.H = np.array([[1., 0.]])    # Measurement Function
    kf.P *= 1000.                  # Covariance Matrix
    kf.R = 5                       # Measurement Noise
    kf.Q = np.array([[(0.05**2)/4, (0.05**2)/2], [(0.05**2)/2, 0.05**2]]) # Process Noise

    # Unpack the tuple returned by batch_filter, keeping only the first item (predictions)
    predictions, *_ = kf.batch_filter(noisy_signal)
    smoothed_signal_kf = predictions[:, 0, 0]

    # 5. Format for frontend
    chart_data = [
        {
            "time_step": i,
            "raw_noisy_data": noisy_signal[i],
            "moving_average": smoothed_signal_ma.iloc[i] if pd.notna(smoothed_signal_ma.iloc[i]) else None,
            "kalman_filter": smoothed_signal_kf[i],
        }
        for i in range(len(true_signal))
    ]

    return chart_data
# backend/main.py

# @app.get("/visualize/kalman-filter/{machine_id}") # Now takes machine_id
# def get_kalman_filter_visualization_live(machine_id: str):
#     """
#     Applies Kalman filter to the most recent 'live' sensor data received
#     for the specified machine_id.
#     """
#     global recent_readings # Access the global deque dictionary

#     if machine_id not in recent_readings or len(recent_readings[machine_id]) < 2:
#         return {"error": "Not enough recent data received for Kalman filtering."}

#     # 1. Get the recent noisy data from the deque
#     # --- IMPORTANT: Ensure 'temperature_c' matches the key used in receive_live_data ---
#     noisy_signal = np.array([reading['temperature_c'] for reading in recent_readings[machine_id]])
#     timestamps = np.array([reading['timestamp'] for reading in recent_readings[machine_id]]) # Optional: for x-axis

#     # 2. Apply a Moving Average (window size might need tuning)
#     smoothed_signal_ma = pd.Series(noisy_signal).rolling(window=min(5, len(noisy_signal)-1)).mean() # Adjust window

#     # 3. Apply a Kalman Filter
#     kf = KalmanFilter(dim_x=2, dim_z=1)
#     initial_value = noisy_signal[0]
#     kf.x = np.array([[initial_value], [0.]]) # Initial state [temp; rate_of_change]
#     kf.F = np.array([[1., 1.], [0., 1.]])   # State Transition Matrix (assuming constant velocity)
#     kf.H = np.array([[1., 0.]])    # Measurement Function
#     kf.P *= 100.                   # Initial Covariance (uncertainty)
#     kf.R = 5                       # Measurement Noise Variance (tune this!)
#     kf.Q = np.array([[(0.1**2)/4, (0.1**2)/2], [(0.1**2)/2, 0.1**2]]) # Process Noise Variance (tune this!)

#     predictions, *_ = kf.batch_filter(noisy_signal)
#     smoothed_signal_kf = predictions[:, 0, 0]

#     # 4. Format for frontend
#     chart_data = [
#         {
#             "time_step": i, # Use index or formatted timestamp
#             "raw_noisy_data": noisy_signal[i],
#             "moving_average": smoothed_signal_ma.iloc[i] if pd.notna(smoothed_signal_ma.iloc[i]) else None,
#             "kalman_filter": smoothed_signal_kf[i],
#         }
#         for i in range(len(noisy_signal))
#     ]

#     return chart_data
# ...existing code...
# In backend/main.py, section 3

@app.post("/anomaly-score/motor")
def get_motor_anomaly_score(data: MotorSequenceData):
    if motor_ae_model is None or motor_ae_scaler is None:
        return {"error": "LSTM Autoencoder model or scaler not loaded."}

    try:
        # 1. Convert input sequence to NumPy array
        input_sequence = np.array(data.sequence)

        # Ensure the sequence has the correct shape (samples, timesteps, features)
        if input_sequence.shape != (SEQUENCE_LEN, 5):
             return {"error": f"Input sequence shape is incorrect. Expected ({SEQUENCE_LEN}, 5), got {input_sequence.shape}"}

        # 2. Scale the data using the loaded scaler
        scaled_sequence = motor_ae_scaler.transform(input_sequence)

        # 3. Reshape for LSTM input (add batch dimension: 1 sequence)
        reshaped_sequence = np.reshape(scaled_sequence, (1, SEQUENCE_LEN, scaled_sequence.shape[1]))

        # 4. Get the model's reconstruction
        reconstruction = motor_ae_model.predict(reshaped_sequence)

        # 5. Calculate the Mean Absolute Error (MAE)
        mae = np.mean(np.abs(reshaped_sequence - reconstruction), axis=(1, 2))
        anomaly_score = mae[0] # Get the scalar value

        # Optional: Define a threshold based on training/validation data
        threshold = 0.05 # Example threshold - tune this!
        is_anomaly = anomaly_score > threshold

        return {
            "anomaly_score": float(anomaly_score),
            "is_anomaly": bool(is_anomaly),
            "threshold": threshold
        }

    except Exception as e:
        print(f"Error during anomaly score calculation: {e}")
        return {"error": "Failed to calculate anomaly score."}
    
# In backend/main.py, near other endpoints

# ==============================================================================
# 9. MAINTENANCE PATTERN MINING (SIMULATED)
# ==============================================================================
@app.get("/maintenance-patterns")
def get_maintenance_patterns():
    """
    Returns a list of pre-defined association rules discovered
    from historical maintenance data (simulated for demo).
    """
    # In a real system, these rules would be generated by Apriori/FP-Growth
    simulated_rules = [
        {
            "id": 1,
            "conditions": ["Pump Vibration > 0.8 mm/s", "Battery RUL < 40 cycles"],
            "outcome": "Motor Failure",
            "confidence": 0.60, # 60% probability increase
            "lift": 2.5 # How much more likely this outcome is given the conditions
        },
        {
            "id": 2,
            "conditions": ["Motor Torque > 65 Nm", "Tool Wear > 200 min"],
            "outcome": "Heat Dissipation Failure (HDF)",
            "confidence": 0.75,
            "lift": 3.1
        },
        {
            "id": 3,
            "conditions": ["Hydraulic Pressure (PS3) < 1.8 bar", "Cooler Efficiency (CE) < 40%"],
            "outcome": "Hydraulic Cooler Failure",
            "confidence": 0.80,
            "lift": 4.0
        }
    ]
    return {"patterns": simulated_rules}

# In backend/main.py, section 3

# ==============================================================================
# 10. REAL-TIME SENSOR DATA ENDPOINTS
# ==============================================================================
# # In backend/main.py
# @app.post("/live-sensor-data")
# async def receive_live_data(data: LiveSensorData):
#     """
#     Receives live sensor data and stores the latest reading.
#     """
#     global latest_reading

#     reading = data.dict()
#     if reading["timestamp"] is None:
#         reading["timestamp"] = time.time() # Add timestamp if not provided

#     latest_reading = reading # Update global variable

#     print(f"Received LIVE data for {data.machine_id}: Temp={data.temperature_c}°C, Hum={data.humidity_percent}%")
#     return {"status": "success", "message": "Data received"}

# --- LIVE DATA ENDPOINTS ---

@app.post("/live-sensor-data")
async def receive_live_data(data: LiveSensorData):
    """
    Receives data from Python Bridge and stores it in memory.
    """
    machine_id = data.machine_id
    
    # Create a storage list for this machine if it doesn't exist
    if machine_id not in recent_readings:
        recent_readings[machine_id] = deque(maxlen=MAX_RECENT_READINGS)
    
    reading = data.dict()
    # Add server-side timestamp if the sensor didn't send one
    if reading["timestamp"] is None:
        reading["timestamp"] = time.time()
        
    # Save the reading to memory
    recent_readings[machine_id].append(reading)
    
    print(f"✅ Stored data for {machine_id}: {reading['temperature_c']}°C")
    return {"status": "success", "message": "Data stored"}

@app.get("/live-data/latest/{machine_id}")
async def get_latest_live_data(machine_id: str):
    """
    Returns the single newest reading for the frontend card.
    """
    if machine_id in recent_readings and len(recent_readings[machine_id]) > 0:
        return recent_readings[machine_id][-1]
    
    # Return N/A if no data has been received yet
    return {"temperature_c": "N/A", "humidity_percent": "N/A", "timestamp": None}

@app.get("/live-data/recent/{machine_id}")
async def get_recent_live_data(machine_id: str):
    """
    Returns the list of recent readings for the small trend chart.
    """
    if machine_id in recent_readings:
        return list(recent_readings[machine_id])
    return []

# @app.get("/live-data/latest/{machine_id}")
# async def get_latest_live_data(machine_id: str):
#     """
#     Returns the single most recently received sensor reading for a machine.
#     (This is what your frontend LiveDataDisplay component will call)
#     """
#     if machine_id in recent_readings and len(recent_readings[machine_id]) > 0:
#         # Return the last item in the list
#         return recent_readings[machine_id][-1]
#     else:
#         return {"temperature_c": "N/A", "humidity_percent": "N/A", "timestamp": None}

# @app.get("/live-data/recent/{machine_id}")
# async def get_recent_live_data(machine_id: str):
#     """
#     Returns a list of the most recent sensor readings for a machine.
#     (This is for the small trend chart in LiveDataDisplay)
#     """
#     if machine_id in recent_readings:
#         return list(recent_readings[machine_id])
#     else:
#         return [] # Return empty list if no data