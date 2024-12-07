from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import threading
import uvicorn
import streamlit as st
import requests
import time  # Added for delay

# FastAPI setup
app = FastAPI()

# CORS setup (optional, for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler (Make sure this model is for cardiovascular disease prediction)
model = joblib.load("random_forest_model_final.joblib")
scaler = joblib.load("scaler.joblib")

# Define input schema for FastAPI
class InputData(BaseModel):
    AGE: float
    GENDER: float
    HEIGHT: float
    WEIGHT: float
    AP_HIGH: float
    AP_LOW: float
    CHOLESTEROL: float
    GLUCOSE: float
    SMOKE: float
    ALCOHOL: float
    PHYSICAL_ACTIVITY: float

@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Specify correct column order
    columns_order = [
        'AGE', 'GENDER', 'HEIGHT', 'WEIGHT', 'AP_HIGH', 'AP_LOW',
        'CHOLESTEROL', 'GLUCOSE', 'SMOKE', 'ALCOHOL', 'PHYSICAL_ACTIVITY'
    ]
    input_df = input_df[columns_order]

    # Scale data and make prediction
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return {"prediction": int(prediction[0])}

# Function to run FastAPI
def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8001)

# Streamlit UI
def run_streamlit():
    # Update the title for cardiovascular disease prediction
    st.title("Cardiovascular Disease Prediction")
    st.write("Enter patient details below:")

    # Collect user inputs
    age = st.number_input("Age", min_value=0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 0 if gender == "Male" else 1
    height = st.number_input("Height (cm)", min_value=0)
    weight = st.number_input("Weight (kg)", min_value=0)
    ap_high = st.number_input("Systolic BP", min_value=0)
    ap_low = st.number_input("Diastolic BP", min_value=0)
    cholesterol = st.selectbox("Cholesterol Level", ["Normal", "Above Normal", "Well Above Normal"])
    cholesterol = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}[cholesterol]
    glucose = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
    glucose = {"Normal": 0, "Above Normal": 1, "Well Above Normal": 2}[glucose]
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    smoke = 0 if smoke == "No" else 1
    alcohol = st.selectbox("Alcohol Intake", ["No", "Yes"])
    alcohol = 0 if alcohol == "No" else 1
    physical_activity = st.selectbox("Physical Activity", ["No", "Yes"])
    physical_activity = 0 if physical_activity == "No" else 1

    # Prediction button
    if st.button("Predict"):
        # Prepare input data
        input_data = {
            "AGE": age,
            "GENDER": gender,
            "HEIGHT": height,
            "WEIGHT": weight,
            "AP_HIGH": ap_high,
            "AP_LOW": ap_low,
            "CHOLESTEROL": cholesterol,
            "GLUCOSE": glucose,
            "SMOKE": smoke,
            "ALCOHOL": alcohol,
            "PHYSICAL_ACTIVITY": physical_activity,
        }
        
        # Send data to FastAPI backend
        try:
            response = requests.post("http://127.0.0.1:8001/predict", json=input_data)
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Prediction Result: {prediction}")
            else:
                st.error("Error in prediction. Check backend.")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

# Run both FastAPI and Streamlit concurrently
if __name__ == "__main__":
    # Start FastAPI in a background thread
    fastapi_thread = threading.Thread(target=run_fastapi)
    fastapi_thread.start()

    # Wait for FastAPI to initialize (added delay)
    time.sleep(3)  # Allow time for FastAPI to start up

    # Now start Streamlit UI
    run_streamlit()
