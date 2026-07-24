import pandas as pd
import joblib
from config import MODEL_FILE, SCALER_FILE
from riskscore import calculate_probability, calculate_riskscore, classify_risk

def load_artifacts():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

def preprocess_input(customer_data, scaler):
    scaled_data = scaler.transform(customer_data)
    return scaler

