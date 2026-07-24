import joblib
from config import MODEL_FILE

def load_model():
    model = joblib.load(MODEL_FILE)
    return model

def calculate_probability(model, customer_data):
    probability = model.predict_proba(customer_data)[0][1]
    return probability

def calculate_riskscore(probability):
    score = round(probability * 100, 2)
    return score

def classify_risk(score):
    if score < 30:
        return "low risk"
    elif score < 70:
        return "medium risk"
    return "high risk"

def generate_risk_report(customer_data):
    model = load_model()
    probability = calculate_probability(model, customer_data)
    score = calculate_riskscore(probability)
    category = classify_risk(score)
    return {
        "Probability": probability,
        "Risk Score": score,
        "Category": category
    }
     