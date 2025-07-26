import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("accident_risk_model.pkl")
encoders = joblib.load("accident_risk_encoders.pkl")

# Sample input
sample = pd.DataFrame([{
    "Location": "Wuse",
    "Time": "08:00",
    "Day": "Monday",
    "Weather": "Rainy",
    "Event": "Yes"
}])

# Encode input
for col in sample.columns:
    sample[col] = encoders[col].transform(sample[col])

# Predict
prediction_encoded = model.predict(sample)[0]
prediction_label = encoders["AccidentRisk"].inverse_transform([prediction_encoded])[0]

print(f"🚨 Predicted accident risk level: {prediction_label}")
