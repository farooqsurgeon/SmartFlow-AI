import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("travel_time_model.pkl")
encoders = joblib.load("travel_time_encoders.pkl")

# Sample input
sample = pd.DataFrame([{
    "Start": "Wuse",
    "End": "Garki",
    "Time": "08:00",
    "Day": "Monday",
    "Weather": "Rainy",
    "Event": "Yes"
}])

# Encode input
for col in sample.columns:
    sample[col] = encoders[col].transform(sample[col])

# Predict
prediction = model.predict(sample)[0]
print(f"🕒 Estimated travel time: {round(prediction)} minutes")
