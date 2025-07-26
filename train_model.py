import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("model.pkl")
le_location = joblib.load("le_location.pkl")
le_time = joblib.load("le_time.pkl")
le_day = joblib.load("le_day.pkl")
le_weather = joblib.load("le_weather.pkl")
le_event = joblib.load("le_event.pkl")
le_congestion = joblib.load("le_congestion.pkl")

# Sample input
sample = pd.DataFrame([{
    "Location": "Wuse",
    "Time": "08:00",
    "Day": "Monday",
    "Weather": "Rainy",
    "Event": "Yes"
}])

# Encode features
sample["Location"] = le_location.transform(sample["Location"])
sample["Time"] = le_time.transform(sample["Time"])
sample["Day"] = le_day.transform(sample["Day"])
sample["Weather"] = le_weather.transform(sample["Weather"])
sample["Event"] = le_event.transform(sample["Event"])

# Predict
prediction_encoded = model.predict(sample)[0]
prediction_label = le_congestion.inverse_transform([prediction_encoded])[0]

print(f"🚦 Predicted congestion level: {prediction_label}")
