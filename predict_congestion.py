import pickle
import pandas as pd

# Load model and encoders
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Sample input
sample = pd.DataFrame([{
    "Location": "Wuse",
    "Time": "08:00",
    "Day": "Monday",
    "Weather": "Rainy",
    "Event": "Yes"
}])

# Encode features
for col in sample.columns:
    sample[col] = encoders[col].transform(sample[col])

# Predict
prediction = model.predict(sample)[0]
print(f"🚦 Predicted congestion level: {prediction}")
