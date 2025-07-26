import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Sample data
data = pd.DataFrame([
    {"Start": "Wuse", "End": "Garki", "Time": "08:00", "Day": "Monday", "Weather": "Rainy", "Event": "Yes", "TravelTime": 25},
    {"Start": "Garki", "End": "Maitama", "Time": "14:00", "Day": "Tuesday", "Weather": "Sunny", "Event": "No", "TravelTime": 15},
    {"Start": "Wuse", "End": "Maitama", "Time": "18:00", "Day": "Friday", "Weather": "Cloudy", "Event": "Yes", "TravelTime": 30},
    {"Start": "Garki", "End": "Wuse", "Time": "07:00", "Day": "Wednesday", "Weather": "Rainy", "Event": "No", "TravelTime": 20},
])

# Encode categorical features
encoders = {}
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

# Train model
X = data.drop("TravelTime", axis=1)
y = data["TravelTime"]
model = RandomForestRegressor()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "travel_time_model.pkl")
joblib.dump(encoders, "travel_time_encoders.pkl")

print("✅ Travel time model trained and saved.")
