import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Sample data
data = pd.DataFrame([
    {"Location": "Wuse", "Time": "08:00", "Day": "Monday", "Weather": "Rainy", "Event": "Yes", "AccidentRisk": "High"},
    {"Location": "Garki", "Time": "14:00", "Day": "Tuesday", "Weather": "Sunny", "Event": "No", "AccidentRisk": "Low"},
    {"Location": "Maitama", "Time": "18:00", "Day": "Friday", "Weather": "Cloudy", "Event": "Yes", "AccidentRisk": "Medium"},
    {"Location": "Wuse", "Time": "07:00", "Day": "Wednesday", "Weather": "Rainy", "Event": "No", "AccidentRisk": "High"},
])

# Encode categorical features
encoders = {}
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

# Train model
X = data.drop("AccidentRisk", axis=1)
y = data["AccidentRisk"]
model = RandomForestClassifier()
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "accident_risk_model.pkl")
joblib.dump(encoders, "accident_risk_encoders.pkl")

print("✅ Accident risk model trained and saved.")
