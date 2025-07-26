import streamlit as st
import pandas as pd
import joblib
from weather_adjustment import adjust_travel_time
from route_optimizer import optimize_route

# Load models and encoders
congestion_model = joblib.load("congestion_model.pkl")
congestion_encoders = joblib.load("congestion_encoders.pkl")
travel_model = joblib.load("travel_time_model.pkl")
accident_model = joblib.load("accident_risk_model.pkl")

# === App Title and Sidebar ===
st.set_page_config(page_title="SmartFlow-AI", layout="wide")
st.title("🚦 SmartFlow-AI Traffic Predictor")
st.sidebar.title("🛠️ SmartFlow-AI Settings")

# === User Input ===
location = st.sidebar.selectbox("Location", congestion_encoders["Location"].classes_)
time = st.sidebar.time_input("Time")
day = st.sidebar.selectbox("Day", congestion_encoders["Day"].classes_)
weather = st.sidebar.selectbox("Weather", congestion_encoders["Weather"].classes_)
event = st.sidebar.selectbox("Event", congestion_encoders["Event"].classes_)
distance = st.sidebar.slider("Distance (km)", 1, 50, 10)
speed = st.sidebar.slider("Average Speed (km/h)", 10, 100, 40)

# === Prediction Logic ===
if st.button("Predict Traffic"):
    hour = time.hour

    # Encode input
    encoded_input = [
        congestion_encoders["Location"].transform([location])[0],
        hour,
        congestion_encoders["Day"].transform([day])[0],
        congestion_encoders["Weather"].transform([weather])[0],
        congestion_encoders["Event"].transform([event])[0]
    ]
    congestion_pred = congestion_model.predict([encoded_input])[0]
    congestion_label = congestion_encoders["Congestion"].inverse_transform([congestion_pred])[0]

    # Travel time
    travel_input = pd.DataFrame([{
        "distance_km": distance,
        "avg_speed_kmph": speed
    }])
    base_time = travel_model.predict(travel_input)[0]
    adjusted_time = adjust_travel_time(base_time, weather)

    # Accident risk
    accident_input = pd.DataFrame([{
        "road_type": 2,
        "weather": {"clear": 0, "rain": 1, "fog": 2, "cloudy": 3, "sunny": 4}.get(weather.lower(), 0),
        "time_of_day": hour
    }])
    accident_risk = accident_model.predict(accident_input)[0]

    # Route optimization
    routes = [
        {"route": "Route A", "travel_time": adjusted_time, "accident_risk": accident_risk},
        {"route": "Route B", "travel_time": adjusted_time * 1.1, "accident_risk": accident_risk * 0.9},
        {"route": "Route C", "travel_time": adjusted_time * 0.95, "accident_risk": accident_risk * 1.2}
    ]
    best_route = optimize_route(routes)

    # === Output ===
    st.markdown("### 🧠 Traffic Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Congestion", congestion_label)

    with col2:
        st.metric("Travel Time (min)", f"{adjusted_time:.2f}")

    with col3:
        st.metric("Accident Risk", accident_risk)

    st.markdown("### 🛣️ Route Suggestion")
    st.success(f"Best Route: **{best_route['route']}**")

st.markdown("---")
st.markdown("© 2025 SmartFlow-AI | Built by Mr. Surgeon 🚀")
