import pandas as pd
import pickle
from route_optimizer import optimize_route
from weather_adjustment import adjust_travel_time

# Load models
with open('congestion_model.pkl', 'rb') as f:
    congestion_model = pickle.load(f)

with open('travel_time_model.pkl', 'rb') as f:
    travel_model = pickle.load(f)

with open('accident_risk_model.pkl', 'rb') as f:
    accident_model = pickle.load(f)

# === INPUT SECTION ===
hour = int(input("Enter hour of day (0–23): "))
day_of_week = int(input("Enter day of week (0=Mon, 6=Sun): "))
location_id = int(input("Enter location ID: "))

distance_km = float(input("Enter route distance (km): "))
avg_speed_kmph = float(input("Enter average speed (km/h): "))

road_type = int(input("Enter road type (1=Highway, 2=Urban, 3=Rural): "))
weather = input("Enter weather condition (clear, rain, fog): ")
time_of_day = int(input("Enter time of day (0–23): "))

# === PREDICTIONS ===
# Congestion
congestion_input = pd.DataFrame([{
    'hour': hour,
    'day_of_week': day_of_week,
    'location_id': location_id
}])
congestion_level = congestion_model.predict(congestion_input)[0]

# Travel time
travel_input = pd.DataFrame([{
    'distance_km': distance_km,
    'avg_speed_kmph': avg_speed_kmph
}])
base_travel_time = travel_model.predict(travel_input)[0]
adjusted_time = adjust_travel_time(base_travel_time, weather)

# Accident risk
accident_input = pd.DataFrame([{
    'road_type': road_type,
    'weather': {'clear': 0, 'rain': 1, 'fog': 2}.get(weather, 0),
    'time_of_day': time_of_day
}])
accident_risk = accident_model.predict(accident_input)[0]

# Route optimization
routes = [
    {'route': 'Route A', 'travel_time': adjusted_time, 'accident_risk': accident_risk},
    {'route': 'Route B', 'travel_time': adjusted_time * 1.1, 'accident_risk': accident_risk * 0.9},
    {'route': 'Route C', 'travel_time': adjusted_time * 0.95, 'accident_risk': accident_risk * 1.2}
]
best_route = optimize_route(routes)

# === OUTPUT ===
print("\n--- SmartFlow-AI Dashboard ---")
print(f"Predicted congestion level: {congestion_level}")
print(f"Base travel time: {base_travel_time:.2f} minutes")
print(f"Adjusted travel time (weather): {adjusted_time:.2f} minutes")
print(f"Predicted accident risk level: {accident_risk}")
print(f"Best route suggestion: {best_route['route']} (score: {best_route['travel_time'] + best_route['accident_risk'] * 10:.2f})")
