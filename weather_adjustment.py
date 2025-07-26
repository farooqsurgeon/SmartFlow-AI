def adjust_travel_time(base_time, weather_condition):
    if weather_condition == 'rain':
        return base_time * 1.2
    elif weather_condition == 'fog':
        return base_time * 1.3
    elif weather_condition == 'clear':
        return base_time
    else:
        return base_time * 1.1

# Example
base_time = 30
weather = 'rain'
adjusted = adjust_travel_time(base_time, weather)
print(f"Adjusted travel time for {weather}: {adjusted:.2f} minutes")
