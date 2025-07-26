def optimize_route(routes):
    best_route = min(routes, key=lambda r: r['travel_time'] + r['accident_risk'] * 10)
    return best_route

# Example routes
routes = [
    {'route': 'A', 'travel_time': 30, 'accident_risk': 0.2},
    {'route': 'B', 'travel_time': 25, 'accident_risk': 0.4},
    {'route': 'C', 'travel_time': 35, 'accident_risk': 0.1}
]

best = optimize_route(routes)
print(f"Best route: {best['route']} with score: {best['travel_time'] + best['accident_risk'] * 10}")
