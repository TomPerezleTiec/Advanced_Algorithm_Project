import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
import time as sleep_module
import re
from deap import base, creator, tools, algorithms
import random
from datetime import datetime, timedelta, time as datetime_time

# Load client data from an Excel file
file_path = 'clients_depot_barcelone.xlsx'
df = pd.read_excel(file_path)

def time_to_minutes(t):
    """Convert various time formats (datetime.time, int, float, or HH:MM string) to minutes since midnight.
    Returns the time in minutes as an integer. Returns 0 if format is unrecognized."""
    if isinstance(t, datetime_time):
        return t.hour * 60 + t.minute
    elif isinstance(t, (int, float)):
        return int(t)
    elif isinstance(t, str):
        try:
            match = re.match(r"(\d{1,2}):(\d{2})", t)
            if match:
                hours, minutes = map(int, match.groups())
                return hours * 60 + minutes
            return int(t)
        except ValueError:
            print(f"Unrecognized time format: {t}. Defaulting to 0 minutes.")
            return 0
    return 0

depot_address = df['Adresse'].iloc[0]
client_addresses = df['Adresse'].iloc[1:].tolist()

# Get client time windows, excluding depot; add depot time window separately
time_windows = [(time_to_minutes(df['Ouverture'].iloc[i]), time_to_minutes(df['Fermeture'].iloc[i])) for i in range(1, len(df))]
depot_time_window = (0, 1440)

# Initialize geocoder and get coordinates for depot and clients
geolocator = Nominatim(user_agent="vrp_app")
locations = []
addresses = [depot_address] + client_addresses
for address in addresses:
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            locations.append((location.latitude, location.longitude))
        else:
            print(f"Address not found: {address}")
            locations.append((None, None))
    except Exception as e:
        print(f"Error with address {address}: {e}")
        locations.append((None, None))
    sleep_module.sleep(1)

# Filter valid addresses
locations = [coords for coords in locations if coords != (None, None)]
depot_location = locations[0]
client_locations = locations[1:]

# Load Barcelona graph
graph = ox.graph_from_place('Barcelona, Spain', network_type='drive')
graph = graph.to_undirected().subgraph(max(nx.connected_components(graph.to_undirected()), key=len)).copy()

def get_nearest_nodes(graph, locations):
    """Find the nearest nodes in the graph to each location.
    Returns a list of node IDs corresponding to each input location."""
    return [ox.distance.nearest_nodes(graph, loc[1], loc[0]) for loc in locations]

nodes = get_nearest_nodes(graph, client_locations)
depot_node = get_nearest_nodes(graph, [depot_location])[0]
nodes_with_depot = [depot_node] + nodes

def parse_speed(speed_str):
    """Parse speed from string format or use default if unavailable.
    Returns speed as a float."""
    if isinstance(speed_str, list):
        speed_str = speed_str[0]
    if isinstance(speed_str, str):
        match = re.match(r"(\d+)", speed_str)
        if match:
            return float(match.group(1))
    return 50.0

def calculate_neighbor_times(graph):
    """Calculate travel time between neighboring nodes in the graph based on distance and speed.
    Returns a dictionary with edge travel times."""
    neighbor_times = {}
    for u, v, data in graph.edges(data=True):
        distance = data.get('length', 0) / 1000
        speed = parse_speed(data.get('maxspeed', '50'))
        travel_time = (distance / speed) * 60
        neighbor_times[(u, v)] = travel_time
    return neighbor_times

neighbor_times = calculate_neighbor_times(graph)

def calculate_time_matrix(graph, nodes):
    """Compute travel time and distance matrices between all pairs of specified nodes.
    Returns two matrices: time_matrix with travel times, and distance_matrix with distances."""
    time_matrix = []
    distance_matrix = []
    for node1 in nodes:
        time_row = []
        distance_row = []
        for node2 in nodes:
            if node1 == node2:
                time_row.append(0)
                distance_row.append(0)
            else:
                try:
                    path = nx.shortest_path(graph, node1, node2, weight='length')
                    total_time = 0
                    total_distance = 0
                    for u, v in zip(path[:-1], path[1:]):
                        edge_data = graph.get_edge_data(u, v)
                        edge = edge_data[0] if isinstance(edge_data, dict) else edge_data
                        distance = edge.get('length', 0) / 1000
                        speed = parse_speed(edge.get('maxspeed', '50'))
                        travel_time = distance / speed * 60
                        total_time += travel_time
                        total_distance += distance
                    time_row.append(total_time)
                    distance_row.append(total_distance)
                except nx.NetworkXNoPath:
                    time_row.append(float('inf'))
                    distance_row.append(float('inf'))
        time_matrix.append(time_row)
        distance_matrix.append(distance_row)
    return time_matrix, distance_matrix

time_matrix, distance_matrix = calculate_time_matrix(graph, nodes_with_depot)

# Convert all time windows to minutes
time_windows = [
    (time_to_minutes(open_time), time_to_minutes(close_time))
    for open_time, close_time in time_windows
]

# DEAP VRPTW Implementation
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(1, len(nodes)), len(nodes) - 1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

population = toolbox.population(n=100)
for ind in population:
    assert len(ind) == len(nodes) - 1, "Individual has incorrect length!"

def evaluate(individual):
    """Evaluate an individual's route, penalizing violations of time windows.
    Returns the route's total travel time with penalties."""
    route = [0] + individual + [0]
    total_time = 0
    last_departure = 0
    penalty = 0
    
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        travel_time = time_matrix[from_node][to_node]
        arrival_time = last_departure + travel_time
        window_open, window_close = time_windows[to_node]
        
        if arrival_time < window_open:
            waiting_time = window_open - arrival_time
            arrival_time = window_open
            penalty += waiting_time
        elif arrival_time > window_close:
            penalty += (arrival_time - window_close) * 10  # Penalty for arriving late
        
        total_time += travel_time
        last_departure = arrival_time

    return total_time + penalty,

toolbox.register("evaluate", evaluate)

def custom_cxOrdered(ind1, ind2):
    """Perform ordered crossover with length checks for VRP individuals.
    Returns two offspring individuals with mixed parent genes."""
    size = len(ind1)
    if len(ind1) != len(ind2):
        raise ValueError("Individuals must be of the same length")

    start, end = sorted(random.sample(range(size), 2))

    child1, child2 = toolbox.clone(ind1), toolbox.clone(ind2)
    child1[start:end + 1] = ind2[start:end + 1]
    
    pos = end + 1
    for gene in ind1:
        if gene not in child1[start:end + 1]:
            if pos >= size:
                pos = 0
            while start <= pos <= end:
                pos += 1
                if pos >= size:
                    pos = 0
            child1[pos] = gene
            pos += 1
    
    child2[start:end + 1] = ind1[start:end + 1]
    pos = end + 1
    for gene in ind2:
        if gene not in child2[start:end + 1]:
            if pos >= size:
                pos = 0
            while start <= pos <= end:
                pos += 1
                if pos >= size:
                    pos = 0
            child2[pos] = gene
            pos += 1

    return child1, child2


# Register the new custom crossover in the DEAP toolbox
toolbox.register("mate", custom_cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA with validation after mating
population_size = 100
generations = 50

# Evaluation of each generation and selection
for gen in range(generations):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
    
    # Evaluate only valid individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Filter offspring that violate time windows
    valid_offspring = [ind for ind in offspring if ind.fitness.values[0] < float('inf')]
    
    # If no valid offspring, penalize heavily to encourage mutations towards valid solutions
    if not valid_offspring:
        for ind in offspring:
            ind.fitness.values = (float('inf'),)
        valid_offspring = population  # Reuse current population as backup
    
    population = toolbox.select(valid_offspring + population, k=population_size)

# Select the best solution after the loop
best_individual = tools.selBest(population, k=1)[0]
best_route = [0] + best_individual + [0]

# Display route details
total_time = 0
print("\nOptimized route details with time windows:")
for i in range(len(best_route) - 1):
    from_node = best_route[i]
    to_node = best_route[i + 1]
    travel_time = time_matrix[from_node][to_node]
    print(f"From {from_node} to {to_node}: Travel time = {travel_time:.2f} min")
    total_time += travel_time

print(f"\nTotal optimized time: {total_time:.2f} minutes")

# Create a map with the optimized route
map_center = depot_location
carte = folium.Map(location=map_center, zoom_start=13)

# Depot marker
folium.Marker(depot_location, popup="Depot", icon=folium.Icon(color="black")).add_to(carte)

def minutes_to_time(minutes):
    """Convert minutes to HH:MM format for readability. Returns a formatted time string."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02}:{mins:02}"

def plot_routes_with_time_details(solution, graph, nodes, time_windows, time_matrix):
    """Plot the route on a map with markers at each client location, displaying time details.
    Adds the complete route and each client’s time window, arrival, and departure times."""
    route_coords = []
    cumulative_time = 480  # Start cumulative time at 08:00 (480 minutes)

    # Plot the route
    for i in range(len(solution) - 1):
        try:
            path = nx.shortest_path(graph, nodes[solution[i]], nodes[solution[i + 1]], weight='length')
            coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in path]
            route_coords.extend(coords)
        except nx.NetworkXNoPath:
            print(f"No path between {solution[i]} and {solution[i + 1]}")
            continue
    folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=0.8).add_to(carte)

    # Initial marker for the depot (starting point)
    depot_location = (graph.nodes[nodes[solution[0]]]['y'], graph.nodes[nodes[solution[0]]]['x'])
    depot_departure_time = minutes_to_time(cumulative_time)
    folium.Marker(
        depot_location,
        popup=f"Depot (Client 0)<br>Departure Time: {depot_departure_time}",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(carte)
    
    print(f"Depot (Client 0): Location {depot_location}")
    print(f"  Departure Time: {depot_departure_time}\n")

    # Place markers for each client with time details
    for idx in range(1, len(solution) - 1):
        location = (graph.nodes[nodes[solution[idx]]]['y'], graph.nodes[nodes[solution[idx]]]['x'])
        time_window = time_windows[solution[idx]]
        travel_time = time_matrix[solution[idx - 1]][solution[idx]]
        arrival_time = cumulative_time + travel_time
        opening_time, closing_time = time_window
        waiting_time = max(0, opening_time - arrival_time)
        departure_time = arrival_time + waiting_time
        arrival_time_str = minutes_to_time(arrival_time)
        departure_time_str = minutes_to_time(departure_time)
        waiting_time_str = f"{int(waiting_time)} min" if waiting_time > 0 else "No waiting"
        cumulative_time = departure_time

        popup_text = (
            f"Client {solution[idx]}<br>"
            f"Time Window: {minutes_to_time(opening_time)} - {minutes_to_time(closing_time)}<br>"
            f"Arrival Time: {arrival_time_str}<br>"
            f"Departure Time: {departure_time_str}<br>"
            f"Waiting Time: {waiting_time_str}"
        )
        
        print(f"Client {solution[idx]}: Location {location}")
        print(f"  Time Window: {minutes_to_time(opening_time)} - {minutes_to_time(closing_time)}")
        print(f"  Arrival Time: {arrival_time_str}")
        print(f"  Departure Time: {departure_time_str}")
        print(f"  Waiting Time: {waiting_time_str}\n")

        folium.Marker(
            location,
            popup=popup_text,
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(carte)

    last_client_id = solution[-2]
    final_travel_time = time_matrix[last_client_id][solution[-1]]
    depot_arrival_time = cumulative_time + final_travel_time
    depot_arrival_time_str = minutes_to_time(depot_arrival_time)

    print(f"Return to Depot (Client 0): Location {depot_location}")
    print(f"  Arrival Time: {depot_arrival_time_str}\n")

    folium.Marker(
        depot_location,
        popup=f"Depot (Client 0)<br>Arrival Time: {depot_arrival_time_str}",
        icon=folium.Icon(color="black", icon="info-sign")
    ).add_to(carte)

graph_route = best_route

plot_routes_with_time_details(graph_route, graph, nodes, time_windows, time_matrix)
carte.save("map_vrptw_barcelona.html")
print("VRPTW map with client locations and time details saved as 'map_vrptw_barcelona.html'")

def calculate_shortest_path_metrics(graph, nodes):
    """Calculate minimum travel distance and time using shortest paths for consecutive nodes.
    Returns the total shortest distance in km and time in minutes."""
    total_shortest_distance = 0
    total_shortest_time = 0
    for i in range(len(nodes) - 1):
        node_start = nodes[i]
        node_end = nodes[i + 1]
        try:
            path = nx.shortest_path(graph, node_start, node_end, weight='length')
            path_edges = ox.utils_graph.get_route_edge_attributes(graph, path, 'length')
            segment_distance = sum(path_edges)
            total_shortest_distance += segment_distance
            
            segment_time = 0
            for u, v, edge_data in graph.subgraph(path).edges(data=True):
                edge_length = edge_data.get('length', 0) / 1000
                speed = parse_speed(edge_data.get('maxspeed', '50'))
                travel_time = (edge_length / speed) * 60
                segment_time += travel_time
            total_shortest_time += segment_time
        except nx.NetworkXNoPath:
            print(f"No path found between {node_start} and {node_end}.")
            total_shortest_distance += float('inf')
            total_shortest_time += float('inf')
    
    return total_shortest_distance / 1000, total_shortest_time

min_nodes_with_depot = [depot_node] + nodes + [depot_node]
shortest_distance, shortest_time = calculate_shortest_path_metrics(graph, min_nodes_with_depot)

optimized_distance = sum(
    distance_matrix[best_route[i]][best_route[i + 1]] for i in range(len(best_route) - 1)
)
optimized_time = sum(
    time_matrix[best_route[i]][best_route[i + 1]] for i in range(len(best_route) - 1)
)

print(f"Minimal theoretical distance (shortest_path): {shortest_distance:.2f} km")
print(f"Minimal theoretical time (shortest_path): {shortest_time:.2f} minutes\n")
print(f"Distance obtained by genetic algorithm: {optimized_distance:.2f} km")
print(f"Time obtained by genetic algorithm: {optimized_time:.2f} minutes\n")

if optimized_distance <= shortest_distance and optimized_time <= shortest_time:
    print("The genetic algorithm found a route matching or better than the minimal theoretical values.")
else:
    print("The genetic algorithm did not reach the minimal theoretical values; further tuning may help.")




