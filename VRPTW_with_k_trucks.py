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
import matplotlib.colors as mcolors
import math
import tkinter as tk
from tkinter import messagebox, Label
import webbrowser
from sklearn.cluster import KMeans
from adress_generator import create_excel_with_addresses

MAP_FILE = "map_vrptw_barcelona_2.html"

def start_vrp_program():
    """Launches the VRP program, initializes settings, and loads client data."""
    try:
        num_trucks = int(num_trucks_entry.get())
        root.destroy()
        loading_window = tk.Tk()
        loading_window.title("Loading...")
        loading_label = Label(loading_window, text="Calculating VRP solution, please wait...")
        loading_label.pack(padx=20, pady=20)
        loading_window.update()
        file_path = 'clients_depot_barcelone.xlsx'
        df = pd.read_excel(file_path)
        print("Columns in df:", df.columns)
        min_generations = 12
        max_generations = 100
        generations = max(max_generations, min_generations)
        depot_address = df['Adresse'].iloc[0]
        client_addresses = df['Adresse'].iloc[1:].tolist()
        time_windows = [(df['Ouverture'].iloc[i], df['Fermeture'].iloc[i]) for i in range(1, len(df))]
        time_windows.insert(0, (0, 1440))
        geolocator = Nominatim(user_agent="vrp_app")
        locations = []
        addresses = [depot_address] + client_addresses
        for address in addresses:
            try:
                location = geolocator.geocode(address, timeout=10)
                if location:
                    locations.append((location.latitude, location.longitude))
                else:
                    print(f"Adresse non trouvée: {address}")
                    locations.append((None, None))
            except Exception as e:
                print(f"Erreur pour l'adresse {address}: {e}")
                locations.append((None, None))
            sleep_module.sleep(1)
        locations = [coords for coords in locations if coords != (None, None)]
        depot_location = locations[0]
        graph = ox.graph_from_place('Barcelona, Spain', network_type='drive')
        graph = graph.to_undirected().subgraph(max(nx.connected_components(graph.to_undirected()), key=len)).copy()

        def get_nearest_nodes(graph, locations):
            """Finds the nearest nodes in the graph to the specified locations.
            Returns a list of nearest node IDs for each location."""
            return [ox.distance.nearest_nodes(graph, loc[1], loc[0]) for loc in locations]

        nodes = get_nearest_nodes(graph, locations)

        def parse_speed(speed_str):
            """Parses speed information from string to a numeric value.
            Returns speed as float or default value if not found."""
            if isinstance(speed_str, list):
                speed_str = speed_str[0]
            if isinstance(speed_str, str):
                match = re.match(r"(\d+)", speed_str)
                if match:
                    return float(match.group(1))
            return 50.0

        def calculate_neighbor_times(graph):
            """Calculates travel times between direct neighbors based on speed and distance.
            Returns a dictionary with travel times for each edge."""
            neighbor_times = {}
            for u, v, data in graph.edges(data=True):
                distance = data.get('length', 0) / 1000
                speed = parse_speed(data.get('maxspeed', '50'))
                travel_time = (distance / speed) * 60
                neighbor_times[(u, v)] = travel_time
            return neighbor_times

        neighbor_times = calculate_neighbor_times(graph)

        def calculate_time_matrix(graph, nodes):
            """Calculates travel time and distance matrices for the specified nodes.
            Returns matrices containing travel times and distances between each pair of nodes."""
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

        time_matrix, distance_matrix = calculate_time_matrix(graph, nodes)
        
        def time_to_minutes(t):
            """Convert a datetime.time object to minutes since midnight.
            Returns the time in minutes as an integer."""
            return t.hour * 60 + t.minute

        def convert_to_minutes(time_str):
            """Convert a time string in 'HH:MM' format to minutes since midnight.
            Returns the time in minutes as an integer."""
            try:
                time_obj = datetime.strptime(time_str, '%H:%M')
                return time_obj.hour * 60 + time_obj.minute
            except ValueError:
                raise ValueError(f"Invalid time format: '{time_str}'")

        time_windows = [
            (convert_to_minutes(df['Ouverture'].iloc[i]), convert_to_minutes(df['Fermeture'].iloc[i]))
            for i in range(len(df))
        ]

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(1, len(nodes)), len(nodes) - 1)

        def init_individual_with_trucks():
            """Creates an initial individual solution that includes all clients, with separators between trucks.
            Returns a DEAP Individual with clients and separators."""
            individual = random.sample(range(1, len(nodes)), len(nodes) - 1)
            separators = [-1] * (num_trucks - 1)
            for i in range(num_trucks - 1):
                pos = random.randint(1, len(individual))
                individual.insert(pos, -1)
            return creator.Individual(individual + separators)

        toolbox.register("individual", init_individual_with_trucks)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        def init_individual_time_window_clusters():
            """Initializes an individual by clustering clients based on time windows and locations.
            Returns a DEAP Individual arranged with clustered clients and separators."""
            client_data = [(locations[i][0], locations[i][1], time_windows[i][0]) for i in range(1, len(nodes))]
            kmeans = KMeans(n_clusters=num_trucks).fit(client_data)
            clusters = kmeans.labels_
            individual = []
            for cluster_id in range(num_trucks):
                clients_in_cluster = [i + 1 for i, c in enumerate(clusters) if c == cluster_id]
                individual.extend(clients_in_cluster + [-1])
            return creator.Individual(individual[:-1])

        toolbox.register("individual", init_individual_time_window_clusters)

        def evaluate_time_window_balanced(individual):
            """Evaluates a route solution considering time windows, penalties, and truck constraints.
            Returns a tuple with total time and penalties for the evaluated individual."""
            routes = []
            route = [0]
            served_clients = set()
            visited_clients = set()
            total_time = 0
            total_penalty = 0
            for gene in individual:
                if gene == -1:
                    route.append(0)
                    if len(route) > 2:
                        routes.append(route)
                    route = [0]
                else:
                    route.append(gene)
                    served_clients.add(gene)
                    if gene in visited_clients:
                        total_penalty += 5000
                    else:
                        visited_clients.add(gene)
            route.append(0)
            if len(route) > 2:
                routes.append(route)

            unused_trucks_penalty = 3000 * (num_trucks - len(routes))
            total_penalty += unused_trucks_penalty

            all_clients = set(range(1, len(nodes)))
            missed_clients = all_clients - served_clients
            total_penalty += 1000 * len(missed_clients)

            max_clients_per_truck = len(nodes) // num_trucks
            for route in routes:
                if len(route) - 2 > max_clients_per_truck:
                    total_penalty += 2000
                last_departure = 0
                for i in range(len(route) - 1):
                    from_node = route[i]
                    to_node = route[i + 1]
                    travel_time = time_matrix[from_node][to_node]
                    arrival_time = last_departure + travel_time
                    window_open, window_close = time_windows[to_node]
                    if arrival_time < window_open:
                        waiting_time = window_open - arrival_time
                        arrival_time = window_open
                        total_penalty += waiting_time
                    elif arrival_time > window_close:
                        total_penalty += (arrival_time - window_close) * 1000
                    total_time += travel_time
                    last_departure = arrival_time
            return total_time + total_penalty,

        toolbox.register("evaluate", evaluate_time_window_balanced)

        def custom_cxOrdered_with_time_window(ind1, ind2):
            """Performs an ordered crossover with time window constraints.
            Returns two offspring individuals with valid truck separation and client sequences."""
            temp1 = [gene for gene in ind1 if gene != -1]
            temp2 = [gene for gene in ind2 if gene != -1]
            start, end = sorted(random.sample(range(len(temp1)), 2))
            child1_genes = temp1[:start] + [gene for gene in temp2[start:end + 1] if gene not in temp1[:start]]
            child1_genes += [gene for gene in temp1[end + 1:] if gene not in child1_genes]
            child2_genes = temp2[:start] + [gene for gene in temp1[start:end + 1] if gene not in temp2[:start]]
            child2_genes += [gene for gene in temp2[end + 1:] if gene not in child2_genes]
            for _ in range(num_trucks - 1):
                pos1 = random.randint(0, len(child1_genes))
                child1_genes.insert(pos1, -1)
                pos2 = random.randint(0, len(child2_genes))
                child2_genes.insert(pos2, -1)
            return creator.Individual(child1_genes), creator.Individual(child2_genes)

        def mutate_with_time_window_constraint(individual):
            """Mutates an individual by shuffling clients within each truck's route.
            Returns a mutated individual that maintains truck separation and route validity."""
            routes = []
            route = []
            for gene in individual:
                if gene == -1:
                    if route:
                        random.shuffle(route)
                        routes.append(route)
                    routes.append([-1])
                    route = []
                else:
                    route.append(gene)
            if route:
                random.shuffle(route)
                routes.append(route)
            mutated_individual = [gene for route in routes for gene in route]
            return creator.Individual(mutated_individual),

        toolbox.register("mate", custom_cxOrdered_with_time_window)
        toolbox.register("mutate", mutate_with_time_window_constraint)

        toolbox.register("select", tools.selTournament, tournsize=3)

        population_size = 100
        population = toolbox.population(n=population_size)

        for gen in range(generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
            for ind in offspring:
                if ind.count(-1) != num_trucks - 1:
                    print(f"Generation {gen}")
                    continue
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            population = toolbox.select(offspring + population, k=population_size)

        best_individual = tools.selBest(population, k=1)[0]
        best_routes = []
        route = [0]
        for gene in best_individual:
            if gene == -1:
                route.append(0)
                best_routes.append(route)
                route = [0]
            else:
                route.append(gene)
        route.append(0)
        best_routes.append(route)

        for truck_idx, route in enumerate(best_routes):
            total_time = 0
            print(f"\nOptimized route details for truck {truck_idx + 1} with time windows:")
            for i in range(len(route) - 1):
                from_node = route[i]
                to_node = route[i + 1]
                travel_time = time_matrix[from_node][to_node]
                print(f"From {from_node} to {to_node}: Travel time = {travel_time:.2f} min")
                total_time += travel_time
            print(f"Total time for truck {truck_idx + 1}: {total_time:.2f} minutes\n")

        map_center = depot_location
        carte = folium.Map(location=map_center, zoom_start=13)

        def minutes_to_time(minutes):
            """Convert minutes since midnight to 'HH:MM' format.
            Returns a formatted string with hours and minutes."""
            hours = int(minutes // 60)
            mins = int(minutes % 60)
            return f"{hours:02}:{mins:02}"

        def get_gradient_colors(start_color, end_color, steps):
            """Generate a list of color values forming a gradient between two specified colors.
            Returns a list of color codes for each step in the gradient."""
            color_map = mcolors.LinearSegmentedColormap.from_list("route_gradient", [start_color, end_color])
            return [mcolors.rgb2hex(color_map(i / (steps - 1))) for i in range(steps)]

        def calculate_bearing(lat1, lon1, lat2, lon2):
            """Calculate the bearing angle from one geographic point to another.
            Returns the angle in degrees between the two points."""
            lat1, lat2 = math.radians(lat1), math.radians(lat2)
            diff_lon = math.radians(lon2 - lon1)
            x = math.sin(diff_lon) * math.cos(lat2)
            y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(diff_lon)
            initial_bearing = math.atan2(x, y)
            return math.degrees(initial_bearing)

        def plot_routes_with_visual_details_for_multiple_trucks(truck_routes, graph, nodes, time_windows, time_matrix):
            """Visualizes each truck's route on a map, adding detailed information about client locations,
            time windows, arrival and departure times, and route segments. Saves the map to an HTML file."""
            colors = ["blue", "green", "purple", "orange", "red", "cyan", "magenta"]
            truck_colors = [colors[i % len(colors)] for i in range(num_trucks)]

            for truck_idx, solution in enumerate(truck_routes):
                cumulative_time = 480
                route_coords = []
                color = truck_colors[truck_idx]

                if len(solution) <= 2:
                    continue

                print(f"\nRoute details for Truck {truck_idx + 1}:")
                for i in range(len(solution) - 1):
                    try:
                        path = nx.shortest_path(graph, nodes[solution[i]], nodes[solution[i + 1]], weight='length')
                        coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in path]
                        route_coords.extend(coords)
                        folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8).add_to(carte)
                        midpoint = coords[len(coords) // 2]
                        folium.Marker(
                            location=midpoint,
                            icon=folium.DivIcon(html=f"<div style='font-size: 16px; color: black;'><b>{i + 1}</b></div>")
                        ).add_to(carte)
                    except nx.NetworkXNoPath:
                        print(f"No path between {solution[i]} and {solution[i + 1]}")
                        continue

                depot_location = (graph.nodes[nodes[solution[0]]]['y'], graph.nodes[nodes[solution[0]]]['x'])
                depot_departure_time = minutes_to_time(cumulative_time)
                folium.Marker(
                    depot_location,
                    popup=f"Depot (Truck {truck_idx + 1})<br>Departure Time: {depot_departure_time}",
                    icon=folium.Icon(color="black", icon="info-sign")
                ).add_to(carte)

                print(f"Depot (Truck {truck_idx + 1}): Location {depot_location}")
                print(f"  Departure Time: {depot_departure_time}\n")

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
                        icon=folium.Icon(color=color, icon="info-sign")
                    ).add_to(carte)

                last_client_id = solution[-2]
                final_travel_time = time_matrix[last_client_id][solution[-1]]
                depot_arrival_time = cumulative_time + final_travel_time
                depot_arrival_time_str = minutes_to_time(depot_arrival_time)

                print(f"Return to Depot (Truck {truck_idx + 1}): Location {depot_location}")
                print(f"  Arrival Time: {depot_arrival_time_str}\n")

                folium.Marker(
                    depot_location,
                    popup=f"Depot (Truck {truck_idx + 1})<br>Arrival Time: {depot_arrival_time_str}",
                    icon=folium.Icon(color="black", icon="info-sign")
                ).add_to(carte)

        plot_routes_with_visual_details_for_multiple_trucks(best_routes, graph, nodes, time_windows, time_matrix)
        carte.save("map_vrptw_barcelona_2.html")
        print("Program finished.")

        loading_window.destroy()

        webbrowser.open(MAP_FILE)
        print("VRPTW map saved and opened in browser.")

    except ValueError as e:
        messagebox.showerror("Invalid input", "Please enter a valid positive integer for the number of vehicles.")
        return


root = tk.Tk()
root.title("VRP Configuration")
root.geometry("300x150")

label_num_trucks = tk.Label(root, text="Enter the number of trucks:")
label_num_trucks.pack(pady=10)

num_trucks_entry = tk.Entry(root)
num_trucks_entry.pack(pady=5)

start_button = tk.Button(root, text="Start VRP Program", command=start_vrp_program)
start_button.pack(pady=20)

root.mainloop()

