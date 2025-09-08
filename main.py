import pandas as pd
import streamlit as st
import pydeck as pdk
import heapq
from collections import deque
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# Algoritmo BFS
def bfs_search(graph, start, goal):
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        current, path = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        if current in graph:
            for neighbor in graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

# Algoritmo DFS https://www.geeksforgeeks.org/python/python-program-for-depth-first-search-or-dfs-for-a-graph/
def dfs_search(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    while stack:
        current, path = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path
        if current in graph:
            for neighbor in reversed(graph.get(current, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

# Algoritmo Dijkstra https://www.dio.me/articles/o-algoritmo-de-dijkstra-em-python-encontrando-o-caminho-mais-curto
def dijkstra_search(graph, start, goal):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    previous_nodes = {node: None for node in graph}
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_distance > distances[current_node]:
            continue
            
        if current_node == goal:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            return path[::-1]

        if current_node in graph:
            for neighbor in graph[current_node]:
                distance = 1 
                new_dist = current_distance + distance
                
                if new_dist < distances.get(neighbor, float('infinity')):
                    distances[neighbor] = new_dist
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (new_dist, neighbor))
    return None

# Algoritmo A* visualizar : https://github.com/malufreitas/a-estrela
#.
#.
#.
@st.cache_data
def load_data():
    airportsDat = './data/airports.dat'
    routesDat = './data/routes.dat'
    colunas_airports = ["AirportID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", "Timezone", "DST", "Tz", "Type", "Source"]
    dfAirports = pd.read_csv(airportsDat, header=None, names=colunas_airports, dtype={'IATA': str})
    
    valid_airports_mask = (dfAirports['IATA'].str.len() == 3) & (dfAirports['IATA'].notna())
    valid_airports = dfAirports[valid_airports_mask].copy()
    
    valid_airports.set_index('IATA', inplace=True)

    colunas_routes = ["Airline", "AirlineID", "SourceAirport", "SourceAirportID", "DestinationAirport", "DestinationAirportID", "Codeshare", "Stops", "Equipment"]
    dfRoutes = pd.read_csv(routesDat, header=None, names=colunas_routes)
    
    return valid_airports, dfRoutes

dfAirports, dfRoutes = load_data()
valid_iata_codes = set(dfAirports.index)

@st.cache_data
def build_graph(dfRoutes, valid_iata_codes):  
    graph = {}
    for _, row in dfRoutes.iterrows():
        src = row['SourceAirport']
        dst = row['DestinationAirport']
        if src in valid_iata_codes and dst in valid_iata_codes:
            if src not in graph:
                graph[src] = []
            graph[src].append(dst)
    return graph

graph = build_graph(dfRoutes, valid_iata_codes)

st.title("Dashboard de Busca de Rotas Aéreas")

if 'path' not in st.session_state:
    st.session_state.path = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'distances_info' not in st.session_state:
    st.session_state.distances_info = []

col1, col2, col3 = st.columns(3)
with col1:
    source_airport = st.text_input("Aeroporto de origem (IATA)", "BVH").upper()
with col2:
    destination_airport = st.text_input("Aeroporto de destino (IATA)", "JFK").upper()
with col3:
    algorithm = st.selectbox("Algoritmo de busca", ["A*", "BFS", "DFS", "Dijkstra"])


if st.button("Buscar Rota"):
    st.session_state.path = None
    st.session_state.current_step = 0
    st.session_state.distances_info = []

    if source_airport not in valid_iata_codes:
        st.error(f"Aeroporto de origem inválido: {source_airport}")
    elif destination_airport not in valid_iata_codes:
        st.error(f"Aeroporto de destino inválido: {destination_airport}")
    elif source_airport == destination_airport:
        st.warning("Aeroporto de origem e destino são iguais!")
    else:
        with st.spinner(f"Buscando rota usando {algorithm}..."):
            path_found = None
            if algorithm == "BFS":
                path_found = bfs_search(graph, source_airport, destination_airport)
            elif algorithm == "DFS":
                path_found = dfs_search(graph, source_airport, destination_airport)
            elif algorithm == "Dijkstra":
                path_found = dijkstra_search(graph, source_airport, destination_airport)
            elif algorithm == "A*":
                path_found = astar_search(graph, source_airport, destination_airport, dfAirports)
        
        if path_found:
            st.session_state.path = path_found
            st.success(f"Rota encontrada com {algorithm}! Clique em 'Próximo Passo' para iniciar a viagem.")
        else:
            st.error(f"Nenhuma rota encontrada de {source_airport} para {destination_airport} usando {algorithm}")

if st.session_state.path:
    path = st.session_state.path
    max_steps = len(path) - 1
    
    if st.session_state.current_step < max_steps:
        if st.button("Próximo Passo"):
            st.session_state.current_step += 1
            
            step_index = st.session_state.current_step
            src_iata = path[step_index - 1]
            dst_iata = path[step_index]

            src_coords = dfAirports.loc[src_iata]
            dst_coords = dfAirports.loc[dst_iata]

            dist = haversine_distance(src_coords['Latitude'], src_coords['Longitude'], dst_coords['Latitude'], dst_coords['Longitude'])
            
            st.session_state.distances_info.append({
                'leg': f"{src_iata} → {dst_iata}",
                'distance': f"{dist:.2f} km",
                'coords': {
                    'start': [src_coords['Longitude'], src_coords['Latitude']],
                    'end': [dst_coords['Longitude'], dst_coords['Latitude']]
                }
            })
    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.5, pitch=45)

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=dfAirports.reset_index(),
        get_position=["Longitude", "Latitude"],
        get_color="[200, 30, 0, 160]",
        get_radius=10000,
        pickable=True,
    )
    
    route_coords_data = [info['coords'] for info in st.session_state.distances_info]
    line_layer = pdk.Layer(
        'LineLayer',
        data=route_coords_data,
        get_source_position='start',
        get_target_position='end',
        get_color='[0, 255, 0, 200]', 
        get_width=3,
        pickable=True
    )
    
    st.pydeck_chart(pdk.Deck(
        layers=[scatter_layer, line_layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{Name}</b><br/>{City} ({IATA})"}
    ))
    if st.session_state.current_step > 0:
        st.markdown("---")
        
        current_path_str = ' → '.join(path[:st.session_state.current_step + 1])
        st.write(f"**Passo {st.session_state.current_step}/{max_steps}:** `{current_path_str}`")

        st.subheader("Distâncias percorridas:")
        total_distance = 0
        for info in st.session_state.distances_info:
            dist_val = float(info['distance'].replace(' km', ''))
            total_distance += dist_val
            st.metric(label=info['leg'], value=info['distance'], delta=None)

        st.markdown("---")
        st.metric(label="Distância total até o momento:", value=f"{total_distance:.2f} km")

        if st.session_state.current_step == max_steps:
            st.success("Destino alcançado!")