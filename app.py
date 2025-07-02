import streamlit as st
import math, random, requests, numpy as np
import geopandas as gpd, osmnx as ox, networkx as nx
from shapely.geometry import Point, shape
from sqlalchemy import create_engine, text

# --- Database config ---
DB_NAME = "PredictiveModel"
DB_USER = "postgres"
DB_PASS = "Soccer.37"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "block_population_4326"
ENGINE = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- Core functions (copied from your script) ---
def get_city_polygon(city_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": city_name,
        "format": "json",
        "polygon_geojson": 1,
        "limit": 1,
        "countrycodes": "us,ca"
    }
    headers = {"User-Agent": "PredictiveModelApp (Calderwoodnolan@gmail.com)"}
    r = requests.get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()
    poly = shape(data[0]["geojson"])
    return gpd.GeoDataFrame([{"geometry": poly}], crs="EPSG:4326")

def query_population(city_gdf):
    wkt = city_gdf.iloc[0].geometry.wkt
    sql = text(f"""
        SELECT SUM(unique_pop) as total_population
        FROM {TABLE_NAME}
        WHERE ST_Intersects(geom, ST_GeomFromText(:wkt,4326))
    """)
    with ENGINE.connect() as conn:
        return conn.execute(sql, {"wkt": wkt}).scalar() or 0

def calculate_target_mart(pop, pop_min=1000, pop_max=8810767):
    pop_log     = math.log10(pop) if pop > 0 else 0
    pop_log_min = math.log10(pop_min)
    pop_log_max = math.log10(pop_max)
    norm        = 1 - (pop_log - pop_log_min) / (pop_log_max - pop_log_min)
    norm        = max(0, min(1, norm))
    return 10 + norm * (60 - 10)

def assign_edge_speeds(G):
    defaults = {
        'motorway':100, 'trunk':80, 'primary':60, 'secondary':50,
        'tertiary':40,'residential':30,'unclassified':40,
        'service':20,'living_street':10
    }
    for u, v, k, data in G.edges(keys=True, data=True):
        ms = data.get("maxspeed")
        speed = None
        if ms:
            ms = ms[0] if isinstance(ms, list) else ms
            num = "".join(c for c in str(ms) if c.isdigit() or c == ".")
            try:
                speed = float(num) * (1.60934 if "mph" in str(ms).lower() else 1)
            except:
                speed = None
        if not speed:
            hw = data.get("highway")
            if isinstance(hw, list): hw = hw[0]
            speed = defaults.get(hw, 40)
        data["speed_kph"] = speed
    return G

def get_intersection_delays(G):
    delays = {n:0 for n in G.nodes}
    for n in G.nodes:
        ctrls = set()
        for nbr in G.neighbors(n):
            ed = G.get_edge_data(n, nbr)
            for key in ed:
                c = ed[key].get("traffic_control")
                if c: ctrls.add(c.lower())
        if "traffic_signals" in ctrls:
            delays[n] = 15
        elif "stop" in ctrls:
            delays[n] = 15
        elif ctrls:
            delays[n] = 5
    return delays

def simulate(G, city_gdf, accounts, comm_pct, target_mart):
    G = assign_edge_speeds(G)
    delays = get_intersection_delays(G)
    bounds = city_gdf.iloc[0].geometry.bounds

    def rand_nodes(n):
        pts = []
        x0, y0, x1, y1 = bounds
        while len(pts) < n:
            x = random.uniform(x0, x1)
            y = random.uniform(y0, y1)
            p = Point(x, y)
            if city_gdf.iloc[0].geometry.contains(p):
                pts.append(p)
        return ox.distance.nearest_nodes(G, [p.x for p in pts], [p.y for p in pts])

    acct_nodes = rand_nodes(accounts)
    for gcount in range(1, 500):
        guard_nodes = rand_nodes(gcount)
        times = []
        for a in acct_nodes:
            best = float("inf")
            for g in guard_nodes:
                try:
                    length = nx.shortest_path_length(G, g, a, weight="length")
                    path   = nx.shortest_path(G, g, a, weight="length")
                    delay  = sum(delays[n] for n in path[1:])
                    spd    = G.edges[path[0], path[1], 0]["speed_kph"]
                    t = length / (spd * 1000/3600) / 60 + delay/60
                    best = min(best, t)
                except:
                    pass
            times.append(best)
        if np.median(times) <= target_mart:
            return gcount, round(np.median(times), 2)
    return None, None

# ─── Streamlit UI ────────────────────────────────────────────────────────────
st.title("📍 Predictive Guard Model")

city  = st.text_input("City (e.g. Toronto, Canada)", "Toronto, Canada")
accts = st.number_input("# of accounts", 100, step=50)
comm  = st.slider("% commercial accounts", 0.0, 1.0, 0.5)

if st.button("Run Simulation"):
    city_gdf = get_city_polygon(city)
    pop      = query_population(city_gdf)
    mart     = calculate_target_mart(pop)
    st.write(f"• Population: **{int(pop):,}**")
    st.write(f"• Target MART: **{mart:.1f} min**")

    G = ox.graph_from_polygon(city_gdf.iloc[0].geometry, network_type="drive")
    guards, resp = simulate(G, city_gdf, accts, comm, mart)

    if guards:
        st.success(f"{guards} guards needed → median response: {resp} min")
    else:
        st.error("No solution up to 500 guards.")
