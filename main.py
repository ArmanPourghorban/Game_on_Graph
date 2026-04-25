from __future__ import annotations

from pathlib import Path
from typing import Tuple

import networkx as nx
import osmnx as ox


PLACE_NAME = "Austin, Texas, USA"
BBOX_DIST_METERS = 8000
NETWORK_TYPE = "drive"

OUTPUT_DIR = Path("output")
GRAPHML_PATH = OUTPUT_DIR / "graph.graphml"


def geocode_compat(query: str) -> Tuple[float, float]:
    if hasattr(ox, "geocode"):
        return ox.geocode(query)
    if hasattr(ox, "geocoder") and hasattr(ox.geocoder, "geocode"):
        return ox.geocoder.geocode(query)
    raise RuntimeError("Could not find geocode function in osmnx.")


def add_edge_lengths_compat(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if hasattr(ox, "distance") and hasattr(ox.distance, "add_edge_lengths"):
        return ox.distance.add_edge_lengths(G)
    if hasattr(ox, "add_edge_lengths"):
        return ox.add_edge_lengths(G)
    return G


def build_graph_bbox(place: str, bbox_dist_m: int, network_type: str) -> nx.MultiDiGraph:
    center_latlon = geocode_compat(place)
    lat, lon = center_latlon

    north, south, east, west = ox.utils_geo.bbox_from_point((lat, lon), dist=bbox_dist_m)
    bbox = (north, south, east, west)

    G = ox.graph_from_bbox(
        bbox,
        network_type=network_type,
        simplify=True,
    )

    G = add_edge_lengths_compat(G)
    G = ox.project_graph(G)
    return G


def main() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building road graph...")
    G = build_graph_bbox(PLACE_NAME, BBOX_DIST_METERS, NETWORK_TYPE)

    print(f"Graph built: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

    ox.save_graphml(G, GRAPHML_PATH)
    print(f"Saved graph: {GRAPHML_PATH}")
    return GRAPHML_PATH


if __name__ == "__main__":
    main()