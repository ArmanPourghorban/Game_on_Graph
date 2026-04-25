from __future__ import annotations

import random
import networkx as nx


def strategy(state: dict) -> None:
    """
    Attacker strategy:
    move toward the closest flag using weighted shortest path.
    The strategy writes the selected node into state["action"].
    """

    current_node = state["curr_pos"]
    flag_positions = state["flag_pos"]
    graph = state["graph"]
    speed = state.get("speed", 1)

    closest_flag = None
    min_distance = float("inf")

    for flag in flag_positions:
        try:
            dist = nx.shortest_path_length(graph, source=current_node, target=flag, weight="length")
            if dist < min_distance:
                min_distance = dist
                closest_flag = flag
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if closest_flag is None:
        neighbors = list(graph.neighbors(current_node))
        state["action"] = random.choice(neighbors) if neighbors else current_node
        return

    try:
        path = nx.shortest_path(graph, source=current_node, target=closest_flag, weight="length")
        if len(path) <= 1:
            state["action"] = current_node
        else:
            step_index = min(speed, len(path) - 1)
            state["action"] = path[step_index]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        neighbors = list(graph.neighbors(current_node))
        state["action"] = random.choice(neighbors) if neighbors else current_node