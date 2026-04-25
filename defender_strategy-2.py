from __future__ import annotations

import random
import networkx as nx


def strategy(state: dict) -> None:
    """
    Defender strategy:
    - identify attackers that can still threaten flags
    - if none, move toward the closest flag
    - otherwise move toward an interception target
    The strategy writes the selected node into state["action"].
    """

    current_node = state["curr_pos"]
    flag_positions = state["flag_pos"]
    attacker_positions = state["attacker_positions"]
    graph = state["graph"]
    speed = state.get("speed", 1)
    capture_radius = state.get("capture_radius", 1)

    alpha = 0.5
    reachable_attackers = []

    for attacker in attacker_positions:
        try:
            attacker_can_reach_any_flag = False
            defender_can_block = False

            for flag in flag_positions:
                attacker_to_flag = nx.shortest_path_length(graph, source=attacker, target=flag, weight="length")
                defender_to_flag = nx.shortest_path_length(graph, source=current_node, target=flag, weight="length")

                if attacker_to_flag < float("inf"):
                    attacker_can_reach_any_flag = True

                if defender_to_flag <= attacker_to_flag + capture_radius:
                    defender_can_block = True

            if attacker_can_reach_any_flag and defender_can_block:
                reachable_attackers.append(attacker)

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not reachable_attackers:
        best_flag = None
        best_dist = float("inf")

        for flag in flag_positions:
            try:
                dist = nx.shortest_path_length(graph, source=current_node, target=flag, weight="length")
                if dist < best_dist:
                    best_dist = dist
                    best_flag = flag
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        if best_flag is None:
            neighbors = list(graph.neighbors(current_node))
            state["action"] = random.choice(neighbors) if neighbors else current_node
            return

        try:
            path = nx.shortest_path(graph, source=current_node, target=best_flag, weight="length")
            if len(path) <= 1:
                state["action"] = current_node
            else:
                step_index = min(speed, len(path) - 1)
                state["action"] = path[step_index]
            return
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            neighbors = list(graph.neighbors(current_node))
            state["action"] = random.choice(neighbors) if neighbors else current_node
            return

    closest_attacker = None
    min_score = float("inf")
    chosen_flag = None

    for attacker in reachable_attackers:
        try:
            defender_to_attacker = nx.shortest_path_length(graph, source=current_node, target=attacker, weight="length")

            for flag in flag_positions:
                try:
                    attacker_to_flag = nx.shortest_path_length(graph, source=attacker, target=flag, weight="length")
                    total_score = alpha * attacker_to_flag + (1 - alpha) * defender_to_attacker

                    if total_score < min_score:
                        min_score = total_score
                        closest_attacker = attacker
                        chosen_flag = flag
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if closest_attacker is None:
        neighbors = list(graph.neighbors(current_node))
        state["action"] = random.choice(neighbors) if neighbors else current_node
        return

    try:
        attacker_path_to_flag = nx.shortest_path(graph, source=closest_attacker, target=chosen_flag, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        attacker_path_to_flag = [closest_attacker]

    defender_target = chosen_flag

    for node in attacker_path_to_flag:
        try:
            defender_time = nx.shortest_path_length(graph, source=current_node, target=node, weight="length")
            attacker_time = nx.shortest_path_length(graph, source=closest_attacker, target=node, weight="length")
            if defender_time <= attacker_time:
                defender_target = node
                break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    try:
        path = nx.shortest_path(graph, source=current_node, target=defender_target, weight="length")
        if len(path) <= 1:
            state["action"] = current_node
        else:
            step_index = min(speed, len(path) - 1)
            state["action"] = path[step_index]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        neighbors = list(graph.neighbors(current_node))
        state["action"] = random.choice(neighbors) if neighbors else current_node