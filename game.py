from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import networkx as nx
import numpy as np
import osmnx as ox
import pygame

import main
import attacker_strategy
import defender_strategy


# =========================================================
# PATHS / CONFIG
# =========================================================
GRAPHML_PATH = Path("output/graph.graphml")
OUTPUT_DIR = Path("output")

RANDOM_SEED = 7
NUM_ATTACKERS = 5
NUM_DEFENDERS = 5
NUM_FLAGS = 3
MAX_STEPS = 500

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 900
FPS = 30
STEP_DELAY_MS = 300

BACKGROUND_COLOR = (245, 245, 245)
EDGE_COLOR = (180, 180, 180)
NODE_COLOR = (210, 210, 210)
FLAG_COLOR = (0, 180, 0)
ATTACKER_COLOR = (220, 30, 30)
DEFENDER_COLOR = (30, 80, 220)
TEXT_COLOR = (20, 20, 20)

EDGE_WIDTH = 1
BASE_NODE_RADIUS = 2
FLAG_RADIUS = 8
ATTACKER_RADIUS = 7
DEFENDER_RADIUS = 7

PERIPHERY_FRACTION = 0.12

RECORD_VIDEO = True
VIDEO_FILENAME = "game_recording.mp4"   # change to .avi if needed
VIDEO_FPS = 30
FINAL_HOLD_SECONDS = 2.0

ATTACKER_SPEED = 1
DEFENDER_SPEED = 1
ATTACKER_CAPTURE_RADIUS = 2
DEFENDER_CAPTURE_RADIUS = 1


# =========================================================
# AGENT DATA
# =========================================================
@dataclass
class Attacker:
    attacker_id: int
    current_node: int
    active: bool = True
    captured: bool = False
    reached_flag: bool = False
    speed: int = ATTACKER_SPEED
    capture_radius: int = ATTACKER_CAPTURE_RADIUS


@dataclass
class Defender:
    defender_id: int
    current_node: int
    active: bool = True
    speed: int = DEFENDER_SPEED
    capture_radius: int = DEFENDER_CAPTURE_RADIUS


# =========================================================
# VIDEO WRITER
# =========================================================
class VideoRecorder:
    def __init__(self, output_path: Path, fps: int = 30, enabled: bool = True) -> None:
        self.output_path = output_path
        self.fps = fps
        self.enabled = enabled
        self.writer = None
        self.frame_count = 0

    def start(self) -> None:
        if not self.enabled:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.output_path.suffix.lower()

        try:
            if suffix == ".mp4":
                self.writer = imageio.get_writer(
                    self.output_path,
                    format="FFMPEG",
                    mode="I",
                    fps=self.fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    macro_block_size=None,
                    ffmpeg_log_level="error",
                )
            elif suffix == ".avi":
                self.writer = imageio.get_writer(
                    self.output_path,
                    format="FFMPEG",
                    mode="I",
                    fps=self.fps,
                    codec="mpeg4",
                    macro_block_size=None,
                    ffmpeg_log_level="error",
                )
            else:
                raise ValueError("VIDEO_FILENAME must end with .mp4 or .avi")

            print(f"Recording video to: {self.output_path}")

        except Exception as e:
            self.enabled = False
            self.writer = None
            print(f"Video recording disabled: {e}")

    def capture(self, screen: pygame.Surface) -> None:
        if not self.enabled or self.writer is None:
            return

        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1, 0, 2))
        self.writer.append_data(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        if self.enabled and self.frame_count > 0:
            print(f"Saved video: {self.output_path} ({self.frame_count} frames)")


# =========================================================
# GRAPH
# =========================================================
def load_or_build_graph(force_rebuild: bool = False) -> nx.MultiDiGraph:
    if force_rebuild or not GRAPHML_PATH.exists():
        print("Graph not found or rebuild requested. Calling main.py...")
        main.main()

    print("Loading graph...")
    G = ox.load_graphml(GRAPHML_PATH)

    mapping = {}
    for n in G.nodes:
        try:
            mapping[n] = int(n)
        except (ValueError, TypeError):
            pass

    if mapping:
        G = nx.relabel_nodes(G, mapping)

    return G


def get_node_positions(G: nx.MultiDiGraph) -> Dict[int, Tuple[float, float]]:
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    positions = {}
    for node_id, row in nodes_gdf.iterrows():
        positions[int(node_id)] = (float(row.geometry.x), float(row.geometry.y))
    return positions


def choose_flag_nodes(G: nx.MultiDiGraph, num_flags: int = 3) -> List[int]:
    """
    Choose flags spread across the graph using farthest-point sampling.
    This makes flags much more separated across the map.
    """
    all_nodes = [int(n) for n in G.nodes]
    if len(all_nodes) <= num_flags:
        return all_nodes

    selected_flags = [random.choice(all_nodes)]

    while len(selected_flags) < num_flags:
        best_candidate = None
        best_score = -1.0

        for candidate in all_nodes:
            if candidate in selected_flags:
                continue

            min_dist_to_selected = float("inf")

            for chosen in selected_flags:
                try:
                    dist = nx.shortest_path_length(
                        G,
                        source=candidate,
                        target=chosen,
                        weight="length",
                    )
                    if dist < min_dist_to_selected:
                        min_dist_to_selected = dist
                except nx.NetworkXNoPath:
                    continue

            if min_dist_to_selected > best_score:
                best_score = min_dist_to_selected
                best_candidate = candidate

        if best_candidate is None:
            break

        selected_flags.append(best_candidate)

    return selected_flags


def shortest_path_length_safe(G: nx.MultiDiGraph, source: int, target: int) -> float:
    try:
        return nx.shortest_path_length(G, source=source, target=target, weight="length")
    except nx.NetworkXNoPath:
        return float("inf")


def choose_spawn_nodes(
    G: nx.MultiDiGraph,
    flag_nodes: List[int],
    num_agents: int,
    rng: random.Random,
) -> List[int]:
    scored_nodes = []

    for node in G.nodes:
        node = int(node)
        if node in flag_nodes:
            continue

        best_flag_dist = min(shortest_path_length_safe(G, node, flag) for flag in flag_nodes)
        if math.isfinite(best_flag_dist):
            scored_nodes.append((node, best_flag_dist))

    if not scored_nodes:
        raise RuntimeError("No valid spawn nodes found.")

    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    candidate_count = max(num_agents * 4, int(len(scored_nodes) * PERIPHERY_FRACTION))
    candidate_count = min(candidate_count, len(scored_nodes))
    candidates = [node for node, _ in scored_nodes[:candidate_count]]

    if len(candidates) < num_agents:
        raise RuntimeError("Not enough spawn nodes available.")

    return rng.sample(candidates, num_agents)


def choose_defender_start_nodes(
    G: nx.MultiDiGraph,
    flag_nodes: List[int],
    num_defenders: int,
) -> List[int]:
    starts = []
    used = set()

    # Put first defenders directly on different flags
    for flag in flag_nodes:
        if len(starts) < num_defenders:
            starts.append(flag)
            used.add(flag)

    # Expand outward around each flag in round robin
    frontier_by_flag = {flag: [flag] for flag in flag_nodes}
    visited_by_flag = {flag: {flag} for flag in flag_nodes}

    while len(starts) < num_defenders:
        added_any = False

        for flag in flag_nodes:
            if len(starts) >= num_defenders:
                break

            new_frontier = []
            for node in frontier_by_flag[flag]:
                for nbr in G.neighbors(node):
                    nbr = int(nbr)
                    if nbr not in visited_by_flag[flag]:
                        visited_by_flag[flag].add(nbr)
                        new_frontier.append(nbr)

                        if nbr not in used:
                            starts.append(nbr)
                            used.add(nbr)
                            added_any = True
                            if len(starts) >= num_defenders:
                                break
                if len(starts) >= num_defenders:
                    break

            frontier_by_flag[flag] = new_frontier

        if not added_any:
            break

    while len(starts) < num_defenders:
        starts.append(flag_nodes[len(starts) % len(flag_nodes)])

    return starts[:num_defenders]


# =========================================================
# STATE BUILDERS
# =========================================================
def build_attacker_state(
    G: nx.MultiDiGraph,
    attacker: Attacker,
    attackers: List[Attacker],
    defenders: List[Defender],
    flag_nodes: List[int],
    time_step: int,
) -> dict:
    return {
        "name": f"attacker_{attacker.attacker_id}",
        "team": "attacker",
        "time": time_step,
        "graph": G,
        "curr_pos": attacker.current_node,
        "flag_pos": flag_nodes,
        "flag_weight": [1] * len(flag_nodes),
        "attacker_positions": [a.current_node for a in attackers if a.active],
        "defender_positions": [d.current_node for d in defenders if d.active],
        "speed": attacker.speed,
        "capture_radius": attacker.capture_radius,
        "action": attacker.current_node,
    }


def build_defender_state(
    G: nx.MultiDiGraph,
    defender: Defender,
    attackers: List[Attacker],
    defenders: List[Defender],
    flag_nodes: List[int],
    time_step: int,
) -> dict:
    return {
        "name": f"defender_{defender.defender_id}",
        "team": "defender",
        "time": time_step,
        "graph": G,
        "curr_pos": defender.current_node,
        "flag_pos": flag_nodes,
        "flag_weight": [1] * len(flag_nodes),
        "attacker_positions": [a.current_node for a in attackers if a.active],
        "defender_positions": [d.current_node for d in defenders if d.active],
        "speed": defender.speed,
        "capture_radius": defender.capture_radius,
        "action": defender.current_node,
    }


# =========================================================
# DRAWING
# =========================================================
def build_screen_positions(
    graph_positions: Dict[int, Tuple[float, float]],
    width: int,
    height: int,
    margin: int = 40,
) -> Dict[int, Tuple[int, int]]:
    xs = [p[0] for p in graph_positions.values()]
    ys = [p[1] for p in graph_positions.values()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    scale_x = (width - 2 * margin) / span_x
    scale_y = (height - 2 * margin) / span_y
    scale = min(scale_x, scale_y)

    screen_positions = {}
    for node, (x, y) in graph_positions.items():
        sx = int(margin + (x - min_x) * scale)
        sy = int(height - (margin + (y - min_y) * scale))
        screen_positions[node] = (sx, sy)

    return screen_positions


def draw_graph(screen: pygame.Surface, G: nx.MultiDiGraph, screen_positions: Dict[int, Tuple[int, int]]) -> None:
    for u, v, _ in G.edges(keys=True):
        u = int(u)
        v = int(v)
        if u in screen_positions and v in screen_positions:
            pygame.draw.line(screen, EDGE_COLOR, screen_positions[u], screen_positions[v], EDGE_WIDTH)

    for _, pos in screen_positions.items():
        pygame.draw.circle(screen, NODE_COLOR, pos, BASE_NODE_RADIUS)


def draw_agents(
    screen: pygame.Surface,
    screen_positions: Dict[int, Tuple[int, int]],
    flag_nodes: List[int],
    attackers: List[Attacker],
    defenders: List[Defender],
) -> None:
    for flag in flag_nodes:
        if flag in screen_positions:
            pygame.draw.circle(screen, FLAG_COLOR, screen_positions[flag], FLAG_RADIUS)

    for attacker in attackers:
        if attacker.active and attacker.current_node in screen_positions:
            pygame.draw.circle(screen, ATTACKER_COLOR, screen_positions[attacker.current_node], ATTACKER_RADIUS)
        elif attacker.captured and attacker.current_node in screen_positions:
            pygame.draw.circle(screen, (150, 150, 150), screen_positions[attacker.current_node], ATTACKER_RADIUS)

    for defender in defenders:
        if defender.active and defender.current_node in screen_positions:
            pygame.draw.circle(screen, DEFENDER_COLOR, screen_positions[defender.current_node], DEFENDER_RADIUS)


def draw_text(
    screen: pygame.Surface,
    font: pygame.font.Font,
    step: int,
    attackers: List[Attacker],
    final_message: str,
) -> None:
    active_attackers = sum(1 for a in attackers if a.active)
    captured_attackers = sum(1 for a in attackers if a.captured)
    reached_attackers = sum(1 for a in attackers if a.reached_flag)

    line1 = (
        f"Step: {step} | "
        f"Active attackers: {active_attackers} | "
        f"Captured: {captured_attackers} | "
        f"Reached flag: {reached_attackers}"
    )
    surface1 = font.render(line1, True, TEXT_COLOR)
    screen.blit(surface1, (20, 20))

    if final_message:
        surface2 = font.render(final_message, True, TEXT_COLOR)
        screen.blit(surface2, (20, 55))


def render_frame(
    screen: pygame.Surface,
    font: pygame.font.Font,
    G: nx.MultiDiGraph,
    screen_positions: Dict[int, Tuple[int, int]],
    flag_nodes: List[int],
    attackers: List[Attacker],
    defenders: List[Defender],
    step: int,
    final_message: str,
) -> None:
    screen.fill(BACKGROUND_COLOR)
    draw_graph(screen, G, screen_positions)
    draw_agents(screen, screen_positions, flag_nodes, attackers, defenders)
    draw_text(screen, font, step, attackers, final_message)
    pygame.display.flip()


# =========================================================
# GAME LOGIC
# =========================================================
def update_attackers(
    G: nx.MultiDiGraph,
    attackers: List[Attacker],
    defenders: List[Defender],
    flag_nodes: List[int],
    time_step: int,
) -> None:
    for attacker in attackers:
        if not attacker.active:
            continue

        state = build_attacker_state(G, attacker, attackers, defenders, flag_nodes, time_step)
        attacker_strategy.strategy(state)

        next_node = state.get("action", attacker.current_node)
        if next_node in G.nodes:
            attacker.current_node = int(next_node)

        if attacker.current_node in flag_nodes:
            attacker.reached_flag = True
            attacker.active = False


def update_defenders(
    G: nx.MultiDiGraph,
    attackers: List[Attacker],
    defenders: List[Defender],
    flag_nodes: List[int],
    time_step: int,
) -> None:
    for defender in defenders:
        if not defender.active:
            continue

        state = build_defender_state(G, defender, attackers, defenders, flag_nodes, time_step)
        defender_strategy.strategy(state)

        next_node = state.get("action", defender.current_node)
        if next_node in G.nodes:
            defender.current_node = int(next_node)


def resolve_captures(attackers: List[Attacker], defenders: List[Defender]) -> int:
    captures = 0
    defender_nodes = {d.current_node for d in defenders if d.active}

    for attacker in attackers:
        if attacker.active and attacker.current_node in defender_nodes:
            attacker.active = False
            attacker.captured = True
            captures += 1

    return captures


def game_over(attackers: List[Attacker]) -> Tuple[bool, str]:
    if any(a.reached_flag for a in attackers):
        return True, "Attackers win: at least one attacker reached a flag."
    if all(not a.active for a in attackers):
        return True, "Defenders win: all attackers were captured."
    return False, ""


# =========================================================
# MAIN LOOP
# =========================================================
def run_game(force_rebuild_graph: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    G = load_or_build_graph(force_rebuild=force_rebuild_graph)
    graph_positions = get_node_positions(G)
    screen_positions = build_screen_positions(graph_positions, WINDOW_WIDTH, WINDOW_HEIGHT)

    flag_nodes = choose_flag_nodes(G, num_flags=NUM_FLAGS)
    attacker_spawn_nodes = choose_spawn_nodes(G, flag_nodes, NUM_ATTACKERS, rng)
    defender_start_nodes = choose_defender_start_nodes(G, flag_nodes, NUM_DEFENDERS)

    attackers = [Attacker(attacker_id=i, current_node=attacker_spawn_nodes[i]) for i in range(NUM_ATTACKERS)]
    defenders = [Defender(defender_id=i, current_node=defender_start_nodes[i]) for i in range(NUM_DEFENDERS)]

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Capture the Flag on Road Graph")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    recorder = VideoRecorder(output_path=OUTPUT_DIR / VIDEO_FILENAME, fps=VIDEO_FPS, enabled=RECORD_VIDEO)
    recorder.start()

    print("Flag nodes:", flag_nodes)
    print("Attacker starts:", [a.current_node for a in attackers])
    print("Defender starts:", [d.current_node for d in defenders])

    running = True
    step = 0
    last_update_time = pygame.time.get_ticks()
    final_message = ""
    final_hold_frames = int(FINAL_HOLD_SECONDS * VIDEO_FPS)
    final_hold_counter = 0

    render_frame(screen, font, G, screen_positions, flag_nodes, attackers, defenders, step, final_message)
    recorder.capture(screen)

    try:
        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if final_message == "" and current_time - last_update_time >= STEP_DELAY_MS:
                step += 1

                update_attackers(G, attackers, defenders, flag_nodes, step)

                is_over, message = game_over(attackers)
                if is_over:
                    final_message = message
                else:
                    update_defenders(G, attackers, defenders, flag_nodes, step)
                    resolve_captures(attackers, defenders)

                    is_over, message = game_over(attackers)
                    if is_over:
                        final_message = message

                if step >= MAX_STEPS and final_message == "":
                    final_message = "Maximum number of steps reached."

                last_update_time = current_time

            render_frame(screen, font, G, screen_positions, flag_nodes, attackers, defenders, step, final_message)
            recorder.capture(screen)

            if final_message:
                final_hold_counter += 1
                if final_hold_counter >= final_hold_frames:
                    running = False

            clock.tick(FPS)

    finally:
        recorder.close()
        pygame.quit()


if __name__ == "__main__":
    run_game(force_rebuild_graph=False)
