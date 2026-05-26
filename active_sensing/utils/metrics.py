from pathlib import Path

import numpy as np
from utils.plotting import *

try:
    from scipy.stats import spearmanr
except ImportError as exc:
    raise ImportError("scipy is required for spearmanr") from exc


def extract_monotonic_runs(nodes, min_run_nodes=3):
    """Return monotonic runs as (start_idx, end_idx, direction)."""
    nodes = np.asarray(nodes).astype(int)
    n_nodes = len(nodes)
    runs = []

    i = 0
    while i < n_nodes - 1:
        diff = nodes[i + 1] - nodes[i]

        if diff > 0:
            run_dir = "out"
        elif diff < 0:
            run_dir = "in"
        else:
            i += 1
            continue

        j = i + 1
        while j < n_nodes - 1:
            next_diff = nodes[j + 1] - nodes[j]
            if run_dir == "out" and next_diff > 0:
                j += 1
            elif run_dir == "in" and next_diff < 0:
                j += 1
            else:
                break

        if j - i + 1 >= min_run_nodes:
            runs.append((i, j, run_dir))

        i = j

    return runs


def compute_monotonic_trajectory_tortuosity(
    mouse_sources: dict,
    mouse: str,
    z_key: str = "z_p",
    group_frac=(0.0, 1.0),
    exclude_first_n: int = 0,
    direction: str = "both",
    min_run_nodes: int = 2,
    eps: float = 1e-8,
):
    """Compute mean tortuosity across valid monotonic trajectory runs."""
    if mouse not in mouse_sources:
        raise KeyError(f"mouse='{mouse}' not in mouse_sources")

    if direction not in {"in", "out", "both"}:
        raise ValueError("direction must be 'in', 'out', or 'both'")

    if min_run_nodes < 2:
        raise ValueError("min_run_nodes must be >= 2")

    metrics_dir = Path(mouse_sources[mouse]["metrics_dir"])
    bouts_all = load_bouts_from_dir(metrics_dir)
    bouts_sel, _ = select_bouts_by_percent(bouts_all, group_frac)

    if len(bouts_sel) == 0:
        raise ValueError("No bouts selected.")

    ratios = []

    for bout in bouts_sel:
        if "target_node" not in bout or z_key not in bout:
            continue

        nodes = np.asarray(bout["target_node"]).astype(int)
        z_values = np.asarray(bout[z_key])

        if len(nodes) != z_values.shape[0]:
            raise ValueError(
                f"Length mismatch: len(target_node)={len(nodes)} vs "
                f"{z_key}.shape[0]={z_values.shape[0]}"
            )

        if len(nodes) <= exclude_first_n:
            continue

        nodes = nodes[exclude_first_n:]
        z_values = z_values[exclude_first_n:]

        if len(nodes) < 2:
            continue

        runs = extract_monotonic_runs(nodes, min_run_nodes=min_run_nodes)

        for start_idx, end_idx, run_dir in runs:
            if direction != "both" and run_dir != direction:
                continue

            z_run = z_values[start_idx:end_idx + 1]

            if len(z_run) < 2:
                continue

            path_length = np.sum(np.linalg.norm(z_run[1:] - z_run[:-1], axis=1))
            displacement = np.linalg.norm(z_run[-1] - z_run[0])

            if displacement < eps:
                continue

            ratios.append(path_length / displacement)

    if len(ratios) == 0:
        raise ValueError("No valid monotonic trajectories found.")

    return float(np.mean(ratios))


def compute_monotonic_depth_spearman(
    mouse_sources: dict,
    mouse: str,
    z_key: str = "z_p",
    group_frac=(0.0, 1.0),
    exclude_first_n: int = 0,
    direction: str = "both",
    min_run_nodes: int = 3,
    average_mode: str = "mean",
    abs_value: bool = True,
    eps: float = 1e-8,
):
    """Compute Spearman depth ordering across valid monotonic trajectory runs."""
    if mouse not in mouse_sources:
        raise KeyError(f"mouse='{mouse}' not in mouse_sources")

    if direction not in {"in", "out", "both"}:
        raise ValueError("direction must be 'in', 'out', or 'both'")

    if min_run_nodes < 3:
        raise ValueError("min_run_nodes must be >= 3 for Spearman correlation")

    if average_mode not in {"mean", "median"}:
        raise ValueError("average_mode must be 'mean' or 'median'")

    def node_depth(node_idx: np.ndarray) -> np.ndarray:
        node_idx = np.asarray(node_idx).astype(np.int64)
        return np.floor(np.log2(node_idx + 1)).astype(np.int64)

    def pc1_projection(z_run: np.ndarray) -> np.ndarray:
        z_centered = z_run - z_run.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(z_centered, full_matrices=False)
        return z_centered @ vt[0]

    metrics_dir = Path(mouse_sources[mouse]["metrics_dir"])
    bouts_all = load_bouts_from_dir(metrics_dir)
    bouts_sel, _ = select_bouts_by_percent(bouts_all, group_frac)

    if len(bouts_sel) == 0:
        raise ValueError("No bouts selected.")

    rho_list = []

    for bout in bouts_sel:
        if "target_node" not in bout or z_key not in bout:
            continue

        nodes = np.asarray(bout["target_node"]).astype(int)
        z_values = np.asarray(bout[z_key])

        if len(nodes) != z_values.shape[0]:
            raise ValueError(
                f"Length mismatch: len(target_node)={len(nodes)} vs "
                f"{z_key}.shape[0]={z_values.shape[0]}"
            )

        if len(nodes) <= exclude_first_n:
            continue

        nodes = nodes[exclude_first_n:]
        z_values = z_values[exclude_first_n:]

        if len(nodes) < min_run_nodes:
            continue

        runs = extract_monotonic_runs(nodes, min_run_nodes=min_run_nodes)

        for start_idx, end_idx, run_dir in runs:
            if direction != "both" and run_dir != direction:
                continue

            nodes_run = nodes[start_idx:end_idx + 1]
            z_run = z_values[start_idx:end_idx + 1]

            if len(nodes_run) < min_run_nodes:
                continue

            depths = node_depth(nodes_run)

            if np.all(depths == depths[0]):
                continue

            coord = pc1_projection(z_run)

            if np.std(coord) < eps:
                continue

            rho, _ = spearmanr(coord, depths)

            if np.isnan(rho):
                continue

            if abs_value:
                rho = abs(rho)
            elif run_dir == "out" and rho < 0:
                rho = -rho
            elif run_dir == "in" and rho > 0:
                rho = -rho

            rho_list.append(float(rho))

    if len(rho_list) == 0:
        raise ValueError("No valid monotonic trajectories found for depth-Spearman metric.")

    if average_mode == "mean":
        return float(np.mean(rho_list))

    return float(np.median(rho_list))


def compute_transition_type_consistency_in_out(
    mouse_sources: dict,
    mouse: str,
    z_key: str = "z_p",
    group_frac=(0.0, 1.0),
    exclude_first_n: int = 0,
    k: int = 10,
    node_min: int = 31,
    node_max: int = 126,
    return_by_direction: bool = True,
):
    """Compute nearest-neighbor transition direction consistency."""
    if mouse not in mouse_sources:
        raise KeyError(f"mouse='{mouse}' not in mouse_sources")

    if k < 1:
        raise ValueError("k must be >= 1")

    def direction_from_transition(x: int, y: int):
        if y == 2 * x + 1 or y == 2 * x + 2:
            return "OUT"
        if x > 0 and y == (x - 1) // 2:
            return "IN"
        return None

    metrics_dir = Path(mouse_sources[mouse]["metrics_dir"])
    bouts_all = load_bouts_from_dir(metrics_dir)
    bouts_sel, _ = select_bouts_by_percent(bouts_all, group_frac)

    if len(bouts_sel) == 0:
        raise ValueError("No bouts selected.")

    x_values = []
    transition_ids = []
    directions = []

    for bout in bouts_sel:
        if "target_node" not in bout or z_key not in bout:
            continue

        nodes = np.asarray(bout["target_node"]).astype(int)
        z_values = np.asarray(bout[z_key])

        if len(nodes) != z_values.shape[0]:
            raise ValueError(
                f"Length mismatch: len(target_node)={len(nodes)} "
                f"vs {z_key}.shape[0]={z_values.shape[0]}"
            )

        if len(nodes) <= exclude_first_n + 1:
            continue

        nodes = nodes[exclude_first_n:]
        z_values = z_values[exclude_first_n:]

        if len(nodes) < 2:
            continue

        for t in range(len(nodes) - 1):
            x = int(nodes[t])
            y = int(nodes[t + 1])

            if min(x, y) < node_min or max(x, y) > node_max:
                continue

            direction = direction_from_transition(x, y)
            if direction is None:
                continue

            x_values.append(z_values[t])
            transition_ids.append((x, y))
            directions.append(direction)

    if len(x_values) == 0:
        raise ValueError("No valid transitions found.")

    x_values = np.asarray(x_values, dtype=float)
    directions = np.asarray(directions)
    n_points = len(x_values)

    if n_points < 2:
        raise ValueError("Need at least 2 transition points.")

    squared_norms = np.sum(x_values * x_values, axis=1, keepdims=True)
    squared_distances = squared_norms + squared_norms.T - 2.0 * (x_values @ x_values.T)
    squared_distances = np.maximum(squared_distances, 0.0)
    np.fill_diagonal(squared_distances, np.inf)

    neighbor_order = np.argsort(squared_distances, axis=1)

    consistency_per_point = []
    point_directions = []
    skipped_not_enough_unique = 0

    for i in range(n_points):
        selected = []
        used_transition_ids = set()
        center_transition_id = transition_ids[i]

        for j in neighbor_order[i]:
            transition_id = transition_ids[j]

            if transition_id == center_transition_id:
                continue
            if transition_id in used_transition_ids:
                continue

            selected.append(j)
            used_transition_ids.add(transition_id)

            if len(selected) == k:
                break

        if len(selected) < k:
            skipped_not_enough_unique += 1
            continue

        same_direction = sum(directions[j] == directions[i] for j in selected)
        consistency_per_point.append(same_direction / k)
        point_directions.append(directions[i])

    if len(consistency_per_point) == 0:
        raise ValueError(
            "Could not find k unique non-identical transition neighbors "
            "for any point."
        )

    consistency_per_point = np.asarray(consistency_per_point, dtype=float)
    point_directions = np.asarray(point_directions)

    overall = float(np.mean(consistency_per_point))

    if not return_by_direction:
        return overall

    out_mask = point_directions == "OUT"
    in_mask = point_directions == "IN"

    return {
        "overall": overall,
        "OUT": float(np.mean(consistency_per_point[out_mask])) if np.any(out_mask) else np.nan,
        "IN": float(np.mean(consistency_per_point[in_mask])) if np.any(in_mask) else np.nan,
        "n_points_used": int(len(consistency_per_point)),
        "n_points_total": int(n_points),
        "n_skipped_not_enough_unique": int(skipped_not_enough_unique),
        "n_out_used": int(np.sum(out_mask)),
        "n_in_used": int(np.sum(in_mask)),
    }