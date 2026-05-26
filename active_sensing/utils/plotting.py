
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from umap import UMAP


def load_bouts_from_dir(metrics_dir: str | Path) -> list[dict[str, Any]]:
    """Load bout metric files."""
    metrics_dir = Path(metrics_dir)
    files = sorted(metrics_dir.glob("bout_*_metrics.npz"))

    bouts = []
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        bout = {k: d[k] for k in d.files}
        try:
            bid = int(fp.name.split("bout_")[-1].split("_")[0])
        except Exception:
            bid = len(bouts)
        bout["bout_id"] = int(bid)
        bouts.append(bout)
    return bouts


def select_bouts_by_percent(
    bouts_sorted: list[dict[str, Any]],
    group_frac: tuple[float, float] = (0.0, 1.0),
):
    """Select bouts by fractional index range."""
    lo_frac, hi_frac = float(group_frac[0]), float(group_frac[1])
    if not (0.0 <= lo_frac < hi_frac <= 1.0):
        raise ValueError("group_frac must satisfy 0 <= lo < hi <= 1.")

    n = len(bouts_sorted)
    if n == 0:
        return [], (0, 0, 0)

    start = int(np.floor(n * lo_frac))
    end = int(np.floor(n * hi_frac))

    if end <= start:
        return [], (start, end, n)

    return bouts_sorted[start:end], (start, end, n)


def _is_numeric_array(arr: Any) -> bool:
    arr = np.asarray(arr)
    return np.issubdtype(arr.dtype, np.number)


def _flatten_trailing_dims(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[:, None]
    return arr.reshape(arr.shape[0], -1)


def reduce_metric_per_timestep(
    arr: Any,
    vector_reduce: str = "mean",
) -> np.ndarray:
    """Reduce a per-timestep scalar/vector metric to one value per timestep."""
    arr = np.asarray(arr)

    if arr.ndim == 0:
        raise ValueError("Metric must vary across timesteps; got scalar with no time axis.")

    if arr.ndim == 1:
        return arr.astype(float)

    flat = _flatten_trailing_dims(arr).astype(float)

    vector_reduce = vector_reduce.lower()
    if vector_reduce in {"mean", "avg", "average"}:
        return np.nanmean(flat, axis=1)
    if vector_reduce == "max":
        return np.nanmax(flat, axis=1)
    if vector_reduce == "norm":
        return np.linalg.norm(flat, axis=1)
    if vector_reduce == "first":
        return flat[:, 0]
    raise ValueError("vector_reduce must be one of: 'mean', 'max', 'norm', 'first'")


def list_available_metric_keys(
    metrics_dir: str | Path,
    z_key: str | None = None,
    require_length_match: bool = True,
) -> list[str]:
    """List numeric per-timestep metric keys."""
    bouts = load_bouts_from_dir(metrics_dir)
    keys = set()

    for b in bouts:
        if "target_node" not in b:
            continue
        t = len(np.asarray(b["target_node"]))
        for k, v in b.items():
            if k in {"target_node", "bout_id"}:
                continue
            if z_key is not None and k == z_key:
                continue

            arr = np.asarray(v)
            if not _is_numeric_array(arr):
                continue
            if arr.ndim == 0:
                continue
            if require_length_match and len(arr) != t:
                continue
            keys.add(k)

    return sorted(keys)


def _resolve_color_values(
    saved: dict[str, Any],
    color_by: str,
    vector_reduce: str = "mean",
) -> tuple[np.ndarray, str]:
    """Resolve color values for plotting."""
    color_by = str(color_by)

    if color_by == "node":
        nodes = np.asarray(saved["nodes_used"]).astype(int)
        node_to_idx = saved.get("node_to_idx", {int(n): int(n) for n in range(128)})
        mapped = np.array([node_to_idx[int(n)] for n in nodes], dtype=float)
        return mapped, "logical node index (remapped exactly like maze plot)"

    if color_by == "prev_node":
        return np.asarray(saved["prev_nodes_used"]).astype(float), "prev_node"

    if color_by == "bout_id":
        return np.asarray(saved["bout_id_used"]).astype(float), "bout id"

    if color_by == "time":
        return np.asarray(saved["step_idx_used"]).astype(float), "step index within bout"

    if color_by == "time_normalized":
        step_idx = np.asarray(saved["step_idx_used"]).astype(int)
        bout_id = np.asarray(saved["bout_id_used"]).astype(int)
        out = np.zeros(len(step_idx), dtype=float)

        for b in np.unique(bout_id):
            mask = bout_id == b
            steps = step_idx[mask]
            smin = np.min(steps)
            smax = np.max(steps)
            out[mask] = 0.0 if smax == smin else (steps - smin) / (smax - smin)
        return out, "normalized time within bout"

    metrics_used = saved.get("metrics_used", {})
    if color_by not in metrics_used:
        available = sorted(metrics_used.keys())
        raise KeyError(
            f"color_by='{color_by}' not found. "
            f"Available built-ins: node, prev_node, bout_id, time, time_normalized. "
            f"Saved metric keys: {available}"
        )

    values = reduce_metric_per_timestep(metrics_used[color_by], vector_reduce=vector_reduce)
    return values, f"{color_by} ({vector_reduce})"


def save_umap_embedding(
    mouse_sources: dict,
    mouse: str,
    z_key: str = "z_p",
    group_frac: tuple[float, float] = (0.0, 1.0),
    max_points_total: int | None = 40000,
    random_state: int = 0,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    exclude_first_n: int = 0,
    save_path: str | Path | None = None,
    save_all_metrics: bool = True,
    metric_keys: list[str] | None = None,
):
    """Fit UMAP and optionally save embedding metadata."""
    if mouse not in mouse_sources:
        raise KeyError(f"mouse='{mouse}' not in mouse_sources keys: {list(mouse_sources.keys())}")

    metrics_dir = Path(mouse_sources[mouse]["metrics_dir"])
    bouts_all = load_bouts_from_dir(metrics_dir)
    bouts_sel, (start, end, n_total) = select_bouts_by_percent(bouts_all, group_frac)

    if len(bouts_sel) == 0:
        raise ValueError(f"No bouts selected for mouse={mouse} with group_frac={group_frac}")

    unique_nodes = np.arange(128, dtype=int)
    node_to_idx = {int(n): int(n) for n in unique_nodes}

    z_list = []
    node_list = []
    prev_node_list = []
    bout_id_list = []
    step_idx_list = []
    metrics_accum: dict[str, list[np.ndarray]] = {}

    for local_bout_idx, b in enumerate(bouts_sel):
        if z_key not in b or "target_node" not in b:
            continue

        z = np.asarray(b[z_key])
        nodes = np.asarray(b["target_node"]).astype(int)

        if z.ndim == 0 or len(z) == 0:
            continue
        if len(nodes) != len(z):
            raise ValueError(
                f"Length mismatch in bout_id={b.get('bout_id', local_bout_idx)}: "
                f"len(target_node)={len(nodes)} vs len({z_key})={len(z)}"
            )

        t = len(nodes)
        step_mask = np.arange(t) >= exclude_first_n
        if not np.any(step_mask):
            continue

        prev_nodes = np.full(t, -1, dtype=int)
        if t > 1:
            prev_nodes[1:] = nodes[:-1]

        z_list.append(z[step_mask])
        node_list.append(nodes[step_mask])
        prev_node_list.append(prev_nodes[step_mask])
        step_idx_list.append(np.arange(t, dtype=int)[step_mask])
        bout_id_list.append(np.full(np.sum(step_mask), int(b.get("bout_id", local_bout_idx)), dtype=int))

        if save_all_metrics:
            candidate_keys = [
                k for k, v in b.items()
                if k not in {"target_node", "bout_id"}
                and k != z_key
                and np.asarray(v).ndim > 0
                and _is_numeric_array(v)
                and len(np.asarray(v)) == t
            ]
        else:
            if metric_keys is None:
                candidate_keys = []
            else:
                candidate_keys = [
                    k for k in metric_keys
                    if k in b
                    and np.asarray(b[k]).ndim > 0
                    and _is_numeric_array(b[k])
                    and len(np.asarray(b[k])) == t
                ]

        for k in candidate_keys:
            metrics_accum.setdefault(k, []).append(np.asarray(b[k])[step_mask])

    if len(z_list) == 0:
        raise ValueError(f"No usable z data found for z_key='{z_key}'")

    z_all = np.vstack(z_list)
    nodes_all = np.concatenate(node_list)
    prev_nodes_all = np.concatenate(prev_node_list)
    bout_id_all = np.concatenate(bout_id_list)
    step_idx_all = np.concatenate(step_idx_list)
    metrics_used = {k: np.concatenate(v, axis=0) for k, v in metrics_accum.items() if len(v) > 0}

    if max_points_total is not None and len(z_all) > max_points_total:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(z_all), size=max_points_total, replace=False)

        z_all = z_all[idx]
        nodes_all = nodes_all[idx]
        prev_nodes_all = prev_nodes_all[idx]
        bout_id_all = bout_id_all[idx]
        step_idx_all = step_idx_all[idx]
        for k in list(metrics_used.keys()):
            metrics_used[k] = metrics_used[k][idx]

    reducer = UMAP(
        n_components=2,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    z_umap = reducer.fit_transform(z_all)

    payload = {
        "mouse": mouse,
        "metrics_dir": str(metrics_dir),
        "z_key": z_key,
        "group_frac": tuple(group_frac),
        "selected_bout_index_range": (start, end, n_total),
        "exclude_first_n": int(exclude_first_n),
        "random_state": int(random_state),
        "umap_n_neighbors": int(umap_n_neighbors),
        "umap_min_dist": float(umap_min_dist),
        "Z_used": z_all,
        "Z_umap": z_umap,
        "nodes_used": nodes_all,
        "prev_nodes_used": prev_nodes_all,
        "bout_id_used": bout_id_all,
        "step_idx_used": step_idx_all,
        "metrics_used": metrics_used,
        "available_metric_keys": sorted(metrics_used.keys()),
        "unique_nodes": unique_nodes,
        "node_to_idx": node_to_idx,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(payload, f)

    return payload


def plot_saved_umap(
    umap_path: str | Path,
    color_by: str = "node",
    vector_reduce: str = "mean",
    cmap: str = "nipy_spectral",
    point_size: float = 6,
    alpha: float = 0.85,
    title_suffix: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
):
    """Plot a saved UMAP embedding with a selected color variable."""
    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    color_vals, color_label = _resolve_color_values(
        saved,
        color_by=color_by,
        vector_reduce=vector_reduce
    )

    if len(color_vals) != len(z_umap):
        raise ValueError(
            f"Length mismatch: len(color_vals)={len(color_vals)} vs len(Z_umap)={len(z_umap)}"
        )

    plt.figure(figsize=(7.2, 6.2))
    sc = plt.scatter(
        z_umap[:, 0],
        z_umap[:, 1],
        c=color_vals,
        s=point_size,
        cmap=cmap,
        vmin=np.nanmin(color_vals) if vmin is None else vmin,
        vmax=np.nanmax(color_vals) if vmax is None else vmax,
        alpha=alpha,
        linewidths=0,
    )

    title = f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP colored by {color_label}"
    if title_suffix:
        title += f"\n{title_suffix}"

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label(color_label)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix.lower() == ".svg":
            plt.savefig(save_path, format="svg")
        else:
            plt.savefig(save_path, dpi=dpi)

    plt.show()

    return {
        "Z_umap": z_umap,
        "color_vals": color_vals,
        "color_label": color_label,
        "saved": saved,
    }


def plot_saved_umap_highlight_node(
    umap_path: str | Path,
    highlight_node: int,
    background_color: str = "lightgray",
    highlight_color: str = "red",
    s_bg: float = 6,
    s_hi: float = 18,
    alpha_bg: float = 0.35,
    alpha_hi: float = 1.0,
):
    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    nodes = np.asarray(saved["nodes_used"]).astype(int)
    mask = nodes == int(highlight_node)

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=background_color, s=s_bg, alpha=alpha_bg, linewidths=0)

    if np.any(mask):
        plt.scatter(
            z_umap[mask, 0],
            z_umap[mask, 1],
            c=highlight_color,
            s=s_hi,
            alpha=alpha_hi,
            linewidths=0,
            label=f"node {highlight_node}",
        )
        plt.legend()
    else:
        print(f"Node {highlight_node} not found in saved UMAP.")

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP\nhighlight node {highlight_node}")
    plt.tight_layout()
    plt.show()

    return {
        "highlight_node": highlight_node,
        "num_highlighted_points": int(mask.sum()),
        "mask": mask,
    }


def plot_saved_umap_highlight_transition(
    umap_path: str | Path,
    target_node: int,
    prev_node: int | None = None,
    background_color: str = "lightgray",
    highlight_color: str = "red",
    s_bg: float = 6,
    s_hi: float = 18,
    alpha_bg: float = 0.30,
    alpha_hi: float = 1.0,
):
    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    nodes = np.asarray(saved["nodes_used"]).astype(int)
    prev_nodes = np.asarray(saved["prev_nodes_used"]).astype(int)

    if prev_node is None:
        mask = nodes == int(target_node)
        label = f"node {target_node}"
        title_suffix = f"highlight node {target_node}"
    else:
        mask = (nodes == int(target_node)) & (prev_nodes == int(prev_node))
        label = f"{prev_node} -> {target_node}"
        title_suffix = f"highlight transition {prev_node} -> {target_node}"

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=background_color, s=s_bg, alpha=alpha_bg, linewidths=0)

    if np.any(mask):
        plt.scatter(
            z_umap[mask, 0],
            z_umap[mask, 1],
            c=highlight_color,
            s=s_hi,
            alpha=alpha_hi,
            linewidths=0,
            label=label,
        )
        plt.legend()
    else:
        print(f"No points found for {label}.")

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP\n{title_suffix}")
    plt.tight_layout()
    plt.show()

    return {
        "target_node": target_node,
        "prev_node": prev_node,
        "num_highlighted_points": int(mask.sum()),
        "mask": mask,
    }


def plot_saved_umap_highlight_path_multicolor(
    umap_path: str | Path,
    path_nodes: list[int],
    background_color: str = "lightgray",
    transition_colors: list[str] | None = None,
    s_bg: float = 6,
    s_hi: float = 18,
    alpha_bg: float = 0.25,
    alpha_hi: float = 1.0,
):
    if len(path_nodes) < 2:
        raise ValueError("path_nodes must contain at least 2 nodes.")

    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    nodes = np.asarray(saved["nodes_used"]).astype(int)
    prev_nodes = np.asarray(saved["prev_nodes_used"]).astype(int)

    transitions = list(zip(path_nodes[:-1], path_nodes[1:]))

    if transition_colors is None:
        transition_colors = [
            "red", "blue", "green", "orange", "purple",
            "brown", "black", "magenta", "cyan",
        ]

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=background_color, s=s_bg, alpha=alpha_bg, linewidths=0)

    transition_counts = {}
    for i, (prev_node, curr_node) in enumerate(transitions):
        mask = (prev_nodes == int(prev_node)) & (nodes == int(curr_node))
        transition_counts[(prev_node, curr_node)] = int(mask.sum())

        if np.any(mask):
            plt.scatter(
                z_umap[mask, 0],
                z_umap[mask, 1],
                c=transition_colors[i % len(transition_colors)],
                s=s_hi,
                alpha=alpha_hi,
                linewidths=0,
                label=f"{prev_node} -> {curr_node}",
            )

    path_str = " -> ".join(str(x) for x in path_nodes)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP\npath transitions: {path_str}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "path_nodes": path_nodes,
        "transitions": transitions,
        "transition_counts": transition_counts,
    }


def plot_saved_umap_highlight_exact_full_path_multicolor(
    umap_path: str | Path,
    path_nodes: list[int],
    background_color: str = "lightgray",
    transition_colors: list[str] | None = None,
    s_bg: float = 6,
    s_hi: float = 18,
    alpha_bg: float = 0.25,
    alpha_hi: float = 1.0,
):
    if len(path_nodes) < 2:
        raise ValueError("path_nodes must contain at least 2 nodes.")

    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    nodes = np.asarray(saved["nodes_used"]).astype(int)
    bout_id = np.asarray(saved["bout_id_used"]).astype(int)
    step_idx = np.asarray(saved["step_idx_used"]).astype(int)

    if transition_colors is None:
        transition_colors = [
            "red", "blue", "green", "orange", "purple",
            "brown", "black", "magenta", "cyan",
        ]

    point_lookup = {(int(b), int(t)): i for i, (b, t) in enumerate(zip(bout_id, step_idx))}

    L = len(path_nodes)
    transition_masks = [np.zeros(len(nodes), dtype=bool) for _ in range(L - 1)]
    full_match_count = 0

    start_candidates = np.where(nodes == int(path_nodes[0]))[0]

    for i0 in start_candidates:
        b = int(bout_id[i0])
        t0 = int(step_idx[i0])

        matched_indices = []
        ok = True
        for offset, node in enumerate(path_nodes):
            key = (b, t0 + offset)
            if key not in point_lookup:
                ok = False
                break
            idx = point_lookup[key]
            if int(nodes[idx]) != int(node):
                ok = False
                break
            matched_indices.append(idx)

        if not ok:
            continue

        full_match_count += 1
        for j in range(1, L):
            transition_masks[j - 1][matched_indices[j]] = True

    plt.figure(figsize=(7.2, 6.2))
    plt.scatter(z_umap[:, 0], z_umap[:, 1], c=background_color, s=s_bg, alpha=alpha_bg, linewidths=0)

    transition_counts = {}
    for j, mask in enumerate(transition_masks, start=1):
        prev_node = path_nodes[j - 1]
        curr_node = path_nodes[j]
        transition_counts[(prev_node, curr_node)] = int(mask.sum())

        if np.any(mask):
            plt.scatter(
                z_umap[mask, 0],
                z_umap[mask, 1],
                c=transition_colors[(j - 1) % len(transition_colors)],
                s=s_hi,
                alpha=alpha_hi,
                linewidths=0,
                label=f"{prev_node} -> {curr_node}",
            )

    path_str = " -> ".join(str(x) for x in path_nodes)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(
        f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP\n"
        f"full exact path: {path_str} | matches={full_match_count}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "path_nodes": path_nodes,
        "full_match_count": full_match_count,
        "transition_counts": transition_counts,
    }


def plot_saved_umap_by_in_out_transition(
    umap_path: str | Path,
    cmap: str = "bwr",
    point_size: float = 6,
    alpha: float = 0.85,
    title_suffix: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    in_label: str = "in",
    out_label: str = "out",
):
    """Plot a saved UMAP colored by in/out transition direction."""
    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    nodes_used = np.asarray(saved["nodes_used"]).astype(int)

    if "prev_nodes_used" not in saved:
        raise KeyError(
            "Saved file does not contain 'prev_nodes_used'. "
            "Re-save the UMAP with save_umap_embedding(...), which stores previous-node info."
        )

    prev_nodes_used = np.asarray(saved["prev_nodes_used"]).astype(int)

    if len(z_umap) != len(nodes_used) or len(z_umap) != len(prev_nodes_used):
        raise ValueError(
            "Length mismatch among Z_umap, nodes_used, and prev_nodes_used."
        )

    color_vals = np.full(len(nodes_used), np.nan, dtype=float)

    valid_prev = prev_nodes_used >= 0
    is_out = valid_prev & (nodes_used > prev_nodes_used)
    is_in = valid_prev & (nodes_used < prev_nodes_used)

    color_vals[is_in] = 0.0
    color_vals[is_out] = 1.0

    plt.figure(figsize=(7.2, 6.2))

    undefined_mask = ~np.isfinite(color_vals)
    if np.any(undefined_mask):
        plt.scatter(
            z_umap[undefined_mask, 0],
            z_umap[undefined_mask, 1],
            c="lightgray",
            s=point_size,
            alpha=0.25,
            linewidths=0,
        )

    defined_mask = np.isfinite(color_vals)
    if np.any(defined_mask):
        sc = plt.scatter(
            z_umap[defined_mask, 0],
            z_umap[defined_mask, 1],
            c=color_vals[defined_mask],
            s=point_size,
            cmap=cmap,
            vmin=0,
            vmax=1,
            alpha=alpha,
            linewidths=0,
        )

        cbar = plt.colorbar(sc, ticks=[0, 1])
        cbar.ax.set_yticklabels([in_label, out_label])
        cbar.set_label("transition type")
    else:
        sc = None

    title = (
        f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP "
        f"colored by in/out transition"
    )
    if title_suffix:
        title += f"\n{title_suffix}"

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix.lower() == ".svg":
            plt.savefig(save_path, format="svg")
        else:
            plt.savefig(save_path, dpi=dpi)

    plt.show()

    return {
        "Z_umap": z_umap,
        "nodes_used": nodes_used,
        "prev_nodes_used": prev_nodes_used,
        "color_vals": color_vals,
        "n_in": int(np.sum(is_in)),
        "n_out": int(np.sum(is_out)),
        "n_undefined": int(np.sum(~np.isfinite(color_vals))),
        "saved": saved,
    }


def describe_saved_umap(umap_path: str | Path) -> dict[str, Any]:
    """Summarize a saved UMAP payload."""
    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    metrics_used = saved.get("metrics_used", {})
    metric_shapes = {k: tuple(np.asarray(v).shape) for k, v in metrics_used.items()}

    summary = {
        "mouse": saved.get("mouse"),
        "z_key": saved.get("z_key"),
        "n_points": len(saved.get("Z_umap", [])),
        "group_frac": saved.get("group_frac"),
        "exclude_first_n": saved.get("exclude_first_n"),
        "available_color_by": [
            "node",
            "prev_node",
            "bout_id",
            "time",
            "time_normalized",
            *sorted(metrics_used.keys()),
        ],
        "metric_shapes": metric_shapes,
    }
    return summary


def plot_saved_umap_highlight_path_node_colors(
    umap_path: str | Path,
    path_nodes: list[int],
    background_color: str = "lightgray",
    cmap: str = "nipy_spectral",
    s_bg: float = 6,
    s_hi: float = 18,
    alpha_bg: float = 0.25,
    alpha_hi: float = 1.0,
    save_path: str | Path | None = None,
    dpi: int = 300,
):
    if len(path_nodes) < 2:
        raise ValueError("path_nodes must contain at least 2 nodes.")

    umap_path = Path(umap_path)
    with open(umap_path, "rb") as f:
        saved = pickle.load(f)

    z_umap = np.asarray(saved["Z_umap"])
    nodes = np.asarray(saved["nodes_used"]).astype(int)
    prev_nodes = np.asarray(saved["prev_nodes_used"]).astype(int)

    transitions = list(zip(path_nodes[:-1], path_nodes[1:]))

    norm = Normalize(vmin=np.nanmin(nodes), vmax=np.nanmax(nodes))
    cmap_obj = colormaps[cmap]

    plt.figure(figsize=(7.2, 6.2))

    plt.scatter(
        z_umap[:, 0],
        z_umap[:, 1],
        c=background_color,
        s=s_bg,
        alpha=alpha_bg,
        linewidths=0,
    )

    transition_counts = {}

    for prev_node, curr_node in transitions:
        mask = (prev_nodes == int(prev_node)) & (nodes == int(curr_node))
        transition_counts[(prev_node, curr_node)] = int(mask.sum())

        if np.any(mask):
            color = cmap_obj(norm(curr_node))

            plt.scatter(
                z_umap[mask, 0],
                z_umap[mask, 1],
                color=color,
                s=s_hi,
                alpha=alpha_hi,
                linewidths=0,
                label=f"{prev_node} -> {curr_node}",
            )

    path_str = " -> ".join(str(x) for x in path_nodes)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(
        f"{saved.get('mouse', 'mouse')} | {saved.get('z_key', 'z')} UMAP\n"
        f"path transitions: {path_str}"
    )

    plt.legend(title="Transitions")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix.lower() == ".svg":
            plt.savefig(save_path, format="svg")
        else:
            plt.savefig(save_path, dpi=dpi)

    plt.show()

    return {
        "path_nodes": path_nodes,
        "transitions": transitions,
        "transition_counts": transition_counts,
    }