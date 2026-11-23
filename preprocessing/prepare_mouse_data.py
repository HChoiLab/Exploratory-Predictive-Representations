
import argparse
import pickle
from pathlib import Path

import numpy as np


# Maze adjacency

def build_maze_adjacency():
    """
    127-node binary maze.

        -1: left child
         1: right child
         2: parent
         3: neighboring child (CC)
    """
    adj = np.zeros((127, 127), dtype=np.int64)

    adj[0, 1] = -1
    adj[0, 2] = 1
    adj[1, 0] = 2
    adj[2, 0] = 2
    adj[1, 2] = 3

    levels = [
        (range(1, 3), -1),
        (range(3, 7), 1),
        (range(7, 15), -1),
        (range(15, 31), 1),
        (range(31, 63), -1),
    ]

    for nodes, start_action in levels:
        coef = 1

        for i in nodes:
            left = 2 * i + 1
            right = 2 * i + 2

            adj[i, left] = start_action * coef
            adj[i, right] = -start_action * coef

            adj[left, i] = 2
            adj[right, i] = 2

            adj[left, right] = 3
            adj[right, left] = 3

            coef *= -1

    return adj


# Convert frame information to node sequence

def node_per_frame_from_no(no_bout, T):
    node_per_frame = np.full(T, np.nan, dtype=float)
    no_arr = np.asarray(no_bout)

    if no_arr.size == 0:
        return node_per_frame

    for i, (node_id, start_frame) in enumerate(no_arr):
        start = int(start_frame)
        end = int(no_arr[i + 1, 1]) if i + 1 < len(no_arr) else T

        start = max(0, min(start, T))
        end = max(start, min(end, T))

        node_per_frame[start:end] = int(node_id)

    return node_per_frame


def compress_nodes(node_pf):
    valid = ~np.isnan(node_pf)
    seq = node_pf[valid].astype(int)

    if seq.size == 0:
        return np.array([], dtype=np.int64)

    keep = np.ones(len(seq), dtype=bool)
    keep[1:] = seq[1:] != seq[:-1]

    return seq[keep]


# Convert transitions into actions

def actions_from_adjacency(obs, adj, tf_name, bout_idx):
    actions = np.zeros(len(obs), dtype=np.int64)

    action_mapping = {
        -1: 0,  # left child
         1: 1,  # right child
         2: 2,  # parent
         3: 3,  # neighboring child
    }

    for i in range(1, len(obs)):
        u = int(obs[i - 1])
        v = int(obs[i])

        val = int(adj[u, v])

        if val not in action_mapping:
            raise ValueError(
                f"[{tf_name}] bout={bout_idx}, step={i}: "
                f"Invalid transition {u} -> {v}, "
                f"adjacency value={val}"
            )

        actions[i] = action_mapping[val]

    return actions


# Dataset generation

def prepare_mouse_data(tf_dir, out_dir, min_transitions=5):
    tf_dir = Path(tf_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adj = build_maze_adjacency()
    tf_files = sorted(tf_dir.glob("*-tf"))

    print(f"Found {len(tf_files)} mouse recordings")

    total_bouts = 0
    saved = 0
    skipped_empty = 0
    skipped_short = 0
    sibling_transitions = 0

    for tf_path in tf_files:
        print(f"Processing {tf_path.name}")

        # Only load trusted pickle files.
        with open(tf_path, "rb") as f:
            content = pickle.load(f)

        num_bouts = len(content.ke)
        total_bouts += num_bouts

        for b in range(num_bouts):
            ke = np.asarray(content.ke[b])
            no = np.asarray(content.no[b])

            if ke.size == 0 or no.size == 0:
                skipped_empty += 1
                continue

            T = ke.shape[0]

            node_pf = node_per_frame_from_no(no, T)
            obs = compress_nodes(node_pf)

            if obs.size == 0:
                skipped_empty += 1
                continue

            if len(obs) - 1 < min_transitions:
                skipped_short += 1
                continue

            actions = actions_from_adjacency(
                obs, adj, tf_path.name, b
            )

            sibling_transitions += int(
                np.count_nonzero(actions[1:] == 3)
            )

            mouse_dir = out_dir / tf_path.name
            mouse_dir.mkdir(parents=True, exist_ok=True)

            out_path = (
                mouse_dir /
                f"{tf_path.name}__bout_{b:04d}.npz"
            )

            np.savez(
                out_path,
                observations=obs.astype(np.int64),
                actions=actions.astype(np.int64),
            )

            saved += 1

    print("\nDone")
    print("Total recordings:", len(tf_files))
    print("Total bouts:", total_bouts)
    print("Saved bouts:", saved)
    print("Skipped empty:", skipped_empty)
    print("Skipped short:", skipped_short)
    print("Sibling transitions (action 3):", sibling_transitions)
    print("Output directory:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare mouse maze trajectories"
    )

    parser.add_argument(
        "--tf-dir",
        type=str,
        required=True,
        help="Directory containing original *-tf files",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="data",
        help="Directory for preprocessed .npz bouts",
    )

    parser.add_argument(
        "--min-transitions",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    prepare_mouse_data(
        tf_dir=args.tf_dir,
        out_dir=args.out_dir,
        min_transitions=args.min_transitions,
    )

