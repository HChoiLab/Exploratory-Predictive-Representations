from pathlib import Path
from typing import Optional, List
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


# Graph utilities
def build_tree_adj_127() -> np.ndarray:
    """Build a 127-node adjacency matrix."""
    l1 = [0]
    l2 = range(1, 3)
    l3 = range(3, 7)
    l4 = range(7, 15)
    l5 = range(15, 31)
    l6 = range(31, 63)
    l7 = range(63, 127)

    adj = np.zeros((127, 127), dtype=np.int8)

    # layer 1
    adj[0][1] = -1
    adj[0][2] = 1
    adj[1][0] = 2
    adj[2][0] = 2
    adj[1][2] = 3

    # layer 2
    start_action = -1
    i = min(l2)
    coef = 1
    while i * 2 + 2 <= max(l3):
        adj[i][i * 2 + 1] = start_action * coef
        adj[i][i * 2 + 2] = -1 * start_action * coef
        coef *= -1
        adj[i * 2 + 1][i] = 2
        adj[i * 2 + 2][i] = 2
        adj[i * 2 + 1][i * 2 + 2] = 3
        adj[i * 2 + 2][i * 2 + 1] = 3
        i += 1

    # layer 3
    start_action = 1
    i = min(l3)
    coef = 1
    while i * 2 + 2 <= max(l4):
        adj[i][i * 2 + 1] = start_action * coef
        adj[i][i * 2 + 2] = -1 * start_action * coef
        coef *= -1
        adj[i * 2 + 1][i] = 2
        adj[i * 2 + 2][i] = 2
        adj[i * 2 + 1][i * 2 + 2] = 3
        adj[i * 2 + 2][i * 2 + 1] = 3
        i += 1

    # layer 4
    start_action = -1
    i = min(l4)
    coef = 1
    while i * 2 + 2 <= max(l5):
        adj[i][i * 2 + 1] = start_action * coef
        adj[i][i * 2 + 2] = -1 * start_action * coef
        coef *= -1
        adj[i * 2 + 1][i] = 2
        adj[i * 2 + 2][i] = 2
        adj[i * 2 + 1][i * 2 + 2] = 3
        adj[i * 2 + 2][i * 2 + 1] = 3
        i += 1

    # layer 5
    start_action = 1
    i = min(l5)
    coef = 1
    while i * 2 + 2 <= max(l6):
        adj[i][i * 2 + 1] = start_action * coef
        adj[i][i * 2 + 2] = -1 * start_action * coef
        coef *= -1
        adj[i * 2 + 1][i] = 2
        adj[i * 2 + 2][i] = 2
        adj[i * 2 + 1][i * 2 + 2] = 3
        adj[i * 2 + 2][i * 2 + 1] = 3
        i += 1

    # layer 6
    start_action = -1
    i = min(l6)
    coef = 1
    while i * 2 + 2 <= max(l7):
        adj[i][i * 2 + 1] = start_action * coef
        adj[i][i * 2 + 2] = -1 * start_action * coef
        coef *= -1
        adj[i * 2 + 1][i] = 2
        adj[i * 2 + 2][i] = 2
        adj[i * 2 + 1][i * 2 + 2] = 3
        adj[i * 2 + 2][i * 2 + 1] = 3
        i += 1

    return adj


def build_neighbors_from_adj(adj: np.ndarray) -> List[List[int]]:
    neighbors = []
    for i in range(adj.shape[0]):
        neighbors.append(np.flatnonzero(adj[i] != 0).tolist())
    return neighbors


def bfs_distances(neighbors: List[List[int]], start: int, n_nodes: int) -> np.ndarray:
    dist = np.full(n_nodes, fill_value=np.inf, dtype=np.float32)
    dist[start] = 0.0
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in neighbors[u]:
            if dist[v] == np.inf:
                dist[v] = du + 1.0
                q.append(v)
    return dist


def make_distance_sequence(obs_pad: torch.Tensor, dist_to_water: np.ndarray, *, mode: str) -> torch.Tensor:
    """Return distance targets for padded observations."""
    B, T = obs_pad.shape
    obs_np = obs_pad.detach().cpu().numpy()
    dist_now = dist_to_water[obs_np].astype(np.float32)

    if mode == "now":
        out = dist_now
    elif mode == "prev":
        out = np.zeros_like(dist_now, dtype=np.float32)
        if T > 1:
            out[:, 1:] = dist_now[:, :-1]
        out[:, 0] = 0.0
    else:
        raise ValueError(f"Unknown dist_mode: {mode}")

    return torch.from_numpy(out)


# Data loading
class SequenceDataset(Dataset):
    def __init__(self, observations, locations=None):
        self.obs = observations
        self.locs = locations

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        obs_t = torch.from_numpy(np.asarray(obs)).long()

        if self.locs is None or self.locs[idx] is None:
            return obs_t, None

        loc = self.locs[idx]
        loc_t = torch.from_numpy(np.asarray(loc)).long()
        return obs_t, loc_t


def load_npz(path: str):
    p = Path(path)
    if p.is_dir():
        obs_list, loc_list = [], []
        for f in sorted(p.glob("*.npz")):
            d = np.load(f, allow_pickle=True)
            obs_list.append(d["observations"])
            loc_list.append(d["locations"] if "locations" in d else None)
        return obs_list, loc_list
    else:
        d = np.load(p, allow_pickle=True)
        return d["observations"], d["locations"] if "locations" in d else None


# Collation
def pad_collate(
    batch,
    *,
    pad_value_obs=0,
    pad_value_act=0,
    target_mode: str = "reward",
    reward_node: int = 116,
    dist_to_water: Optional[np.ndarray] = None,
    dist_mode: str = "prev",
):
    """Pad a batch and build reward or distance targets."""
    obs_list, act_list = zip(*batch)
    B = len(obs_list)
    lengths = torch.tensor([o.size(0) for o in obs_list], dtype=torch.long)
    T_max = int(lengths.max().item())

    obs_pad = torch.full((B, T_max), pad_value_obs, dtype=torch.long)
    mask = torch.zeros((B, T_max), dtype=torch.bool)

    if any(a is not None for a in act_list):
        act_pad = torch.full((B, T_max), pad_value_act, dtype=torch.long)
    else:
        act_pad = None

    for i, (o, a) in enumerate(zip(obs_list, act_list)):
        t = o.size(0)
        obs_pad[i, :t] = o
        mask[i, :t] = True
        if act_pad is not None and a is not None:
            assert a.size(0) == t
            act_pad[i, :t] = a

    if target_mode == "reward":
        target_pad = (obs_pad == reward_node).float()
        target_pad = target_pad * mask.float()
        return obs_pad, act_pad, target_pad, mask

    if target_mode == "distance":
        if dist_to_water is None:
            raise RuntimeError("target_mode='distance' requires dist_to_water (precomputed BFS distances).")
        target_pad = make_distance_sequence(obs_pad, dist_to_water, mode=dist_mode).float()
        target_pad = target_pad * mask.float()
        return obs_pad, act_pad, target_pad, mask

    raise ValueError(f"Unknown target_mode: {target_mode}")


# Lightning data module
class SequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        val_path: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle_train: bool = True,
        *,
        target_mode: str = "reward",
        waterport_node: int = 116,
        dist_mode: str = "prev",
        num_nodes: int = 127,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path or train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train

        self.target_mode = target_mode
        self.waterport_node = int(waterport_node)
        self.dist_mode = dist_mode
        self.num_nodes = int(num_nodes)

        self.dist_to_water: Optional[np.ndarray] = None

    def setup(self, stage: Optional[str] = None):
        obs, locs = load_npz(self.train_path)
        self.train_dataset = SequenceDataset(obs, locs)
        obs_val, locs_val = load_npz(self.val_path)
        self.val_dataset = SequenceDataset(obs_val, locs_val)

        # Precompute distances for distance targets.
        if self.target_mode == "distance":
            if self.num_nodes != 127:
                raise ValueError("build_tree_adj_127() is fixed for 127 nodes; set num_nodes=127.")
            adj = build_tree_adj_127()
            neighbors = build_neighbors_from_adj(adj)
            dist = bfs_distances(neighbors, start=self.waterport_node, n_nodes=adj.shape[0])
            if not np.isfinite(dist[0]):
                raise RuntimeError("Distance BFS failed.")
            self.dist_to_water = dist.astype(np.float32)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            collate_fn=lambda batch: pad_collate(
                batch,
                target_mode=self.target_mode,
                reward_node=self.waterport_node,
                dist_to_water=self.dist_to_water,
                dist_mode=self.dist_mode,
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda batch: pad_collate(
                batch,
                target_mode=self.target_mode,
                reward_node=self.waterport_node,
                dist_to_water=self.dist_to_water,
                dist_mode=self.dist_mode,
            ),
        )