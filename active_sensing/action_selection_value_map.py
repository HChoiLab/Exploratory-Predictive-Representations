import argparse
import csv
from pathlib import Path
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_sensing.utils.saving import _append_summary_csv, _save_bout_npz
from active_sensing.core.perception import SequentialMazePerception


from collections import deque


# ============================================================
# Action mapping
# Maze adjacency uses:
#   -1 = left
#    1 = right
#    2 = back-to-parent
# Model action indices must be in [0, 1, 2]
# ============================================================
ACTION_TO_INDEX = {-1: 0, 1: 1, 2: 2}
INDEX_TO_ACTION = {0: -1, 1: 1, 2: 2}
ACTION_NAME = {0: "left", 1: "right", 2: "back"}


# ============================================================
# Utilities
# ============================================================


def build_value_map_from_bouts(
    bouts,
    num_nodes: int = 127,
    gamma: float = 0.95,
    n_iters: int = 100,
    tau=0.1,
):
    """
    Build value map from exploratory bouts:
      1) reward_map from reward_anticipation_p
      2) transition graph from observed transitions
      3) value propagation over that graph

    Returns
    -------
    value_map : (num_nodes,)
    reward_map : (num_nodes,)
    transition_graph : dict[node] -> set(next_nodes)
    """

    # --------------------------------------------------
    # reward map
    # --------------------------------------------------
    reward_sum = np.zeros(num_nodes, dtype=np.float64)
    counts = np.zeros(num_nodes, dtype=np.int64)

    # --------------------------------------------------
    # transition graph
    # --------------------------------------------------
    transition_graph = {i: set() for i in range(num_nodes)}

    for b in bouts:
        obs = np.asarray(b["obs_full"]).astype(int)
        rew_pred = np.asarray(b["reward_anticipation_p"])

        L = min(len(rew_pred), len(obs) - 1)
        if L <= 0:
            continue

        nodes_next = obs[1:1+L]
        nodes_prev = obs[:L]
        rew_pred = rew_pred[:L]

        # reward map
        for node, val in zip(nodes_next, rew_pred):
            reward_sum[node] += float(val)
            counts[node] += 1

        for u, v in zip(nodes_prev, nodes_next):
            transition_graph[int(u)].add(int(v))

    reward_map = np.zeros(num_nodes, dtype=np.float64)
    valid_nodes = counts > 0
    reward_map[valid_nodes] = reward_sum[valid_nodes] / counts[valid_nodes]

    # --------------------------------------------------
    # value propagation
    # --------------------------------------------------
    value = reward_map.copy()

    for _ in range(n_iters):
        new_value = value.copy()

        for node in range(num_nodes):
            next_nodes = transition_graph[node]

            if len(next_nodes) == 0:
                continue
            
            vals = np.array([value[n] for n in next_nodes])
            soft = np.log(np.sum(np.exp(vals / tau))) * tau

            new_value[node] = reward_map[node] + gamma * soft

        value = new_value

    return value, reward_map, transition_graph



def beta_warmup(bout_id: int, *, start: int, end: int, beta_final: float, kind: str = "linear") -> float:
    if bout_id <= start:
        return 0.0
    if bout_id >= end:
        return float(beta_final)

    x = (bout_id - start) / max(1, (end - start))

    if kind == "linear":
        w = x
    elif kind == "sigmoid":
        k = 10.0
        w = 1 / (1 + np.exp(-k * (x - 0.5)))
        w0 = 1 / (1 + np.exp(-k * (0 - 0.5)))
        w1 = 1 / (1 + np.exp(-k * (1 - 0.5)))
        w = (w - w0) / (w1 - w0)
    else:
        raise ValueError("kind must be 'linear' or 'sigmoid'")

    return float(beta_final * w)


# ===========================
# Build binary-maze adjacency
# ===========================
def build_binary_maze_adjacency(num_nodes: int = 127) -> np.ndarray:
    if num_nodes != 127:
        raise ValueError("This builder currently assumes only 1 maze with 127 nodes")

    l1 = [0]
    l2 = range(1, 3)
    l3 = range(3, 7)
    l4 = range(7, 15)
    l5 = range(15, 31)
    l6 = range(31, 63)
    l7 = range(63, 127)

    adj = np.zeros((127, 127), dtype=np.int64)

    # layer 1
    adj[0][1] = -1
    adj[0][2] = 1
    adj[1][0] = 2
    adj[2][0] = 2

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
        i += 1

    return adj


# ============================================================
# Maze environment
# ============================================================
class BinaryMazeEnv:
    def __init__(self, adj: np.ndarray, start_node: int = 0, waterport_node: int = 116):
        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.start_node = int(start_node)
        self.waterport_node = int(waterport_node)
        self.node = self.start_node

    def reset(self) -> int:
        self.node = self.start_node
        return self.node

    def valid_moves(self, node: int = None) -> List[Tuple[int, int]]:
        """
        Returns list of (action_idx, next_node) pairs.
        action_idx is in {0,1,2} for model compatibility.
        """
        node = self.node if node is None else int(node)
        next_nodes = np.where(self.adj[node] != 0)[0]
        moves = []
        for nxt in next_nodes:
            raw_action = int(self.adj[node, nxt])
            action_idx = ACTION_TO_INDEX[raw_action]
            moves.append((action_idx, int(nxt)))
        return moves

    def valid_action_indices(self, node: int = None) -> List[int]:
        return [a for a, _ in self.valid_moves(node)]

    def step(self, action_idx: int):
        raw_action = INDEX_TO_ACTION[int(action_idx)]
        indices = np.where(self.adj[self.node] == raw_action)[0]
        assert len(indices) == 1, "There should be only 1 node given previous node and action"

        next_node = int(indices[0])
        self.node = next_node
        reward = 1 if next_node == self.waterport_node else 0
        done = False
        info = {
            "raw_action": raw_action,
            "action_name": ACTION_NAME[int(action_idx)],
        }
        return next_node, reward, done, info


# ============================================================
# Model / loading utilities
# ============================================================
def _load_model(args, device: torch.device):
    model = SequentialMazePerception(
        num_nodes=args.num_nodes,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        lr=args.lr,
        beta=args.beta,
        reward_beta=args.reward_beta,
        sparse_beta=args.sparse_beta,
    )

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            print('------Found key "state_dict", loading that part of the checkpoint')
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            print('------Did not find key "state_dict", loading entire state dict directly')
            model.load_state_dict(state, strict=True)
        print("Model loaded.")
    else:
        print("No checkpoint provided. Training model from scratch.")

    
    return model.to(device)


# ============================================================
# Metric computation
# ============================================================
@torch.no_grad()
def _compute_eval_metrics(
    model: SequentialMazePerception,
    out: Dict[str, torch.Tensor],
    obs_b: torch.Tensor,
    action_scores_full: np.ndarray,
    valid_actions_mask_full: np.ndarray,
    chosen_action_names_full: np.ndarray,
    rewards_full: np.ndarray,
) -> Dict[str, np.ndarray]:
    assert obs_b.dim() == 2 and obs_b.size(0) == 1
    B, T = obs_b.shape
    assert B == 1

    targets = obs_b[:, 1:]

    logits_q = out["logits_q"]
    logits_p = out["logits_p"]

    mu_q, logvar_q = out["mu_q"], out["logvar_q"]
    mu_p, logvar_p = out["mu_p"], out["logvar_p"]
    z_q, z_p = out["z_q"], out["z_p"]
    h_list = out["h_list"]

    reward_logits_q = out["reward_logits_q"]
    reward_logits_p = out["reward_logits_p"]

    p_reward_q = torch.sigmoid(reward_logits_q)
    p_reward_p = torch.sigmoid(reward_logits_p)

    ce_q = F.cross_entropy(
        logits_q.reshape(-1, model.num_nodes),
        targets.reshape(-1),
        reduction="none",
    ).reshape(1, T - 1)

    ce_p = F.cross_entropy(
        logits_p.reshape(-1, model.num_nodes),
        targets.reshape(-1),
        reduction="none",
    ).reshape(1, T - 1)

    expected_H_q = out["expected_H_q"]
    expected_IG_kl = out["expected_IG_kl"]
    EIG = out["EIG"]
    H_p = out["H_p"]

    kl = model.kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p)

    pred_node_p = torch.argmax(logits_p, dim=-1)
    pred_node_q = torch.argmax(logits_q, dim=-1)
    mask_t = torch.ones((1, T - 1), dtype=torch.bool, device=obs_b.device)

    metrics: Dict[str, torch.Tensor] = {
        "mask": mask_t,
        "target_node": targets,
        "ce_q": ce_q,
        "ce_p": ce_p,
        "expected_H_q": expected_H_q,
        "H_p": H_p,
        "KL_qp": kl,
        "expected_IG_kl": expected_IG_kl,
        "EIG": EIG,
        "reward_anticipation_q": p_reward_q,
        "reward_anticipation_p": p_reward_p,
        "pred_node_p": pred_node_p,
        "pred_node_q": pred_node_q,
        "mu_q": mu_q,
        "logvar_q": logvar_q,
        "mu_p": mu_p,
        "logvar_p": logvar_p,
        "z_q": z_q,
        "z_p": z_p,
        "h_list": h_list,
    }

    out_np: Dict[str, np.ndarray] = {}
    for k, v in metrics.items():
        out_np[k] = v.squeeze(0).detach().cpu().numpy()

    out_np["action_scores_full"] = action_scores_full.astype(np.float32)
    out_np["valid_actions_mask_full"] = valid_actions_mask_full.astype(np.int64)
    out_np["chosen_action_names_full"] = chosen_action_names_full
    out_np["rewards_full"] = rewards_full.astype(np.int64)

    return out_np


def _compute_train_losses(
    model: SequentialMazePerception,
    out: Dict[str, torch.Tensor],
    obs_b: torch.Tensor,
    rew_targets: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    assert obs_b.dim() == 2 and obs_b.size(0) == 1
    B, T = obs_b.shape
    assert B == 1

    targets = obs_b[:, 1:]
    mask_t = torch.ones((1, T - 1), dtype=torch.bool, device=obs_b.device)

    logits_q = out["logits_q"]
    ce = F.cross_entropy(
        logits_q.reshape(-1, model.num_nodes),
        targets.reshape(-1),
        reduction="none",
    ).reshape(1, T - 1)
    rec_loss = model._masked_mean(ce, mask_t)

    mu_q, logvar_q = out["mu_q"], out["logvar_q"]
    mu_p, logvar_p = out["mu_p"], out["logvar_p"]
    kl_bt = model.kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p)
    kl_loss = model._masked_mean(kl_bt, mask_t)

    sparse_loss = out["mu_q"].abs().mean()

    reward_logits = out["reward_logits_q"]
    pos = rew_targets.sum()
    neg = rew_targets.numel() - pos

    model.rew_pos_count += pos.detach()
    model.rew_neg_count += neg.detach()

    pos_weight = (model.rew_neg_count / (model.rew_pos_count + 1e-8)).clamp(min=1.0)
    pos_weight = pos_weight.to(obs_b.device)

    reward_loss = F.binary_cross_entropy_with_logits(
        reward_logits,
        rew_targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    reward_loss = model._masked_mean(reward_loss, mask_t)

    total = rec_loss + model.beta * kl_loss + model.reward_beta * reward_loss + model.sparse_beta * sparse_loss

    return {
        "total": total,
        "rec": rec_loss,
        "kl": kl_loss,
        "sparse": sparse_loss,
        "reward": reward_loss,
    }




# ============================================================
# Action selection using reward driven rule
# ============================================================
@torch.no_grad()
def choose_action_value_map_driven(
    env: BinaryMazeEnv,
    current_node: int,
    valid_actions: List[int],
    value_map: np.ndarray,
):
    """
    Choose action leading to next node with highest learned value.
    """
    if len(valid_actions) == 0:
        raise RuntimeError(f"No valid actions at node {current_node}")

    best_action = None
    best_value = -np.inf
    action_scores_all = np.full((3,), -np.inf, dtype=np.float32)

    for a_idx, next_node in env.valid_moves(current_node):
        if a_idx not in valid_actions:
            continue

        v = float(value_map[next_node])
        action_scores_all[a_idx] = v

        if v > best_value:
            best_value = v
            best_action = a_idx

    if best_action is None:
        raise RuntimeError(f"Could not find value-map action at node {current_node}")

    return int(best_action), action_scores_all



# ============================================================
# Action selection using expected_IG_kl
# ============================================================
@torch.no_grad()
def choose_action_expected_ig(
    model: SequentialMazePerception,
    h_t: torch.Tensor,
    current_node: int,
    valid_actions: List[int],
    device: torch.device,
    k_info_gain: int = 10,
):
    """
    Scores each valid candidate action with model.calculate_EIG(...)
    and chooses random action with weights proportional to EIG.

    Returns
    -------
    chosen_action : int
        Action index in {0,1,2}
    action_scores_all : np.ndarray
        Shape (num_actions,), filled with -inf for invalid actions
    """
    x_t = F.one_hot(
        torch.tensor([current_node], device=device),
        num_classes=model.num_nodes,
    ).float()

    action_scores_all = np.full((model.num_actions,), -np.inf, dtype=np.float32)

    valid_scores = []
    valid_action_list = []


    for a_idx in valid_actions:
        a_vec = F.one_hot(
            torch.tensor([a_idx], device=device),
            num_classes=model.num_actions,
        ).float()

        prior_in = torch.cat([h_t, a_vec, x_t], dim=-1)
        mu_p, logvar_p = torch.chunk(model.prior_net(prior_in), 2, dim=-1)

        info = model.calculate_EIG(
            mu_p=mu_p,
            logvar_p=logvar_p,
            a_prev=a_vec,
            h_prev=h_t,
            k=k_info_gain,
            deterministic_z=True,
        )

        score = float(info["EIG"].item())
        action_scores_all[a_idx] = score

        valid_scores.append(score)
        valid_action_list.append(a_idx)


        
    valid_scores = np.maximum(valid_scores, 0.0)
    valid_scores = np.asarray(valid_scores, dtype=np.float64)
    # avoid division by zero
    if valid_scores.sum()  <= 1e-12:
        # print(f"Warning: all valid actions have almost zero {valid_scores.sum()} expected_IG_kl, using uniform probabilities")
        probs = np.ones_like(valid_scores) / len(valid_scores)
    else:
        probs = valid_scores / valid_scores.sum()

    chosen_action = int(np.random.choice(valid_action_list, p=probs))

    return chosen_action, action_scores_all


# ============================================================
# Generate one active bout from scratch
# ============================================================
@torch.no_grad()
def generate_active_bout(
    env: BinaryMazeEnv,
    model: SequentialMazePerception,
    horizon: int,
    device: torch.device,
    k_info_gain: int = 10,
    deterministic_belief_update: bool = True,
    reward_driven_p: float = 0.0,
    explore_after_reward_steps: int = 0,
    value_map: np.ndarray = None,
):
    """
    Generates one trajectory by starting from node 0 and selecting actions that maximize
    expected_IG_kl or value among valid actions.
    
      obs[t] is the node at time t
      act[t] is the action that led from obs[t-1] to obs[t]
      act[0] is a dummy action and is ignored by the model
    """
    model.eval()

    start_node = env.reset()

    obs = [int(start_node)]
    acts = [0]  # dummy action at t=0
    rews = [1 if start_node == env.waterport_node else 0]

    action_scores_per_t = [np.full((model.num_actions,), np.nan, dtype=np.float32)]
    valid_actions_mask_per_t = [np.zeros((model.num_actions,), dtype=np.int64)]
    chosen_action_names = ["dummy"]

    h = torch.zeros(1, model.hidden_dim, device=device)
    
    if value_map is None:
        raise ValueError("value_map must be provided")

    forced_explore_countdown = 0
    action_mode_per_t = ["dummy"]
    reward_drive_active = False


    for t in range(1, horizon):
        current_node = obs[-1]
        valid_actions = env.valid_action_indices(current_node)

        valid_mask = np.zeros((model.num_actions,), dtype=np.int64)
        for a in valid_actions:
            valid_mask[a] = 1

        use_reward_driven = False

        if forced_explore_countdown > 0:
            use_reward_driven = False
            forced_explore_countdown -= 1
            reward_drive_active = False
        elif reward_drive_active:
            use_reward_driven = True
        elif np.random.rand() < reward_driven_p:
            use_reward_driven = True
            reward_drive_active = True

        if use_reward_driven:
            chosen_action, action_scores = choose_action_value_map_driven(
                env=env,
                current_node=current_node,
                valid_actions=valid_actions,
                value_map=value_map,
            )
            action_mode_per_t.append("reward_driven")
        else:
            chosen_action, action_scores = choose_action_expected_ig(
                model=model,
                h_t=h,
                current_node=current_node,
                valid_actions=valid_actions,
                device=device,
                k_info_gain=k_info_gain,
            )
            action_mode_per_t.append("explore")


        # --------------------------------------------------
        # retry once if going to previous node
        # --------------------------------------------------
        if not use_reward_driven and len(obs) >= 2:
            prev_node = obs[-2]
            # find where chosen action goes
            next_node_candidate = None
            for a_idx, nxt in env.valid_moves(current_node):
                if a_idx == chosen_action:
                    next_node_candidate = nxt
                    break
            if next_node_candidate == prev_node:
                # resample once
                chosen_action, action_scores = choose_action_expected_ig(
                    model=model,
                    h_t=h,
                    current_node=current_node,
                    valid_actions=valid_actions,
                    device=device,
                    k_info_gain=k_info_gain,
                )
        next_node, reward, done, info = env.step(chosen_action)
        
        if reward == 1:
            forced_explore_countdown = explore_after_reward_steps
            reward_drive_active = False

        obs.append(int(next_node))
        acts.append(int(chosen_action))
        rews.append(int(reward))
        action_scores_per_t.append(action_scores)
        valid_actions_mask_per_t.append(valid_mask)
        chosen_action_names.append(info["action_name"])

        # belief update with the actual observed next node
        x_next = F.one_hot(
            torch.tensor([next_node], device=device),
            num_classes=model.num_nodes,
        ).float()

        a_vec = F.one_hot(
            torch.tensor([chosen_action], device=device),
            num_classes=model.num_actions,
        ).float()

        post_in = torch.cat([h, a_vec, x_next], dim=-1)
        mu_q, logvar_q = torch.chunk(model.post_net(post_in), 2, dim=-1)

        if deterministic_belief_update:
            z_q = mu_q
        else:
            std_q = torch.exp(0.5 * logvar_q)
            z_q = mu_q + torch.randn_like(std_q) * std_q

        rnn_in = torch.cat([a_vec, z_q], dim=-1)
        h = model.rnn(rnn_in, h)

    return {
        "observations": np.asarray(obs, dtype=np.int64),
        "actions": np.asarray(acts, dtype=np.int64),
        "rewards": np.asarray(rews, dtype=np.int64),
        "action_scores_full": np.stack(action_scores_per_t, axis=0),
        "valid_actions_mask_full": np.stack(valid_actions_mask_per_t, axis=0),
        "chosen_action_names_full": np.asarray(chosen_action_names, dtype=object),
        "action_mode_full": np.asarray(action_mode_per_t, dtype=object),
    }


def save_checkpoint(model: torch.nn.Module, ckpt_dir: Path, bout_id: int):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"bout_{bout_id:05d}.pt"
    torch.save({"state_dict": model.state_dict(), "bout_id": bout_id}, ckpt_path)



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




def main(args):
    
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if args.num_nodes != 127:
        raise ValueError("This script currently assumes the 127-node binary maze.")
    if args.num_actions != 3:
        raise ValueError("This script expects num_actions=3 for left/right/back.")

    adj = build_binary_maze_adjacency(num_nodes=args.num_nodes)
    env = BinaryMazeEnv(
        adj=adj,
        start_node=args.start_node,
        waterport_node=args.waterport_node,
    )

    model = _load_model(args, device)
    params = model.configure_params()
    optimizer_prior = torch.optim.Adam(params[0], lr=args.online_lr_prior)
    optimizer_posterior = torch.optim.Adam(params[1], lr=args.online_lr_posterior)
    optimizer_other = torch.optim.Adam(params[2], lr=args.online_lr_other)

    metrics_out_dir = Path(args.metrics_out_dir)
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    value_maps_dir = metrics_out_dir / "value_maps"
    value_maps_dir.mkdir(parents=True, exist_ok=True)


    if args.initial_value_map is not None:
        loaded = np.load(args.initial_value_map)
        value_map = loaded["value_map"]
        reward_map = loaded["reward_map"]
        print("Loaded initial maps from:", args.initial_value_map)
        assert value_map.shape[0] == args.num_nodes
        assert reward_map.shape[0] == args.num_nodes
    else:
        # raise ValueError("No initial value map")
        print("No initial value map provided, starting from zero map (pure exploration first)")
        value_map = np.zeros(args.num_nodes, dtype=np.float64)
        reward_map = np.zeros(args.num_nodes, dtype=np.float64)


    recent_bouts_for_value = []

    summary_csv = Path(args.summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(args.online_ckpt_dir) if args.online_ckpt_dir else None

    print("WATERPORT NODE:", args.waterport_node)
    print("START NODE:", args.start_node)
    print("HORIZON:", args.horizon)
    print("NUM BOUTS:", args.num_bouts)

    for bout_id in tqdm(range(1, args.num_bouts + 1), desc="Active bouts"):
        model.beta = beta_warmup(
            bout_id,
            start=args.beta_warmup_start,
            end=args.beta_warmup_end,
            beta_final=args.beta,
            kind=args.beta_warmup_kind,
        )

        # ----------------------------------------------------
        # Generate a fresh active trajectory from scratch
        # ----------------------------------------------------
        if args.horizon == 0:
            horizon = np.random.randint(50, 91)  # 50 to 90 inclusive
        else:
            horizon = args.horizon

        traj = generate_active_bout(
            env=env,
            model=model,
            horizon=horizon,
            device=device,
            k_info_gain=args.k_info_gain,
            deterministic_belief_update=not args.sample_online_belief,
            reward_driven_p=args.reward_driven_p,
            explore_after_reward_steps=args.explore_after_reward_steps,
            value_map=value_map,
        )

        obs_np = traj["observations"]
        act_np = traj["actions"]
        rew_np = traj["rewards"]
        action_scores_full = traj["action_scores_full"]
        valid_actions_mask_full = traj["valid_actions_mask_full"]
        chosen_action_names_full = traj["chosen_action_names_full"]
        action_mode_full = traj["action_mode_full"]

        obs_b = torch.from_numpy(obs_np).long().to(device).unsqueeze(0)
        act_b = torch.from_numpy(act_np).long().to(device).unsqueeze(0)
        rew_b = torch.from_numpy(rew_np).long().to(device).unsqueeze(0)
        rew_targets = rew_b[:, 1:].float()

        # ----------------------------------------------------
        # Evaluate before training
        # ----------------------------------------------------
        model.eval()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        with torch.no_grad():
            out_eval = model.forward(
                obs_b,
                act_b,
                rew_b,
                deterministic_z=True,
                calculate_expected_info=True,
            )
            eval_arrays = _compute_eval_metrics(
                model,
                out_eval,
                obs_b,
                action_scores_full=action_scores_full,
                valid_actions_mask_full=valid_actions_mask_full,
                chosen_action_names_full=chosen_action_names_full,
                rewards_full=rew_np,
            )

        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

        eval_arrays["obs_full"] = obs_np
        eval_arrays["act_full"] = act_np
        eval_arrays["rew_full"] = rew_np
        eval_arrays["action_mode_full"] = action_mode_full

        # ----------------------------------------------------
        # Train on the generated bout
        # ----------------------------------------------------
        model.train()
        train_window = args.train_window
        last_losses = None
        h_online = None

        for start in range(0, obs_b.size(1) - 1, train_window - 1):
            end = min(start + train_window, obs_b.size(1))

            obs_chunk = obs_b[:, start:end]
            act_chunk = act_b[:, start:end]
            rew_chunk = rew_b[:, start:end]

            if obs_chunk.size(1) < 2:
                continue

            rew_targets_chunk = rew_chunk[:, 1:].float()

            optimizer_prior.zero_grad(set_to_none=True)
            optimizer_posterior.zero_grad(set_to_none=True)
            optimizer_other.zero_grad(set_to_none=True)

            out_train = model.forward(
                obs_chunk,
                act_chunk,
                rew_chunk,
                deterministic_z=False,
                h_init=h_online,
                calculate_expected_info=False,
            )

            losses = _compute_train_losses(
                model,
                out_train,
                obs_chunk,
                rew_targets=rew_targets_chunk,
            )

            losses["total"].backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer_prior.step()
            optimizer_posterior.step()
            optimizer_other.step()

            h_online = out_train["h"].detach()
            last_losses = losses
        

        # ----------------------------------------------------
        # Save full per-bout arrays
        # ----------------------------------------------------
        _save_bout_npz(metrics_out_dir, bout_id, eval_arrays)

        recent_bouts_for_value.append(eval_arrays)

        if len(recent_bouts_for_value) > args.value_update_window:
            recent_bouts_for_value = recent_bouts_for_value[-args.value_update_window:]

        if bout_id % args.value_update_every == 0 and len(recent_bouts_for_value) > 0:
            value_map, reward_map, transition_graph = build_value_map_from_bouts(
                recent_bouts_for_value,
                num_nodes=args.num_nodes,
                gamma=args.value_gamma,
                n_iters=args.value_iters,
                tau=args.value_tau,
            )

            np.savez(
                value_maps_dir / f"value_map_bout_{bout_id:05d}.npz",
                value_map=value_map,
                reward_map=reward_map,
                counts=np.array([
                    len(transition_graph[i]) for i in range(args.num_nodes)
                ]),
            )

        mask_eval = eval_arrays["mask"].astype(bool)

        def _mmean(x: np.ndarray) -> float:
            if mask_eval.any():
                return float(x[mask_eval].mean())
            return float(np.nan)

        summary_row = {
            "bout_id": int(bout_id),
            "beta": float(model.beta),
            "total_loss": float(last_losses["total"].item()),
            "rec_loss": float(last_losses["rec"].item()),
            "kl_loss": float(last_losses["kl"].item()),
            "reward_loss": float(last_losses["reward"].item()),
            "sparse_loss": float(last_losses["sparse"].item()),
            "mean_ce_q": _mmean(eval_arrays["ce_q"]),
            "mean_ce_p": _mmean(eval_arrays["ce_p"]),
            "mean_expected_H_q": _mmean(eval_arrays["expected_H_q"]),
            "mean_H_p": _mmean(eval_arrays["H_p"]),
            "mean_KL_qp": _mmean(eval_arrays["KL_qp"]),
            "mean_expected_IG_kl": _mmean(eval_arrays["expected_IG_kl"]),
            "mean_EIG": _mmean(eval_arrays["EIG"]),
            "mean_reward_anticipation_q": _mmean(eval_arrays["reward_anticipation_q"]),
            "mean_reward_anticipation_p": _mmean(eval_arrays["reward_anticipation_p"]),
            "traj_len": int(len(obs_np)),
            "start_node": int(obs_np[0]),
            "final_node": int(obs_np[-1]),
            "n_rewards": int(rew_np.sum()),
        }
        _append_summary_csv(summary_csv, summary_row)

    if ckpt_dir is not None:
        save_checkpoint(model, ckpt_dir, bout_id)

    print("Done.")
    print("Per-bout metrics saved in:", metrics_out_dir)
    print("Summary CSV:", summary_csv)
    if ckpt_dir is not None:
        print("Checkpoints saved in:", ckpt_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--ckpt", default=None, help="Optional initial checkpoint to start active training from")
    p.add_argument("--metrics-out-dir", default="runs/active_maze_metrics", help="Where to save per-bout metrics .npz")
    p.add_argument("--summary-csv", default="runs/active_maze_metrics/summary.csv", help="Where to append bout summary rows")
    p.add_argument("--online-ckpt-dir", default=None, help="If set, save checkpoint each bout to this dir")

    # Model hyperparameters
    p.add_argument("--num-nodes", type=int, default=127)
    p.add_argument("--num-actions", type=int, default=3)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--waterport-node", type=int, default=116)
    p.add_argument("--start-node", type=int, default=0)
    p.add_argument("--beta", type=float, default=1e-2)
    p.add_argument("--sparse-beta", type=float, default=1e-2)
    p.add_argument("--reward-beta", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)

    # Optimization
    p.add_argument("--online-lr-prior", type=float, default=1e-4)
    p.add_argument("--online-lr-posterior", type=float, default=1e-4)
    p.add_argument("--online-lr-other", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--train-window", type=int, default=10)

    # Active data generation
    p.add_argument("--num-bouts", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=20, help="Number of observations per generated bout")
    p.add_argument("--k-info-gain", type=int, default=10, help="Monte Carlo samples used in expected_IG_kl estimation")
    p.add_argument("--sample-online-belief", action="store_true", help="If set, sample z_q during online acting instead of using mu_q")

    p.add_argument("--value-update-every", type=int, default=50)
    p.add_argument("--value-update-window", type=int, default=50)
    p.add_argument("--value-gamma", type=float, default=0.9)
    p.add_argument("--value-iters", type=int, default=100)
    p.add_argument("--value-tau", type=float, default=0.1)
    p.add_argument("--initial-value-map", default=None)

    # Beta warmup
    p.add_argument("--beta-warmup-start", type=int, default=50)
    p.add_argument("--beta-warmup-end", type=int, default=100)
    p.add_argument("--beta-warmup-kind", type=str, default="sigmoid", choices=["linear", "sigmoid"])

    p.add_argument(
        "--reward-driven-p",
        type=float,
        default=0.0,
        help="Probability of choosing action using learned value map"
    )
    p.add_argument(
        "--explore-after-reward-steps",
        type=int,
        default=7,
        help="Number of exploratory steps forced after reaching the reward node"
    )

    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    main(args)