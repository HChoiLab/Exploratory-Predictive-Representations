
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from active_sensing.utils.saving import _append_summary_csv, _save_bout_npz
from active_sensing.core.perception import PerceptionModel, SequentialMazePerception


BOUT_RE = re.compile(r"bout_(\d+)", re.IGNORECASE) # for file reading
def parse_bout_number(filename: str) -> int:
    m = BOUT_RE.search(filename)
    if not m:
        raise ValueError(f"Could not parse bout number from filename: {filename}")
    return int(m.group(1))


def beta_warmup(bout_id: int, *, start: int, end: int, beta_final: float, kind: str = "linear") -> float:
    """
    bout_id: actual bout number (1-indexed)
    start:   bout where warmup begins (beta ~ 0 before this)
    end:     bout where warmup finishes (beta = beta_final at/after this)
    kind:    "linear" or "sigmoid"
    """

    if bout_id <= start:
        return 0.0
    if bout_id >= end:
        return float(beta_final)

    x = (bout_id - start) / max(1, (end - start))  # in (0,1)

    if kind == "linear":
        w = x
    elif kind == "sigmoid":
        # smooth S-curve: slow start, faster middle, slow end
        # map x in (0,1) to sigmoid around 0.5
        k = 10.0
        w = 1 / (1 + np.exp(-k * (x - 0.5)))
        # normalize so w(0)=0, w(1)=1 approximately
        w0 = 1 / (1 + np.exp(-k * (0 - 0.5)))
        w1 = 1 / (1 + np.exp(-k * (1 - 0.5)))
        w = (w - w0) / (w1 - w0)
    else:
        raise ValueError("kind must be 'linear' or 'sigmoid'")

    return float(beta_final * w)




def _load_model(args, device: torch.device):    
    model = SequentialMazePerception(
                num_nodes=args.num_nodes,
                z_dim=args.z_dim,
                hidden_dim=args.hidden_dim,
                num_actions=args.num_actions,
                lr=args.lr,
                beta=args.beta, 
                reward_beta=args.reward_beta, 
                sparse_beta = args.sparse_beta
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
    return model.to(device)


# --------------------------
# Metrics computation
# --------------------------

def gaussian_entropy(logvar):
    # logvar shape: (B, T-1, z_dim)
    return 0.5 * (logvar + np.log(2 * np.pi * np.e)).sum(dim=-1)



@torch.no_grad()
def _compute_eval_metrics(
    model: PerceptionModel,
    out: Dict[str, torch.Tensor],
    obs_b: torch.Tensor
) -> Dict[str, np.ndarray]:
    """
    Compute per-timestep metrics for a SINGLE bout (B=1).
    Returns numpy arrays with time length (T-1).

    obs_b: (1, T) long node ids
    """
    
    assert obs_b.dim() == 2 and obs_b.size(0) == 1
    B, T = obs_b.shape
    assert B == 1

    # Targets correspond to x_t for t=1..T-1
    targets = obs_b[:, 1:]  # (1, T-1)

    # model outputs
    logits_q = out["logits_q"]  # (1, T-1, S)
    logits_p = out["logits_p"]  # (1, T-1, S)

    mu_q, logvar_q = out["mu_q"], out["logvar_q"]  # (1, T-1, z)
    mu_p, logvar_p = out["mu_p"], out["logvar_p"]  # (1, T-1, z)
    z_q, z_p = out["z_q"], out["z_p"]              # (1, T-1, z)

    h_list = out["h_list"]  # (1, T-1, H)

    reward_logits_q = out["reward_logits_q"] # (1, T-1)
    reward_logits_p = out["reward_logits_p"]  # (1, T-1)

    p_reward_q = torch.sigmoid(reward_logits_q)  # (1, T-1)
    p_reward_p = torch.sigmoid(reward_logits_p)  # (1, T-1)

    
    # --- Cross-entropies (NLL) per timestep ---
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

    # calculate expected IG
    expected_H_q = out["expected_H_q"]
    expected_IG_kl = out["expected_IG_kl"]
    EIG = out["EIG"]
    H_p = out["H_p"]
    
    # --- KL(q||p) per timestep (latent) ---
    kl = model.kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p)  # (1, T-1)

    # --- Predictions / reconstructions as node IDs ---
    pred_node_p = torch.argmax(logits_p, dim=-1)  # (1, T-1)
    pred_node_q = torch.argmax(logits_q, dim=-1)  # (1, T-1)
    mask_t = torch.ones((1, T - 1), dtype=torch.bool, device=obs_b.device)

    metrics: Dict[str, torch.Tensor] = {
        "mask": mask_t,  # (T-1,) bool
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

    # Convert to numpy, squeeze batch dim
    out_np: Dict[str, np.ndarray] = {}
    for k, v in metrics.items():
        out_np[k] = v.squeeze(0).detach().cpu().numpy()

    return out_np


def _compute_train_losses(
    model: PerceptionModel,
    out: Dict[str, torch.Tensor],
    obs_b: torch.Tensor,
    rew_targets: torch.Tensor
) -> Dict[str, torch.Tensor]:
    
    assert obs_b.dim() == 2 and obs_b.size(0) == 1
    B, T = obs_b.shape
    assert B == 1


    targets = obs_b[:, 1:]  # (1, T-1)

    mask_t = torch.ones((1, T - 1), dtype=torch.bool, device=obs_b.device)

    # Reconstruction CE per timestep
    logits_q = out["logits_q"]  # (1, T-1, S)
    ce = F.cross_entropy(
        logits_q.reshape(-1, model.num_nodes),
        targets.reshape(-1),
        reduction="none",
    ).reshape(1, T - 1)
    
    rec_loss = model._masked_mean(ce, mask_t)

    mu_q, logvar_q = out["mu_q"], out["logvar_q"]
    mu_p, logvar_p = out["mu_p"], out["logvar_p"]
    kl_bt = model.kl_diag_gauss(mu_q, logvar_q, mu_p, logvar_p)  # (1, T-1)
    # ADD L1 loss
    sparse_loss = out["mu_q"].abs().mean()

    kl_loss = model._masked_mean(kl_bt, mask_t)

    reward_logits = out["reward_logits_q"]  # (1, T-1)

    pos = rew_targets.sum()
    neg = rew_targets.numel() - pos

    model.rew_pos_count += pos.detach()
    model.rew_neg_count += neg.detach()

    pos_weight = (model.rew_neg_count / (model.rew_pos_count + 1e-8)).clamp(min=1.0)
    pos_weight = pos_weight.to(obs_b.device)

    reward_loss = F.binary_cross_entropy_with_logits(
    reward_logits, rew_targets, pos_weight=pos_weight, reduction="none"
    )
    reward_loss = model._masked_mean(reward_loss, mask_t)

    total = rec_loss + model.beta * kl_loss + model.reward_beta * reward_loss + model.sparse_beta * sparse_loss

    return {"total": total, "rec": rec_loss, "kl": kl_loss, "sparse":sparse_loss, "reward": reward_loss}





def main(args):

    if args.bouts_dir == "data/ALL-tf":
        mice = ["A1a", "A1b", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C1", "C3", "C6", "C7", "C8", "C9", "D3", "D4", "D5", "D7", "D7a", "D7b", "D8", "D9", "D9a", "D9b", "F2a", "F2b"]

        files_by_mouse = {}
        for mouse in mice:
            mouse_bouts_dir = Path(f"data/{mouse}-tf")
            mouse_files = sorted(
                mouse_bouts_dir.glob("*.npz"),
                key=lambda f: parse_bout_number(f.name)
            )
            if not mouse_files:
                raise RuntimeError(f"No .npz files found in {mouse_bouts_dir}")
            files_by_mouse[mouse] = mouse_files

        # interleave: all 1st bouts, then all 2nd bouts, etc.
        files = []
        max_n_bouts = max(len(v) for v in files_by_mouse.values())
        print("Max number of bouts: ", max_n_bouts)
        for bout_pos in range(max_n_bouts):
            for mouse in mice:
                mouse_files = files_by_mouse[mouse]
                if bout_pos < len(mouse_files):
                    files.append(mouse_files[bout_pos])

    else:
        bouts_dir = Path(args.bouts_dir)
        files = sorted(bouts_dir.glob("*.npz"), key=lambda f: parse_bout_number(f.name))
        if not files:
            raise RuntimeError(f"No .npz files found in {bouts_dir}")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = _load_model(args, device)
    params = model.configure_params()
    optimizer_prior = torch.optim.Adam(params[0], lr=args.online_lr_prior)
    optimizer_posterior = torch.optim.Adam(params[1], lr=args.online_lr_posterior)
    optimizer_other = torch.optim.Adam(params[2], lr=args.online_lr_other)

    metrics_out_dir = Path(args.metrics_out_dir)
    ckpt_dir = Path(args.online_ckpt_dir) if args.online_ckpt_dir else None
    summary_csv = Path(args.summary_csv)

    waterport_node = args.waterport_node
    # test random port 
    waterport_node = np.random.randint(0, args.num_nodes-1) if waterport_node < 0 else waterport_node
    print("WATERPORT NODE:", waterport_node)

    for global_bout_idx, f in enumerate(tqdm(files, desc="Online bouts"), start=1):
        d = np.load(f, allow_pickle=True)

        # Required keys
        obs_np = np.asarray(d["observations"]).astype(np.int64)  # (T,)
        act_np = np.asarray(d["actions"]).astype(np.int64)

        # Skip too-short bouts
        if obs_np.size < 2:
            continue

        if args.bouts_dir == "data/ALL-tf":
            bout_id = global_bout_idx
        else:
            bout_id = parse_bout_number(f.name)


        model.beta = beta_warmup(bout_id, start=50, end=100, beta_final=args.beta, kind="sigmoid")

        obs_t = torch.from_numpy(obs_np).long().to(device)  # (T,)
        act_t = torch.from_numpy(act_np).long().to(device) if act_np is not None else None

        obs_b = obs_t.unsqueeze(0)          # (1,T)
        
        rew_seq = (obs_b == waterport_node).long()
        rew_targets = rew_seq[:, 1:].float()

        act_b = act_t.unsqueeze(0) if act_t is not None else None  # (1,T)


        # -----------------------------------
        # evaluate before training
        # -----------------------------------
        model.eval()
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        with torch.no_grad():
            out_eval = model.forward(
                obs_b, act_b, rew_seq,
                deterministic_z=True,
                calculate_expected_info=True,
            )
            eval_arrays = _compute_eval_metrics(model, out_eval, obs_b)

        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)


        eval_arrays["obs_full"] = obs_np                      # (T,)
        eval_arrays["act_full"] = act_np if act_np is not None else np.zeros_like(obs_np)

        
        
        # -----------------------------------
        # train online with short windows
        # -----------------------------------
        model.train()

        train_window = args.train_window
        last_losses = None
        h_online = None

        for start in range(0, obs_b.size(1) - 1, train_window - 1):
            end = min(start + train_window, obs_b.size(1))

            obs_chunk = obs_b[:, start:end]
            act_chunk = act_b[:, start:end] if act_b is not None else None
            rew_chunk = rew_seq[:, start:end]

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
                calculate_expected_info=False,  # no need to calculate expected IG during training
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

        if last_losses is None:
            continue

        # -----------------------------------
        # save metrics per bout
        # -----------------------------------
        _save_bout_npz(metrics_out_dir, bout_id, eval_arrays)

        # summary stats (masked mean over timesteps)
        mask_eval = eval_arrays["mask"].astype(bool)  # (T-1,)
        def _mmean(x: np.ndarray) -> float:
            if mask_eval.any():
                return float(x[mask_eval].mean())
            return float(np.nan)

    # -----------------------------------
    # save final checkpoint
    # -----------------------------------
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / "final.ckpt"
        torch.save({
            "state_dict": model.state_dict(),
            "bout_id": bout_id,
        }, ckpt_path)

        print(f"Saved final checkpoint at bout {bout_id}")

    
    print("Per-bout metrics saved in:", metrics_out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Data / paths
    p.add_argument("--bouts-dir", required=True, help="Folder with per-bout .npz files")
    p.add_argument("--ckpt", default=None, help="Optional initial checkpoint to start online training from")
    p.add_argument("--metrics-out-dir", default="runs/online_metrics", help="Where to save per-bout metrics .npz")
    p.add_argument("--summary-csv", default="runs/online_metrics/summary.csv", help="Where to append bout summary rows")
    p.add_argument("--online-ckpt-dir", default=None, help="If set, save checkpoint each bout to this dir")

    # Model hyperparams
    p.add_argument("--num-nodes", type=int, default=127)
    p.add_argument("--num-actions", type=int, default=3)
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--waterport-node", type=int, default=116, help="Node ID of water port")
    p.add_argument("--beta", type=float, default=1e-2)
    p.add_argument("--sparse_beta", type=float, default=1e-2)
    p.add_argument("--reward_beta", type=float, default=1)

    # Optimization
    p.add_argument("--lr", type=float, default=1e-3, help="(Unused here) base lr passed into model")
    p.add_argument("--online-lr-prior", type=float, default=1e-4, help="Online optimizer LR (bout-by-bout)")
    p.add_argument("--online-lr-posterior", type=float, default=1e-4, help="Online optimizer LR (bout-by-bout)")
    p.add_argument("--online-lr-other", type=float, default=1e-4, help="Online optimizer LR (bout-by-bout)")
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 to disable gradient clipping")

    p.add_argument("--train-window", type=int, default=10,
               help="Number of consecutive timesteps per online update")

    args = p.parse_args()
    main(args)
