import torch
import numpy as np


def _entropy_from_dist(dist) -> torch.Tensor:
    """Return entropy for common distribution-like objects."""
    if hasattr(dist, "torch_dist") and hasattr(dist.torch_dist, "entropy"):
        return dist.torch_dist.entropy()

    if hasattr(dist, "entropy"):
        try:
            return dist.entropy()
        except Exception:
            pass

    if hasattr(dist, "scale"):
        var = dist.scale ** 2
        return 0.5 * torch.log(2 * torch.pi * torch.e * var).sum(dim=-1)

    raise RuntimeError("Cannot compute entropy for the given distribution object.")


def _find_posterior_obj(outputs):
    """Find a distribution-like object in model output."""
    items = outputs if isinstance(outputs, (list, tuple)) else [outputs]

    for item in reversed(items):
        if (
            hasattr(item, "torch_dist")
            or hasattr(item, "entropy")
            or hasattr(item, "sigma")
            or hasattr(item, "scale")
        ):
            return item

    raise RuntimeError("Could not find a posterior-like object in model.forward output.")


def compute_entropy_reductions(
    model,
    obs_seq: torch.Tensor,
    loc_seq: torch.Tensor | None,
) -> np.ndarray:
    """Compute entropy reduction from step t to t+1."""
    if obs_seq.ndim == 2:
        obs_seq = obs_seq.unsqueeze(0)
        loc_seq = loc_seq.unsqueeze(0) if loc_seq is not None else None

    n_steps = obs_seq.shape[1]
    reductions = []

    for t in range(1, n_steps):
        cur_obs = obs_seq[:, :t, :]
        cur_loc = loc_seq[:, :t, :] if loc_seq is not None else None
        next_obs = obs_seq[:, : t + 1, :]
        next_loc = loc_seq[:, : t + 1, :] if loc_seq is not None else None

        post_before = _find_posterior_obj(model.forward(cur_obs, cur_loc))
        post_after = _find_posterior_obj(model.forward(next_obs, next_loc))

        ent_before = _entropy_from_dist(post_before).detach().cpu().numpy()
        ent_after = _entropy_from_dist(post_after).detach().cpu().numpy()

        reductions.append((ent_before - ent_after).reshape(-1))

    reductions = np.stack(reductions, axis=0)
    return reductions.mean(axis=1)


def score_path(
    model,
    obs_seq: torch.Tensor,
    loc_seq: torch.Tensor | None,
) -> dict:
    reductions = compute_entropy_reductions(model, obs_seq, loc_seq)

    return {
        "entropy_reduction": reductions,
        "total_reduction": float(reductions.sum()),
    }