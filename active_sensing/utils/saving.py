"""Helper functions to save checkpoints, predictions, trajectories and scoring outputs."""
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict
import csv


def save_checkpoint(model, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path: str, device: str = "cpu"):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def save_npz(path: str, **arrays):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)


def _to_serializable(o):
    if isinstance(o, dict):
        return {k: _to_serializable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_serializable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, torch.Tensor):
        return _to_serializable(o.detach().cpu().numpy())
    return o


def save_json(path: str, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    serial = _to_serializable(obj)
    with open(p, "w") as f:
        json.dump(serial, f, indent=2)



# --------------------------
# Saving utilities
# --------------------------
def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _append_summary_csv(csv_path: Path, row: Dict[str, object]):
    """
    Appends a row to CSV. Creates header if file does not exist.
    """
    _ensure_parent(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _save_bout_npz(out_dir: Path, bout_id: int, arrays: Dict[str, np.ndarray]):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bout_{bout_id:05d}_metrics.npz"
    np.savez_compressed(out_path, **arrays)
