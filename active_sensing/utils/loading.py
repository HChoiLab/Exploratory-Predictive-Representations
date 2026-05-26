from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from active_sensing.core.data import load_npz


def load_sequences(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load observations and optional locations from .npz or directory of .npz files."""
    return load_npz(path)


