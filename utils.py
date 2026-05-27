"""
utils.py — Configuration loading, seeding, logging helpers.
"""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """Make experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Faster, slightly less reproducible
    torch.backends.cudnn.benchmark = True       # Speed up convs


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def report_gpu_memory() -> str:
    """Report current GPU memory usage."""
    if not torch.cuda.is_available():
        return "No GPU"
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    return f"GPU memory: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved"
