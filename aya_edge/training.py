"""Small, explicit helpers for reproducible prototype training."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .config import AyaEdgeConfig


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs; enable deterministic CUDA algorithms."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    *,
    step: int = 0,
    tokens_processed: int = 0,
    config: AyaEdgeConfig | dict[str, Any] | None = None,
    dataset_version: str | None = None,
) -> None:
    """Save model, optimizer, RNG, and run metadata in one portable payload."""
    if step < 0 or tokens_processed < 0:
        raise ValueError("step and tokens_processed must be non-negative")
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "tokens_processed": tokens_processed,
        "rng_state": _rng_state(),
        "config": config.to_dict() if isinstance(config, AyaEdgeConfig) else config,
        "dataset_version": dataset_version,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return its metadata, including the step."""
    payload = torch.load(Path(path), map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and payload.get("rng_state") is not None:
        _restore_rng_state(payload["rng_state"])
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "scheduler", "scaler", "rng_state"}
    }
