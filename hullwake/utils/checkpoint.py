from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hullwake.utils.seed import capture_rng_state, restore_rng_state


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric: float,
    config: dict[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "config": config,
        "rng_state": capture_rng_state(),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    map_location: str | torch.device = "cpu",
) -> tuple[int, float, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    restore_rng_state(payload["rng_state"])
    return int(payload["epoch"]) + 1, float(payload["best_metric"]), payload["config"]


def load_model_checkpoint(
    path: str | Path, model: torch.nn.Module, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return payload
