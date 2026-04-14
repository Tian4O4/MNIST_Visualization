from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

## choose GPU or CPU
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

## set seed for reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

## checkpoint payload
@dataclass(frozen=True)
class CheckpointPayload:
    model_state: Dict[str, Any]
    optimizer_state: Optional[Dict[str, Any]] = None
    epoch: Optional[int] = None
    best_acc: Optional[float] = None

## save checkpoint
def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_acc: Optional[float] = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)

## load checkpoint
def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> CheckpointPayload:
    raw = torch.load(path, map_location=map_location)
    return CheckpointPayload(
        model_state=raw["model_state"],
        optimizer_state=raw.get("optimizer_state"),
        epoch=raw.get("epoch"),
        best_acc=raw.get("best_acc"),
    )

## calculate accuracy from logits
@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()