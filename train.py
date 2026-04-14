from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from mnist_model import create_model
from utils import accuracy_from_logits, get_device, load_checkpoint, save_checkpoint, set_seed

# set all the parameters for the training
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    epochs: int = 12
    lr: float = 3e-3
    weight_decay: float = 1e-2
    num_workers: int = 0
    val_split: int = 5_000
    seed: int = 42
    scheduler: str = "cosine"  # cosine | step | none
    cpu_threads: int | None = None
    interop_threads: int | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    max_gpu_mem_fraction: float | None = None

# build the transforms for the training and testing, also known as data augmentation
def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean, std = (0.1307,), (0.3081,)
    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, test_tf

## evaluate the model on the validation set
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, *, max_batches: int | None = None) -> float:
    model.eval()
    acc_sum = 0.0
    n = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        acc_sum += accuracy_from_logits(logits, y) * x.size(0)
        n += x.size(0)
        if max_batches is not None and (step + 1) >= max_batches:
            break
    return acc_sum / max(1, n)

## train the model for one epoch
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    max_batches: int | None = None,
) -> float:
    model.train()
    loss_sum = 0.0
    n = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        if max_batches is not None and (step + 1) >= max_batches:
            break
    return loss_sum / max(1, n)


def main() -> None:
    ## set the arguments for the training
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="Where to download/store MNIST")
    parser.add_argument("--ckpt", default="checkpoints/mnist_cnn.pth", help="Checkpoint output path")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from (optional)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--scheduler", default=None, choices=["cosine", "step", "none"], help="LR scheduler")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (Windows: 0 is safest)")
    parser.add_argument("--cpu-threads", type=int, default=None, help="Limit CPU threads used by PyTorch (e.g. 4)")
    parser.add_argument("--interop-threads", type=int, default=None, help="Limit inter-op threads (advanced)")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Cap batches per training epoch")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Cap batches for val/test evaluation")
    parser.add_argument(
        "--max-gpu-mem-fraction",
        type=float,
        default=None,
        help="If using CUDA, cap this process' GPU memory fraction (0-1)",
    )
    args = parser.parse_args()

    ## set the configuration for the training
    cfg = TrainConfig()
    if args.epochs is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "epochs": args.epochs})
    if args.batch_size is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "batch_size": args.batch_size})
    if args.lr is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "lr": args.lr})
    if args.scheduler is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "scheduler": args.scheduler})
    if args.num_workers is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "num_workers": args.num_workers})
    if args.cpu_threads is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "cpu_threads": args.cpu_threads})
    if args.interop_threads is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "interop_threads": args.interop_threads})
    if args.max_train_batches is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "max_train_batches": args.max_train_batches})
    if args.max_eval_batches is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "max_eval_batches": args.max_eval_batches})
    if args.max_gpu_mem_fraction is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "max_gpu_mem_fraction": args.max_gpu_mem_fraction})

    set_seed(cfg.seed)

    ## set the number of threads for the training
    if cfg.cpu_threads is not None:
        torch.set_num_threads(int(cfg.cpu_threads))
    if cfg.interop_threads is not None:
        torch.set_num_interop_threads(int(cfg.interop_threads))

    ## set the device for the training
    device = get_device()
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")
    if cfg.cpu_threads is not None:
        print(f"torch.num_threads={torch.get_num_threads()}")
    if cfg.interop_threads is not None:
        print(f"torch.num_interop_threads={torch.get_num_interop_threads()}")
    if device.type == "cuda" and cfg.max_gpu_mem_fraction is not None:
        frac = float(cfg.max_gpu_mem_fraction)
        frac = max(0.05, min(1.0, frac))
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"cuda.per_process_memory_fraction={frac}")

    ## prepare for the training and test set
    train_tf, test_tf = build_transforms()
    full_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=test_tf)

    val_size = min(cfg.val_split, len(full_train) // 10)
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed)
    )
    # Validation should not use random affine.
    val_ds.dataset.transform = test_tf  # type: ignore[attr-defined]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin
    )

    ## create model
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    ## start training(settings)
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(ckpt.model_state)
        if ckpt.optimizer_state is not None:
            optimizer.load_state_dict(ckpt.optimizer_state)
        if ckpt.epoch is not None:
            start_epoch = int(ckpt.epoch) + 1
        if ckpt.best_acc is not None:
            best_acc = float(ckpt.best_acc)
        print(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_acc={best_acc:.4f})")

    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "step":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[cfg.epochs // 2, (cfg.epochs * 3) // 4], gamma=0.3)
    else:
        scheduler = None

    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)

    ## train for n epochs
    for epoch in range(start_epoch, cfg.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            max_batches=cfg.max_train_batches,
        )
        val_acc = evaluate(model, val_loader, device, max_batches=cfg.max_eval_batches)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | lr={lr_now:.4g} | train_loss={train_loss:.4f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(args.ckpt, model=model, optimizer=optimizer, epoch=epoch, best_acc=best_acc)
            print(f"  Saved best checkpoint to {args.ckpt} (best_acc={best_acc*100:.2f}%)")

    # Final test report with best checkpoint (if saved).
    if os.path.exists(args.ckpt):
        ckpt = load_checkpoint(args.ckpt, map_location=device)
        model.load_state_dict(ckpt.model_state)
    test_acc = evaluate(model, test_loader, device, max_batches=cfg.max_eval_batches)
    print(f"Test accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    main()