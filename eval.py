from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_model import create_model
from utils import accuracy_from_logits, get_device, load_checkpoint


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data", help="Where MNIST is stored/downloaded")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path, e.g. checkpoints/mnist_cnn.pth")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device} (cuda_available={torch.cuda.is_available()})")

    mean, std = (0.1307,), (0.3081,)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=tf)
    pin = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin
    )

    model = create_model().to(device)
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.model_state)
    model.eval()

    acc_sum = 0.0
    n = 0
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        acc_sum += accuracy_from_logits(logits, y) * x.size(0)
        n += x.size(0)

    acc = acc_sum / max(1, n)
    print(f"Test accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()