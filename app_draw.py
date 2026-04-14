from __future__ import annotations

import argparse
import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import torch
import tkinter as tk
from PIL import Image, ImageChops, ImageDraw, ImageOps

from mnist_model import create_model
from utils import get_device, load_checkpoint


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def center_of_mass(img: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(img > 0)
    if len(xs) == 0:
        return (img.shape[1] / 2.0, img.shape[0] / 2.0)
    weights = img[ys, xs].astype(np.float64)
    wsum = float(weights.sum())
    cx = float((xs * weights).sum() / wsum)
    cy = float((ys * weights).sum() / wsum)
    return cx, cy


def preprocess_to_mnist(pil_img: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    """
    Input: grayscale PIL image (0..255), black background, white strokes.
    Output: normalized torch tensor (1,1,28,28) and the 28x28 PIL image (for debug).
    """
    img = pil_img.convert("L")

    # Trim empty borders (based on non-black content)
    bbox = ImageOps.invert(img).getbbox()
    if bbox is None:
        small = Image.new("L", (28, 28), 0)
        arr = np.array(small, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)[None, None, :, :]
        t = (t - MNIST_MEAN) / MNIST_STD
        return t, small

    img = img.crop(bbox)

    # Resize so that the digit fits into 20x20, keep aspect ratio
    w, h = img.size
    if w == 0 or h == 0:
        small = Image.new("L", (28, 28), 0)
        arr = np.array(small, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr)[None, None, :, :]
        t = (t - MNIST_MEAN) / MNIST_STD
        return t, small

    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    # Paste into 28x28 with rough centering first
    canvas = Image.new("L", (28, 28), 0)
    off_x = (28 - new_w) // 2
    off_y = (28 - new_h) // 2
    canvas.paste(img, (off_x, off_y))

    # Center by center-of-mass (MNIST-like)
    arr_u8 = np.array(canvas, dtype=np.uint8)
    cx, cy = center_of_mass(arr_u8)
    shift_x = int(round(14 - cx))
    shift_y = int(round(14 - cy))
    canvas = ImageChops.offset(canvas, shift_x, shift_y)

    # Normalize to MNIST mean/std
    arr = np.array(canvas, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)[None, None, :, :]
    t = (t - MNIST_MEAN) / MNIST_STD
    return t, canvas


class DrawApp:
    def __init__(self, root: tk.Tk, *, model: torch.nn.Module, device: torch.device, canvas_px: int = 280) -> None:
        self.root = root
        self.model = model
        self.device = device

        self.canvas_px = canvas_px
        self.brush = 14
        self.debounce_ms = 10
        self._scheduled: Optional[str] = None

        self.last_x = None
        self.last_y = None

        root.title("Hello MNIST")

        self.canvas = tk.Canvas(root, width=canvas_px, height=canvas_px, bg="black", highlightthickness=1)
        self.canvas.grid(row=0, column=0, rowspan=12, padx=10, pady=10)

        # Offscreen buffer (PIL) for preprocessing.
        self.buffer = Image.new("L", (canvas_px, canvas_px), 0)
        self.draw = ImageDraw.Draw(self.buffer)

        self.canvas.bind("<B1-Motion>", self.on_paint)
        self.canvas.bind("<Button-1>", self.on_paint)

        self.pred_label = tk.Label(root, text="Draw a digit...", font=("Consolas", 14))
        self.pred_label.grid(row=0, column=1, sticky="w", padx=10)

        self.prob_labels = []
        for i in range(10):
            lbl = tk.Label(root, text=f"{i}: 0.0000", font=("Consolas", 12))
            lbl.grid(row=i + 1, column=1, sticky="w", padx=10)
            self.prob_labels.append(lbl)

        btn_frame = tk.Frame(root)
        btn_frame.grid(row=11, column=1, sticky="w", padx=30, pady=(10, 10))

        tk.Button(btn_frame, text="Clear", command=self.clear).grid(row=0, column=0, padx=(0, 8))
        ## tk.Button(btn_frame, text="Predict now", command=self.predict_now).grid(row=0, column=1)

        self._update_probs(np.zeros(10, dtype=np.float32))
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def clear(self) -> None:
        self.canvas.delete("all")
        self.buffer = Image.new("L", (self.canvas_px, self.canvas_px), 0)
        self.draw = ImageDraw.Draw(self.buffer)
        self.pred_label.config(text="Draw a digit...")
        self._update_probs(np.zeros(10, dtype=np.float32))

        self.last_x = None
        self.last_y = None

    def on_release(self, event: tk.Event) -> None:
        self.last_x = None
        self.last_y = None

    #def on_paint(self, event: tk.Event) -> None:
     #   x, y = int(event.x), int(event.y)
      #  r = self.brush // 2
       # x0, y0, x1, y1 = x - r, y - r, x + r, y + r

        #self.canvas.create_oval(x0, y0, x1, y1, fill="white", outline="white")
        #self.draw.ellipse([x0, y0, x1, y1], fill=255)

        #if self._scheduled is not None:
         #   try:
          #      self.root.after_cancel(self._scheduled)
           # except Exception:
            #    pass
#        self._scheduled = self.root.after(self.debounce_ms, self.predict_now)

    def on_paint(self, event: tk.Event) -> None:
        x, y = int(event.x), int(event.y)

        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=self.brush, fill="white", capstyle="round", smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill=255, width=self.brush
            )

        r = self.brush // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        self.last_x = x
        self.last_y = y

        if self._scheduled is not None:
            try:
                self.root.after_cancel(self._scheduled)
            except Exception:
                pass
        self._scheduled = self.root.after(self.debounce_ms, self.predict_now)

    @torch.no_grad()
    def predict_now(self) -> None:
        t, _debug28 = preprocess_to_mnist(self.buffer)
        t = t.to(self.device)
        logits = self.model(t)[0].detach().float().cpu().numpy()
        probs = softmax_np(logits).astype(np.float32)
        self._update_probs(probs)

    def _update_probs(self, probs: np.ndarray) -> None:
        top = int(np.argmax(probs)) if probs.size == 10 else -1
        if top >= 0 and probs.sum() > 0:
            self.pred_label.config(text=f"Prediction: {top}    (confidence={probs[top]:.4f})")
        else:
            self.pred_label.config(text="Draw a digit...")

        for i in range(10):
            txt = f"{i}: {probs[i]:.4f}"
            self.prob_labels[i].config(text=txt, fg=("lime" if i == top else "grey"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path, e.g. checkpoints/mnist_cnn.pth")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else get_device()
    model = create_model()
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model.load_state_dict(ckpt.model_state)
    model.to(device)
    model.eval()

    root = tk.Tk()
    _app = DrawApp(root, model=model, device=device)
    root.mainloop()


if __name__ == "__main__":
    main()