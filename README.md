# MNIST CNN + Drawing App (PyTorch)

This project trains a custom CNN (with residual blocks) on MNIST, then lets you draw digits in a small window and see live probabilities for 0–9.

## 0) Environment

- Python 3.9+ recommended (works on Windows)
- GPU is optional (CUDA will be used automatically if available)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 1) Train

```bash
python train.py
```

This will download MNIST automatically and save a checkpoint to:

- `checkpoints/mnist_cnn.pth`

## 2) Evaluate (test set)

```bash
python eval.py --ckpt checkpoints/mnist_cnn.pth
```

## 3) Launch drawing app

```bash
python app_draw.py --ckpt checkpoints/mnist_cnn.pth
```

Tips:
- Draw a clear digit (0–9) with your mouse.
- Use **Clear** to reset the canvas.

## Files

- `mnist_model.py`: model architecture (custom CNN with residual blocks)
- `train.py`: training loop + checkpoint saving
- `eval.py`: load checkpoint + evaluate on MNIST test set
- `app_draw.py`: Tkinter drawing window + live inference
- `utils.py`: device/seed/checkpoint helper functions

