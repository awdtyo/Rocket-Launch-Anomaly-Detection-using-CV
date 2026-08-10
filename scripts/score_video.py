#!/usr/bin/env python3
"""Score one feature-vector file with the trained temporal anomaly autoencoder.

Loads models/temporal_autoencoder.pt (GRU autoencoder) and models/scaler.pkl
(train-only standardization stats), applies the scaler, windows the sequence
exactly like training (WINDOW frames, STRIDE 4), reconstructs each window, and
computes a per-frame reconstruction error identical to the Phase-5 evaluation:
a frame's error is the mean MSE over every window that covers it.

Prints the peak-error frame + its elapsed timestamp, saves a PNG plot of error
over elapsed seconds (x-axis from data/features/{name}.frame_index.json
timestamps), and returns 0.

Usage:
    python scripts/score_video.py data/features/spacex_crs-7.npy
    python scripts/score_video.py --name spacex_crs-7 --out data/plots/crs-7.png
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = ROOT / "data" / "features"
DEFAULT_MODEL = ROOT / "models" / "temporal_autoencoder.pt"
DEFAULT_SCALER = ROOT / "models" / "scaler.pkl"
DEFAULT_PLOTS = ROOT / "data" / "plots"


class GRUAutoencoder(nn.Module):
    """Same architecture as the training notebook."""

    def __init__(self, input_dim: int, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc_enc = nn.Linear(hidden, latent)
        self.fc_dec = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc_out = nn.Linear(hidden, input_dim)

    def forward(self, x):
        _, h = self.encoder(x)
        z = torch.tanh(self.fc_enc(h[-1]))
        h0 = torch.tanh(self.fc_dec(z)).unsqueeze(0)
        src = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        out, _ = self.decoder(src, h0)
        return self.fc_out(out)


def load_artifacts(model_path: Path, scaler_path: Path):
    """Return (model, scaler_dict, ckpt_dict)."""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = GRUAutoencoder(ckpt["input_dim"], ckpt["hidden"], ckpt["latent"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, scaler, ckpt


def make_windows(X: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Split a scaled sequence into overlapping windows (same as training)."""
    starts = range(0, len(X) - window + 1, stride)
    return np.stack([X[s:s + window] for s in starts])


def per_frame_error(model, X: np.ndarray, window: int, stride: int, device: str) -> np.ndarray:
    """Mean reconstruction MSE per frame, averaged over all covering windows."""
    W = make_windows(X, window, stride)
    model.eval()
    with torch.no_grad():
        recon = model(torch.from_numpy(W).to(device)).cpu().numpy()
    window_err = ((W - recon) ** 2).mean(axis=(1, 2))          # [n_windows]
    per_frame = np.zeros(len(X), np.float32)
    cover = np.zeros(len(X), np.int32)
    for i, s in enumerate(range(0, len(X) - window + 1, stride)):
        per_frame[s:s + window] += window_err[i]
        cover[s:s + window] += 1
    return per_frame / np.maximum(1, cover)


def load_timestamps(npy_path: Path) -> np.ndarray | None:
    """Elapsed-seconds per frame from the matching .frame_index.json (or None)."""
    jp = npy_path.parent / f"{npy_path.stem}.frame_index.json"
    if not jp.exists():
        return None
    with open(jp) as f:
        idx = json.load(f)
    frames = idx.get("frames", [])
    return np.array([fr.get("timestamp", 0.0) for fr in frames], np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy", nargs="?", help="path to the feature .npy to score")
    ap.add_argument("--name", help="instead of a path, look up "
                                   "data/features/{name}.npy")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="trained autoencoder checkpoint (default: %(default)s)")
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER),
                    help="scaler pickle (default: %(default)s)")
    ap.add_argument("--window", type=int, default=None,
                    help="override window size (default: from checkpoint)")
    ap.add_argument("--stride", type=int, default=None,
                    help="override stride (default: from checkpoint)")
    ap.add_argument("--out", default=None,
                    help="PNG output path (default: data/plots/{stem}.png)")
    ap.add_argument("--no-save", action="store_true",
                    help="plot to screen instead of saving a PNG")
    args = ap.parse_args(argv)

    if args.name and not args.npy:
        npy_path = Path(DEFAULT_FEATURES) / f"{args.name}.npy"
    elif args.npy:
        npy_path = Path(args.npy)
    else:
        print("error: give a feature .npy path or --name", file=sys.stderr)
        return 1
    if not npy_path.exists():
        print(f"error: no such file: {npy_path}", file=sys.stderr)
        return 1

    X = np.load(npy_path).astype(np.float32)
    if X.ndim != 2:
        print(f"error: expected a 2D feature matrix, got {X.shape}", file=sys.stderr)
        return 1

    model, scaler, ckpt = load_artifacts(Path(args.model), Path(args.scaler))
    window = args.window or ckpt["window"]
    stride = args.stride or ckpt["stride"]
    if X.shape[1] != ckpt["input_dim"]:
        print(f"error: feature dim {X.shape[1]} != model input {ckpt['input_dim']}",
              file=sys.stderr)
        return 1
    if len(X) < window:
        print(f"error: only {len(X)} frames, need at least {window} for one window",
              file=sys.stderr)
        return 1

    scaled = (X - scaler["mean"]) / scaler["std"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    err = per_frame_error(model, scaled, window, stride, device)

    ts = load_timestamps(npy_path)
    if ts is None or len(ts) != len(X):
        ts = np.arange(len(X), dtype=np.float64) / 2.0   # 2fps default sampling
    peak = int(np.argmax(err))

    name = npy_path.stem
    print(f"{name}: {len(X)} frames, window {window}, stride {stride}, {device}")
    print(f"  error: mean {err.mean():.5f}  p95 {np.percentile(err, 95):.5f}  "
          f"max {err.max():.5f}")
    print(f"  peak error {err[peak]:.5f} at frame {peak} "
          f"(elapsed {ts[peak]:.2f}s)")

    if args.no_save:
        plt.ion()
        show = True
    else:
        out = Path(args.out) if args.out else DEFAULT_PLOTS / f"{name}_anomaly.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        show = False

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(ts, err, lw=0.8, color="tab:blue")
    ax.axvline(ts[peak], color="tab:red", ls="--", alpha=0.8,
               label=f"peak {err[peak]:.3f} @ {ts[peak]:.1f}s")
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("reconstruction error")
    ax.set_title(f"{name} — anomaly score (mean {err.mean():.4f}, "
                 f"p95 {np.percentile(err, 95):.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  plot saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
