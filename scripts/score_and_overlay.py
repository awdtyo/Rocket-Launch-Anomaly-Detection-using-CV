#!/usr/bin/env python3
"""Render an anomaly-score overlay demo video for a raw launch video.

For each frame of the raw video this draws:
  - the tracked ROI bounding box from roi_boxes.json (a thin red full-frame
    outline on fallback frames),
  - a corner overlay with a live-updating rolling (default 60 s) anomaly-score
    line chart — the per-frame reconstruction error from score_video.py — with
    the current position marked and the threshold / peak lines drawn,
  - a flashing "ANOMALY RISING" banner when the score crosses
    threshold = multiplier * that video's own p95 error (default 3x), upgraded
    to a steady red "EVENT CONFIRMED" banner above peak_frac * observed max.

The chart figure is created once and re-used every frame (artists updated and
redrawn via buffer_rgba), so rendering stays fast enough for multi-minute video.

Saves demo/{video_name}_annotated.mp4.

Usage:
    python scripts/score_and_overlay.py --video data/raw/anomaly/spacex_crs-7.mp4
    python scripts/score_and_overlay.py --video data/raw/anomaly/spacex_crs-7.mp4 \\
        --threshold-multiplier 5.0 --window-sec 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from score_video import load_artifacts, per_frame_error, load_timestamps

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = ROOT / "data" / "features"
DEFAULT_MODEL = ROOT / "models" / "temporal_autoencoder.pt"
DEFAULT_SCALER = ROOT / "models" / "scaler.pkl"
DEFAULT_DEMO = ROOT / "demo"
DEFAULT_DECODED = ROOT / "data" / "frames" / "_decoded"

COLOR_BOX = (0, 255, 255)      # BGR cyan for the tracked ROI box
COLOR_FALLBACK = (0, 0, 255)   # BGR red for full-frame fallback outline
COLOR_RISING = (0, 180, 255)   # BGR amber banner
COLOR_PEAK = (0, 0, 255)       # BGR red banner


class RollingChart:
    """Cached matplotlib figure for the live anomaly-score chart."""

    def __init__(self, window_sec: float, y_max: float, threshold: float,
                 peak_line: float, ts: np.ndarray, err: np.ndarray):
        self.window = window_sec
        self.ts = ts
        self.err = err
        self.fig = Figure(figsize=(3.4, 2.1), dpi=100)
        self.fig.patch.set_facecolor((0.04, 0.04, 0.09, 0.85))
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_axes([0.14, 0.20, 0.82, 0.72])
        ax.set_facecolor("none")
        ax.axhline(threshold, color=(1.0, 0.85, 0.35), ls="--", lw=1.1,
                   alpha=0.9, label="threshold")
        ax.axhline(peak_line, color=(1.0, 0.35, 0.35), ls=":", lw=1.1,
                   alpha=0.9, label="peak")
        (self.line,) = ax.plot([], [], color=(0.95, 0.55, 0.15), lw=1.6)
        (self.marker,) = ax.plot([], [], marker="o", ms=5.5, color=(1.0, 0.25, 0.2))
        ax.set_ylim(0, max(1e-6, y_max))
        ax.set_xlim(0, window_sec)
        ax.tick_params(labelsize=6, colors="white")
        ax.set_ylabel("anomaly", fontsize=7, color="white")
        ax.set_xlabel("sec", fontsize=7, color="white")
        ax.grid(True, alpha=0.3, color="white")
        for s in ax.spines.values():
            s.set_color("white")
        self.ax = ax

    def render(self, t: float) -> np.ndarray:
        """Update the chart to time t and return the RGBA buffer."""
        t0 = max(0.0, t - self.window)
        mask = (self.ts >= t0) & (self.ts <= t)
        self.line.set_data(self.ts[mask], self.err[mask])
        self.marker.set_data([t], [np.interp(t, self.ts, self.err)])
        self.ax.set_xlim(t0, t0 + self.window)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())


def load_roi(roi_path: Path):
    """Return (ts, boxes, fallback) aligned with the scored rows.

    Row i corresponds to the i-th sorted ROI crop (same order as features.py),
    so boxes[i] / fallback[i] line up with timestamps[i] / error[i].
    """
    with open(roi_path) as f:
        boxes_json = json.load(f)
    files = sorted(boxes_json["boxes"])
    fallback_set = set(boxes_json.get("fallback_frames", []))
    boxes = [boxes_json["boxes"][fn] for fn in files]
    fallback = np.array([fn in fallback_set for fn in files], bool)
    return np.array(boxes, np.int32), fallback


def draw_banner(frame: np.ndarray, text: str, color, n: int, *, steady: bool = False) -> None:
    """Draw a flashing (or steady) banner bar across the top of the frame."""
    h, w = frame.shape[:2]
    flash_on = steady or (n // 4) % 2 == 0
    if not flash_on:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 44), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    scale = max(0.8, w / 1100.0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(frame, text, ((w - tw) // 2, 31), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2, cv2.LINE_AA)


def render_overlay(video_path: Path, roi_path: Path, ts: np.ndarray, err: np.ndarray,
                   out_path: Path, *, multiplier: float = 3.0, window_sec: float = 60.0,
                   peak_frac: float = 0.9, out_fps: float | None = None,
                   max_frames: int | None = None, progress=None) -> dict:
    """Render the annotated demo video; returns a stats dict."""
    boxes, fallback = load_roi(roi_path)
    if len(ts) != len(err) or len(boxes) != len(err):
        raise ValueError(
            f"length mismatch: ts {len(ts)}, err {len(err)}, boxes {len(boxes)}")

    p95 = float(np.percentile(err, 95))
    threshold = multiplier * p95
    peak_line = peak_frac * float(err.max())
    y_max = max(float(err.max()), threshold, peak_line) * 1.05

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if out_fps is None:
        out_fps = src_fps
    step = max(1, round(src_fps / out_fps))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             out_fps, (width, height))
    chart = RollingChart(window_sec, y_max, threshold, peak_line, ts, err)
    seen = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if seen % step != 0:
            seen += 1
            continue
        t = seen / src_fps

        idx = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1))
        if fallback[idx]:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1),
                          COLOR_FALLBACK, 1)
        else:
            x, y, bw, bh = boxes[idx]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_BOX, 2)

        e = float(np.interp(t, ts, err))
        chart_rgba = chart.render(t)
        ch, cw = chart_rgba.shape[:2]
        scale = min(1.0, (width - 10) / cw, (height - 10) / ch)
        if scale < 1.0:
            cw, ch = max(1, int(cw * scale)), max(1, int(ch * scale))
            chart_rgba = cv2.resize(chart_rgba, (cw, ch),
                                    interpolation=cv2.INTER_AREA)
        x0 = max(0, width - cw - 10)
        y0 = max(0, height - ch - 10)
        alpha = chart_rgba[..., 3:] / 255.0
        region = frame[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
        frame[y0:y0 + ch, x0:x0 + cw] = (
            region * (1 - alpha) + chart_rgba[..., :3] * alpha).astype(np.uint8)

        if e > peak_line:
            draw_banner(frame, "EVENT CONFIRMED - ANOMALY PEAK", COLOR_PEAK,
                        seen, steady=True)
        elif e > threshold:
            draw_banner(frame, "ANOMALY RISING", COLOR_RISING, seen)

        cv2.putText(frame, f"t={t:.1f}s  score={e:.2f}  thr={threshold:.2f}",
                    (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        seen += 1
        if progress and seen % 500 == 0:
            progress(seen, total // step)
        if max_frames and seen >= max_frames:
            break

    cap.release()
    writer.release()

    return {"video": video_path.stem, "frames": seen, "src_fps": src_fps,
            "out_fps": out_fps, "p95": p95, "max": float(err.max()),
            "multiplier": multiplier, "threshold": threshold,
            "peak_line": peak_line, "out_path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="raw video path (mp4)")
    ap.add_argument("--roi-boxes", default=None,
                    help="roi_boxes.json path (default: data/frames/{stem}/roi_boxes.json)")
    ap.add_argument("--features", default=None,
                    help="feature .npy path (default: data/features/{stem}.npy)")
    ap.add_argument("--scores", default=None,
                    help="precomputed per-frame error .npy (else compute via score_video)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="trained autoencoder checkpoint (default: %(default)s)")
    ap.add_argument("--scaler", default=str(DEFAULT_SCALER),
                    help="scaler pickle (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output mp4 path (default: demo/{stem}_annotated.mp4)")
    ap.add_argument("--threshold-multiplier", type=float, default=3.0,
                    help="threshold = multiplier * video p95 error (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="absolute threshold (overrides --threshold-multiplier)")
    ap.add_argument("--window-sec", type=float, default=60.0,
                    help="rolling chart window in seconds (default: %(default)s)")
    ap.add_argument("--peak-frac", type=float, default=0.9,
                    help="fraction of observed max error for the peak banner "
                         "(default: %(default)s)")
    ap.add_argument("--fps", type=float, default=None,
                    help="output fps (default: source fps)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after this many written frames (testing)")
    args = ap.parse_args(argv)

    video = Path(args.video)
    if not video.exists():
        print(f"error: no such video: {video}", file=sys.stderr)
        return 1
    stem = video.stem
    roi_path = Path(args.roi_boxes) if args.roi_boxes else ROOT / "data" / "frames" / stem / "roi_boxes.json"
    feat_path = Path(args.features) if args.features else DEFAULT_FEATURES / f"{stem}.npy"

    if args.scores:
        err = np.load(args.scores).astype(np.float32)
        ts = load_timestamps(Path(feat_path))
    else:
        X = np.load(feat_path).astype(np.float32)
        model, scaler, ckpt = load_artifacts(Path(args.model), Path(args.scaler))
        scaled = (X - scaler["mean"]) / scaler["std"]
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        err = per_frame_error(model, scaled, ckpt["window"], ckpt["stride"], device)
        ts = load_timestamps(feat_path)
    if ts is None or len(ts) != len(err):
        ts = np.arange(len(err), dtype=np.float64) / 2.0

    multiplier = args.threshold_multiplier
    if args.threshold is not None:
        threshold = args.threshold
        multiplier = None
    else:
        threshold = None
    p95 = float(np.percentile(err, 95))
    threshold = threshold if threshold is not None else multiplier * p95

    out_path = Path(args.out) if args.out else DEFAULT_DEMO / f"{stem}_annotated.mp4"

    print(f"video : {stem}")
    print(f"  frames scored : {len(err)}   raw : {video.name}")
    print(f"  error stats   : mean {err.mean():.4f}  p95 {p95:.4f}  "
          f"max {err.max():.4f}")
    print(f"  threshold     : {'abs ' + format(threshold, '.4f') if multiplier is None else format(threshold, '.4f') + '  (= %.2fx p95)' % multiplier}")
    print(f"  peak line     : {args.peak_frac:.2f} x max = {args.peak_frac * err.max():.4f}")
    if threshold >= err.max():
        print(f"  note: threshold {threshold:.3f} >= observed max {err.max():.3f} — "
              f"no banner will trigger", file=sys.stderr)

    start = time.time()

    def progress(seen, total):
        print(f"  {stem}: {seen}/{total} frames ...", file=sys.stderr, end="\r")

    stats = render_overlay(video, roi_path, ts, err, out_path,
                           multiplier=threshold / p95 if threshold else 3.0,
                           window_sec=args.window_sec, peak_frac=args.peak_frac,
                           out_fps=args.fps, max_frames=args.max_frames,
                           progress=progress)
    elapsed = time.time() - start
    print(f"  rendered {stats['frames']} frames ({stats['out_fps']:.1f} fps out) "
          f"in {elapsed:.1f}s -> {stats['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
