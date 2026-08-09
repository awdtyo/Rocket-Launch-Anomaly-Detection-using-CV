#!/usr/bin/env python3
"""Compute per-frame feature vectors from ROI-cropped launch frames.

For every ROI crop in data/frames/{video}/roi/ this computes a fixed-length,
normalized feature vector (plume shape, HSV color histogram, edge/debris
density outside the plume, frame-to-frame optical flow) and saves it as
data/features/{video}.npy (float32, rows = frames) plus
data/features/{video}.frame_index.json mapping each row to its frame file,
timestamp, fallback flag, and the video's lighting tag.

Features are made relative/normalized where possible so day and night footage
stay comparable (e.g. plume brightness is scored as contrast against the local
background, not an absolute gray level). Frames that used the full-frame
fallback in roi_track are flagged in the frame index rather than dropped.

Processed videos (existing .npy) are skipped on rerun; --force redoes, --only
runs a single video. Works as a CLI or importable (process_video / batch) for
use from a Colab notebook. All outputs are cached to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRAMES = ROOT / "data" / "frames"
DEFAULT_FEATURES = ROOT / "data" / "features"

WORK_SIZE = 480  # max dimension used for color/edge/flow; shape stats use full res

H_BINS, S_BINS, V_BINS = 12, 6, 6

FEATURE_NAMES = (
    ["plume_area_frac", "plume_aspect_ratio", "plume_symmetry",
     "plume_centroid_x", "plume_centroid_y", "plume_bg_contrast"]
    + [f"hue_bin{i}" for i in range(H_BINS)]
    + [f"sat_bin{i}" for i in range(S_BINS)]
    + [f"val_bin{i}" for i in range(V_BINS)]
    + ["edge_density_outside", "flow_mean_mag", "flow_p95_mag"]
)
N_FEATURES = len(FEATURE_NAMES)


def iter_video_dirs(frames_base: Path):
    """Yield video dirs under frames_base that have ROI crops to process."""
    for video_dir in sorted(frames_base.iterdir()):
        roi_dir = video_dir / "roi"
        if roi_dir.is_dir() and any(roi_dir.glob("frame_*.jpg")):
            yield video_dir


def plume_mask(gray: np.ndarray) -> np.ndarray:
    """Return a uint8 mask of the largest bright blob (the plume core).

    Same brightness-first logic as roi_track: Otsu threshold, raised to the
    top-7% hot pixels when a scene is uniformly bright (day launches).
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = blur >= otsu_val
    if bright.mean() > 0.5:
        bright = blur >= float(np.percentile(blur, 93))
    mask = (bright.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros((h, w), np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def symmetry_score(mask: np.ndarray, cx: int) -> float:
    """Mirror-overlap symmetry of the mask about a vertical axis at column cx.

    1.0 = perfectly symmetric, 0.0 = no overlap between mirrored halves.
    """
    h, w = mask.shape
    left_w, right_w = int(round(cx)), w - int(round(cx))
    if left_w <= 0 or right_w <= 0:
        return 0.0
    L = mask[:, :left_w].astype(np.float32)
    R = mask[:, left_w:][:, ::-1].astype(np.float32)
    m = min(L.shape[1], R.shape[1])
    L, R = L[:, :m], R[:, :m]
    total = float(L.sum() + R.sum())
    if total == 0:
        return 0.0
    return max(0.0, 1.0 - float(np.abs(L - R).sum()) / total)


def shape_features(mask: np.ndarray) -> np.ndarray:
    """Plume shape stats from the full-resolution mask (all normalized)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return np.array([0.0, 0.0, 0.0, 0.5, 0.5], np.float32)
    area_frac = len(xs) / (w * h)
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    aspect = bw / max(1.0, bh)
    cx = int(round(xs.mean()))
    sym = symmetry_score(mask, cx)
    cx_norm = (xs.mean() + 0.5) / w
    cy_norm = (ys.mean() + 0.5) / h
    return np.array([area_frac, aspect, sym, cx_norm, cy_norm], np.float32)


def plume_bg_contrast(gray_work: np.ndarray, mask_work: np.ndarray) -> float:
    """Plume brightness relative to a surrounding annulus, in [-1, 1]."""
    m = mask_work > 0
    if m.sum() < 10:
        return 0.0
    d = max(3, round(0.05 * (gray_work.shape[0] + gray_work.shape[1]) / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    annulus = (cv2.dilate(m.astype(np.uint8), kernel) > 0) & (~m)
    inside = float(gray_work[m].mean())
    outside = float(gray_work[annulus].mean()) if annulus.sum() >= 10 else inside
    return float(np.clip((inside - outside) / 255.0, -1.0, 1.0))


def hsv_histogram(img_work: np.ndarray) -> np.ndarray:
    """Coarse-binned, normalized HSV histograms concatenated (24 dims)."""
    hsv = cv2.cvtColor(img_work, cv2.COLOR_BGR2HSV)
    hh, _ = np.histogram(hsv[:, :, 0].ravel(), bins=H_BINS, range=(0, 180))
    sh, _ = np.histogram(hsv[:, :, 1].ravel(), bins=S_BINS, range=(0, 256))
    vh, _ = np.histogram(hsv[:, :, 2].ravel(), bins=V_BINS, range=(0, 256))
    total = hsv.shape[0] * hsv.shape[1]
    return np.concatenate([hh, sh, vh]).astype(np.float32) / total


def edge_density_outside(gray_work: np.ndarray, mask_work: np.ndarray) -> float:
    """Canny edge fraction outside the plume mask (debris / separation debris)."""
    med = float(np.median(gray_work))
    lo = max(1, int(0.66 * med))
    hi = min(255, int(1.33 * med))
    if hi <= lo:
        hi = lo + 1
    edges = cv2.Canny(gray_work, lo, hi)
    outside = ~(mask_work > 0)
    return float(np.count_nonzero(edges & outside) / max(1, int(outside.sum())))


def flow_stats(prev_gray: np.ndarray | None, gray_work: np.ndarray) -> tuple[float, float]:
    """Mean and p95 Farneback flow magnitude, normalized by frame diagonal."""
    if prev_gray is None:
        return 0.0, 0.0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_work, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    diag = np.hypot(gray_work.shape[0], gray_work.shape[1])
    return float(mag.mean() / diag), float(np.percentile(mag, 95) / diag)


def extract_features(img: np.ndarray, gray_full: np.ndarray, mask_full: np.ndarray,
                     gray_work: np.ndarray, prev_gray: np.ndarray | None) -> np.ndarray:
    """Build the fixed-length feature vector for one crop."""
    mask_work = cv2.resize(mask_full, gray_work.shape[::-1], interpolation=cv2.INTER_NEAREST)
    shape = shape_features(mask_full)
    contrast = plume_bg_contrast(gray_work, mask_work)
    hist = hsv_histogram(img)
    edge_density = edge_density_outside(gray_work, mask_work)
    flow_mean, flow_p95 = flow_stats(prev_gray, gray_work)
    return np.concatenate([shape, [contrast], hist,
                           [edge_density, flow_mean, flow_p95]]).astype(np.float32)


def process_video(video_dir, features_base, *, work_size: int = WORK_SIZE) -> dict:
    """Compute features for one video into data/features/; return a summary dict."""
    video_dir = Path(video_dir)
    features_base = Path(features_base)
    name = video_dir.name
    crops = sorted((video_dir / "roi").glob("frame_*.jpg"))
    if not crops:
        return {"video": name, "error": "no roi crops", "frames": 0}

    fallback = set()
    boxes_path = video_dir / "roi_boxes.json"
    if boxes_path.exists():
        with open(boxes_path) as f:
            fallback = set(json.load(f).get("fallback_frames", []))

    lighting, timestamps = "unknown", {}
    manifest_path = video_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            lighting = manifest.get("lighting", "unknown")
            timestamps = {fr["file"]: fr["timestamp"]
                          for fr in manifest.get("frames", [])}
        except (json.JSONDecodeError, OSError):
            pass

    features_base.mkdir(parents=True, exist_ok=True)
    rows, frame_index = [], []
    prev_gray = None
    for fp in crops:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_full = plume_mask(gray_full)
        gray_work = cv2.resize(gray_full, (work_size, work_size),
                               interpolation=cv2.INTER_AREA)
        rows.append(extract_features(img, gray_full, mask_full, gray_work, prev_gray))
        frame_index.append({
            "row": len(rows) - 1,
            "file": fp.name,
            "timestamp": timestamps.get(fp.name, 0.0),
            "fallback": fp.name in fallback,
        })
        prev_gray = gray_work

    X = np.stack(rows).astype(np.float32)
    np.save(features_base / f"{name}.npy", X)

    video_meta = {"name": name, "lighting": lighting}
    if manifest_path.exists():
        video_meta.update({k: manifest.get("video", {}).get(k)
                           for k in ("source", "mission", "label")})
    frame_index_json = {
        "video": video_meta,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "frames": frame_index,
    }
    with open(features_base / f"{name}.frame_index.json", "w") as f:
        json.dump(frame_index_json, f, indent=2)

    return {"video": name, "frames": len(rows),
            "fallbacks": sum(1 for fr in frame_index if fr["fallback"])}


def batch(frames_base, features_base, *, only: str | None = None, force: bool = False,
          skip: set | None = None, after_video=None, **kw) -> int:
    """Run process_video over every video dir; `after_video(name)` is called per video."""
    frames_base = Path(frames_base)
    features_base = Path(features_base)
    dirs = [d for d in iter_video_dirs(frames_base)
            if not only or only in d.name]
    if not dirs:
        print("no video dirs with ROI crops found"
              + (f" matching '{only}'" if only else ""), file=sys.stderr)
        return 1

    n_done = 0
    for video_dir in dirs:
        name = video_dir.name
        if skip and name in skip:
            print(f"{name}: skipped (already on Drive)")
            continue
        out = features_base / f"{name}.npy"
        if out.exists() and not force:
            print(f"{name}: skipped (cached .npy)")
            continue
        print(f"processing {name} ...", file=sys.stderr)
        summary = process_video(video_dir, features_base, **kw)
        if "error" in summary:
            print(f"{name}: skipped ({summary['error']})", file=sys.stderr)
            continue
        n_done += 1
        print(f"{summary['video']}: {summary['frames']} frames, "
              f"{summary['fallbacks']} fallback crops", flush=True)
        if after_video:
            after_video(name)
    print(f"\n{n_done} videos processed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES),
                    help="frames base dir (default: %(default)s)")
    ap.add_argument("--features", default=str(DEFAULT_FEATURES),
                    help="output features base dir (default: %(default)s)")
    ap.add_argument("--only", default=None,
                    help="only process videos whose name contains this substring")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if .npy exists")
    ap.add_argument("--work-size", type=int, default=WORK_SIZE,
                    help="max crop dimension for color/edge/flow (default: %(default)s)")
    args = ap.parse_args(argv)
    return batch(args.frames, args.features, only=args.only, force=args.force,
                 work_size=args.work_size)


if __name__ == "__main__":
    sys.exit(main())
