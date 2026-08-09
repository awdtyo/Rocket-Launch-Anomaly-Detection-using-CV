#!/usr/bin/env python3
"""Detect and crop the rocket + exhaust plume region for each extracted frame.

Reads data/frames/{video_name}/frame_*.jpg, finds the plume bounding box with a
brightness + motion blob detector (the plume is the brightest, most dynamic
region of an ascent shot), temporally smooths the box across frames so it does
not jitter, expands it by a configurable margin (default 20%), and writes
crops to data/frames/{video_name}/roi/ plus data/frames/{video_name}/roi_boxes.json
recording per-frame box coordinates.

Frames where no confident blob is found fall back to a full-frame crop; those
frames are logged to stderr and recorded in roi_boxes.json['fallback_frames']
for spot-checking. Videos that already have roi_boxes.json are skipped on rerun
(--force to redo); videos with no extracted frames are skipped gracefully.

Works both as a CLI and as an importable function (process_video) so it can be
driven from a Colab notebook. Outputs are cached to disk, never recomputed.
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


def iter_video_dirs(frames_base: Path):
    """Yield every video frame dir under frames_base that has frame images."""
    for video_dir in sorted(frames_base.iterdir()):
        if video_dir.is_dir() and any(video_dir.glob("frame_*.jpg")):
            yield video_dir


def load_frames(video_dir: Path) -> list[Path]:
    return sorted(video_dir.glob("frame_*.jpg"))


def detect_plume(gray, prev_gray, motion_thresh: int, min_area_frac: float):
    """Return (box, area_fraction) for the brightest, most dynamic blob.

    Brightness-first: Otsu isolates the bright plume against a dark sky; when a
    scene is uniformly bright (day launches) the threshold is raised to the top
    hot pixels so the sun/white tower don't swallow the plume. A temporal motion
    mask (absdiff from the previous frame) *boosts* the score of blobs that are
    also dynamic rather than gating on it, since frames are sampled 0.5 s apart
    and the whole scene shifts with camera pan. Returns (None, 0.0) when no
    plausible blob survives.
    """
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_val, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = blur >= otsu_val
    if bright.mean() > 0.5:
        bright = blur >= float(np.percentile(blur, 93))
    mask = (bright.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    motion = None
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        motion = diff > motion_thresh

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    best, best_score, best_frac = None, 0.0, 0.0
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        area_frac = (bw * bh) / (w * h)
        score = area_frac
        if motion is not None:
            region = mask[y:y + bh, x:x + bw]
            dynamism = np.count_nonzero(region & motion[y:y + bh, x:x + bw]) / max(
                1, np.count_nonzero(region))
            score += 0.5 * area_frac * dynamism
        if score > best_score:
            best, best_score, best_frac = (x, y, bw, bh), score, area_frac
    if best is None or best_frac < min_area_frac or best_frac > 0.85:
        return None, 0.0
    return best, best_frac


def expand_margin(box, frame_w, frame_h, margin: float):
    """Grow a box by `margin` x its own size on each side, clamped to the frame."""
    x, y, bw, bh = box
    pw, ph = int(bw * margin), int(bh * margin)
    x0 = max(0, x - pw)
    y0 = max(0, y - ph)
    x1 = min(frame_w, x + bw + pw)
    y1 = min(frame_h, y + bh + ph)
    return (x0, y0, x1 - x0, y1 - y0)


def process_video(video_dir, *, margin: float = 0.2, min_confidence: float = 0.005,
                  alpha: float = 0.4, motion_thresh: int = 12,
                  min_area_frac: float = 0.002, jpeg_quality: int = 95,
                  progress=None) -> dict:
    """Crop every frame of one video into its roi/ dir; return a summary dict.

    `progress` is an optional callback called as progress(i, total) after each
    frame. roi_boxes.json is written into video_dir with per-frame boxes.
    """
    video_dir = Path(video_dir)
    frames = load_frames(video_dir)
    name = video_dir.name
    if not frames:
        return {"video": name, "error": "no frames", "frames": 0,
                "boxes": 0, "fallbacks": 0, "fallback_frames": []}

    out_dir = video_dir / "roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes, confs, fallbacks = {}, {}, []
    prev_gray, smooth = None, None
    width = height = 0
    for i, fp in enumerate(frames):
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_box, frac = detect_plume(gray, prev_gray, motion_thresh, min_area_frac)
        prev_gray = gray

        fname = fp.name
        if det_box is None or frac < min_confidence:
            box = (0, 0, width, height)
            confs[fname] = 0.0
            fallbacks.append(fname)
            crop = frame
        else:
            if smooth is None:
                smooth = det_box
            else:
                smooth = [round(alpha * t + (1 - alpha) * s)
                          for t, s in zip(det_box, smooth)]
            box = expand_margin(smooth, width, height, margin)
            confs[fname] = round(frac, 4)
            crop = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

        cv2.imwrite(str(out_dir / fname), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        boxes[fname] = [int(v) for v in box]
        if progress:
            progress(i + 1, len(frames))

    meta = {"name": name}
    manifest_path = video_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            meta = manifest.get("video", {})
            meta["lighting"] = manifest.get("lighting", "unknown")
        except (json.JSONDecodeError, OSError):
            pass

    boxes_json = {
        "video": meta,
        "frame_width": width,
        "frame_height": height,
        "config": {
            "margin": margin,
            "min_confidence": min_confidence,
            "alpha": alpha,
            "motion_thresh": motion_thresh,
            "min_area_frac": min_area_frac,
        },
        "boxes": boxes,
        "confidences": confs,
        "fallback_frames": fallbacks,
    }
    with open(video_dir / "roi_boxes.json", "w") as f:
        json.dump(boxes_json, f, indent=2)

    return {"video": name, "frames": len(frames), "boxes": len(boxes),
            "fallbacks": len(fallbacks), "fallback_frames": fallbacks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES),
                    help="frames base dir (default: %(default)s)")
    ap.add_argument("--only", default=None,
                    help="only process videos whose name contains this substring")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if roi_boxes.json exists")
    ap.add_argument("--margin", type=float, default=0.2,
                    help="crop margin as a fraction of the blob box (default: %(default)s)")
    ap.add_argument("--min-confidence", type=float, default=0.01,
                    help="blob area fraction below this triggers full-frame fallback (default: %(default)s)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="temporal smoothing weight for the new detection (default: %(default)s)")
    ap.add_argument("--motion-thresh", type=int, default=12,
                    help="absolute-difference threshold for the motion mask (default: %(default)s)")
    ap.add_argument("--min-area-frac", type=float, default=0.002,
                    help="blobs smaller than this area fraction are ignored (default: %(default)s)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default: %(default)s)")
    args = ap.parse_args(argv)

    frames_base = Path(args.frames)
    dirs = [d for d in iter_video_dirs(frames_base)
            if not args.only or args.only in d.name]
    if not dirs:
        print("no video frame dirs found"
              + (f" matching '{args.only}'" if args.only else ""), file=sys.stderr)
        return 1

    n_done = 0
    for video_dir in dirs:
        name = video_dir.name
        boxes_path = video_dir / "roi_boxes.json"
        if boxes_path.exists() and not args.force:
            print(f"{name}: skipped (cached roi_boxes.json)")
            continue
        print(f"processing {name} ...", file=sys.stderr)

        def progress(i, total):
            if i % 500 == 0 or i == total:
                print(f"  {name}: {i}/{total} frames ...", file=sys.stderr, end="\r")

        summary = process_video(video_dir, margin=args.margin,
                                min_confidence=args.min_confidence, alpha=args.alpha,
                                motion_thresh=args.motion_thresh,
                                min_area_frac=args.min_area_frac,
                                jpeg_quality=args.quality, progress=progress)
        if "error" in summary:
            print(f"{name}: skipped ({summary['error']})", file=sys.stderr)
            continue
        n_done += 1
        fb = summary["fallback_frames"]
        print(f"{name}: {summary['boxes']} crops, {summary['fallbacks']} fallback frames", flush=True)
        if fb:
            shown = ", ".join(fb[:20])
            print(f"  fallback frames ({len(fb)}): {shown}"
                  + (" ..." if len(fb) > 20 else ""), flush=True)
    print(f"\n{n_done} videos processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
