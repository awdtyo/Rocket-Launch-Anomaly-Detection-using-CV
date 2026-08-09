#!/usr/bin/env python3
"""Extract evenly sampled frames from raw launch videos into data/frames/.

Per video (data/raw/{normal,anomaly}/*.mp4):
  - PySceneDetect shot detection (ContentDetector) over the full video.
  - Shots shorter than --min-shot seconds are discarded (countdown graphics and
    rapid cutaways, not sustained ascent footage).
  - Frames are sampled at --fps (default 2fps) inside the kept shots and written
    to data/frames/{video_name}/frame_{idx:05d}.jpg.
  - A rough per-video lighting tag (day/dusk/night) is derived from the mean
    brightness of a sample of frames across the whole video, since downstream
    training splits by lighting condition.
  - data/frames/{video_name}/manifest.json records every frame's timestamp and
    source shot index plus the lighting tag (and shot/video metadata).

Processed videos (manifest present) are skipped; use --force to re-extract and
--only <substring> to process a single video for quick testing. All outputs are
cached to disk, never recomputed from raw video more than once.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector
except ImportError:
    sys.exit("scenedetect is required (pip install scenedetect). See requirements.txt.")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_FRAMES = ROOT / "data" / "frames"

_VIDEO_EXTS = ("*.mp4", "*.mkv", "*.webm")


def iter_videos(raw_base: Path):
    """Yield (label, path) for every video file under raw_base/{label}/."""
    for label in ("normal", "anomaly"):
        dirpath = raw_base / label
        if not dirpath.is_dir():
            continue
        for pattern in _VIDEO_EXTS:
            yield from ((label, p) for p in sorted(dirpath.glob(pattern)))


def parse_name(stem: str) -> tuple[str, str]:
    """Split '{source}_{mission}' back into (source, mission); fall back to stem."""
    if "_" in stem:
        return stem.split("_", 1)
    return stem, stem


def scene_seconds(scene) -> tuple[float, float]:
    """Return (start_sec, end_sec) from a scene, tolerant of API versions."""
    if hasattr(scene, "start"):
        s, e = scene.start, scene.end
    else:
        s, e = scene

    def sec(tc) -> float:
        return float(getattr(tc, "seconds", tc.get_seconds()))

    return sec(s), sec(e)


def _cv2_readable(path: Path) -> bool:
    cap = cv2.VideoCapture(str(path))
    ok = cap.read()[0]
    cap.release()
    return ok


def ensure_decodable(video_path: Path, frames_base: Path) -> Path:
    """Return a cv2-decodable copy of the video, transcoding AV1/VP9 once to H.264.

    YouTube webcasts often arrive as AV1/VP9 which OpenCV cannot decode; the
    transcoded copy is cached under {frames_base}/_decoded/ and reused.
    """
    if _cv2_readable(video_path):
        return video_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{video_path} is not OpenCV-decodable and ffmpeg is missing")

    out = frames_base / "_decoded" / f"{video_path.stem}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
    return out


def detect_shots(video_path: Path, threshold: float) -> list[tuple[float, float]]:
    """Detect shot boundaries with ContentDetector; returns list of (start, end) secs."""
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold))
    sm.detect_scenes(video, show_progress=False)
    return [scene_seconds(s) for s in sm.get_scene_list(start_in_scene=True)]


def classify_lighting(mean_brightness: float, night_th: float, day_th: float) -> str:
    if mean_brightness < night_th:
        return "night"
    if mean_brightness > day_th:
        return "day"
    return "dusk"


def write_frame(path: Path, frame: np.ndarray, quality: int) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"failed to encode JPEG for {path}")
    path.write_bytes(buf.tobytes())


def extract_video(label: str, video_path: Path, frames_base: Path, args) -> dict:
    """Extract frames + manifest for one video. Returns the summary dict."""
    name = video_path.stem
    source, mission = parse_name(name)
    out_dir = frames_base / name
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = video_path
    video_path = ensure_decodable(video_path, frames_base)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return {"label": label, "name": name, "error": "cannot open video"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0.0
    sample_every = max(1, round(fps / args.fps))

    # --- shots ---
    shots = detect_shots(video_path, args.threshold)
    if not shots:  # paranoia: fall back to a single shot spanning the video
        shots = [(0.0, duration)]
    kept, discarded = [], []
    for start, end in (kept_shots := [(s, e) for s, e in shots]):
        (kept if (end - start) >= args.min_shot else discarded).append((start, end))

    # Map kept shots to cv2 frame ranges (decoupled from scenedetect's fps).
    shot_frames = []
    for start, end in kept:
        shot_frames.append((int(round(start * fps)), int(round(end * fps))))

    # --- lighting sample points across the whole video ---
    n_samp = max(2, args.brightness_samples)
    sample_nums = {round(i * (total_frames - 1) / (n_samp - 1)) for i in range(n_samp)}
    brightness, seen = 0.0, 0

    # --- single decode pass: write sampled frames + gather brightness ---
    shot_idx, shot_frames_written = 0, 0
    frames_meta, written, frame_num = [], 0, 0
    while True:
        if args.progress and frame_num % 10000 == 0:
            print(f"  {name}: {frame_num}/{total_frames} frames ...", file=sys.stderr, end="\r")
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num in sample_nums:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness += float(gray.mean())
            seen += 1
        while shot_idx < len(shot_frames) and frame_num >= shot_frames[shot_idx][1]:
            shot_idx += 1
        if shot_idx < len(shot_frames):
            s_start, s_end = shot_frames[shot_idx]
            if s_start <= frame_num < s_end and (frame_num - s_start) % sample_every == 0:
                fname = f"frame_{written:05d}.jpg"
                write_frame(out_dir / fname, frame, args.quality)
                frames_meta.append({
                    "index": written,
                    "timestamp": round(frame_num / fps, 3),
                    "shot_index": shot_idx,
                    "file": fname,
                })
                written += 1
        frame_num += 1
    cap.release()

    if seen:
        brightness /= seen
    lighting = classify_lighting(brightness, args.night_threshold, args.day_threshold)

    manifest = {
        "video": {
            "name": name,
            "label": label,
            "source": source,
            "mission": mission,
            "path": str(raw_path),
        },
        "lighting": lighting,
        "fps": round(fps, 4),
        "duration_seconds": round(duration, 3),
        "sampling": {
            "fps": args.fps,
            "min_shot_seconds": args.min_shot,
            "content_threshold": args.threshold,
        },
        "shot_count": len(shots),
        "discarded_shot_count": len(discarded),
        "shots": [
            {"index": i, "start_seconds": round(s, 3), "end_seconds": round(e, 3),
             "duration_seconds": round(e - s, 3), "discarded": (s, e) in discarded}
            for i, (s, e) in enumerate(shots)
        ],
        "frames": frames_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "label": label,
        "name": name,
        "frames": written,
        "shots_kept": len(kept),
        "shots_discarded": len(discarded),
        "lighting": lighting,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="raw video base dir (default: %(default)s)")
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES),
                    help="output frames base dir (default: %(default)s)")
    ap.add_argument("--fps", type=float, default=2.0, help="sample rate inside kept shots (default: %(default)s)")
    ap.add_argument("--min-shot", type=float, default=2.0,
                    help="discard shots shorter than this many seconds (default: %(default)s)")
    ap.add_argument("--threshold", type=float, default=27.0,
                    help="PySceneDetect ContentDetector threshold (default: %(default)s)")
    ap.add_argument("--brightness-samples", type=int, default=24,
                    help="frames sampled for lighting estimate (default: %(default)s)")
    ap.add_argument("--night-threshold", type=float, default=45.0,
                    help="mean brightness below this tags night (default: %(default)s)")
    ap.add_argument("--day-threshold", type=float, default=100.0,
                    help="mean brightness above this tags day (default: %(default)s)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default: %(default)s)")
    ap.add_argument("--only", default=None, help="only process videos whose name contains this substring")
    ap.add_argument("--force", action="store_true", help="re-extract even if manifest exists")
    ap.add_argument("--progress", action="store_true", help="print frame-level progress")
    args = ap.parse_args(argv)

    raw_base, frames_base = Path(args.raw), Path(args.frames)
    videos = [v for v in iter_videos(raw_base) if not args.only or args.only in v[1].stem]
    if not videos:
        print("no videos found" + (f" matching '{args.only}'" if args.only else ""), file=sys.stderr)
        return 1

    n_new = 0
    for label, path in videos:
        name = path.stem
        manifest_path = frames_base / name / "manifest.json"
        if manifest_path.exists() and not args.force:
            try:
                with open(manifest_path) as f:
                    cached = json.load(f)
                is_cached = bool(cached.get("frames"))
            except (json.JSONDecodeError, OSError):
                is_cached = False
            if is_cached:
                print(f"normal {name}: skipped (cached manifest)")
                continue
        print(f"processing {label} {name} ...", file=sys.stderr)
        summary = extract_video(label, path, frames_base, args)
        if "error" in summary:
            print(f"error {name}: {summary['error']}", file=sys.stderr)
            continue
        n_new += 1
        print(f"{summary['label']:7s} {summary['name']}: {summary['frames']} frames, "
              f"{summary['shots_kept']} shots kept, {summary['shots_discarded']} discarded "
              f"(<{args.min_shot}s), lighting={summary['lighting']}", flush=True)
    print(f"\n{n_new} videos extracted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
