#!/usr/bin/env python3
"""Download rocket launch footage from YouTube (max 1080p) into data/raw/.

Reads a YAML config listing YouTube video IDs grouped under `normal:` and
`anomaly:` keys with a `mission` name and `source` per entry, downloads each
into data/raw/{label}/{source}_{mission}.mp4, skips files that already exist,
and prints a summary table (mission, duration, resolution, filesize).

Requires ffmpeg on PATH to merge separate video/audio streams into an mp4
container; falls back to a single combined stream when ffmpeg is missing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import yaml
from yt_dlp import YoutubeDL

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "videos.yaml"
DEFAULT_BASE = Path(__file__).resolve().parent.parent / "data" / "raw"


def sanitize(name: str) -> str:
    """Normalize a name for use in a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())


def load_config(path: str | Path) -> list[dict]:
    """Load video entries from a YAML config, tagged with their group label."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    entries = []
    for label in ("normal", "anomaly"):
        for item in cfg.get(label, []) or []:
            entry = dict(item)
            entry["label"] = label
            entries.append(entry)
    if not entries:
        sys.exit("No video entries found in config (expected 'normal:' / 'anomaly:' keys).")
    return entries


def target_path(base: Path, entry: dict) -> Path:
    """Resolve the output path for an entry: base/{label}/{source}_{mission}.mp4."""
    fname = f"{sanitize(entry['source'])}_{sanitize(entry['mission'])}.mp4"
    return base / entry["label"] / fname


def probe_info(entry: dict) -> dict | None:
    """Fetch video metadata without downloading (used for already-cached files)."""
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            return ydl.extract_info(entry["video_id"], download=False)
    except Exception:
        return None


def download_one(entry: dict, base: Path, max_height: int = 1080, verbose: bool = False):
    """Download a single video at max max_height. Returns (path, status, info).

    status is 'ok', 'skipped', or 'error: <detail>'.
    """
    out = target_path(base, entry)
    if out.exists():
        return out, "skipped", probe_info(entry)

    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        fmt = f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]"
    else:
        fmt = f"best[height<={max_height}]"
    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(out),
        "quiet": not verbose,
        "no_warnings": not verbose,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(entry["video_id"])
        return out, "ok", info
    except Exception as exc:
        return out, f"error: {exc}", None


def _fmt_duration(seconds) -> str:
    if not seconds:
        return "-"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_size(num_bytes: int) -> str:
    return f"{num_bytes / 1e6:6.1f} MB"


def _resolution(info: dict | None) -> str:
    if not info:
        return "-"
    if info.get("width") and info.get("height"):
        return f"{info['width']}x{info['height']}"
    return str(info.get("resolution", "-"))


def _format_table(headers: list[str], rows: list[dict]) -> str:
    widths = {h: max(len(h), *(len(str(r.get(h, ""))) for r in rows)) for h in headers}
    lines = ["  ".join(h.ljust(widths[h]) for h in headers)]
    lines.append("  ".join("-" * widths[h] for h in headers))
    for r in rows:
        lines.append("  ".join(str(r.get(h, "")).ljust(widths[h]) for h in headers))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="YAML config with video entries (default: %(default)s)")
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="output base dir, videos land in {base}/{label}/ (default: %(default)s)")
    ap.add_argument("--max-height", type=int, default=1080,
                    help="max resolution to download (default: %(default)s)")
    ap.add_argument("--verbose", action="store_true", help="show yt-dlp progress output")
    args = ap.parse_args(argv)

    base = Path(args.base)
    entries = load_config(args.config)

    rows, n_ok, n_skip, n_err = [], 0, 0, 0
    for entry in entries:
        label, source, mission = entry["label"], entry["source"], entry["mission"]
        print(f"  {label:7s} {source}_{mission} ({entry['video_id']}) ...", file=sys.stderr)
        out, status, info = download_one(entry, base, args.max_height, args.verbose)
        if status == "ok":
            n_ok += 1
        elif status == "skipped":
            n_skip += 1
        else:
            n_err += 1
        rows.append({
            "label": label,
            "source": source,
            "mission": mission,
            "duration": _fmt_duration(info.get("duration") if info else None),
            "resolution": _resolution(info),
            "size": _fmt_size(out.stat().st_size) if out.exists() else "-",
            "status": status[:60],
        })

    print()
    print(_format_table(
        ["label", "source", "mission", "duration", "resolution", "size", "status"], rows))
    print(f"\n{len(entries)} videos: {n_ok} downloaded, {n_skip} skipped (cached), {n_err} failed.")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
