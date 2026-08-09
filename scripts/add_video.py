#!/usr/bin/env python3
"""Append a hand-picked YouTube launch video to configs/videos.yaml.

Small CLI helper that replaces the API-driven fetch_launch_list.py approach:
you paste in URLs you've verified by hand, and this script records them.

Usage:
    python scripts/add_video.py <youtube_url> --mission "Name" --source spacex --tag normal

Extracts the video ID from any standard YouTube URL (youtube.com/watch?v=,
youtu.be/, embed/, live/, shorts/), errors out if that video_id is already in
the config, then appends the entry under the right `normal:` / `anomaly:` key.
The file is edited textually so existing comments and formatting are preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "videos.yaml"

_YT_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?[^#]*v=|embed/|live/|shorts/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)
_GROUP_KEY_RE = re.compile(r"^(normal|anomaly):")


def extract_youtube_id(url: str) -> str | None:
    """Return the YouTube video ID from a URL, or None if not a YouTube link."""
    m = _YT_RE.search(url)
    return m.group(1) if m else None


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def existing_video_id(cfg: dict, video_id: str) -> str | None:
    """Return the group label where video_id already appears, or None."""
    for group in ("normal", "anomaly"):
        for item in cfg.get(group, []) or []:
            if item.get("video_id") == video_id:
                return group
    return None


def entry_text(entry: dict) -> str:
    """Render a config entry as indented YAML list lines (no trailing newline).

    Produces:
        - mission: Name
          source: spacex
          video_id: abc123
    """
    body = yaml.safe_dump(entry, default_flow_style=False, sort_keys=False).rstrip("\n")
    lines = body.split("\n")
    return "\n".join(["  - " + lines[0]] + ["    " + ln for ln in lines[1:]])


def append_entry(lines: list[str], group: str, text_lines: list[str]) -> list[str]:
    """Insert text_lines (each ending in \\n) under the top-level group key.

    Keeps the surrounding comments/blank lines intact by editing line-by-line.
    """
    out = list(lines)
    key_idx = {m.group(1): i for i, ln in enumerate(out) if (m := _GROUP_KEY_RE.match(ln))}

    if group in key_idx:
        start = key_idx[group]
        end = min((i for g, i in key_idx.items() if g != group and i > start), default=len(out))
        last = start
        for i in range(start, end):
            if out[i].strip() and not out[i].lstrip().startswith("#"):
                last = i
        out[last + 1:last + 1] = text_lines
    else:
        if out and out[-1].strip():
            if not out[-1].endswith("\n"):
                out[-1] += "\n"
            out.append("\n")
        out.append(f"{group}:\n")
        out.extend(text_lines)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", help="YouTube URL in any standard format")
    ap.add_argument("--mission", required=True, help="mission name, e.g. 'Starlink 9-2'")
    ap.add_argument("--source", required=True, help="provider/agency, e.g. spacex, isro, nasa")
    ap.add_argument("--tag", required=True, choices=("normal", "anomaly"),
                    help="normal = nominal ascent (training), anomaly = known failure (eval only)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="YAML config to update (default: %(default)s)")
    args = ap.parse_args(argv)

    video_id = extract_youtube_id(args.url)
    if not video_id:
        print(f"error: not a recognized YouTube URL: {args.url}", file=sys.stderr)
        return 1

    path = Path(args.config)
    cfg = load_config(path)
    dup = existing_video_id(cfg, video_id)
    if dup:
        print(f"error: video_id {video_id} already in {path.name} under '{dup}'.",
              file=sys.stderr)
        return 1

    entry = {"mission": args.mission.strip(),
             "source": args.source.strip(),
             "video_id": video_id}
    text_lines = [ln + "\n" for ln in entry_text(entry).split("\n")]

    with open(path) as f:
        lines = f.readlines()
    updated = append_entry(lines, args.tag, text_lines)
    with open(path, "w") as f:
        f.writelines(updated)

    # Sanity check: the file must still parse and contain the new entry.
    cfg = load_config(path)
    assert existing_video_id(cfg, video_id) == args.tag

    print(f"added to {path.name} under '{args.tag}':")
    for ln in text_lines:
        print("  " + ln.rstrip("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
