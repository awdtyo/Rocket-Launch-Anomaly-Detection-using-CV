#!/usr/bin/env python3
"""Add anomaly launch videos from a URL list, then download them.

Convenience wrapper around add_video.py + download.py for the failure-footage
gathering workflow: you paste one YouTube URL per line (stdin or --file), and
for each URL this script

  1. extracts the video ID (same regex as add_video.py),
  2. fetches the real title via yt-dlp metadata (no download yet),
  3. asks for a one-line confirmation (y = use auto-detected title as mission,
     n = skip, edit = type a clean mission name like "CRS-7"),
  4. infers the source from the channel name (spacex / isro / nasa) or asks,
  5. appends the entry under the `anomaly:` key of configs/videos.yaml
     (skipping links already in the config),
then prints a summary and runs scripts/download.py so the videos land in
data/raw/anomaly/.

Usage:
    python scripts/add_anomaly_videos.py < urls.txt
    python scripts/add_anomaly_videos.py --file urls.txt
    cat urls.txt | python scripts/add_anomaly_videos.py --verbose
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from yt_dlp import YoutubeDL

from add_video import (DEFAULT_CONFIG, append_entry, entry_text,
                       existing_video_id, extract_youtube_id, load_config)
from download import DEFAULT_BASE, sanitize

SCRIPT_DIR = Path(__file__).resolve().parent

_SOURCE_HINTS = {
    "spacex": ("spacex", "falcon"),
    "isro": ("isro",),
    "nasa": ("nasa",),
}


def read_urls(file_arg: str | None) -> list[str]:
    """Read one URL per line from --file, or from stdin. Blank/# lines ignored."""
    if file_arg:
        lines = Path(file_arg).read_text().splitlines()
    else:
        lines = sys.stdin.read().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def fetch_metadata(video_id: str) -> dict | None:
    """Fetch video metadata (title/channel) without downloading. None on error."""
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            return ydl.extract_info(video_id, download=False)
    except Exception as exc:
        print(f"  error: could not fetch metadata: {exc}", file=sys.stderr)
        return None


def channel_name(info: dict) -> str:
    return str(info.get("channel") or info.get("uploader") or "-")


def infer_source(info: dict) -> str | None:
    """Map the channel name to a known source, or None if it's not one of ours."""
    lower = channel_name(info).lower()
    for source, hints in _SOURCE_HINTS.items():
        if any(h in lower for h in hints):
            return source
    return None


def resolve_source(info: dict) -> str:
    src = infer_source(info)
    if src:
        print(f"  source: {src} (channel: {channel_name(info)})")
        return src
    while True:
        ans = input(f"  source? channel is '{channel_name(info)}' "
                    f"({', '.join(_SOURCE_HINTS)}): ").strip().lower()
        if ans in _SOURCE_HINTS:
            return ans
        print(f"  unknown source '{ans}' — pick one of {', '.join(_SOURCE_HINTS)}")


def confirm_mission(info: dict, video_id: str) -> str | None:
    """Ask y/n/edit; returns the mission name, or None to skip this video."""
    title = info.get("title") or video_id
    auto = sanitize(title)
    print(f"  title: {title}")
    print(f"  video: https://youtu.be/{video_id}")
    while True:
        ans = input(f"  add as anomaly? mission [{auto}] (y/n/edit): ").strip()
        if ans == "" or ans.lower() in ("y", "yes"):
            return auto
        if ans.lower() in ("n", "no"):
            return None
        if ans.lower() in ("e", "edit"):
            cleaned = input("  cleaned mission name: ").strip()
            if cleaned:
                return cleaned
            print("  empty name — try again (or n to skip)")
            continue
        return ans  # treat anything else as an inline mission name


def add_entry(video_id: str, mission: str, source: str, config: str) -> None:
    """Append an entry under the anomaly: key, preserving existing formatting."""
    entry = {"mission": mission, "source": source, "video_id": video_id}
    text_lines = [ln + "\n" for ln in entry_text(entry).split("\n")]
    with open(config) as f:
        lines = f.readlines()
    updated = append_entry(lines, "anomaly", text_lines)
    with open(config, "w") as f:
        f.writelines(updated)
    cfg = load_config(config)
    assert existing_video_id(cfg, video_id) == "anomaly"
    print("  added:")
    for ln in text_lines:
        print("    " + ln.rstrip("\n"))


def run_download(args) -> int:
    cmd = [sys.executable, str(SCRIPT_DIR / "download.py"),
           "--config", args.config, "--base", args.base,
           "--max-height", str(args.max_height)]
    if args.verbose:
        cmd.append("--verbose")
    print(f"\nRunning scripts/download.py ({cmd[0]} {cmd[1]} ...)")
    return subprocess.run(cmd).returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="file with one YouTube URL per line "
                                   "(default: read from stdin)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="YAML config to update (default: %(default)s)")
    ap.add_argument("--base", default=str(DEFAULT_BASE),
                    help="output base dir, videos land in {base}/{label}/ "
                         "(default: %(default)s)")
    ap.add_argument("--max-height", type=int, default=1080,
                    help="max resolution for the follow-up download (default: %(default)s)")
    ap.add_argument("--verbose", action="store_true",
                    help="show yt-dlp download progress")
    args = ap.parse_args(argv)

    urls = read_urls(args.file)
    if not urls:
        print("no URLs given (feed stdin or use --file)", file=sys.stderr)
        return 1

    config = Path(args.config)
    cfg = load_config(config)
    added, skipped, failed = [], 0, 0

    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        video_id = extract_youtube_id(url)
        if not video_id:
            print("  error: not a recognized YouTube URL", file=sys.stderr)
            failed += 1
            continue

        dup = existing_video_id(cfg, video_id)
        if dup:
            print(f"  skipped: video_id {video_id} already in {config.name} "
                  f"under '{dup}'.")
            skipped += 1
            continue

        info = fetch_metadata(video_id)
        if info is None:
            failed += 1
            continue

        mission = confirm_mission(info, video_id)
        if mission is None:
            print("  skipped.")
            continue
        source = resolve_source(info)

        add_entry(video_id, mission, source, str(config))
        added.append((mission, source, video_id))
        cfg = load_config(config)  # refresh for later dedup checks

    print(f"\n=== summary: {len(urls)} URLs, {len(added)} added, "
          f"{skipped} skipped (already in config), {failed} failed. ===")
    for mission, source, video_id in added:
        print(f"  - {mission} ({source}) https://youtu.be/{video_id}")

    return run_download(args)


if __name__ == "__main__":
    sys.exit(main())
