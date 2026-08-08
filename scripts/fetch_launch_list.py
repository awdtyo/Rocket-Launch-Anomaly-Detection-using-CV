#!/usr/bin/env python3
"""Auto-generate a videos.yaml-style config from public launch APIs.

Queries:
  - SpaceX: https://api.spacexdata.com/v5/launches/past — mission name,
    success flag, links.webcast YouTube URL (skips entries without one).
  - Launch Library 2: https://ll.thespacedevs.com/2.3.0/launches/ —
    queried with lsp__name__icontains filters for ISRO/NASA providers
    (server-side filtering keeps this to a handful of pages, staying well
    under LL2's 15-calls-per-hour free tier), pulls mission name, status
    (Launch Successful / Launch Failure / Partial Failure), and vid_urls
    webcasts from the launch or its mission.

YouTube IDs are extracted from youtube.com/watch?v= / youtu.be/ URLs (plus
embed/live/shorts forms). Entries are auto-classified: success -> normal,
failure -> anomaly. Results are written to configs/videos_generated.yaml —
videos.yaml is never overwritten unless --out says so — with a count summary
printed for review before you merge.

Raw API responses are cached in data/api_cache/ so reruns don't re-hit the
APIs. Launch Library 2 free-tier calls are rate-limited between pages.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import requests
import yaml

SPACEX_URL = "https://api.spacexdata.com/v5/launches/past"
LL2_URL = "https://ll.thespacedevs.com/2.3.0/launches/"
LL2_LIMIT = 100
LL2_PROVIDER_QUERIES = ("isro", "nasa")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "configs" / "videos_generated.yaml"
DEFAULT_CACHE = ROOT / "data" / "api_cache"
DEFAULT_MANUAL = ROOT / "configs" / "videos_manual_additions.yaml"

_YT_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?[^#]*v=|embed/|live/|shorts/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})"
)
_ISRO_RE = re.compile(r"isro|indian space research", re.I)
_RELEVANT_RE = re.compile(r"isro|indian space research|nasa|national aeronautics", re.I)


def extract_youtube_id(url: str | None) -> str | None:
    """Return the YouTube video ID from a URL, or None if not a YouTube link."""
    if not url:
        return None
    m = _YT_RE.search(url)
    return m.group(1) if m else None


def _cached_get_json(url: str, cache_key: str, cache_dir: Path,
                     use_cache: bool = True, rate: float = 0.0,
                     retries: int = 4) -> dict:
    """GET a JSON endpoint, caching the raw response to disk between runs."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_key
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, timeout=30,
                headers={"User-Agent": "rocket-anomaly-downloader/0.1"})
            if resp.status_code == 429:
                raise requests.HTTPError("rate limited (HTTP 429)")
            resp.raise_for_status()
            data = resp.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            if rate:
                time.sleep(rate + random.uniform(0.5, 1.5))  # jittered
            return data
        except requests.RequestException as exc:
            last_exc = exc
            is_429 = isinstance(exc, requests.HTTPError) and "429" in str(exc)
            wait = 30 * (attempt + 1) if is_429 else 5 * (2 ** attempt)
            print(f"  retry {attempt + 1}/{retries} for {url} in {wait}s ({exc})",
                  file=sys.stderr)
            time.sleep(wait)
    raise last_exc


def fetch_spacex_launches(cache_dir: Path, use_cache: bool = True,
                          url: str = SPACEX_URL):
    """Fetch all past SpaceX launches; returns (entries, skipped_stats)."""
    data = _cached_get_json(url, "spacex_v5_launches_past.json", cache_dir,
                            use_cache=use_cache)
    entries, skipped = [], {"no_webcast": 0, "upcoming_or_unknown": 0}
    for launch in data:
        links = launch.get("links") or {}
        vid = extract_youtube_id(links.get("webcast") or links.get("video_link"))
        success = launch.get("success")
        if not vid:
            skipped["no_webcast"] += 1
            continue
        if success is None:
            skipped["upcoming_or_unknown"] += 1
            continue
        entries.append({
            "mission": launch.get("name") or launch.get("id"),
            "source": "spacex",
            "video_id": vid,
            "outcome": "success" if success else "failure",
            "date": (launch.get("date_utc") or "")[:10],
        })
    return entries, skipped


def ll2_video_url(launch: dict) -> str | None:
    """Return the first YouTube webcast URL from launch- or mission-level vid_urls."""
    for group in (launch.get("vid_urls"), (launch.get("mission") or {}).get("vid_urls")):
        for vu in group or []:
            url = vu.get("url") if isinstance(vu, dict) else vu
            if url and extract_youtube_id(url):
                return url
    return None


def ll2_outcome(status_name: str | None) -> str | None:
    s = (status_name or "").lower()
    if "success" in s:
        return "success"
    if "fail" in s:  # covers "Launch Failure" and "Partial Failure"
        return "failure"
    return None


def fetch_ll2_launches(cache_dir: Path, queries: tuple[str, ...] = LL2_PROVIDER_QUERIES,
                       max_pages: int = 20, rate: float = 4.0,
                       use_cache: bool = True) -> list[dict]:
    """Fetch LL2 2.3.0 launches, paginated per ISRO/NASA provider query.

    Provider filtering is done server-side (lsp__name__icontains) so the whole
    relevant history fits in a handful of pages — no need to paginate every
    launch ever attempted (which would blow past LL2's 15-calls/hour limit).
    """
    out = []
    for q in queries:
        url = f"{LL2_URL}?limit={LL2_LIMIT}&lsp__name__icontains={q}"
        page = 0
        while url and page < max_pages:
            data = _cached_get_json(url, f"ll2_{q}_{page}.json", cache_dir,
                                    use_cache=use_cache, rate=rate)
            out.extend(data.get("results") or [])
            url = data.get("next")
            page += 1
    return out


def build_ll2_entries(launches: list[dict]):
    """Filter raw LL2 results to ISRO/NASA launches; returns (entries, skipped)."""
    entries, skipped = [], {"provider_filtered": 0, "no_youtube": 0, "unknown_outcome": 0}
    for ll in launches:
        provider = (ll.get("launch_service_provider") or {}).get("name") or ""
        if not _RELEVANT_RE.search(provider):
            skipped["provider_filtered"] += 1
            continue
        vid = extract_youtube_id(ll2_video_url(ll))
        outcome = ll2_outcome((ll.get("status") or {}).get("name"))
        if not vid:
            skipped["no_youtube"] += 1
            continue
        if not outcome:
            skipped["unknown_outcome"] += 1
            continue
        source = "isro" if _ISRO_RE.search(provider) else "nasa"
        entries.append({
            "mission": (ll.get("name") or (ll.get("mission") or {}).get("name")
                        or str(ll.get("id"))),
            "source": source,
            "video_id": vid,
            "outcome": outcome,
            "date": (ll.get("net") or ll.get("window_start") or "")[:10],
        })
    return entries, skipped


def dedupe(entries: list[dict]) -> list[dict]:
    """Drop entries sharing a video_id (first occurrence wins)."""
    seen, out = set(), []
    for e in entries:
        if e["video_id"] in seen:
            continue
        seen.add(e["video_id"])
        out.append(e)
    return out


def write_config(path: str | Path, entries: list[dict]) -> None:
    groups = {"normal": [], "anomaly": []}
    for e in entries:
        groups["anomaly" if e["outcome"] == "failure" else "normal"].append(e)
    docs = {}
    for label in ("normal", "anomaly"):
        rows = []
        for e in groups[label]:
            row = {"mission": e["mission"], "source": e["source"], "video_id": e["video_id"]}
            for key in ("date", "outcome"):
                if e.get(key):
                    row[key] = e[key]
            rows.append(row)
        docs[label] = rows
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(docs, f, sort_keys=False, default_flow_style=False)


MANUAL_TEMPLATE = """\
# Manually verified historical cases the launch APIs don't cover well (no
# webcast link in the API, removed YouTube videos, or footage only on
# unofficial channels). Same format as videos.yaml — confirm each video_id
# yourself before filling it in. download.py picks these up alongside the
# generated config once you merge them into configs/videos.yaml.
normal: []
anomaly:
  # Space Shuttle Challenger (STS-51-L), 1986 — lost ~T+73s
  # - mission: sts-51-l-challenger
  #   source: nasa
  #   video_id: YOUR_VERIFIED_ID
  # Antares Orb-3, 2014 — pad abort ~T+6s
  # - mission: antares-orb-3
  #   source: nasa
  #   video_id: YOUR_VERIFIED_ID
  # ISRO GSLV MkII F02, 2010 — launch failure
  # - mission: gslv-mk2-f02
  #   source: isro
  #   video_id: YOUR_VERIFIED_ID
"""


def write_manual_template(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        path.write_text(MANUAL_TEMPLATE)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output config path (default: %(default)s). "
                         "Use --out configs/videos.yaml to merge for real.")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                    help="raw API response cache (default: %(default)s)")
    ap.add_argument("--pages", type=int, default=20,
                    help="max LL2 pages per provider query, 100 launches each "
                         "(default: %(default)s)")
    ap.add_argument("--rate", type=float, default=4.0,
                    help="seconds between uncached LL2 requests "
                         "(default: %(default)s)")
    ap.add_argument("--spacex-url", default=SPACEX_URL,
                    help="SpaceX API URL override (default: %(default)s)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached API responses")
    args = ap.parse_args(argv)

    cache_dir = Path(args.cache_dir)
    use_cache = not args.no_cache

    try:
        spacex, spx_skip = fetch_spacex_launches(cache_dir, use_cache=use_cache,
                                                 url=args.spacex_url)
        print(f"SpaceX: {len(spacex)} launches with a YouTube webcast "
              f"({spx_skip['no_webcast']} without one, "
              f"{spx_skip['upcoming_or_unknown']} upcoming/unknown skipped).",
              file=sys.stderr)
    except Exception as exc:
        print(f"SpaceX API failed ({exc}); continuing with Launch Library 2 only.",
              file=sys.stderr)
        spacex, spx_skip = [], {"no_webcast": 0, "upcoming_or_unknown": 0}

    print("Fetching Launch Library 2 previous launches (ISRO/NASA)...",
          file=sys.stderr)
    ll2_raw = fetch_ll2_launches(cache_dir, max_pages=args.pages,
                                 rate=args.rate, use_cache=use_cache)
    ll2, ll2_skip = build_ll2_entries(ll2_raw)
    print(f"LL2: {len(ll2)} ISRO/NASA launches with a YouTube webcast "
          f"(checked {len(ll2_raw)} launches; "
          f"{ll2_skip['no_youtube']} relevant without YouTube video, "
          f"{ll2_skip['unknown_outcome']} unknown outcome skipped).",
          file=sys.stderr)

    before = len(spacex) + len(ll2)
    entries = dedupe(spacex + ll2)
    n_normal = sum(1 for e in entries if e["outcome"] == "success")
    n_anomaly = len(entries) - n_normal
    sources: dict[str, int] = {}
    for e in entries:
        sources[e["source"]] = sources.get(e["source"], 0) + 1

    write_config(args.out, entries)
    write_manual_template(DEFAULT_MANUAL)

    print(f"\nWrote {args.out}")
    print(f"  {len(entries)} entries = {n_normal} normal + {n_anomaly} anomaly "
          f"({before - len(entries)} duplicate webcasts dropped)")
    print("  by source: " + ", ".join(f"{k} {v}" for k, v in sorted(sources.items())))
    for label, cond in (("normal", lambda e: e["outcome"] == "success"),
                        ("anomaly", lambda e: e["outcome"] == "failure")):
        examples = [e for e in entries if cond(e)][:3]
        print(f"  {label} examples:")
        for e in examples:
            print(f"    - {e['source']} {e['mission']} ({e['date']}) {e['video_id']}")
    print("\nconfigs/videos.yaml untouched. Review the generated file, then merge:")
    print(f"  python3 scripts/fetch_launch_list.py --out configs/videos.yaml")
    print(f"Manual template (if needed): {DEFAULT_MANUAL}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
