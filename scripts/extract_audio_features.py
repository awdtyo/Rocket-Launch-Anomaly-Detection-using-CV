#!/usr/bin/env python3
"""Compute per-0.5s-window audio feature vectors from raw launch webcasts.

For every video in data/raw/{normal,anomaly}/ this decodes the audio track
(ffmpeg -> mono 16 kHz PCM), computes a log-mel spectrogram, and reduces each
0.5 s window to a fixed-length feature vector: mean and std of the mel-band
energies in the window, plus window-mean spectral flatness and spectral
centroid (a rough proxy for "engine roar" vs "sudden transient" — the kind of
shift an overpressure or structural event might cause).

Windows are anchored to the exact frame timestamps in
data/frames/{video}/manifest.json, one audio row per video frame row, so the
audio stream is frame-index aligned with the video stream for later
correlation. Videos without a manifest fall back to uniform 0.5 s windows.

Outputs (mirroring the video feature files):
  data/audio_features/{video}.npy                float32, rows = windows
  data/audio_features/{video}.frame_index.json   row -> window timestamp

Processed videos (existing .npy) are skipped on rerun; --force redoes, --only
runs a single video. Audio processing is lightweight and runs locally (no
GPU, no Colab). All outputs are cached to disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_FRAMES = ROOT / "data" / "frames"
DEFAULT_OUTPUT = ROOT / "data" / "audio_features"

SAMPLE_RATE = 16000  # mono 16 kHz
WINDOW_SECONDS = 0.5  # matches the video pipeline's 2 fps sampling
N_MELS = 32
N_FFT = 2048
HOP_LENGTH = 512
_BLOCK = 4096  # STFT frames computed per block, to bound peak memory

FEATURE_NAMES = (
    [f"mel_mean_b{i}" for i in range(N_MELS)]
    + [f"mel_std_b{i}" for i in range(N_MELS)]
    + ["spectral_flatness", "spectral_centroid_hz"]
)
N_FEATURES = len(FEATURE_NAMES)


def iter_raw_videos(raw_base: Path):
    """Yield mp4s under raw_base (group subdirs like normal/ and anomaly/)."""
    for group in sorted(raw_base.iterdir()):
        if not group.is_dir():
            continue
        for mp4 in sorted(group.glob("*.mp4")):
            yield mp4


def load_manifest(frames_base: Path, name: str) -> dict | None:
    """Return the video frame manifest, or None if missing/unreadable."""
    path = frames_base / name / "manifest.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def decode_audio(video_path: Path, sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Decode the first audio track to mono float32 PCM in [-1, 1].

    Returns None if ffmpeg fails (e.g. no audio stream in the container).
    """
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error",
           "-i", str(video_path), "-map", "0:a:0",
           "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH — required for audio decode")
    if proc.returncode != 0 or not proc.stdout:
        return None
    raw = np.frombuffer(proc.stdout, dtype=np.int16)
    if raw.size == 0:
        return None
    return raw.astype(np.float32) / 32768.0


def mel_band_features(y: np.ndarray, sr: int, n_mels: int = N_MELS,
                      n_fft: int = N_FFT, hop: int = HOP_LENGTH):
    """Full-signal log-mel spectrogram, spectral flatness, and centroid.

    STFT frames are computed in blocks to bound peak memory. Returns
    (mel_db [n_mels, T], flatness [T], centroid_hz [T]) where T is the number
    of STFT frames (time axis at hop/sr seconds per frame).
    """
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (n_mels, 1+n_fft/2)
    cents = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2)
    win = np.hanning(n_fft)

    n_frames = max(0, (len(y) - n_fft) // hop + 1)
    mel_pows = []
    for s in range(0, n_frames, _BLOCK):
        e = min(n_frames, s + _BLOCK)
        idx = np.arange(s, e)[:, None] * hop + np.arange(n_fft)[None, :]
        X = np.fft.rfft(y[idx] * win, axis=1)
        mel_pows.append(fb @ (np.abs(X) ** 2).T)  # (n_mels, block)
    if not mel_pows:
        return (np.zeros((n_mels, 0), np.float32),
                np.zeros(0, np.float32), np.zeros(0, np.float32))
    mel_pow = np.concatenate(mel_pows, axis=1)  # (n_mels, T)

    mel_db = librosa.power_to_db(mel_pow, ref=mel_pow.max(), top_db=None)
    eps = 1e-10
    flat = (np.exp(np.mean(np.log(mel_pow + eps), axis=0))
            / np.maximum(np.mean(mel_pow, axis=0), eps))
    cent = (cents @ mel_pow) / np.maximum(mel_pow.sum(axis=0), eps)
    return mel_db, flat, cent


def window_slice(t: float, sr: int, hop: int, window_seconds: float) -> tuple[int, int]:
    """Half-open range of STFT frames whose centers fall in [t, t + window)."""
    i0 = int(np.ceil(t * sr / hop))
    i1 = int(np.ceil((t + window_seconds) * sr / hop))
    return i0, i1


def process_video(raw_path: Path, frames_base: Path, output_base: Path,
                  *, sr: int = SAMPLE_RATE, n_mels: int = N_MELS,
                  window_seconds: float = WINDOW_SECONDS) -> dict:
    """Compute audio features for one video into data/audio_features/."""
    name = raw_path.stem

    manifest = load_manifest(frames_base, name)
    if manifest and manifest.get("frames"):
        timestamps = [fr["timestamp"] for fr in manifest["frames"]]
        aligned = True
        video_meta = {"name": name}
        if manifest.get("video"):
            video_meta.update({k: manifest["video"].get(k)
                               for k in ("source", "mission", "label")})
        video_meta.setdefault("lighting", manifest.get("lighting", "unknown"))
    else:
        timestamps, aligned, video_meta = None, False, {"name": name,
                                                        "lighting": "unknown"}

    y = decode_audio(raw_path, sr=sr)
    if y is None:
        return {"video": name, "error": "no audio track"}
    duration = len(y) / sr

    mel_db, flat, cent = mel_band_features(y, sr, n_mels=n_mels)
    n_stft = mel_db.shape[1]
    if n_stft == 0:
        return {"video": name, "error": f"audio shorter than one FFT window ({duration:.3f}s)"}

    if timestamps is None:
        timestamps = np.arange(0.0, duration, window_seconds)
    timestamps = [float(t) for t in timestamps]

    rows, frame_index, prev_row = [], [], None
    for row, t in enumerate(timestamps):
        i0, i1 = window_slice(t, sr, HOP_LENGTH, window_seconds)
        i0, i1 = min(max(i0, 0), n_stft), min(max(i1, 0), n_stft)
        if i1 > i0:
            w = mel_db[:, i0:i1]
            vec = np.concatenate([
                w.mean(axis=1), w.std(axis=1),
                [float(np.mean(flat[i0:i1])), float(np.mean(cent[i0:i1]))],
            ]).astype(np.float32)
        elif prev_row is not None:
            vec = prev_row  # window past the end of audio: replicate last row
        else:
            vec = np.zeros(N_FEATURES, np.float32)
        prev_row = vec
        rows.append(vec)
        frame_index.append({"row": row, "timestamp": t,
                            "window_end": t + window_seconds})

    X = np.stack(rows).astype(np.float32)
    output_base.mkdir(parents=True, exist_ok=True)
    np.save(output_base / f"{name}.npy", X)

    frame_index_json = {
        "video": video_meta,
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "audio": {"sample_rate": sr, "window_seconds": window_seconds,
                  "n_mels": n_mels, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                  "aligned_to_manifest": aligned},
        "frames": frame_index,
    }
    with open(output_base / f"{name}.frame_index.json", "w") as f:
        json.dump(frame_index_json, f, indent=2)

    return {"video": name, "duration": duration, "windows": len(rows),
            "n_features": N_FEATURES, "aligned": aligned}


def batch(raw_base, frames_base, output_base, *, only: str | None = None,
          force: bool = False, **kw) -> int:
    """Run process_video over every raw video; skip ones already processed."""
    raw_base, frames_base, output_base = map(Path, (raw_base, frames_base, output_base))
    videos = [p for p in iter_raw_videos(raw_base)
              if not only or only in p.stem]
    if not videos:
        print("no videos found under " + str(raw_base)
              + (f" matching '{only}'" if only else ""), file=sys.stderr)
        return 1

    n_done, n_skipped = 0, 0
    for path in videos:
        name = path.stem
        out = output_base / f"{name}.npy"
        if out.exists() and not force:
            print(f"{name}: skipped (cached .npy)")
            n_skipped += 1
            continue
        summary = process_video(path, frames_base, output_base, **kw)
        if "error" in summary:
            print(f"{name}: skipped ({summary['error']})", file=sys.stderr)
            n_skipped += 1
            continue
        n_done += 1
        mode = "aligned to manifest" if summary["aligned"] else "uniform 0.5s windows"
        print(f"{summary['video']}: duration {summary['duration']:.2f}s, "
              f"{summary['windows']} windows, {summary['n_features']} dims "
              f"({mode})", flush=True)
    print(f"\n{n_done} videos processed, {n_skipped} skipped.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default=str(DEFAULT_RAW),
                    help="raw video base dir (default: %(default)s)")
    ap.add_argument("--frames", default=str(DEFAULT_FRAMES),
                    help="frames base dir holding manifest.json files "
                         "(default: %(default)s)")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="output audio features base dir (default: %(default)s)")
    ap.add_argument("--only", default=None,
                    help="only process videos whose name contains this substring")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if .npy exists")
    ap.add_argument("--sr", type=int, default=SAMPLE_RATE,
                    help="audio sample rate in Hz (default: %(default)s)")
    ap.add_argument("--n-mels", type=int, default=N_MELS,
                    help="mel bands per window (default: %(default)s)")
    args = ap.parse_args(argv)
    return batch(args.raw, args.frames, args.output, only=args.only,
                 force=args.force, sr=args.sr, n_mels=args.n_mels)


if __name__ == "__main__":
    sys.exit(main())
