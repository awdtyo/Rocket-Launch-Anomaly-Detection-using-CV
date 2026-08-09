# Rocket Launch Anomaly Detection

Unsupervised video anomaly detection on rocket ascent footage.

**Core idea:** train only on *normal* launch footage so a temporal model learns what a
typical ascent looks like (plume shape, color, debris pattern, vehicle silhouette), then
score new footage by how much it deviates from that learned normal. There is no labeled
failure dataset, so nothing assumes balanced classes or failure examples for training.

## Pipeline

```
configs/videos.yaml ──► scripts/download.py ──► data/raw/{normal,anomaly}/
    │                                                │
    └─ scripts/add_video.py (hand-picked entries)    ▼
                                          scripts/extract_frames.py
                                                    │
                                                    ▼
                          data/frames/{video}/frame_*.jpg + manifest.json
                                                    │
                                          scripts/roi_track.py
                                                    │
                                                    ▼
                             data/frames/{video}/roi/ + roi_boxes.json
                                                    │
                                          scripts/features.py        (pending)
                                                    │
                                                    ▼
                                  data/features/*.npz feature vectors
                                                    │
                                                    ▼
                          notebooks/train_temporal_model.ipynb        (pending)
                              (ConvLSTM / temporal autoencoder, normal-only)
                                                    │
                                                    ▼
                                   scripts/score_and_overlay.py       (pending)
                                  (anomaly score per frame + overlay video)
```

| Stage | Script | Status |
|-------|--------|--------|
| 1. Download | `scripts/download.py` + `scripts/fetch_launch_list.py` + `scripts/add_video.py` | Done |
| 2. Frame extraction | `scripts/extract_frames.py` | Done |
| 3. ROI tracking | `scripts/roi_track.py` | Done |
| 4. Features | `scripts/features.py` | Pending |
| 5. Temporal model | `notebooks/train_temporal_model.ipynb` | Pending |
| 6. Scoring/overlay | `scripts/score_and_overlay.py` | Pending |

## Current dataset

- **13 normal videos** (`data/raw/normal/`, all SpaceX webcasts, Feb–Sep 2023), listed in `configs/videos.yaml`.
- **0 anomaly videos** (`data/raw/anomaly/` is empty) — anomaly footage is for inference/eval only and not needed to train.
- **29,597 frames** extracted at 2 fps across 13 videos (`data/frames/{video}/`).
- Lighting split (needed because baselines are trained per lighting condition):
  1× day, 4× dusk, 8× night. Currently only one day-launch video exists.

## Completed stages (technical details)

### 1. Video list + download

- **`configs/videos.yaml`** — YAML grouped by `normal:` / `anomaly:`; each entry is
  `{mission, source, video_id}`. `scripts/download.py` expects this exact schema.
- **`scripts/add_video.py`** — CLI helper for hand-picked URLs (replaces the
  API-driven approach):
  ```
  python scripts/add_video.py <youtube_url> --mission "Name" --source spacex --tag normal
  ```
  Extracts the 11-char video ID from any standard YouTube URL (`watch?v=`, `youtu.be/`,
  `embed/`, `live/`, `shorts/`), rejects non-YouTube URLs and duplicate IDs, appends under
  the right group in `configs/videos.yaml`, and preserves the file's existing comments by
  editing text line-by-line rather than round-tripping YAML.
- **`scripts/fetch_launch_list.py`** — auto-generates `configs/videos_generated.yaml` from
  the SpaceX v5 API and Launch Library 2 (LL2 2.3.0, provider-side filters for
  ISRO/NASA); caches raw responses in `data/api_cache/`. Superseded by `add_video.py` for
  day-to-day use but kept.
- **`scripts/download.py`** — reads `configs/videos.yaml`, downloads each video at ≤1080p
  into `data/raw/{label}/{source}_{mission}.mp4`, skips existing files, merges streams with
  ffmpeg when available, prints a summary table.

### 2. Frame extraction (`scripts/extract_frames.py`)

Per video:
1. **Shot detection** with PySceneDetect `ContentDetector` (threshold 27.0) over the full
   video.
2. **Short-shot discard** — shots < 2 s are dropped (countdown graphics / rapid cuts, not
   sustained ascent footage).
3. **Sampling** — frames written at 2 fps inside kept shots (sampling cadence derived from
   fps, e.g. every 15th frame at 29.97 fps) to `data/frames/{video}/frame_{idx:05d}.jpg`.
4. **Lighting tag** — mean gray brightness over ~24 frames sampled across the whole video;
   `< 45` → `night`, `> 100` → `day`, else `dusk`. Stored in the manifest so training can
   split by lighting condition.
5. **`manifest.json`** — records per-frame `{index, timestamp, shot_index, file}` plus
   video metadata, shot list (kept/discarded flags), and the lighting tag.

Extraction happens in a single decode pass (shot boundaries come from scene detection, then
one read loop writes frames and gathers brightness). Already-processed videos (valid
manifest with frames) are skipped; `--force` redoes, `--only <substring>` tests one video.

**Codec handling:** SpaceX webcasts arrive as AV1/VP9, which OpenCV cannot decode. A
one-time ffmpeg transcode to H.264 (`libx264`, `-crf 23`, `-pix_fmt yuv420p`, no audio) is
cached in `data/frames/_decoded/` and reused; only the raw path is recorded in the
manifest. Requires `ffmpeg` on PATH.

### 3. ROI tracking (`scripts/roi_track.py`)

Detects the rocket + exhaust plume region per frame and crops it with a 20% margin.

- **Detection** is brightness-primary: Otsu threshold isolates the plume against dark sky;
  when a scene is uniformly bright (day launches) the threshold is raised to the
  93rd-percentile of gray so the sun/white sky don't swallow the plume. A temporal motion
  mask (absdiff from the previous frame) *boosts* the blob score rather than gating on it —
  frames are sampled 0.5 s apart and camera pan dominates the raw diff.
- **Temporal smoothing** — an EMA over the box (α = 0.4) keeps the crop from jittering
  frame to frame.
- **Fallback** — frames with no confident blob (area fraction < 0.005 of the frame) get a
  full-frame crop; those frames are logged and recorded in
  `roi_boxes.json["fallback_frames"]` for spot-checking (4.8% of frames overall, higher on
  night launches where the plume is tiny).
- **Outputs** — `data/frames/{video}/roi/frame_*.jpg` crops + `roi_boxes.json` with
  per-frame `{x, y, w, h}` boxes, confidences, fallback list, config, and the lighting tag
  pulled from the manifest.

Works as CLI (`python scripts/roi_track.py`) or importable `process_video()` for use from a
Colab notebook. Skipped on rerun when `roi_boxes.json` exists; `--force` redoes.

## Pending stages (technical notes)

### 4. `scripts/features.py` — per-frame feature vector

Classical CV first; CNN embeddings only if classical features prove insufficient. Per-frame
features on the ROI crop (roughly KB-scale per frame):

- Plume shape statistics (blob area fraction, aspect ratio, centroid offset from crop
  center, compactness).
- Color histogram (HSV) — relevant because plume color varies by lighting condition
  (orange fire at night vs saturated-white at day).
- Edge / debris density (Canny or gradient magnitude normalized by crop area).
- Optical flow magnitude within the crop (frame-to-frame motion intensity).

Follow the same CLI + importable-function pattern, cache to `data/features/{video}/`, and
use the `manifest.json` lighting tag + `roi_boxes.json` boxes as inputs.

### 5. `notebooks/train_temporal_model.ipynb` — the model

ConvLSTM or temporal autoencoder over cached feature sequences, trained **only on normal
footage**. Feature vectors are cheap, so training never needs GPU even though extraction
did. Train separate baselines per lighting condition (day / dusk / night) — hence the
lighting tag collected in stage 2. Also run as Colab cells with Google Drive I/O (no local
filesystem assumptions beyond a configurable base path).

### 6. `scripts/score_and_overlay.py` — inference

Runs the trained model over a full video, outputs an anomaly score per frame + a rendered
overlay video. No failure examples are required to run it.

## Conventions

- Raw PyTorch; no Ultralytics / high-level wrappers.
- Every GPU-heavy script must also run unchanged as Colab notebook cells with Google Drive
  I/O (configurable base path, no hardcoded local paths).
- Cache every expensive intermediate (frames, features) to disk — never recompute from raw
  video more than once.
- One function per pipeline stage, testable independently on a single video (`--only`)
  before running batch.
- Feature vectors are KB-scale, so the temporal model never needs GPU.

## Requirements

`requirements.txt`: `yt-dlp`, `pyyaml`, `requests`, `scenedetect`, `opencv-python`,
`numpy`. `ffmpeg` is required on PATH for download merging and AV1/VP9 transcoding.

## Known gaps / next steps

1. Only one day-launch video — the day-lighting baseline is undersampled.
2. `data/raw/anomaly/` is empty; anomaly footage is needed only for eval, not training.
3. ROI tracking uses a classical brightness/motion heuristic; verify quality on more
   footage before building features on top (the `fallback_frames` list is the review tool).
4. Batch feature extraction and the training notebook are the next pipeline stages to build.
