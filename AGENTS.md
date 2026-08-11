# Project: rocket launch anomaly detection

Unsupervised anomaly detection on rocket ascent footage. The core idea: train
only on NORMAL launch footage to learn what a typical ascent looks like, then
score new footage by how much it deviates from that learned normal. No labeled
failure dataset exists — do not design anything that assumes balanced classes
or requires failure examples to train.

## Two parallel anomaly streams

The project has two parallel, independent anomaly streams, built to the same
unsupervised normal-only template. They are **correlated later, not merged or
replaced**: both reduce each source to a fixed-length per-0.5s feature vector
(timestamp-aligned to `data/frames/{video}/manifest.json`), so the two streams
can be compared frame-index-to-frame-index in a downstream fusion stage.

1. **Video stream** (built): shot-filtered frames → plume ROI tracking →
   per-frame CV features → GRU autoencoder trained on normal-only video →
   per-frame anomaly score + overlay.
2. **Audio stream** (in progress): ffmpeg mono 16 kHz track → mel-spectrogram →
   per-0.5s-window stats (mean/std of mel-band energies + spectral flatness /
   centroid) → its own normal-only temporal model + scoring later.

Video pipeline stages:
1. `scripts/download.py` — yt-dlp based downloader, organizes by
   source/mission into `data/raw/`
2. `scripts/extract_frames.py` — shot/cut detection + 2 fps frame sampling
   into `data/frames/`, tagged by lighting (day/dusk/night)
3. `scripts/roi_track.py` — detects and crops the rocket + exhaust plume
   region per frame
4. `scripts/features.py` — per-frame feature vector (plume shape stats,
   color histogram, edge/debris density, optical flow magnitude). Classical CV
   first, CNN embeddings only if classical features prove insufficient.
5. `notebooks/train_temporal_model.ipynb` — GRU temporal autoencoder over
   cached feature sequences, trained on normal-only data
6. `scripts/score_video.py` + `scripts/score_and_overlay.py` — run the trained
   model over a full video, output anomaly score per frame + rendered overlay

Audio pipeline stages:
1. `scripts/extract_audio_features.py` — ffmpeg audio decode + mel-spectrogram,
   reduced to per-0.5s windowed feature vectors in `data/audio_features/`,
   timestamp-aligned to the video frame manifest
2. (next) audio temporal model + scoring, mirroring the video stages 5–6

Conventions:
- Raw PyTorch, no Ultralytics/high-level wrappers.
- GPU-heavy video stages also run as Colab notebook cells with Google Drive
  I/O — write them so they work unchanged in Colab (no local filesystem
  assumptions beyond a configurable base path). Audio extraction is
  lightweight and runs locally.
- Cache every expensive intermediate result (frames, features) to disk —
  never recompute from raw video more than once.
- One function per pipeline stage, testable independently on a single video
  before running batch.
- Feature vectors are cheap (KB scale) — training the temporal model never
  needs GPU even if extraction did.
- Parallel streams must stay aligned: audio windows are anchored to the exact
  frame timestamps in `data/frames/{video}/manifest.json`, one audio row per
  video frame row.
