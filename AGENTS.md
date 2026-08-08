# Project: rocket launch anomaly detection

Unsupervised video anomaly detection on rocket ascent footage. The core idea:
train only on NORMAL launch footage to learn what a typical ascent looks like
(plume shape, color, debris pattern, vehicle silhouette), then score new
footage by how much it deviates from that learned normal. No labeled failure
dataset exists — do not design anything that assumes balanced classes or
requires failure examples to train.

Pipeline stages (build in this order, one script/notebook per stage):
1. scripts/download.py — yt-dlp based downloader, organizes by
   source/mission into data/raw/
2. scripts/extract_frames.py — shot/cut detection + frame sampling into
   data/frames/, tagged by camera-angle cluster
3. scripts/roi_track.py — detects and crops the rocket + exhaust plume
   region per frame
4. scripts/features.py — per-frame feature vector (plume shape stats,
   color histogram, edge/debris density, optical flow magnitude).
   Classical CV first, CNN embeddings only if classical features prove
   insufficient.
5. notebooks/train_temporal_model.ipynb — ConvLSTM or temporal
   autoencoder over cached feature sequences, trained on normal-only data
6. scripts/score_and_overlay.py — runs the trained model over a full
   video, outputs anomaly score per frame + rendered overlay video

Conventions:
- Raw PyTorch, no Ultralytics/high-level wrappers.
- All GPU-heavy scripts must also run as Colab notebook cells with
  Google Drive I/O — write them so they work unchanged in Colab (no local
  filesystem assumptions beyond a configurable base path).
- Cache every expensive intermediate result (frames, features) to disk —
  never recompute from raw video more than once.
- One function per pipeline stage, testable independently on a single
  video before running batch.
- Feature vectors are cheap (KB scale) — training the temporal model
  never needs GPU even if extraction did.
