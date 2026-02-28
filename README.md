# Tracking

Experiments in object detection and tracking from nadir (top-down) drone footage.  
Three tracker families are implemented: YOLO + DeepSORT (multi-object), SiamRPN with YOLO drift correction (single-object), and a segmentation-anchored dual-template SiamRPN with Kalman-based recovery (single-object, hard re-ID).

---

## Quick-start

```bash
# 1. clone
git clone https://github.com/<you>/tracking.git
cd tracking

# 2. create the virtualenv (Python 3.10+ recommended)
python3 -m venv tracking
source tracking/bin/activate
pip install -r requirements.txt   # or install manually, see below

# 3. download all test videos + model weights
python setup_assets.py

# or selectively:
python setup_assets.py --videos      # test videos only
python setup_assets.py --weights     # model weights only
python setup_assets.py --dry-run     # preview what would be downloaded
```

### Manual requirements

```
pip install ultralytics deep-sort-realtime opencv-python \
            yt-dlp gdown pyyaml requests tqdm
```
For pysot: no pip install needed—it is included as a sub-directory under `siamese/pysot/` and injected via `sys.path`.

---

## Repository layout

```
tracking/
├── scratch/                   # multi-object trackers
│   ├── track.py               # YOLO + DeepSORT, full flag set
│   ├── detect.py              # detection-only viewer (auto OBB / bbox)
│   └── click_to_track.py      # click to lock DeepSORT to one object
│
├── siamese/                   # single-object SiamRPN trackers
│   ├── siam_track.py                # basic pysot SiamRPN wrapper
│   ├── siam_track_drift_correct.py  # hybrid: SiamRPN + YOLO drift correction
│   ├── siam_dual_template.py        # dual template (permanent anchor + YOLO-corrected)
│   ├── siam_seg_anchor.py           # seg-anchor + Kalman + perimeter recovery  ← main
│   └── pysot/                       # STVIR/pysot (git clone, not installed)
│
├── models/                    # custom YOLO weights (not in git, see below)
│   └── reid/                  # re-ID classification models
│
├── video_test/                # test clips (not in git – downloaded by setup_assets.py)
│
├── assets_manifest.yaml       # URLs / GDrive IDs for all large files
├── setup_assets.py            # download script
└── .gitignore
```

---

## Scripts

### `scratch/track.py` — YOLO + DeepSORT multi-object tracker

```
python scratch/track.py --source video_test/nadir_crossroads.mp4 \
        --model models/yolov26nobbnew_merged_1024.pt \
        --classes -1 --show
```

Key flags: `--classes` (−1 = all), `--conf`, `--iou`, `--nms-max-overlap`,
`--det-color`, `--track-color`, `--det-thickness`, `--track-thickness`,
`--hide-det`, `--hide-track`, `--hide-conf`, `--hide-id`, `--hide-class`

---

### `scratch/detect.py` — detection-only viewer

```
python scratch/detect.py --source video_test/nadir_crossroads.mp4 \
        --model models/yolov26nobbnew_merged_1024.pt
```

Automatically selects OBB or bbox head based on model type.

---

### `scratch/click_to_track.py` — click-to-lock DeepSORT

Launch, click on an object in the first frame; only that track is highlighted.

```
python scratch/click_to_track.py --source video_test/nadir_pedestrians_cars.mp4 \
        --model models/yolov26nobbnew_merged_1024.pt
```

---

### `siamese/siam_dual_template.py` — dual-template SiamRPN

Maintains two templates simultaneously:
- `zf_anchor` — frozen from user-click on frame 0
- `zf_yolo` — periodically refreshed by the highest-confidence YOLO detection

Per frame, both templates score independently; the higher-scoring one drives the bounding-box output.

```bash
# file
python siamese/siam_dual_template.py \
    --source  video_test/nadir_pedestrians_cars.mp4 \
    --config  siamese/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --weights siamese/pysot/experiments/siamrpn_r50_l234_dwxcorr/model/model.pth \
    --yolo    models/yolov26nobbnew_merged_1024.pt \
    --show

# webcam
python siamese/siam_dual_template.py --source 0 --yolo yolov8n.pt --show
```

| arg | default | description |
|---|---|---|
| `--corr-interval` | 10 | YOLO correction every N frames |
| `--corr-conf` | 0.60 | min YOLO confidence to trigger correction |
| `--corr-iou` | 0.35 | IoU threshold for matching detection to current box |
| `--score-thr` | 0.20 | min SiamRPN score to accept a template |

---

### `siamese/siam_seg_anchor.py` — segmentation-anchored tracker (main)

Uses a heavy segmentation model **on frame 0 only** to:
1. Let you click a person / vehicle
2. Extract a clean, background-masked anchor template (`zf_anchor`)

Subsequent frames use a lightweight YOLO detector + SiamRPN dual-template.  
A pure-numpy Kalman filter predicts position when tracking becomes unreliable.  
When drift is detected for several consecutive frames, a perimeter search recovers
the target by scoring YOLO candidates against `zf_anchor`.

```bash
python siamese/siam_seg_anchor.py \
    --source  video_test/nadir_pedestrians_cars.mp4 \
    --config  siamese/pysot/experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --weights siamese/pysot/experiments/siamrpn_r50_l234_dwxcorr/model/model.pth \
    --seg     yolov8l-seg.pt \
    --yolo    models/yolov26nobbnew_merged_1024.pt \
    --show
```

| arg | default | description |
|---|---|---|
| `--seg` | `yolov8l-seg.pt` | seg model used **only on frame 0** |
| `--yolo` | custom pt | lightweight per-frame detector |
| `--corr-interval` | 10 | YOLO correction cadence |
| `--anchor-warn-ratio` | 0.5 | similarity ratio below which drift banner shows |
| `--drift-patience` | 5 | frames below `warn-ratio` before perimeter search |
| `--search-radius` | 150 | px radius around Kalman prediction to search |
| `--recover-thr` | 0.65 | min anchor similarity to accept a recovery candidate |
| `--show-mask` | flag | overlay first-frame segmentation mask |
| `--show-kalman` | flag | draw Kalman-predicted box |

---

## Models

| file | source | status |
|---|---|---|
| `siamrpn_r50_l234_dwxcorr/model/model.pth` | Google Drive | downloaded by `setup_assets.py` |
| `siamrpn_alex_dwxcorr/model/model.pth` | Google Drive | downloaded by `setup_assets.py` |
| `yolov8n.pt` | ultralytics | auto-downloaded |
| `yolov8l-seg.pt` | ultralytics | auto-downloaded |
| `models/yolov26nobbnew_merged_1024.pt` | custom-trained | **obtain separately** |

Custom-trained weights are not redistributable. Contact the owner or train your own
using the YOLO training scripts.

---

## Notes

- Tested on **macOS (Apple Silicon, MPS)** with Python 3.14 and PyTorch 2.10.
- `pysot` is not installed via pip; it lives at `siamese/pysot/` and is injected
  via `sys.path` in each siamese script.
- The `tracking/` virtualenv directory is excluded from git (large, platform-specific).
  Recreate it with the quick-start steps above.
