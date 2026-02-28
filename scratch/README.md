# Scratch — YOLO + DeepSORT / BoT-SORT Tracking

Quick-start tracking experiments on video files.  
All scripts are run from the **repo root** (`/home/jetson2/gaurav/`).

---

## Files

| File | Purpose |
|---|---|
| `scratch/track.py` | **DeepSORT** tracker — main tracking script |
| `scratch/track_botsort.py` | **BoT-SORT** tracker with built-in Ultralytics ReID |
| `scratch/make_challenge.py` | Generate an adversarial test video from a source clip |

---

## 1. Run DeepSORT tracking

### Basic (default: challenge video, mobilenet ReID, saves to `scratch/out_tracked.mp4`)
```bash
python3 scratch/track.py
```

### On a custom video
```bash
python3 scratch/track.py --source video_test/13657722_640x640.mp4
```

### Save annotated output to a specific path
```bash
python3 scratch/track.py \
  --source video_test/13657722_640x640.mp4 \
  --save scratch/my_output.mp4
```

### Disable ReID (IoU-only, faster)
```bash
python3 scratch/track.py --embedder none
```

### Full options
```
--weights   Path to detection model (.pt or TRT .engine)   [yolov8n.pt]
--source    Input video file                                [scratch/challenge.mp4]
--imgsz     Inference image size                           [640]
--conf      Detection confidence threshold                 [0.25]
--iou       NMS IoU threshold                              [0.45]
--embedder  ReID embedder: mobilenet | torchreid | none    [mobilenet]
--max-age   Frames to keep a lost track alive              [150]
--n-init    Detections needed to confirm a new track       [5]
--save      Output annotated video path                    [scratch/out_tracked.mp4]
--show      Display a live window while processing
```

---

## 2. Run BoT-SORT tracking

### With a specific ReID model
```bash
python3 scratch/track_botsort.py \
  --source scratch/challenge.mp4 \
  --reid models/reid/yolov8n-cls.pt
```

### Without ReID (IoU-only)
```bash
python3 scratch/track_botsort.py --reid none
```

### Benchmark ALL available ReID models (auto-saves one video per model)
```bash
python3 scratch/track_botsort.py --reid all
```

### Full options
```
--weights   Detection model path                           [yolov8n.pt]
--source    Input video file                               [scratch/challenge.mp4]
--reid      ReID model path | none | all                   [models/reid/yolov8n-cls.pt]
--imgsz     Inference image size                           [640]
--conf      Detection confidence threshold                 [0.25]
--save      Output annotated video path                    [auto-named by reid model]
```

---

## 3. Generate a challenge video

Creates an adversarially augmented version of any source clip with:
- **Occlusion bar** — opaque rectangle sweeps across the frame (frames 80–180, ~2 s)
- **Blink dropout** — random blackouts over the person region (65 frames spread throughout)
- **Motion blur** — heavy horizontal blur ramped in/out (frames 200–300, ~2 s)

### Default (uses `video_test/13657722_640x640.mp4`, saves to `scratch/challenge.mp4`)
```bash
python3 scratch/make_challenge.py
```

### Custom input/output
```bash
python3 scratch/make_challenge.py \
  --input  video_test/my_video.mp4 \
  --output scratch/my_challenge.mp4
```

---

## 4. Typical workflow

```bash
# Step 1 — build the challenge video
python3 scratch/make_challenge.py

# Step 2 — run DeepSORT on it and save result
python3 scratch/track.py \
  --source scratch/challenge.mp4 \
  --save   scratch/challenge_tracked.mp4

# Step 3 — compare with BoT-SORT
python3 scratch/track_botsort.py \
  --source scratch/challenge.mp4 \
  --reid   models/reid/yolov8n-cls.pt \
  --save   scratch/challenge_botsort.mp4
```

---

## 5. Key tracker parameters explained

### Why the ID changes after a long occlusion
Both trackers delete a lost track after `max_age` / `track_buffer` frames.  
If the occlusion is longer than that window, the track dies and a new ID is assigned —  
**ReID cannot re-link to a track that no longer exists.**

Rule of thumb:
```
max_age = fps × max_expected_occlusion_seconds
# e.g. 50 fps × 3 s = 150  (current default)
```

### Parameter cheat-sheet

| Parameter | Script | Default | Effect |
|---|---|---|---|
| `--max-age` | `track.py` | `150` | Frames to keep lost track alive (DeepSORT) |
| `--n-init` | `track.py` | `5` | Hits to confirm a new track (higher → fewer false tracks) |
| `max_cosine_distance` | `track.py` (code) | `0.4` | ReID similarity gate — lower = stricter re-link |
| `track_buffer` | `track_botsort.py` (code) | `150` | Same as max_age, for BoT-SORT |
| `appearance_thresh` | `track_botsort.py` (code) | `0.5` | BoT-SORT ReID gate (cosine distance) |
| `proximity_thresh` | `track_botsort.py` (code) | `0.3` | BoT-SORT spatial IoU gate |

---

## 6. Available ReID models

Located in `models/reid/`:

| Model | Speed (on challenge video) |
|---|---|
| `yolov8n-cls.pt` | **~18 fps** (fastest) |
| `yolo26n-cls.pt` | ~17 fps |
| `yolo26s-cls.pt` | ~17 fps |
| `yolo11s-cls.pt` | ~17 fps |
| `yolo11n-cls.pt` | ~17 fps |
| `yolo26m-cls.pt` | ~16 fps |

`yolov8n-cls.pt` is the recommended default — fastest with identical accuracy on this dataset.
