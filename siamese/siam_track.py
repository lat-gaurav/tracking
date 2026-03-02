#!/usr/bin/env python3
"""
SiamRPN single-object tracker using the pysot library.

On the FIRST FRAME a ROI selector opens:
  - Click and drag to draw a bounding box around the object to track
  - Press ENTER or SPACE to confirm, ESC to cancel

Controls during tracking
------------------------
  Q / ESC   quit
  SPACE     pause / resume
  R         re-select target (pauses, shows ROI selector again)
  S         save snapshot of current frame

Usage
-----
    python3 siam_track.py --video ../video_test/nadir_ped_crossing_crop640.mp4  \\
                          --config siamrpn_alex_dwxcorr                         \\
                          --show

    # save output video
    python3 siam_track.py --video ../video_test/nadir_city_rushhour.mp4 \\
                          --config siamrpn_r50_l234_dwxcorr              \\
                          --show --save out_siam.mp4

Downloading pretrained weights
-------------------------------
The model weights must be placed at:
    siamese/pysot/experiments/<config_name>/model/model.pth

Weights are available from the pysot MODEL_ZOO (download in browser):
  siamrpn_alex_dwxcorr  →  https://drive.google.com/open?id=1t62x56Jl7baUzPTo0QrC4jJnwvPZm-2m
  siamrpn_r50_l234_dwxcorr → https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG
  siammask_r50_l3       →  https://drive.google.com/open?id=1YbPUQVTYw_slAvk_DchvRY-7B6rnSXP9

  All models also on Baidu Yun: https://pan.baidu.com/s/1GB9-aTtjG57SebraVoBfuQ
  Extraction Code: j9yb
"""
import argparse
import sys
import time
from pathlib import Path

# ── make pysot importable without installing it ───────────────────────────────
_PYSOT_ROOT = Path(__file__).resolve().parent / "pysot"
if str(_PYSOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYSOT_ROOT))

import cv2
import numpy as np
import torch

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

# ── defaults ──────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent
_EXPERIMENTS    = _PYSOT_ROOT / "experiments"
_WEIGHTS_ROOT   = _HERE.parent / "resources" / "weights"
DEFAULT_CONFIG  = "siamrpn_alex_dwxcorr"
DEFAULT_VIDEO   = str(_HERE.parent / "resources" / "video_test" / "nadir_ped_crossing_crop640.mp4")
DEFAULT_DEVICE  = "cpu"   # pysot runs on cpu/cuda (MPS not fully supported)

TRACK_COLOR  = (0, 255,   0)   # green  – track box
TRAIL_COLOR  = (0, 200, 255)   # yellow – trail dots


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="pysot SiamRPN single-object tracker")
    p.add_argument("--video",   default=DEFAULT_VIDEO,
                   help="Input video file or camera index (0, 1, …)")
    p.add_argument("--config",  default=DEFAULT_CONFIG,
                   help="Experiment config name under siamese/pysot/experiments/")
    p.add_argument("--weights", default="",
                   help="Path to model .pth file. "
                        "Defaults to <experiments>/<config>/model/model.pth")
    p.add_argument("--save",    default="",
                   help="Output annotated video path (empty = no save)")
    p.add_argument("--show",    action="store_true",
                   help="Display live window")
    p.add_argument("--trail",   type=int, default=80,
                   help="Number of past centre-points to draw as trail (0 = off)")
    p.add_argument("--thickness", type=int, default=2,
                   help="Track box line thickness")
    return p.parse_args()


# ── model + tracker loader ────────────────────────────────────────────────────
def load_tracker(config_name: str, weights_path: str, device: str):
    cfg_file = _EXPERIMENTS / config_name / "config.yaml"
    if not cfg_file.exists():
        print(f"ERROR: config not found: {cfg_file}", file=sys.stderr)
        print(f"  Available configs: {sorted(p.name for p in _EXPERIMENTS.iterdir() if p.is_dir())}")
        sys.exit(1)

    if not weights_path:
        weights_path = str(_WEIGHTS_ROOT / config_name / "model" / "model.pth")
    if not Path(weights_path).exists():
        print(f"\nERROR: Model weights not found at:\n  {weights_path}", file=sys.stderr)
        print("""
  Download the pretrained weights (open link in browser):

  siamrpn_alex_dwxcorr (smallest, fastest):
    https://drive.google.com/open?id=1t62x56Jl7baUzPTo0QrC4jJnwvPZm-2m

  siamrpn_r50_l234_dwxcorr (more accurate):
    https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG

  Place the downloaded file at:
    siamese/pysot/experiments/<config>/model/model.pth

  All models also available on Baidu Yun:
    https://pan.baidu.com/s/1GB9-aTtjG57SebraVoBfuQ  (code: j9yb)
""", file=sys.stderr)
        sys.exit(1)

    print(f"  Config   : {cfg_file}")
    print(f"  Weights  : {weights_path}")

    cfg.merge_from_file(str(cfg_file))
    cfg.CUDA = (torch.cuda.is_available() and cfg.CUDA)

    model = ModelBuilder()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval().to(torch.device("cuda" if cfg.CUDA else "cpu"))

    tracker = build_tracker(model)
    print(f"  Tracker  : {cfg.TRACK.TYPE}  (CUDA={cfg.CUDA})")
    return tracker


# ── ROI selector helper ───────────────────────────────────────────────────────
def select_roi(frame: np.ndarray, win_name: str):
    """
    Open cv2.selectROI, return (x, y, w, h) or None if cancelled.
    Shows an overlay hint so the user knows what to do.
    """
    hint = frame.copy()
    cv2.putText(hint, "Draw bbox around target, then press ENTER / SPACE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hint, "Press ESC to quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.imshow(win_name, hint)
    try:
        rect = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
    except Exception:
        return None
    if rect[2] == 0 or rect[3] == 0:
        return None
    return rect   # (x, y, w, h)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── open video ────────────────────────────────────────────────────────────
    raw = args.video.strip()
    if raw.lstrip("-").isdigit():
        cap = cv2.VideoCapture(int(raw))
        n_frames = 0
        is_live = True
        src_label = f"webcam {raw}"
    else:
        cap = cv2.VideoCapture(str(Path(raw).resolve()))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_live  = False
        src_label = Path(raw).name

    if not cap.isOpened():
        print(f"ERROR: cannot open {raw}", file=sys.stderr)
        return 1

    fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── optional writer ───────────────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (W, H))

    # ── load pysot model ──────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Source : {src_label}  ({W}x{H} @ {fps_in:.1f} fps)")
    tracker = load_tracker(args.config, args.weights,
                           "cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)

    WIN = "SiamRPN Tracker"
    if args.show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # ── read first frame & get initial bbox ───────────────────────────────────
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: could not read first frame", file=sys.stderr)
        return 1

    if args.show:
        init_rect = select_roi(first_frame, WIN)
    else:
        # headless: can't select → abort
        print("ERROR: --show is required for interactive ROI selection", file=sys.stderr)
        return 1

    if init_rect is None:
        print("No target selected. Exiting.")
        return 0

    tracker.init(first_frame, init_rect)
    print(f"  Initialised: x={init_rect[0]} y={init_rect[1]} "
          f"w={init_rect[2]} h={init_rect[3]}")

    # trail
    from collections import deque
    trail: deque = deque(maxlen=max(1, args.trail))

    frame_idx = 1          # first frame already read
    t0 = time.time()
    paused = False
    snap_n = 0
    annotated = first_frame.copy()

    # draw init box on first frame
    ix, iy, iw, ih = [int(v) for v in init_rect]
    cv2.rectangle(annotated, (ix, iy), (ix+iw, iy+ih), TRACK_COLOR, args.thickness)
    cv2.putText(annotated, "INIT", (ix, max(14, iy-4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TRACK_COLOR, 2, cv2.LINE_AA)
    if writer:
        writer.write(annotated)
    if args.show:
        cv2.imshow(WIN, annotated)
        cv2.waitKey(1)

    # ── tracking loop ─────────────────────────────────────────────────────────
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = False
                print("  [resumed]")
            if key == ord("r"):
                # re-select
                new_rect = select_roi(frame, WIN)
                if new_rect and new_rect[2] > 0 and new_rect[3] > 0:
                    tracker.init(frame, new_rect)
                    trail.clear()
                    print(f"  Re-initialised at {new_rect}")
                paused = False
            if key == ord("s"):
                fname = f"snap_{snap_n:04d}.png"
                cv2.imwrite(fname, annotated)
                print(f"  Saved snapshot → {fname}")
                snap_n += 1
            continue

        # ── track ─────────────────────────────────────────────────────────────
        outputs = tracker.track(frame)
        annotated = frame.copy()

        if "polygon" in outputs:
            poly = np.array(outputs["polygon"]).astype(np.int32)
            cv2.polylines(annotated, [poly.reshape((-1, 1, 2))],
                          True, TRACK_COLOR, args.thickness)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            # optional mask overlay (SiamMask)
            if "mask" in outputs:
                mask = ((outputs["mask"] > cfg.TRACK.MASK_THERSHOLD) * 255).astype(np.uint8)
                mask_overlay = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis=-1)
                annotated = cv2.addWeighted(annotated, 0.80, mask_overlay, 0.20, 0)
        else:
            bbox = list(map(int, outputs["bbox"]))   # x, y, w, h
            x, y, bw, bh = bbox
            cv2.rectangle(annotated, (x, y), (x+bw, y+bh), TRACK_COLOR, args.thickness)
            cx = x + bw // 2
            cy = y + bh // 2

            # score bar (if available)
            score = outputs.get("best_score", None)
            if score is not None:
                score_label = f"score {score:.3f}"
            else:
                score_label = f"ID: tracked"
            (tw, th), _ = cv2.getTextSize(score_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x, max(0, y-th-6)), (x+tw+4, y), TRACK_COLOR, -1)
            cv2.putText(annotated, score_label, (x+2, max(th, y-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # ── trail ─────────────────────────────────────────────────────────────
        if args.trail > 0:
            trail.append((cx, cy))
            for i, (px, py) in enumerate(trail):
                r = max(2, i // 8)
                # cv2.circle(annotated, (px, py), r, TRAIL_COLOR, -1)

        # ── HUD ───────────────────────────────────────────────────────────────
        elapsed = max(time.time() - t0, 1e-9)
        fps_now = frame_idx / elapsed
        frame_str = f"frame {frame_idx}" if is_live else f"frame {frame_idx}/{n_frames}"
        cv2.putText(annotated, f"FPS {fps_now:.1f}  |  {frame_str}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, "R=reselect  SPACE=pause  S=snap  Q=quit",
                    (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        if writer:
            writer.write(annotated)

        if args.show:
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = True
                print("  [paused]")
            if key == ord("r"):
                new_rect = select_roi(frame, WIN)
                if new_rect and new_rect[2] > 0 and new_rect[3] > 0:
                    tracker.init(frame, new_rect)
                    trail.clear()
                    print(f"  Re-initialised at {new_rect}")
            if key == ord("s"):
                fname = f"snap_{snap_n:04d}.png"
                cv2.imwrite(fname, annotated)
                print(f"  Saved snapshot → {fname}")
                snap_n += 1

        frame_idx += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total = max(time.time() - t0, 1e-9)
    print("=" * 60)
    print(f"  Done. {frame_idx} frames in {total:.1f}s ({frame_idx/total:.1f} fps avg)")
    if args.save and Path(args.save).exists():
        print(f"  Saved → {args.save}  ({Path(args.save).stat().st_size/1e6:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
