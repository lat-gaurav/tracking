#!/usr/bin/env python3
"""
Hybrid SiamRPN + YOLO drift-correction tracker.

Pipeline
--------
  1. User draws ROI on the first frame.
  2. YOLO runs on that frame → the nearest detection box is used to REFINE
     the initial ROI before SiamRPN is initialised.
  3. Every frame SiamRPN tracks the object.
  4. YOLO also runs every frame (background detection).
  5. Drift correction happens when EITHER:
       a. HIGH-CONF trigger   – a YOLO detection overlaps the current SiamRPN
                                box with IoU ≥ --corr-iou AND confidence ≥ --corr-conf
       b. PERIODIC trigger    – every --corr-interval frames, the best-matching
                                YOLO detection (IoU ≥ --min-iou) is accepted.
     On correction the SiamRPN tracker is RE-INITIALISED on the detection box,
     snapping it back to the detector's tighter, less-drifted estimate.

Visual legend
-------------
  Green  solid        – current SiamRPN track box
  Cyan   dashed       – YOLO detection box that triggered the last correction
  Orange dotted trail – centroid history
  Yellow text HUD     – frame / fps / score
  Red    "LOST"       – SiamRPN score fell below --score-thr

Controls (--show)
-----------------
  R         re-select target (ROI selector re-opens)
  SPACE     pause / resume
  S         save snapshot PNG
  Q / ESC   quit

Usage
-----
    # from /Users/gaurav/tracking
    tracking/bin/python3 siamese/siam_track_drift_correct.py \\
        --video video_test/nadir_ped_crossing_crop640.mp4    \\
        --config siamrpn_alex_dwxcorr                        \\
        --yolo  models/yolov26nobbnew_merged_1024.pt         \\
        --show

    tracking/bin/python3 siamese/siam_track_drift_correct.py \\
        --video video_test/nadir_crosswalk_ped.mp4           \\
        --config siamrpn_alex_dwxcorr                        \\
        --yolo  models/yolov26nobbnew_merged_1024.pt         \\
        --show --save out_hybrid.mp4                         \\
        --corr-interval 10 --corr-conf 0.6 --corr-iou 0.35
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

# ── pysot on sys.path ─────────────────────────────────────────────────────────
_PYSOT_ROOT = Path(__file__).resolve().parent / "pysot"
if str(_PYSOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYSOT_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

# ── paths / colours ───────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_EXPERIMENTS = _PYSOT_ROOT / "experiments"
DEFAULT_VIDEO  = str(_HERE.parent / "video_test" / "nadir_ped_crossing_crop640.mp4")
DEFAULT_CONFIG = "siamrpn_alex_dwxcorr"
DEFAULT_YOLO   = str(_HERE.parent / "models" / "yolov26nobbnew_merged_1024.pt")
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

COL_SIAM    = (  0, 255,   0)   # green   – SiamRPN live box
COL_CORR    = (255, 200,   0)   # cyan    – YOLO correction box
COL_INIT    = (  0, 200, 255)   # yellow  – initial refined box (first frame)
COL_TRAIL   = (  0, 165, 255)   # orange  – trail
COL_LOST    = (  0,   0, 255)   # red     – lost text
COL_DET_DIM = ( 60,  60,  60)   # grey    – background detections


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def iou(a_xywh, b_xywh):
    """IoU between two (x, y, w, h) boxes."""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(*a_xywh)
    bx1, by1, bx2, by2 = xywh_to_xyxy(*b_xywh)
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-9)


def best_matching_detection(track_box, detections, min_iou=0.15):
    """
    Return (best_det_box, best_conf, best_iou) or (None, 0, 0).
    detections: list of (x, y, w, h, conf)
    """
    best_box  = None
    best_conf = 0.0
    best_iou  = 0.0
    for det_box_xywh, det_conf in detections:
        ov = iou(track_box, det_box_xywh)
        if ov >= min_iou and ov > best_iou:
            best_iou  = ov
            best_box  = det_box_xywh
            best_conf = det_conf
    return best_box, best_conf, best_iou


def nearest_detection(roi_xywh, detections):
    """
    Return the detection whose centre is closest to the ROI centre.
    Used to refine the user-drawn initial ROI.
    detections: list of (x, y, w, h, conf)
    """
    rx = roi_xywh[0] + roi_xywh[2] / 2
    ry = roi_xywh[1] + roi_xywh[3] / 2
    best     = None
    best_d2  = float("inf")
    for (dx, dy, dw, dh), conf in detections:
        cx = dx + dw / 2; cy = dy + dh / 2
        d2 = (cx - rx)**2 + (cy - ry)**2
        if d2 < best_d2:
            best_d2 = d2
            best    = (dx, dy, dw, dh)
    return best   # None if no detections


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_dashed_rect(img, x1, y1, x2, y2, color, thick=2, dash=10):
    for (ax, ay), (bx, by) in [
        ((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
        ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))
    ]:
        length = int(np.hypot(bx-ax, by-ay))
        if length == 0:
            continue
        dx, dy = (bx-ax)/length, (by-ay)/length
        for i in range(0, length, dash*2):
            sx = int(ax + dx*i);      sy = int(ay + dy*i)
            ex = int(ax + dx*min(i+dash,length)); ey = int(ay + dy*min(i+dash,length))
            cv2.line(img, (sx, sy), (ex, ey), color, thick)


def put_label(img, text, x, y, fg=(0,0,0), bg=COL_SIAM, scale=0.55, thick=1):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, max(0, y-th-bl)), (x+tw+4, y+bl), bg, -1)
    cv2.putText(img, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_siam(config_name, weights_path):
    cfg_file = _EXPERIMENTS / config_name / "config.yaml"
    if not cfg_file.exists():
        avail = sorted(p.name for p in _EXPERIMENTS.iterdir() if p.is_dir())
        print(f"ERROR: pysot config not found: {cfg_file}\n  Available: {avail}", file=sys.stderr)
        sys.exit(1)
    if not weights_path:
        weights_path = str(_EXPERIMENTS / config_name / "model" / "model.pth")
    if not Path(weights_path).exists():
        print(f"""
ERROR: SiamRPN weights not found at:
  {weights_path}

Download (open in browser):
  siamrpn_alex_dwxcorr  → https://drive.google.com/open?id=1t62x56Jl7baUzPTo0QrC4jJnwvPZm-2m
  siamrpn_r50_l234_dwxcorr → https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG

Place at: siamese/pysot/experiments/<config>/model/model.pth
Also on Baidu Yun: https://pan.baidu.com/s/1GB9-aTtjG57SebraVoBfuQ  (code: j9yb)
""", file=sys.stderr)
        sys.exit(1)

    cfg.merge_from_file(str(cfg_file))
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    model = ModelBuilder()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval().to(torch.device("cuda" if cfg.CUDA else "cpu"))
    tracker = build_tracker(model)
    print(f"  SiamRPN  : {cfg.TRACK.TYPE}  config={config_name}  CUDA={cfg.CUDA}")
    return tracker


def run_yolo(detector, frame, conf_thr, iou_thr, device, imgsz):
    """Run YOLO and return list of ((x,y,w,h), conf) in pixel coords."""
    result = detector.predict(frame, imgsz=imgsz, conf=conf_thr, iou=iou_thr,
                              device=device, verbose=False)[0]
    boxes  = result.obb if detector.task == "obb" else result.boxes
    dets   = []
    if boxes is not None and len(boxes) > 0:
        for xyxy, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = xyxy.tolist()
            dets.append(((x1, y1, x2-x1, y2-y1), float(score)))
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# ROI selector
# ══════════════════════════════════════════════════════════════════════════════

def select_roi(frame, win_name):
    hint = frame.copy()
    cv2.putText(hint, "Draw bbox around target, press ENTER to confirm",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(win_name, hint)
    try:
        rect = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
    except Exception:
        return None
    if rect[2] == 0 or rect[3] == 0:
        return None
    return rect   # (x, y, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SiamRPN + YOLO drift-correction hybrid tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── inputs ────────────────────────────────────────────────────────────────
    p.add_argument("--video",   default=DEFAULT_VIDEO,   help="Video file or camera index")
    p.add_argument("--config",  default=DEFAULT_CONFIG,  help="pysot experiment config name")
    p.add_argument("--weights", default="",              help="SiamRPN .pth path (auto-resolved)")
    p.add_argument("--yolo",    default=DEFAULT_YOLO,    help="YOLO model (.pt / .engine)")
    p.add_argument("--device",  default=DEFAULT_DEVICE,  help="YOLO device (mps / cpu / cuda)")
    p.add_argument("--imgsz",   type=int,   default=640, help="YOLO inference size")
    p.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO detection confidence threshold")
    p.add_argument("--yolo-iou",  type=float, default=0.45, help="YOLO NMS IoU threshold")

    # ── drift correction ──────────────────────────────────────────────────────
    p.add_argument("--corr-interval", type=int,   default=10,
                   help="Periodic correction: reinit SiamRPN every N frames if a matching detection exists")
    p.add_argument("--corr-conf",     type=float, default=0.60,
                   help="High-conf trigger: detection confidence required to trigger immediate correction")
    p.add_argument("--corr-iou",      type=float, default=0.35,
                   help="High-conf trigger: minimum IoU with current track box to accept detection")
    p.add_argument("--min-iou",       type=float, default=0.15,
                   help="Periodic trigger: minimum IoU with track box to accept a matching detection")
    p.add_argument("--score-thr",     type=float, default=0.20,
                   help="SiamRPN score below this → declare object lost (no correction attempted)")

    # ── output / display ──────────────────────────────────────────────────────
    p.add_argument("--save",      default="",    help="Output annotated video path")
    p.add_argument("--show",      action="store_true", help="Display live window")
    p.add_argument("--trail",     type=int, default=0,  help="Trail length (0 = off)")
    p.add_argument("--thickness", type=int, default=1,  help="Box line thickness")
    p.add_argument("--show-dets", action="store_true",  help="Draw all YOLO detections (dim grey)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── open video ────────────────────────────────────────────────────────────
    raw = args.video.strip()
    if raw.lstrip("-").isdigit():
        cap = cv2.VideoCapture(int(raw))
        n_frames, is_live, src_label = 0, True, f"webcam {raw}"
    else:
        cap      = cv2.VideoCapture(str(Path(raw).resolve()))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        is_live  = False
        src_label = Path(raw).name

    if not cap.isOpened():
        print(f"ERROR: cannot open {raw}", file=sys.stderr); return 1

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (W, H))

    # ── load models ───────────────────────────────────────────────────────────
    print("=" * 62)
    print(f"  Source  : {src_label}  ({W}x{H} @ {fps_in:.1f} fps)")
    siam_tracker = load_siam(args.config, args.weights)
    detector = YOLO(args.yolo)
    detector.to(args.device)
    print(f"  YOLO    : {Path(args.yolo).name}  device={args.device}")
    print(f"  Correction interval : every {args.corr_interval} frames")
    print(f"  High-conf trigger   : conf≥{args.corr_conf}  IoU≥{args.corr_iou}")
    print(f"  Periodic min-IoU    : {args.min_iou}")
    print("=" * 62)

    WIN = "SiamRPN + YOLO Drift Correction"
    if args.show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # ── read first frame ──────────────────────────────────────────────────────
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: cannot read first frame", file=sys.stderr); return 1

    if not args.show:
        print("ERROR: --show is required for interactive ROI selection", file=sys.stderr)
        return 1

    # ── step 1: user draws ROI ────────────────────────────────────────────────
    roi = select_roi(first_frame, WIN)
    if roi is None:
        print("No target selected. Exiting."); return 0
    roi_xywh = list(map(float, roi))

    # ── step 2: YOLO detects on first frame → refine roi ─────────────────────
    dets_first = run_yolo(detector, first_frame,
                          args.yolo_conf, args.yolo_iou, args.device, args.imgsz)
    refined_box = nearest_detection(roi_xywh, dets_first)
    if refined_box is not None:
        print(f"  ROI     : {[int(v) for v in roi_xywh]}")
        print(f"  Refined : {[int(v) for v in refined_box]}  (snapped to nearest YOLO det)")
        init_box = refined_box
    else:
        print(f"  ROI     : {[int(v) for v in roi_xywh]}  (no YOLO det found, using raw ROI)")
        init_box = tuple(roi_xywh)

    # ── step 3: init SiamRPN on the (potentially refined) box ────────────────
    siam_tracker.init(first_frame, init_box)
    current_box = list(map(int, init_box))   # (x, y, w, h) – updated each frame

    # ── state ─────────────────────────────────────────────────────────────────
    trail           = deque(maxlen=max(1, args.trail))
    frames_since_corr = 0            # frames since last correction
    last_corr_box   = None           # YOLO box used in last correction (for drawing)
    last_corr_frame = -1             # frame index of last correction
    last_corr_type  = ""             # "high-conf" | "periodic"
    snap_n          = 0
    paused          = False
    frame_idx       = 1              # first frame already consumed
    t0              = time.time()

    # annotate first frame
    annotated_first = first_frame.copy()
    ix, iy, iw, ih = map(int, init_box)
    cv2.rectangle(annotated_first, (ix, iy), (ix+iw, iy+ih), COL_INIT, args.thickness)
    put_label(annotated_first, "INIT (refined)" if refined_box else "INIT",
              ix, max(14, iy-4), bg=COL_INIT)
    if writer:
        writer.write(annotated_first)
    cv2.imshow(WIN, annotated_first)
    cv2.waitKey(1)

    # ══════════════════════════════════════════════════════════════════════════
    # Tracking loop
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        # ── pause handler ─────────────────────────────────────────────────────
        if paused:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = False; print("  [resumed]")
            elif key == ord("r"):
                new_roi = select_roi(frame, WIN)
                if new_roi and new_roi[2] > 0 and new_roi[3] > 0:
                    dets_r = run_yolo(detector, frame,
                                      args.yolo_conf, args.yolo_iou,
                                      args.device, args.imgsz)
                    ref = nearest_detection(list(map(float, new_roi)), dets_r)
                    init_b = ref if ref else tuple(map(float, new_roi))
                    siam_tracker.init(frame, init_b)
                    current_box = list(map(int, init_b))
                    trail.clear()
                    frames_since_corr = 0
                    last_corr_box = None
                    print(f"  Re-initialised at {[int(v) for v in init_b]}")
                paused = False
            elif key == ord("s"):
                fname = f"snap_{snap_n:04d}.png"
                cv2.imwrite(fname, annotated); snap_n += 1
                print(f"  Snapshot → {fname}")
            continue

        # ── read frame ────────────────────────────────────────────────────────
        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()

        # ── SiamRPN track ─────────────────────────────────────────────────────
        outputs     = siam_tracker.track(frame)
        siam_box    = list(map(int, outputs["bbox"]))       # (x, y, w, h)
        siam_score  = float(outputs.get("best_score", 1.0))
        is_lost     = siam_score < args.score_thr
        current_box = siam_box

        # update trail
        cx = siam_box[0] + siam_box[2] // 2
        cy = siam_box[1] + siam_box[3] // 2
        trail.append((cx, cy))

        # ── YOLO detect ───────────────────────────────────────────────────────
        dets = run_yolo(detector, frame,
                        args.yolo_conf, args.yolo_iou, args.device, args.imgsz)

        # ── drift correction decision ─────────────────────────────────────────
        corrected   = False
        corr_type   = ""
        corr_det    = None

        if not is_lost and dets:
            best_box, best_conf, best_ov = best_matching_detection(
                siam_box, dets, min_iou=args.min_iou)

            # trigger A: high-confidence immediate correction
            if (best_box is not None
                    and best_conf >= args.corr_conf
                    and best_ov  >= args.corr_iou):
                corrected = True
                corr_type = "high-conf"
                corr_det  = best_box

            # trigger B: periodic correction (every N frames)
            elif (not corrected
                  and frames_since_corr >= args.corr_interval
                  and best_box is not None):
                corrected = True
                corr_type = "periodic"
                corr_det  = best_box

        if corrected:
            siam_tracker.init(frame, (corr_det[0], corr_det[1],
                                       corr_det[2], corr_det[3]))
            current_box       = list(map(int, corr_det))
            cx = current_box[0] + current_box[2] // 2
            cy = current_box[1] + current_box[3] // 2
            trail.append((cx, cy))
            last_corr_box     = current_box[:]
            last_corr_frame   = frame_idx
            last_corr_type    = corr_type
            frames_since_corr = 0
            print(f"  [frame {frame_idx:4d}] CORRECTION ({corr_type})  "
                  f"IoU={best_ov:.2f}  conf={best_conf:.2f}  "
                  f"box={[int(v) for v in corr_det]}")
        else:
            frames_since_corr += 1

        # ══════════════════════════════════════════════════════════════════════
        # Annotation
        # ══════════════════════════════════════════════════════════════════════

        # 1. background YOLO detections (dim grey, optional)
        if args.show_dets:
            for (dx, dy, dw, dh), _ in dets:
                cv2.rectangle(annotated,
                              (int(dx), int(dy)),
                              (int(dx+dw), int(dy+dh)),
                              COL_DET_DIM, 1)

        # 3. last correction box (dashed cyan), fades after 5 frames
        if last_corr_box and (frame_idx - last_corr_frame) <= 5:
            lx, ly, lw, lh = last_corr_box
            draw_dashed_rect(annotated, lx, ly, lx+lw, ly+lh, COL_CORR,
                             thick=max(1, args.thickness-1))
            put_label(annotated, f"DET ({last_corr_type})",
                      lx, max(14, ly-4), fg=(0,0,0), bg=COL_CORR, scale=0.5)

        # 4. main SiamRPN track box
        sx, sy, sw, sh = current_box
        if is_lost:
            # dashed red when lost
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, COL_LOST, args.thickness)
            cv2.putText(annotated, f"LOST (score={siam_score:.2f})",
                        (sx+2, sy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_LOST, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(annotated, (sx, sy), (sx+sw, sy+sh), COL_SIAM, args.thickness)
            # corner accents
            ca = 16
            for (ax2, ay2), (ddx, ddy) in [
                ((sx,sy),(1,1)), ((sx+sw,sy),(-1,1)),
                ((sx+sw,sy+sh),(-1,-1)), ((sx,sy+sh),(1,-1))
            ]:
                cv2.line(annotated,(ax2,ay2),(ax2+ddx*ca,ay2),COL_SIAM,args.thickness)
                cv2.line(annotated,(ax2,ay2),(ax2,ay2+ddy*ca),COL_SIAM,args.thickness)
            score_lbl = f"score {siam_score:.3f}"
            put_label(annotated, score_lbl, sx, max(14, sy-4),
                      fg=(0,0,0), bg=COL_SIAM, scale=0.55)

        # 5. next-correction countdown bar
        bar_w = W - 20
        prog  = min(1.0, frames_since_corr / max(1, args.corr_interval))
        bar_h = 6
        cv2.rectangle(annotated, (10, H-22), (10+bar_w, H-22+bar_h), (60,60,60), -1)
        cv2.rectangle(annotated, (10, H-22), (10+int(bar_w*prog), H-22+bar_h),
                      COL_CORR if prog >= 1.0 else (0, 180, 100), -1)
        cv2.putText(annotated,
                    f"corr in {max(0, args.corr_interval - frames_since_corr)} frames",
                    (10, H-26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

        # 6. HUD
        elapsed  = max(time.time() - t0, 1e-9)
        fps_now  = frame_idx / elapsed
        fr_str   = f"frame {frame_idx}" if is_live else f"frame {frame_idx}/{n_frames}"
        cv2.putText(annotated, f"FPS {fps_now:.1f}  |  {fr_str}  |  last_corr={last_corr_frame}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(annotated, "R=reselect  SPACE=pause  S=snap  Q=quit",
                    (10, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1, cv2.LINE_AA)

        # 7. colour legend
        cv2.putText(annotated, "SiamRPN", (W-140, H-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_SIAM, 1, cv2.LINE_AA)
        cv2.putText(annotated, "Det correction", (W-140, H-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CORR, 1, cv2.LINE_AA)

        # ── output ────────────────────────────────────────────────────────────
        if writer:
            writer.write(annotated)

        if args.show:
            cv2.imshow(WIN, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                paused = True; print("  [paused]")
            elif key == ord("r"):
                new_roi = select_roi(frame, WIN)
                if new_roi and new_roi[2] > 0 and new_roi[3] > 0:
                    dets_r = run_yolo(detector, frame,
                                      args.yolo_conf, args.yolo_iou,
                                      args.device, args.imgsz)
                    ref = nearest_detection(list(map(float, new_roi)), dets_r)
                    init_b = ref if ref else tuple(map(float, new_roi))
                    siam_tracker.init(frame, init_b)
                    current_box = list(map(int, init_b))
                    trail.clear()
                    frames_since_corr = 0
                    last_corr_box = None
                    print(f"  Re-initialised at {[int(v) for v in init_b]}")
            elif key == ord("s"):
                fname = f"snap_{snap_n:04d}.png"
                cv2.imwrite(fname, annotated); snap_n += 1
                print(f"  Snapshot → {fname}")

        frame_idx += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total = max(time.time() - t0, 1e-9)
    print("=" * 62)
    print(f"  Done. {frame_idx} frames in {total:.1f}s  ({frame_idx/total:.1f} fps avg)")
    if args.save and Path(args.save).exists():
        print(f"  Saved → {args.save}  ({Path(args.save).stat().st_size/1e6:.1f} MB)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
