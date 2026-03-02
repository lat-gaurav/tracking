#!/usr/bin/env python3
"""
Dual-Template SiamRPN + YOLO tracker.

Template bank
-------------
  zf_anchor  – extracted from the user's initial ROI (reset only on R key)
  zf_yolo    – extracted from the last YOLO correction box

Every frame both templates are tested independently against the same
search crop.  The one that returns the higher SiamRPN score "wins" and
its predicted box becomes the current tracker state.

YOLO still runs every frame to trigger two kinds of correction:
  a. HIGH-CONF   – det conf ≥ --corr-conf  AND  IoU ≥ --corr-iou
  b. PERIODIC    – every --corr-interval frames (if matching det found)
A correction updates ONLY zf_yolo; zf_anchor is permanent.

Visual legend
-------------
  Green  solid  – anchor template is winning
  Blue   solid  – yolo template is winning
  Cyan   dashed – last YOLO correction box (fades after 5 frames)
  Red    dashed – lost (both template scores < --score-thr)

Controls (--show)
-----------------
  R         re-select target (resets BOTH templates)
  SPACE     pause / resume
  S         save snapshot PNG
  Q / ESC   quit

Usage
-----
    # from /Users/gaurav/tracking
    tracking/bin/python3 siamese/siam_dual_template.py \\
        --video video_test/nadir_ped_crossing_crop640.mp4 \\
        --config siamrpn_alex_dwxcorr                     \\
        --yolo  models/yolov26nobbnew_merged_1024.pt       \\
        --show
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── pysot on sys.path ─────────────────────────────────────────────────────────
_PYSOT_ROOT = Path(__file__).resolve().parent / "pysot"
if str(_PYSOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYSOT_ROOT))

import cv2
import torch
from ultralytics import YOLO

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

# ── paths / constants ─────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_EXPERIMENTS = _PYSOT_ROOT / "experiments"
_WEIGHTS_ROOT  = _HERE.parent / "resources" / "weights"
DEFAULT_VIDEO  = str(_HERE.parent / "resources" / "video_test" / "nadir_ped_crossing_crop640.mp4")
DEFAULT_CONFIG = "siamrpn_alex_dwxcorr"
DEFAULT_YOLO   = str(_HERE.parent / "resources" / "models" / "yolov26nobbnew_merged_1024.pt")
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

COL_ANCHOR  = (  0, 255,   0)   # green  – anchor template winning
COL_YOLO_T  = (255, 140,   0)   # orange – yolo template winning
COL_CORR    = (255, 200,   0)   # cyan   – last correction box
COL_INIT    = (  0, 200, 255)   # yellow – initial box (first frame)
COL_LOST    = (  0,   0, 255)   # red    – lost
COL_DET_DIM = ( 60,  60,  60)   # grey   – background detections


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def iou(a, b):
    """IoU of two (x, y, w, h) boxes."""
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(*a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(*b)
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-9)


def best_matching_detection(track_box, detections, min_iou=0.15):
    best_box  = None
    best_conf = 0.0
    best_iou  = 0.0
    for det_box, det_conf in detections:
        ov = iou(track_box, det_box)
        if ov >= min_iou and ov > best_iou:
            best_iou  = ov
            best_box  = det_box
            best_conf = det_conf
    return best_box, best_conf, best_iou


def nearest_detection(roi_xywh, detections):
    rx = roi_xywh[0] + roi_xywh[2] / 2
    ry = roi_xywh[1] + roi_xywh[3] / 2
    best    = None
    best_d2 = float("inf")
    for (dx, dy, dw, dh), _ in detections:
        cx = dx + dw / 2; cy = dy + dh / 2
        d2 = (cx - rx)**2 + (cy - ry)**2
        if d2 < best_d2:
            best_d2 = d2
            best    = (dx, dy, dw, dh)
    return best


# ══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_dashed_rect(img, x1, y1, x2, y2, color, thick=1, dash=10):
    for (ax, ay), (bx, by) in [
        ((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
        ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))
    ]:
        length = int(np.hypot(bx-ax, by-ay))
        if length == 0:
            continue
        ddx, ddy = (bx-ax)/length, (by-ay)/length
        for i in range(0, length, dash*2):
            sx = int(ax + ddx*i);            sy = int(ay + ddy*i)
            ex = int(ax + ddx*min(i+dash, length)); ey = int(ay + ddy*min(i+dash, length))
            cv2.line(img, (sx, sy), (ex, ey), color, thick)


def put_label(img, text, x, y, fg=(0,0,0), bg=(0,255,0), scale=0.55, thick=1):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, max(0, y-th-bl)), (x+tw+4, y+bl), bg, -1)
    cv2.putText(img, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def draw_box(img, bbox, color, thick, label=None):
    sx, sy, sw, sh = [int(v) for v in bbox]
    cv2.rectangle(img, (sx, sy), (sx+sw, sy+sh), color, thick)
    # corner accents
    ca = 16
    for (px, py), (ddx, ddy) in [
        ((sx, sy), (1, 1)), ((sx+sw, sy), (-1, 1)),
        ((sx+sw, sy+sh), (-1, -1)), ((sx, sy+sh), (1, -1))
    ]:
        cv2.line(img, (px, py), (px + ddx*ca, py), color, thick)
        cv2.line(img, (px, py), (px, py + ddy*ca), color, thick)
    if label:
        put_label(img, label, sx, max(14, sy-4), fg=(0,0,0), bg=color)


# ══════════════════════════════════════════════════════════════════════════════
# Dual-template helper
# ══════════════════════════════════════════════════════════════════════════════

def track_with_zf(tracker, frame, zf):
    """
    Run tracker.track() substituting a specific template feature (zf),
    then RESTORE the tracker's internal state so the next call can start
    from the same position.  Returns the outputs dict.
    """
    saved_center = tracker.center_pos.copy()
    saved_size   = tracker.size.copy()
    saved_zf     = tracker.model.zf          # keep reference to current zf

    tracker.model.zf = zf
    outputs = tracker.track(frame)

    # Restore state – the caller will apply whichever outputs it chooses
    tracker.center_pos = saved_center
    tracker.size       = saved_size
    tracker.model.zf   = saved_zf

    return outputs


def apply_outputs(tracker, outputs, zf):
    """
    Permanently update tracker state from the chosen outputs and template.
    pysot stores center_pos / size from the last track(); we reconstruct them.
    """
    bbox = outputs["bbox"]                        # (x, y, w, h) floats
    tracker.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                   bbox[1] + bbox[3] / 2])
    tracker.size       = np.array([bbox[2], bbox[3]])
    tracker.model.zf   = zf


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
        weights_path = str(_WEIGHTS_ROOT / config_name / "model" / "model.pth")
    if not Path(weights_path).exists():
        print(f"""
ERROR: SiamRPN weights not found at:
  {weights_path}

Download (open in browser):
  siamrpn_alex_dwxcorr       → https://drive.google.com/open?id=1t62x56Jl7baUzPTo0QrC4jJnwvPZm-2m
  siamrpn_r50_l234_dwxcorr  → https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG

Place at: siamese/pysot/experiments/<config>/model/model.pth
Baidu Yun: https://pan.baidu.com/s/1GB9-aTtjG57SebraVoBfuQ  (code: j9yb)
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
        description="Dual-Template SiamRPN + YOLO tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source",  default=DEFAULT_VIDEO,
                   help="Video file path, or camera index (0, 1, …) for webcam")
    p.add_argument("--config",  default=DEFAULT_CONFIG)
    p.add_argument("--weights", default="",             help="SiamRPN .pth (auto-resolved)")
    p.add_argument("--yolo",    default=DEFAULT_YOLO)
    p.add_argument("--device",  default=DEFAULT_DEVICE)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou",  type=float, default=0.45)

    p.add_argument("--corr-interval", type=int,   default=10,
                   help="Reinit zf_yolo every N frames if matching det found")
    p.add_argument("--corr-conf",     type=float, default=0.60,
                   help="Immediate correction if det conf ≥ this")
    p.add_argument("--corr-iou",      type=float, default=0.35,
                   help="Immediate correction if IoU with track box ≥ this")
    p.add_argument("--min-iou",       type=float, default=0.15,
                   help="Periodic correction: min IoU to accept detection")
    p.add_argument("--score-thr",     type=float, default=0.20,
                   help="Both scores below this → lost")

    p.add_argument("--save",      default="")
    p.add_argument("--show",      action="store_true")
    p.add_argument("--thickness", type=int, default=1)
    p.add_argument("--show-dets", action="store_true",
                   help="Draw all YOLO detections (dim grey)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ── open video ────────────────────────────────────────────────────────────
    raw = args.source.strip() if isinstance(args.source, str) else str(args.source)
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
    tracker  = load_siam(args.config, args.weights)
    detector = YOLO(args.yolo)
    detector.to(args.device)
    print(f"  YOLO    : {Path(args.yolo).name}  device={args.device}")
    print(f"  Correction interval : every {args.corr_interval} frames")
    print(f"  High-conf trigger   : conf≥{args.corr_conf}  IoU≥{args.corr_iou}")
    print("=" * 62)

    WIN = "Dual-Template SiamRPN"
    if args.show:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: cannot read first frame", file=sys.stderr); return 1

    if not args.show:
        print("ERROR: --show is required for ROI selection", file=sys.stderr)
        return 1

    # ── step 1: user draws ROI ────────────────────────────────────────────────
    roi = select_roi(first_frame, WIN)
    if roi is None:
        print("No target selected. Exiting."); return 0
    roi_xywh = list(map(float, roi))

    # ── step 2: YOLO on first frame → refine ROI ─────────────────────────────
    dets_first = run_yolo(detector, first_frame,
                          args.yolo_conf, args.yolo_iou, args.device, args.imgsz)
    refined = nearest_detection(roi_xywh, dets_first)
    if refined:
        print(f"  ROI     : {[int(v) for v in roi_xywh]}")
        print(f"  Refined : {[int(v) for v in refined]}  (snapped to nearest YOLO det)")
        init_box = refined
    else:
        print(f"  ROI     : {[int(v) for v in roi_xywh]}  (no YOLO det found)")
        init_box = tuple(roi_xywh)

    # ── step 3: init tracker → capture zf_anchor ─────────────────────────────
    tracker.init(first_frame, init_box)
    # zf_anchor is the template from the user's ROI – never changes unless R key
    zf_anchor = tracker.model.zf
    # zf_yolo starts as a copy of anchor (no YOLO correction yet)
    zf_yolo   = zf_anchor          # same tensor until first YOLO correction

    current_box      = list(map(int, init_box))
    active_template  = "anchor"    # which template won last frame

    # ── state ─────────────────────────────────────────────────────────────────
    frames_since_corr = 0
    last_corr_box     = None
    last_corr_frame   = -1
    last_corr_type    = ""
    snap_n            = 0
    paused            = False
    frame_idx         = 1
    t0                = time.time()

    # annotate and show first frame
    annotated_first = first_frame.copy()
    ix, iy, iw, ih  = map(int, init_box)
    cv2.rectangle(annotated_first, (ix, iy), (ix+iw, iy+ih), COL_INIT, args.thickness)
    put_label(annotated_first, "INIT", ix, max(14, iy-4), bg=COL_INIT)
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
            elif key == ord("s"):
                fname = f"snap_{snap_n:04d}.png"; snap_n += 1
                cv2.imwrite(fname, annotated); print(f"  Snapshot → {fname}")
            elif key == ord("r"):
                new_roi = select_roi(frame, WIN)
                if new_roi and new_roi[2] > 0 and new_roi[3] > 0:
                    dets_r = run_yolo(detector, frame,
                                      args.yolo_conf, args.yolo_iou,
                                      args.device, args.imgsz)
                    ref = nearest_detection(list(map(float, new_roi)), dets_r)
                    init_b = ref if ref else tuple(map(float, new_roi))
                    tracker.init(frame, init_b)
                    # R key resets BOTH templates to new ROI
                    zf_anchor = tracker.model.zf
                    zf_yolo   = zf_anchor
                    current_box = list(map(int, init_b))
                    active_template = "anchor"
                    frames_since_corr = 0
                    last_corr_box = None
                    print(f"  Re-init BOTH templates → {[int(v) for v in init_b]}")
                paused = False
            continue

        # ── read frame ────────────────────────────────────────────────────────
        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()

        # ── dual-template tracking ────────────────────────────────────────────
        # Run tracker twice (save/restore state between calls), pick higher score.
        out_a = track_with_zf(tracker, frame, zf_anchor)
        out_y = track_with_zf(tracker, frame, zf_yolo)

        score_a = float(out_a.get("best_score", 0))
        score_y = float(out_y.get("best_score", 0))
        best_score = max(score_a, score_y)

        if score_a >= score_y:
            winner_out  = out_a
            winner_zf   = zf_anchor
            active_template = "anchor"
        else:
            winner_out  = out_y
            winner_zf   = zf_yolo
            active_template = "yolo"

        # Apply winner – permanently update tracker state
        apply_outputs(tracker, winner_out, winner_zf)
        current_box = list(map(int, winner_out["bbox"]))
        is_lost     = best_score < args.score_thr

        # ── YOLO detect ───────────────────────────────────────────────────────
        dets = run_yolo(detector, frame,
                        args.yolo_conf, args.yolo_iou, args.device, args.imgsz)

        # ── drift correction → updates zf_yolo only ───────────────────────────
        corrected  = False
        corr_type  = ""
        corr_det   = None

        if not is_lost and dets:
            best_box, best_conf, best_ov = best_matching_detection(
                current_box, dets, min_iou=args.min_iou)

            if (best_box is not None
                    and best_conf >= args.corr_conf
                    and best_ov  >= args.corr_iou):
                corrected  = True
                corr_type  = "high-conf"
                corr_det   = best_box

            elif (not corrected
                  and frames_since_corr >= args.corr_interval
                  and best_box is not None):
                corrected  = True
                corr_type  = "periodic"
                corr_det   = best_box

        if corrected:
            # Extract new zf_yolo without disturbing tracker state
            saved_center = tracker.center_pos.copy()
            saved_size   = tracker.size.copy()
            saved_zf     = tracker.model.zf

            tracker.init(frame, (corr_det[0], corr_det[1], corr_det[2], corr_det[3]))
            zf_yolo = tracker.model.zf          # capture new yolo template

            # Restore state to winner's position (init() resets it)
            tracker.center_pos = saved_center
            tracker.size       = saved_size
            tracker.model.zf   = saved_zf       # keep current winner's zf active

            last_corr_box   = list(map(int, corr_det))
            last_corr_frame = frame_idx
            last_corr_type  = corr_type
            frames_since_corr = 0
            print(f"  [frame {frame_idx:4d}] zf_yolo updated ({corr_type})  "
                  f"IoU={best_ov:.2f}  conf={best_conf:.2f}")
        else:
            frames_since_corr += 1

        # ══════════════════════════════════════════════════════════════════════
        # Annotation
        # ══════════════════════════════════════════════════════════════════════

        # 1. background detections (optional)
        if args.show_dets:
            for (dx, dy, dw, dh), _ in dets:
                cv2.rectangle(annotated,
                              (int(dx), int(dy)), (int(dx+dw), int(dy+dh)),
                              COL_DET_DIM, 1)

        # 2. last correction box
        if last_corr_box and (frame_idx - last_corr_frame) <= 5:
            lx, ly, lw, lh = last_corr_box
            draw_dashed_rect(annotated, lx, ly, lx+lw, ly+lh, COL_CORR, 1)
            put_label(annotated, f"DET ({last_corr_type})",
                      lx, max(14, ly-4), fg=(0,0,0), bg=COL_CORR, scale=0.45)

        # 3. main track box
        sx, sy, sw, sh = current_box
        if is_lost:
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, COL_LOST, args.thickness)
            cv2.putText(annotated, f"LOST a={score_a:.2f} y={score_y:.2f}",
                        (sx+2, sy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_LOST, 1, cv2.LINE_AA)
        else:
            col = COL_ANCHOR if active_template == "anchor" else COL_YOLO_T
            lbl = f"A({score_a:.2f})" if active_template == "anchor" else f"Y({score_y:.2f})"
            draw_box(annotated, current_box, col, args.thickness, label=lbl)

        # 4. mini score bar (anchor vs yolo) – top-right corner
        bar_x = W - 160
        bar_y = 10
        bar_h = 14
        bar_w = 140

        # anchor bar
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (40,40,40), -1)
        cv2.rectangle(annotated, (bar_x, bar_y),
                      (bar_x + int(bar_w * min(1, score_a)), bar_y+bar_h),
                      COL_ANCHOR, -1)
        cv2.putText(annotated, f"A {score_a:.2f}", (bar_x+2, bar_y+bar_h-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)

        # yolo bar
        by2 = bar_y + bar_h + 4
        cv2.rectangle(annotated, (bar_x, by2), (bar_x+bar_w, by2+bar_h), (40,40,40), -1)
        cv2.rectangle(annotated, (bar_x, by2),
                      (bar_x + int(bar_w * min(1, score_y)), by2+bar_h),
                      COL_YOLO_T, -1)
        cv2.putText(annotated, f"Y {score_y:.2f}", (bar_x+2, by2+bar_h-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 1, cv2.LINE_AA)

        # 5. countdown bar
        bw  = W - 20
        prog = min(1.0, frames_since_corr / max(1, args.corr_interval))
        cv2.rectangle(annotated, (10, H-22), (10+bw, H-22+6), (60,60,60), -1)
        cv2.rectangle(annotated, (10, H-22),
                      (10+int(bw*prog), H-22+6),
                      COL_CORR if prog >= 1.0 else (0, 180, 100), -1)
        cv2.putText(annotated,
                    f"zf_yolo update in {max(0, args.corr_interval - frames_since_corr)} frames",
                    (10, H-26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160,160,160), 1, cv2.LINE_AA)

        # 6. HUD
        elapsed = max(time.time() - t0, 1e-9)
        fr_str  = f"{frame_idx}" if is_live else f"{frame_idx}/{n_frames}"
        cv2.putText(annotated,
                    f"FPS {frame_idx/elapsed:.1f}  |  frame {fr_str}  |  tmpl={active_template}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,0), 1, cv2.LINE_AA)
        cv2.putText(annotated, "R=reselect  SPACE=pause  S=snap  Q=quit",
                    (10, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,100), 1, cv2.LINE_AA)

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
            elif key == ord("s"):
                fname = f"snap_{snap_n:04d}.png"; snap_n += 1
                cv2.imwrite(fname, annotated); print(f"  Snapshot → {fname}")
            elif key == ord("r"):
                new_roi = select_roi(frame, WIN)
                if new_roi and new_roi[2] > 0 and new_roi[3] > 0:
                    dets_r = run_yolo(detector, frame,
                                      args.yolo_conf, args.yolo_iou,
                                      args.device, args.imgsz)
                    ref = nearest_detection(list(map(float, new_roi)), dets_r)
                    init_b = ref if ref else tuple(map(float, new_roi))
                    tracker.init(frame, init_b)
                    zf_anchor = tracker.model.zf
                    zf_yolo   = zf_anchor
                    current_box = list(map(int, init_b))
                    active_template = "anchor"
                    frames_since_corr = 0
                    last_corr_box = None
                    print(f"  Re-init BOTH templates → {[int(v) for v in init_b]}")

        frame_idx += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total = max(time.time() - t0, 1e-9)
    print("=" * 62)
    print(f"  Done. {frame_idx} frames  {total:.1f}s  ({frame_idx/total:.1f} fps avg)")
    if args.save and Path(args.save).exists():
        print(f"  Saved → {args.save}  ({Path(args.save).stat().st_size/1e6:.1f} MB)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
