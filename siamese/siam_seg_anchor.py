#!/usr/bin/env python3
"""
Segmentation-anchored Dual-Template SiamRPN tracker.

Key idea
--------
A YOLO segmentation model (yolov8n-seg by default) provides a per-pixel
person mask.  That mask is used to CLEAN the template crop before SiamRPN
sees it: background pixels are replaced with the mean foreground colour so
the network learns only the person's appearance, not whatever is behind them.

Template bank (same dual-template logic as siam_dual_template.py)
-----------------------------------------------------------------
  zf_anchor – extracted from the segmentation-masked first frame.
              PERMANENT – only reset when the user presses R.
  zf_yolo   – updated whenever the seg/YOLO correction trigger fires.
              Uses the new mask-cleaned crop so background never pollutes it.

Selection UI
------------
Instead of dragging a box, the user simply CLICKS on the person to track.
All detected persons are shown as coloured overlays; click inside any one
of them.  If segmentation finds no person at the click point a fallback
ROI-drag is offered.

Controls (--show)
-----------------
  R         re-select target (resets BOTH templates)
  SPACE     pause / resume
  S         save snapshot PNG
  Q / ESC   quit

Visual legend
-------------
  Green  solid  – anchor template winning
  Orange solid  – yolo template winning
  Cyan   dashed – last correction box (fades after 5 frames)
  Red    dashed – LOST
  Purple overlay – seg mask of current tracked person (--show-mask)

Usage
-----
    # from /Users/gaurav/tracking
    tracking/bin/python3 siamese/siam_seg_anchor.py \\
        --video video_test/nadir_ped_crossing_crop640.mp4 \\
        --config siamrpn_alex_dwxcorr                     \\
        --seg   yolov8n-seg.pt                            \\
        --show  --show-mask

    # use a heavier seg model for better masks
    tracking/bin/python3 siamese/siam_seg_anchor.py \\
        --video video_test/nadir_crosswalk_ped.mp4    \\
        --config siamrpn_alex_dwxcorr                 \\
        --seg   yolov8s-seg.pt                        \\
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
DEFAULT_VIDEO  = str(_HERE.parent / "video_test" / "nadir_ped_crossing_crop640.mp4")
DEFAULT_CONFIG = "siamrpn_r50_l234_dwxcorr"
DEFAULT_SEG    = "yolov8n-seg.pt"          # auto-downloaded by ultralytics if absent
DEFAULT_YOLO   = str(_HERE.parent / "models" / "yolov26nobbnew_merged_1024.pt")
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PERSON_CLASS   = 0                         # COCO class id for 'person'

COL_ANCHOR  = (  0, 255,   0)   # green  – anchor winning
COL_YOLO_T  = (  0, 140, 255)   # orange – yolo winning
COL_CORR    = (255, 200,   0)   # cyan   – last correction
COL_INIT    = (  0, 200, 255)   # yellow – init box
COL_LOST    = (  0,   0, 255)   # red    – lost
COL_DET_DIM = ( 40,  40,  40)   # grey   – background detections
COL_RECOVER = (255,   0, 200)   # magenta – anchor perimeter recovery
COL_KALMAN  = (180, 180,   0)   # dim yellow – Kalman prediction
MASK_COLORS = [                  # overlay colours for click-select UI
    (  0, 200, 255), (  0, 255, 100), (200,   0, 255),
    (255, 200,   0), (  0, 255, 200), (255,  60, 120),
]


# ══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ══════════════════════════════════════════════════════════════════════════════

def xywh_to_xyxy(x, y, w, h):
    return x, y, x + w, y + h


def iou(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(*a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(*b)
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-9)


def best_matching_detection(track_box, detections, min_iou=0.15):
    """detections: list of ((x,y,w,h), conf) as returned by run_yolo."""
    best_box  = None
    best_conf = 0.0
    best_iou  = 0.0
    for (det_box, det_conf) in detections:
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
    for (det_box, _conf, mask) in detections:
        cx = det_box[0] + det_box[2] / 2
        cy = det_box[1] + det_box[3] / 2
        d2 = (cx - rx)**2 + (cy - ry)**2
        if d2 < best_d2:
            best_d2 = d2
            best    = (det_box, mask)
    return best   # (xywh, mask_or_None) or None


# ══════════════════════════════════════════════════════════════════════════════
# Segmentation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _resize_mask(mask_tensor, H, W):
    """Resize a pysot/ultralytics float mask tensor to (H, W) numpy bool array."""
    m = mask_tensor.cpu().float().numpy()
    if m.shape != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
    return m > 0.5


def run_seg(seg_model, frame, conf_thr, iou_thr, device, imgsz):
    """
    Run YOLO-seg on *frame*.  Returns list of (bbox_xywh, conf, mask_HW_bool).
    mask_HW_bool is None when the model is a plain detection model (no masks).
    Only PERSON_CLASS detections are returned.
    """
    H, W = frame.shape[:2]
    result = seg_model.predict(frame, imgsz=imgsz, conf=conf_thr, iou=iou_thr,
                               device=device, verbose=False)[0]
    boxes  = result.boxes
    masks  = result.masks    # None for plain detect models

    dets = []
    if boxes is None or len(boxes) == 0:
        return dets

    for i, (xyxy, score, cls) in enumerate(zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy())):
        if int(cls) != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = xyxy.tolist()
        bbox = (x1, y1, x2-x1, y2-y1)
        mask = None
        if masks is not None and i < len(masks.data):
            mask = _resize_mask(masks.data[i], H, W)
        dets.append((bbox, float(score), mask))
    return dets


def run_yolo(det_model, frame, conf_thr, iou_thr, device, imgsz):
    """
    Lightweight per-frame detection (no segmentation masks).
    Returns list of ((x, y, w, h), conf) for ALL classes.
    """
    result = det_model.predict(frame, imgsz=imgsz, conf=conf_thr, iou=iou_thr,
                               device=device, verbose=False)[0]
    boxes  = result.obb if det_model.task == "obb" else result.boxes
    dets   = []
    if boxes is not None and len(boxes) > 0:
        for xyxy, score in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = xyxy.tolist()
            dets.append(((x1, y1, x2-x1, y2-y1), float(score)))
    return dets


def tight_bbox_from_mask(mask_bool):
    """Return (x, y, w, h) tight bounding box of the True region."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return (x1, y1, x2 - x1, y2 - y1)


def mask_background(frame, mask_bool, fill_color=None):
    """
    Replace all pixels where mask_bool is False with fill_color.
    fill_color defaults to the mean colour of the foreground pixels.
    This removes background context from the SiamRPN template crop.
    """
    out = frame.copy()
    if fill_color is None:
        fg_pixels = frame[mask_bool]
        fill_color = tuple(int(v) for v in fg_pixels.mean(axis=0)) \
                     if len(fg_pixels) > 0 else (114, 114, 114)
    out[~mask_bool] = fill_color
    return out


def detection_at_click(dets, click_xy, H, W):
    """
    Return (det_idx, bbox, mask) for the detection whose mask contains click_xy.
    Falls back to nearest centroid if no mask data available.
    """
    cx, cy = click_xy
    # Prefer mask containment
    for idx, (bbox, _conf, mask) in enumerate(dets):
        if mask is not None and 0 <= cy < H and 0 <= cx < W:
            if mask[cy, cx]:
                return idx, bbox, mask
    # Fallback: nearest centroid
    best_idx  = None
    best_d2   = float("inf")
    for idx, (bbox, _conf, _mask) in enumerate(dets):
        bx, by, bw, bh = bbox
        d2 = (bx + bw/2 - cx)**2 + (by + bh/2 - cy)**2
        if d2 < best_d2:
            best_d2  = d2
            best_idx = idx
    if best_idx is not None:
        b, _, m = dets[best_idx]
        return best_idx, b, m
    return None, None, None


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


def put_label(img, text, x, y, fg=(0,0,0), bg=(0,255,0), scale=0.50, thick=1):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, max(0, y-th-bl)), (x+tw+4, y+bl), bg, -1)
    cv2.putText(img, text, (x+2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def draw_box_with_accents(img, bbox, color, thick, label=None):
    sx, sy, sw, sh = [int(v) for v in bbox]
    cv2.rectangle(img, (sx, sy), (sx+sw, sy+sh), color, thick)
    ca = min(16, sw//4, sh//4)
    for (px, py), (ddx, ddy) in [
        ((sx, sy),(1,1)), ((sx+sw, sy),(-1,1)),
        ((sx+sw, sy+sh),(-1,-1)), ((sx, sy+sh),(1,-1))
    ]:
        cv2.line(img, (px, py), (px+ddx*ca, py), color, thick)
        cv2.line(img, (px, py), (px, py+ddy*ca), color, thick)
    if label:
        put_label(img, label, sx, max(14, sy-4), fg=(0,0,0), bg=color)


def overlay_mask(img, mask_bool, color, alpha=0.35):
    """Semi-transparent colour overlay for a binary mask."""
    overlay = img.copy()
    overlay[mask_bool] = (
        np.array(img[mask_bool], dtype=float) * (1 - alpha) +
        np.array(color, dtype=float) * alpha
    ).astype(np.uint8)
    return overlay


# ══════════════════════════════════════════════════════════════════════════════
# Kalman filter  (constant-velocity, 6-state [cx,cy,w,h,vx,vy])
# ══════════════════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Single-target Kalman filter over (cx, cy, w, h) with constant velocity.
    No external library required – uses numpy only.
    """
    def __init__(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        # State: [cx, cy, w, h, vx, vy]
        self._x = np.array([x+w/2, y+h/2, float(w), float(h), 0., 0.], dtype=float)
        # Transition matrix (cx+=vx, cy+=vy each step)
        self._F = np.eye(6)
        self._F[0, 4] = 1.0
        self._F[1, 5] = 1.0
        # Measurement matrix – we observe [cx,cy,w,h]
        self._H      = np.zeros((4, 6))
        self._H[0,0] = self._H[1,1] = self._H[2,2] = self._H[3,3] = 1.0
        # Noise
        self._Q = np.diag([1., 1., 2., 2., 10., 10.])  # process
        self._R = np.diag([2., 2., 8., 8.])             # measurement
        self._P = np.eye(6) * 20.0

    def predict(self):
        """Advance state; return predicted (x,y,w,h) before measurement."""
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        cx, cy, w, h = self._x[:4]
        return (cx - w/2, cy - h/2, max(1, w), max(1, h))

    def update(self, bbox_xywh):
        """Correct state with measurement (x,y,w,h)."""
        x, y, w, h  = bbox_xywh
        z           = np.array([x+w/2, y+h/2, float(w), float(h)])
        S           = self._H @ self._P @ self._H.T + self._R
        K           = self._P @ self._H.T @ np.linalg.inv(S)
        self._x     = self._x + K @ (z - self._H @ self._x)
        self._P     = (np.eye(6) - K @ self._H) @ self._P

    def state_xywh(self):
        cx, cy, w, h = self._x[:4]
        return (cx - w/2, cy - h/2, max(1, w), max(1, h))

    def reset(self, bbox_xywh):
        self.__init__(bbox_xywh)


# ══════════════════════════════════════════════════════════════════════════════
# Anchor similarity scorer
# ══════════════════════════════════════════════════════════════════════════════

def score_anchor_similarity(tracker_model, frame, bbox_xywh, zf_ref):
    """
    Crop *bbox_xywh* from *frame*, run the SiamRPN backbone, and return the
    cosine similarity between the resulting feature map and *zf_ref*.
    Result is in [-1, 1]; higher = more similar to the anchor template.
    """
    x, y, w, h = [int(v) for v in bbox_xywh]
    fH, fW     = frame.shape[:2]
    x  = max(0, x);   y = max(0, y)
    w  = min(w, fW - x); h = min(h, fH - y)
    if w < 8 or h < 8:
        return 0.0
    crop = frame[y:y+h, x:x+w]
    # Resize to SiamRPN exemplar size (127 by default)
    esz  = cfg.TRACK.EXEMPLAR_SIZE
    crop = cv2.resize(crop, (esz, esz))
    tensor = torch.from_numpy(
        crop.transpose(2, 0, 1).astype(np.float32)
    ).unsqueeze(0) / 255.0
    device = next(tracker_model.parameters()).device
    tensor = tensor.to(device)
    with torch.no_grad():
        feat = tracker_model.backbone(tensor)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
    a = zf_ref.flatten().float()
    b = feat.flatten().float()
    sim = torch.dot(a, b) / (a.norm() * b.norm() + 1e-9)
    return float(sim.cpu())


# ══════════════════════════════════════════════════════════════════════════════
# Dual-template helpers  (same mechanics as siam_dual_template.py)
# ══════════════════════════════════════════════════════════════════════════════

def track_with_zf(tracker, frame, zf):
    saved_center = tracker.center_pos.copy()
    saved_size   = tracker.size.copy()
    saved_zf     = tracker.model.zf

    tracker.model.zf = zf
    outputs = tracker.track(frame)

    tracker.center_pos = saved_center
    tracker.size       = saved_size
    tracker.model.zf   = saved_zf
    return outputs


def apply_outputs(tracker, outputs, zf):
    bbox = outputs["bbox"]
    tracker.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                   bbox[1] + bbox[3] / 2])
    tracker.size       = np.array([bbox[2], bbox[3]])
    tracker.model.zf   = zf


# ══════════════════════════════════════════════════════════════════════════════
# SiamRPN loader
# ══════════════════════════════════════════════════════════════════════════════

def load_siam(config_name, weights_path):
    cfg_file = _EXPERIMENTS / config_name / "config.yaml"
    if not cfg_file.exists():
        avail = sorted(p.name for p in _EXPERIMENTS.iterdir() if p.is_dir())
        print(f"ERROR: pysot config not found: {cfg_file}\n  Available: {avail}",
              file=sys.stderr)
        sys.exit(1)
    if not weights_path:
        weights_path = str(_EXPERIMENTS / config_name / "model" / "model.pth")
    if not Path(weights_path).exists():
        print(f"""
ERROR: SiamRPN weights not found at:
  {weights_path}

Download (open in browser):
  siamrpn_alex_dwxcorr       → https://drive.google.com/open?id=1t62x56Jl7baUzPTo0QrC4jJnwvPZm-2m
  siamrpn_r50_l234_dwxcorr  → https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG
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


# ══════════════════════════════════════════════════════════════════════════════
# Click-to-select UI  (first frame)
# ══════════════════════════════════════════════════════════════════════════════

def click_select_person(frame, dets, win_name):
    """
    Show all detected persons as coloured overlays.
    User clicks on the one to track.
    Returns (bbox_xywh, mask_bool_or_None).
    Falls back to selectROI if no persons detected or user presses F.
    """
    H, W = frame.shape[:2]

    # Build overlay with all person masks
    vis = frame.copy()
    for i, (bbox, conf, mask) in enumerate(dets):
        col = MASK_COLORS[i % len(MASK_COLORS)]
        if mask is not None:
            vis = overlay_mask(vis, mask, col, alpha=0.40)
        # Draw bbox outline
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis, (x, y), (x+w, y+h), col, 1)
        put_label(vis, f"#{i}  {conf:.2f}", x, max(14, y-4),
                  fg=(0,0,0), bg=col, scale=0.45)

    instruction = "Click on person to track   |   F = drag fallback   |   Q = quit"
    cv2.putText(vis, instruction, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    result = {"done": False, "bbox": None, "mask": None, "fallback": False}

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            idx, bbox, mask = detection_at_click(dets, (x, y), H, W)
            if bbox is not None:
                result["bbox"] = bbox
                result["mask"] = mask
                result["done"] = True

    cv2.setMouseCallback(win_name, mouse_cb)
    cv2.imshow(win_name, vis)

    while not result["done"]:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            result["fallback"] = True
            result["done"]     = True

    cv2.setMouseCallback(win_name, lambda *a: None)

    if result["fallback"] or result["bbox"] is None:
        # Fallback: classic ROI drag
        hint = frame.copy()
        cv2.putText(hint, "Draw bbox around person, ENTER to confirm",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win_name, hint)
        try:
            rect = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
        except Exception:
            return None, None
        if rect[2] == 0 or rect[3] == 0:
            return None, None
        return tuple(map(float, rect)), None

    return result["bbox"], result["mask"]


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Segmentation-anchored dual-template SiamRPN tracker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video",   default=DEFAULT_VIDEO)
    p.add_argument("--config",  default=DEFAULT_CONFIG,
                   help="pysot experiment config name")
    p.add_argument("--weights", default="",
                   help="SiamRPN .pth path (auto-resolved)")
    p.add_argument("--seg",     default=DEFAULT_SEG,
                   help="YOLO segmentation model (.pt). "
                        "yolov8n-seg.pt is auto-downloaded on first use.")
    p.add_argument("--device",  default=DEFAULT_DEVICE)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--seg-conf",  type=float, default=0.30,
                   help="Segmentation confidence threshold (first frame only)")
    p.add_argument("--seg-iou",   type=float, default=0.45,
                   help="Segmentation NMS IoU threshold (first frame only)")
    p.add_argument("--yolo",      default=DEFAULT_YOLO,
                   help="YOLO detection model for per-frame drift correction")
    p.add_argument("--det-conf",  type=float, default=0.25,
                   help="Detection confidence threshold (per-frame YOLO)")
    p.add_argument("--det-iou",   type=float, default=0.45,
                   help="Detection NMS IoU threshold (per-frame YOLO)")

    # drift correction
    p.add_argument("--corr-interval", type=int,   default=10)
    p.add_argument("--corr-conf",     type=float, default=0.55,
                   help="Immediate correction if detection conf ≥ this")
    p.add_argument("--corr-iou",      type=float, default=0.35,
                   help="Immediate correction if IoU with track ≥ this")
    p.add_argument("--min-iou",       type=float, default=0.15)
    p.add_argument("--score-thr",     type=float, default=0.20,
                   help="Both template scores below this → lost")
    p.add_argument("--anchor-warn-ratio", type=float, default=0.5,
                   help="Raise ANCHOR DRIFT warning when score_a < ratio * score_y")
    p.add_argument("--drift-patience",    type=int,   default=2,
                   help="Consecutive drift frames before perimeter search activates")
    p.add_argument("--search-radius",     type=int,   default=150,
                   help="Pixel radius around Kalman prediction to search candidates")
    p.add_argument("--recover-thr",       type=float, default=0.99,
                   help="Min cosine similarity vs zf_anchor to accept a recovery candidate")

    # display / output
    p.add_argument("--show",      action="store_true")
    p.add_argument("--show-mask", action="store_true",
                   help="Overlay current seg mask on tracking display")
    p.add_argument("--show-dets", action="store_true",
                   help="Draw all per-frame person detections (dim grey)")
    p.add_argument("--show-kalman", action="store_true",
                   help="Draw Kalman predicted box each frame (dim yellow dashed)")
    p.add_argument("--save",      default="")
    p.add_argument("--thickness", type=int, default=1)
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
    print(f"  Source   : {src_label}  ({W}x{H} @ {fps_in:.1f} fps)")

    tracker  = load_siam(args.config, args.weights)

    print(f"  Seg YOLO : {args.seg}  device={args.device}  (loading…)")
    seg_model = YOLO(args.seg)
    seg_model.to(args.device)
    has_masks = (seg_model.task == "segment")
    print(f"  Seg task : {seg_model.task}  (masks={'yes' if has_masks else 'NO – bbox only'})")
    print(f"  Det YOLO : {Path(args.yolo).name}  device={args.device}  (loading…)")
    det_model = YOLO(args.yolo)
    det_model.to(args.device)
    print(f"  Det task : {det_model.task}  (per-frame drift correction)")
    print(f"  Correction interval : every {args.corr_interval} frames")
    print(f"  High-conf trigger   : conf≥{args.corr_conf}  IoU≥{args.corr_iou}")
    print("=" * 62)

    if not args.show:
        print("ERROR: --show is required for interactive person selection.",
              file=sys.stderr)
        return 1

    WIN = "Seg-Anchor Dual-Template SiamRPN"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # ── read first frame ──────────────────────────────────────────────────────
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: cannot read first frame", file=sys.stderr); return 1

    # ── segment first frame ───────────────────────────────────────────────────
    print("  Running segmentation on first frame…")
    dets_first = run_seg(seg_model, first_frame,
                         args.seg_conf, args.seg_iou, args.device, args.imgsz)
    print(f"  Found {len(dets_first)} person(s) on first frame.")

    # ── user selects person ───────────────────────────────────────────────────
    init_bbox, init_mask = click_select_person(first_frame, dets_first, WIN)
    if init_bbox is None:
        print("No target selected. Exiting."); return 0

    # If we have a mask, use the tighter mask-derived bbox
    if init_mask is not None:
        tight = tight_bbox_from_mask(init_mask)
        if tight is not None:
            print(f"  Click bbox : {[int(v) for v in init_bbox]}")
            print(f"  Mask tight : {list(tight)}  (using mask-derived bbox)")
            init_bbox = tight

    # ── mask background in first frame → clean template ──────────────────────
    if init_mask is not None:
        template_frame = mask_background(first_frame, init_mask)
        print("  Template background masked using seg mask.")
    else:
        template_frame = first_frame
        print("  No mask available – using raw frame as template.")

    # ── init SiamRPN → capture zf_anchor ─────────────────────────────────────
    tracker.init(template_frame, init_bbox)
    zf_anchor        = tracker.model.zf
    zf_yolo          = zf_anchor          # starts identical; updated by corrections
    current_box      = list(map(int, init_bbox))
    current_mask     = init_mask          # latest seg mask for display
    active_template  = "anchor"

    # ── state ─────────────────────────────────────────────────────────────────
    frames_since_corr = 0
    last_corr_box     = None
    last_corr_frame   = -1
    last_corr_type    = ""
    # Kalman + perimeter-search state
    kalman            = KalmanBoxTracker(current_box)
    drift_consec      = 0              # consecutive frames with anchor_drift flag
    last_recover_box  = None           # box used in last anchor recovery
    last_recover_frame = -1
    snap_n            = 0
    paused            = False
    frame_idx         = 1
    t0                = time.time()

    # show annotated first frame
    ann_first = first_frame.copy()
    if init_mask is not None and args.show_mask:
        ann_first = overlay_mask(ann_first, init_mask, COL_ANCHOR, alpha=0.30)
    ix, iy, iw, ih = map(int, init_bbox)
    cv2.rectangle(ann_first, (ix, iy), (ix+iw, iy+ih), COL_INIT, args.thickness)
    put_label(ann_first, "INIT (seg-masked)" if init_mask is not None else "INIT",
              ix, max(14, iy-4), bg=COL_INIT)
    if writer:
        writer.write(ann_first)
    cv2.imshow(WIN, ann_first)
    cv2.waitKey(1)

    def do_reinit(frame_img, win):
        """Helper: re-select person and reinitialise BOTH templates."""
        nonlocal zf_anchor, zf_yolo, current_box, current_mask
        nonlocal active_template, frames_since_corr, last_corr_box
        nonlocal drift_consec, last_recover_box, last_recover_frame

        dets_r = run_seg(seg_model, frame_img,
                         args.seg_conf, args.seg_iou, args.device, args.imgsz)
        new_bbox, new_mask = click_select_person(frame_img, dets_r, win)
        if new_bbox is None:
            return False

        if new_mask is not None:
            tight = tight_bbox_from_mask(new_mask)
            if tight:
                new_bbox = tight

        tpl_frame = mask_background(frame_img, new_mask) \
                    if new_mask is not None else frame_img

        tracker.init(tpl_frame, new_bbox)
        zf_anchor        = tracker.model.zf
        zf_yolo          = zf_anchor
        current_box      = list(map(int, new_bbox))
        current_mask     = new_mask
        active_template  = "anchor"
        frames_since_corr = 0
        last_corr_box    = None
        kalman.reset(current_box)
        drift_consec     = 0
        last_recover_box = None
        print(f"  Re-init BOTH templates → {[int(v) for v in new_bbox]}"
              f"  mask={'yes' if new_mask is not None else 'no'}")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Tracking loop
    # ══════════════════════════════════════════════════════════════════════════
    annotated = ann_first   # ensure variable exists for pause-snapshot

    while True:
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
                do_reinit(frame, WIN)
                paused = False
            continue

        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame.copy()

        # ── Kalman prediction (before SiamRPN, uses velocity from last frame) ─
        kalman_pred = kalman.predict()   # (x, y, w, h) of expected location

        # ── dual-template tracking ────────────────────────────────────────────
        out_a = track_with_zf(tracker, frame, zf_anchor)
        out_y = track_with_zf(tracker, frame, zf_yolo)

        score_a = float(out_a.get("best_score", 0))
        score_y = float(out_y.get("best_score", 0))

        if score_a >= score_y:
            winner_out  = out_a
            winner_zf   = zf_anchor
            active_template = "anchor"
        else:
            winner_out  = out_y
            winner_zf   = zf_yolo
            active_template = "yolo"

        apply_outputs(tracker, winner_out, winner_zf)
        current_box = list(map(int, winner_out["bbox"]))
        best_score  = max(score_a, score_y)
        is_lost     = best_score < args.score_thr

        # Update Kalman with the chosen tracking box
        kalman.update(current_box)

        # ── detection (per-frame, lightweight model) ──────────────────────────
        dets = run_yolo(det_model, frame,
                        args.det_conf, args.det_iou, args.device, args.imgsz)

        # ── drift correction → update zf_yolo only ───────────────────────────
        corrected = False
        corr_type = ""

        if not is_lost and dets:
            best_box, best_conf, best_ov = best_matching_detection(
                current_box, dets, min_iou=args.min_iou)

            trigger_hi = (best_box is not None
                          and best_conf >= args.corr_conf
                          and best_ov  >= args.corr_iou)
            trigger_per = (not trigger_hi
                           and frames_since_corr >= args.corr_interval
                           and best_box is not None)

            if trigger_hi or trigger_per:
                corrected  = True
                corr_type  = "high-conf" if trigger_hi else "periodic"

                # det model has no mask – use raw frame for zf_yolo
                # Extract new zf_yolo without disturbing tracker position
                saved_center = tracker.center_pos.copy()
                saved_size   = tracker.size.copy()
                saved_zf     = tracker.model.zf

                tracker.init(frame, (best_box[0], best_box[1],
                                     best_box[2], best_box[3]))
                zf_yolo = tracker.model.zf

                # Restore position
                tracker.center_pos = saved_center
                tracker.size       = saved_size
                tracker.model.zf   = saved_zf

                last_corr_box   = list(map(int, best_box))
                last_corr_frame = frame_idx
                last_corr_type  = corr_type
                frames_since_corr = 0
                print(f"  [frame {frame_idx:4d}] zf_yolo updated ({corr_type})  "
                      f"IoU={best_ov:.2f}  conf={best_conf:.2f}")

        if not corrected:
            frames_since_corr += 1

        # ── anchor drift flag + consecutive counter ───────────────────────────
        anchor_drift = (not is_lost
                        and score_y > 0
                        and score_a < args.anchor_warn_ratio * score_y)
        drift_consec = (drift_consec + 1) if anchor_drift else 0

        # ── perimeter search (fires when drift persists ≥ drift_patience) ─────
        recovered = False
        if drift_consec >= args.drift_patience and dets:
            # Use Kalman predicted centre as search origin – it is independent
            # of SiamRPN which may already be at the wrong location.
            kx, ky, kw, kh = kalman_pred
            kcx = kx + kw / 2
            kcy = ky + kh / 2
            candidates = [
                (bbox_c, conf_c)
                for (bbox_c, conf_c) in dets
                if np.hypot(bbox_c[0] + bbox_c[2]/2 - kcx,
                            bbox_c[1] + bbox_c[3]/2 - kcy) <= args.search_radius
            ]
            best_sim  = -1.0
            best_cand = None
            for (bbox_c, _) in candidates:
                sim = score_anchor_similarity(
                    tracker.model, frame, bbox_c, zf_anchor)
                if sim > best_sim:
                    best_sim  = sim
                    best_cand = bbox_c

            if best_cand is not None and best_sim >= args.recover_thr:
                # Re-init tracker at the recovered location
                tracker.init(frame, (best_cand[0], best_cand[1],
                                     best_cand[2], best_cand[3]))
                zf_yolo            = tracker.model.zf   # update yolo template
                current_box        = list(map(int, best_cand))
                kalman.update(current_box)              # snap Kalman to truth
                last_recover_box   = current_box[:]
                last_recover_frame = frame_idx
                drift_consec       = 0
                recovered          = True
                print(f"  [frame {frame_idx:4d}] PERIMETER RECOVERY "
                      f"sim={best_sim:.3f}  box={current_box}  "
                      f"searched {len(candidates)} candidate(s)")
            else:
                n_cands = len(candidates)
                best_s  = f"{best_sim:.3f}" if best_cand else "–"
                print(f"  [frame {frame_idx:4d}] perimeter search: "
                      f"{n_cands} candidates, best_sim={best_s}  (thr={args.recover_thr})")

        # ══════════════════════════════════════════════════════════════════════
        # Annotation
        # ══════════════════════════════════════════════════════════════════════

        # 1. current seg mask overlay
        if args.show_mask and current_mask is not None:
            col = COL_ANCHOR if active_template == "anchor" else COL_YOLO_T
            annotated = overlay_mask(annotated, current_mask, col, alpha=0.25)

        # 2. background person detections (optional)
        if args.show_dets:
            for (bx, by, bw, bh), _ in dets:
                cv2.rectangle(annotated,
                              (int(bx), int(by)), (int(bx+bw), int(by+bh)),
                              COL_DET_DIM, 1)

        # 2b. Kalman predicted box (optional, very subtle)
        if args.show_kalman:
            kx, ky, kw, kh = [int(v) for v in kalman_pred]
            draw_dashed_rect(annotated, kx, ky, kx+kw, ky+kh, COL_KALMAN, 1, dash=6)

        # 2c. perimeter search radius circle (while drift is active)
        if drift_consec >= args.drift_patience:
            kx, ky, kw, kh = [int(v) for v in kalman_pred]
            kcx, kcy = kx + kw//2, ky + kh//2
            cv2.circle(annotated, (kcx, kcy), args.search_radius, (100, 0, 180), 1)
            put_label(annotated, f"searching… {drift_consec}f",
                      kcx - args.search_radius, max(14, kcy - args.search_radius - 4),
                      fg=(255,255,255), bg=(100, 0, 180), scale=0.40)

        # 2d. last recovery box (magenta dashed, fades after 8 frames)
        if last_recover_box and (frame_idx - last_recover_frame) <= 8:
            rx, ry, rw, rh = last_recover_box
            draw_dashed_rect(annotated, rx, ry, rx+rw, ry+rh, COL_RECOVER, 1)
            put_label(annotated, "RECOVERED",
                      rx, max(14, ry-4), fg=(0,0,0), bg=COL_RECOVER, scale=0.40)

        # 3. last correction box (dashed, fades after 5 frames)
        if last_corr_box and (frame_idx - last_corr_frame) <= 5:
            lx, ly, lw, lh = last_corr_box
            draw_dashed_rect(annotated, lx, ly, lx+lw, ly+lh, COL_CORR, 1)
            put_label(annotated, f"DET ({last_corr_type})",
                      lx, max(14, ly-4), fg=(0,0,0), bg=COL_CORR, scale=0.40)

        # 4. main track box
        sx, sy, sw, sh = current_box
        if is_lost:
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, COL_LOST, args.thickness)
            cv2.putText(annotated, f"LOST a={score_a:.2f} y={score_y:.2f}",
                        (sx+2, sy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_LOST, 1,
                        cv2.LINE_AA)
        else:
            col = COL_ANCHOR if active_template == "anchor" else COL_YOLO_T
            lbl = f"A {score_a:.2f}" if active_template == "anchor" \
                  else f"Y {score_y:.2f}"
            draw_box_with_accents(annotated, current_box, col, args.thickness, label=lbl)

        # 5. dual score bars (top-right)
        # anchor_drift already computed above (perimeter search section)
        bx0   = W - 155
        by0   = 10
        bw_bar = 130
        bh_bar = 13
        # anchor bar  (highlight red border if drifting)
        bar_border = (0, 0, 200) if anchor_drift else (40, 40, 40)
        cv2.rectangle(annotated, (bx0-1, by0-1), (bx0+bw_bar+1, by0+bh_bar+1), bar_border, 1)
        cv2.rectangle(annotated, (bx0, by0), (bx0+bw_bar, by0+bh_bar), (40,40,40), -1)
        cv2.rectangle(annotated, (bx0, by0),
                      (bx0+int(bw_bar*min(1,score_a)), by0+bh_bar), COL_ANCHOR, -1)
        cv2.putText(annotated, f"A {score_a:.2f}", (bx0+2, by0+bh_bar-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)
        by1 = by0 + bh_bar + 3
        cv2.rectangle(annotated, (bx0, by1), (bx0+bw_bar, by1+bh_bar), (40,40,40), -1)
        cv2.rectangle(annotated, (bx0, by1),
                      (bx0+int(bw_bar*min(1,score_y)), by1+bh_bar), COL_YOLO_T, -1)
        cv2.putText(annotated, f"Y {score_y:.2f}", (bx0+2, by1+bh_bar-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,0), 1, cv2.LINE_AA)

        # ANCHOR DRIFT warning banner (below score bars)
        if anchor_drift:
            warn_x = bx0
            warn_y = by1 + bh_bar + 6
            warn_text = f"ANCHOR DRIFT  a/y={score_a/max(score_y,1e-9):.2f}"
            (tw, th), bl = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (warn_x-2, warn_y-th-2),
                          (warn_x+tw+4, warn_y+bl+2), (0, 0, 180), -1)
            cv2.putText(annotated, warn_text, (warn_x+2, warn_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 6. correction countdown bar
        prog = min(1.0, frames_since_corr / max(1, args.corr_interval))
        bww  = W - 20
        cv2.rectangle(annotated, (10, H-22), (10+bww, H-22+6), (60,60,60), -1)
        cv2.rectangle(annotated, (10, H-22),
                      (10+int(bww*prog), H-22+6),
                      COL_CORR if prog >= 1.0 else (0, 180, 100), -1)
        cv2.putText(annotated,
                    f"zf_yolo update in {max(0, args.corr_interval-frames_since_corr)} frames",
                    (10, H-26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1, cv2.LINE_AA)

        # 7. HUD
        elapsed = max(time.time() - t0, 1e-9)
        fr_str  = str(frame_idx) if is_live else f"{frame_idx}/{n_frames}"
        mask_str = "mask" if (current_mask is not None) else "no-mask"
        cv2.putText(annotated,
                    f"FPS {frame_idx/elapsed:.1f}  frame {fr_str}  "
                    f"tmpl={active_template}  {mask_str}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1, cv2.LINE_AA)
        cv2.putText(annotated, "R=reselect  SPACE=pause  S=snap  Q=quit",
                    (10, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100,100,100), 1, cv2.LINE_AA)

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
                do_reinit(frame, WIN)

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
