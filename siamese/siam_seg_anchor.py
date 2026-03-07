#!/usr/bin/env python3
"""
Segmentation-anchored Dual-Template SiamRPN tracker.

Key idea
--------
A YOLO segmentation model (yolov8n-seg by default) provides a per-pixel
person mask.  That mask is used to CLEAN the template crop before SiamRPN
sees it: background pixels are replaced with the mean foreground colour so
the network learns only the person's appearance, not whatever is behind them.

Template banks
--------------
  anchor_bank – 3-4 permanent reference templates captured at different times.
                The first is from the segmentation-masked initial selection.
                Additional ones are auto-captured when tracking is confident
                (high SiamRPN score + ReID confirmation).  Never evicted.
  yolo_bank   – rolling FIFO of recent YOLO-correction templates.
                Updated whenever the seg/YOLO correction trigger fires.

Crowd robustness
----------------
  - Kalman-proximity weighted scoring: templates predicting positions
    closer to the Kalman prediction get a score bonus.
  - Centroid jump rejection: if the winner jumps too far from Kalman
    prediction, fall back to the closest-to-Kalman template.
  - Size-consistency guard: penalises templates predicting boxes whose
    size deviates too much from the recent median.

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

torch.set_num_threads(11)

# ── paths / constants ─────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent
_EXPERIMENTS = _PYSOT_ROOT / "experiments"
_WEIGHTS_ROOT  = _HERE.parent / "resources" / "weights"
DEFAULT_VIDEO  = str(_HERE.parent / "resources" / "video_test" / "nadir_ped_crossing_crop640.mp4")
DEFAULT_CONFIG = "siamrpn_r50_l234_dwxcorr"
DEFAULT_SEG    = str(_HERE.parent / "resources" / "models" / "yolov8n-seg.pt")
DEFAULT_YOLO   = str(_HERE.parent / "resources" / "models" / "yolov26nobbnew_merged_1024.pt")
# DEFAULT_DEVICE = "cuda" if torch.backends.mps.is_available() else "cpu"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
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


def run_seg(seg_model, frame, conf_thr, iou_thr, device, imgsz, half=False):
    """
    Run YOLO-seg on *frame*.  Returns list of (bbox_xywh, conf, mask_HW_bool).
    mask_HW_bool is None when the model is a plain detection model (no masks).
    Only PERSON_CLASS detections are returned.
    """
    H, W = frame.shape[:2]
    result = seg_model.predict(frame, imgsz=imgsz, conf=conf_thr, iou=iou_thr,
                               device=device, half=half, verbose=False)[0]
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


def run_yolo(det_model, frame, conf_thr, iou_thr, device, imgsz, half=False):
    """
    Lightweight per-frame detection (no segmentation masks).
    Returns list of ((x, y, w, h), conf) for ALL classes.
    """
    result = det_model.predict(frame, imgsz=imgsz, conf=conf_thr, iou=iou_thr,
                               device=device, half=half, verbose=False)[0]
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


def crop_masked_template(frame, mask_bool, bbox_xywh, margin=0.15):
    """
    Crop tightly around the mask, apply mask, then pad to a square.

    Unlike mask_background (which operates on the full frame),
    this crops FIRST — so when SiamRPN adds context padding,
    it samples from the masked crop instead of distant background.
    The target fills a larger portion of the 127×127 template.

    margin: fractional padding around tight mask bounds (0.15 = 15%).
    Returns (cropped_image, adjusted_bbox_xywh).
    """
    H, W = frame.shape[:2]
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        # No mask pixels — fall back to full-frame mask
        return mask_background(frame, mask_bool), bbox_xywh

    # Tight mask bounds
    mx1, my1 = int(xs.min()), int(ys.min())
    mx2, my2 = int(xs.max()), int(ys.max())
    mw, mh = mx2 - mx1, my2 - my1

    # Add margin and make square (SiamRPN expects roughly square context)
    pad_w = int(mw * margin)
    pad_h = int(mh * margin)
    cx, cy = (mx1 + mx2) // 2, (my1 + my2) // 2
    half_side = max(mw + 2 * pad_w, mh + 2 * pad_h) // 2

    # Crop bounds (clamp to frame)
    c_x1 = max(0, cx - half_side)
    c_y1 = max(0, cy - half_side)
    c_x2 = min(W, cx + half_side)
    c_y2 = min(H, cy + half_side)

    # Crop frame and mask
    crop = frame[c_y1:c_y2, c_x1:c_x2].copy()
    crop_mask = mask_bool[c_y1:c_y2, c_x1:c_x2]

    # Compute fill colour from foreground pixels
    fg = crop[crop_mask]
    fill = tuple(int(v) for v in fg.mean(axis=0)) if len(fg) > 0 else (114, 114, 114)
    crop[~crop_mask] = fill

    # Adjust bbox to crop coordinates
    bx, by, bw, bh = bbox_xywh
    new_bbox = (bx - c_x1, by - c_y1, bw, bh)

    return crop, new_bbox


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
    color_layer = np.full_like(img, color, dtype=np.uint8)
    blended = cv2.addWeighted(img, 1 - alpha, color_layer, alpha, 0)
    out = img.copy()
    out[mask_bool] = blended[mask_bool]
    return out


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
# Velocity-direction scorer  (uses Kalman velocity for trajectory gating)
# ══════════════════════════════════════════════════════════════════════════════

def velocity_direction_score(kalman_tracker, candidate_bbox):
    """
    Cosine of angle between Kalman velocity vector and displacement from
    current Kalman position to the candidate bbox centre.

    Returns 0-1 (1 = same direction, 0 = opposite).
    Returns 1.0 if the object is nearly stationary (< 2 px/frame).
    """
    vx, vy = kalman_tracker._x[4], kalman_tracker._x[5]
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 2.0:
        return 1.0   # stationary → no directional constraint

    # Kalman current centre
    kcx, kcy = kalman_tracker._x[0], kalman_tracker._x[1]
    # Candidate centre
    cx = candidate_bbox[0] + candidate_bbox[2] / 2
    cy = candidate_bbox[1] + candidate_bbox[3] / 2
    dx, dy = cx - kcx, cy - kcy
    disp = np.sqrt(dx**2 + dy**2)
    if disp < 1.0:
        return 1.0   # nearly overlapping → accept

    cos_angle = (vx * dx + vy * dy) / (speed * disp)
    return float(np.clip((cos_angle + 1.0) / 2.0, 0.0, 1.0))  # map [-1,1] → [0,1]


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


def score_anchor_similarity(model, frame, bbox_xywh, zf_anchor):
    """Cosine similarity between a candidate crop's feature and the anchor template.
    Used by main_v1 for recovery search."""
    import torch.nn.functional as F
    x, y, w, h = [int(v) for v in bbox_xywh]
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE))
    crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).unsqueeze(0).float().to(
        next(model.parameters()).device)
    with torch.no_grad():
        zf_cand = model.backbone(crop_tensor)
        if cfg.ADJUST.ADJUST:
            zf_cand = model.neck(zf_cand)
    a = zf_anchor[-1] if isinstance(zf_anchor, (list, tuple)) else zf_anchor
    c = zf_cand[-1] if isinstance(zf_cand, (list, tuple)) else zf_cand
    a_flat = a.flatten()
    c_flat = c.flatten()
    return float(F.cosine_similarity(a_flat.unsqueeze(0), c_flat.unsqueeze(0)))


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
        weights_path = str(_WEIGHTS_ROOT / config_name / "model" / "model.pth")
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
    p.add_argument("--source", "--video", dest="source", default=DEFAULT_VIDEO)
    p.add_argument("--config",  default=DEFAULT_CONFIG,
                   help="pysot experiment config name")
    p.add_argument("--weights", default="",
                   help="SiamRPN .pth path (auto-resolved)")
    p.add_argument("--seg",     default=DEFAULT_SEG,
                   help="YOLO segmentation model (.pt). "
                        "yolov8n-seg.pt is auto-downloaded on first use.")
    p.add_argument("--device",  default=DEFAULT_DEVICE)
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--half",    action="store_true", default=True,
                   help="Use FP16 half-precision for YOLO models (faster on GPU)")
    p.add_argument("--no-half", dest="half", action="store_false",
                   help="Disable FP16 half-precision")
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
    p.add_argument("--det-interval", type=int, default=3,
                   help="Run YOLO detection every N frames (1=every frame, 3=default)")

    # drift correction
    p.add_argument("--corr-interval", type=int,   default=10)
    p.add_argument("--corr-conf",     type=float, default=0.55,
                   help="Immediate correction if detection conf ≥ this")
    p.add_argument("--corr-iou",      type=float, default=0.35,
                   help="Immediate correction if IoU with track ≥ this")
    p.add_argument("--min-iou",       type=float, default=0.15)
    p.add_argument("--score-thr",     type=float, default=0.12,
                   help="Final LOST threshold (below this → lost)")
    p.add_argument("--tracked-thr",   type=float, default=0.40,
                   help="Score above this → fully tracked (corrections allowed)")
    p.add_argument("--partial-thr",   type=float, default=0.25,
                   help="Score above this → partial occlusion (freeze templates)")
    p.add_argument("--occ-patience",  type=int,   default=30,
                   help="Frames of full occlusion before declaring LOST")
    p.add_argument("--anchor-warn-ratio", type=float, default=0.5,
                   help="Raise ANCHOR DRIFT warning when score_a < ratio * score_y")
    p.add_argument("--drift-patience",    type=int,   default=2,
                   help="Consecutive drift frames before perimeter search activates")
    p.add_argument("--search-radius",     type=int,   default=150,
                   help="Pixel radius around Kalman prediction to search candidates")
    p.add_argument("--recover-thr",       type=float, default=0.30,
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

    # ── accuracy improvements (v2) ───────────────────────────────────────
    p.add_argument("--reid-thr",   type=float, default=0.35,
                   help="ReID cosine similarity gate for YOLO corrections")
    p.add_argument("--bank-size",  type=int,   default=3,
                   help="Yolo template bank size (number of recent templates)")
    p.add_argument("--window-influence", type=float, default=0.0,
                   help="Override cfg WINDOW_INFLUENCE (0=use config default, "
                        "higher = stickier, resists drift)")
    p.add_argument("--penalty-k",        type=float, default=0.0,
                   help="Override cfg PENALTY_K (0=use config default)")
    p.add_argument("--lr-scale",         type=float, default=1.0,
                   help="Multiply cfg.TRACK.LR by this (1.0=no change, "
                        "lower = slower template update)")
    p.add_argument("--kalman-search-blend", type=float, default=0.0,
                   help="Blend search centre toward Kalman prediction each frame "
                        "(0=disabled, higher = more motion-driven)")
    p.add_argument("--seg-refine-interval", type=int, default=15,
                   help="Run seg model every N frames to snap box to person mask (0=disabled)")

    # ── crowd robustness (v3) ────────────────────────────────────────
    p.add_argument("--anchor-bank-size", type=int, default=4,
                   help="Number of permanent anchor templates to maintain")
    p.add_argument("--anchor-capture-interval", type=int, default=30,
                   help="Min frames between anchor template captures")
    p.add_argument("--anchor-capture-score", type=float, default=0.45,
                   help="Min SiamRPN score to capture a new permanent anchor")
    p.add_argument("--max-jump-factor", type=float, default=2.5,
                   help="Max jump distance as factor of box diagonal (0=disabled)")
    p.add_argument("--kalman-weight", type=float, default=0.15,
                   help="Weight for Kalman proximity bonus in template scoring")
    p.add_argument("--size-guard-ratio", type=float, default=0.5,
                   help="Reject tracking result if size deviates by more than this ratio from recent median")
    p.add_argument("--max-size-change", type=float, default=0.15,
                   help="Max per-frame size change as fraction of median size "
                        "(0=disabled, 0.15=default, prevents box growing to cover two objects)")

    # ── multi-cue gate (v4 – crowd ID-switch fix) ────────────────────
    p.add_argument("--reid-cls-model", default="",
                   help="Path to YOLO-cls model for embedding cue (empty = color-only mode)")
    p.add_argument("--color-weight",   type=float, default=0.55,
                   help="Weight for HSV color histogram cue")
    p.add_argument("--cls-weight",     type=float, default=0.25,
                   help="Weight for YOLO-cls embedding cue")
    p.add_argument("--velocity-weight", type=float, default=0.0,
                   help="Reserved (velocity gating is hard-coded threshold)")
    p.add_argument("--hist-bins",      type=int, nargs=3, default=[16, 16, 16],
                   help="HSV histogram resolution (H S V)")
    p.add_argument("--neg-bank-size",  type=int, default=8,
                   help="Negative exemplar FIFO size")
    p.add_argument("--crowd-thr",      type=int, default=3,
                   help="Nearby detection count to trigger crowd-adaptive mode")
    p.add_argument("--crowd-lockdown-thr", type=int, default=5,
                   help="Nearby detection count to FREEZE all corrections "
                        "(dense crowd = maximum tracker stickiness)")

    # ── YOLO-primary mode (v5 – nadir/drone tracking) ────────────────
    p.add_argument("--yolo-primary", action="store_true",
                   help="YOLO+Kalman drives position, SiamRPN is fallback only. "
                        "Auto-enabled when target bbox area < 4000 px².")
    p.add_argument("--yolo-blend", type=float, default=0.70,
                   help="In YOLO-primary mode, blend weight toward YOLO detection "
                        "(0.7 = 70%% YOLO, 30%% SiamRPN)")
    p.add_argument("--yolo-accept-radius", type=float, default=1.5,
                   help="Accept YOLO detection if within this many box-diagonals "
                        "of Kalman prediction")

    # ── nadir / scale-lock ─────────────────────────────────────────────
    p.add_argument("--scale-lock", action="store_true", default=False,
                   help="Lock box size after warmup (nadir view: people don't change "
                        "size). Prevents absorbing neighbours. Auto-enabled by --crowd.")
    p.add_argument("--no-scale-lock", dest="scale_lock", action="store_false")
    p.add_argument("--crowd-kalman-boost", type=float, default=0.55,
                   help="Kalman search blend when nearby_count >= crowd-thr "
                        "(0=disabled, higher = more motion-driven in crowds)")

    # ── preset modes ──────────────────────────────────────────────────
    p.add_argument("--crowd", action="store_true",
                   help="Dense crowd preset: cranks up anti-drift parameters "
                        "(window-influence=0.70, penalty-k=0.16, lr-scale=0.3, "
                        "kalman-search-blend=0.20, max-jump-factor=1.5, "
                        "max-size-change=0.20, scale-lock). "
                        "Use for nadir/top-down dense crowds "
                        "where everyone looks alike.")

    args = p.parse_args()

    # Apply --crowd preset (only overrides values left at their defaults)
    if args.crowd:
        if args.window_influence == 0.0:
            args.window_influence = 0.70
        if args.penalty_k == 0.0:
            args.penalty_k = 0.16
        if args.lr_scale == 1.0:
            args.lr_scale = 0.3
        if args.kalman_search_blend == 0.0:
            args.kalman_search_blend = 0.20
        if args.max_jump_factor == 2.5:
            args.max_jump_factor = 1.5
        if args.max_size_change == 0.15:
            args.max_size_change = 0.20
        if args.det_interval == 3:
            args.det_interval = 1  # crowd needs every-frame detection
        if not args.scale_lock:
            args.scale_lock = True  # nadir crowds: people don't change size

    return args


# ══════════════════════════════════════════════════════════════════════════════
# ORIGINAL main – to use, change sys.exit(main()) → sys.exit(main_v1())
# ══════════════════════════════════════════════════════════════════════════════

def main_v1():
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

        # ── detection (run every N frames, not every frame) ──────────────────
        if frame_idx % 5 == 0:
            dets = run_yolo(det_model, frame,
                            args.det_conf, args.det_iou, args.device, args.imgsz)
        else:
            dets = []

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


class MultiCueGate:
    """
    Multi-cue appearance gate for YOLO corrections.  Replaces ReIDGate.

    Cues fused (weighted sum):
      - HSV color histogram  (Bhattacharyya distance via cv2.compareHist)
      - YOLO-cls embedding   (cosine similarity, optional)
      - Negative exemplar penalty (max sim to known non-targets subtracted)

    Key design choices:
      - Gated EMA: only updates EMA when new histogram matches the REFERENCE
        (not the drifting EMA itself), preventing positive-feedback drift.
      - Negative exemplar bank: FIFO of nearby non-target histograms.
    """

    def __init__(self, frame, bbox_xywh, mask=None, *,
                 cls_model=None, hist_bins=(16, 16, 16),
                 color_weight=0.55, cls_weight=0.25,
                 neg_bank_size=8, ema_alpha=0.85):
        self._hist_bins = list(hist_bins)
        self._ema_alpha = ema_alpha
        self._cls_model = cls_model
        self._neg_bank_size = neg_bank_size

        # Redistribute weights if no cls model
        if cls_model is None:
            self._w_color = color_weight + cls_weight
            self._w_cls   = 0.0
        else:
            self._w_color = color_weight
            self._w_cls   = cls_weight
        self._w_neg = 0.15   # subtracted penalty (was 0.30, too aggressive)

        # Reference features (from initial selection)
        self._ref_hist = self._extract_hist(frame, bbox_xywh, mask)
        self._ema_hist = self._ref_hist.copy()
        self._ref_cls  = self._extract_cls(frame, bbox_xywh) if cls_model else None
        self._ema_cls  = self._ref_cls.copy() if self._ref_cls is not None else None

        # Negative exemplar bank (FIFO)
        self._neg_bank = []   # list of HSV histograms

    def _crop(self, frame, bbox_xywh, mask=None):
        x, y, w, h = [int(v) for v in bbox_xywh]
        fH, fW = frame.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, fW - x); h = min(h, fH - y)
        if w < 8 or h < 8:
            return None
        crop = frame[y:y+h, x:x+w].copy()
        if mask is not None:
            mc = mask[y:y+h, x:x+w]
            if mc.shape == crop.shape[:2]:
                crop[~mc] = 0  # zero out background for histogram
        return crop

    def _extract_hist(self, frame, bbox_xywh, mask=None):
        crop = self._crop(frame, bbox_xywh, mask)
        if crop is None:
            h = np.zeros(self._hist_bins[0] * self._hist_bins[1] * self._hist_bins[2],
                         dtype=np.float32)
            return h
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, self._hist_bins,
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def _extract_cls(self, frame, bbox_xywh):
        if self._cls_model is None:
            return None
        x, y, w, h = [int(v) for v in bbox_xywh]
        fH, fW = frame.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, fW - x); h = min(h, fH - y)
        if w < 8 or h < 8:
            return np.zeros(128, dtype=np.float32)
        crop = frame[y:y+h, x:x+w]
        results = self._cls_model.predict(crop, embed=[-2], verbose=False)
        if results and len(results) > 0:
            embed = results[0]
            if hasattr(embed, 'cpu'):
                embed = embed.cpu().numpy()
            return np.array(embed, dtype=np.float32).flatten()
        return np.zeros(128, dtype=np.float32)

    @staticmethod
    def _hist_sim(a, b):
        """Bhattacharyya-based similarity in [0, 1]."""
        a_cv = a.reshape(-1).astype(np.float32)
        b_cv = b.reshape(-1).astype(np.float32)
        # HISTCMP_BHATTACHARYYA returns distance in [0,1]; convert to similarity
        dist = cv2.compareHist(a_cv, b_cv, cv2.HISTCMP_BHATTACHARYYA)
        return float(1.0 - dist)

    @staticmethod
    def _cosine_sim(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def similarity(self, frame, bbox_xywh, mask=None):
        """
        Multi-cue similarity score for a candidate detection.
        Returns float in roughly [0, 1] (can go slightly negative with penalty).
        """
        cand_hist = self._extract_hist(frame, bbox_xywh, mask)

        # Color histogram: max of ref and EMA
        sim_ref = self._hist_sim(self._ref_hist, cand_hist)
        sim_ema = self._hist_sim(self._ema_hist, cand_hist)
        color_score = max(sim_ref, sim_ema)

        score = self._w_color * color_score

        # YOLO-cls embedding (optional)
        if self._cls_model is not None and self._w_cls > 0:
            cand_cls = self._extract_cls(frame, bbox_xywh)
            if cand_cls is not None and self._ref_cls is not None:
                sim_cls_ref = self._cosine_sim(self._ref_cls, cand_cls)
                sim_cls_ema = self._cosine_sim(self._ema_cls, cand_cls) \
                              if self._ema_cls is not None else sim_cls_ref
                cls_score = max(sim_cls_ref, sim_cls_ema)
                score += self._w_cls * cls_score

        # Negative exemplar penalty
        if self._neg_bank:
            max_neg_sim = max(self._hist_sim(neg_h, cand_hist)
                              for neg_h in self._neg_bank)
            score -= self._w_neg * max_neg_sim

        return score

    def update(self, frame, bbox_xywh, mask=None):
        """
        Gated EMA update: only blends when new observation matches the
        REFERENCE histogram (not the EMA), preventing drift-toward-wrong-person.
        """
        new_hist = self._extract_hist(frame, bbox_xywh, mask)
        ref_sim = self._hist_sim(self._ref_hist, new_hist)

        # Only update EMA if sufficiently similar to the original reference
        if ref_sim >= 0.5:
            self._ema_hist = (self._ema_alpha * self._ema_hist
                              + (1 - self._ema_alpha) * new_hist)
            if self._cls_model is not None:
                new_cls = self._extract_cls(frame, bbox_xywh)
                if new_cls is not None and self._ema_cls is not None:
                    self._ema_cls = (self._ema_alpha * self._ema_cls
                                     + (1 - self._ema_alpha) * new_cls)

    def add_negative(self, frame, bbox_xywh, mask=None):
        """Add a nearby non-target's histogram to the negative bank."""
        neg_hist = self._extract_hist(frame, bbox_xywh, mask)
        self._neg_bank.append(neg_hist)
        if len(self._neg_bank) > self._neg_bank_size:
            self._neg_bank.pop(0)

    def reset(self, frame, bbox_xywh, mask=None):
        """Full reset: new reference, clear EMA and negative bank."""
        self._ref_hist = self._extract_hist(frame, bbox_xywh, mask)
        self._ema_hist = self._ref_hist.copy()
        if self._cls_model is not None:
            self._ref_cls = self._extract_cls(frame, bbox_xywh)
            self._ema_cls = self._ref_cls.copy() if self._ref_cls is not None else None
        self._neg_bank = []


# ══════════════════════════════════════════════════════════════════════════════
# Efficient multi-template tracking  (1 backbone pass, N rpn-head passes)
# ══════════════════════════════════════════════════════════════════════════════

def track_multi_zf(tracker, frame, zf_list):
    """
    Run SiamRPN tracking with multiple template features efficiently.
    Backbone + neck evaluated ONCE; only the lightweight RPN head runs
    per-template.

    Returns: list of (outputs_dict, zf) sorted by score descending.
    Tracker state is NOT modified – caller should apply_outputs for winner.
    """
    saved_center = tracker.center_pos.copy()
    saved_size   = tracker.size.copy()
    saved_zf     = tracker.model.zf

    # ── compute search crop (mirrors SiamRPNTracker.track preamble) ──────
    w_z = saved_size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(saved_size)
    h_z = saved_size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(saved_size)
    s_z = np.sqrt(w_z * h_z)
    scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
    s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
    x_crop = tracker.get_subwindow(frame, saved_center,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), tracker.channel_average)

    # ── backbone + neck ONCE ─────────────────────────────────────────────
    with torch.no_grad():
        xf = tracker.model.backbone(x_crop)
        if cfg.MASK.MASK:
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = tracker.model.neck(xf)

    # ── per-template: RPN head + post-processing ─────────────────────────
    results = []
    for zf in zf_list:
        with torch.no_grad():
            cls, loc = tracker.model.rpn_head(zf, xf)

        score     = tracker._convert_score(cls)
        pred_bbox = tracker._convert_bbox(loc, tracker.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     sz(saved_size[0]*scale_z, saved_size[1]*scale_z))
        r_c = change((saved_size[0] / saved_size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore  = penalty * score
        pscore  = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                  tracker.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox_raw = pred_bbox[:, best_idx] / scale_z
        lr       = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        cx       = bbox_raw[0] + saved_center[0]
        cy       = bbox_raw[1] + saved_center[1]
        width    = saved_size[0] * (1 - lr) + bbox_raw[2] * lr
        height   = saved_size[1] * (1 - lr) + bbox_raw[3] * lr
        cx, cy, width, height = tracker._bbox_clip(
            cx, cy, width, height, frame.shape[:2])

        results.append(({
            'bbox': [cx - width/2, cy - height/2, width, height],
            'best_score': float(score[best_idx]),
        }, zf))

    # ── restore tracker state ────────────────────────────────────────────
    tracker.center_pos = saved_center
    tracker.size       = saved_size
    tracker.model.zf   = saved_zf

    return sorted(results, key=lambda r: r[0]['best_score'], reverse=True)


def best_reid_detection(track_box, dets, reid_gate, frame,
                        min_iou=0.15, reid_thr=0.90):
    """
    Like best_matching_detection but gates candidates by ReID similarity.
    Primary sort: ReID similarity.  Secondary: IoU.
    Returns (best_box, best_conf, best_iou, best_reid_sim) or (None,0,0,0).
    """
    candidates = []
    for (det_box, det_conf) in dets:
        ov = iou(track_box, det_box)
        if ov >= min_iou:
            sim = reid_gate.similarity(frame, det_box)
            candidates.append((det_box, det_conf, ov, sim))
    if not candidates:
        return None, 0.0, 0.0, 0.0

    # Primary: ReID similarity, secondary: IoU
    candidates.sort(key=lambda c: (c[3], c[2]), reverse=True)
    best = candidates[0]

    if best[3] < reid_thr:
        return None, 0.0, 0.0, best[3]

    return best   # (box, conf, iou, reid_sim)


# ══════════════════════════════════════════════════════════════════════════════
# Improved Main (v2) – ReID gate + template bank + tuned for crowds
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
    print(f"  Source   : {src_label}  ({W}x{H} @ {fps_in:.1f} fps)")

    tracker  = load_siam(args.config, args.weights)

    # ── tune hyperparameters for crowded scenes ──────────────────────────────
    if args.window_influence > 0:
        cfg.TRACK.WINDOW_INFLUENCE = args.window_influence
        print(f"  Override : WINDOW_INFLUENCE = {cfg.TRACK.WINDOW_INFLUENCE}")
    if args.penalty_k > 0:
        cfg.TRACK.PENALTY_K = args.penalty_k
        print(f"  Override : PENALTY_K = {cfg.TRACK.PENALTY_K}")
    if args.lr_scale != 1.0:
        old_lr = cfg.TRACK.LR
        cfg.TRACK.LR = old_lr * args.lr_scale
        print(f"  Override : LR = {old_lr} × {args.lr_scale} = {cfg.TRACK.LR:.3f}"
              f"  (slower template update = resists absorbing wrong person)")
    # Regenerate hanning window in case score_size changed
    tracker.window = np.tile(
        np.outer(np.hanning(tracker.score_size),
                 np.hanning(tracker.score_size)).flatten(),
        tracker.anchor_num)

    use_half = args.half and args.device.startswith("cuda")
    print(f"  Seg YOLO : {args.seg}  device={args.device}  half={use_half}  (loading…)")
    seg_model = YOLO(args.seg)
    seg_model.to(args.device)
    has_masks = (seg_model.task == "segment")
    print(f"  Seg task : {seg_model.task}  (masks={'yes' if has_masks else 'NO – bbox only'})")
    print(f"  Det YOLO : {Path(args.yolo).name}  device={args.device}  half={use_half}  (loading…)")
    det_model = YOLO(args.yolo)
    det_model.to(args.device)
    print(f"  Det task : {det_model.task}  (every {args.det_interval} frames)")
    # ── optional YOLO-cls model for embedding cue ─────────────────────────
    cls_model = None
    if args.reid_cls_model and Path(args.reid_cls_model).exists():
        print(f"  YOLO-cls : {args.reid_cls_model}  (loading…)")
        cls_model = YOLO(args.reid_cls_model)
        cls_model.to(args.device)
        print(f"  YOLO-cls : loaded (cls_weight={args.cls_weight})")
    else:
        if args.reid_cls_model:
            print(f"  YOLO-cls : not found at {args.reid_cls_model}, using color-only mode")
        else:
            print(f"  YOLO-cls : disabled (color-only mode)")

    print(f"  ReID thr : {args.reid_thr}   Yolo bank : {args.bank_size}")
    print(f"  MultiCue : color_w={args.color_weight}  cls_w={args.cls_weight}  "
          f"neg_bank={args.neg_bank_size}  hist_bins={args.hist_bins}")
    print(f"  Anchor bank : {args.anchor_bank_size} permanent templates")
    print(f"  Kalman weight : {args.kalman_weight}   Max jump : {args.max_jump_factor}x diag")
    print(f"  Crowd thr : {args.crowd_thr}  Lockdown thr : {args.crowd_lockdown_thr}"
          f"  Recover thr : {args.recover_thr}")
    print(f"  Seg refine : every {args.seg_refine_interval} frames  "
          f"(box snaps to person mask)")
    print(f"  Correction interval : every {args.corr_interval} frames")
    print(f"  High-conf trigger   : conf≥{args.corr_conf}  IoU≥{args.corr_iou}")
    if args.lr_scale != 1.0:
        print(f"  LR (template update): {cfg.TRACK.LR:.3f}  (scaled {args.lr_scale}×)")
    if args.kalman_search_blend > 0:
        print(f"  Kalman search blend : {args.kalman_search_blend}")
    if args.max_size_change > 0:
        print(f"  Max size change     : {args.max_size_change}")
    if args.scale_lock:
        print(f"  Scale lock          : ON (size frozen after warmup)")
    if args.crowd_kalman_boost > 0:
        print(f"  Crowd Kalman boost  : {args.crowd_kalman_boost}")
    if args.crowd:
        print(f"  MODE: --crowd (anti-drift preset active)")
    print("=" * 62)

    if not args.show:
        print("ERROR: --show is required for interactive person selection.",
              file=sys.stderr)
        return 1

    WIN = "Seg-Anchor SiamRPN v3 (Multi-Anchor+Crowd)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    # ── read first frame ──────────────────────────────────────────────────────
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: cannot read first frame", file=sys.stderr); return 1

    # ── segment first frame ───────────────────────────────────────────────────
    print("  Running segmentation on first frame…")
    dets_first = run_seg(seg_model, first_frame,
                         args.seg_conf, args.seg_iou, args.device, args.imgsz, args.half)
    print(f"  Found {len(dets_first)} person(s) on first frame.")

    # ── user selects person ───────────────────────────────────────────────────
    init_bbox, init_mask = click_select_person(first_frame, dets_first, WIN)
    if init_bbox is None:
        print("No target selected. Exiting."); return 0

    if init_mask is not None:
        tight = tight_bbox_from_mask(init_mask)
        if tight is not None:
            print(f"  Click bbox : {[int(v) for v in init_bbox]}")
            print(f"  Mask tight : {list(tight)}  (using mask-derived bbox)")
            init_bbox = tight
    else:
        # ── AUTO-TIGHTEN: user drew a big box → find nearest detection inside ──
        # Only tighten if a detection's center falls INSIDE the drawn box AND
        # the detection is much smaller than the drawn box (user likely intended
        # to select that specific object, not the whole region).
        tightened = False
        if dets_first:
            nearest = nearest_detection(init_bbox, dets_first)
            if nearest:
                det_bbox, det_mask = nearest
                dcx = det_bbox[0] + det_bbox[2] / 2
                dcy = det_bbox[1] + det_bbox[3] / 2
                bx, by, bw, bh = init_bbox
                det_area = det_bbox[2] * det_bbox[3]
                drawn_area = bw * bh
                # Only tighten if detection is inside drawn box AND
                # drawn box is at least 2x larger than the detection
                if (bx <= dcx <= bx + bw and by <= dcy <= by + bh
                        and drawn_area > 2 * det_area):
                    if det_mask is not None:
                        tight = tight_bbox_from_mask(det_mask)
                        if tight:
                            print(f"  Auto-tighten: drawn box {[int(v) for v in init_bbox]}"
                                  f" → mask tight {list(tight)}")
                            init_bbox = tight
                            init_mask = det_mask
                            tightened = True
                    else:
                        print(f"  Auto-tighten: drawn box {[int(v) for v in init_bbox]}"
                              f" → detection bbox {[int(v) for v in det_bbox]}")
                        init_bbox = det_bbox
                        tightened = True
        if not tightened:
            print(f"  Using drawn box as-is: {[int(v) for v in init_bbox]}")

    # ── mask background → clean template ──────────────────────────────────────
    if init_mask is not None:
        template_frame, init_bbox_adj = crop_masked_template(
            first_frame, init_mask, init_bbox)
        print("  Template: crop-then-mask (tight crop for cleaner features).")
    else:
        template_frame = first_frame
        init_bbox_adj = init_bbox
        print("  No mask available – using raw frame as template.")

    # ── YOLO-primary mode (opt-in only, useful for sparse nadir scenes) ─────
    init_area = init_bbox[2] * init_bbox[3]
    yolo_primary = args.yolo_primary   # manual flag only, no auto-enable
    if yolo_primary:
        print(f"  YOLO-PRIMARY mode: enabled (flag)")
        print(f"    Position driven by YOLO+Kalman, SiamRPN as fallback")
        print(f"    blend={args.yolo_blend}  accept_radius={args.yolo_accept_radius}x diag")
    print(f"  Target area: {int(init_area)} px²")

    # ── init SiamRPN → capture first permanent anchor ───────────────────────
    tracker.init(template_frame, init_bbox_adj)
    zf_init          = tracker.model.zf
    anchor_bank      = [zf_init]         # permanent references (grows up to anchor_bank_size)
    ANCHOR_BANK_MAX  = max(1, args.anchor_bank_size)
    current_box      = list(map(int, init_bbox))
    current_mask     = init_mask
    active_template  = "anchor"

    # ── MultiCue gate  (initialise from selected target) ────────────────────
    reid_gate = MultiCueGate(first_frame, init_bbox, init_mask,
                              cls_model=cls_model,
                              hist_bins=tuple(args.hist_bins),
                              color_weight=args.color_weight,
                              cls_weight=args.cls_weight,
                              neg_bank_size=args.neg_bank_size)
    print(f"  MultiCue gate : initialised (hist_bins={args.hist_bins}  "
          f"color_w={reid_gate._w_color:.2f}  cls_w={reid_gate._w_cls:.2f})")

    # ── template bank (yolo templates; rolling FIFO) ─────────────────────────
    yolo_bank = [zf_init]             # start with anchor copy; replaced by corrections
    BANK_MAX  = max(1, args.bank_size)

    # ── crowd robustness state ───────────────────────────────────────────────
    recent_sizes = [(current_box[2], current_box[3])]   # (w, h) rolling history
    SIZE_HISTORY = 30
    last_anchor_capture_frame = 0

    # ── state ─────────────────────────────────────────────────────────────────
    frames_since_corr  = 0
    last_corr_box      = None
    last_corr_frame    = -1
    last_corr_type     = ""
    kalman             = KalmanBoxTracker(current_box)
    drift_consec       = 0
    last_recover_box   = None
    last_recover_frame = -1
    last_seg_refine    = 0              # frame index of last seg refinement
    track_state        = "tracked"     # tracked | partial | occluded | lost
    prev_track_state   = "tracked"     # state from the previous frame
    occ_consec         = 0             # consecutive full-occlusion frames
    recovery_frames    = 0             # countdown: frames of Kalman-blend after occlusion
    RECOVERY_DURATION  = 15            # how many frames to stay in recovery mode
    locked_size        = None          # (w, h) set after warmup when --scale-lock
    nearby_count       = 0             # nearby detections (updated each frame)
    snap_n             = 0
    paused             = False
    frame_idx          = 1
    t0                 = time.time()

    # show annotated first frame
    ann_first = first_frame.copy()
    if init_mask is not None and args.show_mask:
        ann_first = overlay_mask(ann_first, init_mask, COL_ANCHOR, alpha=0.30)
    ix, iy, iw, ih = map(int, init_bbox)
    cv2.rectangle(ann_first, (ix, iy), (ix+iw, iy+ih), COL_INIT, args.thickness)
    put_label(ann_first, "INIT (seg+MultiCue+multi-anchor)" if init_mask is not None else "INIT",
              ix, max(14, iy-4), bg=COL_INIT)
    if writer:
        writer.write(ann_first)
    cv2.imshow(WIN, ann_first)
    cv2.waitKey(1)

    def do_reinit(frame_img, win):
        """Helper: re-select person and reinitialise everything."""
        nonlocal anchor_bank, current_box, current_mask, yolo_bank
        nonlocal active_template, frames_since_corr, last_corr_box
        nonlocal drift_consec, last_recover_box, last_recover_frame
        nonlocal recent_sizes, last_anchor_capture_frame, last_seg_refine
        nonlocal yolo_primary, track_state, occ_consec, init_frame_idx
        nonlocal prev_track_state, recovery_frames, locked_size

        dets_r = run_seg(seg_model, frame_img,
                         args.seg_conf, args.seg_iou, args.device, args.imgsz, args.half)
        new_bbox, new_mask = click_select_person(frame_img, dets_r, win)
        if new_bbox is None:
            return False

        if new_mask is not None:
            tight = tight_bbox_from_mask(new_mask)
            if tight:
                new_bbox = tight

        if new_mask is not None:
            tpl_frame, new_bbox_adj = crop_masked_template(
                frame_img, new_mask, new_bbox)
        else:
            tpl_frame, new_bbox_adj = frame_img, new_bbox

        tracker.init(tpl_frame, new_bbox_adj)
        anchor_bank      = [tracker.model.zf]   # reset to single permanent anchor
        yolo_bank        = [tracker.model.zf]
        current_box      = list(map(int, new_bbox))
        current_mask     = new_mask
        active_template  = "anchor"
        frames_since_corr = 0
        last_corr_box    = None
        kalman.reset(current_box)
        drift_consec     = 0
        last_recover_box = None
        recent_sizes     = [(current_box[2], current_box[3])]
        last_anchor_capture_frame = frame_idx
        last_seg_refine = frame_idx
        track_state      = "tracked"
        prev_track_state = "tracked"
        occ_consec       = 0
        recovery_frames  = 0
        locked_size      = None        # re-lock after new warmup
        init_frame_idx   = frame_idx   # restart warmup
        reid_gate.reset(frame_img, new_bbox, new_mask)
        yolo_primary = args.yolo_primary
        print(f"  Re-init ALL + MultiCue → {[int(v) for v in new_bbox]}"
              f"  mask={'yes' if new_mask is not None else 'no'}")
        return True

    def do_begin_new(win):
        """Seek to frame 0, let user select a new target, track from start."""
        nonlocal anchor_bank, current_box, current_mask, yolo_bank
        nonlocal active_template, frames_since_corr, last_corr_box
        nonlocal drift_consec, last_recover_box, last_recover_frame
        nonlocal recent_sizes, last_anchor_capture_frame, last_seg_refine
        nonlocal yolo_primary, track_state, occ_consec, init_frame_idx
        nonlocal prev_track_state, recovery_frames, locked_size
        nonlocal frame_idx, last_corr_frame, last_corr_type
        nonlocal t0, writer

        if is_live:
            print("  Cannot seek to beginning on a live source.")
            return False

        # Seek to frame 0 and read it
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok_f0, frame0 = cap.read()
        if not ok_f0:
            print("  ERROR: cannot read frame 0"); return False

        # Segment + select on frame 0
        dets_r = run_seg(seg_model, frame0,
                         args.seg_conf, args.seg_iou, args.device, args.imgsz, args.half)
        new_bbox, new_mask = click_select_person(frame0, dets_r, win)
        if new_bbox is None:
            return False

        if new_mask is not None:
            tight = tight_bbox_from_mask(new_mask)
            if tight:
                new_bbox = tight

        if new_mask is not None:
            tpl_frame, new_bbox_adj = crop_masked_template(
                frame0, new_mask, new_bbox)
        else:
            tpl_frame, new_bbox_adj = frame0, new_bbox

        # Re-create writer from scratch (new video from frame 0)
        if writer:
            writer.release()
            writer = cv2.VideoWriter(args.save,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_in, (W, H))

        # Reset tracker
        tracker.init(tpl_frame, new_bbox_adj)
        anchor_bank      = [tracker.model.zf]
        yolo_bank        = [tracker.model.zf]
        current_box      = list(map(int, new_bbox))
        current_mask     = new_mask
        active_template  = "anchor"
        frames_since_corr = 0
        last_corr_box    = None
        last_corr_frame  = -1
        last_corr_type   = ""
        kalman.reset(current_box)
        drift_consec     = 0
        last_recover_box = None
        last_recover_frame = -1
        recent_sizes     = [(current_box[2], current_box[3])]
        last_anchor_capture_frame = 0
        last_seg_refine  = 0
        track_state      = "tracked"
        prev_track_state = "tracked"
        occ_consec       = 0
        recovery_frames  = 0
        locked_size      = None        # re-lock after new warmup
        frame_idx        = 1
        init_frame_idx   = 1
        t0               = time.time()
        reid_gate.reset(frame0, new_bbox, new_mask)
        yolo_primary = args.yolo_primary

        print(f"  BEGIN NEW from frame 0 → {[int(v) for v in new_bbox]}"
              f"  mask={'yes' if new_mask is not None else 'no'}")
        return True

    # ── warmup: let SiamRPN converge freely before enabling guards ──────────
    # When user draws an oversized box, SiamRPN needs ~30 frames to naturally
    # shrink it to the actual target.  During warmup, disable state machine
    # gates, Kalman proximity bias, and jump rejection that would fight this.
    WARMUP_FRAMES = 30
    init_frame_idx = 1   # reset on reinit so warmup restarts

    # ══════════════════════════════════════════════════════════════════════════
    # Tracking loop (v2 – ReID + template bank + efficient multi-template)
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
            elif key == ord("b"):
                if do_begin_new(WIN):
                    paused = False
            continue

        ok, frame = cap.read()
        if not ok:
            break

        annotated = frame   # draw on frame directly (avoid copy)

        # ── Kalman prediction ─────────────────────────────────────────────────
        kalman_pred = kalman.predict()

        # ── Kalman-guided search centre ──────────────────────────────────────
        # Nudge SiamRPN's search centre toward Kalman prediction so the
        # search crop is biased toward where MOTION says the target is,
        # not where appearance (unreliable when everyone looks alike) says.
        frames_since_init = frame_idx - init_frame_idx
        if args.kalman_search_blend > 0 and frames_since_init > WARMUP_FRAMES:
            kx, ky, kw, kh = kalman_pred
            kcx, kcy = kx + kw / 2, ky + kh / 2
            old_cx, old_cy = tracker.center_pos
            # Dynamic boost: stronger Kalman pull when crowd is nearby
            blend = args.kalman_search_blend
            if args.crowd_kalman_boost > 0 and nearby_count >= args.crowd_thr:
                blend = max(blend, args.crowd_kalman_boost)
            tracker.center_pos = np.array([
                old_cx * (1 - blend) + kcx * blend,
                old_cy * (1 - blend) + kcy * blend])

        in_warmup = (frames_since_init <= WARMUP_FRAMES)

        # ── Scale lock: capture size at end of warmup ──────────────────────
        if (args.scale_lock
                and locked_size is None
                and not in_warmup
                and len(recent_sizes) >= 5):
            locked_size = (
                float(np.median([s[0] for s in recent_sizes])),
                float(np.median([s[1] for s in recent_sizes])))
            print(f"  [frame {frame_idx:4d}] SCALE LOCKED  "
                  f"{locked_size[0]:.0f}x{locked_size[1]:.0f}")

        # ── Kalman prediction context ────────────────────────────────────────
        kx, ky, kw, kh = kalman_pred
        kalman_cx = kx + kw / 2
        kalman_cy = ky + kh / 2
        kalman_diag = np.sqrt(kw**2 + kh**2) if (kw > 0 and kh > 0) else 50.0

        if in_warmup:
            # ── WARMUP: track exactly like main_v1 (simple dual-template) ────
            # Use tracker.track() directly so SiamRPN's natural size EMA
            # works identically to main_v1, allowing oversized boxes to shrink.
            with torch.no_grad():
                out_a = track_with_zf(tracker, frame, anchor_bank[0])
                out_y = track_with_zf(tracker, frame, yolo_bank[0])
            score_a_raw = float(out_a.get("best_score", 0))
            score_y_raw = float(out_y.get("best_score", 0))
            if score_a_raw >= score_y_raw:
                winner_out = out_a
                winner_zf  = anchor_bank[0]
            else:
                winner_out = out_y
                winner_zf  = yolo_bank[0]
            winner_raw = max(score_a_raw, score_y_raw)
            all_templates = [anchor_bank[0]]  # for display

        else:
            # ── POST-WARMUP: multi-template + scoring + guards ───────────────
            # Use track_with_zf (= tracker.track()) for each template so the
            # SiamRPN size-EMA and post-processing are IDENTICAL to main_v1.
            # This avoids the box-growth divergence that track_multi_zf had.
            # During "partial" state use ONLY anchor templates — the original
            # identity references — to prevent yolo_bank noise from matching
            # a nearby similar object.
            if track_state == "partial" or recovery_frames > 0:
                all_templates = list(anchor_bank)
            else:
                all_templates = list(anchor_bank) + yolo_bank
            seen_ids = set()
            unique_templates = []
            for t in all_templates:
                tid = id(t)
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    unique_templates.append(t)
            all_templates = unique_templates

            tmpl_results = []
            with torch.no_grad():
                for zf in all_templates:
                    out = track_with_zf(tracker, frame, zf)
                    tmpl_results.append((out, zf))

            # Compute median recent size for size-consistency guard
            if recent_sizes:
                med_w = float(np.median([s[0] for s in recent_sizes]))
                med_h = float(np.median([s[1] for s in recent_sizes]))
            else:
                med_w, med_h = kw, kh

            # Boost spatial weight when scores are low (partial occlusion).
            # During occlusion, appearance is unreliable but position is still
            # predicted well by Kalman — trust position more to avoid switching
            # to a nearby similar object.
            max_raw = max(out['best_score'] for (out, _) in tmpl_results)
            if max_raw < args.tracked_thr:
                # Low-confidence: 3× Kalman weight to anchor on predicted position
                eff_kalman_w = args.kalman_weight * 3.0
            else:
                eff_kalman_w = args.kalman_weight

            scored_results = []
            for (out, zf) in tmpl_results:
                raw_score = out['best_score']
                bbox = out['bbox']
                pred_cx = bbox[0] + bbox[2] / 2
                pred_cy = bbox[1] + bbox[3] / 2
                pred_w, pred_h = bbox[2], bbox[3]

                # Kalman proximity bonus (boosted during low-score frames)
                dist = np.sqrt((pred_cx - kalman_cx)**2 + (pred_cy - kalman_cy)**2)
                proximity = max(0.0, 1.0 - dist / max(kalman_diag * 2, 1.0))
                adjusted_score = raw_score + eff_kalman_w * proximity

                # Size-consistency penalty: only penalize GROWTH beyond median
                if med_w > 0 and med_h > 0:
                    w_growth = max(0, pred_w - med_w) / med_w
                    h_growth = max(0, pred_h - med_h) / med_h
                    size_dev = max(w_growth, h_growth)
                    if size_dev > args.size_guard_ratio:
                        penalty = 0.3 * (size_dev - args.size_guard_ratio)
                        adjusted_score -= penalty

                scored_results.append((adjusted_score, raw_score, out, zf))

            scored_results.sort(key=lambda r: r[0], reverse=True)
            winner_adj, winner_raw, winner_out, winner_zf = scored_results[0]

        # ── centroid jump rejection (post-warmup only) ────────────────────
        if not in_warmup and args.max_jump_factor > 0 and kalman_diag > 10:
            # Tighter jump limit when scores are low or already degraded —
            # during partial occlusion the tracker should stay near the
            # predicted position, not jump to a nearby similar object.
            jf = args.max_jump_factor
            cur_raw = winner_out['best_score']
            if track_state in ("partial", "occluded") or cur_raw < args.tracked_thr:
                jf = min(jf, 1.0)
            max_jump = jf * kalman_diag
            w_bbox = winner_out['bbox']
            w_cx = w_bbox[0] + w_bbox[2] / 2
            w_cy = w_bbox[1] + w_bbox[3] / 2
            jump_dist = np.sqrt((w_cx - kalman_cx)**2 + (w_cy - kalman_cy)**2)

            if jump_dist > max_jump:
                # Winner jumped too far — find closest-to-Kalman template instead
                best_fallback = None
                best_fb_dist  = float('inf')
                for (adj, raw, out, zf) in scored_results:
                    fb_bbox = out['bbox']
                    fb_cx = fb_bbox[0] + fb_bbox[2] / 2
                    fb_cy = fb_bbox[1] + fb_bbox[3] / 2
                    fb_dist = np.sqrt((fb_cx - kalman_cx)**2 + (fb_cy - kalman_cy)**2)
                    if fb_dist < best_fb_dist:
                        best_fb_dist = fb_dist
                        best_fallback = (adj, raw, out, zf)
                if best_fallback and best_fb_dist <= max_jump:
                    winner_adj, winner_raw, winner_out, winner_zf = best_fallback
                    print(f"  [frame {frame_idx:4d}] JUMP REJECTED "
                          f"dist={jump_dist:.0f} > {max_jump:.0f}  → fallback")

        best_score = winner_out['best_score']

        # Determine which template type won
        anchor_ids = {id(t) for t in anchor_bank}
        is_anchor_winner = id(winner_zf) in anchor_ids
        active_template  = "anchor" if is_anchor_winner else "yolo"

        # Collect scores for display
        if in_warmup:
            score_a = score_a_raw if score_a_raw else best_score
            score_y = score_y_raw if score_y_raw else 0.0
        else:
            score_a = 0.0
            best_yolo_score = 0.0
            for (_adj, _raw, out, zf) in scored_results:
                if id(zf) in anchor_ids:
                    score_a = max(score_a, out['best_score'])
                else:
                    best_yolo_score = max(best_yolo_score, out['best_score'])
            score_y = best_yolo_score

        # ══════════════════════════════════════════════════════════════════
        # Tracking state machine:
        #   tracked  → score high, normal tracking
        #   partial  → score medium, use SiamRPN but freeze templates
        #   occluded → score low, coast on Kalman, ignore SiamRPN
        #   lost     → score very low or occ_patience exceeded
        # During warmup: everything above score_thr = "tracked" (like v1),
        # so SiamRPN can freely shrink an oversized initial box.
        # ══════════════════════════════════════════════════════════════════
        in_warmup = (frames_since_init <= WARMUP_FRAMES)

        # Save previous state BEFORE updating it
        prev_track_state = track_state

        if in_warmup:
            # Warmup: simple binary tracked/lost (matches main_v1 behaviour)
            if best_score >= args.score_thr:
                track_state = "tracked"
                occ_consec = 0
            else:
                track_state = "lost"
        elif best_score >= args.tracked_thr:
            track_state = "tracked"
            occ_consec = 0
        elif best_score >= args.partial_thr:
            track_state = "partial"
            occ_consec = 0
        elif best_score >= args.score_thr:
            occ_consec += 1
            track_state = "lost" if occ_consec >= args.occ_patience else "occluded"
        else:
            track_state = "lost"

        is_lost = (track_state == "lost")
        kalman_coast = False

        # ── Recovery cooldown after occlusion ─────────────────────────────
        # When transitioning from occluded/partial → tracked, don't
        # immediately trust SiamRPN.  It may have latched onto the
        # occluder.  Blend toward Kalman for RECOVERY_DURATION frames.
        if (prev_track_state in ("occluded", "partial")
                and track_state == "tracked"
                and not in_warmup):
            recovery_frames = RECOVERY_DURATION
            print(f"  [frame {frame_idx:4d}] RECOVERY START  "
                  f"{prev_track_state} → tracked  "
                  f"(blend for {RECOVERY_DURATION} frames)")

        if track_state in ("occluded", "partial", "lost"):
            # While degraded, don't count down recovery
            recovery_frames = 0
        # Note: recovery_frames countdown happens AFTER the apply-outputs
        # section so the full duration is used for blending.

        # Debug: print size convergence for first 50 frames
        if frames_since_init <= 50:
            bbox_w = winner_out['bbox'][2]
            bbox_h = winner_out['bbox'][3]
            print(f"  [frame {frame_idx:4d}] warmup={in_warmup}  "
                  f"size={bbox_w:.0f}x{bbox_h:.0f}  "
                  f"score={best_score:.3f}  state={track_state}")

        if track_state == "occluded":
            # ── FULL OCCLUSION: ignore SiamRPN, coast on Kalman ──────────
            # SiamRPN is probably seeing the occluder, not our target.
            kx, ky, kw, kh = kalman_pred
            kcx, kcy = kx + kw / 2, ky + kh / 2
            current_box = [int(kcx - kw / 2), int(kcy - kh / 2),
                           int(max(1, kw)), int(max(1, kh))]
            tracker.center_pos = np.array([kcx, kcy])
            tracker.size = np.array([max(1.0, kw), max(1.0, kh)])
            tracker.model.zf = winner_zf   # keep template, ignore position
            kalman_coast = True
        elif track_state == "partial":
            # ── PARTIAL OCCLUSION: blend SiamRPN toward Kalman ───────────
            # During partial occlusion SiamRPN may latch onto a nearby
            # similar object.  Blending toward Kalman keeps the box on
            # the predicted position while still allowing gentle updates.
            apply_outputs(tracker, winner_out, winner_zf)
            siam_bbox = winner_out["bbox"]
            siam_cx = siam_bbox[0] + siam_bbox[2] / 2
            siam_cy = siam_bbox[1] + siam_bbox[3] / 2
            siam_w, siam_h = siam_bbox[2], siam_bbox[3]

            kx, ky, kw, kh = kalman_pred
            kcx, kcy = kx + kw / 2, ky + kh / 2

            # 70% Kalman, 30% SiamRPN — strongly resist drift
            blend_k = 0.70
            bcx = siam_cx * (1 - blend_k) + kcx * blend_k
            bcy = siam_cy * (1 - blend_k) + kcy * blend_k
            bw  = siam_w  * (1 - blend_k) + max(1.0, kw) * blend_k
            bh  = siam_h  * (1 - blend_k) + max(1.0, kh) * blend_k

            tracker.center_pos = np.array([bcx, bcy])
            tracker.size       = np.array([max(1.0, bw), max(1.0, bh)])
            current_box = [int(bcx - bw / 2), int(bcy - bh / 2),
                           int(max(1, bw)), int(max(1, bh))]
        else:
            # ── TRACKED / LOST: apply SiamRPN result ─────────────────────
            apply_outputs(tracker, winner_out, winner_zf)
            current_box = list(map(int, winner_out["bbox"]))

            # ── Recovery blend: after occlusion, distrust SiamRPN ────────
            # SiamRPN may latch onto the occluder that just moved away.
            # During recovery, blend toward Kalman (like partial state).
            # Also: if Kalman says target was stationary but SiamRPN
            # implies sudden movement, reject SiamRPN completely.
            if recovery_frames > 0 and not in_warmup:
                kx, ky, kw, kh = kalman_pred
                kcx, kcy = kx + kw / 2, ky + kh / 2

                siam_bbox = winner_out["bbox"]
                siam_cx = siam_bbox[0] + siam_bbox[2] / 2
                siam_cy = siam_bbox[1] + siam_bbox[3] / 2

                # Velocity gate: if Kalman thinks target is ~stationary
                # but SiamRPN jumped far, coast on Kalman entirely.
                kvx, kvy = kalman._x[4], kalman._x[5]
                k_speed = np.sqrt(kvx**2 + kvy**2)
                siam_dist = np.sqrt((siam_cx - kcx)**2 + (siam_cy - kcy)**2)
                kalman_diag_local = np.sqrt(kw**2 + kh**2) if (kw > 0 and kh > 0) else 1.0

                if k_speed < 2.0 and siam_dist > kalman_diag_local * 0.3:
                    # Target was stationary, SiamRPN jumped — full coast
                    current_box = [int(kcx - kw / 2), int(kcy - kh / 2),
                                   int(max(1, kw)), int(max(1, kh))]
                    tracker.center_pos = np.array([kcx, kcy])
                    tracker.size = np.array([max(1.0, kw), max(1.0, kh)])
                    kalman_coast = True
                    print(f"  [frame {frame_idx:4d}] RECOVERY VELOCITY GATE  "
                          f"k_speed={k_speed:.1f} siam_dist={siam_dist:.1f}  "
                          f"→ coast on Kalman  (recovery={recovery_frames})")
                else:
                    # Blend: decay from 60% Kalman → 0% over RECOVERY_DURATION
                    blend_k = 0.60 * (recovery_frames / RECOVERY_DURATION)
                    bcx = siam_cx * (1 - blend_k) + kcx * blend_k
                    bcy = siam_cy * (1 - blend_k) + kcy * blend_k
                    bw  = siam_bbox[2] * (1 - blend_k) + max(1.0, kw) * blend_k
                    bh  = siam_bbox[3] * (1 - blend_k) + max(1.0, kh) * blend_k
                    current_box = [int(bcx - bw / 2), int(bcy - bh / 2),
                                   int(max(1, bw)), int(max(1, bh))]
                    tracker.center_pos = np.array([bcx, bcy])
                    tracker.size = np.array([max(1.0, bw), max(1.0, bh)])
                    if recovery_frames % 5 == 0:
                        print(f"  [frame {frame_idx:4d}] RECOVERY BLEND  "
                              f"k={blend_k:.2f}  (recovery={recovery_frames})")

            # ── Per-frame size clamp ──────────────────────────────────────
            # Prevents box from suddenly growing to cover two similar
            # adjacent objects (e.g. two cars side by side).
            if args.max_size_change > 0 and len(recent_sizes) >= 3:
                med_w = float(np.median([s[0] for s in recent_sizes]))
                med_h = float(np.median([s[1] for s in recent_sizes]))
                cur_w, cur_h = current_box[2], current_box[3]
                # Only clamp GROWTH — never fight natural shrinkage
                max_w = med_w * (1 + args.max_size_change)
                max_h = med_h * (1 + args.max_size_change)
                new_w = min(max_w, cur_w)
                new_h = min(max_h, cur_h)
                if abs(new_w - cur_w) > 1 or abs(new_h - cur_h) > 1:
                    cx = current_box[0] + cur_w / 2
                    cy = current_box[1] + cur_h / 2
                    current_box = [int(cx - new_w / 2), int(cy - new_h / 2),
                                   int(max(1, new_w)), int(max(1, new_h))]
                    tracker.center_pos = np.array([cx, cy])
                    tracker.size = np.array([max(1.0, new_w), max(1.0, new_h)])
                    if abs(cur_w - new_w) > 5 or abs(cur_h - new_h) > 5:
                        print(f"  [frame {frame_idx:4d}] SIZE CLAMP  "
                              f"{cur_w:.0f}x{cur_h:.0f} → {new_w:.0f}x{new_h:.0f}  "
                              f"(median {med_w:.0f}x{med_h:.0f})")

            # ── Scale lock: force box back to locked dimensions ─────────
            # In nadir view, target size is constant (altitude doesn't
            # change). This prevents the box from absorbing a neighbour.
            if locked_size is not None and track_state == "tracked":
                lw, lh = locked_size
                cx = current_box[0] + current_box[2] / 2
                cy = current_box[1] + current_box[3] / 2
                current_box = [int(cx - lw / 2), int(cy - lh / 2),
                               int(max(1, lw)), int(max(1, lh))]
                tracker.size = np.array([max(1.0, lw), max(1.0, lh)])

            # Kalman-IoU gate: if SiamRPN box barely overlaps Kalman
            # prediction, it probably jumped to a different object.
            # Active after warmup for tracked state (was crowd-only before,
            # but warmup handling now prevents false triggers during box
            # convergence).
            if (not in_warmup
                    and track_state == "tracked"
                    and frame_idx > 10):
                kx, ky, kw, kh = kalman_pred
                k_box = [kx, ky, kw, kh]
                siam_kalman_iou = iou(current_box, k_box)
                if siam_kalman_iou < 0.10 and kalman_diag > 10:
                    kcx, kcy = kx + kw / 2, ky + kh / 2
                    current_box = [int(kcx - kw / 2), int(kcy - kh / 2),
                                   int(max(1, kw)), int(max(1, kh))]
                    tracker.center_pos = np.array([kcx, kcy])
                    tracker.size = np.array([max(1.0, kw), max(1.0, kh)])
                    kalman_coast = True
                    print(f"  [frame {frame_idx:4d}] KALMAN-IOU GATE  "
                          f"IoU={siam_kalman_iou:.3f} < 0.10  → coast")

            # Velocity-consistency gate (catches opposite-direction occluders)
            # Only active in --crowd mode.
            if (args.crowd
                    and track_state == "tracked"
                    and frame_idx > 10 and not kalman_coast):
                kvx, kvy = kalman._x[4], kalman._x[5]
                k_speed = np.sqrt(kvx**2 + kvy**2)
                if k_speed > 2.0:
                    kx, ky, kw, kh = kalman_pred
                    kcx, kcy = kx + kw / 2, ky + kh / 2
                    prev_cx = kcx - kvx
                    prev_cy = kcy - kvy
                    siam_cx = current_box[0] + current_box[2] / 2
                    siam_cy = current_box[1] + current_box[3] / 2
                    siam_vx = siam_cx - prev_cx
                    siam_vy = siam_cy - prev_cy
                    siam_speed = np.sqrt(siam_vx**2 + siam_vy**2)
                    if siam_speed > 2.0:
                        cos_angle = (kvx * siam_vx + kvy * siam_vy) / \
                                    (k_speed * siam_speed)
                        if cos_angle < -0.5:
                            kx, ky, kw, kh = kalman_pred
                            kcx, kcy = kx + kw / 2, ky + kh / 2
                            current_box = [int(kcx - kw / 2), int(kcy - kh / 2),
                                           int(max(1, kw)), int(max(1, kh))]
                            tracker.center_pos = np.array([kcx, kcy])
                            tracker.size = np.array([max(1.0, kw), max(1.0, kh)])
                            kalman_coast = True
                            print(f"  [frame {frame_idx:4d}] VELOCITY COAST  "
                                  f"cos={cos_angle:.2f}  k={k_speed:.1f}  "
                                  f"s={siam_speed:.1f}")

        # Count down recovery AFTER apply-outputs used the value
        if recovery_frames > 0 and track_state == "tracked":
            recovery_frames -= 1

        # Update recent size history (only when tracking or partial)
        if track_state in ("tracked", "partial"):
            recent_sizes.append((current_box[2], current_box[3]))
            if len(recent_sizes) > SIZE_HISTORY:
                recent_sizes.pop(0)

        if not kalman_coast:
            kalman.update(current_box)

        # ── YOLO detection (every det_interval frames, or every frame in degraded state)
        if (frame_idx % args.det_interval == 0
                or track_state != "tracked"
                or recovery_frames > 0):
            dets = run_yolo(det_model, frame,
                            args.det_conf, args.det_iou, args.device, args.imgsz, args.half)
        else:
            dets = []

        # ── YOLO-primary position override ────────────────────────────────────
        # In sparse nadir/drone views, YOLO detection + Kalman is more reliable
        # than SiamRPN appearance.  Disabled in dense crowds where proximity
        # matching is unreliable (too many detections clustered together).
        yolo_overrode = False
        # Count nearby for quick crowd check (full count computed below)
        _quick_nearby = sum(1 for (db, _) in dets
                            if np.sqrt((db[0]+db[2]/2 - kalman_pred[0]-kalman_pred[2]/2)**2 +
                                       (db[1]+db[3]/2 - kalman_pred[1]-kalman_pred[3]/2)**2)
                            <= 2 * (np.sqrt(kalman_pred[2]**2 + kalman_pred[3]**2) or 50))
        if yolo_primary and dets and not is_lost and _quick_nearby < args.crowd_lockdown_thr:
            kx, ky, kw, kh = kalman_pred
            kcx, kcy = kx + kw / 2, ky + kh / 2
            k_diag = np.sqrt(kw**2 + kh**2) if (kw > 0 and kh > 0) else 50.0

            # Find detection closest to KALMAN prediction (not SiamRPN box)
            best_det = None
            best_dist = float('inf')
            for (det_box, det_conf) in dets:
                dcx = det_box[0] + det_box[2] / 2
                dcy = det_box[1] + det_box[3] / 2
                d = np.sqrt((dcx - kcx)**2 + (dcy - kcy)**2)
                if d < best_dist:
                    best_dist = d
                    best_det = (det_box, det_conf)

            # Accept if within N box-diagonals of Kalman prediction
            if best_det and best_dist < k_diag * args.yolo_accept_radius:
                det_box, det_conf = best_det
                # Velocity-direction check (reject if moving opposite direction)
                vel_ok = velocity_direction_score(kalman, det_box) >= 0.25

                if vel_ok:
                    det_cx = det_box[0] + det_box[2] / 2
                    det_cy = det_box[1] + det_box[3] / 2
                    # Blend: mostly YOLO, some SiamRPN for smoothness
                    a = args.yolo_blend   # YOLO weight (default 0.7)
                    b = 1.0 - a           # SiamRPN weight
                    cur_cx = tracker.center_pos[0]
                    cur_cy = tracker.center_pos[1]
                    new_cx = cur_cx * b + det_cx * a
                    new_cy = cur_cy * b + det_cy * a
                    new_w  = tracker.size[0] * b + det_box[2] * a
                    new_h  = tracker.size[1] * b + det_box[3] * a

                    tracker.center_pos = np.array([new_cx, new_cy])
                    tracker.size       = np.array([new_w, new_h])
                    current_box = [int(new_cx - new_w / 2),
                                   int(new_cy - new_h / 2),
                                   int(new_w), int(new_h)]
                    kalman.update(current_box)
                    yolo_overrode = True

        # ── crowd-adaptive thresholds ─────────────────────────────────────────
        # Count detections within 2x box diagonal to measure crowd density
        cb_diag = np.sqrt(current_box[2]**2 + current_box[3]**2)
        cb_cx = current_box[0] + current_box[2] / 2
        cb_cy = current_box[1] + current_box[3] / 2
        nearby_count = 0
        for (db, _dc) in dets:
            dcx = db[0] + db[2] / 2
            dcy = db[1] + db[3] / 2
            if np.sqrt((dcx - cb_cx)**2 + (dcy - cb_cy)**2) <= 2 * cb_diag:
                nearby_count += 1

        # ── DENSE CROWD LOCKDOWN ─────────────────────────────────────────────
        # When many people are nearby, every correction risks snapping to the
        # wrong person.  The safest approach: FREEZE all corrections and rely
        # solely on SiamRPN + strong cosine window (maximum stickiness).
        dense_crowd = (nearby_count >= args.crowd_lockdown_thr)

        if nearby_count >= args.crowd_thr:
            effective_reid_thr = min(0.95, args.reid_thr + 0.05 * (nearby_count - 2))
            effective_corr_iou = min(0.60, args.corr_iou + 0.05 * (nearby_count - 2))
        else:
            effective_reid_thr = args.reid_thr
            effective_corr_iou = args.corr_iou

        # ── MultiCue-gated drift correction → updates yolo_bank ─────────────
        corrected = False
        corr_type = ""
        corr_reid = 0.0

        if track_state == "tracked" and dets and not dense_crowd:
            best_box, best_conf, best_ov, best_sim = best_reid_detection(
                current_box, dets, reid_gate, frame,
                min_iou=args.min_iou, reid_thr=effective_reid_thr)

            if best_box is not None:
                # Count how many detections overlap the track box —
                # if multiple detections are close, re-centering risks
                # snapping to the wrong one → only do gentle template update.
                n_overlapping = sum(1 for (db, _) in dets
                                    if iou(current_box, db) >= args.min_iou)
                ambiguous = (n_overlapping >= 2)

                trigger_hi  = (not ambiguous
                               and best_conf >= args.corr_conf
                               and best_ov  >= effective_corr_iou)
                trigger_per = (not trigger_hi
                               and frames_since_corr >= args.corr_interval)

                # ── velocity-direction gating ────────────────────────────────
                vel_score = velocity_direction_score(kalman, best_box)
                vel_reject = (vel_score < 0.25 and best_ov < 0.5)

                if vel_reject:
                    print(f"  [frame {frame_idx:4d}] VELOCITY REJECT  "
                          f"vel={vel_score:.2f}  IoU={best_ov:.2f}  "
                          f"crowd={nearby_count}")
                elif trigger_hi or trigger_per:
                    corrected  = True
                    corr_type  = "high-conf" if trigger_hi else "periodic"
                    corr_reid  = best_sim

                    # Extract new yolo template
                    saved_center = tracker.center_pos.copy()
                    saved_size   = tracker.size.copy()
                    saved_zf     = tracker.model.zf

                    tracker.init(frame, (best_box[0], best_box[1],
                                         best_box[2], best_box[3]))
                    new_zf = tracker.model.zf

                    if trigger_hi:
                        # HIGH-CONF: re-centre search toward detection (0.5 blend)
                        det_cx = best_box[0] + best_box[2] / 2
                        det_cy = best_box[1] + best_box[3] / 2
                        new_cx = saved_center[0] * 0.5 + det_cx * 0.5
                        new_cy = saved_center[1] * 0.5 + det_cy * 0.5
                        new_w  = saved_size[0] * 0.5 + best_box[2] * 0.5
                        new_h  = saved_size[1] * 0.5 + best_box[3] * 0.5
                        tracker.center_pos = np.array([new_cx, new_cy])
                        tracker.size       = np.array([new_w, new_h])
                        current_box = [int(new_cx - new_w/2), int(new_cy - new_h/2),
                                       int(new_w), int(new_h)]
                        kalman.update(current_box)
                    else:
                        # PERIODIC: gentle template update only, preserve position
                        tracker.center_pos = saved_center
                        tracker.size       = saved_size
                    tracker.model.zf = saved_zf

                    # Add to template bank (FIFO)
                    yolo_bank.append(new_zf)
                    if len(yolo_bank) > BANK_MAX:
                        yolo_bank.pop(0)

                    # Update MultiCue EMA with confirmed detection
                    reid_gate.update(frame, best_box)

                    last_corr_box   = list(map(int, best_box))
                    last_corr_frame = frame_idx
                    last_corr_type  = corr_type
                    frames_since_corr = 0
                    print(f"  [frame {frame_idx:4d}] zf_yolo updated ({corr_type}"
                          f"{'+ RECENTRE' if trigger_hi else ''})  "
                          f"IoU={best_ov:.2f}  conf={best_conf:.2f}  "
                          f"ReID={corr_reid:.3f}  vel={vel_score:.2f}  "
                          f"crowd={nearby_count}  bank={len(yolo_bank)}")

            # ── negative exemplar collection ─────────────────────────────────
            # For every nearby detection that is NOT the correction target,
            # add its histogram to the negative bank.
            if dets:
                corr_box = best_box if corrected else None
                for (det_box, _det_conf) in dets:
                    dcx = det_box[0] + det_box[2] / 2
                    dcy = det_box[1] + det_box[3] / 2
                    if np.sqrt((dcx - cb_cx)**2 + (dcy - cb_cy)**2) > 2 * cb_diag:
                        continue   # too far away, not relevant
                    if corr_box is not None and iou(corr_box, det_box) > 0.3:
                        continue   # this is the correction target itself
                    if iou(current_box, det_box) > 0.3:
                        continue   # overlaps with our track
                    reid_gate.add_negative(frame, det_box)

        if not corrected:
            frames_since_corr += 1

        # ── periodic seg refinement (snap box to person mask) ─────────────
        # Only in "tracked" state; skip in dense crowds or occlusion.
        if (track_state == "tracked"
                and not dense_crowd
                and args.seg_refine_interval > 0
                and (frame_idx - last_seg_refine) >= args.seg_refine_interval):
            seg_dets = run_seg(seg_model, frame,
                               args.seg_conf, args.seg_iou, args.device, args.imgsz, args.half)
            if seg_dets:
                tcx = current_box[0] + current_box[2] / 2
                tcy = current_box[1] + current_box[3] / 2
                # Find nearest person detection to current tracking centre
                best_seg = None
                best_seg_dist = float('inf')
                for (det_bbox, det_conf, det_mask) in seg_dets:
                    dcx = det_bbox[0] + det_bbox[2] / 2
                    dcy = det_bbox[1] + det_bbox[3] / 2
                    d = np.sqrt((dcx - tcx)**2 + (dcy - tcy)**2)
                    if d < best_seg_dist:
                        best_seg_dist = d
                        best_seg = (det_bbox, det_conf, det_mask)

                # Accept if within 1.5x box diagonal AND no ambiguity
                # (skip when multiple seg detections are close — risk of
                # snapping to the wrong person)
                n_close_seg = sum(1 for (db, _, _) in seg_dets
                                  if np.sqrt((db[0]+db[2]/2 - tcx)**2 +
                                             (db[1]+db[3]/2 - tcy)**2)
                                  < cb_diag * 1.5)
                if (best_seg and best_seg_dist < cb_diag * 1.5
                        and n_close_seg <= 1):
                    seg_bbox, seg_conf, seg_mask = best_seg
                    seg_reid = reid_gate.similarity(frame, seg_bbox, seg_mask)
                    if seg_reid >= effective_reid_thr:
                        refined = False
                        if seg_mask is not None:
                            tight = tight_bbox_from_mask(seg_mask)
                            if tight:
                                tracker.center_pos = np.array(
                                    [tight[0] + tight[2] / 2,
                                     tight[1] + tight[3] / 2])
                                tracker.size = np.array(
                                    [float(tight[2]), float(tight[3])])
                                current_box = list(map(int, tight))
                                current_mask = seg_mask
                                refined = True
                        if not refined:
                            # No mask – use detection bbox to correct size
                            tracker.center_pos = np.array(
                                [seg_bbox[0] + seg_bbox[2] / 2,
                                 seg_bbox[1] + seg_bbox[3] / 2])
                            tracker.size = np.array(
                                [float(seg_bbox[2]), float(seg_bbox[3])])
                            current_box = list(map(int, seg_bbox))

                        kalman.update(current_box)
                        reid_gate.update(frame, current_box, seg_mask)
                        # Update size history with corrected size
                        recent_sizes.append((current_box[2], current_box[3]))
                        if len(recent_sizes) > SIZE_HISTORY:
                            recent_sizes.pop(0)
                        last_seg_refine = frame_idx
                        print(f"  [frame {frame_idx:4d}] SEG REFINE → "
                              f"{current_box}  ReID={seg_reid:.3f}  "
                              f"mask={'yes' if seg_mask is not None else 'no'}")
                    else:
                        last_seg_refine = frame_idx  # don't retry immediately

        # ── auto-capture new permanent anchor ────────────────────────────────
        # Captures diverse reference templates at regular intervals.
        # More permissive than before: any winning template type is OK,
        # threshold lowered so anchors actually get captured for tiny targets.
        if (not in_warmup
                and track_state == "tracked"
                and recovery_frames == 0
                and len(anchor_bank) < ANCHOR_BANK_MAX
                and best_score >= args.anchor_capture_score
                and (frame_idx - last_anchor_capture_frame) >= args.anchor_capture_interval):
            reid_sim = reid_gate.similarity(frame, current_box)
            if reid_sim >= args.reid_thr:
                # Extract a new permanent anchor template
                saved_center = tracker.center_pos.copy()
                saved_size   = tracker.size.copy()
                saved_zf     = tracker.model.zf

                tracker.init(frame, tuple(current_box))
                new_anchor = tracker.model.zf

                tracker.center_pos = saved_center
                tracker.size       = saved_size
                tracker.model.zf   = saved_zf

                anchor_bank.append(new_anchor)
                last_anchor_capture_frame = frame_idx
                reid_gate.update(frame, current_box)
                print(f"  [frame {frame_idx:4d}] NEW PERMANENT ANCHOR captured "
                      f"({len(anchor_bank)}/{ANCHOR_BANK_MAX})  "
                      f"score={best_score:.3f}  ReID={reid_sim:.3f}")

        # ── anchor drift flag ─────────────────────────────────────────────────
        anchor_drift = (not is_lost
                        and score_y > 0
                        and score_a < args.anchor_warn_ratio * score_y)
        drift_consec = (drift_consec + 1) if anchor_drift else 0

        # ── perimeter search (ReID-gated recovery) ────────────────────────────
        recovered = False
        if drift_consec >= args.drift_patience and dets:
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
                sim = reid_gate.similarity(frame, bbox_c)
                if sim > best_sim:
                    best_sim  = sim
                    best_cand = bbox_c

            if best_cand is not None and best_sim >= args.recover_thr:
                tracker.init(frame, (best_cand[0], best_cand[1],
                                     best_cand[2], best_cand[3]))
                new_zf = tracker.model.zf
                yolo_bank.append(new_zf)
                if len(yolo_bank) > BANK_MAX:
                    yolo_bank.pop(0)
                current_box        = list(map(int, best_cand))
                kalman.update(current_box)
                last_recover_box   = current_box[:]
                last_recover_frame = frame_idx
                drift_consec       = 0
                recovered          = True
                reid_gate.update(frame, best_cand)
                print(f"  [frame {frame_idx:4d}] PERIMETER RECOVERY "
                      f"ReID={best_sim:.3f}  box={current_box}  "
                      f"searched {len(candidates)} candidate(s)")
            else:
                n_cands = len(candidates)
                best_s  = f"{best_sim:.3f}" if best_cand else "–"
                print(f"  [frame {frame_idx:4d}] perimeter search: "
                      f"{n_cands} candidates, best_ReID={best_s}  "
                      f"(thr={args.recover_thr})")

        # ══════════════════════════════════════════════════════════════════════
        # Annotation
        # ══════════════════════════════════════════════════════════════════════

        # 1. current seg mask overlay
        if args.show_mask and current_mask is not None:
            col = COL_ANCHOR if active_template == "anchor" else COL_YOLO_T
            annotated = overlay_mask(annotated, current_mask, col, alpha=0.25)

        # 2. background person detections
        if args.show_dets:
            for (bx, by, bw, bh), _ in dets:
                cv2.rectangle(annotated,
                              (int(bx), int(by)), (int(bx+bw), int(by+bh)),
                              COL_DET_DIM, 1)

        # 2b. Kalman predicted box
        if args.show_kalman:
            kx, ky, kw, kh = [int(v) for v in kalman_pred]
            draw_dashed_rect(annotated, kx, ky, kx+kw, ky+kh, COL_KALMAN, 1, dash=6)

        # 2c. perimeter search radius circle
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

        # 4. main track box (state-dependent rendering)
        sx, sy, sw, sh = current_box
        if track_state == "lost":
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, COL_LOST, args.thickness)
            put_label(annotated, f"LOST {best_score:.2f}",
                      sx, max(14, sy-4), fg=(255,255,255), bg=COL_LOST, scale=0.45)
        elif track_state == "occluded":
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, COL_KALMAN, args.thickness, dash=6)
            put_label(annotated, f"OCCLUDED {occ_consec}/{args.occ_patience}",
                      sx, max(14, sy-4), fg=(0,0,0), bg=COL_KALMAN, scale=0.45)
        elif track_state == "partial":
            col_partial = (0, 180, 255)  # orange-ish
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, col_partial, args.thickness, dash=12)
            put_label(annotated, f"PARTIAL {best_score:.2f}",
                      sx, max(14, sy-4), fg=(0,0,0), bg=col_partial, scale=0.45)
        elif kalman_coast:
            # Velocity coast (direction reversal detected in tracked state)
            draw_dashed_rect(annotated, sx, sy, sx+sw, sy+sh, COL_KALMAN, args.thickness, dash=6)
            put_label(annotated, "V-COAST", sx, max(14, sy-4),
                      fg=(0,0,0), bg=COL_KALMAN, scale=0.45)
        elif recovery_frames > 0:
            col_recovery = (255, 165, 0)  # orange
            cv2.rectangle(annotated, (sx, sy), (sx+sw, sy+sh), col_recovery, args.thickness)
            put_label(annotated, f"RECOVERY {recovery_frames}",
                      sx, max(14, sy-4), fg=(0,0,0), bg=col_recovery, scale=0.45)
        else:
            # Normal tracked
            col = COL_ANCHOR if active_template == "anchor" else COL_YOLO_T
            lbl = f"A {score_a:.2f}" if active_template == "anchor" \
                  else f"Y {score_y:.2f}"
            draw_box_with_accents(annotated, current_box, col, args.thickness, label=lbl)

        # 5. dual score bars (top-right)
        bx0    = W - 155
        by0    = 10
        bw_bar = 130
        bh_bar = 13
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

        # ANCHOR DRIFT warning banner
        if anchor_drift:
            warn_x = bx0
            warn_y = by1 + bh_bar + 6
            warn_text = f"ANCHOR DRIFT  a/y={score_a/max(score_y,1e-9):.2f}"
            (tw, th), bl = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(annotated, (warn_x-2, warn_y-th-2),
                          (warn_x+tw+4, warn_y+bl+2), (0, 0, 180), -1)
            cv2.putText(annotated, warn_text, (warn_x+2, warn_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # dense crowd lockdown indicator
        if dense_crowd:
            lock_y = by1 + bh_bar + (26 if anchor_drift else 6)
            lock_text = f"LOCKDOWN  nearby={nearby_count}"
            put_label(annotated, lock_text, bx0, lock_y,
                      fg=(255,255,255), bg=(0, 100, 200), scale=0.38)

        # template bank indicator
        bank_offset = 20 if dense_crowd else 0
        bank_y = by1 + bh_bar + (26 if anchor_drift else 6) + bank_offset
        bank_text = f"anc:{len(anchor_bank)}/{ANCHOR_BANK_MAX}  yolo:{len(yolo_bank)}/{BANK_MAX}"
        if locked_size is not None:
            bank_text += "  SL"
        put_label(annotated, bank_text, bx0, bank_y,
                  fg=(200,200,200), bg=(40,40,40), scale=0.35)

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
                    f"tmpl={active_template}  {mask_str}  anc={len(anchor_bank)}"
                    f"  [{track_state.upper()}]",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1, cv2.LINE_AA)
        cv2.putText(annotated, "R=reselect  B=new from start  SPACE=pause  S=snap  Q=quit",
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
            elif key == ord("b"):
                do_begin_new(WIN)

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
    # To compare: change main() → main_v1()
    sys.exit(main())
