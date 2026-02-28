#!/usr/bin/env python3
"""
Generate a harder version of a tracking video with three adversarial augmentations:

1. Occlusion bar   – a wide opaque rectangle sweeps across the frame,
                     fully covering the tracked person for ~2 seconds.
2. Blink dropout   – random "blackout" bursts over the person region.
3. Motion blur     – heavy directional blur applied to the whole frame.

The source clip is trimmed to frames 0–CLIP_END, then looped N times
to produce a longer consistent video before augmentations are applied.

Usage:
    python3 scratch/make_challenge.py
    python3 scratch/make_challenge.py --clip-end 300 --loops 4
    python3 scratch/make_challenge.py --input video_test/foo.mp4 --output scratch/challenge.mp4
"""
import argparse, random
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent

DEFAULT_INPUT  = str(_HERE / ".." / "video_test" / "13657722_640x640.mp4")
DEFAULT_OUTPUT = str(_HERE / "challenge.mp4")

SEED = 42


def make_blur_kernel(angle_deg: float, length: int) -> np.ndarray:
    """Return a motion-blur kernel of given length and angle."""
    k = np.zeros((length, length), dtype=np.float32)
    cx = length // 2
    for i in range(length):
        x = int(cx + (i - cx) * np.cos(np.radians(angle_deg)))
        y = int(cx + (i - cx) * np.sin(np.radians(angle_deg)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    default=DEFAULT_INPUT)
    ap.add_argument("--output",   default=DEFAULT_OUTPUT)
    ap.add_argument("--clip-end", type=int, default=300,
                    help="Use frames 0..CLIP_END-1 from the source (default: 300)")
    ap.add_argument("--loops",    type=int, default=4,
                    help="How many times to repeat the clip (default: 4)")
    args = ap.parse_args()

    rng = random.Random(SEED)
    np.random.seed(SEED)

    cap = cv2.VideoCapture(str(Path(args.input).resolve()))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")

    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps       = cap.get(cv2.CAP_PROP_FPS) or 50.0
    W         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_end = min(args.clip_end, src_total)  # clamp to actual length

    # ── pre-read clip frames into memory ─────────────────────────────────────
    print(f"Reading frames 0–{clip_end-1} from source...", flush=True)
    clip: list[np.ndarray] = []
    for i in range(clip_end):
        ok, frame = cap.read()
        if not ok:
            break
        clip.append(frame)
    cap.release()

    clip_end = len(clip)   # actual frames read (may be < requested)

    # build full looped frame index sequence
    frame_seq = [clip[i % clip_end] for _ in range(args.loops) for i in range(clip_end)]
    total = len(frame_seq)   # clip_end * loops

    print(f"Clip : {clip_end} frames  x{args.loops} loops = {total} frames total  "
          f"({total/fps:.1f} s @ {fps} fps)")

    out = cv2.VideoWriter(
        str(Path(args.output).resolve()),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )
    if not out.isOpened():
        raise SystemExit(f"Cannot open output {args.output}")

    # ── pre-compute challenge parameters (scaled to full looped length) ──────

    # 1. OCCLUSION EVENTS: multiple bars across all four loops
    #    Each entry: (start_frame, end_frame, bar_w_frac, bar_h_frac, bar_y_frac, colour, left_to_right)
    #    Spread one per loop at different offsets so the tracker gets hit repeatedly
    occlusion_events = [
        # loop 1 — short, narrow, left→right  (1.6 s)
        (clip_end * 0 + 50,  clip_end * 0 + 130, 0.30, 0.50, 0.25, (60,  60, 200), True),
        # loop 2 — long, wide, right→left     (3.0 s)
        (clip_end * 1 + 30,  clip_end * 1 + 180, 0.45, 0.60, 0.20, (30, 160,  60), False),
        # loop 3 — medium, tall, left→right   (2.0 s)
        (clip_end * 2 + 80,  clip_end * 2 + 180, 0.35, 0.70, 0.15, (180, 50,  50), True),
        # loop 4 — very long full-width freeze (4.0 s)
        (clip_end * 3 + 20,  clip_end * 3 + 220, 0.60, 0.55, 0.22, (50,  50, 180), False),
    ]

    # pre-compute sweep x-positions for each event
    def make_bar_xs(start, end, bar_w, left_to_right):
        n = end - start
        if left_to_right:
            return np.linspace(-bar_w, W, n).astype(int)
        else:
            return np.linspace(W, -bar_w, n).astype(int)

    occ_data = []
    for (s, e, wf, hf, yf, col, ltr) in occlusion_events:
        bw = int(W * wf)
        bh = int(H * hf)
        by = int(H * yf)
        xs = make_bar_xs(s, e, bw, ltr)
        occ_data.append((s, e, bw, bh, by, col, xs))

    # 2. BLINK DROPOUT: spread across the full looped video
    BLINK_REGION = (270, 140, 340, 280)
    blink_frames: set[int] = set()
    for _ in range(16):
        start = rng.randint(10, total - 10)
        length = rng.randint(3, 10)
        for f in range(start, min(start + length, total)):
            blink_frames.add(f)

    # 3. MOTION BLUR: placed in third loop
    BLUR_START = clip_end * 2 + 50
    BLUR_END   = clip_end * 2 + 150
    BLUR_LEN = 29       # kernel size (must be odd)
    BLUR_ANGLE = 0.0    # horizontal blur
    blur_kernel = make_blur_kernel(BLUR_ANGLE, BLUR_LEN)

    print(f"Input : {args.input}  ({W}x{H} @ {fps} fps, {src_total} src frames)")
    print(f"Output: {args.output}")
    print(f"Challenges:")
    occ_summary = "  ".join(f"[{s}-{e}]" for s, e, *_ in occlusion_events)
    print(f"  Occlusions     : {len(occlusion_events)} events — {occ_summary}")
    print(f"  Blink dropout  : {len(blink_frames)} frames randomly blacked out (all loops)")
    print(f"  Motion blur    : frames {BLUR_START}-{BLUR_END}  (loop 3)")

    for fidx, frame in enumerate(frame_seq):
        f = frame.copy()

        # ── 4. motion blur (applied first so other effects are still sharp) ──
        if BLUR_START <= fidx < BLUR_END:
            # ramp in/out for naturalness
            t = (fidx - BLUR_START) / (BLUR_END - BLUR_START)
            intensity = np.sin(t * np.pi)     # 0→1→0
            blurred = cv2.filter2D(f, -1, blur_kernel)
            f = cv2.addWeighted(f, 1 - intensity, blurred, intensity, 0)

        # ── 2. blink dropout ──────────────────────────────────────────────────
        if fidx in blink_frames:
            x1, y1, x2, y2 = BLINK_REGION
            x1, x2 = max(0, x1), min(W, x2)
            y1, y2 = max(0, y1), min(H, y2)
            f[y1:y2, x1:x2] = 0

        # ── 1. occlusion bars (multiple events) ──────────────────────────────
        for (s, e, bw, bh, by, col, xs) in occ_data:
            if s <= fidx < e:
                bx = xs[fidx - s]
                x1 = max(0, bx)
                x2 = min(W, bx + bw)
                if x2 > x1:
                    f[by : by + bh, x1:x2] = col

        out.write(f)

        if fidx % 100 == 0:
            print(f"  {fidx}/{total} frames processed...")

    out.release()

    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nDone. Saved: {args.output}  ({size_mb:.1f} MB)")
    print(f"Now run the tracker on it:")
    print(f"  python3 scratch/track.py --source scratch/challenge.mp4 --save scratch/challenge_tracked.mp4")


if __name__ == "__main__":
    main()
