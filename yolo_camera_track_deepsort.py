#!/usr/bin/env python3
import argparse
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as exc:  # pragma: no cover
    print("ERROR: deep-sort-realtime is not available.", file=sys.stderr)
    print("Install with: pip install deep-sort-realtime", file=sys.stderr)
    raise SystemExit(1) from exc


DEFAULT_WEIGHTS = "/home/jetson2/gaurav/models/yolon26obb-newmerged-imgsz640-epochs75_fp16.engine"


class AsyncVideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int], max_queue: int = 240):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, float(fps), frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open output video {output_path}")

        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop_token = object()
        self.written_frames = 0
        self.dropped_frames = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is self._stop_token:
                    return
                self.writer.write(item)
                self.written_frames += 1
            finally:
                self.queue.task_done()

    def write(self, frame) -> None:
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped_frames += 1

    def close(self) -> None:
        self.queue.put(self._stop_token)
        self.queue.join()
        self.thread.join(timeout=5.0)
        self.writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ultralytics YOLO + DeepSORT (ReID) on camera input.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Detection model path (.engine/.pt)")
    parser.add_argument("--device", default="/dev/video0", help="Camera device path")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--embedder", default=None,
                        help="DeepSORT ReID embedder: mobilenet, torchreid, clip_RN50, clip_ViT-B/32, etc. Default: None (IoU-only, no ReID)")
    parser.add_argument("--max-age", type=int, default=30, help="Max missed frames before track deletion")
    parser.add_argument("--n-init", type=int, default=3, help="Frames needed before a track is confirmed")
    parser.add_argument("--max-iou-distance", type=float, default=0.7, help="Max IoU distance for association")
    parser.add_argument("--show", action="store_true", help="Show live annotated window")
    parser.add_argument("--save", default="", help="Optional output video path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.device}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if cam_fps <= 1.0:
        cam_fps = 60.0

    writer = None
    if args.save:
        try:
            writer = AsyncVideoWriter(args.save, cam_fps, (width, height))
        except RuntimeError as exc:
            cap.release()
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    detector = YOLO(args.weights)
    tracker = DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=args.max_iou_distance,
        embedder=args.embedder,     # None = pure IoU, otherwise e.g. 'mobilenet'
        half=args.embedder is not None,
        bgr=True,
    )

    running = True

    def stop_handler(signum, frame):  # noqa: ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    print(f"Camera: {args.device} ({width}x{height} @ ~{cam_fps:.2f} fps)")
    print(f"Detection model: {args.weights}")
    reid_on = args.embedder is not None
    print("Tracker backend: deepsort")
    print(f"ReID enabled: {reid_on}")
    print(f"ReID model/embedder: {args.embedder if reid_on else 'None (IoU-only tracking)'}")
    if not reid_on:
        print("Tip: pass --embedder mobilenet to enable appearance ReID")
    else:
        print("WARNING: If active_trackers stays 0, embedder may be crashing silently; omit --embedder to use IoU-only mode.")
    if args.show:
        print("Press q in window or Ctrl+C in terminal to stop.")
    else:
        print("Running headless. Press Ctrl+C to stop.")

    processed = 0
    started = time.time()
    last_print = started

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: Camera frame read failed, stopping.", file=sys.stderr)
                break

            result = detector.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]

            detections = []
            det_count = 0
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                for box, score, cls_id in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = box.tolist()
                    w_box = max(0.0, x2 - x1)
                    h_box = max(0.0, y2 - y1)
                    # skip crops too small for embedder to process
                    if w_box < 4 or h_box < 4:
                        continue
                    detections.append(([x1, y1, w_box, h_box], float(score), int(cls_id)))
                det_count = len(detections)

            # IoU-only mode: pass zero-vector embeddings so cosine distance stays valid
            if args.embedder is None:
                zero_embed = np.zeros(128, dtype=np.float32)
                update_kwargs = {"embeds": [zero_embed] * len(detections)}
            else:
                update_kwargs = {}
            tracks = tracker.update_tracks(detections, frame=frame, **update_kwargs)
            active_tracks = [track for track in tracks if track.is_confirmed()]

            annotated = frame
            for track in active_tracks:
                ltrb = track.to_ltrb()
                tid = track.track_id
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"ID {tid}",
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            processed += 1
            now = time.time()
            fps_now = processed / max(now - started, 1e-9)
            cv2.putText(
                annotated,
                f"FPS: {fps_now:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("YOLO + DeepSORT", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if now - last_print >= 1.0:
                print(
                    f"Processed={processed} | infer_fps={fps_now:.2f} "
                    f"| detections={det_count} | active_trackers={len(active_tracks)}"
                )
                last_print = now
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        cv2.destroyAllWindows()

    total = max(time.time() - started, 1e-9)
    print(f"Done. Frames={processed} in {total:.2f}s ({processed / total:.2f} fps)")
    if writer is not None:
        print(f"Saved: {args.save}")
        print(f"Writer frames={writer.written_frames} | dropped={writer.dropped_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
