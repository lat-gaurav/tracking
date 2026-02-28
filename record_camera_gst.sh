#!/usr/bin/env bash
set -euo pipefail

DEVICE="/dev/video0"
OUTPUT="${1:-recording.mp4}"
BITRATE=4000

echo "Recording from ${DEVICE} -> ${OUTPUT}"
echo "No display window (SSH-safe). Press Ctrl+C to stop."

run_raw_pipeline() {
  gst-launch-1.0 -e \
    v4l2src device="${DEVICE}" ! \
    videoconvert ! \
    x264enc tune=zerolatency speed-preset=ultrafast bitrate=${BITRATE} ! \
    mp4mux ! \
    filesink location="${OUTPUT}"
}

run_mjpeg_pipeline() {
  gst-launch-1.0 -e \
    v4l2src device="${DEVICE}" ! \
    image/jpeg ! \
    jpegdec ! \
    videoconvert ! \
    x264enc tune=zerolatency speed-preset=ultrafast bitrate=${BITRATE} ! \
    mp4mux ! \
    filesink location="${OUTPUT}"
}

set +e
run_raw_pipeline
status=$?
set -e

if [ $status -ne 0 ]; then
  echo "Raw pipeline failed. Retrying with MJPEG decode fallback..."
  run_mjpeg_pipeline
fi

echo "Saved: ${OUTPUT}"
