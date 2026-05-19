#!/bin/bash
# Download a small demo dataset so first-time users can run the framework
# end-to-end with no manual data prep.
#
# We use Ultralytics' COCO128 (128 images from COCO val2017, ~7 MB) — it's
# their canonical "is everything working?" dataset, 80 classes, detect task.
#
# This script keeps the binary off the git repo; runs once on demand.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_DIR="$ROOT_DIR/datasets"
DEMO_DIR="$DATASETS_DIR/demo"

URL="https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
ZIP="$DATASETS_DIR/coco128.zip"

mkdir -p "$DATASETS_DIR"

if [ -d "$DEMO_DIR/images" ]; then
    echo "[demo] dataset already present at $DEMO_DIR — skipping download"
    exit 0
fi

echo "[demo] downloading COCO128 (~7 MB) from Ultralytics..."
if ! wget -q --show-progress -O "$ZIP" "$URL"; then
    echo "[demo] ERROR: download failed. Check network or grab manually from:" >&2
    echo "  $URL" >&2
    exit 1
fi

echo "[demo] extracting..."
cd "$DATASETS_DIR"
if ! unzip -q "$ZIP"; then
    echo "[demo] ERROR: unzip failed. Is 'unzip' installed? (apt install unzip)" >&2
    exit 1
fi

# coco128.zip extracts to coco128/ — rename to demo/ for friendlier naming
if [ -d "$DATASETS_DIR/coco128" ]; then
    mv "$DATASETS_DIR/coco128" "$DEMO_DIR"
fi
rm -f "$ZIP"

# Count what we got
N_IMG=$(find "$DEMO_DIR/images" -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l)
N_LBL=$(find "$DEMO_DIR/labels" -type f -name '*.txt' | wc -l)
echo "[demo] ✓ $N_IMG images / $N_LBL labels in $DEMO_DIR"
echo ""
echo "Next:"
echo "  bash start_self_training.sh --dataset $DEMO_DIR --rounds 3"
