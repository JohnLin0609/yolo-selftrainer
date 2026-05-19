#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$ROOT_DIR/models/pretrained"

mkdir -p "$MODELS_DIR"

BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.3.0"

TASK="all"
SIZE="nano"

while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --size) SIZE="$2"; shift 2 ;;
        --all)  TASK="all"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

NANO_MODELS=()
SMALL_MODELS=()

case "$TASK" in
    detect)
        NANO_MODELS=(yolo11n yolov8n)
        SMALL_MODELS=(yolo11s)
        ;;
    obb)
        NANO_MODELS=(yolo11n-obb yolov8n-obb)
        SMALL_MODELS=(yolo11s-obb)
        ;;
    segment)
        NANO_MODELS=(yolo11n-seg yolov8n-seg)
        SMALL_MODELS=(yolo11s-seg)
        ;;
    pose)
        NANO_MODELS=(yolo11n-pose yolov8n-pose)
        SMALL_MODELS=(yolo11s-pose)
        ;;
    classify)
        NANO_MODELS=(yolo11n-cls yolov8n-cls)
        SMALL_MODELS=(yolo11s-cls)
        ;;
    all)
        NANO_MODELS=(
            yolo11n yolov8n
            yolo11n-obb yolov8n-obb
            yolo11n-seg yolov8n-seg
            yolo11n-pose yolov8n-pose
            yolo11n-cls yolov8n-cls
        )
        SMALL_MODELS=(
            yolo11s yolo11s-obb yolo11s-seg yolo11s-pose yolo11s-cls
        )
        ;;
    *)
        echo "ERROR: Unknown task '$TASK'. Use: detect, obb, segment, pose, classify, all"
        exit 1
        ;;
esac

MODELS=("${NANO_MODELS[@]}")
if [ "$SIZE" = "small" ] || [ "$SIZE" = "all" ]; then
    MODELS+=("${SMALL_MODELS[@]}")
fi

OK=0
FAIL=0
SKIP=0

for model in "${MODELS[@]}"; do
    FILE="$MODELS_DIR/${model}.pt"
    if [ -f "$FILE" ]; then
        SKIP=$((SKIP + 1))
        continue
    fi
    URL="$BASE_URL/${model}.pt"
    # Fail-loud (Harness §二): keep wget's stderr. -q suppresses progress;
    # real errors (404, DNS, perm denied) must surface so the operator can
    # fix the root cause rather than only seeing "FAILED".
    if wget -q -nc -P "$MODELS_DIR/" "$URL"; then
        FSIZE=$(du -h "$FILE" | cut -f1)
        echo "  Downloaded: ${model}.pt ($FSIZE)"
        OK=$((OK + 1))
    else
        echo "  FAILED: ${model}.pt — see wget error above. Manual: wget $URL" >&2
        FAIL=$((FAIL + 1))
    fi
done

echo "  Models: $OK downloaded, $SKIP already present, $FAIL failed"
if [ $FAIL -gt 0 ]; then
    echo "  Manual download: $BASE_URL/<model>.pt"
fi
