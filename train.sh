#!/bin/bash
# Do NOT use set -e — we need to catch training failures and still wake Claude

# Activate the yolov11obb venv (has ultralytics + torch installed)
source /home/johnlin/workspace/aoi/yolov11obb/.venv/bin/activate

DATASET="/home/johnlin/workspace/aoi/yolo_selftrainer/dataset_merge.yaml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Parameters — edit these each cycle using sed
WEIGHTS=${1:-"/home/johnlin/workspace/aoi/yolov11obb/models/pretrained/yolo11n-obb.pt"}
EPOCHS=200
LR=0.001
LR_FINAL=0.001
BATCH=-1
IMGSZ=1024
PATIENCE=60
OPTIMIZER="auto"
MOMENTUM=0.937
WEIGHT_DECAY=0.0005
WARMUP_EPOCHS=3.0
WARMUP_MOMENTUM=0.8
COS_LR=false
HSV_H=0.015
HSV_S=0.7
HSV_V=0.4
DEGREES=0.0
TRANSLATE=0.1
SCALE=0.5
FLIPLR=0.5
FLIPUD=0.5
MOSAIC=1.0
MIXUP=0.0
COPY_PASTE=0.0
ERASING=0.4
CLOSE_MOSAIC=10

# ─── Generate run name: YYYYMMDD_HHMMSS_claude_MODEL ────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Extract model short name from weights path (e.g., yolo11n-obb.pt → v11, yolov8n-obb.pt → v8)
if echo "$WEIGHTS" | grep -q "best.pt"; then
  # Fine-tuning from a previous run's best.pt — try to extract model from parent dir name
  PARENT_DIR=$(basename "$(dirname "$(dirname "$WEIGHTS")")")
  MODEL_SHORT=$(echo "$PARENT_DIR" | grep -oE 'v[0-9]+' | head -1)
  MODEL_SHORT="${MODEL_SHORT:-ft}_ft"
else
  MODEL_SHORT=$(basename "$WEIGHTS" | sed -E 's/.*yolo[v]?([0-9]+).*/v\1/')
fi
RUN_NAME="${TIMESTAMP}_claude_${MODEL_SHORT}"

# Write run name immediately so start_claude.sh can find it for logging
echo "$RUN_NAME" > "$SCRIPT_DIR/last_run_name"

echo "[train] Run: $RUN_NAME | weights: $WEIGHTS | lr: $LR | epochs: $EPOCHS"

yolo obb train \
  data="$DATASET" \
  model="$WEIGHTS" \
  epochs="$EPOCHS" \
  lr0="$LR" \
  lrf="$LR_FINAL" \
  batch="$BATCH" \
  imgsz="$IMGSZ" \
  patience="$PATIENCE" \
  optimizer="$OPTIMIZER" \
  momentum="$MOMENTUM" \
  weight_decay="$WEIGHT_DECAY" \
  warmup_epochs="$WARMUP_EPOCHS" \
  warmup_momentum="$WARMUP_MOMENTUM" \
  cos_lr="$COS_LR" \
  hsv_h="$HSV_H" \
  hsv_s="$HSV_S" \
  hsv_v="$HSV_V" \
  degrees="$DEGREES" \
  translate="$TRANSLATE" \
  scale="$SCALE" \
  fliplr="$FLIPLR" \
  flipud="$FLIPUD" \
  mosaic="$MOSAIC" \
  mixup="$MIXUP" \
  copy_paste="$COPY_PASTE" \
  erasing="$ERASING" \
  close_mosaic="$CLOSE_MOSAIC" \
  project="/home/johnlin/workspace/aoi/yolov11obb/runs/obb" \
  name="$RUN_NAME" \
  exist_ok=false \
  amp=true \
  device=0 \
  workers=8

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
  echo "[train] FAILED (exit code $TRAIN_EXIT) — waking Claude to diagnose..."
  PARAMS=$(grep -E '^(WEIGHTS|EPOCHS|LR|LR_FINAL|BATCH|IMGSZ|PATIENCE|OPTIMIZER|DATASET)=' "$SCRIPT_DIR/train.sh" 2>/dev/null || echo "(could not read train.sh params)")
  cat > "$SCRIPT_DIR/next_instruction.md" <<CRASH_EOF
## Training CRASHED

Run $RUN_NAME failed with exit code $TRAIN_EXIT.

## Diagnose
1. Check the log: cat $SCRIPT_DIR/current.log | tail -50
2. Look for: OOM, NaN, CUDA error, dataset error, missing file
3. Check if a partial run exists: ls -la /home/johnlin/workspace/aoi/yolov11obb/runs/obb/$RUN_NAME/
4. If OOM: reduce BATCH or IMGSZ in train.sh with sed
5. If other error: fix and relaunch
6. If unfixable: write STOP in next_instruction.md

## Previous params
$PARAMS

Rewrite next_instruction.md with full history before exiting.
CRASH_EOF
else
  echo "[train] Done: $RUN_NAME"
fi

echo "[train] Waking Claude..."
bash "$SCRIPT_DIR/start_claude.sh"
