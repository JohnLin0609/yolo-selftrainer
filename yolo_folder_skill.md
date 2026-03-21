# YOLO OBB Self-Trainer Folder Skill

## Environment

- **Python venv**: `/home/johnlin/workspace/aoi/yolov11obb/.venv/` — already has ultralytics + torch
- **Activation**: `source /home/johnlin/workspace/aoi/yolov11obb/.venv/bin/activate`
- **DO NOT run pip install ultralytics** — it is already installed (v8.4.23)
- train.sh and start_claude.sh both activate this venv automatically

## Project Context

This is an **Oriented Bounding Box (OBB)** defect detection project for industrial wheel inspection.

- **Task**: OBB detection (rotated bounding boxes, NOT standard detection)
- **YOLO command**: `yolo obb train` (NOT `yolo detect train`)
- **Class**: Single class — `defect` (class_id=0)
- **Images**: 3840×2748 PNG from Basler acA3800-14um industrial camera, trained at 640×640
- **Label format**: YOLO OBB 4-corner — `0 x1 y1 x2 y2 x3 y3 x4 y4` (normalized coordinates)
- **OBB-specific loss**: `angle_loss` tracks rotation accuracy (in addition to box/cls/dfl)

## Datasets

Three datasets available (all in `/home/johnlin/workspace/aoi/yolov11obb/dataset/`):

| Dataset | YAML | Train | Val | Total | Split? | Notes |
|---------|------|-------|-----|-------|--------|-------|
| **20260320_wheel_merge** | `dataset_merge.yaml` | 501 | 126 | 627 | YES | Start with this. Updated 2026-03-20. |
| **20260320_wheel** | `dataset_wheel.yaml` | — | — | 332 | NO | Fallback #1. Needs split before use. |
| **20260317_wheel_john** | `dataset_john.yaml` | — | — | 295 | NO | Fallback #2. Needs split before use. |

**Priority order**: merge → wheel → john

**When to switch datasets:**
- Start with **merge** (most data, 627 images)
- If mAP50 plateaus below 0.75 after 4+ runs AND overfitting gap is low → try **wheel** or **john**
- To switch:
```bash
sed -i 's|^DATASET=.*|DATASET="/home/johnlin/workspace/aoi/yolo_selftrainer/dataset_wheel.yaml"|' ./train.sh
# or
sed -i 's|^DATASET=.*|DATASET="/home/johnlin/workspace/aoi/yolo_selftrainer/dataset_john.yaml"|' ./train.sh
```

**IMPORTANT — Splitting unsplit datasets before use:**
If switching to `wheel` or `john`, you MUST split them first (images are flat, not in train/val dirs).
Run this Python script to split:
```bash
python3 -c "
import random, shutil
from pathlib import Path
dataset_dir = Path('/home/johnlin/workspace/aoi/yolov11obb/dataset/DATASET_NAME_HERE')
src_images = dataset_dir / 'images'
src_labels = dataset_dir / 'labels'
if (src_images / 'train').is_dir():
    print('Already split'); exit()
exts = ('.png', '.jpg', '.jpeg')
images = sorted([f for f in src_images.iterdir() if f.suffix.lower() in exts])
random.seed(0); random.shuffle(images)
split_idx = max(1, int(len(images) * 0.8))
for subset, img_list in [('train', images[:split_idx]), ('val', images[split_idx:])]:
    (src_images / subset).mkdir(exist_ok=True)
    (src_labels / subset).mkdir(exist_ok=True)
    for img in img_list:
        shutil.move(str(img), str(src_images / subset / img.name))
        lbl = src_labels / (img.stem + '.txt')
        if lbl.exists(): shutil.move(str(lbl), str(src_labels / subset / lbl.name))
print(f'Split: train={split_idx}, val={len(images)-split_idx}')
"
```
Replace `DATASET_NAME_HERE` with `20260320_wheel` or `20260317_wheel_john`.

## Pretrained Models

All in `/home/johnlin/workspace/aoi/yolov11obb/models/pretrained/`:

| Model | Path | Size | Notes |
|-------|------|------|-------|
| **yolo11n-obb.pt** | `.../models/pretrained/yolo11n-obb.pt` | 5.6M | Best mAP50 on merge (0.784). Start here. |
| **yolov8n-obb.pt** | `.../models/pretrained/yolov8n-obb.pt` | 6.3M | Close second (0.781 on merge). Mature arch. |
| ~~yolo26n-obb.pt~~ | — | — | Do NOT use — underperforms on this task |

## Previous Baseline Results (from yolov11obb project)

All trained with: 2000 epochs, batch=-1, imgsz=640, patience=500, default augmentation.

| Model × Dataset | Best mAP50 | Best Epoch | Total Epochs |
|----------------|-----------|------------|--------------|
| v11 × merge | **0.7840** | 376 | 945 |
| v8 × merge | 0.7810 | — | — |
| v11 × john | 0.7570 | — | — |
| v8 × john | 0.7707 | — | — |
| ~~v26 × merge~~ | ~~0.7152~~ | — | Do not use |
| ~~v26 × john~~ | ~~0.7373~~ | — | Do not use |

**Target**: Beat 0.784 mAP50. Then push toward 0.85+.

## Project Layout

```
/home/johnlin/workspace/aoi/yolo_selftrainer/   ← scripts & config live here
├── start_claude.sh
├── train.sh                          # Edit params with sed
├── hyperparameter_strategy.md
├── yolo_folder_skill.md
├── next_instruction.md               # Memory between sessions
├── dataset_merge.yaml
├── dataset_john.yaml
├── train.pid                         # PID of running training job
└── current.log                       # Stdout/stderr of current training

/home/johnlin/workspace/aoi/yolov11obb/runs/obb/  ← training outputs go here
└── train_YYYYMMDD_HHMMSS/
    ├── results.csv                   # Epoch-by-epoch metrics
    ├── args.yaml                     # Hyperparameters used
    └── weights/
        ├── best.pt                   # Best checkpoint
        └── last.pt                   # Last checkpoint
```

## results.csv Columns (OBB-Specific)

```
epoch, time,
train/box_loss, train/cls_loss, train/dfl_loss, train/angle_loss,
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
val/box_loss, val/cls_loss, val/dfl_loss, val/angle_loss,
lr/pg0, lr/pg1, lr/pg2
```

**Key columns:**
- `metrics/mAP50(B)` → primary quality signal (column 9, 1-indexed)
- `metrics/mAP50-95(B)` → stricter metric (column 10)
- `metrics/precision(B)` → column 7
- `metrics/recall(B)` → column 8
- `train/box_loss` vs `val/box_loss` → overfitting gap
- `train/angle_loss` vs `val/angle_loss` → OBB rotation accuracy gap
- `val/angle_loss` → should decrease; if rising while box_loss falls, rotation fitting is degrading

## Find the Latest Run

```bash
LATEST=$(ls -td /home/johnlin/workspace/aoi/yolov11obb/runs/obb/train_*/ 2>/dev/null | head -1)
```

## Read Results

```bash
# Header
head -1 "$LATEST/results.csv"
# Last 5 epochs
tail -5 "$LATEST/results.csv"
# Best mAP50 epoch
awk -F',' 'NR>1 {if($9+0 > max) {max=$9+0; line=$0}} END {print max, line}' "$LATEST/results.csv"
# Check total epochs trained (if < EPOCHS, patience triggered early stop)
tail -1 "$LATEST/results.csv" | cut -d',' -f1
```

## Check Params Used

```bash
cat "$LATEST/args.yaml"
```

## Find Best Weights

```bash
BEST_PT="$LATEST/weights/best.pt"
[ -f "$BEST_PT" ] && echo "Found: $BEST_PT" || echo "NOT FOUND"
```

## Edit train.sh (ALWAYS use sed, NEVER rewrite the file)

```bash
# Learning rate
sed -i 's/^LR=.*/LR=0.005/' ./train.sh
sed -i 's/^LR_FINAL=.*/LR_FINAL=0.005/' ./train.sh

# Epochs and patience
sed -i 's/^EPOCHS=.*/EPOCHS=150/' ./train.sh
sed -i 's/^PATIENCE=.*/PATIENCE=40/' ./train.sh

# Model weights (pretrained or previous best.pt)
sed -i 's|^WEIGHTS=.*|WEIGHTS=${1:-"/home/johnlin/workspace/aoi/yolov11obb/models/pretrained/yolov8n-obb.pt"}|' ./train.sh

# Optimizer
sed -i 's/^OPTIMIZER=.*/OPTIMIZER="AdamW"/' ./train.sh

# Augmentation
sed -i 's/^MOSAIC=.*/MOSAIC=0.8/' ./train.sh
sed -i 's/^MIXUP=.*/MIXUP=0.1/' ./train.sh
sed -i 's/^DEGREES=.*/DEGREES=15.0/' ./train.sh
sed -i 's/^SCALE=.*/SCALE=0.5/' ./train.sh
sed -i 's/^ERASING=.*/ERASING=0.4/' ./train.sh
sed -i 's/^CLOSE_MOSAIC=.*/CLOSE_MOSAIC=10/' ./train.sh

# Image size (for small defect experiments)
sed -i 's/^IMGSZ=.*/IMGSZ=1024/' ./train.sh
sed -i 's/^BATCH=.*/BATCH=8/' ./train.sh

# Switch dataset to john
sed -i 's|^DATASET=.*|DATASET="/home/johnlin/workspace/aoi/yolo_selftrainer/dataset_john.yaml"|' ./train.sh
```

## Verify Edits

```bash
grep -E '^(WEIGHTS|EPOCHS|LR|LR_FINAL|BATCH|IMGSZ|PATIENCE|OPTIMIZER|MOSAIC|MIXUP|DEGREES|DATASET)=' ./train.sh
```

## Launch Training (ALWAYS detached)

```bash
nohup bash ./train.sh /path/to/weights.pt > /home/johnlin/workspace/aoi/yolo_selftrainer/current.log 2>&1 &
echo $! > /home/johnlin/workspace/aoi/yolo_selftrainer/train.pid
```

**NEVER run train.sh without `nohup &`** — it blocks the session.

When fine-tuning from a previous best.pt:
```bash
BEST_PT=$(ls -td /home/johnlin/workspace/aoi/yolov11obb/runs/obb/train_*/ | head -1)/weights/best.pt
nohup bash ./train.sh "$BEST_PT" > /home/johnlin/workspace/aoi/yolo_selftrainer/current.log 2>&1 &
echo $! > /home/johnlin/workspace/aoi/yolo_selftrainer/train.pid
```

## Check if Training is Already Running

```bash
PID=$(cat /home/johnlin/workspace/aoi/yolo_selftrainer/train.pid 2>/dev/null)
kill -0 $PID 2>/dev/null && echo "Already running" || echo "Not running"
```

If already running: do **NOT** launch another. Exit immediately.

## Write next_instruction.md (do this BEFORE every exit)

This is the ONLY memory your next self has. Be specific, detailed, and include everything needed to make the next decision.

**REQUIRED sections:**

```
---
## Current training
Run folder: <EXACT folder name, e.g., 20260320_170000_claude_v11>
  Located at: /home/johnlin/workspace/aoi/yolov11obb/runs/obb/<folder_name>/
  Status: IN PROGRESS (or COMPLETED if you read the final results)
  Weights used: <full path to .pt file used>
  Key params: epochs=X lr=X lrf=X batch=X imgsz=X patience=X optimizer=X
  Augmentation: degrees=X mosaic=X mixup=X flipud=X fliplr=X erasing=X scale=X

## Run history & analysis
Run #1 — <folder_name> | <model> | <dataset>
  mAP50: X.XX | mAP50-95: X.XX | P: X.XX | R: X.XX
  val_box: X.XX | train_box: X.XX | val_angle: X.XX | train_angle: X.XX
  Epochs trained: X/X (patience triggered? or full?)
  Analysis: <what this result tells us — overfitting? underfitting? good fit?>
  Decision: <what was changed and WHY based on the analysis>
    e.g., "mAP50 improved 0.72→0.79 (+0.07) but val_box gap widening (1.85 vs 0.48),
    suggesting early overfitting. Reduced LR from 0.01→0.005 to slow convergence
    and resumed from best.pt to keep the good features learned."

Run #2 — <folder_name> | <model> | <dataset>
  ... (same format, always include analysis and reasoning)

## Best result so far
Best mAP50: X.XX from run #N (<folder_name>)
Best weights: /home/johnlin/workspace/aoi/yolov11obb/runs/obb/<folder_name>/weights/best.pt
Key params that achieved this: lr=X imgsz=X degrees=X optimizer=X ...

## Dataset
Currently using: <which yaml> (<N> images)
Switch consideration: <reason if applicable, or "none — results still improving">

## What I already changed in train.sh
<List EXACT sed commands already applied this round. These are FACTS — verify they are still in train.sh.>
e.g.:
  - sed -i 's/^DEGREES=.*/DEGREES=15.0/' → DEGREES now 15.0
  - Weights arg: using /path/to/best.pt from run #2

## Recommended action for next round
<What the NEXT Claude should do, with reasoning. This is a SUGGESTION — the next Claude
should read the actual results first and may override if the data says otherwise.>
e.g.:
  "If DEGREES=15 improved mAP50 → keep it and try IMGSZ=1024 next.
   If DEGREES=15 hurt mAP50 → revert to DEGREES=0 and try IMGSZ=1024 instead.
   If mAP50 > 0.85 → trigger Action K (long production run)."

## Next task
1. Check if training <folder_name> is still running:
   PID=$(cat /home/johnlin/workspace/aoi/yolo_selftrainer/train.pid 2>/dev/null)
   kill -0 $PID 2>/dev/null && echo "Still running" || echo "Done"
2. If still running: EXIT immediately, do not launch another.
3. If done: read results from /home/johnlin/workspace/aoi/yolov11obb/runs/obb/<folder_name>/results.csv
4. Compare with best result so far (mAP50=X.XX from run #N)
5. READ "Recommended action for next round" above — consider this advice seriously.
6. Then check the actual results against hyperparameter_strategy.md decision tree.
   If results confirm the recommendation → follow it.
   If results contradict it → override with reasoning and explain why in your Decision.
7. Edit train.sh with sed. Launch:
   nohup bash /home/johnlin/workspace/aoi/yolo_selftrainer/train.sh <weights> > /home/johnlin/workspace/aoi/yolo_selftrainer/current.log 2>&1 &
   echo $! > /home/johnlin/workspace/aoi/yolo_selftrainer/train.pid
8. Rewrite next_instruction.md with updated run history before exiting.

## Watch out for
- <specific anomalies, e.g., "angle_loss has been rising since run #3">
- <what to try next if current approach fails>
- <any observations about the dataset, e.g., "precision is low on merge, may be label noise">
---
```

## OBB-Specific Considerations

1. **angle_loss** is unique to OBB — track it alongside box_loss. If angle_loss plateaus or rises while box_loss falls, the model is losing rotation accuracy.
2. **DEGREES augmentation** matters for OBB — unlike standard detection, rotation augmentation can help the model learn better orientation. Try 10–30° if not overfitting.
3. **High-res images** (3840×2748) downsampled to 640 — lots of detail lost. If small defects are missed, try `IMGSZ=1024` or `IMGSZ=1280` (reduce BATCH accordingly).
4. **Single class** means cls_loss should converge quickly — focus attention on box_loss and angle_loss.
5. **38 background images** (no labels) in merge train set — this is fine, teaches the model to not hallucinate detections.

## Stop Conditions — Write STOP in next_instruction.md, do NOT launch training

- dataset.yaml missing or has broken paths
- Training job already running (PID alive)
- GPU out-of-memory that persists after reducing batch
- 6 consecutive runs with mAP50 change < 0.01
- Any NaN in loss columns
- angle_loss > 0.1 (rotation learning has collapsed)
- val_box_loss increasing for 3+ consecutive runs (severe overfitting)
