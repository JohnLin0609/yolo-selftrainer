#!/bin/bash
# Scaffold a new YOLO self-training project from templates.
# Usage: bash scripts/new_project.sh --dataset PATH [options]
# Minimal: bash scripts/new_project.sh --dataset /path/to/dataset
# Everything else is auto-detected: task, classes, resolution, model, split.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATES_DIR="$ROOT_DIR/templates"

# ─── Defaults ─────────────────────────────────────────────────────────
NAME=""
TASK=""
DATASET=""
MODEL=""
FALLBACK=""
IMGSZ=""
CLASSES=""
IMAGE_INFO=""
BASELINE=""
MAX_ROUNDS=10
DEVICE=0
KPT_SHAPE=""
FORCE=false
AUTO_SPLIT=true
NO_AUTO_SPLIT=false
# P6 multi-LLM agent mode. Default "claude" preserves the original Claude-CLI
# flow; "agent" scaffolds start_agent.sh + agent.env (used by run_agent.py).
LOOP_MODE="claude"
LLM_PROVIDER="anthropic"
LLM_MODEL="claude-opus-4-7"
LLM_API_BASE=""
# Held-out test split (agent-invisible). 0 disables. Seed locks reproducibility.
TEST_SPLIT="0.15"
TEST_SEED="42"
# Whether the operator passed --test-seed explicitly. Used in strict-heldout
# mode to decide between auto-randomize (default) and pinned-seed.
TEST_SEED_EXPLICIT=false
# Baseline-mode RNG seed (only used when --mode baseline). Same seed → same
# random-search trajectory, so two baseline projects are byte-for-byte equal.
BASELINE_SEED="42"
# LeetCode-mode held-out test split. When true:
#   - dataset.eval.yaml carries the test: key, dataset.yaml does NOT
#   - .heldout_strict marker activates claude_bash_guard's heldout patterns
#   - .heldout_seed records the (possibly randomized) test seed
#   - heldout-cut event emitted for the audit trail
STRICT_HELDOUT=false

# ─── Parse arguments ─────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --name)       NAME="$2"; shift 2 ;;
        --task)       TASK="$2"; shift 2 ;;
        --dataset)    DATASET="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --fallback)   FALLBACK="$2"; shift 2 ;;
        --imgsz)      IMGSZ="$2"; shift 2 ;;
        --classes)    CLASSES="$2"; shift 2 ;;
        --image-info) IMAGE_INFO="$2"; shift 2 ;;
        --baseline)   BASELINE="$2"; shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --kpt-shape)  KPT_SHAPE="$2"; shift 2 ;;
        --mode)         LOOP_MODE="$2"; shift 2 ;;
        --llm-provider) LLM_PROVIDER="$2"; shift 2 ;;
        --llm-model)    LLM_MODEL="$2"; shift 2 ;;
        --llm-api-base) LLM_API_BASE="$2"; shift 2 ;;
        --test-split)    TEST_SPLIT="$2"; shift 2 ;;
        --test-seed)     TEST_SEED="$2"; TEST_SEED_EXPLICIT=true; shift 2 ;;
        --baseline-seed) BASELINE_SEED="$2"; shift 2 ;;
        --strict-heldout) STRICT_HELDOUT=true; shift ;;
        --force)         FORCE=true; shift ;;
        --auto-split)    AUTO_SPLIT=true; shift ;;
        --no-auto-split) NO_AUTO_SPLIT=true; AUTO_SPLIT=false; shift ;;
        -h|--help)
            echo "Usage: bash scripts/new_project.sh --dataset PATH [options]"
            echo ""
            echo "Required:"
            echo "  --dataset PATH     Path to dataset directory"
            echo ""
            echo "Auto-detected (override with flags):"
            echo "  --name NAME        Project name (default: dataset folder name)"
            echo "  --task TASK        detect, obb, segment, pose, classify (default: auto-detect from labels)"
            echo "  --classes STR      Class mapping (default: from classes.txt or labels)"
            echo "  --imgsz N          Image size (default: 1024 if >1920px, else 640)"
            echo "  --image-info STR   Image description (default: auto-detect from first image)"
            echo ""
            echo "Optional:"
            echo "  --model FILE       Primary model filename (default: auto per task)"
            echo "  --fallback FILE    Fallback model filename"
            echo "  --baseline N       Baseline metric value to beat"
            echo "  --max-rounds N     Max training rounds (default: 10)"
            echo "  --device N         GPU device (default: 0)"
            echo "  --kpt-shape STR    Keypoint shape for pose (e.g., '17 3')"
            echo "  --force            Overwrite existing project"
            echo "  --no-auto-split    Don't auto-split flat datasets (default: auto-split)"
            echo "  --test-split RATIO Held-out test split fraction (default: 0.15; 0 disables)"
            echo "                     Test images are agent-invisible — used for post-hoc"
            echo "                     validation only, never feed into prompts."
            echo "  --test-seed SEED   RNG seed locking the test split (default: 42)"
            echo "  --mode MODE        claude | agent | baseline (default: claude)"
            echo "                     baseline = LLM-free control loop using"
            echo "                     scripts/baseline_policy.py to pick params."
            echo "  --baseline-seed N  RNG seed for --mode baseline (default: 42)"
            echo "  --strict-heldout   LeetCode mode: agent process cannot read the"
            echo "                     held-out test split. dataset.yaml loses its"
            echo "                     test: key (moved to dataset.eval.yaml), the"
            echo "                     Bash guard rejects yolo val split=test and"
            echo "                     reads under datasets/<name>/(images|labels)/test/."
            echo "                     The agent can submit via scripts/run_test_tool.py"
            echo "                     once per round, getting only a single aggregate"
            echo "                     score. When set without --test-seed, the test"
            echo "                     split is freshly randomized per scaffold."
            exit 0
            ;;
        *) echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validate required args ──────────────────────────────────────────
if [ -z "$DATASET" ]; then echo "ERROR: --dataset is required."; exit 1; fi

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET"
    exit 1
fi
# Fail-loud (Harness §二): after the dir check, realpath failure is unexpected.
RESOLVED="$(realpath "$DATASET")"
if [ -z "$RESOLVED" ]; then
    echo "ERROR: realpath failed on $DATASET" >&2
    exit 1
fi
DATASET="$RESOLVED"

# ─── Auto-detect name from dataset folder ─────────────────────────────
if [ -z "$NAME" ]; then
    NAME=$(basename "$DATASET" | sed 's/[^a-zA-Z0-9_-]/_/g')
    echo "[AUTO] Project name: $NAME (from dataset folder)"
fi

# ─── Auto-detect task from label format ───────────────────────────────
if [ -z "$TASK" ]; then
    echo "[AUTO] Detecting task type from labels..."
    # Check for classify (folder-per-class, no labels dir)
    if [ -d "$DATASET/train" ] && [ ! -d "$DATASET/labels" ] && [ ! -d "$DATASET/images" ]; then
        TASK="classify"
    elif [ -d "$DATASET/images" ] && [ ! -d "$DATASET/labels" ]; then
        TASK="classify"
    else
        # Detect from label file field count
        LABEL_DIR="$DATASET/labels"
        [ -d "$LABEL_DIR/train" ] && LABEL_DIR="$LABEL_DIR/train"
        FIRST_LABEL=$(find "$LABEL_DIR" -name "*.txt" -type f ! -empty 2>/dev/null | head -1)
        if [ -z "$FIRST_LABEL" ]; then
            echo "ERROR: No non-empty label files found. Cannot auto-detect task."
            echo "Specify --task manually: detect, obb, segment, pose, classify"
            exit 1
        fi
        FIELD_COUNT=$(awk '{print NF; exit}' "$FIRST_LABEL")
        case "$FIELD_COUNT" in
            5)  TASK="detect" ;;
            9)  TASK="obb" ;;
            *)
                # >9 variable fields = likely segment (polygon), or pose
                # Check if field count is consistent (segment varies, pose is fixed)
                FIELD_COUNTS=$(awk '{print NF}' "$FIRST_LABEL" | sort -u | wc -l)
                if [ "$FIELD_COUNTS" -gt 1 ]; then
                    TASK="segment"
                else
                    # Fixed field count >5: could be pose
                    # Pose: 5 + 3*num_keypoints (with visibility) or 5 + 2*num_keypoints
                    EXTRA=$((FIELD_COUNT - 5))
                    if [ $((EXTRA % 3)) -eq 0 ] && [ "$EXTRA" -gt 0 ]; then
                        TASK="pose"
                        KPT_NUM=$((EXTRA / 3))
                        KPT_SHAPE="$KPT_NUM 3"
                        echo "[AUTO] Detected pose with $KPT_NUM keypoints (3D)"
                    elif [ $((EXTRA % 2)) -eq 0 ] && [ "$EXTRA" -gt 0 ]; then
                        TASK="pose"
                        KPT_NUM=$((EXTRA / 2))
                        KPT_SHAPE="$KPT_NUM 2"
                        echo "[AUTO] Detected pose with $KPT_NUM keypoints (2D)"
                    else
                        TASK="segment"
                    fi
                fi
                ;;
        esac
    fi
    echo "[AUTO] Task type: $TASK (from label format: $FIELD_COUNT fields)"
fi

case "$TASK" in
    detect|obb|segment|pose|classify) ;;
    *) echo "ERROR: --task must be one of: detect, obb, segment, pose, classify"; exit 1 ;;
esac

if [ "$TASK" = "pose" ] && [ -z "$KPT_SHAPE" ]; then
    echo "ERROR: --kpt-shape is required for pose task (could not auto-detect)."
    echo "Format: --kpt-shape 'NUM_KEYPOINTS DIMENSIONS'"
    echo "Example: --kpt-shape '17 3'  (COCO pose: 17 keypoints, x/y/visibility)"
    exit 1
fi

# ─── Auto-detect image resolution ────────────────────────────────────
if [ -z "$IMAGE_INFO" ]; then
    echo "[AUTO] Detecting image resolution..."
    IMG_DIR="$DATASET/images"
    [ -d "$IMG_DIR/train" ] && IMG_DIR="$IMG_DIR/train"
    [ "$TASK" = "classify" ] && IMG_DIR="$DATASET/train" && [ -d "$IMG_DIR" ] || IMG_DIR="$DATASET/images"
    FIRST_IMG=$(find "$IMG_DIR" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.tif" \) 2>/dev/null | head -1)
    if [ -n "$FIRST_IMG" ]; then
        IMG_EXT=$(echo "${FIRST_IMG##*.}" | tr '[:lower:]' '[:upper:]')
        IMG_RES=$(source "$ROOT_DIR/.venv/bin/activate" 2>/dev/null; python3 -c "from PIL import Image; img=Image.open('$FIRST_IMG'); print(f'{img.size[0]}x{img.size[1]}')" 2>/dev/null)
        if [ -n "$IMG_RES" ]; then
            IMAGE_INFO="$IMG_RES $IMG_EXT"
            echo "[AUTO] Image info: $IMAGE_INFO"
        else
            IMAGE_INFO="unknown resolution"
        fi
    else
        IMAGE_INFO="unknown resolution"
    fi
fi

# ─── Check existing project ──────────────────────────────────────────
PROJECT_DIR="$ROOT_DIR/projects/$NAME"
if [ -d "$PROJECT_DIR" ]; then
    if [ "$FORCE" = "true" ]; then
        if [ -f "$PROJECT_DIR/train.pid" ]; then
            OLD_PID=$(cat "$PROJECT_DIR/train.pid")
            if kill -0 "$OLD_PID" 2>/dev/null; then
                echo "ERROR: Training is actively running (PID $OLD_PID). Cannot overwrite."
                echo "Wait for training to finish, or kill it: kill $OLD_PID"
                exit 1
            fi
        fi
        echo "[WARN] Overwriting existing project: $PROJECT_DIR"
        if [ -d "$PROJECT_DIR/logs" ]; then
            BACKUP="$PROJECT_DIR/logs_backup_$(date +%Y%m%d_%H%M%S)"
            mv "$PROJECT_DIR/logs" "$BACKUP"
            echo "[WARN] Existing logs moved to: $BACKUP"
        fi
    else
        echo "ERROR: Project '$NAME' already exists at: $PROJECT_DIR"
        echo "Options:"
        echo "  1. Choose a different --name"
        echo "  2. Delete it manually: rm -rf $PROJECT_DIR"
        echo "  3. Overwrite with: --force"
        exit 1
    fi
fi

# ─── Strict-heldout: randomize test seed when not pinned ────────────
# In strict-heldout mode the test split should be a fresh random sample
# for every scaffold unless the operator explicitly pinned it (e.g. for
# benchmark sweeps where every provider needs the same test data).
if [ "$STRICT_HELDOUT" = "true" ] && [ "$TEST_SEED_EXPLICIT" = "false" ]; then
    if command -v shuf >/dev/null 2>&1; then
        TEST_SEED=$(shuf -i 1-1000000 -n 1)
    else
        TEST_SEED=$(python3 -c 'import random; print(random.randint(1, 1000000))')
    fi
    echo "[strict-heldout] randomized test seed: $TEST_SEED"
fi

# ─── Check dataset split status ───────────────────────────────────────
IS_SPLIT=false
TEST_EXISTS=false
if [ "$TASK" = "classify" ]; then
    if [ -d "$DATASET/train" ]; then
        IS_SPLIT=true
    fi
    [ -d "$DATASET/test" ] && TEST_EXISTS=true
else
    if [ -d "$DATASET/images/train" ]; then
        IS_SPLIT=true
    fi
    [ -d "$DATASET/images/test" ] && TEST_EXISTS=true
fi

# Validate --test-split is a number in [0, 0.5]
if ! python3 -c "
v=float('$TEST_SPLIT')
assert 0 <= v <= 0.5, f'--test-split {v} out of range [0, 0.5]'
" 2>/dev/null; then
    echo "ERROR: --test-split must be a number in [0, 0.5] (got '$TEST_SPLIT')" >&2
    exit 1
fi

# Classify: not supported for now — disable test split silently.
if [ "$TASK" = "classify" ] && [ "$TEST_EXISTS" = "false" ]; then
    if [ "$TEST_SPLIT" != "0" ] && [ "$TEST_SPLIT" != "0.0" ]; then
        echo "[INFO] --test-split disabled for classify task (not yet supported)"
    fi
    TEST_SPLIT="0"
fi

if [ "$IS_SPLIT" = "false" ]; then
    if [ "$NO_AUTO_SPLIT" = "true" ]; then
        echo "ERROR: Dataset is not split into train/val."
        echo "Remove --no-auto-split to auto-split 80/20, or split manually."
        exit 1
    fi
    if [ "$TEST_SPLIT" = "0" ] || [ "$TEST_SPLIT" = "0.0" ]; then
        echo "[AUTO] Auto-splitting dataset 80/20 (no test split)..."
    else
        echo "[AUTO] Auto-splitting dataset 3-way (test=$TEST_SPLIT, then 80/20 train/val)..."
    fi
    if [ "$TASK" = "classify" ]; then
        echo "ERROR: Auto-split for classify datasets is not yet supported. Please split manually."
        exit 1
    fi
    TEST_SPLIT="$TEST_SPLIT" TEST_SEED="$TEST_SEED" DATASET="$DATASET" python3 - <<'PY'
import os, random, shutil
from pathlib import Path

dataset_dir = Path(os.environ['DATASET'])
test_ratio = float(os.environ['TEST_SPLIT'])
test_seed  = int(os.environ['TEST_SEED'])

src_images = dataset_dir / 'images'
src_labels = dataset_dir / 'labels'
if (src_images / 'train').is_dir():
    print('Already split'); raise SystemExit

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
images = sorted(f for f in src_images.iterdir() if f.suffix.lower() in exts)

# Carve test first with a seed that's locked separately from train/val,
# so the test set is stable even if train/val logic changes later.
rng_test = random.Random(test_seed)
rng_test.shuffle(images)
n_test = int(len(images) * test_ratio)
test_imgs = images[:n_test]
rest      = images[n_test:]

# Train/val split: keep the legacy seed=0 so existing projects (with
# test_ratio=0) reproduce their old split byte-for-byte.
rng_val = random.Random(0)
rng_val.shuffle(rest)
split_idx = max(1, int(len(rest) * 0.8))
train_imgs = rest[:split_idx]
val_imgs   = rest[split_idx:]

subsets = [('train', train_imgs), ('val', val_imgs)]
if test_imgs:
    subsets.append(('test', test_imgs))

for subset, img_list in subsets:
    (src_images / subset).mkdir(exist_ok=True)
    (src_labels / subset).mkdir(exist_ok=True)
    for img in img_list:
        shutil.move(str(img), str(src_images / subset / img.name))
        lbl = src_labels / (img.stem + '.txt')
        if lbl.exists():
            shutil.move(str(lbl), str(src_labels / subset / lbl.name))

parts = [f'train={len(train_imgs)}', f'val={len(val_imgs)}']
if test_imgs:
    parts.append(f'test={len(test_imgs)}  (agent-invisible)')
print('Split: ' + ', '.join(parts))
PY
    IS_SPLIT=true
    [ -d "$DATASET/images/test" ] && TEST_EXISTS=true
fi

# Pre-split dataset that's missing a test split — carve it out of train using
# the locked seed. The agent will train on fewer images but gains a held-out
# unbiased benchmark. Skip silently when --test-split 0.
if [ "$IS_SPLIT" = "true" ] && [ "$TEST_EXISTS" = "false" ] && [ "$TASK" != "classify" ] && \
   [ "$TEST_SPLIT" != "0" ] && [ "$TEST_SPLIT" != "0.0" ]; then
    echo "[AUTO] Carving $TEST_SPLIT of TRAIN into a held-out test split (seed=$TEST_SEED)..."
    TEST_SPLIT="$TEST_SPLIT" TEST_SEED="$TEST_SEED" DATASET="$DATASET" python3 - <<'PY'
import os, random, shutil
from pathlib import Path

dataset_dir = Path(os.environ['DATASET'])
test_ratio = float(os.environ['TEST_SPLIT'])
test_seed  = int(os.environ['TEST_SEED'])

src_images = dataset_dir / 'images' / 'train'
src_labels = dataset_dir / 'labels' / 'train'
dst_images = dataset_dir / 'images' / 'test'
dst_labels = dataset_dir / 'labels' / 'test'

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
images = sorted(f for f in src_images.iterdir() if f.suffix.lower() in exts)

# Same RNG strategy as the 3-way split above — test_seed shuffles once,
# first N% becomes test. Locked across runs.
rng_test = random.Random(test_seed)
rng_test.shuffle(images)
n_test = int(len(images) * test_ratio)
test_imgs = images[:n_test]

dst_images.mkdir(exist_ok=True)
dst_labels.mkdir(exist_ok=True)
for img in test_imgs:
    shutil.move(str(img), str(dst_images / img.name))
    lbl = src_labels / (img.stem + '.txt')
    if lbl.exists():
        shutil.move(str(lbl), str(dst_labels / lbl.name))

print(f'Carved test={len(test_imgs)} from train (agent-invisible).')
PY
    TEST_EXISTS=true
fi

# ─── Count images ────────────────────────────────────────────────────
TEST_COUNT=0
if [ "$TASK" = "classify" ]; then
    TRAIN_COUNT=$(find "$DATASET/train" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    VAL_COUNT=$(find "$DATASET/val" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
    [ -d "$DATASET/test" ] && TEST_COUNT=$(find "$DATASET/test" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
else
    TRAIN_COUNT=$(find "$DATASET/images/train" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.tif" \) 2>/dev/null | wc -l)
    VAL_COUNT=$(find "$DATASET/images/val" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.tif" \) 2>/dev/null | wc -l)
    [ -d "$DATASET/images/test" ] && TEST_COUNT=$(find "$DATASET/images/test" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.bmp" -o -name "*.tif" \) 2>/dev/null | wc -l)
fi
TOTAL_COUNT=$((TRAIN_COUNT + VAL_COUNT + TEST_COUNT))
if [ "$TEST_COUNT" -gt 0 ]; then
    echo "[INFO] Dataset: $TRAIN_COUNT train + $VAL_COUNT val + $TEST_COUNT test (agent-invisible) = $TOTAL_COUNT images"
    # Warn (not block) — fewer than 30 test images is too noisy for per-class
    # metrics. The session can still run; we just want the operator to know.
    if [ "$TEST_COUNT" -lt 30 ]; then
        echo "[WARN] Test split has only $TEST_COUNT images — per-class metrics will be very noisy. Consider a larger --test-split, or collect more data." >&2
    fi
else
    echo "[INFO] Dataset: $TRAIN_COUNT train + $VAL_COUNT val = $TOTAL_COUNT images"
fi

# ─── Detect classes ───────────────────────────────────────────────────
if [ -z "$CLASSES" ]; then
    if [ "$TASK" = "classify" ]; then
        CLASSES=$(ls -1 "$DATASET/train" 2>/dev/null | sort | awk '{printf "%d:%s,", NR-1, $0}' | sed 's/,$//')
    elif [ -f "$DATASET/classes.txt" ]; then
        CLASSES=$(awk '{printf "%d:%s,", NR-1, $0}' "$DATASET/classes.txt" | sed 's/,$//')
    else
        # Scan ALL label files for the max class ID. NUM_CLASSES = max_id + 1
        # (YOLO requires names to cover [0, NUM_CLASSES-1]; gaps are fine).
        # Reading just the first file misses any class that doesn't appear in
        # image #1, which on multi-class datasets undercounts wildly.
        MAX_CLASS_ID=$(find "$DATASET/labels" -name "*.txt" -type f 2>/dev/null \
            -exec awk '{print $1}' {} + 2>/dev/null \
            | sort -un | tail -1)
        if [ -n "$MAX_CLASS_ID" ]; then
            NUM_CLASSES=$((MAX_CLASS_ID + 1))
            CLASSES=$(seq 0 $((NUM_CLASSES - 1)) | awk '{printf "%d:class_%d,", $1, $1}' | sed 's/,$//')
        else
            CLASSES="0:class_0"
        fi
    fi
fi
echo "[INFO] Classes: $CLASSES"

NUM_CLASSES=$(echo "$CLASSES" | tr ',' '\n' | wc -l)

# ─── Validate model ──────────────────────────────────────────────────
MODELS_DIR="$ROOT_DIR/models/pretrained"

if [ -z "$MODEL" ]; then
    case "$TASK" in
        detect)   MODEL="yolo11n.pt" ;;
        obb)      MODEL="yolo11n-obb.pt" ;;
        segment)  MODEL="yolo11n-seg.pt" ;;
        pose)     MODEL="yolo11n-pose.pt" ;;
        classify) MODEL="yolo11n-cls.pt" ;;
    esac
fi

if [ ! -f "$MODELS_DIR/$MODEL" ]; then
    echo "ERROR: Model not found: $MODELS_DIR/$MODEL"
    echo "Run: bash scripts/download_models.sh --task $TASK"
    exit 1
fi

if [ -z "$FALLBACK" ]; then
    case "$TASK" in
        detect)   FALLBACK="yolov8n.pt" ;;
        obb)      FALLBACK="yolov8n-obb.pt" ;;
        segment)  FALLBACK="yolov8n-seg.pt" ;;
        pose)     FALLBACK="yolov8n-pose.pt" ;;
        classify) FALLBACK="yolov8n-cls.pt" ;;
    esac
fi

# ─── Set task-specific defaults ───────────────────────────────────────
VENV_PATH="$ROOT_DIR/.venv"
RUNS_DIR="$ROOT_DIR/runs"
DATASET_YAML_PATH="$ROOT_DIR/datasets/$NAME/dataset.yaml"

if [ -z "$IMGSZ" ]; then
    WIDTH=$(echo "$IMAGE_INFO" | grep -oP '^\d+' | head -1)
    if [ -n "$WIDTH" ] && [ "$WIDTH" -gt 1920 ] 2>/dev/null; then
        IMGSZ=1024
        echo "[AUTO] IMGSZ=1024 (source images ${WIDTH}px wide > 1920px)"
    else
        IMGSZ=640
        echo "[AUTO] IMGSZ=640"
    fi
fi

case "$TASK" in
    detect)
        DEGREES=0.0; FLIPUD=0.0; FLIPLR=0.5; MOSAIC=1.0; MIXUP=0.0; COPY_PASTE=0.0; ERASING=0.4; SCALE=0.5
        PRIMARY_METRIC="mAP50(B)"
        PRIMARY_METRIC_COL=8
        BEST_METRIC_AWK="awk -F',' 'NR>1 {if(\$8+0 > max) {max=\$8+0; line=\$0}} END {print max, line}' \"\$LATEST/results.csv\""
        ;;
    obb)
        DEGREES=0.0; FLIPUD=0.5; FLIPLR=0.5; MOSAIC=1.0; MIXUP=0.0; COPY_PASTE=0.0; ERASING=0.4; SCALE=0.5
        PRIMARY_METRIC="mAP50(B)"
        PRIMARY_METRIC_COL=9
        BEST_METRIC_AWK="awk -F',' 'NR>1 {if(\$9+0 > max) {max=\$9+0; line=\$0}} END {print max, line}' \"\$LATEST/results.csv\""
        ;;
    segment)
        DEGREES=0.0; FLIPUD=0.0; FLIPLR=0.5; MOSAIC=1.0; MIXUP=0.0; COPY_PASTE=0.1; ERASING=0.4; SCALE=0.5
        PRIMARY_METRIC="mAP50(M)"
        PRIMARY_METRIC_COL=13
        BEST_METRIC_AWK="awk -F',' 'NR>1 {if(\$13+0 > max) {max=\$13+0; line=\$0}} END {print max, line}' \"\$LATEST/results.csv\""
        ;;
    pose)
        DEGREES=0.0; FLIPUD=0.0; FLIPLR=0.5; MOSAIC=1.0; MIXUP=0.0; COPY_PASTE=0.0; ERASING=0.4; SCALE=0.5
        PRIMARY_METRIC="mAP50(P)"
        PRIMARY_METRIC_COL=13
        BEST_METRIC_AWK="awk -F',' 'NR>1 {if(\$13+0 > max) {max=\$13+0; line=\$0}} END {print max, line}' \"\$LATEST/results.csv\""
        ;;
    classify)
        DEGREES=0.0; FLIPUD=0.0; FLIPLR=0.5; MOSAIC=1.0; MIXUP=0.1; COPY_PASTE=0.0; ERASING=0.4; SCALE=0.5
        PRIMARY_METRIC="accuracy_top1"
        PRIMARY_METRIC_COL=4
        BEST_METRIC_AWK="awk -F',' 'NR>1 {if(\$4+0 > max) {max=\$4+0; line=\$0}} END {print max, line}' \"\$LATEST/results.csv\""
        ;;
esac

BASELINE_VALUE="${BASELINE:-none}"

# ─── Generate dataset.yaml ───────────────────────────────────────────
echo "[INFO] Creating dataset YAML..."
mkdir -p "$ROOT_DIR/datasets/$NAME"

if [ "$TASK" = "classify" ]; then
    {
        echo "path: $DATASET"
        echo "train: train"
        echo "val: val"
        [ "$TEST_COUNT" -gt 0 ] && echo "test: test"
    } > "$DATASET_YAML_PATH"
elif [ "$TASK" = "pose" ]; then
    KPT_NUM=$(echo "$KPT_SHAPE" | awk '{print $1}')
    KPT_DIM=$(echo "$KPT_SHAPE" | awk '{print $2}')
    {
        echo "path: $DATASET"
        echo "train: images/train"
        echo "val: images/val"
        [ "$TEST_COUNT" -gt 0 ] && echo "test: images/test"
        echo ""
        echo "kpt_shape: [$KPT_NUM, $KPT_DIM]"
        echo ""
        echo "names:"
        echo "$CLASSES" | tr ',' '\n' | while IFS=: read -r idx cname; do
            echo "  $idx: $cname"
        done
    } > "$DATASET_YAML_PATH"
else
    {
        echo "path: $DATASET"
        echo "train: images/train"
        echo "val: images/val"
        [ "$TEST_COUNT" -gt 0 ] && echo "test: images/test"
        echo ""
        echo "names:"
        echo "$CLASSES" | tr ',' '\n' | while IFS=: read -r idx cname; do
            echo "  $idx: $cname"
        done
    } > "$DATASET_YAML_PATH"
fi

echo "[INFO] Dataset YAML: $DATASET_YAML_PATH"

# ─── Strict-heldout dual-yaml split ──────────────────────────────────
# When strict mode is on, the agent-visible dataset.yaml MUST NOT mention
# the test split. Move the test: line into a sibling dataset.eval.yaml
# that only run_test_eval.py and run_test_tool.py read.
DATASET_EVAL_YAML_PATH="$ROOT_DIR/datasets/$NAME/dataset.eval.yaml"
if [ "$STRICT_HELDOUT" = "true" ] && [ "$TEST_COUNT" -gt 0 ]; then
    echo "[strict-heldout] splitting dataset.yaml into agent-visible + eval-only"
    TEST_LINE=$(grep -E '^test:' "$DATASET_YAML_PATH" || true)
    # Build the eval yaml: path + just test: + names (so yolo val can resolve classes)
    {
        grep -E '^path:' "$DATASET_YAML_PATH"
        echo "$TEST_LINE"
        echo ""
        # Re-emit names: block (works for both `names:` mapping and `names: [...]`
        awk '/^names:/{p=1;print;next} p&&/^[^[:space:]]/{p=0} p' "$DATASET_YAML_PATH"
    } > "$DATASET_EVAL_YAML_PATH"
    # Strip test: from the agent-visible yaml
    grep -v '^test:' "$DATASET_YAML_PATH" > "${DATASET_YAML_PATH}.tmp" \
        && mv "${DATASET_YAML_PATH}.tmp" "$DATASET_YAML_PATH"
    echo "[strict-heldout] dataset.yaml      (agent-visible): $DATASET_YAML_PATH"
    echo "[strict-heldout] dataset.eval.yaml (operator-only): $DATASET_EVAL_YAML_PATH"
fi

# ─── Generate task-specific multi-line blocks ─────────────────────────
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# --- CLASSES block (for markdown) ---
echo "$CLASSES" | tr ',' '\n' | while IFS=: read -r idx cname; do
    echo "  - $idx: $cname"
done > "$TMPDIR/classes.md"

# --- DATASET_TABLE ---
cat > "$TMPDIR/dataset_table.md" <<TABLE_EOF
| Dataset | YAML | Train | Val | Total | Notes |
|---------|------|-------|-----|-------|-------|
| **$NAME** | \`$DATASET_YAML_PATH\` | $TRAIN_COUNT | $VAL_COUNT | $TOTAL_COUNT | Primary dataset |
TABLE_EOF

# --- BASELINE_SECTION ---
if [ "$BASELINE_VALUE" = "none" ]; then
    echo "No prior baseline. This is the first training session." > "$TMPDIR/baseline_section.md"
else
    cat > "$TMPDIR/baseline_section.md" <<BASE_EOF
| Metric | Value | Notes |
|--------|-------|-------|
| $PRIMARY_METRIC | $BASELINE_VALUE | Previous best baseline — target to beat |
BASE_EOF
fi

# --- RESULTS_CSV_COLUMNS ---
case "$TASK" in
    detect)
        cat > "$TMPDIR/results_csv_columns.md" <<'COLS_EOF'
```
epoch, time,
train/box_loss, train/cls_loss, train/dfl_loss,
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
val/box_loss, val/cls_loss, val/dfl_loss,
lr/pg0, lr/pg1, lr/pg2
```

**Key columns (1-indexed):**
- `metrics/mAP50(B)` → primary quality signal (column 8)
- `metrics/mAP50-95(B)` → stricter metric (column 9)
- `metrics/precision(B)` → column 6
- `metrics/recall(B)` → column 7
- `train/box_loss` vs `val/box_loss` → overfitting gap
COLS_EOF
        ;;
    obb)
        cat > "$TMPDIR/results_csv_columns.md" <<'COLS_EOF'
```
epoch, time,
train/box_loss, train/cls_loss, train/dfl_loss, train/angle_loss,
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
val/box_loss, val/cls_loss, val/dfl_loss, val/angle_loss,
lr/pg0, lr/pg1, lr/pg2
```

**Key columns (1-indexed):**
- `metrics/mAP50(B)` → primary quality signal (column 9)
- `metrics/mAP50-95(B)` → stricter metric (column 10)
- `metrics/precision(B)` → column 7
- `metrics/recall(B)` → column 8
- `train/box_loss` vs `val/box_loss` → overfitting gap
- `train/angle_loss` vs `val/angle_loss` → OBB rotation accuracy gap
COLS_EOF
        ;;
    segment)
        cat > "$TMPDIR/results_csv_columns.md" <<'COLS_EOF'
```
epoch, time,
train/box_loss, train/cls_loss, train/dfl_loss, train/seg_loss,
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M),
val/box_loss, val/cls_loss, val/dfl_loss, val/seg_loss,
lr/pg0, lr/pg1, lr/pg2
```

**Key columns (1-indexed):**
- `metrics/mAP50(M)` → primary mask quality signal (column 13)
- `metrics/mAP50(B)` → box quality (column 9)
- `train/seg_loss` vs `val/seg_loss` → mask overfitting gap
- Monitor mAP50(M) vs mAP50(B) — they can diverge
COLS_EOF
        ;;
    pose)
        cat > "$TMPDIR/results_csv_columns.md" <<'COLS_EOF'
```
epoch, time,
train/box_loss, train/cls_loss, train/dfl_loss, train/pose_loss,
metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B),
metrics/precision(P), metrics/recall(P), metrics/mAP50(P), metrics/mAP50-95(P),
val/box_loss, val/cls_loss, val/dfl_loss, val/pose_loss,
lr/pg0, lr/pg1, lr/pg2
```

**Key columns (1-indexed):**
- `metrics/mAP50(P)` → primary pose quality signal (column 13)
- `metrics/mAP50(B)` → box quality (column 9)
- `train/pose_loss` vs `val/pose_loss` → keypoint overfitting gap
COLS_EOF
        ;;
    classify)
        cat > "$TMPDIR/results_csv_columns.md" <<'COLS_EOF'
```
epoch, time,
train/loss,
metrics/accuracy_top1, metrics/accuracy_top5,
val/loss,
lr/pg0, lr/pg1, lr/pg2
```

**Key columns (1-indexed):**
- `metrics/accuracy_top1` → primary quality signal (column 4)
- `metrics/accuracy_top5` → top-5 accuracy (column 5)
- `train/loss` vs `val/loss` → overfitting gap
COLS_EOF
        ;;
esac

# --- TASK_CONSIDERATIONS ---
case "$TASK" in
    detect)
        cat > "$TMPDIR/task_considerations.md" <<'TC_EOF'
1. **box/cls/dfl loss balance** — monitor all three. If cls_loss converges fast but box_loss doesn't, localization is the bottleneck.
2. **IMGSZ impact** — if original images are high-res, IMGSZ=1024 can dramatically improve detection of small objects.
3. **Augmentation basics** — MOSAIC and SCALE are the most impactful. FLIPLR=0.5 is standard. Try DEGREES only if objects appear at varied rotations.
4. **Single class vs multi-class** — for single-class, cls_loss should converge quickly. Focus on box_loss.
TC_EOF
        ;;
    obb)
        cat > "$TMPDIR/task_considerations.md" <<'TC_EOF'
1. **angle_loss** is unique to OBB — track it alongside box_loss. If angle_loss plateaus or rises while box_loss falls, the model is losing rotation accuracy.
2. **DEGREES augmentation** matters for OBB — rotation augmentation can help the model learn better orientation. Try 10-30 deg if not overfitting.
3. **High-res images** — if original images are large, IMGSZ=1024 helps significantly. OBB needs precise corners.
4. **Single class** means cls_loss should converge quickly — focus attention on box_loss and angle_loss.
5. **Background images** (no labels) in the training set teach the model to not hallucinate detections.
TC_EOF
        ;;
    segment)
        cat > "$TMPDIR/task_considerations.md" <<'TC_EOF'
1. **seg_loss** tracks mask quality separately from box quality. Monitor both.
2. **COPY_PASTE augmentation** is especially powerful for segmentation — it creates new training examples by pasting object masks onto different backgrounds.
3. **Monitor mAP50(M) vs mAP50(B)** — they can diverge. If box mAP is high but mask mAP is low, the model detects objects but segments them poorly.
4. **If seg_loss > 5.0**, mask learning has collapsed — consider reducing augmentation or checking label quality.
TC_EOF
        ;;
    pose)
        cat > "$TMPDIR/task_considerations.md" <<'TC_EOF'
1. **pose_loss** tracks keypoint accuracy. It should decrease steadily.
2. **Be careful with aggressive geometric augmentation** — strong rotation (DEGREES) or perspective transforms can distort keypoint positions in labels.
3. **Monitor mAP50(P)** for pose quality alongside mAP50(B) for detection quality.
4. **kpt_shape must match the dataset** — verify keypoint count and dimensions are correct.
5. **If pose_loss > 10.0**, keypoint learning has collapsed — check augmentation settings and label quality.
TC_EOF
        ;;
    classify)
        cat > "$TMPDIR/task_considerations.md" <<'TC_EOF'
1. **Use accuracy_top1 as primary metric** (NOT mAP50). accuracy_top5 is useful for multi-class problems.
2. **No bounding box concepts** — simpler loss landscape than detection tasks.
3. **Watch for class imbalance** — check confusion matrix if some classes have much fewer images.
4. **Augmentation focus**: HSV, erasing, mixup are most useful. Geometric transforms (mosaic, scale) are less relevant for classification.
5. **train/loss vs val/loss gap** is the key overfitting indicator.
TC_EOF
        ;;
esac

# --- STOP_CONDITIONS ---
case "$TASK" in
    detect)
        cat > "$TMPDIR/stop_conditions.md" <<'SC_EOF'
- NaN in any loss column
- val_box_loss increasing for 3+ consecutive runs (severe overfitting)
- 6 consecutive runs with metric change < 0.01
- GPU OOM persisting after reducing batch size
- Dataset YAML missing or broken paths
- Training job already running (PID alive)
SC_EOF
        ;;
    obb)
        cat > "$TMPDIR/stop_conditions.md" <<'SC_EOF'
- NaN in any loss column
- angle_loss > 0.1 (rotation learning has collapsed)
- val_box_loss increasing for 3+ consecutive runs (severe overfitting)
- 6 consecutive runs with metric change < 0.01
- GPU OOM persisting after reducing batch size
- Dataset YAML missing or broken paths
- Training job already running (PID alive)
SC_EOF
        ;;
    segment)
        cat > "$TMPDIR/stop_conditions.md" <<'SC_EOF'
- NaN in any loss column
- seg_loss > 5.0 (mask learning has collapsed)
- val_box_loss increasing for 3+ consecutive runs (severe overfitting)
- 6 consecutive runs with metric change < 0.01
- GPU OOM persisting after reducing batch size
- Dataset YAML missing or broken paths
- Training job already running (PID alive)
SC_EOF
        ;;
    pose)
        cat > "$TMPDIR/stop_conditions.md" <<'SC_EOF'
- NaN in any loss column
- pose_loss > 10.0 (keypoint learning has collapsed)
- val_box_loss increasing for 3+ consecutive runs (severe overfitting)
- 6 consecutive runs with metric change < 0.01
- GPU OOM persisting after reducing batch size
- Dataset YAML missing or broken paths
- Training job already running (PID alive)
SC_EOF
        ;;
    classify)
        cat > "$TMPDIR/stop_conditions.md" <<'SC_EOF'
- NaN in loss
- val/loss increasing for 3+ consecutive runs (severe overfitting)
- 6 consecutive runs with accuracy change < 0.01
- GPU OOM persisting after reducing batch size
- Dataset path missing or broken
- Training job already running (PID alive)
SC_EOF
        ;;
esac

# --- NEXT_INSTRUCTION_METRICS (for next_instruction.md template) ---
case "$TASK" in
    detect)
        echo '  mAP50: X.XX | mAP50-95: X.XX | P: X.XX | R: X.XX
  val_box: X.XX | train_box: X.XX' > "$TMPDIR/next_instruction_metrics.md"
        ;;
    obb)
        echo '  mAP50: X.XX | mAP50-95: X.XX | P: X.XX | R: X.XX
  val_box: X.XX | train_box: X.XX | val_angle: X.XX | train_angle: X.XX' > "$TMPDIR/next_instruction_metrics.md"
        ;;
    segment)
        echo '  mAP50(B): X.XX | mAP50(M): X.XX | P: X.XX | R: X.XX
  val_box: X.XX | train_box: X.XX | val_seg: X.XX | train_seg: X.XX' > "$TMPDIR/next_instruction_metrics.md"
        ;;
    pose)
        echo '  mAP50(B): X.XX | mAP50(P): X.XX | P: X.XX | R: X.XX
  val_box: X.XX | train_box: X.XX | val_pose: X.XX | train_pose: X.XX' > "$TMPDIR/next_instruction_metrics.md"
        ;;
    classify)
        echo '  accuracy_top1: X.XX | accuracy_top5: X.XX
  train_loss: X.XX | val_loss: X.XX' > "$TMPDIR/next_instruction_metrics.md"
        ;;
esac

# --- READ_METRICS_BLOCK (for hyperparameter strategy) ---
case "$TASK" in
    detect)
        cat > "$TMPDIR/read_metrics_block.md" <<'RM_EOF'
Extract from the **best epoch** (highest mAP50):
```bash
awk -F',' 'NR>1 {if($8+0 > max) {max=$8+0; ep=$1; line=$0}} END {print "best_epoch="ep, "best_mAP50="max}' "$LATEST/results.csv"
```

Key metrics:
- **mAP50** = `metrics/mAP50(B)` — primary quality signal (column 8)
- **mAP50-95** = `metrics/mAP50-95(B)` — stricter metric (column 9)
- **Precision** = `metrics/precision(B)` (column 6)
- **Recall** = `metrics/recall(B)` (column 7)
- **train_box_loss** = `train/box_loss` (column 3)
- **val_box_loss** = `val/box_loss` (column 10)

Compute:
- **Overfitting gap** = `val_box_loss - train_box_loss`
- **Improvement** = current best mAP50 - previous run best mAP50
- **Early stop?** = actual epochs trained < EPOCHS setting
RM_EOF
        ;;
    obb)
        cat > "$TMPDIR/read_metrics_block.md" <<'RM_EOF'
Extract from the **best epoch** (highest mAP50):
```bash
awk -F',' 'NR>1 {if($9+0 > max) {max=$9+0; ep=$1; line=$0}} END {print "best_epoch="ep, "best_mAP50="max}' "$LATEST/results.csv"
```

Key metrics:
- **mAP50** = `metrics/mAP50(B)` — primary quality signal (column 9)
- **mAP50-95** = `metrics/mAP50-95(B)` — stricter metric (column 10)
- **Precision** = `metrics/precision(B)` (column 7)
- **Recall** = `metrics/recall(B)` (column 8)
- **train_box_loss** = `train/box_loss` (column 3)
- **val_box_loss** = `val/box_loss` (column 11)
- **train_angle_loss** = `train/angle_loss` (column 6) — **OBB-specific**
- **val_angle_loss** = `val/angle_loss` (column 14) — **OBB-specific**

Compute:
- **Overfitting gap** = `val_box_loss - train_box_loss`
- **Angle health** = val_angle_loss trend — should be decreasing or stable
- **Improvement** = current best mAP50 - previous run best mAP50
- **Early stop?** = actual epochs trained < EPOCHS setting
RM_EOF
        ;;
    segment)
        cat > "$TMPDIR/read_metrics_block.md" <<'RM_EOF'
Extract from the **best epoch** (highest mAP50(M)):
```bash
awk -F',' 'NR>1 {if($13+0 > max) {max=$13+0; ep=$1; line=$0}} END {print "best_epoch="ep, "best_mAP50_M="max}' "$LATEST/results.csv"
```

Key metrics:
- **mAP50(M)** = `metrics/mAP50(M)` — primary mask quality (column 13)
- **mAP50(B)** = `metrics/mAP50(B)` — box quality (column 9)
- **train_seg_loss** = `train/seg_loss` (column 6)
- **val_seg_loss** = `val/seg_loss` — mask overfitting

Compute:
- **Overfitting gap** = `val_box_loss - train_box_loss` (and seg_loss gap)
- **Box vs mask divergence** = mAP50(B) - mAP50(M)
- **Improvement** = current best mAP50(M) - previous run best
- **Early stop?** = actual epochs trained < EPOCHS setting
RM_EOF
        ;;
    pose)
        cat > "$TMPDIR/read_metrics_block.md" <<'RM_EOF'
Extract from the **best epoch** (highest mAP50(P)):
```bash
awk -F',' 'NR>1 {if($13+0 > max) {max=$13+0; ep=$1; line=$0}} END {print "best_epoch="ep, "best_mAP50_P="max}' "$LATEST/results.csv"
```

Key metrics:
- **mAP50(P)** = `metrics/mAP50(P)` — primary pose quality (column 13)
- **mAP50(B)** = `metrics/mAP50(B)` — box quality (column 9)
- **train_pose_loss** = `train/pose_loss` (column 6)
- **val_pose_loss** — pose overfitting

Compute:
- **Overfitting gap** = `val_box_loss - train_box_loss` (and pose_loss gap)
- **Improvement** = current best mAP50(P) - previous run best
- **Early stop?** = actual epochs trained < EPOCHS setting
RM_EOF
        ;;
    classify)
        cat > "$TMPDIR/read_metrics_block.md" <<'RM_EOF'
Extract from the **best epoch** (highest accuracy_top1):
```bash
awk -F',' 'NR>1 {if($4+0 > max) {max=$4+0; ep=$1; line=$0}} END {print "best_epoch="ep, "best_acc="max}' "$LATEST/results.csv"
```

Key metrics:
- **accuracy_top1** = `metrics/accuracy_top1` — primary quality signal (column 4)
- **accuracy_top5** = `metrics/accuracy_top5` (column 5)
- **train_loss** = `train/loss` (column 3)
- **val_loss** = `val/loss` (column 6)

Compute:
- **Overfitting gap** = `val_loss - train_loss`
- **Improvement** = current best accuracy - previous run best
- **Early stop?** = actual epochs trained < EPOCHS setting
RM_EOF
        ;;
esac

# --- DIAGNOSE_TABLE ---
case "$TASK" in
    classify)
        cat > "$TMPDIR/diagnose_table.md" <<'DT_EOF'
| Condition | Diagnosis |
|-----------|-----------|
| accuracy < 0.50 | Model is struggling — needs more epochs, LR adjustment, or augmentation |
| accuracy 0.50–0.70 | Moderate — try LR schedule, augmentation, or model swap |
| accuracy 0.70–0.85 | Good — fine-tune with lower LR, more epochs |
| accuracy 0.85–0.95 | Strong — small LR, careful not to overfit |
| accuracy > 0.95 | Excellent — near ceiling. Micro-adjustments only |
| val_loss >> train_loss (gap > 1.0) | Overfitting — increase augmentation, weight_decay |
| val_loss ≈ train_loss | Good fit — can try more capacity (epochs, model size) |
| Patience triggered (epochs < EPOCHS) | Converged early — increase PATIENCE or try LR warmup |
| NaN in loss | **STOP** — do not launch next run |
DT_EOF
        ;;
    *)
        cat > "$TMPDIR/diagnose_table.md" <<'DT_EOF'
| Condition | Diagnosis |
|-----------|-----------|
| mAP50 < 0.50 | Model is struggling — needs more epochs, LR adjustment, or augmentation tuning |
| mAP50 0.50–0.70 | Moderate — room for improvement, try LR schedule, augmentation, or model swap |
| mAP50 0.70–0.80 | Good — close to typical. Fine-tune with lower LR |
| mAP50 0.80–0.85 | Strong — beating typical. Small LR, more epochs, careful not to overfit |
| mAP50 > 0.85 | Excellent — near ceiling for dataset. Micro-adjustments only |
| val_box_loss >> train_box_loss (gap > 1.0) | Overfitting — increase augmentation, weight_decay, reduce epochs |
| val_box_loss ≈ train_box_loss | Good fit — can try more capacity (epochs, model size, imgsz) |
| Precision high, Recall low | Missing objects — try more augmentation, lower LR, or more epochs |
| Precision low, Recall high | False positives — possible label noise. Consider switching dataset |
| Patience triggered (epochs < EPOCHS) | Converged early — increase PATIENCE, or try LR warmup restart |
| NaN in any loss | **STOP** — do not launch next run |
DT_EOF
        ;;
esac

# --- AUGMENTATION_GUIDANCE ---
case "$TASK" in
    obb)
        cat > "$TMPDIR/augmentation_guidance.md" <<'AG_EOF'
Set augmentation knobs directly in `next_params.json`. Required keys must
still be present; the snippets below show only the augmentation deltas
(merge with your base required-keys block):

```json
// Start: enable rotation augmentation (OBB-specific advantage)
{ "DEGREES": 15.0, "FLIPLR": 0.5, "FLIPUD": 0.5, "CLOSE_MOSAIC": 15 }
```

```json
// Stronger rotation for more variety
{ "DEGREES": 30.0, "FLIPLR": 0.5, "FLIPUD": 0.5 }
```

**OBB augmentation notes:**
- `DEGREES` is especially powerful for OBB — it teaches the model to handle arbitrary rotations
- Start with 15 deg, increase to 30 if not overfitting
- Don't exceed 45 deg without verifying it makes sense for the objects
AG_EOF
        ;;
    segment)
        cat > "$TMPDIR/augmentation_guidance.md" <<'AG_EOF'
Set augmentation knobs in `next_params.json` (merge into your base block):

```json
{ "COPY_PASTE": 0.2, "MIXUP": 0.1, "HSV_H": 0.02, "HSV_S": 0.5 }
```

**Segment augmentation notes:**
- `COPY_PASTE` is the most powerful augmentation for segmentation tasks
- Creates new training examples by pasting object masks onto different backgrounds
AG_EOF
        ;;
    pose)
        cat > "$TMPDIR/augmentation_guidance.md" <<'AG_EOF'
Set augmentation knobs in `next_params.json` (merge into your base block):

```json
// Pose-safe augmentation — DEGREES capped at 15 by the validator
{ "DEGREES": 10.0, "HSV_V": 0.3, "SCALE": 0.5 }
```

**Pose augmentation notes:**
- Be cautious with `DEGREES` — large rotation can distort keypoint labels
- The validator caps `DEGREES` at 15 for pose tasks (anything higher is rejected)
- `SCALE` and `TRANSLATE` are safe augmentations for pose
AG_EOF
        ;;
    classify)
        cat > "$TMPDIR/augmentation_guidance.md" <<'AG_EOF'
Set augmentation knobs in `next_params.json` (merge into your base block):

```json
{ "HSV_H": 0.02, "HSV_S": 0.5, "HSV_V": 0.3, "ERASING": 0.5, "MIXUP": 0.2 }
```

**Classify augmentation notes:**
- Focus on HSV, erasing, and mixup — geometric transforms are less relevant
- `MOSAIC` and `COPY_PASTE` are force-zeroed by the validator for classify
AG_EOF
        ;;
    *)
        cat > "$TMPDIR/augmentation_guidance.md" <<'AG_EOF'
Set augmentation knobs in `next_params.json` (merge into your base block):

```json
// General-purpose augmentation
{
  "DEGREES": 15.0,
  "FLIPLR": 0.5, "FLIPUD": 0.5,
  "MOSAIC": 1.0, "MIXUP": 0.1,
  "CLOSE_MOSAIC": 15
}
```
AG_EOF
        ;;
esac

# --- DATASET_SWITCH_GUIDANCE ---
echo "Currently only one dataset configured. To add more datasets, create additional YAML files in \`$ROOT_DIR/datasets/$NAME/\` and update this section." > "$TMPDIR/dataset_switch_guidance.md"

# --- DECISION_TREE ---
cat > "$TMPDIR/decision_tree.md" <<DT2_EOF
\`\`\`
START: Read \`## Verified facts\` and \`## Run history\` from your prompt.
  │
  ├─ Is this the first run (cold start)?
  │   YES → Use defaults in train.sh (pretrained model, lr=0.01, epochs=100)
  │
  ├─ Did training fail (no results.csv, NaN loss)?
  │   YES → Check current.log for OOM → Action E (batch=-1)
  │         Check for NaN → STOP
  │
  ├─ $PRIMARY_METRIC < 0.50?
  │   YES → Run count < 3?
  │         YES → Action A (try lr=0.02) + Action B (epochs=200)
  │         NO  → Action C (swap model architecture)
  │
  ├─ Overfitting (val_loss - train_loss > 1.0)?
  │   YES → Action D (fight overfitting)
  │
  ├─ $PRIMARY_METRIC improved by > 0.02 from last run?
  │   YES → Keep same strategy. Use best.pt (Action F).
  │         Maybe increase epochs (Action B, within 50-500 bound).
  │
  ├─ $PRIMARY_METRIC plateaued (< 0.01 improvement for 2+ runs)?
  │   YES → Which lever hasn't been tried?
  │         ├─ LR not reduced → Action A (halve LR)
  │         ├─ IMGSZ still 640 → Action G (try 1024)
  │         ├─ Model not swapped → Action C (try different arch)
  │         ├─ Optimizer not changed → Action J (try AdamW)
  │         └─ All tried → STOP. Plateau across all levers usually means
  │                       data is the limit. Collect more labels / audit
  │                       label quality / add holdout test set.
  │
  └─ Otherwise → Fine-tune: Action A (lower LR) + Action F (best.pt) + Action B (more epochs)
\`\`\`
DT2_EOF

# --- ACTION_M_BLOCK (strict-heldout LeetCode submit instructions) ---
# Empty file when strict mode is off → placeholder is replaced with nothing.
# Otherwise contains the Action M section describing the run_test_tool.py
# submit + rate limit + DO-NOT-leak rules.
if [ "$STRICT_HELDOUT" = "true" ]; then
    cat > "$TMPDIR/action_m_block.md" <<'AMEOF'
### Action M: Submit to held-out test (LeetCode-style — one shot per round)

**When**: val mAP has plateaued AND you want to know how the current best.pt scores on truly unseen data.

**The rule**: this project is `--strict-heldout` — the test split is invisible to you. You CANNOT `cat` test labels, CANNOT `yolo val split=test`, and the Bash guard rejects anything that touches `datasets/<name>/(images|labels)/test/`. The sanctioned (and only) way to learn the test score is to invoke:

```bash
python3 {{ROOT_DIR}}/scripts/run_test_tool.py --project {{PROJECT_DIR}}
```

The tool returns ONE line: `mAP50=X.XXXX mAP50-95=X.XXXX images=N`. No per-class breakdown, no class names, no file paths — same shape a LeetCode submit gives you.

**Rate limit**: ONE call per round. If you call it twice in the same round, the second call fails with "already queried this round". Use it once at the END of a round when you genuinely want to know the test number, not casually.

**Important**: the score never re-enters subsequent prompts (it's firewalled). If you write the number into `next_instruction.md` it gets compacted away by build_prompt.py. You have one peek per round and the information dies with the round.

AMEOF
else
    : > "$TMPDIR/action_m_block.md"
fi

# ─── Template filling ────────────────────────────────────────────────
echo "[INFO] Filling templates..."
mkdir -p "$PROJECT_DIR"
# .claude/ is only needed in claude-CLI mode (for the PreToolUse hook).
# Agent and baseline modes have no use for it (agent runs guard as a subprocess
# from run_agent.py; baseline does not run an LLM at all).
if [ "$LOOP_MODE" = "claude" ]; then
    mkdir -p "$PROJECT_DIR/.claude"
fi

# Common templates scaffolded in all modes
COMMON_TMPLS=("train.sh" "yolo_folder_skill.md" "hyperparameter_strategy.md" ".gitignore")

# Mode-specific orchestrator templates:
#   claude   → start_claude.sh + .claude/settings.json (PreToolUse hook)
#   agent    → start_agent.sh + agent.env (run_agent.py picks the LLM)
#   baseline → start_baseline.sh (LLM-free; baseline_policy.py picks params)
# All modes share train.sh; the recursive call at the end of train.sh probes
# baseline → agent → claude (most specific first).
case "$LOOP_MODE" in
    claude)   MODE_TMPLS=("start_claude.sh") ;;
    agent)    MODE_TMPLS=("start_agent.sh" "agent.env") ;;
    baseline) MODE_TMPLS=("start_baseline.sh") ;;
    *)        echo "ERROR: --mode must be 'claude' | 'agent' | 'baseline' (got '$LOOP_MODE')" >&2; exit 1 ;;
esac

for tmpl in "${COMMON_TMPLS[@]}" "${MODE_TMPLS[@]}"; do
    TMPL_FILE="$TEMPLATES_DIR/${tmpl}.tmpl"
    OUT_FILE="$PROJECT_DIR/$tmpl"
    if [ ! -f "$TMPL_FILE" ]; then
        echo "WARN: Template not found: $TMPL_FILE"
        continue
    fi
    cp "$TMPL_FILE" "$OUT_FILE"
done

# .claude/settings.json (Claude-CLI hook) is only meaningful for `claude`
# mode. In agent mode, the bash guard runs as a subprocess from
# run_agent.py — no .claude/ dir needed.
if [ "$LOOP_MODE" = "claude" ]; then
    SETTINGS_TMPL="$TEMPLATES_DIR/settings.json.tmpl"
    if [ ! -f "$SETTINGS_TMPL" ]; then
        echo "ERROR: $SETTINGS_TMPL is missing — guard hook will not be installed." >&2
        exit 1
    fi
    cp "$SETTINGS_TMPL" "$PROJECT_DIR/.claude/settings.json"
fi

# First pass: multi-line blocks via sed r/d
for f in "$PROJECT_DIR/yolo_folder_skill.md" "$PROJECT_DIR/hyperparameter_strategy.md"; do
    [ -f "$f" ] || continue

    for block_name in CLASSES DATASET_TABLE BASELINE_SECTION RESULTS_CSV_COLUMNS TASK_CONSIDERATIONS STOP_CONDITIONS NEXT_INSTRUCTION_METRICS BEST_METRIC_AWK READ_METRICS_BLOCK DIAGNOSE_TABLE AUGMENTATION_GUIDANCE DATASET_SWITCH_GUIDANCE DECISION_TREE ACTION_M_BLOCK; do
        block_key=$(echo "$block_name" | tr '[:upper:]' '[:lower:]')
        block_file="$TMPDIR/${block_key}.md"
        if [ -f "$block_file" ]; then
            sed -i -e "/{{${block_name}}}/{r ${block_file}" -e "d}" "$f"
        fi
    done
done

# Also handle BEST_METRIC_AWK in folder skill (it's a single-line but contains special chars)
echo "$BEST_METRIC_AWK" > "$TMPDIR/best_metric_awk.md"
sed -i -e "/{{BEST_METRIC_AWK}}/{r ${TMPDIR}/best_metric_awk.md" -e "d}" "$PROJECT_DIR/yolo_folder_skill.md"

# Second pass: single-line placeholders (use | delimiter for paths)
for f in "$PROJECT_DIR"/*; do
    [ -f "$f" ] || continue
    sed -i "s|{{PROJECT_NAME}}|${NAME}|g" "$f"
    sed -i "s|{{TASK}}|${TASK}|g" "$f"
    sed -i "s|{{VENV_PATH}}|${VENV_PATH}|g" "$f"
    sed -i "s|{{MODELS_DIR}}|${MODELS_DIR}|g" "$f"
    sed -i "s|{{RUNS_DIR}}|${RUNS_DIR}|g" "$f"
    sed -i "s|{{PROJECT_DIR}}|${PROJECT_DIR}|g" "$f"
    sed -i "s|{{DATASET_YAML}}|${DATASET_YAML_PATH}|g" "$f"
    sed -i "s|{{PRIMARY_MODEL}}|${MODEL}|g" "$f"
    sed -i "s|{{FALLBACK_MODEL}}|${FALLBACK}|g" "$f"
    sed -i "s|{{PRIMARY_METRIC}}|${PRIMARY_METRIC}|g" "$f"
    sed -i "s|{{IMGSZ}}|${IMGSZ}|g" "$f"
    sed -i "s|{{DEVICE}}|${DEVICE}|g" "$f"
    sed -i "s|{{MAX_ROUNDS}}|${MAX_ROUNDS}|g" "$f"
    sed -i "s|{{BASELINE_VALUE}}|${BASELINE_VALUE}|g" "$f"
    sed -i "s|{{IMAGE_INFO}}|${IMAGE_INFO}|g" "$f"
    sed -i "s|{{CLASSES}}|${CLASSES}|g" "$f"
    sed -i "s|{{NUM_CLASSES}}|${NUM_CLASSES}|g" "$f"
    sed -i "s|{{DEGREES}}|${DEGREES}|g" "$f"
    sed -i "s|{{FLIPUD}}|${FLIPUD}|g" "$f"
    sed -i "s|{{FLIPLR}}|${FLIPLR}|g" "$f"
    sed -i "s|{{MOSAIC}}|${MOSAIC}|g" "$f"
    sed -i "s|{{MIXUP}}|${MIXUP}|g" "$f"
    sed -i "s|{{COPY_PASTE}}|${COPY_PASTE}|g" "$f"
    sed -i "s|{{ERASING}}|${ERASING}|g" "$f"
    sed -i "s|{{SCALE}}|${SCALE}|g" "$f"
    sed -i "s|{{ROOT_DIR}}|${ROOT_DIR}|g" "$f"
    # P6 multi-LLM agent fields
    sed -i "s|{{LLM_PROVIDER}}|${LLM_PROVIDER}|g" "$f"
    sed -i "s|{{LLM_MODEL}}|${LLM_MODEL}|g" "$f"
    # Baseline-mode seed (only meaningful in start_baseline.sh; the substitution
    # is a no-op for other templates).
    sed -i "s|{{BASELINE_SEED}}|${BASELINE_SEED}|g" "$f"
done

# settings.json is in .claude/ (claude mode only) so the per-file loop misses it.
if [ "$LOOP_MODE" = "claude" ]; then
    sed -i "s|{{ROOT_DIR}}|${ROOT_DIR}|g" "$PROJECT_DIR/.claude/settings.json"
fi

# Append optional LLM_API_BASE when scaffolding agent mode with custom endpoint
if [ "$LOOP_MODE" = "agent" ] && [ -n "$LLM_API_BASE" ]; then
    echo "LLM_API_BASE=\"$LLM_API_BASE\"" >> "$PROJECT_DIR/agent.env"
fi

# Make shell scripts executable — only the one(s) for the chosen mode
chmod +x "$PROJECT_DIR/train.sh"
[ -f "$PROJECT_DIR/start_claude.sh" ]   && chmod +x "$PROJECT_DIR/start_claude.sh"
[ -f "$PROJECT_DIR/start_agent.sh" ]    && chmod +x "$PROJECT_DIR/start_agent.sh"
[ -f "$PROJECT_DIR/start_baseline.sh" ] && chmod +x "$PROJECT_DIR/start_baseline.sh"

# ─── Strict-heldout project markers ──────────────────────────────────
# .heldout_strict — sentinel file the claude_bash_guard walks up looking
#                   for to decide whether the LeetCode-mode patterns fire
# .heldout_seed   — recorded seed for the training_report.md to echo
# Plus emit a `heldout-cut` event for the operator audit trail.
if [ "$STRICT_HELDOUT" = "true" ]; then
    touch "$PROJECT_DIR/.heldout_strict"
    echo "$TEST_SEED" > "$PROJECT_DIR/.heldout_seed"

    # Count test images (best-effort)
    N_TEST=0
    if [ -d "$DATASET/images/test" ]; then
        N_TEST=$(find "$DATASET/images/test" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' \) 2>/dev/null | wc -l)
    elif [ -d "$DATASET/test" ]; then
        N_TEST=$(find "$DATASET/test" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' \) 2>/dev/null | wc -l)
    fi

    python3 "$ROOT_DIR/scripts/event.py" "$PROJECT_DIR" emit heldout-cut \
        --seed "$TEST_SEED" \
        --n-test-images "$N_TEST" \
        --dataset "$NAME" 2>/dev/null || true

    # Inject permissions block into .claude/settings.json (Claude mode only).
    # The deny patterns block direct test-set reads; the allow whitelists
    # the sanctioned LeetCode submit tool.
    SETTINGS_FILE="$PROJECT_DIR/.claude/settings.json"
    if [ -f "$SETTINGS_FILE" ]; then
        # Pass paths via env to avoid quote-escaping hell inside the heredoc.
        # Heredoc is QUOTED ('PY') so bash does no substitution; Python reads
        # via os.environ and builds the patterns itself.
        SETTINGS_FILE="$SETTINGS_FILE" \
        STRICT_ROOT_DIR="$ROOT_DIR" \
        STRICT_NAME="$NAME" \
        python3 - <<'PY'
import json
import os
from pathlib import Path

settings = Path(os.environ["SETTINGS_FILE"])
root = os.environ["STRICT_ROOT_DIR"]
name = os.environ["STRICT_NAME"]
data = json.loads(settings.read_text())
data["permissions"] = {
    "deny": [
        f"Read({root}/datasets/{name}/images/test/**)",
        f"Read({root}/datasets/{name}/labels/test/**)",
        f"Bash(*datasets/{name}/images/test/*)",
        f"Bash(*datasets/{name}/labels/test/*)",
        "Bash(*split=test*)",
        "Bash(*split='test'*)",
        'Bash(*split="test"*)',
    ],
    "allow": [
        "Bash(python3 *scripts/run_test_tool.py*)",
        "Bash(python *scripts/run_test_tool.py*)",
    ],
}
settings.write_text(json.dumps(data, indent=2))
PY
        echo "[strict-heldout] injected permissions block into $SETTINGS_FILE"
    fi
    echo "[strict-heldout] marker:   $PROJECT_DIR/.heldout_strict"
    echo "[strict-heldout] seed:     $TEST_SEED (recorded in $PROJECT_DIR/.heldout_seed)"
fi

# ─── Verify no unreplaced placeholders ────────────────────────────────
LEFTOVER=$(grep -rn '{{[A-Z_]*}}' "$PROJECT_DIR/" 2>/dev/null | grep -v '\.log' | grep -v 'next_instruction' || true)
if [ -n "$LEFTOVER" ]; then
    echo ""
    echo "ERROR: Unreplaced placeholders found:"
    echo "$LEFTOVER"
    exit 1
fi

# ─── Post-scaffolding output ─────────────────────────────────────────
echo ""
echo "========================================"
echo "Project '$NAME' scaffolded successfully!"
echo "========================================"
echo ""
echo "Files:"
ls -la "$PROJECT_DIR/"
echo ""
echo "Dataset:"
cat "$DATASET_YAML_PATH"
echo ""
echo "Model: $MODELS_DIR/$MODEL"
echo ""

echo "Syntax check..."
bash -n "$PROJECT_DIR/train.sh" && echo "  train.sh: OK" || echo "  train.sh: SYNTAX ERROR"
case "$LOOP_MODE" in
    agent)
        bash -n "$PROJECT_DIR/start_agent.sh" && echo "  start_agent.sh: OK" || echo "  start_agent.sh: SYNTAX ERROR"
        LAUNCH_SCRIPT="start_agent.sh"
        ;;
    baseline)
        bash -n "$PROJECT_DIR/start_baseline.sh" && echo "  start_baseline.sh: OK" || echo "  start_baseline.sh: SYNTAX ERROR"
        LAUNCH_SCRIPT="start_baseline.sh"
        ;;
    *)
        bash -n "$PROJECT_DIR/start_claude.sh" && echo "  start_claude.sh: OK" || echo "  start_claude.sh: SYNTAX ERROR"
        LAUNCH_SCRIPT="start_claude.sh"
        ;;
esac

echo ""
echo "To start training:"
echo "  cd $PROJECT_DIR && bash $LAUNCH_SCRIPT"
case "$LOOP_MODE" in
    agent)
        echo ""
        echo "Agent config: $PROJECT_DIR/agent.env (provider=$LLM_PROVIDER, model=$LLM_MODEL)"
        echo "Make sure the appropriate API key env var is set (see comments in agent.env)."
        ;;
    baseline)
        echo ""
        echo "Baseline config: random-search policy, seed=$BASELINE_SEED (no LLM)."
        echo "Same --baseline-seed reproduces the same trajectory exactly."
        ;;
esac
echo ""
echo "To monitor:"
echo "  tail -f $PROJECT_DIR/current.log"
echo "  PID=\$(cat $PROJECT_DIR/train.pid); kill -0 \$PID && echo running || echo done"
echo ""
echo "To stop:"
echo "  kill \$(cat $PROJECT_DIR/train.pid)"
