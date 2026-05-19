#!/bin/bash
# Preflight checks before launching the autonomous loop (Harness §十一).
#
# Bootstrap order discipline:
#   synchronous + necessary  : venv + ultralytics + dataset YAML — exit 1 if missing
#   synchronous + warning    : GPU memory / disk space / nvidia-smi — print loud warn
#   background + incremental : (none here)
#
# Modes:
#   --cold DATASET_PATH   Full check before scaffolding a new project.
#   --quick PROJECT_DIR   Lightweight per-round check (disk + GPU only).
#
# Exit codes:
#   0 — all checks passed (warnings may have printed)
#   1 — a blocker check failed; loop must not run
#
# Why warnings vs blockers (Harness §5.4):
#   "User thinks they have protection but doesn't" is the worst safety state.
#   We surface every missing capability LOUD so the operator sees it before
#   the loop runs unattended for hours.

set -u

MODE=""
TARGET=""
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WARN_COUNT=0
ERR_COUNT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --cold)  MODE="cold";  TARGET="$2"; shift 2 ;;
        --quick) MODE="quick"; TARGET="$2"; shift 2 ;;
        *) echo "preflight: unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Usage: bash scripts/preflight.sh --cold DATASET_PATH | --quick PROJECT_DIR" >&2
    exit 1
fi

err()  { echo "[preflight] ✗ $1" >&2; ERR_COUNT=$((ERR_COUNT + 1)); }
warn() { echo "[preflight] ⚠ $1" >&2; WARN_COUNT=$((WARN_COUNT + 1)); }
ok()   { echo "[preflight] ✓ $1"; }

# ─── Synchronous + necessary checks ──────────────────────────────────

check_venv() {
    local venv="$ROOT_DIR/.venv"
    if [ ! -f "$venv/bin/activate" ]; then
        err "venv missing at $venv — run scripts/setup_env.sh"
        return
    fi
    ok "venv present"
}

check_ultralytics() {
    local venv="$ROOT_DIR/.venv"
    [ -f "$venv/bin/activate" ] || return  # check_venv already errored
    local ver
    if ! ver=$("$venv/bin/python" -c "import ultralytics; print(ultralytics.__version__)" 2>&1); then
        err "ultralytics import failed: $ver"
        return
    fi
    if [[ "$ver" != 8.* ]]; then
        err "ultralytics version $ver — expected 8.x"
        return
    fi
    ok "ultralytics $ver"
}

check_torch_cuda() {
    local venv="$ROOT_DIR/.venv"
    [ -f "$venv/bin/activate" ] || return
    local out
    if ! out=$("$venv/bin/python" -c "
import torch
print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device={torch.cuda.get_device_name(0)}')
    free, total = torch.cuda.mem_get_info(0)
    print(f'mem_free_gb={free / 1e9:.2f}')
    print(f'mem_total_gb={total / 1e9:.2f}')
" 2>&1); then
        err "torch import failed: $out"
        return
    fi
    echo "$out" | while read -r line; do echo "[preflight]   $line"; done
    if ! echo "$out" | grep -q 'cuda_available=True'; then
        warn "CUDA not available — training will run on CPU (very slow). Loop may still work for tiny datasets."
        return
    fi
    local free_gb
    free_gb=$(echo "$out" | awk -F= '/mem_free_gb=/ {print $2}')
    if [ -n "$free_gb" ]; then
        # 4GB minimum for nano models at imgsz=640; 8GB recommended for 1024+
        if awk "BEGIN {exit !($free_gb < 4.0)}"; then
            err "GPU has only ${free_gb}GB free — need ≥4GB. Another process may be using the card: nvidia-smi"
        elif awk "BEGIN {exit !($free_gb < 8.0)}"; then
            warn "GPU has ${free_gb}GB free — fine for nano at 640, may OOM at 1024+. Recommend ≥8GB."
        else
            ok "GPU memory ${free_gb}GB free"
        fi
    fi
}

check_disk() {
    local target_dir="$ROOT_DIR/runs"
    mkdir -p "$target_dir"
    local free_gb
    free_gb=$(df -BG --output=avail "$target_dir" | tail -1 | tr -dc '0-9')
    if [ -z "$free_gb" ]; then
        warn "could not parse disk usage from df — skipping check"
        return
    fi
    # A long training run can write 1-3 GB of checkpoints + logs.
    if [ "$free_gb" -lt 5 ]; then
        err "only ${free_gb}GB free on $target_dir — need ≥5GB to safely run training"
    elif [ "$free_gb" -lt 20 ]; then
        warn "${free_gb}GB free on $target_dir — fine for a few runs, top up before long sessions"
    else
        ok "disk ${free_gb}GB free"
    fi
}

check_nvidia_smi() {
    if ! command -v nvidia-smi >/dev/null; then
        warn "nvidia-smi not on PATH — cannot monitor GPU during training"
        return
    fi
    ok "nvidia-smi present"
}

check_dataset() {
    local ds="$1"
    if [ ! -d "$ds" ]; then
        err "dataset directory does not exist: $ds"
        return
    fi
    # YOLO expects images/{train,val} and labels/{train,val} OR a YAML pointing
    # to the right structure. We don't enforce a specific layout here (some
    # projects use class-folders for classify, train/val/test for OBB); just
    # confirm there's _something_ to train on.
    local img_count
    img_count=$(find "$ds" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | head -10000 | wc -l)
    if [ "$img_count" -eq 0 ]; then
        err "dataset has no images: $ds"
        return
    fi
    if [ "$img_count" -lt 20 ]; then
        warn "dataset has only $img_count images — YOLO needs ≥20 to train meaningfully; ≥100 for usable results"
    else
        ok "dataset has $img_count images"
    fi
}

# ─── Run checks for the requested mode ──────────────────────────────

case "$MODE" in
    cold)
        check_venv
        check_ultralytics
        check_torch_cuda
        check_disk
        check_nvidia_smi
        check_dataset "$TARGET"
        ;;
    quick)
        # Per-round: only the things that change between rounds. Venv and
        # ultralytics are static across the session.
        check_disk
        check_torch_cuda
        ;;
esac

echo "[preflight] summary: $ERR_COUNT errors, $WARN_COUNT warnings"
[ "$ERR_COUNT" -eq 0 ]
