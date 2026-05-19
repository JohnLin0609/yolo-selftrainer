#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "YOLO Self-Trainer Environment Setup"
echo "========================================"
echo "Root: $ROOT_DIR"
echo ""

# ─── Prerequisites ────────────────────────────────────────────────────
echo "[1/6] Checking prerequisites..."

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required. Found none."
    echo "Install with: sudo apt install python3.12 python3.12-venv"
    exit 1
fi
echo "  Python: $($PYTHON --version)"

if ! "$PYTHON" -m venv --help &>/dev/null; then
    echo "ERROR: python3-venv package not installed."
    echo "Install with: sudo apt install python3-venv"
    exit 1
fi
echo "  python3-venv: OK"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo "  GPU: $GPU_NAME (driver $CUDA_VER)"
else
    echo "  GPU: nvidia-smi not found (CPU training only)"
fi

# ─── Create venv ──────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting up virtualenv..."

VENV_DIR="$ROOT_DIR/.venv"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "  .venv/ already exists, reusing."
else
    "$PYTHON" -m venv "$VENV_DIR"
    echo "  Created .venv/"
fi

source "$VENV_DIR/bin/activate"

# ─── Install dependencies ────────────────────────────────────────────
echo ""
echo "[3/6] Installing dependencies..."

# Fail-loud (Harness §二): pip errors must not be swallowed. --quiet
# suppresses progress noise; the `| tail -1` we removed was masking errors.
if ! pip install --upgrade pip --quiet; then
    echo "ERROR: pip upgrade failed. See output above." >&2
    exit 1
fi
if ! pip install "ultralytics==8.4.23" --quiet; then
    echo "ERROR: ultralytics install failed. See output above." >&2
    exit 1
fi
# litellm enables the multi-LLM agent mode (P6). Optional for Claude-CLI mode
# but harmless to install — the venv is project-local. Pinned to 1.x major.
if ! pip install "litellm>=1.45,<2" --quiet; then
    echo "ERROR: litellm install failed (needed for --mode agent). See output above." >&2
    exit 1
fi
echo "  Done."

# ─── Verify installation ─────────────────────────────────────────────
echo ""
echo "[4/6] Verifying installation..."

# Fail-loud: keep stderr so import errors are visible.
if ! ULTRA_VER=$(python -c "import ultralytics; print(ultralytics.__version__)"); then
    echo "ERROR: ultralytics import failed (see error above)." >&2
    exit 1
fi
echo "  ultralytics: $ULTRA_VER"

if ! TORCH_INFO=$(python -c "import torch; print(f'{torch.__version__} CUDA={torch.cuda.is_available()}')"); then
    echo "ERROR: torch import failed (see error above)." >&2
    exit 1
fi
echo "  torch: $TORCH_INFO"

# CUDA absence is a warning, not an error — CPU training works for tiny
# datasets and we want setup to succeed on dev machines without a GPU.
# Loud warn (Harness §5.4): the operator must SEE that no GPU was found.
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    if ! GPU_TORCH=$(python -c "import torch; print(torch.cuda.get_device_name(0))"); then
        echo "  WARN: cuda is available but device name query failed" >&2
    else
        echo "  GPU device: $GPU_TORCH"
    fi
else
    echo "  WARN: CUDA not available — training will run on CPU (very slow). Continuing setup." >&2
fi

# ─── Create directories ──────────────────────────────────────────────
echo ""
echo "[5/6] Creating directory structure..."

for dir in models/pretrained runs datasets projects templates scripts; do
    mkdir -p "$ROOT_DIR/$dir"
done
echo "  OK"

# ─── Download models ─────────────────────────────────────────────────
echo ""
echo "[6/6] Downloading pretrained models..."

if [ -f "$SCRIPT_DIR/download_models.sh" ]; then
    bash "$SCRIPT_DIR/download_models.sh" --all
else
    echo "  WARN: download_models.sh not found. Run it manually later."
fi

# ─── Summary ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo "  Python:       $($PYTHON --version 2>&1)"
echo "  torch:        $TORCH_INFO"
echo "  ultralytics:  $ULTRA_VER"
echo "  venv:         $VENV_DIR"
echo "  models:       $ROOT_DIR/models/pretrained/"
echo "  runs:         $ROOT_DIR/runs/"
DISK_FREE=$(df -h "$ROOT_DIR" 2>/dev/null | awk 'NR==2{print $4}')
echo "  disk free:    $DISK_FREE"
echo ""
echo "To activate: source $VENV_DIR/bin/activate"
