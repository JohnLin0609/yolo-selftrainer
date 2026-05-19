#!/bin/bash
# One-command YOLO self-training: auto-detect everything, scaffold, and start.
# Usage: bash start_self_training.sh --dataset /path/to/dataset [--rounds 10]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Parse args ──────────────────────────────────────────────────────
DATASET=""
ROUNDS=10
# P6: mode + LLM selection. Defaults preserve original Claude-CLI behavior.
LOOP_MODE="claude"
LLM_PROVIDER=""
LLM_MODEL=""
LLM_API_BASE=""
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dataset)      DATASET="$2"; shift 2 ;;
        --rounds)       ROUNDS="$2"; shift 2 ;;
        --mode)         LOOP_MODE="$2"; shift 2 ;;
        --provider)     LLM_PROVIDER="$2"; shift 2 ;;
        --model)        LLM_MODEL="$2"; shift 2 ;;
        --api-base)     LLM_API_BASE="$2"; shift 2 ;;
        -h|--help)
            cat <<'HELP_EOF'
Usage: bash start_self_training.sh --dataset PATH [options]

  --dataset PATH           Path to dataset directory (required)
  --rounds N               Number of training rounds (default: 10)

P6 multi-LLM options:
  --mode {claude,agent}    Agent loop implementation (default: claude)
                             claude → uses `claude` CLI (Anthropic only)
                             agent  → uses scripts/run_agent.py via litellm
  --provider PROVIDER      LLM provider for agent mode:
                             anthropic | openai | gemini | groq | together_ai
                             ollama | vllm
  --model MODEL            Model id for the chosen provider (e.g.
                             claude-opus-4-7, gpt-4o, qwen2.5:32b)
  --api-base URL           Custom OpenAI-compatible endpoint (vLLM / self-host)

Everything else is auto-detected: task type, classes, resolution, model, split.
Pass extra flags to new_project.sh after --:
  bash start_self_training.sh --dataset /path --rounds 5 -- --name myproject --baseline 0.8

Examples:
  # Claude (default)
  bash start_self_training.sh --dataset ./datasets/foo --rounds 10

  # OpenAI GPT-4o
  bash start_self_training.sh --dataset ./datasets/foo --mode agent \
      --provider openai --model gpt-4o

  # Local Ollama
  bash start_self_training.sh --dataset ./datasets/foo --mode agent \
      --provider ollama --model qwen2.5:32b
HELP_EOF
            exit 0
            ;;
        --)         shift; EXTRA_ARGS=("$@"); break ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# Validate mode + LLM args
case "$LOOP_MODE" in
    claude|agent) ;;
    *) echo "ERROR: --mode must be 'claude' or 'agent' (got '$LOOP_MODE')" >&2; exit 1 ;;
esac
if [ "$LOOP_MODE" = "agent" ]; then
    if [ -z "$LLM_PROVIDER" ] || [ -z "$LLM_MODEL" ]; then
        echo "ERROR: --mode agent requires --provider and --model" >&2
        echo "Example: --mode agent --provider openai --model gpt-4o" >&2
        exit 1
    fi
fi

if [ -z "$DATASET" ]; then
    echo "ERROR: --dataset is required."
    echo "Usage: bash start_self_training.sh --dataset /path/to/dataset [--rounds 10]"
    exit 1
fi

if [ ! -d "$DATASET" ]; then
    echo "ERROR: Dataset path does not exist: $DATASET"
    exit 1
fi
# Now that we know the path exists, fail loud if realpath itself errors
# (Harness §二 — no silent fallback to a potentially bogus path).
RESOLVED="$(realpath "$DATASET")"
if [ -z "$RESOLVED" ]; then
    echo "ERROR: realpath failed on $DATASET" >&2
    exit 1
fi
DATASET="$RESOLVED"

# ─── Ensure environment is set up ────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "[setup] First run — setting up environment..."
    bash "$SCRIPT_DIR/scripts/setup_env.sh"
    if [ $? -ne 0 ]; then
        echo "ERROR: Environment setup failed."
        exit 1
    fi
fi

# ─── Preflight (Harness §十一 — synchronous + necessary before render) ──
# Catch problems at boot rather than 1 hour into training.
if ! bash "$SCRIPT_DIR/scripts/preflight.sh" --cold "$DATASET"; then
    echo "" >&2
    echo "ERROR: preflight failed. Fix the issues above before re-running." >&2
    exit 1
fi
echo ""

# ─── Derive project name from dataset folder ─────────────────────────
NAME=$(basename "$DATASET" | sed 's/[^a-zA-Z0-9_-]/_/g')

# Check if this project already has a running training
PROJECT_DIR="$SCRIPT_DIR/projects/$NAME"
if [ -f "$PROJECT_DIR/train.pid" ]; then
    OLD_PID=$(cat "$PROJECT_DIR/train.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Training is already running for '$NAME' (PID $OLD_PID)."
        echo "Monitor: tail -f $PROJECT_DIR/current.log"
        echo "Stop:    kill $OLD_PID"
        exit 0
    fi
fi

# ─── Scaffold project ────────────────────────────────────────────────
echo "========================================"
echo "YOLO Self-Training"
echo "========================================"
echo "Dataset: $DATASET"
echo "Rounds:  $ROUNDS"
echo ""

SCAFFOLD_ARGS=(
    --dataset "$DATASET"
    --max-rounds "$ROUNDS"
    --force
    --mode "$LOOP_MODE"
)
if [ "$LOOP_MODE" = "agent" ]; then
    SCAFFOLD_ARGS+=(--llm-provider "$LLM_PROVIDER" --llm-model "$LLM_MODEL")
    [ -n "$LLM_API_BASE" ] && SCAFFOLD_ARGS+=(--llm-api-base "$LLM_API_BASE")
fi
SCAFFOLD_ARGS+=("${EXTRA_ARGS[@]}")

bash "$SCRIPT_DIR/scripts/new_project.sh" "${SCAFFOLD_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "ERROR: Project scaffolding failed."
    exit 1
fi

# ─── Reset session state ─────────────────────────────────────────────
# --rounds means "run N rounds from now", so always start fresh
rm -f "$PROJECT_DIR/round_counter" "$PROJECT_DIR/session_id" "$PROJECT_DIR/last_run_name" "$PROJECT_DIR/train_completed"

# ─── Start the training loop ─────────────────────────────────────────
echo ""
echo "========================================"
echo "Starting autonomous training loop..."
echo "========================================"
echo ""

cd "$PROJECT_DIR"
if [ "$LOOP_MODE" = "agent" ]; then
    exec bash start_agent.sh
else
    exec bash start_claude.sh
fi
