#!/bin/bash
# Do NOT use set -e — claude non-zero exit must not silently kill the loop

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="/home/johnlin/workspace/aoi/yolov11obb/.venv"
FOLDER_SKILL="$SCRIPT_DIR/yolo_folder_skill.md"
PARAM_SKILL="$SCRIPT_DIR/hyperparameter_strategy.md"
INSTRUCTION="$SCRIPT_DIR/next_instruction.md"
ROUND_FILE="$SCRIPT_DIR/round_counter"
SESSION_FILE="$SCRIPT_DIR/session_id"
LOGS_DIR="$SCRIPT_DIR/logs"
MAX_ROUNDS=10   # ← change this to set how many training cycles to run

# Activate the yolov11obb venv so Claude's Bash tool inherits it
source "$VENV/bin/activate"

for f in "$FOLDER_SKILL" "$PARAM_SKILL"; do
  if [ ! -f "$f" ]; then
    echo "[claude] ERROR: missing $f"
    exit 1
  fi
done

# ─── Round counter ───────────────────────────────────────────────────
ROUND=1
if [ -f "$ROUND_FILE" ]; then
  ROUND=$(cat "$ROUND_FILE")
  ROUND=$((ROUND + 1))
fi
echo "$ROUND" > "$ROUND_FILE"

# ─── Session ID (timestamp folder for this batch of rounds) ──────────
if [ "$ROUND" -eq 1 ] || [ ! -f "$SESSION_FILE" ]; then
  SESSION_ID=$(date +%Y%m%d_%H%M%S)
  echo "$SESSION_ID" > "$SESSION_FILE"
else
  SESSION_ID=$(cat "$SESSION_FILE")
fi
SESSION_LOG_DIR="$LOGS_DIR/$SESSION_ID"
mkdir -p "$SESSION_LOG_DIR"

# Clear stale last_run_name so we only pick up this round's training
rm -f "$SCRIPT_DIR/last_run_name"

echo "[claude] =============================="
echo "[claude] Session: $SESSION_ID | Round $ROUND / $MAX_ROUNDS"

if [ "$ROUND" -gt "$MAX_ROUNDS" ]; then
  echo "[claude] MAX_ROUNDS ($MAX_ROUNDS) reached. Stopping loop."
  echo "[claude] next_instruction.md preserved for future resume."
  echo "[claude] Logs: $SESSION_LOG_DIR/"
  echo "[claude] To continue: rm round_counter session_id && bash start_claude.sh"
  echo "0" > "$ROUND_FILE"
  rm -f "$SESSION_FILE"
  exit 0
fi

# ─── Prepare temp log dir (renamed after Claude launches training) ───
TEMP_LOG_DIR="$SESSION_LOG_DIR/round_${ROUND}_pending"
mkdir -p "$TEMP_LOG_DIR"

SYSTEM_PROMPT="$(cat "$FOLDER_SKILL")

---

$(cat "$PARAM_SKILL")"

# ─── Build prompt ────────────────────────────────────────────────────
ROUND_INFO="Round $ROUND of $MAX_ROUNDS."
PREV_ROUND=$((ROUND - 1))

if [ "$ROUND" -eq "$((MAX_ROUNDS - 1))" ]; then
  ROUND_INFO="$ROUND_INFO THIS IS ROUND $ROUND — THE SECOND-TO-LAST ROUND.
You MUST launch a LONG PRODUCTION TRAINING RUN this round (Action K in hyperparameter_strategy.md):
- Use best.pt from the best run so far
- Set EPOCHS=2000, PATIENCE=500
- Keep the best hyperparameters found during exploration
- This run will finish overnight and produce a stable model."
fi

if [ "$ROUND" -eq "$MAX_ROUNDS" ]; then
  ROUND_INFO="$ROUND_INFO THIS IS THE FINAL ROUND. Do NOT launch training. Instead:
1. Read the latest run results (likely the long production run from round $PREV_ROUND).
2. Write a comprehensive next_instruction.md summarizing all run history, best results, best model path, and recommendations for the next session.
3. Exit."
fi

if [ -f "$INSTRUCTION" ]; then
  echo "[claude] Resuming from next_instruction.md"
  PROMPT="$ROUND_INFO

$(cat "$INSTRUCTION")"
  # Do NOT delete yet — keep as fallback until claude succeeds
else
  echo "[claude] Cold start"
  PROMPT="$ROUND_INFO

Cold start — OBB wheel defect detection.

IMPORTANT: The virtualenv at /home/johnlin/workspace/aoi/yolov11obb/.venv is already activated.
ultralytics is already installed there. Do NOT run pip install.

Do the following in order:
1. Verify ultralytics is available:
   python -c 'import ultralytics; print(ultralytics.__version__)'
2. Validate the dataset YAML from train.sh:
   DATASET_PATH=\$(grep '^DATASET=' $SCRIPT_DIR/train.sh | cut -d'\"' -f2)
   cat \"\$DATASET_PATH\"
   ls \$(grep '^path:' \"\$DATASET_PATH\" | awk '{print \$2}')/images/train/ | wc -l
3. Check current train.sh params:
   grep -E '^(WEIGHTS|EPOCHS|LR|LR_FINAL|BATCH|IMGSZ|PATIENCE|OPTIMIZER|DATASET)=' $SCRIPT_DIR/train.sh
4. Apply hyperparameter_strategy.md to pick starting params for first run.
   This is OBB task — model should be yolo11n-obb.pt.
   Edit train.sh with sed as needed. Verify edits with grep.
5. Launch detached:
   nohup bash $SCRIPT_DIR/train.sh > $SCRIPT_DIR/current.log 2>&1 &
   echo \$! > $SCRIPT_DIR/train.pid
6. Write next_instruction.md (see format in yolo_folder_skill.md)."
fi

# ─── Save instruction to log ─────────────────────────────────────────
printf '%s\n' "$PROMPT" > "$TEMP_LOG_DIR/instruction.md"

# ─── Run Claude and capture full output ──────────────────────────────
# --allowedTools MUST remain last: it is variadic and will consume any
# following positional arguments as tool names.
if printf '%s\n' "$PROMPT" | claude --dangerously-skip-permissions \
  --print \
  --verbose \
  --output-format stream-json \
  --system-prompt "$SYSTEM_PROMPT" \
  --allowedTools "Bash,Write,Read" \
  > "$TEMP_LOG_DIR/claude_raw.jsonl" 2>&1; then

  CLAUDE_EXIT=0
else
  CLAUDE_EXIT=$?
fi

# ─── Extract readable log from stream-json ────────────────────────────
python3 -c "
import json, sys
for line in open('$TEMP_LOG_DIR/claude_raw.jsonl'):
    line = line.strip()
    if not line: continue
    try: msg = json.loads(line)
    except: continue
    t = msg.get('type', '')
    if t == 'assistant':
        for block in msg.get('message', {}).get('content', []):
            if block.get('type') == 'text':
                print('--- CLAUDE ---')
                print(block['text'])
                print()
            elif block.get('type') == 'tool_use':
                name = block.get('name', '?')
                inp = block.get('input', {})
                if 'command' in inp:
                    print(f'--- TOOL: {name} ---')
                    print(f'  \$ {inp[\"command\"]}')
                elif 'file_path' in inp:
                    print(f'--- TOOL: {name} ---')
                    print(f'  file: {inp[\"file_path\"]}')
                    if 'content' in inp:
                        print(f'  (write {len(inp[\"content\"])} chars)')
                    if 'old_string' in inp:
                        print(f'  replacing {len(inp[\"old_string\"])} chars')
                else:
                    print(f'--- TOOL: {name} ---')
                    print(f'  {json.dumps(inp)[:200]}')
                print()
    elif t == 'tool':
        content = msg.get('content', '')
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get('text'):
                    content = c['text']; break
        if isinstance(content, str) and content.strip():
            lines = content.strip().split('\n')
            if len(lines) > 30:
                print('\n'.join(lines[:15]))
                print(f'  ... ({len(lines)-30} lines omitted) ...')
                print('\n'.join(lines[-15:]))
            else:
                print(content.strip())
            print()
    elif t == 'result':
        cost = msg.get('total_cost_usd', 0)
        dur = msg.get('duration_ms', 0)
        turns = msg.get('num_turns', 0)
        print(f'--- RESULT: {msg.get(\"subtype\",\"?\")} | {turns} turns | {dur/1000:.1f}s | \${cost:.4f} ---')
        print()
" > "$TEMP_LOG_DIR/claude_session.log" 2>/dev/null

# ─── Rename log folder to match the training run name ─────────────────
sleep 3
if [ -f "$SCRIPT_DIR/last_run_name" ]; then
  ACTUAL_RUN_NAME=$(cat "$SCRIPT_DIR/last_run_name")
  FINAL_LOG_DIR="$SESSION_LOG_DIR/$ACTUAL_RUN_NAME"
  # Use mv -T to treat dest as file name, not directory (prevents nesting)
  if [ ! -d "$FINAL_LOG_DIR" ]; then
    mv "$TEMP_LOG_DIR" "$FINAL_LOG_DIR" 2>/dev/null
    echo "[claude] Logs: $FINAL_LOG_DIR/"
  else
    # Dest already exists (shouldn't happen) — keep pending name
    echo "[claude] WARN: $FINAL_LOG_DIR already exists, keeping $TEMP_LOG_DIR/"
  fi
else
  echo "[claude] Logs: $TEMP_LOG_DIR/"
fi

if [ $CLAUDE_EXIT -eq 0 ]; then
  # Do NOT delete next_instruction.md — Claude wrote a NEW one for the next round
  echo "[claude] Session complete."
else
  echo "[claude] ERROR: claude exited with code $CLAUDE_EXIT" >&2
  echo "[claude] Preserving next_instruction.md for retry." >&2
  exit 1
fi
