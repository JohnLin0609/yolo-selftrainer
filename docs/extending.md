# Extending the framework

Three common extension points: adding a new LLM provider, adding a new
action playbook, and hooking into `events.jsonl` for external dashboards.

---

## Adding a new LLM provider

LiteLLM already supports 100+ providers; you usually don't need code
changes — just set the env var and use the right `--provider`/`--model`
combo. If your provider needs a specific tweak:

### 1. Add it to `PROVIDER_TOOL_NUDGE` in `scripts/run_agent.py`

If the model is weak at tool calling unless prompted explicitly, append a
nudge to its system prompt:

```python
PROVIDER_TOOL_NUDGE = {
    ...
    "my_provider": (
        "\n\nIMPORTANT: You MUST use the structured tool_calls API, not "
        "emit JSON inline as text."
    ),
}
```

### 2. Add provider-specific kwargs

If the provider needs special LiteLLM params (e.g. disable thinking,
specific stop sequences), add to the `extra_kwargs` block in `main()`:

```python
extra_kwargs = {}
if args.provider == "ollama":
    extra_kwargs["extra_body"] = {"think": False}
elif args.provider == "my_provider":
    extra_kwargs["extra_body"] = {"some_flag": True}
```

### 3. Test with the smoke runner

```bash
python3 scripts/test_agent_smoke.py --providers my_provider --model my-model-id
```

A passing smoke test confirms the provider can do one-tool ReAct. Multi-tool
reliability requires real testing on your own data.

---

## Adding a new action playbook

Actions A–J are documented in `templates/hyperparameter_strategy.md.tmpl`.
The format is loose markdown — Claude reads it as part of the system prompt
and picks actions based on the decision tree.

To add Action L (say, "swap to yolov11s for the final push"):

### 1. Edit `templates/hyperparameter_strategy.md.tmpl`

Insert after Action J:

```markdown
### Action L: Upgrade to Small (yolo11s) for final push

**When**: mAP50 ≥ 0.85, validation loss stable, you've exhausted nano-model
tuning and want to test if a slightly larger model can push higher.

**Cost**: 2-3x slower training, 2-3x memory, ~2x model size.

\`\`\`bash
sed -i 's|^WEIGHTS=.*|WEIGHTS=${1:-"{{MODELS_DIR}}/yolo11s.pt"}|' {{PROJECT_DIR}}/train.sh
sed -i 's/^LR=.*/LR=0.01/' {{PROJECT_DIR}}/train.sh   # reset for new arch
sed -i 's/^EPOCHS=.*/EPOCHS=150/' {{PROJECT_DIR}}/train.sh
\`\`\`

**Rules**:
- Use **pretrained** yolo11s.pt — NOT a nano best.pt (arch incompatible)
- Reset LR to 0.01 (fresh start for new arch)
```

### 2. Make sure `scripts/download_models.sh` pulls yolo11s.pt

Already does by default if you ran `scripts/setup_env.sh`.

### 3. (Optional) Update the cycle decision tree

In `scripts/new_project.sh`, find the `DT2_EOF` heredoc and add a branch:

```bash
  ├─ mAP50 ≥ 0.85 AND no arch swap tried?
  │   YES → Consider Action L (upgrade to yolo11s)
```

Re-scaffold any project (or edit `hyperparameter_strategy.md` in place for
running projects). Claude will see the new action on its next round.

---

## Hooking into events.jsonl

`events.jsonl` is the authoritative log of everything that happens. The
schema is documented in `scripts/event.py` (the `EVENT_TYPES` set). You
can tail it with `jq` for a live dashboard or feed it into anything that
speaks JSON.

### Example: Slack notification on round completion

```bash
# In a separate shell
tail -F projects/my_project/events.jsonl | while read -r line; do
  type=$(echo "$line" | jq -r .type)
  if [ "$type" = "training_metrics" ]; then
    run=$(echo "$line" | jq -r .run_name)
    map=$(echo "$line" | jq -r .best_metric_value)
    curl -X POST -H 'Content-type: application/json' \
      --data "{\"text\":\"Round done: $run mAP50=$map\"}" \
      "$SLACK_WEBHOOK_URL"
  fi
done
```

### Example: Custom report

```python
import json
from pathlib import Path

events = [json.loads(l) for l in Path("projects/my_project/events.jsonl").read_text().splitlines() if l.strip()]

# Total wall-clock time spent training
training_time = sum(
    e["duration_sec"]
    for e in events
    if e["type"] == "training_finished" and e.get("exit_code") == 0
)
print(f"Total training time: {training_time/60:.1f} min")

# All hyperparameter changes (training_started carries params dict)
for e in events:
    if e["type"] == "training_started":
        print(f"Round {e['round']}: {e.get('params')}")
```

---

## Adding a new event type

If you want to record something new in `events.jsonl`:

### 1. Add the type to `EVENT_TYPES` in `scripts/event.py`

```python
EVENT_TYPES = {
    ...
    "my-new-event",
}
```

### 2. Add its required fields to `EVENT_FIELDS`

```python
EVENT_FIELDS = {
    ...
    "my-new-event": {
        "round": int,
        "my_payload_json": "json",
    },
}
```

The `_json` suffix marks a JSON-decoded field. Required ones can be `int`,
`float`, `str`, `bool`, or `"json"`.

### 3. Emit it from anywhere

```bash
python3 scripts/event.py <project> emit my-new-event \
  --round 5 \
  --my-payload-json '{"key": "value"}'
```

### 4. (Optional) Query it

Add a new query case to `q_*` functions in `event.py` and the `query`
subparser's `choices` list.

---

## Writing your own agent backend

If LiteLLM doesn't cover your provider (e.g., a fully custom inference
server), the integration point is `scripts/run_agent.py`. The contract:

- **Input**: system prompt + user prompt (both file-based), tool catalog,
  max turns, guard script path.
- **Output**: stream-json-compatible JSONL file (the format Claude CLI
  emits), so the existing `start_agent.sh` log extractor renders it
  identically.

Look at `msg_to_log_assistant()` in `run_agent.py` for the exact JSON
shape needed. The tool dispatch logic (`dispatch_tool()`, `run_bash()`,
`run_write()`, `run_read()`) is provider-agnostic and can be reused.

Then have your `start_my_backend.sh.tmpl` invoke your script instead of
`run_agent.py`. `train.sh` auto-detects `start_agent.sh` first, falling
back to `start_claude.sh` — extend it to detect your script too.
