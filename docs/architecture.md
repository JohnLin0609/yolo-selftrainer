# Architecture

How the YOLO Self-Trainer is wired, and why each piece is where it is.

The design follows the **Harness Engineering** patterns: event sourcing as
source of truth, real trust boundaries instead of advisory rules, fail-loud
over silent fallback, and budget-aware context engineering.

---

## The loop

```
┌────────────────────┐    1. preflight (GPU / disk / dataset)
│ start_self_train   │ ──→ 2. scaffold projects/<name>/
└──────────┬─────────┘    3. exec start_claude.sh OR start_agent.sh
           │
           ▼
┌────────────────────┐
│  start_*.sh        │     - HALTED check → exit if present
│  (per round)       │     - per-round preflight
│                    │     - bump round counter (round_started event)
│                    │     - build prompt (events.jsonl facts + Claude's notes)
│                    │     - run agent (claude CLI or run_agent.py via LiteLLM)
└──────────┬─────────┘
           │ agent decides what to change, sed-edits train.sh,
           │ launches `nohup bash train.sh &` (detached)
           ▼
┌────────────────────┐
│  train.sh          │     - param validator (range-check, abort on violation)
│                    │     - yolo {task} train ... (foreground)
│                    │     - emit training_finished + training_metrics events
│                    │     - on success: clear consecutive_failures
│                    │     - on crash: circuit breaker (3 strikes → HALTED)
│                    │     - rm train.pid, then recurse: bash start_*.sh
└────────────────────┘
```

Each step that mutates state emits an event. After N rounds, the chain
naturally stops (`MAX_ROUNDS` check at the top of `start_*.sh`).

---

## State: events.jsonl is source of truth

Three categories of state, each with a different lifetime:

| Kind | Examples | Storage |
|---|---|---|
| **Permanent audit** | round_started, training_metrics, halted, claude_finished | `events.jsonl` (append-only, never deleted) |
| **Cache for fast bash access** | `round_counter`, `train_completed`, `last_run_name`, `consecutive_failures` | Single-file mutables in project dir |
| **Per-round artifacts** | prompt, raw LLM output, session log | `logs/<session>/<run_name>/` |

The cache files exist so bash can read state without spawning Python.
**They can always be rebuilt from `events.jsonl`** — events are canonical.

`scripts/event.py` is the only writer/reader for events:

```bash
# Query current round (max round number across all events)
python3 scripts/event.py <project> query current-round

# Per-round eval metrics table (P, R, mAP50, mAP50-95, losses, overfit gap)
python3 scripts/event.py <project> query metrics-table

# Failure count since last successful training
python3 scripts/event.py <project> query consecutive-failures
```

Event schema is fixed in `EVENT_TYPES` — a typo in `event_type` fails at
emit time, so the audit log never accumulates unparseable garbage.

---

## Prompt engineering: three layers

`scripts/build_prompt.py` constructs each round's prompt with three sections,
in increasing order of trust:

### 1. `## Verified facts` — machine-extracted from `events.jsonl`

```
- Last round in log: 5
- Total successful training runs: 4
- Consecutive failures: 0
- BEST so far: mAP50(B)=0.9950 at run `...` epoch 106 (round 5)
- Last 4 runs trajectory: [0.800, 0.891, 0.989, 0.995]
```

These are the only numbers the agent should trust. If its own prose disagrees,
trust facts.

### 2. `## Per-round metrics at best epoch`

```
| # | run | ep | Δ | P | R | mAP50 | mAP50-95 | gap |
| 1 | ... |  89 |   — | 0.68 | 0.61 | 0.80 | 0.34 | 0.06 |
| 2 | ... | 180 | +0.09 | 0.89 | 0.75 | 0.89 | 0.48 | 0.20 |
| 3 | ... |  55 | +0.10 | 0.96 | 0.79 | 0.99 | 0.47 | 0.36 |
```

Every metric in `results.csv` at the best epoch of each run, including the
overfit gap. The agent reads this to spot trends without re-doing CSV math.

### 3. `## Claude's notes from previous round (FREE-FORM — may contain errors)`

The previous round's `next_instruction.md` prose, with auto-generated
duplicate sections stripped. Marked clearly as Claude's hypothesis, not
ground truth.

The prose budget defaults to 12,000 characters and can be overridden via
`YOLO_TRAINER_PROSE_BUDGET=8000`. Over budget → head+tail truncation with
an explicit `[... N chars omitted ...]` marker so the model knows.

---

## Trust boundaries

The framework runs an LLM unattended. Three real boundaries (not advisory):

### Boundary 1 — Bash guard (`scripts/claude_bash_guard.py`)

Runs as a PreToolUse hook (Claude CLI mode) or as a subprocess called by
`run_agent.py` (agent mode). Same code, same deny list, two dispatch
mechanisms.

```python
DENY_HEADS = {
    "rm", "rmdir", "sudo", "chmod", "chown", "mv", "cp", "ln",
    "dd", "mkfs", "shutdown", "reboot", "init",
    "pip", "pip3", "uv", "poetry", "conda",
    "npm", "yarn", "pnpm",
    "ssh", "scp", "sftp", "rsync", "curl", "wget",
    "docker", "podman", "kubectl",
    "systemctl", "service",
}
```

Critical: compound commands are **split per-subcommand** before matching.
`echo hi && rm -rf /` correctly hits `rm`, not the leading `echo`.

### Boundary 2 — Param validator (top of `train.sh`)

Hyperparameters are range-checked before YOLO launches. Out-of-range params
abort the run, emit a `validation-failed` event, and do NOT consume a
round — Claude wakes up, sees the abort message, can fix and retry.

```python
rng('LR',           0.0001, 0.05)
rng('EPOCHS',       50,     500)
rng('PATIENCE',     20,     100)
rng('IMGSZ',        640,    1280)   # must also be multiple of 32
rng('BATCH',        2,      64,    allow=('-1',))
rng('MOSAIC',       0.0,    1.0)
rng('WEIGHT_DECAY', 0.0,    0.005)
opt in {'AdamW', 'SGD', 'auto'}
```

### Boundary 3 — Circuit breaker

Failure types counted: training crash (exit ≠ 0), parameter validation
abort, preflight failure between rounds, LLM exit error.

```
failure 1/3 → next_instruction.md updated with crash report, wake agent
failure 2/3 → same
failure 3/3 → write HALTED with diagnostic instructions, do NOT wake agent
```

`start_*.sh` checks `HALTED` at the top of every wakeup and refuses to
run until the operator removes it.

---

## Multi-LLM via LiteLLM

`scripts/run_agent.py` is a ~500-line LiteLLM-based ReAct loop that mirrors
the Claude CLI's interface (stream-json output, same `Bash`/`Write`/`Read`
tool catalog) so logs render identically across modes.

Per-provider tool-calling nudges live in `PROVIDER_TOOL_NUDGE`:

```python
PROVIDER_TOOL_NUDGE = {
    "anthropic": "",   # Opus 4.x doesn't need extra nudging
    "openai":    "",
    "gemini": "...IMPORTANT: You MUST use the tools...",
    "ollama":  "...EVERY filesystem op MUST be a tool call...",
    ...
}
```

For Ollama specifically, `think=False` is passed via `extra_body` because
thinking mode breaks multi-turn tool calling on Qwen3 / DeepSeek models.

Trade-offs intentionally taken:
- **Non-streaming**: avoids the per-block-index buffer / thinking-block
  preservation complexity. The loop is short-lived (one session per round),
  so buffering the full response is acceptable.
- **Sequential tool dispatch**: parallel tool calls flatten to sequential in
  the loop. Simpler, still correct.

---

## Bootstrap order

`start_self_training.sh` follows the Harness §11 discipline:

| Category | Check | Action on failure |
|---|---|---|
| **Synchronous + necessary** | `.venv` exists | exit, run `setup_env.sh` |
| Synchronous + necessary | `ultralytics` importable | exit |
| Synchronous + necessary | dataset directory non-empty | exit |
| Synchronous + warning | CUDA available | warn (CPU training works for tiny datasets) |
| Synchronous + warning | GPU memory ≥ 4 GB free | warn (might OOM on IMGSZ=1280) |
| Synchronous + warning | Disk free ≥ 5 GB | warn (long runs may fill) |
| Per-round preflight | GPU + disk still OK | write HALTED, exit |

No silent fallback. Every "the operator thought they had X but didn't"
state is surfaced loudly.

---

## File map

```
yolo-selftrainer/
├── start_self_training.sh        # the front door — scaffold + launch
├── scripts/
│   ├── setup_env.sh              # .venv + ultralytics + litellm + models
│   ├── download_models.sh        # pretrained yolo11n.pt, yolov8n.pt, etc.
│   ├── download_demo_dataset.sh  # COCO128 subset for quickstart
│   ├── new_project.sh            # scaffold projects/<name>/ from templates
│   ├── preflight.sh              # --cold (full check) / --quick (per-round)
│   ├── event.py                  # events.jsonl emit + query CLI
│   ├── build_prompt.py           # per-round prompt with facts+metrics+prose
│   ├── run_agent.py              # LiteLLM ReAct loop (agent mode)
│   ├── claude_bash_guard.py      # PreToolUse Bash deny-list filter
│   ├── generate_report.py        # final-round training_report.md
│   └── test_agent_smoke.py       # one-turn smoke test per provider
├── templates/
│   ├── train.sh.tmpl             # YOLO training wrapper (param validator + circuit breaker)
│   ├── start_claude.sh.tmpl      # Claude CLI orchestrator
│   ├── start_agent.sh.tmpl       # multi-LLM orchestrator (run_agent.py)
│   ├── agent.env.tmpl            # per-project LLM config
│   ├── settings.json.tmpl        # Claude CLI PreToolUse hook registration
│   ├── yolo_folder_skill.md.tmpl # system prompt — environment + rules
│   ├── hyperparameter_strategy.md.tmpl # system prompt — decision framework
│   └── .gitignore.tmpl           # per-project gitignore
├── docs/
│   ├── architecture.md           # this file
│   ├── usage.md                  # operator guide
│   ├── extending.md              # adding providers, actions, events
│   └── local-llms.md             # why local LLMs are marginal here
├── projects/                     # user-scaffolded, never committed
├── models/pretrained/            # downloaded by setup_env.sh
├── runs/<task>/                  # YOLO output, ignored
├── datasets/                     # user data, ignored
├── README.md
└── LICENSE
```
