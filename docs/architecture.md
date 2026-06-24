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

The framework runs an LLM unattended. Boundaries 2 and 3 below are
**deterministic gates** — they hard-fail any input outside their
contract. **Boundary 1 (the Bash guard) is best-effort today** — a
denylist of command heads plus train.sh-write regexes. It catches the
obvious dangerous commands (`rm`, `sudo`, `pip`, …) but has known
bypasses; do not treat it as the sole protection for sensitive paths.
The canonical contract for the hardened guard lives in
`tests/features/bypass_attempts.feature` and
`tests/features/sandbox_isolation.feature`.

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

The guard also rejects writes to `train.sh` via any mechanism (sed -i,
> redirect, tee, awk -i inplace, perl -i, cp/mv into the path). The
agent's contract is to write `next_params.json`; the guard is the wall
that backs that contract up if the prompt regresses.

#### Known gaps (Boundary 1 is denylist-based)

The current head-token denylist is bypassable. Any of the following
are **allowed** today and would let the agent reach arbitrary commands:

- **Interpreter wrappers** — the head token is `python3` / `bash` / `sh` /
  `eval`, not on the deny list, but the payload is arbitrary:
  - `python3 -c "import os; os.remove('/x')"`
  - `bash -c 'rm /x'`, `sh -c '...'`, `eval 'rm /x'`
- **find primitives** — head is `find`, payload erases without invoking `rm`:
  - `find . -delete`, `find / -exec rm {} \;`
- **Embedded interpreters** in otherwise-benign tools:
  - `awk 'BEGIN{system("rm /x")}' /dev/null`, `perl -e 'unlink "/x"'`
- **xargs feeding a wrapper** — dangerous tail buried in argv:
  - `xargs -I{} sh -c 'rm {}' < list.txt`
- **Command substitution at the head position** — the outer guard sees
  the substitution syntax, not the resolved command:
  - `$(echo rm) -rf /x`, `` `echo rm` -rf /x ``
- **Heredocs feeding an interpreter** — equivalent to `bash -c`:
  - `bash <<'EOF'\nrm /x\nEOF`

These cases are documented as `xfail` tests in
`tests/unit/test_claude_bash_guard.py::test_bypass_attempts_blocked` and
as scenarios in `tests/features/bypass_attempts.feature`. The xfail
markers flip to passing tests when the predicate is hardened
(approach 2 — allow-list) or when the agent's Bash runs inside an OS
sandbox (approach 1 — container/namespace isolation). The contract
for approach 1 lives in `tests/features/sandbox_isolation.feature`
(SKIPPED today; auto-runs once `scripts/sandbox.run_in_sandbox` exists).

Operational implication: **do not rely on Boundary 1 alone** for
protecting host paths outside the project tree. Boundaries 2 and 3
remain deterministic; Boundary 1 is a partial wall.

### Boundary 2 — Param contract (`next_params.json` + `apply_params.py`)

The agent never edits `train.sh` directly. Each round it writes a flat
JSON file `next_params.json` (one entry per hyperparameter). The contract
has three layers:

1. **Single source of truth** — `scripts/param_bounds.py` holds the
   `BASE_BOUNDS` dict and `REQUIRED_KEYS` set. `bounds_for(task, fine_tune)`
   derives per-context bounds: fine-tune from `best.pt` opens `EPOCHS` floor
   to 10; pose tightens `DEGREES` to ≤15; classify zeros out
   `MOSAIC`/`COPY_PASTE`. `scripts/baseline_policy.py` also reads
   `BASE_BOUNDS` so its sampled hyperparameters fall inside by construction.

2. **Primary boundary** — `scripts/apply_params.py` is the SOLE writer of
   `effective_params.env` (the file train.sh sources). It schema-checks
   (REQUIRED_KEYS present, unknown keys rejected as typos), range-checks
   via `param_bounds.validate`, and on any failure emits the
   `validation-failed` event with structured violations + writes a clear
   "fix this" message to `next_instruction.md`. Round NOT consumed.

3. **Defense-in-depth** — train.sh sources `effective_params.env`, then
   re-calls `python3 scripts/param_bounds.py validate-env --task TASK
   --weights "$WEIGHTS"` against the resolved env. Catches the case where
   apply_params is bypassed (manual launch, broken script, future bug).

```python
# scripts/param_bounds.py
BASE_BOUNDS = {
    "LR":        {"type": "float", "min": 0.0001, "max": 0.05},
    "EPOCHS":    {"type": "int",   "min": 50,     "max": 500},
    "IMGSZ":     {"type": "int",   "min": 640,    "max": 1280, "multiple_of": 32},
    "BATCH":     {"type": "int",   "min": 2,      "max": 64,   "allow": (-1,)},
    "OPTIMIZER": {"type": "choice", "choices": ["AdamW", "SGD", "auto"]},
    # … see source for the full table
}
REQUIRED_KEYS = {"WEIGHTS", "EPOCHS", "LR", "LR_FINAL",
                 "IMGSZ", "BATCH", "OPTIMIZER", "PATIENCE"}
```

The validation-failed event's `violations_json` carries the structured
list `[{key, expected, got, reason}, …]` — replaying events.jsonl tells
you exactly what the agent got wrong each time.

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

### Boundary 4 — Operator-only data (held-out test split)

The agent tunes against val metrics. If val is small or noisy, the agent
can fit val without actually generalizing — and the framework has no way
to tell. The held-out test split is an unbiased post-hoc benchmark that
the agent **never sees**, so val-overfitting becomes visible to the
operator.

How the firewall is enforced:

1. `scripts/new_project.sh` carves a fixed-seed test split at scaffold
   time. Test images live in `images/test/` and `labels/test/`; the
   yaml's `test:` key points at them. The split is locked by
   `--test-seed` (default `42`) — same seed always yields the same test
   set, so it stays comparable across re-scaffolds.

2. `templates/train.sh.tmpl` calls `scripts/run_test_eval.py` after
   each successful training run. The helper runs `yolo val split=test`
   on `best.pt` and emits a `test_metrics` event into `events.jsonl`.

3. `scripts/build_prompt.py` has three independent firewall layers
   keyed on the constant `AGENT_INVISIBLE_EVENT_TYPES = {"test_metrics"}`:

   - **Layer 1** (denylist): single source of truth at module top.
   - **Layer 2** (load filter): `load_events()` drops invisible types,
     so no downstream loop can accidentally read them.
   - **Layer 3** (output sanity check): scans the assembled prompt for
     guard strings (`test_metrics`, `split=test`, `test_mAP`, …) and
     exits non-zero on any hit. This catches future regressions where
     a contributor reads test data via a path other than `load_events`.

4. `scripts/generate_report.py` reads `test_metrics` events (operator
   tool — not agent-facing) and renders side-by-side val/test columns
   plus a divergence-trend insight, so val-overfitting jumps out.

To add a new operator-only event type, list it in
`AGENT_INVISIBLE_EVENT_TYPES` AND add distinctive guard substrings to
`_PROMPT_GUARD_TERMS` in the same file. Both must be updated together.

### Boundary 5 — Plateau circuit

The crash circuit breaker (Boundary 3) handles hard failures. Plateau is the
quieter failure mode: the agent keeps proposing variations of the same
direction and the primary metric stops moving. The plateau circuit is
independent — keyed on training *success*, not failure — and additive to
the crash breaker.

State machine (`q_plateau_status` in `scripts/event.py`):

```
                  improvement ≥ threshold
                  ┌──────────────────┐
                  ▼                  │
   insufficient → ok ── < threshold → warn ── m grace rounds → halt
                                       ▲           passed       │
                                       │                        │
                                       └─── improvement ≥ ──────┘
                                            threshold (cleared)
```

- **ok / insufficient**: no nudge, no halt.
- **warn**: `train.sh` emits a `plateau-detected` event (once per fresh
  warning). `build_prompt.py` injects a "switch to an orthogonal axis"
  block at the top of every subsequent prompt while the warning is active.
- **halt**: M consecutive `training_metrics` events after the warning
  failed to improve by ≥ threshold. `train.sh` writes `HALTED`, emits
  `halted --reason plateau`, and exits 0 (the run itself succeeded — the
  chain stop is the feature).

Defaults: N=3, threshold=0.005, M=2. Override via env vars
`YOLO_TRAINER_PLATEAU_N` / `_THRESHOLD` / `_M`. The query reads them, so
the same overrides apply to `train.sh` and `build_prompt.py` automatically.

State lives entirely in `events.jsonl`. The warning is implicitly cleared
when any post-warn `training_metrics` improves by ≥ threshold above the
pre-warn best — no `plateau-cleared` event needed. `plateau-detected` is
**not** firewalled (it's agent-visible by design — the nudge is the point).

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
