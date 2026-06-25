# Architecture

How the YOLO Self-Trainer is wired, and why each piece is where it is.

The design follows the **Harness Engineering** patterns: event sourcing as
source of truth, real trust boundaries instead of advisory rules, fail-loud
over silent fallback, and budget-aware context engineering.

---

## The loop

```
┌────────────────────┐    1. parse CLI flags, auto-detect task/classes/imgsz
│ new_project.sh     │ ──→ 2. carve held-out test split (locked seed)
└──────────┬─────────┘    3. scaffold projects/<name>/ from templates/
           │                 (start_claude.sh | start_agent.sh | start_baseline.sh
           │                  depending on --mode)
           ▼
┌────────────────────┐
│  start_*.sh        │     - HALTED check → exit if present
│  (per round)       │     - per-round preflight
│                    │     - bump round counter (round_started event)
│                    │     - claude/agent: build prompt + invoke LLM
│                    │     - baseline: invoke baseline_policy.py
└──────────┬─────────┘
           │ claude/agent: agent writes next_params.json (the structured
           │   param contract — apply_params.py validates against
           │   param_bounds.py and writes effective_params.env)
           │ baseline: baseline_policy.py prints KEY=VALUE pairs that
           │   the orchestrator sed-edits into train.sh
           │ in all modes: launches `nohup bash train.sh > current.log 2>&1 &`
           ▼
┌────────────────────┐
│  train.sh          │     - apply_params.py + defense-in-depth param validator
│                    │     - yolo {task} train ... (foreground)
│                    │     - emit training_finished + training_metrics events
│                    │     - per_class_metrics.py: re-run val(), emit per-class
│                    │     - run_test_eval.py: held-out test split (agent-invisible)
│                    │     - plateau circuit: q_plateau_status → warn / halt
│                    │     - on crash: circuit breaker (3 strikes → HALTED)
│                    │     - rm train.pid, then recurse: bash start_*.sh
└────────────────────┘
```

Each step that mutates state emits an event. After N rounds, the chain
naturally stops (`MAX_ROUNDS` check at the top of `start_*.sh`). The
plateau circuit can stop earlier if the primary metric stops moving for
N+M consecutive rounds.

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

The framework runs an LLM unattended. Three boundaries protect it:

- **Boundary 1 (Bash guard)** has TWO layers — a hardened predicate in
  `scripts/claude_bash_guard.py` (denylist + bypass-pattern rejection +
  train.sh-write regex) that runs in both claude CLI mode and run_agent.py
  mode, plus an OS-level `bwrap` sandbox in `scripts/sandbox.py` that
  wraps Bash execution **only in run_agent.py mode**. In claude CLI mode
  the sandbox does not apply (claude controls its own Bash dispatch);
  the predicate is the sole protection there.
- **Boundaries 2 and 3** are deterministic gates — they hard-fail any
  input outside their contract.

The canonical contract for both Boundary 1 layers lives in
`tests/features/bypass_attempts.feature` (predicate) and
`tests/features/sandbox_isolation.feature` (sandbox runtime).

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

The same shape of rule applies to **`datasets/`** (`_DATASETS_WRITE_PATTERNS`):
the dataset is operator-curated, and the per-class diagnostics feature
(Boundary 4 below) deliberately routes data-layer findings into
recommendations the agent *writes about*, never into actions the agent
*takes*. Read-only inspection (`cat`, `find -print`, `grep`, `sed -n`,
`awk` without `-i inplace`) stays allowed — the agent still needs to
inspect dataset YAMLs to reason about class names and sample counts.

#### Bypass-pattern rejection (the predicate's second layer)

The original head-token denylist could be bypassed by wrapping the
dangerous command in an interpreter, a find primitive, or command
substitution. `_BYPASS_PATTERNS` in `claude_bash_guard.py` now rejects
all of those at the same trust boundary:

```python
_BYPASS_PATTERNS = [
    (re.compile(r"\b(?:/[^/\s]+/)*(?:bash|sh|dash|zsh|python3?|node|ruby|perl)\b[^|;&]*\s(?:-[a-zA-Z]*c|--command)\b"),
     "interpreter -c: arbitrary code execution"),
    (re.compile(r"(?<![\w/.-])eval\b"), "eval: arbitrary code execution"),
    (re.compile(r"\bfind\b[^|;&]*\s-delete\b"),
     "find -delete: erases without invoking rm"),
    (re.compile(r"\bfind\b[^|;&]*\s-(?:exec|execdir|ok|okdir)\b"),
     "find -exec/-execdir/-ok: arbitrary command spawn"),
    (re.compile(r"\bawk\b[^|;&]*['\"][^'\"]*\bsystem\s*\("),
     "awk system(): arbitrary command spawn"),
    # … perl -e, xargs sh, heredoc-to-interpreter, $(...) at head, backtick at head
]
```

The check runs between the train.sh-write regex and the head-token deny
check; it's a third filter, not a replacement for the other two. Tests
locking the contract: `tests/unit/test_claude_bash_guard.py::test_bypass_attempts_blocked` (19 cases) and `tests/features/bypass_attempts.feature` (9 scenarios). All pass on `main` as of the hardening PR.

One deliberate exception: **mid-args command substitution is allowed**
(`kill -0 $(cat train.pid)` is a legitimate idiom the agent uses every
round). Only **head-position** substitution (the command itself comes
from a substitution) is rejected. Tests cover both directions.

#### Sandbox runtime (`scripts/sandbox.py`, run_agent.py mode only)

`scripts/sandbox.py` wraps Bash execution in `bubblewrap` (`bwrap`):

- Root mounted **read-only**; `{framework_root}` mounted read-only.
- `{project_dir}` is the ONLY writable real path.
- `/tmp` is a tmpfs — escape attempts (`echo poison > /tmp/marker`) do
  not touch the host.
- `--unshare-net` — no network from in-sandbox commands.
- `--unshare-pid` — own pid namespace (can't kill host processes).
- `--die-with-parent` — sandbox dies if the python parent dies.

`scripts/run_agent.py`'s `run_bash` routes through
`sandbox.run_in_sandbox(...)` when `sandbox.is_available()` returns True
(bwrap on PATH). When unavailable, it logs to stderr and falls back to
host execution — the predicate alone still protects, but operators
get a loud warning at agent startup.

**Asymmetry**: in claude CLI mode the sandbox is NOT wired in — claude
runs Bash itself; the PreToolUse hook can only validate, not relocate
execution. Hardening claude CLI mode further would require launching
claude itself inside bwrap; that's a separate piece of work.

Contract tests for the sandbox: `tests/features/sandbox_isolation.feature`
covers cannot-delete-sibling, cannot-write-outside, cannot-read-sibling,
writes-within-project-succeed, framework-read-only, network-denied.
All scenarios pass on bwrap-capable hosts; they skip gracefully when
bwrap is absent.

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

#### Per-class diagnostics (agent-visible, dataset-read-only)

Per-class P/R/mAP and the confusion matrix are surfaced to the agent —
they're *necessary* for the agent to reason about which class is
dragging down the metric. The asymmetry vs the held-out test split:
per-class data IS visible (so the agent can act on it), but the action
is restricted to writing a **read-only recommendation**, never to
mutating the dataset.

How it's enforced:

1. `scripts/per_class_metrics.py` runs `model.val()` on `best.pt` after
   each successful training round, emits a `per-class-metrics` event
   carrying per-class metrics + the confusion matrix.

2. `scripts/diagnose_classes.py` is a pure function that ranks weak
   classes (ascending by mAP50, tie-broken by support then alphabetic)
   and flags `persistent` when the same class has been worst for N
   consecutive rounds (default N=3, override `YOLO_TRAINER_PERSIST_N`).

3. `scripts/build_prompt.py:build_per_class_section` surfaces the
   ranking + top-3 confused pairs in the agent's prompt. When the
   `persistent` flag fires it adds a "⚠️ Persistent weakness detected"
   callout instructing the agent to write `## Data-layer recommendations`
   in `next_instruction.md` AND reminding it that the Bash guard rejects
   writes under `datasets/`.

4. Boundary 1's `_DATASETS_WRITE_PATTERNS` is the wall behind that
   reminder: sed -i / awk -i inplace / perl -i / > redirect / tee
   targeting `datasets/` are all rejected. Reads stay allowed so the
   agent can still cite specific sample counts and labels in its
   recommendation.

5. `scripts/generate_report.py:render_per_class_weaknesses` reproduces
   both the machine ranking and the agent's `## Data-layer
   recommendations` block (verbatim, read-only) in the final report —
   the human sees the data-layer signal AND the agent's interpretation
   side-by-side.

No firewall changes here: per-class events are agent-VISIBLE by design,
unlike `test_metrics`. The trust boundary is the *write* restriction
on `datasets/`, not the *read* visibility of the diagnostic.

#### Strict-heldout (`--strict-heldout`) — LeetCode mode

The standard Boundary 4 is **prompt-level**: it guarantees the agent's
prompt doesn't carry test metrics. An agent that knows the ultralytics
API can still `cat datasets/<name>/labels/test/*.txt` or
`yolo val split=test ...` itself — `yolo` isn't on `DENY_HEADS`. The
firewall is honest about this with its docstring: "agent never sees
test data **via prompts**".

`--strict-heldout` upgrades the contract to "agent process cannot reach
the test data at all, but can submit its current model and get back a
single aggregate score" — the LeetCode contract. Activated by
`scripts/new_project.sh --strict-heldout`, layered as four pieces:

1. **Dual-yaml**. The agent-visible `datasets/<name>/dataset.yaml` loses
   its `test:` key; an operator-only sibling `dataset.eval.yaml` carries
   it. `run_test_eval.py` and the new `run_test_tool.py` read the eval
   yaml; `train.sh` reads the agent yaml.
2. **Marker file + Bash-guard patterns**. `projects/<name>/.heldout_strict`
   is a sentinel `claude_bash_guard.py` walks up looking for from cwd.
   When present, `_HELDOUT_READ_PATTERNS` reject every direct exfil
   route: any path under `datasets/.../(images|labels)/test/`, any
   `yolo val split=test` (string-literal or quoted), any `split="test"`
   Python keyword smuggling. The sole sanctioned exit is
   `python3 scripts/run_test_tool.py …`, whitelisted in the same
   `check_command` pass.
3. **`.claude/settings.json` permissions** (Claude-CLI mode only).
   `new_project.sh --strict-heldout` injects a `permissions.deny`
   block listing the test paths + `split=test` patterns and a
   `permissions.allow` entry whitelisting the run_test_tool.py
   invocation. `run_agent.py` (LiteLLM) doesn't read this file but is
   covered by #2.
4. **The LeetCode submit tool** (`scripts/run_test_tool.py`).
   Score-only stdout (`mAP50=X.XXXX mAP50-95=X.XXXX images=N`); no
   per-class, no class names, no file paths. Rate-limited to one peek
   per round via an idempotency check on `test_tool_query` events at
   the current round. Every invocation emits a `test_tool_query` event
   that is itself agent-INVISIBLE (added to `AGENT_INVISIBLE_EVENT_TYPES`)
   so the score never re-enters a subsequent prompt.

The randomized test seed (`--strict-heldout` rolls one fresh per scaffold
unless `--test-seed` is pinned) plus the `heldout-cut` event (operator-
audit-only) close the loop: each project sees a fresh randomly-sampled
test split that the agent cannot enumerate, cannot read, and can only
score against once per round.

Out of scope for this layer: LiteLLM-mode equivalent of the submit tool
(currently Claude-only); rotating the test split across rounds within
one project (would break round-to-round comparability).

To add a new operator-only event type, list it in
`AGENT_INVISIBLE_EVENT_TYPES` AND add distinctive guard substrings to
`_PROMPT_GUARD_TERMS` in the same file. Both must be updated together.

### Cross-provider benchmark (out-of-band, observability tooling)

Not a trust boundary — a measurement loop. `scripts/benchmark.py` sweeps
the same training pipeline against multiple LLM providers on the same
demo dataset and emits a Markdown comparison table. The data comes from
each provider's own `projects/<name>/events.jsonl`, aggregated by the
pure functions `scripts/benchmark_aggregate.py` and rendered by
`scripts/benchmark_render.py`. The README's old subjective star ratings
were replaced with this script's output.

For the cost column to be populated, `run_agent.py` now emits
`total_cost_usd` on each `claude_finished` event (the per-turn cost from
`litellm.completion_cost` is accumulated across the session). Older
events.jsonl files lack the field; the aggregator treats missing as 0.0
so old projects still aggregate cleanly.

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

`scripts/new_project.sh` (one-time scaffold) and `scripts/preflight.sh`
(invoked from every `start_*.sh` wakeup) jointly follow the Harness §11
discipline:

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
├── scripts/
│   ├── setup_env.sh              # .venv + ultralytics + litellm + models
│   ├── download_models.sh        # pretrained yolo11n.pt, yolov8n.pt, etc.
│   ├── download_demo_dataset.sh  # COCO128 subset for quickstart
│   ├── new_project.sh            # THE FRONT DOOR — scaffold projects/<name>/
│   ├── preflight.sh              # --cold (full check) / --quick (per-round)
│   ├── event.py                  # events.jsonl emit + query CLI
│   ├── build_prompt.py           # per-round prompt with facts+metrics+prose
│   ├── param_bounds.py           # bounds source-of-truth for next_params.json
│   ├── apply_params.py           # validates next_params.json → effective_params.env
│   ├── baseline_policy.py        # LLM-free random-search policy (--mode baseline)
│   ├── per_class_metrics.py      # model.val() → per_class_metrics event
│   ├── diagnose_classes.py       # pure rank-and-flag for per-class weakness
│   ├── run_test_eval.py          # held-out test split (agent-invisible)
│   ├── run_agent.py              # LiteLLM ReAct loop (agent mode)
│   ├── claude_bash_guard.py      # Bash deny-list + bypass patterns + datasets/ guard
│   ├── sandbox.py                # bwrap wrapper for agent-mode Bash (optional)
│   ├── benchmark.py              # cross-provider sweep orchestrator
│   ├── benchmark_aggregate.py    # pure events.jsonl → ProviderRow
│   ├── benchmark_render.py       # pure ProviderRow[] → Markdown table
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
