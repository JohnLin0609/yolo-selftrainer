# YOLO Self-Trainer

> An autonomous YOLO training loop driven by an LLM agent. Point it at a
> dataset, walk away, come back to a trained model — with a full audit trail
> of every decision the agent made.

Each round the agent reads the prior run's metrics + per-class diagnostics
from a machine-extracted prompt, writes its hyperparameter choice as
`next_params.json` (a structured contract validated against
`scripts/param_bounds.py`), launches training, and repeats until the round
budget is exhausted. Five trust boundaries keep the loop honest: a hardened
Bash guard (predicate + optional bwrap sandbox), the parameter contract, a
crash circuit breaker, an agent-invisible held-out test split, and a
plateau circuit that nudges the agent off dead-end axes before halting.
Every action is logged to an append-only `events.jsonl` you can audit
afterward.

Works for `detect`, `obb`, `segment`, `pose`, `classify` tasks.

---

## Quick start (5 minutes)

```bash
# 1. Clone
git clone https://github.com/<your-user>/yolo-selftrainer
cd yolo-selftrainer

# 2. One-time setup (creates .venv, installs ultralytics + litellm, downloads pretrained weights)
bash scripts/setup_env.sh

# 3. Optional: grab the demo dataset (COCO128 subset, ~3 MB)
bash scripts/download_demo_dataset.sh

# 4. Scaffold a project from your dataset (auto-detects task, classes, resolution)
bash scripts/new_project.sh --dataset ./datasets/demo --max-rounds 5

# 5. Run the autonomous loop — pick ONE of these:
cd projects/demo
bash start_claude.sh            # default — Claude CLI
# OR (after scaffolding with --mode agent ...)
bash start_agent.sh             # LiteLLM-based, any provider
# OR (after scaffolding with --mode baseline)
bash start_baseline.sh          # LLM-free random-search control loop
```

Five training rounds will run autonomously, with the agent making all
hyperparameter decisions. The final round produces a `training_report.md`
with per-round metrics, per-class weakness flags, best model path, and an
ASCII progression chart.

---

## Supported LLMs

The agent is provider-agnostic via [LiteLLM](https://github.com/BerriAI/litellm).
Pick the one you have access to.

| Provider | Mode | API key | Notes |
|---|---|---|---|
| **Claude CLI** | `--mode claude` (default) | Claude.ai subscription or `ANTHROPIC_API_KEY` | Uses the `claude` binary. |
| **Anthropic API** | `--mode agent --llm-provider anthropic --llm-model claude-opus-4-7` | `ANTHROPIC_API_KEY` | |
| **OpenAI** | `--mode agent --llm-provider openai --llm-model gpt-4o` | `OPENAI_API_KEY` | |
| **Gemini** | `--mode agent --llm-provider gemini --llm-model gemini-2.5-pro` | `GEMINI_API_KEY` | |
| **Groq** | `--mode agent --llm-provider groq --llm-model llama-3.3-70b-versatile` | `GROQ_API_KEY` | |
| **Ollama (local)** | `--mode agent --llm-provider ollama --llm-model qwen3:8b` | — (local) | Small local models struggle with multi-tool ReAct. See [docs/local-llms.md](docs/local-llms.md). |
| **vLLM / self-hosted** | `--mode agent --llm-provider vllm --llm-model <name> --llm-api-base <url>` | — (set via `--llm-api-base`) | OpenAI-compatible endpoint. |

The flags above are passed to `scripts/new_project.sh` at scaffold time; they
get written into the project's `agent.env` so subsequent wakeups reuse them.
For agent mode, see [docs/usage.md](docs/usage.md#api-keys) for where to set
API keys.

### Reliability — measure it, don't guess

Subjective star ratings used to live here. They're replaced by a
reproducible benchmark: same demo dataset, same rounds across providers,
with the numbers backed by each project's `events.jsonl`.

```bash
python3 scripts/benchmark.py \
    --providers anthropic openai gemini ollama \
    --models   claude-haiku-4-5-20251001 gpt-4o-mini gemini-2.5-flash qwen2.5:32b \
    --dataset  datasets/demo \
    --rounds   3 \
    --output   benchmark_report.md
```

Required env vars are checked up-front so a missing key fails fast.
Workspaces land under `projects/bench_<timestamp>_<provider>_<modelslug>/`.
The aggregator (`scripts/benchmark_aggregate.py`) sums LLM cost (written
into `claude_finished` events as `total_cost_usd`), total wall time,
val/test mAP, and circuit-breaker trips per provider. The renderer
(`scripts/benchmark_render.py`) sorts by val mAP DESC.

After running, paste the printed Markdown table between the markers below.

<!-- BENCHMARK TABLE START -->
_No benchmark run committed yet. Run the command above and paste the
output here so future readers see real numbers._
<!-- BENCHMARK TABLE END -->

---

## What you get

Every project (`projects/<name>/`) accumulates:

- **`events.jsonl`** — append-only audit trail. Every round start/finish,
  every training run's metrics, per-class diagnostics, plateau warnings,
  halt reasons, every agent invocation timestamped with cost and duration.
  Operator-only event types (`test_metrics` for the held-out split) are
  firewalled from agent prompts at three layers — see
  [docs/architecture.md](docs/architecture.md).
- **`logs/<session>/<run_name>/`** — per-round dump: the exact prompt sent
  to the agent, the raw stream-json from the LLM, a human-readable session
  log, and a snapshot of `next_instruction.md`.
- **`runs/<task>/<run_name>/`** (in the framework root) — the actual YOLO
  output: `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`,
  confusion matrices, prediction samples.
- **`training_report.md`** at the end — best model summary, per-round metrics
  table (P, R, mAP50, mAP50-95, train/val losses, overfit gap), per-class
  weakness ranking, the agent's `## Data-layer recommendations` (verbatim,
  read-only), ASCII progression chart, and auto-generated insights.

---

## Safety

The framework runs an LLM agent unattended for hours, with shell access via
the `Bash` tool. Five trust boundaries — see
[docs/architecture.md](docs/architecture.md) for the full design.

1. **Bash guard** (`scripts/claude_bash_guard.py`) — two sublayers:
   - **Predicate**: deny-list head tokens (`rm`, `sudo`, `pip`, `mv`, `cp`,
     `git push/reset`, `curl`, `ssh`, `docker`, …) on every subcommand of
     a split compound command. Bypass-pattern rejection on top of that
     (interpreter `-c`, `eval`, `find -delete/-exec`, `awk system()`,
     heredoc-to-interpreter, head-position command substitution). Writes
     to `train.sh` and anything under `datasets/` are rejected — agent
     surfaces data-layer issues as recommendations, never as writes.
   - **Sandbox** (agent mode only): when `bwrap` is installed,
     `scripts/sandbox.py` wraps Bash execution in a user-namespace
     sandbox — root read-only, project dir read-write, `/tmp` is tmpfs,
     no network, own pid namespace.
2. **Parameter contract**: the agent writes a flat JSON dict to
   `next_params.json`. `scripts/apply_params.py` validates it against
   `scripts/param_bounds.py` (single source of truth, task-aware and
   fine-tune-aware) and writes the effective env file. Validation failures
   don't consume a round — the agent wakes again to fix the file.
3. **Crash circuit breaker** (in `train.sh`): 3 consecutive training
   failures (crash, validator abort, or LLM error) write a `HALTED` file
   that blocks all subsequent wakeups until the operator inspects it.
4. **Operator-only data**: `scripts/new_project.sh` carves a held-out test
   split at scaffold time with a locked seed. `scripts/run_test_eval.py`
   runs `yolo val split=test` after each successful round and emits
   `test_metrics` events. These events are agent-INVISIBLE
   (`AGENT_INVISIBLE_EVENT_TYPES` in `build_prompt.py`) so the agent can't
   overfit the unbiased benchmark. **Optional `--strict-heldout` mode**
   upgrades the contract to LeetCode-grade: dual `dataset.yaml` /
   `dataset.eval.yaml` so the agent's view never names the test split;
   Bash-guard patterns reject every direct read of `images/test/` /
   `labels/test/` / `split=test`; the agent can submit a model via
   `scripts/run_test_tool.py` for a one-line aggregate score (rate-limited
   one peek per round), but cannot enumerate or read the data.
5. **Plateau circuit** (additive to #3): when the primary metric stops
   moving over a sliding window, `train.sh` emits a `plateau_detected`
   event. The next prompt carries an orthogonal-strategy nudge ("DO NOT
   tweak LR further; MUST try one of: augmentation regime, model size
   swap, or data review"). After M grace rounds without improvement
   above threshold, the chain halts cleanly. Overrides via
   `YOLO_TRAINER_PLATEAU_N` / `_THRESHOLD` / `_M`.

The autonomous loop runs Claude with `--dangerously-skip-permissions` (or
the equivalent in agent mode); the boundaries above are what restore the
safety the UI prompt normally provides.

---

## CLI reference

### Scaffolding

```
bash scripts/new_project.sh --dataset PATH [options]

  --dataset PATH         Dataset directory (Ultralytics layout); auto-splits
                         if no train/val dirs exist (use --no-auto-split to disable)
  --name NAME            Project subdir name under projects/ (default: dataset basename)
  --task TASK            detect | obb | segment | pose | classify
                         (default: auto-detect from label format)
  --max-rounds N         MAX_ROUNDS for the loop (default: 10)
  --device N             GPU device (default: 0)

  --mode MODE            claude (default) | agent | baseline
                         baseline = LLM-free random-search control loop
                         (`scripts/baseline_policy.py`) — for measuring
                         agent uplift on the same dataset / seed.
  --llm-provider P       For --mode agent: anthropic | openai | gemini |
                         groq | together_ai | ollama | vllm
  --llm-model M          Model id for the chosen provider
  --llm-api-base URL     Custom OpenAI-compatible endpoint (vLLM, self-host)

  --test-split RATIO     Held-out test split fraction (default: 0.15; 0 disables)
  --test-seed SEED       RNG seed locking the test split (default: 42; auto-
                         randomized when --strict-heldout and not pinned)
  --baseline-seed SEED   RNG seed for --mode baseline (default: 42)
  --strict-heldout       LeetCode-grade hidden test split. Agent's dataset.yaml
                         loses its test: key (moved to dataset.eval.yaml). The
                         Bash guard rejects all direct reads of the test split.
                         The sole sanctioned route to the score is
                         `python3 scripts/run_test_tool.py --project <dir>`,
                         which returns one line: mAP50=X.XXXX mAP50-95=X.XXXX
                         images=N. Rate-limited to one peek per round.
  --force                Overwrite an existing project
```

Examples:

```bash
# Default — Claude CLI, 5 rounds, demo dataset
bash scripts/new_project.sh --dataset ./datasets/demo --max-rounds 5
cd projects/demo && bash start_claude.sh

# OpenAI gpt-4o-mini (cheap), 10 rounds
bash scripts/new_project.sh --dataset ./datasets/wheel --max-rounds 10 \
    --mode agent --llm-provider openai --llm-model gpt-4o-mini
cd projects/wheel && bash start_agent.sh

# Local Ollama
bash scripts/new_project.sh --dataset ./datasets/wheel --max-rounds 5 \
    --mode agent --llm-provider ollama --llm-model qwen3:8b
cd projects/wheel && bash start_agent.sh

# LLM-free baseline — same dataset, locked seed, for measuring agent uplift
bash scripts/new_project.sh --dataset ./datasets/wheel --max-rounds 10 \
    --mode baseline --baseline-seed 42
cd projects/wheel && bash start_baseline.sh
```

### Multi-provider sweep

`python3 scripts/benchmark.py …` (see "Reliability" above).

### Smoke test

`python3 scripts/test_agent_smoke.py [--providers ...]` — quick provider
connectivity + tool-calling check, no training.

---

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — full loop wiring
  (preflight → agent → train → events), the 5 trust boundaries in detail,
  the prompt-building pipeline, Harness Engineering principles applied.
- **[docs/usage.md](docs/usage.md)** — dataset layout, API key setup,
  monitoring a running session, recovering from `HALTED`.
- **[docs/extending.md](docs/extending.md)** — adding a new LLM provider,
  writing custom action playbooks, hooking into events.jsonl.
- **[docs/local-llms.md](docs/local-llms.md)** — why local Ollama is
  marginal for multi-tool ReAct, what models survive, tuning tips.

---

## Requirements

- **OS**: Linux (tested on Ubuntu 24.04). macOS likely works; Windows via WSL.
- **Python**: 3.10+ (3.12 recommended).
- **GPU**: NVIDIA with CUDA 12+ recommended for IMGSZ ≥ 640. CPU works for
  tiny datasets but is very slow.
- **Disk**: ~5 GB free per long training session (checkpoints + logs).
- **`bubblewrap`** (optional, agent mode only): enables OS-level sandboxing
  of the agent's Bash tool. `apt install bubblewrap`. Without it, the
  predicate guard alone still protects; a stderr warning prints at agent
  startup.
- **Claude Code CLI** if using `--mode claude`, OR an API key for one of
  the providers in agent mode, OR nothing extra for `--mode baseline`.

---

## Acknowledgements

- Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- Agent-mode multi-LLM support powered by [LiteLLM](https://github.com/BerriAI/litellm).
- Default LLM driver is [Claude Code](https://claude.com/claude-code).
- Framework structure follows the [Harness Engineering](https://github.com/anthropics/easy-agent)
  reference patterns (event sourcing, trust boundaries, fail-loud, three-tier
  compaction).

---

## License

[MIT](LICENSE).
