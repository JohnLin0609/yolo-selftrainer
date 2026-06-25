# YOLO Self-Trainer

> An autonomous YOLO training loop driven by an LLM agent. Point it at a dataset,
> walk away, come back to a trained model — with a full audit trail of every
> decision the agent made.

The agent reads the previous round's `results.csv`, decides what hyperparameter
to change (lr / epochs / aug / model / fine-tune from best.pt / ...), edits
`train.sh` with `sed`, launches training, and repeats until your round budget
is exhausted. A circuit breaker stops the loop after 3 consecutive failures,
preflight checks block bad runs before they start, and every action is logged
to an append-only `events.jsonl` you can audit afterward.

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

# 4. Run autonomous training (default: Claude CLI mode, 5 rounds)
bash start_self_training.sh --dataset ./datasets/demo --rounds 5
```

That's it. Five training rounds will run autonomously, with the agent making
all hyperparameter decisions. The final round produces a `training_report.md`
with per-round metrics, best model path, and an ASCII bar chart of progression.

---

## Supported LLMs

The agent is provider-agnostic via [LiteLLM](https://github.com/BerriAI/litellm).
Pick the one you have access to.

| Provider | Mode | API key | Notes |
|---|---|---|---|
| **Claude CLI** | `--mode claude` (default) | Claude.ai subscription or `ANTHROPIC_API_KEY` | Uses the `claude` binary. |
| **Anthropic API** | `--mode agent --provider anthropic --model claude-opus-4-7` | `ANTHROPIC_API_KEY` | |
| **OpenAI** | `--mode agent --provider openai --model gpt-4o` | `OPENAI_API_KEY` | |
| **Gemini** | `--mode agent --provider gemini --model gemini-2.5-pro` | `GEMINI_API_KEY` | |
| **Groq** | `--mode agent --provider groq --model llama-3.3-70b-versatile` | `GROQ_API_KEY` | |
| **Ollama (local)** | `--mode agent --provider ollama --model qwen3:8b` | — (local) | Small local models struggle with multi-tool ReAct. See [docs/local-llms.md](docs/local-llms.md). |
| **vLLM / self-hosted** | `--mode agent --provider vllm --model <name> --api-base <url>` | — (set via `--api-base`) | OpenAI-compatible endpoint. |

For agent mode, see [docs/usage.md](docs/usage.md#api-keys) for where to set
API keys.

### Reliability — measure it, don't guess

Subjective star ratings used to live here. They're replaced by a
reproducible benchmark: same demo dataset, same rounds, same seed across
providers, with the numbers backed by each project's `events.jsonl`.

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
The aggregator (`scripts/benchmark_aggregate.py`) sums LLM cost (now
written into `claude_finished` events as `total_cost_usd`), total wall
time, val/test mAP, and circuit-breaker trips per provider. The renderer
(`scripts/benchmark_render.py`) sorts by val mAP DESC.

After running, paste the printed Markdown table between the markers below.

<!-- BENCHMARK TABLE START -->
_No benchmark run committed yet. Run the command above and paste the
output here so future readers see real numbers._
<!-- BENCHMARK TABLE END -->

---

## What you get

Every project (`projects/<name>/`) accumulates:

- **`events.jsonl`** — append-only audit trail. Every round start/finish, every
  training run's metrics, every halt reason, every Claude/agent invocation
  timestamped with cost and duration.
- **`logs/<session>/<run_name>/`** — per-round dump: the exact prompt sent to
  the agent, the raw stream-json from the LLM, a human-readable session log,
  and a snapshot of `next_instruction.md`.
- **`runs/<task>/<run_name>/`** (in the framework root) — the actual YOLO
  output: `weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`,
  confusion matrices, prediction samples.
- **`training_report.md`** at the end — best model summary, per-round metrics
  table (P, R, mAP50, mAP50-95, train/val losses, overfit gap), ASCII
  progression chart, and auto-generated insights.

---

## Safety

The framework runs an LLM agent unattended for hours, possibly with shell
access via the `Bash` tool. Three layers of defense:

1. **PreToolUse guard** (`scripts/claude_bash_guard.py`): every Bash command
   passes through a deny-list filter. Compound commands (`a && b`) are split
   per-subcommand so `echo hi && rm -rf /` is caught at the `rm`. Default deny:
   `rm`, `sudo`, `chmod`, `chown`, `mv`, `cp`, `pip`, `git push/reset/clean`,
   `curl`, `wget`, `ssh`, `docker`.
2. **Parameter validator** (top of `train.sh`): hyperparameters are
   range-checked against bounds in `hyperparameter_strategy.md` before YOLO
   launches. Out-of-range params abort the run *and don't consume a round*.
3. **Circuit breaker**: 3 consecutive training failures (crash, validation
   abort, or LLM error) write a `HALTED` file that blocks all subsequent
   wakeups until the operator inspects and clears it.

See [docs/architecture.md](docs/architecture.md) for the full design.

---

## CLI reference

```bash
bash start_self_training.sh --dataset PATH [options]

  --dataset PATH         Dataset directory (Ultralytics layout)
  --rounds N             Number of training rounds (default: 10)
                         Round N is summary-only; effective training rounds = N-1

  --mode {claude,agent}  Agent loop implementation (default: claude)
  --provider PROVIDER    For agent mode: anthropic | openai | gemini |
                         groq | together_ai | ollama | vllm
  --model MODEL          Model id for the chosen provider
  --api-base URL         Custom OpenAI-compatible endpoint (vLLM, self-host)

Examples:
  # Claude CLI, 10 rounds
  bash start_self_training.sh --dataset ./datasets/wheel --rounds 10

  # OpenAI gpt-4o-mini (cheap)
  bash start_self_training.sh --dataset ./datasets/wheel --rounds 5 \
      --mode agent --provider openai --model gpt-4o-mini

  # Local Ollama
  bash start_self_training.sh --dataset ./datasets/wheel --rounds 5 \
      --mode agent --provider ollama --model qwen3:8b
```

---

## Documentation

- **[docs/architecture.md](docs/architecture.md)** — how the loop is wired
  (preflight → agent → train → events), the Harness Engineering principles
  applied, the prompt-building pipeline.
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
- **Claude Code CLI** if using `--mode claude`, OR an API key for one of the
  providers in agent mode.

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
