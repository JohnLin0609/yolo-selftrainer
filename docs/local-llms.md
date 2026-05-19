# Local LLMs: why they're marginal for this framework

The autonomous training loop runs a non-trivial multi-tool ReAct task:
roughly 5–15 tool calls per round, with each tool result feeding the next
decision. This is much harder than single-turn chat or code completion,
and small open-source LLMs (≤14B parameters) struggle on it.

This document records what was tested, what failed, and what works — so
you can pick the right local model without going through the same
trial-and-error.

---

## What was tested (8 GB VRAM laptop GPU)

| Model | Tool calling reliability | Failure mode |
|---|---|---|
| `gemma3n:e2b` (5.1B) | ★ | Infinite empty tool-call loop. Hit max_turns. |
| `gemma3n:e4b` (8B) | ★ | Sometimes finishes 1-2 calls, then emits tool call as JSON-in-text. |
| `qwen2.5-coder:7b-instruct` | ★ | Same JSON-in-text pattern after first call. |
| `llama3.1:8b` | ★ | Empty tool-call loops. |
| `qwen3:8b` | ★★ | 3 clean turns then degrades. Best of the small models. |
| `qwen3:14b` | ★ | Surprisingly worse than 8B — emits `{}` as text and stops. |

**None reliably completed a full training round** (which needs ~6+ tool
calls).

The pattern: **all small local models with `tools` capability declare can
do the FIRST tool call correctly, then drift into emitting subsequent
"tool calls" as inline JSON text instead of using the structured
tool_calls API.** LiteLLM correctly identifies this as "no tool calls" and
ends the turn, breaking the chain.

---

## Root cause

Verified via raw Ollama API + LiteLLM tracing:

1. **Long system prompts amplify the problem.** With 26 KB of system
   prompt (the full `yolo_folder_skill.md` + `hyperparameter_strategy.md`),
   small models lose track of the tool-calling format after 1–3 turns.
2. **Code blocks in prompts trigger it.** When the user prompt contains
   example shell commands ("`grep -E '^EPOCHS=' train.sh`"), the model
   sometimes "narrates" the next call by emitting JSON-looking text
   instead of using the API.
3. **Thinking mode worsens it.** Qwen3 / DeepSeek-distill models with
   `think=True` are much worse than with `think=False`. The framework
   sets `think=False` for Ollama via `extra_body`.

These are model-capability limits, not framework bugs. Larger models
(70B+) handle this better; cloud-hosted models (Claude, GPT-4o, Gemini)
handle it perfectly.

---

## What to try if you really want local

### Option 1 — qwen3:8b is the least bad ≤8B option

Smoke test passes; first 3 turns of a real session usually work; circuit
breaker catches the rest. Expect 30%–50% of rounds to fail with empty
tool-call loops; the circuit breaker will eventually HALT.

```bash
ollama pull qwen3:8b
bash start_self_training.sh --dataset ./datasets/my_data --rounds 5 \
    --mode agent --provider ollama --model qwen3:8b
```

Tips:
- Pre-warm the model: `ollama run qwen3:8b "ok"` before launch, so the
  first round's cold-load doesn't stack with YOLO's GPU memory request.
- Set `OLLAMA_KEEP_ALIVE=30s` (curl `/api/generate` with
  `"keep_alive":"30s"`) so the model unloads quickly between agent calls,
  freeing VRAM for YOLO.
- Be ready to manually unstick HALTED — see `docs/usage.md`.

### Option 2 — try a 30B+ MoE model

Qwen3-30B-A3B (MoE: 30B total params, 3B active per token) gives
near-30B-class reasoning at near-3B speed. Needs ~17 GB to load (won't
fit in 8 GB VRAM, but spills to RAM acceptably with the MoE pattern).

```bash
ollama pull qwen3:30b-a3b
```

Untested in this framework but theoretically the best local fit.

### Option 3 — self-host vLLM with Qwen3-32B or larger

```bash
# Separate machine with 24+ GB VRAM
vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes

# Then on your dev machine
bash start_self_training.sh --dataset ./datasets/my_data --rounds 5 \
    --mode agent --provider vllm --model Qwen3-32B \
    --api-base http://<vllm-server>:8000/v1
```

vLLM's tool-call parsing is more reliable than Ollama's.

---

## When to just use a paid API

For this specific task (autonomous training loop with audit trail), the
math usually favors paid:

| Provider | $ per round | $ per 10-round session | Reliability |
|---|---|---|---|
| Claude Opus | ~$1–5 | ~$10–50 | ★★★★★ |
| Claude Haiku | ~$0.05–0.20 | ~$0.50–2 | ★★★★★ |
| OpenAI gpt-4o | ~$0.30–1 | ~$3–10 | ★★★★★ |
| OpenAI gpt-4o-mini | ~$0.02–0.10 | ~$0.20–1 | ★★★★★ |
| Gemini 2.5 Flash | ~$0.02–0.05 | ~$0.20–0.50 | ★★★★ |
| Local Qwen3:8b | $0 | $0 | ★★ |
| Local Llama 3.1:8b | $0 | $0 | ★ |

A failed round on local LLMs wastes 10+ GPU-minutes of YOLO training.
That cost — both compute and your debugging time — usually exceeds the
$0.20 you'd spend on gpt-4o-mini doing it right.

**Recommendation**: develop with local Qwen3:8b for free iteration, but
when running real training sessions you care about, use a paid provider
or Claude CLI.
