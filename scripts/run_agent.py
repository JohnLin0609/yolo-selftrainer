#!/usr/bin/env python3
"""Multi-LLM agent loop for the YOLO self-trainer (Harness §1+§2 portable layer).

Why this exists:
  start_claude.sh shells out to the `claude` CLI. That gave us free tool calling,
  prompt cache, streaming, and a PreToolUse hook system — but pinned the project
  to one provider. P6 reimplements the agentic loop in ~300 lines so any
  LiteLLM-supported model (Anthropic, OpenAI, Gemini, Ollama, Groq, vLLM, ...)
  can drive the training cycle.

Why LiteLLM and not the raw SDKs:
  Each provider's tool calling has slightly different JSON shapes (Anthropic's
  `tool_use` blocks vs OpenAI `function_calls`, etc.). LiteLLM speaks OpenAI
  on the wire and normalizes inbound. Single dependency for 100+ providers.

Why we stay non-streaming for v1:
  Streaming demands per-block-index buffers and thinking-block preservation
  (Harness §1.1 + §1.2). Doing that across 5 providers is its own project. The
  loop is short and infrequent (one Claude session per training round); buffering
  the full message is fine for now.

Logging format:
  We emit lines to stream-json-out in the same shape as `claude --output-format
  stream-json`. That lets start_agent.sh reuse the existing log extractor in
  start_claude.sh and produce identical human-readable session logs. The exact
  shape is documented next to msg_to_log_assistant().

Trust boundary (Harness §5.1):
  Bash commands go through scripts/claude_bash_guard.py via subprocess (--guard-
  script). The guard is unaware of the agent loop — it's just a stdin/exit-code
  contract. That keeps the security layer exactly the same as in CLI mode.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


# ─── Constants ──────────────────────────────────────────────────────

TOOL_BASH = "Bash"
TOOL_WRITE = "Write"
TOOL_READ = "Read"

DEFAULT_MAX_TURNS = 50               # Harness §2.5 hard limit
DEFAULT_TOOL_RESULT_MAX = 30_000     # chars — same order of magnitude as Claude CLI
DEFAULT_BASH_TIMEOUT = 600           # seconds per single shell command
DEFAULT_LLM_TIMEOUT = 900            # seconds per LLM call


# Tool catalog — OpenAI function-calling shape, which LiteLLM speaks on the
# wire and translates per provider. Descriptions are deliberately concrete so
# weaker models (Llama-3.1-class) understand when to call each.
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_BASH,
            "description": (
                "Execute a shell command in the project directory. "
                "Use for file inspection (ls, cat, grep, head, tail, awk), "
                "training launches (nohup bash train.sh > current.log 2>&1 &), "
                "process checks (kill -0 PID). "
                "Hyperparameters are set via next_params.json (use the Write "
                "tool) — sed-editing train.sh is rejected by the safety guard. "
                "Dangerous commands (rm, pip install, sudo, mv, git push, ...) are "
                "blocked by the same guard regardless of the permission model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "description": {"type": "string", "description": "Brief reason for this command."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_WRITE,
            "description": (
                "Create or overwrite a file. Primary uses: writing "
                "next_params.json (the hyperparameter contract for the next "
                "round) and writing next_instruction.md with the structured "
                "sections described in the system prompt before exiting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or project-relative path."},
                    "content":   {"type": "string", "description": "Full file contents."},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_READ,
            "description": "Read a file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute or project-relative path."},
                },
                "required": ["file_path"],
            },
        },
    },
]


# Per-provider nudges appended to the system prompt to improve tool-call
# reliability on weaker models. Empty for the strong providers — adding
# extraneous reminders to Opus/GPT-4o just wastes context.
PROVIDER_TOOL_NUDGE: dict[str, str] = {
    "anthropic": "",
    "openai":    "",
    "gemini": (
        "\n\nIMPORTANT: You have tools (Bash, Write, Read). You MUST use them "
        "to inspect files, write next_params.json (the hyperparameter contract), "
        "and write next_instruction.md. Do NOT just describe what you would do "
        "— actually call the tools. Never edit train.sh."
    ),
    "ollama": (
        "\n\nIMPORTANT: You have three tools — Bash, Write, Read. Every "
        "filesystem or shell operation MUST be a tool call, not prose. After "
        "receiving each tool result, either issue more tool calls or finish "
        "with a final text response."
    ),
    "groq": (
        "\n\nIMPORTANT: You have tools (Bash, Write, Read). You MUST use them; "
        "don't just describe planned actions."
    ),
    "together_ai": (
        "\n\nIMPORTANT: You have tools (Bash, Write, Read). You MUST use them; "
        "don't just describe planned actions."
    ),
}


# ─── Event log integration ──────────────────────────────────────────

def emit_event_subprocess(project: Path, event_type: str, **kwargs) -> None:
    """Append an event via scripts/event.py.

    We shell out instead of importing event.py to keep the dependency one-way
    (run_agent → event.py) and avoid coupling their lifecycles. Event emission
    is rare (once per round), so subprocess overhead is irrelevant.
    """
    event_py = Path(__file__).parent / "event.py"
    if not event_py.exists():
        print(f"[run_agent] WARN: event.py not found at {event_py}", file=sys.stderr)
        return
    cmd = ["python3", str(event_py), str(project), "emit", event_type]
    for k, v in kwargs.items():
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    try:
        subprocess.run(cmd, check=False, timeout=30, capture_output=True)
    except Exception as e:
        print(f"[run_agent] WARN: event emit failed ({event_type}): {e}", file=sys.stderr)


# ─── Tool execution ─────────────────────────────────────────────────

def truncate_for_tool_result(text: str, max_chars: int) -> str:
    """Same head+tail truncation pattern as Harness §7.1.

    Surfacing that truncation happened lets the model decide whether to re-run
    with grep/head/tail to get specific parts of the missing middle. Silent
    truncation would let it believe the file was small.
    """
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return (
        f"{head}\n\n"
        f"[... output truncated: {len(text) - max_chars} chars omitted from middle; "
        f"original length {len(text)} chars ...]\n\n"
        f"{tail}"
    )


def guard_bash_command(guard_script: Path | None, command: str) -> tuple[bool, str]:
    """Validate a shell command via the standalone guard subprocess."""
    if not guard_script:
        return True, ""
    payload = json.dumps({"tool_input": {"command": command}})
    try:
        proc = subprocess.run(
            ["python3", str(guard_script)],
            input=payload,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "guard timed out (>10s) — denying conservatively"
    except Exception as e:
        return False, f"guard subprocess failed: {e}"
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()


def run_bash(command: str, *, guard_script: Path | None, timeout: int, max_chars: int) -> str:
    ok, reason = guard_bash_command(guard_script, command)
    if not ok:
        return (
            "BLOCKED by safety guard:\n"
            f"{reason}\n\n"
            "If this command is genuinely required, a human must edit "
            "scripts/claude_bash_guard.py to allow it. Otherwise plan a different approach."
        )
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s\nCommand: {command!r}"
    except Exception as e:
        return f"ERROR: command failed to launch: {e}\nCommand: {command!r}"

    body = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        body = f"[exit {proc.returncode}]\n{body}"
    return truncate_for_tool_result(body, max_chars)


def run_write(file_path: str, content: str) -> str:
    try:
        p = Path(file_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"ERROR writing {file_path!r}: {e}"


def run_read(file_path: str, *, max_chars: int) -> str:
    try:
        p = Path(file_path).expanduser()
        if not p.exists():
            return f"ERROR: file does not exist: {p}"
        if p.is_dir():
            return f"ERROR: {p} is a directory, not a file"
        return truncate_for_tool_result(p.read_text(errors="replace"), max_chars)
    except Exception as e:
        return f"ERROR reading {file_path!r}: {e}"


def dispatch_tool(
    tool_name: str,
    tool_args: dict,
    *,
    guard_script: Path | None,
    bash_timeout: int,
    max_chars: int,
) -> str:
    if tool_name == TOOL_BASH:
        cmd = tool_args.get("command")
        if not isinstance(cmd, str) or not cmd.strip():
            return "ERROR: Bash requires non-empty 'command' argument"
        return run_bash(cmd, guard_script=guard_script, timeout=bash_timeout, max_chars=max_chars)
    if tool_name == TOOL_WRITE:
        fp = tool_args.get("file_path")
        ct = tool_args.get("content")
        if not isinstance(fp, str):
            return "ERROR: Write requires 'file_path' argument"
        if ct is None:
            return "ERROR: Write requires 'content' argument"
        if not isinstance(ct, str):
            ct = json.dumps(ct, indent=2)
        return run_write(fp, ct)
    if tool_name == TOOL_READ:
        fp = tool_args.get("file_path")
        if not isinstance(fp, str):
            return "ERROR: Read requires 'file_path' argument"
        return run_read(fp, max_chars=max_chars)
    return f"ERROR: unknown tool {tool_name!r}"


# ─── Stream-json log emission ───────────────────────────────────────

class StreamJSONWriter:
    """Append-only writer in the shape of `claude --output-format stream-json`.

    The existing log extractor in start_claude.sh.tmpl parses this format to
    produce the human-readable session log. Mirroring it means agent-mode
    sessions get identical log rendering for free.
    """
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w")

    def write(self, obj: dict) -> None:
        self._f.write(json.dumps(obj, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def msg_to_log_assistant(msg) -> dict:
    """Convert a LiteLLM response message into the stream-json 'assistant' shape."""
    blocks: list[dict] = []
    if getattr(msg, "content", None):
        blocks.append({"type": "text", "text": msg.content})
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            # Some models (looking at you, smaller Llamas) emit raw text instead
            # of valid JSON. Wrap so the extractor still has something to show.
            args = {"_raw_arguments": tc.function.arguments}
        blocks.append({
            "type": "tool_use",
            "id":   tc.id,
            "name": tc.function.name,
            "input": args,
        })
    return {"type": "assistant", "message": {"content": blocks}}


def msg_to_history_dict(msg) -> dict:
    """Convert LiteLLM response message into a dict suitable for re-injection.

    Some providers (Anthropic) require non-empty content even when tool_calls
    are present; we fall back to an empty string in that case.
    """
    d: dict = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
        d.setdefault("content", "")
    return d


# ─── Model id resolution ────────────────────────────────────────────

def resolve_model_id(provider: str, model: str, api_base: str | None) -> str:
    """Translate (provider, model) into a LiteLLM model identifier.

    Most providers accept `provider/model`. vLLM and other OpenAI-compatible
    self-hosted endpoints are accessed by setting OPENAI_API_BASE + using
    `openai/<model>` regardless of what the local model is.
    """
    if provider in ("vllm", "openai-compatible"):
        if not api_base:
            raise ValueError(f"provider={provider!r} requires --api-base")
        os.environ["OPENAI_API_BASE"] = api_base
        os.environ.setdefault("OPENAI_API_KEY", "dummy")
        return f"openai/{model}"
    return f"{provider}/{model}"


# ─── Main loop ──────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-LLM ReAct loop for the YOLO self-trainer")
    ap.add_argument("--project", type=Path, required=True, help="project directory (for events.jsonl)")
    ap.add_argument("--provider", required=True,
                    help="anthropic | openai | gemini | ollama | groq | together_ai | vllm")
    ap.add_argument("--model", required=True,
                    help="model id without provider prefix (e.g. claude-opus-4-7, gpt-4o, qwen2.5:32b)")
    ap.add_argument("--api-base", default=None,
                    help="custom OpenAI-compatible endpoint (vLLM, local servers)")
    ap.add_argument("--system-prompt-file", type=Path, required=True)
    ap.add_argument("--user-prompt-file", type=Path,
                    help="path to user prompt; if omitted, read from stdin")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--guard-script", type=Path, default=None,
                    help="path to claude_bash_guard.py — Bash calls blocked when guard rejects")
    ap.add_argument("--stream-json-out", type=Path, required=True,
                    help="where to write per-turn JSON lines (compatible with claude --output-format stream-json)")
    ap.add_argument("--tool-result-max-chars", type=int, default=DEFAULT_TOOL_RESULT_MAX)
    ap.add_argument("--bash-timeout", type=int, default=DEFAULT_BASH_TIMEOUT)
    ap.add_argument("--llm-timeout", type=int, default=DEFAULT_LLM_TIMEOUT)
    ap.add_argument("--round", type=int, default=0, help="round number for event emission")
    ap.add_argument("--session-id", default="agent", help="session id for event emission")
    args = ap.parse_args()

    # Lazy import — only fail with a friendly message if litellm isn't there.
    try:
        import litellm  # type: ignore
    except ImportError:
        print(
            "ERROR: litellm is not installed in this venv. Install with:\n"
            "  source .venv/bin/activate && pip install 'litellm>=1.45,<2'",
            file=sys.stderr,
        )
        return 2

    # System + user prompts
    if not args.system_prompt_file.exists():
        print(f"ERROR: system prompt file not found: {args.system_prompt_file}", file=sys.stderr)
        return 2
    system_prompt = args.system_prompt_file.read_text()
    nudge = PROVIDER_TOOL_NUDGE.get(args.provider, "")
    if nudge:
        system_prompt = system_prompt + nudge

    if args.user_prompt_file:
        if not args.user_prompt_file.exists():
            print(f"ERROR: user prompt file not found: {args.user_prompt_file}", file=sys.stderr)
            return 2
        user_prompt = args.user_prompt_file.read_text()
    else:
        user_prompt = sys.stdin.read()
    if not user_prompt.strip():
        print("ERROR: user prompt is empty", file=sys.stderr)
        return 2

    try:
        model_id = resolve_model_id(args.provider, args.model, args.api_base)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    writer = StreamJSONWriter(args.stream_json_out)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    start_ts = time.time()
    total_cost = 0.0
    num_turns = 0
    stop_reason = "unknown"

    emit_event_subprocess(
        args.project, "claude-started",
        round=args.round, session_id=args.session_id,
    )

    try:
        for _ in range(args.max_turns):
            num_turns += 1
            # Ollama-specific: `think=False` disables Qwen3 / DeepSeek thinking
            # blocks. With thinking enabled the model burns 800+ tokens per
            # turn on internal monologue AND its multi-turn tool call format
            # breaks (it emits subsequent calls as JSON-in-text rather than
            # structured tool_calls). Verified against raw ollama API: the
            # same prompt with think=false correctly emits parallel structured
            # tool calls; with thinking on, only the first call is structured.
            extra_kwargs = {}
            if args.provider == "ollama":
                extra_kwargs["extra_body"] = {"think": False}
            try:
                resp = litellm.completion(
                    model=model_id,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    stream=False,
                    timeout=args.llm_timeout,
                    **extra_kwargs,
                )
            except Exception as e:
                # Surface the upstream API error in stream-json so the human log
                # extractor shows it, then re-raise to hit the outer handler.
                writer.write({
                    "type": "result", "subtype": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "duration_ms": int((time.time() - start_ts) * 1000),
                    "num_turns": num_turns, "total_cost_usd": total_cost,
                })
                raise

            # Cost — best-effort. Some providers don't ship pricing in litellm.
            try:
                cost = litellm.completion_cost(completion_response=resp)
                if cost:
                    total_cost += float(cost)
            except Exception:
                pass

            choice = resp.choices[0]
            msg = choice.message
            writer.write(msg_to_log_assistant(msg))
            messages.append(msg_to_history_dict(msg))

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                stop_reason = "end_turn"
                break

            for tc in tool_calls:
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {"_raw_arguments": tc.function.arguments}
                result = dispatch_tool(
                    tc.function.name, tool_args,
                    guard_script=args.guard_script,
                    bash_timeout=args.bash_timeout,
                    max_chars=args.tool_result_max_chars,
                )
                writer.write({"type": "tool", "content": result})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # for-else: loop completed without break → hit max_turns
            stop_reason = "max_turns"

    except KeyboardInterrupt:
        stop_reason = "interrupted"
        writer.write({
            "type": "result", "subtype": "interrupted",
            "duration_ms": int((time.time() - start_ts) * 1000),
            "num_turns": num_turns, "total_cost_usd": total_cost,
        })
        writer.close()
        emit_event_subprocess(
            args.project, "claude-finished",
            round=args.round, exit_code=130,
            duration_sec=int(time.time() - start_ts),
        )
        return 130
    except Exception as e:
        traceback.print_exc()
        writer.close()
        emit_event_subprocess(
            args.project, "claude-finished",
            round=args.round, exit_code=1,
            duration_sec=int(time.time() - start_ts),
        )
        return 1

    duration_ms = int((time.time() - start_ts) * 1000)
    writer.write({
        "type": "result",
        "subtype": stop_reason,
        "duration_ms": duration_ms,
        "num_turns": num_turns,
        "total_cost_usd": total_cost,
    })
    writer.close()

    emit_event_subprocess(
        args.project, "claude-finished",
        round=args.round, exit_code=0,
        duration_sec=int(time.time() - start_ts),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
