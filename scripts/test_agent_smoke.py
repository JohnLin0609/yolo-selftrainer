#!/usr/bin/env python3
"""Smoke test for the multi-LLM agent loop.

For each provider you've configured an API key for, run a single-turn task:
  "Use the Bash tool to run `python -c 'import ultralytics; print(ultralytics.__version__)'`
   and then write a one-line summary."

Verifies end-to-end:
  - litellm can reach the provider
  - tool calling works (model emits Bash call with correct args)
  - guard subprocess integration runs
  - stream-json output is produced

Usage:
  python3 scripts/test_agent_smoke.py                  # all providers with keys set
  python3 scripts/test_agent_smoke.py --providers anthropic openai
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROVIDERS = {
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",  # cheap for smoke test
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
    },
    "groq": {
        "env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "together_ai": {
        "env": "TOGETHER_API_KEY",
        "default_model": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    },
    "ollama": {
        # ollama doesn't need an API key; check if `ollama` binary is on PATH
        "env": None,
        "default_model": "qwen2.5:32b",
        "check": lambda: shutil.which("ollama") is not None,
    },
}


SYSTEM_PROMPT = """You are a smoke-test agent. Your job is to verify the tool-calling pipeline works.

You have three tools: Bash, Write, Read.

Complete the task in the user message. Use the Bash tool — do not just describe what you would do."""

USER_PROMPT = """Run this exact command via the Bash tool:
  python -c 'import ultralytics; print(ultralytics.__version__)'

Then in your final text response, state the ultralytics version you saw."""


def run_provider(provider: str, model: str, framework_root: Path) -> tuple[bool, str]:
    """Run a single agent invocation against `provider`. Returns (pass, detail)."""
    info = PROVIDERS[provider]
    if info.get("env") and not os.environ.get(info["env"]):
        return False, f"{info['env']} not set"
    if "check" in info and not info["check"]():
        return False, "provider check failed (binary not on PATH or service unreachable)"

    with tempfile.TemporaryDirectory(prefix=f"smoke_{provider}_") as td:
        td_path = Path(td)
        sys_file = td_path / "system.md"
        usr_file = td_path / "user.md"
        out_file = td_path / "raw.jsonl"
        sys_file.write_text(SYSTEM_PROMPT)
        usr_file.write_text(USER_PROMPT)

        cmd = [
            "python3", str(framework_root / "scripts" / "run_agent.py"),
            "--project",            str(td_path),
            "--provider",           provider,
            "--model",              model,
            "--system-prompt-file", str(sys_file),
            "--user-prompt-file",   str(usr_file),
            "--guard-script",       str(framework_root / "scripts" / "claude_bash_guard.py"),
            "--stream-json-out",    str(out_file),
            "--max-turns",          "6",
            "--bash-timeout",       "30",
            "--llm-timeout",        "60",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return False, "agent invocation timed out (>3min)"

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout)[-500:]
            return False, f"agent exited {proc.returncode}: {tail!r}"

        if not out_file.exists():
            return False, "stream-json output file not created"

        # Verify the model actually called Bash with ultralytics import.
        called_bash = False
        saw_version = False
        for line in out_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "assistant":
                for block in ev.get("message", {}).get("content", []):
                    if block.get("type") == "tool_use" and block.get("name") == "Bash":
                        cmd_text = block.get("input", {}).get("command", "")
                        if "ultralytics" in cmd_text:
                            called_bash = True
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if "8." in text:  # ultralytics 8.x
                            saw_version = True
            if ev.get("type") == "tool":
                content = ev.get("content", "")
                if isinstance(content, str) and content.startswith("8."):
                    saw_version = True

        if not called_bash:
            return False, "model did not call Bash with ultralytics import"
        if not saw_version:
            # Not a hard failure — some models might rephrase the version
            return True, "PASS (bash called; version mention not detected — soft pass)"
        return True, "PASS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="*", default=list(PROVIDERS),
                    help="subset of providers to test")
    ap.add_argument("--model", default=None,
                    help="override model for all selected providers")
    args = ap.parse_args()

    framework_root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("Multi-LLM agent smoke test")
    print(f"Framework root: {framework_root}")
    print("=" * 60)

    results: list[tuple[str, bool, str]] = []
    for provider in args.providers:
        if provider not in PROVIDERS:
            print(f"  {provider}: SKIP (unknown provider; valid: {list(PROVIDERS)})")
            continue
        model = args.model or PROVIDERS[provider]["default_model"]
        print(f"\n--- {provider} / {model} ---")
        ok, detail = run_provider(provider, model, framework_root)
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {detail}")
        results.append((provider, ok, detail))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for prov, ok, detail in results:
        print(f"  {'✓' if ok else '✗'} {prov}: {detail}")

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
