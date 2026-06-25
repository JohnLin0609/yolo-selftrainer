#!/usr/bin/env python3
"""Sweep multiple LLM providers against the same training pipeline.

Why this exists:
  README.md used to carry star ratings (★★★★★ for Claude, ★★ for Ollama).
  The stars were guesswork and drifted as models changed. This script
  replaces the guess with data: run the same demo dataset, same rounds,
  through each provider's agent loop, then aggregate val/test mAP, total
  LLM cost, wall time, and circuit-breaker trips into a single Markdown
  comparison table.

  The aggregator (`benchmark_aggregate.aggregate_from_events`) + renderer
  (`benchmark_render.render_comparison_table`) are pure and unit-tested
  separately. This file is the orchestrator that ties them to the
  existing scaffolding (`scripts/new_project.sh` + `start_agent.sh`).

Usage:
  python3 scripts/benchmark.py \\
      --providers anthropic openai gemini \\
      --models   claude-haiku-4-5-20251001 gpt-4o-mini gemini-2.5-flash \\
      --dataset  datasets/demo \\
      --rounds   3 \\
      --output   benchmark_report.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# Provider → required env var (None = no key needed, e.g. ollama runs locally).
# Mirrors scripts/test_agent_smoke.py:PROVIDERS — duplicated rather than
# imported because the two scripts' lifecycles are independent.
PROVIDER_ENV = {
    "anthropic":   "ANTHROPIC_API_KEY",
    "openai":      "OPENAI_API_KEY",
    "gemini":      "GEMINI_API_KEY",
    "groq":        "GROQ_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
    "ollama":      None,
    "vllm":        None,
}


def slug(s: str) -> str:
    """File-system-safe project-name segment derived from a model id.

    `:` (ollama), `/` (together_ai's Org/Model), `.` (version numbers) all
    map to `_`. Anything outside [A-Za-z0-9_-] follows.
    """
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")


def preflight_keys(providers: list[str]) -> list[str]:
    """Return a list of providers whose required API key is missing."""
    missing: list[str] = []
    for p in providers:
        if p not in PROVIDER_ENV:
            print(f"[benchmark] unknown provider {p!r} — supported: "
                  f"{', '.join(PROVIDER_ENV)}", file=sys.stderr)
            missing.append(p)
            continue
        env = PROVIDER_ENV[p]
        if env and not os.environ.get(env):
            missing.append(p)
    return missing


def scaffold_project(
    framework_root: Path, project_name: str,
    provider: str, model: str, api_base: str | None,
    dataset: Path, rounds: int, task: str | None, device: int,
) -> Path:
    """Invoke scripts/new_project.sh and return the path to the scaffolded
    project directory. Forces overwrite — benchmark runs are
    reproducible; re-running the orchestrator should reset state."""
    cmd = [
        "bash", str(framework_root / "scripts" / "new_project.sh"),
        "--dataset",     str(dataset),
        "--name",        project_name,
        "--mode",        "agent",
        "--llm-provider", provider,
        "--llm-model",   model,
        "--max-rounds",  str(rounds),
        "--device",      str(device),
        "--force",
    ]
    if task:
        cmd.extend(["--task", task])
    if api_base:
        cmd.extend(["--llm-api-base", api_base])

    print(f"[benchmark] scaffolding {project_name} ({provider}/{model})", flush=True)
    proc = subprocess.run(cmd, cwd=framework_root, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"new_project.sh failed for {provider}/{model} (exit {proc.returncode})"
        )
    return framework_root / "projects" / project_name


def run_session(project_dir: Path) -> int:
    """Block on `bash start_agent.sh` until the chain terminates cleanly
    (start_agent.sh exits 0 once ROUND > MAX_ROUNDS). Returns the exit code."""
    start = project_dir / "start_agent.sh"
    if not start.exists():
        raise FileNotFoundError(
            f"no start_agent.sh in {project_dir} — scaffold must have failed"
        )
    print(f"[benchmark] launching {start.name} (blocks until MAX_ROUNDS reached)",
          flush=True)
    proc = subprocess.run(["bash", str(start)], cwd=project_dir)
    return proc.returncode


def read_events(project_dir: Path) -> list[dict]:
    """Read events.jsonl into a list of dicts (skip blank / malformed lines).

    Doesn't import scripts/event.py — that would couple our lifecycles and
    we only need the JSON parse. Malformed lines are tolerated because the
    aggregator is robust to partial data.
    """
    path = project_dir / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[benchmark] WARN: malformed events.jsonl line in {project_dir}: {e}",
                  file=sys.stderr)
    return events


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cross-provider benchmark: sweep providers, emit a "
                    "Markdown comparison table backed by each project's events.jsonl."
    )
    ap.add_argument("--providers", nargs="+", required=True,
                    help="ordered list of providers to sweep")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model id per provider (same count as --providers)")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="dataset directory (passed to new_project.sh --dataset)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="MAX_ROUNDS per provider (default: 3)")
    ap.add_argument("--output", type=Path, required=True,
                    help="path to write the Markdown comparison table")
    ap.add_argument("--workspace", type=Path, default=None,
                    help="naming prefix container; default: benchmarks/<timestamp>")
    ap.add_argument("--task", default=None,
                    help="explicit task (default: auto-detect via new_project.sh)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--api-base", default=None,
                    help="custom OpenAI-compatible endpoint, applied to ALL "
                         "providers that need one (vllm). Per-provider api-base "
                         "is intentionally not supported — keep the sweep simple.")
    args = ap.parse_args()

    if len(args.providers) != len(args.models):
        print(f"ERROR: --providers ({len(args.providers)}) and --models "
              f"({len(args.models)}) must have the same count",
              file=sys.stderr)
        return 2

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    # Preflight: refuse to run if any provider's required env var is unset.
    # Better to fail loud now than after 30 minutes of one provider's training.
    missing = preflight_keys(args.providers)
    if missing:
        for p in missing:
            env = PROVIDER_ENV.get(p)
            if env:
                print(f"ERROR: ${env} not set — required for provider {p!r}",
                      file=sys.stderr)
            else:
                print(f"ERROR: provider {p!r} is unsupported", file=sys.stderr)
        return 2

    framework_root = Path(__file__).resolve().parent.parent

    # Lazy import the pure modules so missing pieces fail loud here, not in
    # the middle of a long sweep.
    sys.path.insert(0, str(framework_root / "scripts"))
    from benchmark_aggregate import aggregate_from_events
    from benchmark_render import render_comparison_table

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_prefix = (args.workspace.name if args.workspace
                        else f"bench_{timestamp}")

    rows: list[dict] = []
    for provider, model in zip(args.providers, args.models):
        project_name = f"{workspace_prefix}_{provider}_{slug(model)}"
        try:
            proj_dir = scaffold_project(
                framework_root, project_name,
                provider, model, args.api_base,
                args.dataset, args.rounds, args.task, args.device,
            )
            exit_code = run_session(proj_dir)
            if exit_code != 0:
                print(f"[benchmark] WARN: {provider}/{model} exited {exit_code} "
                      "— aggregating partial events.jsonl anyway", file=sys.stderr)
        except Exception as e:
            print(f"[benchmark] ERROR while running {provider}/{model}: {e}",
                  file=sys.stderr)
            # Aggregate with an empty events list so the row still appears
            # — that's the honest signal (everything zero / None).
            rows.append(aggregate_from_events([], provider, model))
            continue

        events = read_events(proj_dir)
        rows.append(aggregate_from_events(events, provider, model))

    report = render_comparison_table(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(f"\n[benchmark] wrote {args.output}\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
