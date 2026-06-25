"""Step defs for per_class_diagnostics.feature.

Lazy-imports the module under test so the file collects cleanly even when
diagnose_classes / build_prompt-perclass-helpers aren't on disk yet —
mirrors the sandbox_isolation_steps.py pattern.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when


scenarios("../features/per_class_diagnostics.feature")


# ─── Per-scenario mutable context (passed via fixture) ────────────────

@pytest.fixture
def ctx(tmp_path: Path, repo_root: Path, guard_path: Path) -> dict:
    return {
        "project_dir": tmp_path / "proj",
        "repo_root":   repo_root,
        "guard_path":  guard_path,
        "prompt_stdout": "",
        "prompt_stderr": "",
        "prompt_returncode": None,
        "guard_block_result": None,
        "guard_allow_result": None,
    }


def _write_event(events_path: Path, ev: dict) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a") as f:
        f.write(json.dumps(ev) + "\n")


def _run_build_prompt(ctx: dict, round_num: int, max_rounds: int) -> None:
    r = subprocess.run(
        [sys.executable,
         str(ctx["repo_root"] / "scripts" / "build_prompt.py"),
         str(ctx["project_dir"]),
         "--round", str(round_num),
         "--max-rounds", str(max_rounds)],
        capture_output=True, text=True, timeout=30,
    )
    ctx["prompt_stdout"] = r.stdout
    ctx["prompt_stderr"] = r.stderr
    ctx["prompt_returncode"] = r.returncode


def _emit_training_metrics(events_path: Path, round_num: int, run_name: str,
                           best_metric_value: float, ts: str) -> None:
    _write_event(events_path, {
        "ts": ts,
        "type": "training_metrics",
        "round": round_num,
        "run_name": run_name,
        "best_metric_name": "mAP50(B)",
        "best_metric_value": best_metric_value,
        "best_epoch": 50,
        "final_epoch": 100,
        "total_epochs": 100,
        "patience_triggered": False,
        "extras": {"metrics/mAP50(B)": best_metric_value},
    })


def _emit_per_class(events_path: Path, round_num: int, run_name: str,
                    per_class: dict, confusion: list, class_names: list,
                    ts: str) -> None:
    _write_event(events_path, {
        "ts": ts,
        "type": "per_class_metrics",
        "round": round_num,
        "run_name": run_name,
        "per_class": per_class,
        "confusion": confusion,
        "class_names": class_names,
    })


# ─── Background ──────────────────────────────────────────────────────

@given("a fresh project directory")
def given_fresh_project(ctx):
    ctx["project_dir"].mkdir(parents=True, exist_ok=True)


# ─── Scenario 1 ──────────────────────────────────────────────────────

@given(parsers.parse('a completed training run named "{run_name}" at round {round_num:d}'))
def given_one_run(ctx, run_name, round_num):
    events_path = ctx["project_dir"] / "events.jsonl"
    _emit_training_metrics(
        events_path, round_num, run_name,
        best_metric_value=0.50,
        ts=f"2026-06-25T10:00:0{round_num}+08:00",
    )


@given(parsers.parse(
    'the per_class_metrics event for "{run_name}" records class "{a}" with '
    'mAP50={va:f} and class "{b}" with mAP50={vb:f}'
))
def given_per_class_two(ctx, run_name, a, va, b, vb):
    events_path = ctx["project_dir"] / "events.jsonl"
    per_class = {
        a: {"P": 0.4, "R": 0.3, "mAP50": va, "mAP50_95": 0.05, "support": 50},
        b: {"P": 0.9, "R": 0.85, "mAP50": vb, "mAP50_95": 0.70, "support": 50},
    }
    # Confusion matrix: small off-diagonal so we don't accidentally trigger
    # the noise-floor threshold (we just want to confirm the section renders).
    confusion = [[40, 5], [3, 45]]
    _emit_per_class(
        events_path, round_num=1, run_name=run_name,
        per_class=per_class, confusion=confusion, class_names=[a, b],
        ts="2026-06-25T10:00:05+08:00",
    )


# ─── Scenario 2 ──────────────────────────────────────────────────────

@given(parsers.parse(
    'three consecutive runs where class "{cls}" is the worst class each round'
))
def given_three_runs_persistent(ctx, cls):
    events_path = ctx["project_dir"] / "events.jsonl"
    other = "dent"
    # Round 1, 2, 3 — cls is always the worst.
    for round_num in (1, 2, 3):
        run_name = f"run{round_num}"
        ts = f"2026-06-25T10:00:0{round_num}+08:00"
        _emit_training_metrics(
            events_path, round_num, run_name,
            best_metric_value=0.50 + 0.01 * round_num,
            ts=ts,
        )
        per_class = {
            cls:   {"P": 0.4, "R": 0.3,  "mAP50": 0.10 + 0.01 * round_num,
                    "mAP50_95": 0.05, "support": 60},
            other: {"P": 0.9, "R": 0.85, "mAP50": 0.85,
                    "mAP50_95": 0.70, "support": 80},
        }
        confusion = [[40, 5], [3, 70]]
        _emit_per_class(
            events_path, round_num, run_name,
            per_class=per_class, confusion=confusion,
            class_names=[cls, other], ts=ts,
        )


# ─── When ────────────────────────────────────────────────────────────

@when(parsers.parse('build_prompt.py renders round {n:d} of {m:d}'))
def when_render(ctx, n, m):
    _run_build_prompt(ctx, n, m)


# ─── Then (scenario 1) ───────────────────────────────────────────────

@then(parsers.parse('the prompt contains a "{header}" section'))
def then_prompt_has_section(ctx, header):
    assert header in ctx["prompt_stdout"], (
        f"expected section header {header!r} in prompt, got:\n"
        f"{ctx['prompt_stdout'][:500]}"
    )


@then(parsers.parse('the prompt names "{cls}" as the weakest class'))
def then_weakest(ctx, cls):
    out = ctx["prompt_stdout"]
    # The weakest class must appear in the worst-classes table as rank 1.
    # Format from build_per_class_section: "| 1 | `scratch` | ..."
    assert f"| 1 | `{cls}` |" in out, (
        f"expected rank-1 entry for {cls!r}, prompt was:\n{out[:800]}"
    )


@then('the prompt points the agent at the confusion_matrix.png file rather '
      'than embedding the numeric matrix')
def then_matrix_pointer(ctx):
    out = ctx["prompt_stdout"]
    assert "confusion_matrix.png" in out, (
        "expected a pointer to confusion_matrix.png in the per-class section"
    )
    # The numeric matrix would look like "[[40, 5], [3, 45]]" — make sure we
    # didn't accidentally embed it.
    assert "[[40" not in out and "[40, 5]" not in out, (
        f"prompt unexpectedly embeds raw confusion matrix:\n{out[:800]}"
    )


# ─── Then (scenario 2) ───────────────────────────────────────────────

@then(parsers.parse(
    'the prompt contains a "Persistent weakness detected" block flagging "{cls}"'
))
def then_persistent_block(ctx, cls):
    out = ctx["prompt_stdout"]
    assert "Persistent weakness detected" in out, (
        f"missing persistent-weakness block in prompt:\n{out[:1200]}"
    )
    # The block must name the flagged class.
    callout_start = out.index("Persistent weakness detected")
    callout = out[callout_start:callout_start + 1500]
    assert cls in callout, (
        f"persistent-weakness callout does not mention {cls!r}:\n{callout}"
    )


@then('the block recommends labeling, samples, or augmentation review')
def then_recommends_data_actions(ctx):
    out = ctx["prompt_stdout"]
    lowered = out.lower()
    assert "label" in lowered, "expected a labeling-audit recommendation"
    assert "sample" in lowered, "expected a more-samples recommendation"
    assert "augment" in lowered, "expected an augmentation-review recommendation"


@then('the prompt explicitly tells the agent not to modify the dataset directory')
def then_read_only_warning(ctx):
    out = ctx["prompt_stdout"]
    lowered = out.lower()
    assert "do not modify" in lowered or "do not mutate" in lowered, (
        f"expected an explicit do-not-modify warning, got:\n{out[:1200]}"
    )
    assert "datasets/" in out, (
        "expected the warning to reference the datasets/ directory"
    )


@then('the Bash guard rejects writing under the project\'s datasets directory')
def then_guard_blocks(ctx, bash_guard):
    r = bash_guard("echo bad > datasets/labels/foo.txt")
    assert r.returncode != 0, (
        f"guard FAILED to reject datasets/ write — stdout={r.stdout!r} "
        f"stderr={r.stderr!r}"
    )
    assert "BLOCKED" in r.stderr


@then('the Bash guard still allows reading the project\'s datasets directory')
def then_guard_allows_read(ctx, bash_guard):
    for cmd in ("cat datasets/foo/data.yaml",
                "find datasets/ -name *.yaml",
                "grep -r class_names datasets/"):
        r = bash_guard(cmd)
        assert r.returncode == 0, (
            f"guard wrongly blocked read-only command {cmd!r}: "
            f"stderr={r.stderr!r}"
        )
