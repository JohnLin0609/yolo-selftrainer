#!/usr/bin/env python3
"""Apply the agent's next_params.json — the ONLY writer of effective_params.env.

The agent's parameter contract:
  Each round, the agent writes `next_params.json` — a flat JSON object whose
  keys are members of `param_bounds.BASE_BOUNDS`. This script:

  1. Reads next_params.json (missing on round 1 is OK — train.sh's
     scaffolded defaults take over).
  2. Schema-checks: top-level is a flat object, all keys are known,
     REQUIRED_KEYS are present.
  3. Derives `fine_tune` from WEIGHTS via param_bounds.is_finetune.
  4. Range-checks via param_bounds.validate(task, fine_tune).
  5. On success: writes effective_params.env (POSIX-safe `KEY=value` lines)
     that train.sh sources to override its scaffolded defaults.
  6. On failure: emits a validation-failed event whose violations_json
     carries the structured violations list, writes a clear
     "Parameter validation FAILED" addendum to next_instruction.md, exits 1.

Round-not-consumed semantics:
  This script DOES NOT bump consecutive_failures or write HALTED — train.sh
  already owns the "validator failed" code path (lines 105-148 of the old
  template). When apply_params exits non-zero, train.sh hits its existing
  VALIDATE_EXIT≠0 branch and inherits that handling unchanged. Single
  source of truth for the wake-without-consuming-round flow.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

# Reuse the bounds + REQUIRED_KEYS set without re-importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import param_bounds as pb   # noqa: E402


def _emit_validation_failed(project: Path, round_num: int, violations: list[dict]) -> None:
    """Emit a validation-failed event via event.py.

    We shell out to event.py rather than importing it so the audit log
    writes go through the same code path as train.sh's emits — a single
    point of schema enforcement.
    """
    event_py = Path(__file__).resolve().parent / "event.py"
    try:
        subprocess.run(
            ["python3", str(event_py), str(project), "emit", "validation-failed",
             "--round", str(round_num),
             "--violations-json", json.dumps(violations, default=str)],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        # Don't let an audit-log failure mask the validator failure itself —
        # the violations are still printed to stderr below.
        print(f"[apply_params] WARN: failed to emit validation-failed event: {e}",
              file=sys.stderr)


def _write_next_instruction_addendum(project: Path, violations: list[dict],
                                     reason: str, framework_root: Path) -> None:
    """Overwrite next_instruction.md with a parameter-fix message.

    Replaces (does not append to) any prior content because the prior
    content was the agent's plan from the failed round, which is no
    longer relevant — the round didn't happen.
    """
    lines = [
        "## Parameter validation FAILED — round NOT consumed",
        "",
        f"Reason: **{reason}**",
        "",
        "Your `next_params.json` was rejected by `scripts/apply_params.py`.",
        "This counts toward the consecutive_failures budget (3 in a row → HALTED)",
        "but does NOT consume a round.",
        "",
        "## Violations",
        "",
    ]
    if violations:
        for v in violations:
            got = v.get("got")
            lines.append(
                f"- **{v.get('key','?')}** = `{got!r}` — {v.get('reason','?')} "
                f"(expected {v.get('expected','?')})"
            )
    else:
        lines.append("- (no per-key violations; see Reason above)")
    lines += [
        "",
        "## How to fix",
        "",
        f"1. Inspect the bounds for this task / fine-tune state:",
        f"   ```bash",
        f"   python3 {framework_root}/scripts/param_bounds.py show --task <TASK> [--fine-tune]",
        f"   ```",
        "2. Rewrite `next_params.json` so every violated key is in range",
        "   and every required key is present. Required keys:",
        f"   `{sorted(pb.REQUIRED_KEYS)}`.",
        "3. Re-launch training (this round is reset; you will not lose a slot):",
        "   ```bash",
        "   nohup bash $SCRIPT_DIR/train.sh > $SCRIPT_DIR/current.log 2>&1 &",
        "   ```",
        "",
        "Do NOT touch train.sh directly — the structured contract is the only",
        "supported way for you to change hyperparameters.",
    ]
    (project / "next_instruction.md").write_text("\n".join(lines) + "\n")


def _format_env_value(key: str, value) -> str:
    """Render value as a POSIX-safe `KEY=...` line for bash sourcing.

    Numbers go in unquoted; strings (paths, OPTIMIZER, COS_LR) get quoted.
    """
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (int, float)):
        rendered = repr(value) if isinstance(value, float) else str(value)
    else:
        rendered = shlex.quote(str(value))
    return f"{key}={rendered}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project",      type=Path, required=True)
    ap.add_argument("--task",         required=True,
                    choices=["detect", "obb", "segment", "pose", "classify"])
    ap.add_argument("--round",        type=int, required=True, dest="round_num")
    ap.add_argument("--next-params",  type=Path, required=True)
    ap.add_argument("--out",          type=Path, required=True)
    args = ap.parse_args()

    framework_root = Path(__file__).resolve().parent.parent

    # ─── Step 1: read next_params.json ────────────────────────────────
    if not args.next_params.exists():
        if args.round_num <= 1:
            # Round 1 cold start with no agent picks yet is fine — train.sh's
            # scaffolded defaults are the contract for this round. Write an
            # empty env file so train.sh's `source` is a no-op.
            args.out.write_text("")
            print(f"[apply_params] round {args.round_num}: no next_params.json — "
                  "using train.sh's scaffolded defaults")
            return 0
        viols = [{"key": "<file>", "expected": "next_params.json",
                  "got": str(args.next_params), "reason": "file not found"}]
        _emit_validation_failed(args.project, args.round_num, viols)
        _write_next_instruction_addendum(args.project, viols,
                                         "next_params.json missing",
                                         framework_root)
        print(f"[apply_params] FAIL: next_params.json not at {args.next_params}",
              file=sys.stderr)
        return 1

    try:
        raw = json.loads(args.next_params.read_text())
    except json.JSONDecodeError as e:
        viols = [{"key": "<file>", "expected": "valid JSON object",
                  "got": str(args.next_params), "reason": f"JSON parse error: {e}"}]
        _emit_validation_failed(args.project, args.round_num, viols)
        _write_next_instruction_addendum(args.project, viols,
                                         "next_params.json is not valid JSON",
                                         framework_root)
        print(f"[apply_params] FAIL: next_params.json is not valid JSON: {e}",
              file=sys.stderr)
        return 1

    if not isinstance(raw, dict):
        viols = [{"key": "<root>", "expected": "JSON object",
                  "got": type(raw).__name__,
                  "reason": "next_params.json top level must be a flat object"}]
        _emit_validation_failed(args.project, args.round_num, viols)
        _write_next_instruction_addendum(args.project, viols,
                                         "next_params.json top-level not an object",
                                         framework_root)
        print("[apply_params] FAIL: next_params.json top-level must be a JSON object",
              file=sys.stderr)
        return 1

    # ─── Step 2: REQUIRED_KEYS check ─────────────────────────────────
    missing = sorted(pb.REQUIRED_KEYS - set(raw.keys()))
    schema_viols: list[dict] = []
    for k in missing:
        schema_viols.append({"key": k, "expected": "present (required)",
                             "got": None, "reason": "required key missing"})

    # Unknown-key check happens inside pb.validate() but we run it after
    # the WEIGHTS-driven fine_tune derivation below, so any unknown-key
    # violations land alongside range violations in the same list.

    # ─── Step 3: derive fine_tune ────────────────────────────────────
    weights = raw.get("WEIGHTS", "")
    fine_tune = pb.is_finetune(str(weights)) if isinstance(weights, str) else False

    # ─── Step 4: range + unknown-key check ───────────────────────────
    range_viols = pb.validate(raw, task=args.task, fine_tune=fine_tune)

    all_viols = schema_viols + range_viols
    if all_viols:
        _emit_validation_failed(args.project, args.round_num, all_viols)
        reason = (
            f"{len(all_viols)} parameter violation(s) — "
            f"missing-required: {len(schema_viols)}, range/unknown: {len(range_viols)}"
        )
        _write_next_instruction_addendum(args.project, all_viols, reason, framework_root)
        print(f"[apply_params] FAIL — {len(all_viols)} violation(s):", file=sys.stderr)
        for v in all_viols:
            print(f"  ✗ {v.get('key','?')}={v.get('got')!r} — {v.get('reason','?')} "
                  f"(expected {v.get('expected','?')})", file=sys.stderr)
        return 1

    # ─── Step 5: emit effective_params.env ───────────────────────────
    # Stable key order: BASE_BOUNDS insertion order, then any extras at end
    # (there should be none after the unknown-key check, but defensive).
    out_lines = ["# Auto-generated by scripts/apply_params.py — do NOT edit by hand."]
    out_lines.append(f"# Source: {args.next_params}")
    for key in pb.BASE_BOUNDS:
        if key in raw:
            out_lines.append(_format_env_value(key, raw[key]))

    args.out.write_text("\n".join(out_lines) + "\n")
    print(f"[apply_params] OK ({len(raw)} params, task={args.task}, "
          f"fine_tune={fine_tune}) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
