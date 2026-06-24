#!/usr/bin/env python3
"""Single source of truth for YOLO hyperparameter bounds.

Why this module:
  Pre-refactor, the agent picked params by sed-editing train.sh, and an
  inline Python heredoc at the top of train.sh range-checked the result.
  That coupled the bounds to the shell file and made task / fine-tune
  awareness impossible without forking the validator. Now every consumer
  reads from `BASE_BOUNDS` here:

    - scripts/apply_params.py — primary param boundary; the SOLE writer of
      effective_params.env. Runs at the top of train.sh.
    - templates/train.sh.tmpl  — defense-in-depth `validate-env` CLI call
      after sourcing, so a bypassed apply_params can't slip past.
    - scripts/baseline_policy.py — also reads BASE_BOUNDS so its sampled
      hyperparameters fall inside the validator's window by construction.

Bounds are derived per (task, fine_tune) via `bounds_for(...)`. Fine-tune is
detected from the WEIGHTS path (anything ending in `best.pt` is a fine-tune;
pretrained yolo*.pt is a cold start). Per-task tweaks:
  - pose: DEGREES tightened (large rotations corrupt keypoint labels)
  - classify: MOSAIC / COPY_PASTE zeroed (object-detection augmentations)
  - fine-tune: EPOCHS lower bound relaxed (short fine-tunes are often optimal)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


# ─── The bounds table (THE source of truth) ──────────────────────────
#
# Each entry is a dict {type, min?, max?, multiple_of?, allow?, choices?,
# must_exist?}. validate() walks this dict for every param the caller passes.

BASE_BOUNDS: dict[str, dict] = {
    "LR":              {"type": "float", "min": 0.0001, "max": 0.05},
    "LR_FINAL":        {"type": "float", "min": 0.0001, "max": 0.05},
    "WEIGHT_DECAY":    {"type": "float", "min": 0.0,    "max": 0.005},
    "EPOCHS":          {"type": "int",   "min": 50,     "max": 500},
    "PATIENCE":        {"type": "int",   "min": 20,     "max": 100},
    "IMGSZ":           {"type": "int",   "min": 640,    "max": 1280, "multiple_of": 32},
    # BATCH=-1 means "auto" — ultralytics picks the best batch for the VRAM.
    "BATCH":           {"type": "int",   "min": 2,      "max": 64,   "allow": (-1,)},
    "OPTIMIZER":       {"type": "choice", "choices": ["AdamW", "SGD", "auto"]},
    "MOSAIC":          {"type": "float", "min": 0.0,    "max": 1.0},
    "MIXUP":           {"type": "float", "min": 0.0,    "max": 1.0},
    "COPY_PASTE":      {"type": "float", "min": 0.0,    "max": 1.0},
    "ERASING":         {"type": "float", "min": 0.0,    "max": 1.0},
    "FLIPLR":          {"type": "float", "min": 0.0,    "max": 1.0},
    "FLIPUD":          {"type": "float", "min": 0.0,    "max": 1.0},
    "SCALE":           {"type": "float", "min": 0.0,    "max": 1.0},
    "DEGREES":         {"type": "float", "min": 0.0,    "max": 45.0},
    "MOMENTUM":        {"type": "float", "min": 0.6,    "max": 0.999},
    "WARMUP_EPOCHS":   {"type": "float", "min": 0.0,    "max": 10.0},
    "WARMUP_MOMENTUM": {"type": "float", "min": 0.0,    "max": 1.0},
    "HSV_H":           {"type": "float", "min": 0.0,    "max": 0.1},
    "HSV_S":           {"type": "float", "min": 0.0,    "max": 0.9},
    "HSV_V":           {"type": "float", "min": 0.0,    "max": 0.9},
    "TRANSLATE":       {"type": "float", "min": 0.0,    "max": 0.5},
    "CLOSE_MOSAIC":    {"type": "int",   "min": 0,      "max": 100},
    "COS_LR":          {"type": "choice", "choices": ["true", "false", "True", "False"]},
    "WEIGHTS":         {"type": "path",  "must_exist": True},
}

# Keys the agent MUST set every round. Missing any of these aborts.
# Augmentation knobs are intentionally optional — the agent only specifies
# what it actually wants to change this round, mirroring the old sed-edit
# ergonomics where unchanged params inherited the previous values.
REQUIRED_KEYS: frozenset[str] = frozenset({
    "WEIGHTS", "EPOCHS", "LR", "LR_FINAL",
    "IMGSZ", "BATCH", "OPTIMIZER", "PATIENCE",
})


# ─── Task / fine-tune awareness ──────────────────────────────────────

def is_finetune(weights_path: str) -> bool:
    """True when WEIGHTS points at a previous run's best.pt vs. a pretrained.

    Path inspection — no separate `mode` field for the agent to disagree
    with WEIGHTS on. Ultralytics writes its outputs under
    runs/<task>/<run>/weights/best.pt so this is unambiguous.
    """
    return weights_path.endswith("/best.pt")


def bounds_for(task: str, fine_tune: bool) -> dict[str, dict]:
    """Derive per-(task, fine_tune) bounds from BASE_BOUNDS.

    Returns a deep copy so callers can mutate without poisoning the
    module-level dict.
    """
    b = copy.deepcopy(BASE_BOUNDS)
    if fine_tune:
        # Short fine-tunes from best.pt are often optimal; let the agent
        # propose EPOCHS as low as 10.
        b["EPOCHS"]["min"] = 10
    if task == "pose":
        # Keypoint annotations don't survive large rotations.
        b["DEGREES"]["max"] = 15.0
    if task == "classify":
        # Object-detection augmentations don't apply meaningfully to
        # classification; force-disable.
        b["MOSAIC"]["max"]     = 0.0
        b["COPY_PASTE"]["max"] = 0.0
    return b


# ─── Validation ─────────────────────────────────────────────────────

def _violation(key: str, expected: str, got, reason: str) -> dict:
    return {"key": key, "expected": expected, "got": got, "reason": reason}


def _coerce_numeric(raw, want: str):
    """Convert raw → int or float per `want`; raise ValueError on failure.

    Accepts ints/floats/strings. Strings like "200" parse as int when
    requested. A float-typed bound still accepts an int input (auto-promoted).
    """
    if want == "int":
        if isinstance(raw, bool):
            raise ValueError("bool is not int")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{raw!r} is not whole-valued")
            return int(raw)
        return int(str(raw))
    if want == "float":
        if isinstance(raw, bool):
            raise ValueError("bool is not float")
        if isinstance(raw, (int, float)):
            return float(raw)
        return float(str(raw))
    raise ValueError(f"unknown numeric type {want!r}")


def _check_one(key: str, raw, spec: dict) -> list[dict]:
    """Validate a single key/value against its spec. Returns 0 or 1 violation."""
    t = spec["type"]

    if t == "choice":
        if raw not in spec["choices"]:
            return [_violation(key, f"one of {spec['choices']}", raw, "not in allowed set")]
        return []

    if t == "path":
        # Accept str or os.PathLike; existence checked when must_exist=True.
        if not isinstance(raw, (str, os.PathLike)):
            return [_violation(key, "filesystem path string", raw, "not a string")]
        if spec.get("must_exist") and not Path(str(raw)).exists():
            return [_violation(key, "existing file", raw, "path does not exist")]
        return []

    if t in ("int", "float"):
        try:
            v = _coerce_numeric(raw, t)
        except (ValueError, TypeError):
            return [_violation(key, t, raw, "not a valid number")]
        allow = spec.get("allow", ())
        if v in allow:
            return []
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and v < lo:
            return [_violation(key, f"≥ {lo}", v, "below min")]
        if hi is not None and v > hi:
            return [_violation(key, f"≤ {hi}", v, "above max")]
        mo = spec.get("multiple_of")
        if mo is not None and v % mo != 0:
            return [_violation(key, f"multiple of {mo}", v, "not a multiple")]
        return []

    return [_violation(key, "(unknown spec type)", raw, f"validator bug: type={t!r}")]


def validate(params: dict, task: str, fine_tune: bool) -> list[dict]:
    """Return a list of violation dicts. Empty list = pass.

    Each violation: {"key": str, "expected": str, "got": Any, "reason": str}.
    This shape is what apply_params.py serializes into the validation-failed
    event's violations_json field.

    Does NOT check REQUIRED_KEYS presence — that's apply_params's job (it
    composes schema-level checks with this range-level check).
    """
    bounds = bounds_for(task, fine_tune)
    out: list[dict] = []
    for key, raw in params.items():
        spec = bounds.get(key)
        if spec is None:
            # Unknown key — typo catcher.
            out.append(_violation(key, f"one of {sorted(BASE_BOUNDS)}", raw, "unknown key"))
            continue
        out.extend(_check_one(key, raw, spec))
    return out


# ─── CLI (defense-in-depth from train.sh) ────────────────────────────

def _cli_validate_env(args) -> int:
    """Read env vars whose names appear in BASE_BOUNDS; report violations.

    Used by train.sh as the second wall after apply_params runs upstream
    of it. If apply_params worked, this returns 0 — it's not redundant; a
    bypassed apply_params (broken script, manual launch) would still trip
    this wall before yolo train fires.
    """
    params: dict = {}
    for key in BASE_BOUNDS:
        raw = os.environ.get(key)
        if raw is None or raw == "":
            continue
        params[key] = raw

    if args.weights:
        # CLI override takes precedence so the train.sh caller can pass the
        # actual WEIGHTS even when it's not yet in env.
        params["WEIGHTS"] = args.weights

    weights = params.get("WEIGHTS", "")
    fine_tune = is_finetune(weights) if weights else False

    viols = validate(params, task=args.task, fine_tune=fine_tune)
    if not viols:
        print(f"[param_bounds] OK ({len(params)} params, task={args.task}, fine_tune={fine_tune})")
        return 0

    print(f"[param_bounds] FAIL — {len(viols)} violation(s):", file=sys.stderr)
    for v in viols:
        print(f"  ✗ {v['key']}={v['got']!r} — {v['reason']} (expected {v['expected']})", file=sys.stderr)
    return 1


def _cli_show(args) -> int:
    """Print the resolved bounds table as JSON. Useful for the prompt
    template and for operator debugging."""
    fine_tune = False
    if args.weights:
        fine_tune = is_finetune(args.weights)
    elif args.fine_tune:
        fine_tune = True
    print(json.dumps(bounds_for(args.task, fine_tune), indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    vp = sub.add_parser("validate-env", help="validate the current env vars")
    vp.add_argument("--task",    required=True,
                    choices=["detect", "obb", "segment", "pose", "classify"])
    vp.add_argument("--weights", default="",
                    help="WEIGHTS path; used to derive fine_tune. If omitted, "
                         "the env var WEIGHTS is read.")

    sp = sub.add_parser("show", help="print resolved bounds for a task")
    sp.add_argument("--task",      required=True,
                    choices=["detect", "obb", "segment", "pose", "classify"])
    sp.add_argument("--weights",   default="")
    sp.add_argument("--fine-tune", action="store_true")

    args = ap.parse_args()
    if args.cmd == "validate-env":
        return _cli_validate_env(args)
    if args.cmd == "show":
        return _cli_show(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
