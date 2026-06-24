#!/usr/bin/env python3
"""LLM-free hyperparameter policy for `--mode baseline`.

Pure function: given (round, seed), prints the params train.sh should use this
round. The orchestrator (start_baseline.sh) sed-edits each KEY=VALUE line into
train.sh, exactly like the LLM would have via the agent loop.

Round 1 is intentionally a no-op (the project scaffolded by new_project.sh
already encodes safe defaults — train.sh's initial values are the
"no-tuning floor"). Round 2..N samples within the validator bounds with a
sub-RNG keyed on (seed * 1000 + round), so the same seed reproduces the
same params for any specific round regardless of how many rounds are run.

Output format (consumed by start_baseline.sh):
  #policy=defaults       ← round 1
  #policy=random-search  ← round ≥ 2
  KEY=value
  KEY=value
  ...

KEEP IN SYNC with the rng() bounds at the top of
templates/train.sh.tmpl — they're the trust boundary; this script must
never propose a value the validator would reject.
"""
from __future__ import annotations

import argparse
import math
import random
import sys


IMGSZ_CHOICES = [640, 768, 896, 1024, 1152, 1280]
OPTIMIZER_CHOICES = ["AdamW", "SGD", "auto"]

BOUNDS = {
    "LR":           (0.0001, 0.05),
    "LR_FINAL":     (0.0001, 0.05),
    "WEIGHT_DECAY": (0.0,    0.005),
    "EPOCHS":       (50,     500),
    "PATIENCE":     (20,     100),
    "MOSAIC":       (0.0, 1.0),
    "MIXUP":        (0.0, 1.0),
    "COPY_PASTE":   (0.0, 1.0),
    "ERASING":      (0.0, 1.0),
    "FLIPLR":       (0.0, 1.0),
    "FLIPUD":       (0.0, 1.0),
    "SCALE":        (0.0, 1.0),
    "DEGREES":      (0.0, 45.0),
}


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    if lo <= 0:
        # WEIGHT_DECAY's lower bound is 0; degrade gracefully to linear there.
        return rng.uniform(lo, hi)
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample(round_idx: int, seed: int) -> dict[str, str]:
    """Return KEY → string-value dict for the given round/seed pair."""
    rng = random.Random(seed * 1000 + round_idx)
    out: dict[str, str] = {}

    for k in ("LR", "LR_FINAL", "WEIGHT_DECAY"):
        lo, hi = BOUNDS[k]
        out[k] = f"{_log_uniform(rng, lo, hi):.5f}"

    lo, hi = BOUNDS["EPOCHS"]
    out["EPOCHS"] = str(rng.randint(lo, hi))
    lo, hi = BOUNDS["PATIENCE"]
    out["PATIENCE"] = str(rng.randint(lo, hi))

    out["IMGSZ"] = str(rng.choice(IMGSZ_CHOICES))
    out["OPTIMIZER"] = f'"{rng.choice(OPTIMIZER_CHOICES)}"'
    # Fixed at -1 (auto). Exploring BATCH would just burn rounds on OOMs.
    out["BATCH"] = "-1"

    for k in ("MOSAIC", "MIXUP", "COPY_PASTE", "ERASING",
              "FLIPLR", "FLIPUD", "SCALE", "DEGREES"):
        lo, hi = BOUNDS[k]
        out[k] = f"{rng.uniform(lo, hi):.3f}"

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--seed",  type=int, required=True)
    args = ap.parse_args()

    if args.round < 1:
        print(f"[baseline_policy] --round must be >= 1 (got {args.round})", file=sys.stderr)
        return 1

    if args.round == 1:
        # No edits — train.sh as scaffolded already encodes the no-tuning
        # defaults. The orchestrator's sed loop becomes a no-op.
        print("#policy=defaults")
        return 0

    print("#policy=random-search")
    for k, v in sample(args.round, args.seed).items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
