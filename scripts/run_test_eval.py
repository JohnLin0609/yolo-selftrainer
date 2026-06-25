#!/usr/bin/env python3
"""Evaluate best.pt on the held-out test split and emit a test_metrics event.

This script is called from train.sh AFTER training_metrics has been emitted.
Its output is **operator-only** — the resulting test_metrics event is
firewalled out of every prompt path by build_prompt.py
(AGENT_INVISIBLE_EVENT_TYPES).

Why a separate script:
  Cleanly isolates the test-eval code path from the training pipeline.
  `yolo val` is run via the Python API rather than the CLI so we read
  metrics directly off the return value (no CSV parsing, no version skew).

Failure tolerance:
  Per Harness §二, we don't silently swallow errors. But this script is
  POST-HOC validation — its failure must NOT break the agent loop. train.sh
  wraps the call so a non-zero exit prints a warning and continues. Within
  the script we fail loud (raise + stderr); the wrapper catches it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the event registry + emit. Imports event.py from the same scripts/ dir.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import event  # noqa: E402


def parse_yaml_path_test(yaml_path: Path) -> tuple[str, str | None]:
    """Read `path:` and `test:` keys from an Ultralytics dataset.yaml.

    We don't use pyyaml here — simple line-based parsing is enough and
    keeps the helper standalone-runnable even without yaml installed.
    """
    p = None
    t = None
    for raw in yaml_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("path:"):
            p = line.split(":", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("test:"):
            t = line.split(":", 1)[1].strip().strip('"').strip("'")
    return p, t


def count_test_images(yaml_path: Path) -> int:
    base, test_rel = parse_yaml_path_test(yaml_path)
    if base is None or test_rel is None:
        return 0
    test_dir = Path(base) / test_rel
    if not test_dir.is_dir():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sum(1 for f in test_dir.iterdir() if f.suffix.lower() in exts)


def build_extras(results, task: str) -> dict:
    """Flatten ultralytics val results into a flat dict of metric → float.

    Mirrors the keys used by extract_metrics_from_run() in event.py so the
    report renderer can show val/test side-by-side with matching column names.
    """
    extras: dict = {}

    # Detect / OBB use results.box; segment also has results.seg; pose has
    # results.pose. For now we expose the box metrics for everything that has
    # them — classify is handled separately.
    box = getattr(results, "box", None)
    if box is not None:
        suffix = "(B)"
        extras[f"metrics/precision{suffix}"] = float(getattr(box, "mp", 0.0) or 0.0)
        extras[f"metrics/recall{suffix}"]    = float(getattr(box, "mr", 0.0) or 0.0)
        extras[f"metrics/mAP50{suffix}"]     = float(getattr(box, "map50", 0.0) or 0.0)
        extras[f"metrics/mAP50-95{suffix}"]  = float(getattr(box, "map", 0.0) or 0.0)
        # Per-class — keep them under a sub-key so they don't clutter the table
        # but are still recoverable for deeper inspection.
        names = getattr(results, "names", None) or {}
        per_class = {}
        # box.maps is a per-class mAP50-95 array; ap50 is per-class mAP50
        maps = getattr(box, "maps", None)
        ap50 = getattr(box, "ap50", None)
        if maps is not None and ap50 is not None:
            for i, name in (names.items() if isinstance(names, dict) else enumerate(names)):
                try:
                    per_class[str(name)] = {
                        "mAP50": float(ap50[i]),
                        "mAP50-95": float(maps[i]),
                    }
                except (IndexError, ValueError, TypeError):
                    continue
        if per_class:
            extras["per_class"] = per_class

    if task == "segment":
        seg = getattr(results, "seg", None)
        if seg is not None:
            extras["metrics/precision(M)"] = float(getattr(seg, "mp", 0.0) or 0.0)
            extras["metrics/recall(M)"]    = float(getattr(seg, "mr", 0.0) or 0.0)
            extras["metrics/mAP50(M)"]     = float(getattr(seg, "map50", 0.0) or 0.0)
            extras["metrics/mAP50-95(M)"]  = float(getattr(seg, "map", 0.0) or 0.0)
    elif task == "pose":
        pose = getattr(results, "pose", None)
        if pose is not None:
            extras["metrics/precision(P)"] = float(getattr(pose, "mp", 0.0) or 0.0)
            extras["metrics/recall(P)"]    = float(getattr(pose, "mr", 0.0) or 0.0)
            extras["metrics/mAP50(P)"]     = float(getattr(pose, "map50", 0.0) or 0.0)
            extras["metrics/mAP50-95(P)"]  = float(getattr(pose, "map", 0.0) or 0.0)
    elif task == "classify":
        top1 = getattr(results, "top1", None)
        top5 = getattr(results, "top5", None)
        if top1 is not None:
            extras["accuracy_top1"] = float(top1)
        if top5 is not None:
            extras["accuracy_top5"] = float(top5)

    return extras


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project",  type=Path, required=True, help="project directory (events.jsonl lives here)")
    ap.add_argument("--best-pt",  type=Path, required=True, help="path to best.pt")
    ap.add_argument("--data",     type=Path, required=True, help="dataset.yaml")
    ap.add_argument("--task",     required=True, choices=list(event.TASK_PRIMARY))
    ap.add_argument("--round",    dest="round_num", type=int, required=True)
    ap.add_argument("--run-name", required=True, help="training run name (the _test suffix is appended)")
    args = ap.parse_args()

    if not args.best_pt.exists():
        print(f"[test-eval] ERROR: best.pt not found: {args.best_pt}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"[test-eval] ERROR: dataset yaml not found: {args.data}", file=sys.stderr)
        return 1

    # Refuse to run if the yaml doesn't declare a test split — this prevents
    # accidentally calling `yolo val split=test` and silently falling back to
    # val under the hood.
    _, test_rel = parse_yaml_path_test(args.data)
    if test_rel is None:
        # Strict-heldout mode: dataset.yaml has no test: key (moved to
        # dataset.eval.yaml). Try the sibling file before giving up.
        sibling = args.data.parent / "dataset.eval.yaml"
        if sibling.is_file():
            print(f"[test-eval] dataset.yaml lacks `test:`; using {sibling.name} "
                  "(strict-heldout dual-yaml)", file=sys.stderr)
            args.data = sibling
            _, test_rel = parse_yaml_path_test(args.data)
        if test_rel is None:
            print(f"[test-eval] ERROR: dataset yaml has no `test:` key — refusing to run",
                  file=sys.stderr)
            return 1

    n_test = count_test_images(args.data)
    if n_test == 0:
        print(f"[test-eval] ERROR: test split is empty", file=sys.stderr)
        return 1

    # Output goes into runs/<task>/<run_name>_test/ — kept separate from the
    # training run dir so extract_metrics_from_run() never sees this results.csv.
    # exist_ok=True for idempotency (re-running test eval overwrites).
    runs_task_dir = args.best_pt.parent.parent.parent  # weights/ → run_name/ → task/
    val_name = args.best_pt.parent.parent.name + "_test"

    from ultralytics import YOLO
    print(f"[test-eval] loading {args.best_pt}")
    model = YOLO(str(args.best_pt))
    print(f"[test-eval] running val on split=test ({n_test} images)")
    results = model.val(
        data=str(args.data),
        split="test",
        project=str(runs_task_dir),
        name=val_name,
        exist_ok=True,
        verbose=False,
        plots=False,
        save_json=False,
    )

    extras = build_extras(results, args.task)

    primary_name = event.TASK_PRIMARY[args.task]
    # Look up the primary metric value in extras (key form: "metrics/<name>")
    primary_value = 0.0
    for k, v in extras.items():
        if isinstance(v, (int, float)) and primary_name in k:
            primary_value = float(v)
            break

    payload = {
        "round": args.round_num,
        "run_name": args.run_name,
        "test_split_size": n_test,
        "best_metric_name": primary_name,
        "best_metric_value": round(primary_value, 6),
        # best_epoch is the best.pt's epoch — we don't know it from val alone,
        # so leave -1 as a sentinel. The training_metrics event for the same
        # run_name carries the true best_epoch.
        "best_epoch": -1,
        "extras": extras,
    }
    event.emit_event(args.project, "test-metrics", payload)
    print(
        f"[test-eval] emitted test_metrics: {primary_name}={primary_value:.4f} "
        f"on {n_test} held-out images"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
