#!/usr/bin/env python3
"""Extract per-class metrics + confusion matrix from a completed run.

Why this exists:
  results.csv only carries overall P/R/mAP. The information that lets the
  agent reason about WHICH class is dragging down the metric — per-class
  P/R/mAP and the confusion matrix — only lives in the ultralytics
  `DetMetrics` object returned by `model.val()`. We re-run val once on the
  saved `weights/best.pt` to surface that data as a per-class-metrics
  event, then build_prompt.py + generate_report.py consume it.

  Re-running val is cheap (single forward pass over the val split, no
  backward pass) and runs once per round AFTER the long training has
  finished, so the cost is negligible relative to training itself.

CLI shape mirrors `event.py extract-metrics`:
    python3 per_class_metrics.py extract \\
        --project   <project_dir>       \\
        --run-dir   <runs/.../run_name> \\
        --data-yaml <dataset.yaml>      \\
        --task      detect              \\
        --round     <N>                 \\
        --run-name  <run_name>

On success: emits a `per-class-metrics` event into the project's
events.jsonl and prints a one-line summary to stdout.

On failure (no best.pt, val crashes, etc.): prints a warning to stderr and
returns 1. NEVER halts the chain — per-class extraction is enrichment,
not a trust boundary.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars / arrays to plain Python so json.dumps works."""
    try:
        import numpy as np
    except ImportError:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def extract_from_run(
    run_dir: Path,
    data_yaml: Path,
    task: str,
) -> dict | None:
    """Return {per_class, confusion, class_names} or None on failure.

    Runs `model.val(data=…, split='val', verbose=False)` on the run's
    `weights/best.pt`. Per-class arrays are pulled from `DetMetrics.box`
    (precision, recall, mAP50, mAP50-95) keyed by the class names from
    `results.names`.

    confusion is the full square matrix WITHOUT the background row/col
    that ultralytics appends — the trailing row/col are dropped because
    class_names indexes 0..nc-1 and the agent prompt explains things in
    terms of class-vs-class confusion.
    """
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[per_class_metrics] WARN: no best.pt at {best_pt}", file=sys.stderr)
        return None
    if not data_yaml.exists():
        print(f"[per_class_metrics] WARN: data.yaml not found at {data_yaml}",
              file=sys.stderr)
        return None
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[per_class_metrics] WARN: ultralytics not available: {e}",
              file=sys.stderr)
        return None

    try:
        model = YOLO(str(best_pt))
        results = model.val(
            data=str(data_yaml),
            split="val",
            verbose=False,
            plots=False,
            save_json=False,
        )
    except Exception as e:
        print(f"[per_class_metrics] WARN: val() failed: {e}", file=sys.stderr)
        return None

    # results.names is dict[int, str] {0: "class0", ...}; convert to list in
    # index order so the confusion matrix indices line up.
    names_dict = getattr(results, "names", None) or {}
    if not names_dict:
        print("[per_class_metrics] WARN: results.names empty — model has no classes?",
              file=sys.stderr)
        return None
    nc = max(names_dict.keys()) + 1
    class_names = [str(names_dict.get(i, f"class_{i}")) for i in range(nc)]

    # Per-class metrics via DetMetrics.class_result(i) → (p, r, ap50, ap50_95).
    # Falls back to zeros for classes the model never produced detections for.
    box = getattr(results, "box", None)
    ap_class_index = getattr(box, "ap_class_index", []) if box is not None else []
    ap_class_index = list(_to_jsonable(ap_class_index))

    # Per-class support: ultralytics exposes `nt_per_class` on the DetMetrics
    # instance (number of GT targets per class index that appeared in val).
    nt_per_class = _to_jsonable(getattr(results, "nt_per_class", []))
    if not isinstance(nt_per_class, list):
        nt_per_class = []

    per_class: dict[str, dict] = {}
    for i, name in enumerate(class_names):
        support = int(nt_per_class[i]) if i < len(nt_per_class) else 0
        if box is not None and i in ap_class_index:
            local_idx = ap_class_index.index(i)
            try:
                p, r, ap50, ap = box.class_result(local_idx)
                per_class[name] = {
                    "P": float(p),
                    "R": float(r),
                    "mAP50": float(ap50),
                    "mAP50_95": float(ap),
                    "support": support,
                }
                continue
            except Exception:
                pass
        per_class[name] = {
            "P": 0.0, "R": 0.0, "mAP50": 0.0, "mAP50_95": 0.0,
            "support": support,
        }

    # Confusion matrix. Ultralytics appends a background row/col → matrix is
    # (nc+1, nc+1). We drop the trailing row/col so indices match class_names.
    cm = getattr(results, "confusion_matrix", None)
    confusion_list: list[list[int]] = []
    if cm is not None and getattr(cm, "matrix", None) is not None:
        full = _to_jsonable(cm.matrix)
        if isinstance(full, list) and full:
            # Ultralytics convention: rows = predicted, cols = true. Transpose
            # so rows = true, cols = predicted (matches the prompt's wording).
            try:
                rows = len(full)
                cols = len(full[0]) if full[0] else 0
                # Drop background by slicing down to nc × nc.
                trimmed_rows = min(rows, nc)
                trimmed_cols = min(cols, nc)
                transposed = [
                    [int(full[j][i]) for j in range(trimmed_rows)]
                    for i in range(trimmed_cols)
                ]
                confusion_list = transposed
            except Exception:
                confusion_list = []

    return {
        "per_class": per_class,
        "confusion": confusion_list,
        "class_names": class_names,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Extract per-class metrics from a YOLO run; emit a "
                    "per-class-metrics event into the project's events.jsonl."
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="extract + emit")
    ex.add_argument("--project",   required=True, type=Path)
    ex.add_argument("--run-dir",   required=True, type=Path)
    ex.add_argument("--data-yaml", required=True, type=Path)
    ex.add_argument("--task",      required=True)
    ex.add_argument("--round",     dest="round_num", required=True, type=int)
    ex.add_argument("--run-name",  required=True)
    args = p.parse_args()

    if args.cmd == "extract":
        data = extract_from_run(args.run_dir, args.data_yaml, args.task)
        if data is None:
            return 1

        # Import here so a missing event.py doesn't break a pure-extraction
        # smoke test (`python3 per_class_metrics.py extract` is still useful
        # for one-off inspection).
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from event import emit_event

        payload = {
            "round":      args.round_num,
            "run_name":   args.run_name,
            "per_class":   data["per_class"],
            "confusion":   data["confusion"],
            "class_names": data["class_names"],
        }
        emit_event(args.project, "per-class-metrics", payload)
        worst = min(
            data["per_class"].items(),
            key=lambda kv: kv[1]["mAP50"],
        )
        print(
            f"[per_class] emitted per_class_metrics: {len(data['per_class'])} classes, "
            f"worst = {worst[0]} (mAP50={worst[1]['mAP50']:.4f})"
        )
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
