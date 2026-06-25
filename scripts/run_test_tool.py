#!/usr/bin/env python3
"""LeetCode-style test-submit tool. Strict-heldout mode (--strict-heldout)
is the only mode this is meant for.

The agent invokes this via the Bash tool. The script:

  1. Looks up the latest successful training_metrics event in events.jsonl
     to find the current ROUND and the latest run_name (with its best.pt).
  2. Refuses (exit 1) if a `test_tool_query` event already exists at this
     ROUND — one peek per round, like a single LeetCode submission per
     code revision.
  3. Runs `model.val(data=dataset.eval.yaml, split="test", verbose=False,
     plots=False)` on the latest best.pt.
  4. Emits a `test-tool-query` event recording the round + score. That
     event is firewalled out of every prompt (build_prompt.py's
     AGENT_INVISIBLE_EVENT_TYPES) so the score never re-enters the
     agent's view in a subsequent round.
  5. Prints ONLY one line to stdout:
         mAP50=<X.XXXX> mAP50-95=<X.XXXX> images=<N>
     No per-class, no confusion matrix, no file paths, no class names.

That single line is what the agent sees as the tool's response. Everything
else (per-class, confusion, image paths) stays in the operator's view via
events.jsonl + the eventual training_report.md — never in any prompt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import event  # noqa: E402
from run_test_eval import parse_yaml_path_test, count_test_images  # noqa: E402


def _read_events(project: Path) -> list[dict]:
    p = project / "events.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _latest_training_metrics(events: list[dict]) -> dict | None:
    runs = sorted(
        (e for e in events if e.get("type") == "training_metrics"),
        key=lambda e: e.get("ts", ""),
    )
    return runs[-1] if runs else None


def _query_already_used(events: list[dict], round_num: int) -> bool:
    """True iff a test_tool_query event has already fired for this round."""
    for e in events:
        if e.get("type") == "test_tool_query" and e.get("round") == round_num:
            return True
    return False


def _resolve_eval_yaml(project: Path, dataset_name: str | None = None) -> Path | None:
    """Return the path to dataset.eval.yaml.

    Looks in two places, in order:
      1. <framework_root>/datasets/<name>/dataset.eval.yaml   (Strict-heldout layout)
      2. <project>/dataset.eval.yaml                          (in-project override)
    """
    framework_root = SCRIPT_DIR.parent
    # First try the standard location: framework_root/datasets/<name>/...
    # `name` defaults to the project directory's name (matches new_project.sh's
    # convention where projects/<name> shares its name with datasets/<name>).
    name = dataset_name or project.name
    candidate = framework_root / "datasets" / name / "dataset.eval.yaml"
    if candidate.is_file():
        return candidate
    # Fallback — operator may have placed a per-project override
    candidate = project / "dataset.eval.yaml"
    if candidate.is_file():
        return candidate
    return None


def _resolve_best_pt(latest_tm: dict, framework_root: Path, task: str) -> Path | None:
    """Return path to weights/best.pt for the latest training run."""
    run_name = latest_tm.get("run_name")
    if not run_name:
        return None
    candidate = framework_root / "runs" / task / run_name / "weights" / "best.pt"
    return candidate if candidate.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, required=True,
                    help="project directory (events.jsonl + .heldout_strict live here)")
    ap.add_argument("--task", default=None,
                    help="task (default: read from latest training_metrics event)")
    ap.add_argument("--data", type=Path, default=None,
                    help="explicit dataset.eval.yaml path (default: auto-resolve)")
    args = ap.parse_args()

    if not args.project.is_dir():
        print(f"ERROR: project dir not found: {args.project}", file=sys.stderr)
        return 1

    # Strict-heldout marker is a hard prerequisite. The tool is meaningful
    # only in strict mode — in non-strict mode the agent already has full
    # access to the test split, so a "submit" tool would be pointless and
    # potentially leaking.
    if not (args.project / ".heldout_strict").is_file():
        print(
            "ERROR: this tool only works in --strict-heldout projects. "
            "The marker .heldout_strict is missing.",
            file=sys.stderr,
        )
        return 1

    events = _read_events(args.project)
    latest_tm = _latest_training_metrics(events)
    if not latest_tm:
        print(
            "ERROR: no successful training_metrics events yet — "
            "cannot submit before the first training round completes.",
            file=sys.stderr,
        )
        return 1

    round_num = int(latest_tm.get("round", 0) or 0)
    if _query_already_used(events, round_num):
        print(
            f"ERROR: test-tool already queried this round (round={round_num}). "
            "One submission per round — wait until the next training round.",
            file=sys.stderr,
        )
        return 1

    # Resolve task: argument wins, otherwise infer from training_metrics name
    # (e.g. "mAP50(B)" → detect/obb; "accuracy_top1" → classify; etc).
    task = args.task
    if task is None:
        bmn = str(latest_tm.get("best_metric_name", ""))
        if "(M)" in bmn:
            task = "segment"
        elif "(P)" in bmn:
            task = "pose"
        elif "accuracy" in bmn:
            task = "classify"
        else:
            task = "detect"

    framework_root = SCRIPT_DIR.parent
    eval_yaml = args.data or _resolve_eval_yaml(args.project)
    if eval_yaml is None or not eval_yaml.is_file():
        print(
            "ERROR: dataset.eval.yaml not found. "
            "--strict-heldout projects should have one at "
            "datasets/<name>/dataset.eval.yaml.",
            file=sys.stderr,
        )
        return 1

    best_pt = _resolve_best_pt(latest_tm, framework_root, task)
    if best_pt is None:
        print(
            f"ERROR: best.pt for run {latest_tm.get('run_name')!r} not found",
            file=sys.stderr,
        )
        return 1

    _, test_rel = parse_yaml_path_test(eval_yaml)
    if test_rel is None:
        print("ERROR: dataset.eval.yaml has no `test:` key", file=sys.stderr)
        return 1

    n_test = count_test_images(eval_yaml)
    if n_test == 0:
        print("ERROR: test split is empty", file=sys.stderr)
        return 1

    # Run val. Output goes to a sibling dir so we don't pollute the run.
    runs_task_dir = best_pt.parent.parent.parent
    val_name = best_pt.parent.parent.name + "_test_tool"
    try:
        from ultralytics import YOLO
        model = YOLO(str(best_pt))
        results = model.val(
            data=str(eval_yaml),
            split="test",
            project=str(runs_task_dir),
            name=val_name,
            exist_ok=True,
            verbose=False,
            plots=False,
            save_json=False,
        )
    except Exception as e:
        print(f"ERROR: val() failed: {e}", file=sys.stderr)
        return 1

    # Pull out only the headline numbers — nothing else reaches stdout.
    box = getattr(results, "box", None)
    if box is None:
        print("ERROR: val() returned no box metrics (model has no detections?)",
              file=sys.stderr)
        return 1
    map50 = float(getattr(box, "map50", 0.0) or 0.0)
    map_overall = float(getattr(box, "map", 0.0) or 0.0)

    # Emit the audit event BEFORE printing stdout — that way even if the
    # agent's Bash tool somehow truncates the output, the operator still
    # has the score on record.
    primary_name = event.TASK_PRIMARY.get(task, "mAP50(B)")
    event.emit_event(args.project, "test-tool-query", {
        "round":             round_num,
        "run_name":          str(latest_tm.get("run_name", "")),
        "best_metric_name":  primary_name,
        "best_metric_value": round(map50, 6),
    })

    # The ONLY line the agent sees. No per-class, no class names, no paths.
    print(f"mAP50={map50:.4f} mAP50-95={map_overall:.4f} images={n_test}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
