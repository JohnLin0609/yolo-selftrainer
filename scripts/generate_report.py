#!/usr/bin/env python3
"""Generate a training session report — project-scoped via events.jsonl.

Why this rewrite (P7):
  The old generate_report.py scanned `runs/<task>/*` directly and pulled
  every directory in there. With multiple projects sharing one runs/ folder,
  that meant a final report mixed runs from different projects. Now we read
  the project's events.jsonl (which only contains events for THIS project's
  runs) as the source of truth, then read each run's results.csv for the
  detailed numbers.

Usage:
    python generate_report.py --project-dir /path/to/projects/name \
                              --runs-dir   /path/to/runs/task \
                              --task       detect \
                              --output     /path/to/projects/name/training_report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path


# Same per-task primary metric mapping used in event.py — name-based, not
# index-based, so it survives column reordering between ultralytics versions.
TASK_PRIMARY_METRIC = {
    "detect":   "mAP50(B)",
    "obb":      "mAP50(B)",
    "segment":  "mAP50(M)",
    "pose":     "mAP50(P)",
    "classify": "accuracy_top1",
}
TASK_SECONDARY_METRIC = {
    "detect":   "mAP50-95(B)",
    "obb":      "mAP50-95(B)",
    "segment":  "mAP50-95(M)",
    "pose":     "mAP50-95(P)",
    "classify": "accuracy_top5",
}

# Pretty short labels for the per-round metrics table
LABEL_MAP = {
    "metrics/precision(B)": "P",
    "metrics/recall(B)":    "R",
    "metrics/mAP50(B)":     "mAP50",
    "metrics/mAP50-95(B)":  "mAP50-95",
    "train/box_loss":       "tr_box",
    "train/cls_loss":       "tr_cls",
    "train/dfl_loss":       "tr_dfl",
    "val/box_loss":         "val_box",
    "val/cls_loss":         "val_cls",
    "val/dfl_loss":         "val_dfl",
    "overfit_gap_box":      "gap",
}


def read_events(project_dir: Path) -> list[dict]:
    p = project_dir / "events.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # Match event.py's lenient skip + warn behavior
            print(f"WARN: unparseable events.jsonl line: {line[:80]!r}", file=sys.stderr)
    return out


def metric_runs(events: list[dict]) -> list[dict]:
    return sorted(
        (e for e in events if e.get("type") == "training_metrics"),
        key=lambda e: e.get("ts", ""),
    )


def read_args_yaml(run_dir: Path) -> dict[str, str]:
    """Minimal YAML-key reader for args.yaml (no pyyaml dependency)."""
    out: dict[str, str] = {}
    f = run_dir / "args.yaml"
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        if ":" not in line or line.startswith("#"):
            continue
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (ValueError, TypeError):
        return str(v)


def render_summary(runs: list[dict], task: str, runs_dir: Path) -> str:
    """Top-of-report best-run summary."""
    if not runs:
        return "_No completed training runs in events.jsonl yet._"

    best = max(runs, key=lambda r: float(r.get("best_metric_value", -1) or -1))
    best_name = best.get("run_name", "?")
    best_dir = runs_dir / best_name
    args_y = read_args_yaml(best_dir)
    primary = TASK_PRIMARY_METRIC.get(task, "?")
    secondary = TASK_SECONDARY_METRIC.get(task, "?")
    extras = best.get("extras") or {}

    sec_key = next((k for k in extras if secondary in k), None)
    sec_val = extras.get(sec_key) if sec_key else None

    lines = [
        "## Best Model",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| **{primary}** | **{fmt(best.get('best_metric_value'))}** |",
        f"| {secondary} | {fmt(sec_val)} |",
        f"| Best epoch | {best.get('best_epoch')} |",
        f"| Run | `{best_name}` |",
        f"| Weights | `{best_dir}/weights/best.pt` |",
    ]
    for key in ("imgsz", "lr0", "optimizer", "epochs", "patience"):
        if key in args_y:
            lines.append(f"| {key} | {args_y[key]} |")
    return "\n".join(lines)


def render_metrics_table(runs: list[dict]) -> str:
    """Per-round table of ALL evaluation metrics at the best epoch."""
    if not runs:
        return ""

    # Union of extras keys, preserving first-seen order
    keys: list[str] = []
    for r in runs:
        for k in (r.get("extras") or {}):
            if k not in keys:
                keys.append(k)
    labels = [LABEL_MAP.get(k, k) for k in keys]
    header = ["#", "run_name", "ep", "Δ_primary"] + labels

    lines = [
        "## Per-round metrics at best epoch",
        "",
        "Every value is at the epoch where the primary metric peaked.",
        "`Δ_primary` shows the change in primary metric vs the previous run.",
        "`gap` = val_box_loss − train_box_loss (positive = overfitting).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]

    prev_primary = None
    for r in runs:
        primary = r.get("best_metric_value", 0.0) or 0.0
        delta = "—" if prev_primary is None else f"{primary - prev_primary:+.4f}"
        prev_primary = primary
        run_name = r.get("run_name", "?")
        cells = [
            str(r.get("round", "?")),
            f"`{run_name}`",
            str(r.get("best_epoch", "?")),
            delta,
        ]
        extras = r.get("extras") or {}
        for k in keys:
            v = extras.get(k)
            cells.append(fmt(v) if isinstance(v, (int, float)) else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_progression_bar_chart(runs: list[dict]) -> str:
    """ASCII bar chart of primary-metric progression across rounds."""
    if not runs:
        return ""
    values = [(r.get("round", i + 1), r.get("best_metric_value", 0.0) or 0.0)
              for i, r in enumerate(runs)]
    best_v = max(v for _, v in values) or 1.0
    best_round = max(values, key=lambda x: x[1])[0]

    lines = ["## Primary metric progression", "", "```"]
    for round_n, v in values:
        bar_len = int(round((v / best_v) * 50))
        marker = "  ← best" if round_n == best_round else ""
        lines.append(f"  Round {round_n:>2} | {'█' * bar_len} {fmt(v)}{marker}")
    lines.append("```")
    return "\n".join(lines)


def render_run_history_section(runs: list[dict], runs_dir: Path) -> str:
    """One block per run with args.yaml hyperparameters."""
    if not runs:
        return ""
    lines = ["## All runs (in order)", ""]
    for r in runs:
        run_name = r.get("run_name", "?")
        run_dir = runs_dir / run_name
        args_y = read_args_yaml(run_dir)
        lines.append(f"### Run #{r.get('round', '?')} — `{run_name}`")
        lines.append(
            f"- Best **{r.get('best_metric_name', '?')}** = "
            f"**{fmt(r.get('best_metric_value'))}** at epoch {r.get('best_epoch')}"
        )
        lines.append(
            f"- Trained {r.get('final_epoch', '?')}/{r.get('total_epochs', '?')} epochs"
            + (" (patience triggered)" if r.get("patience_triggered") else "")
        )
        if args_y:
            kv = []
            for k in ("imgsz", "lr0", "lrf", "batch", "optimizer", "patience"):
                if k in args_y:
                    kv.append(f"{k}={args_y[k]}")
            if kv:
                lines.append(f"- Params: `{' '.join(kv)}`")
        lines.append("")
    return "\n".join(lines)


def render_insights(runs: list[dict]) -> str:
    """Quick high-level observations the operator might miss."""
    if not runs:
        return ""
    first_v = runs[0].get("best_metric_value", 0.0) or 0.0
    best_v  = max(r.get("best_metric_value", 0.0) or 0.0 for r in runs)
    pct_gain = ((best_v - first_v) / first_v * 100) if first_v else 0.0

    lines = ["## Insights", ""]
    lines.append(f"- Total improvement: {fmt(first_v)} → {fmt(best_v)} "
                 f"({best_v - first_v:+.4f}, {pct_gain:+.1f}%)")

    # Overfitting watch
    last_gaps = [
        (r.get("round"), (r.get("extras") or {}).get("overfit_gap_box"))
        for r in runs[-3:]
        if (r.get("extras") or {}).get("overfit_gap_box") is not None
    ]
    if last_gaps:
        gaps = ", ".join(f"r{n}={fmt(g, 3)}" for n, g in last_gaps)
        lines.append(f"- Overfit gap last {len(last_gaps)} runs: {gaps}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--runs-dir",    type=Path, required=True)
    ap.add_argument("--task",        required=True, choices=list(TASK_PRIMARY_METRIC))
    ap.add_argument("--output",      type=Path, required=True)
    args = ap.parse_args()

    events = read_events(args.project_dir)
    runs   = metric_runs(events)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "# Training Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Project**: `{args.project_dir.name}`",
        f"**Task**: {args.task}",
        f"**Primary metric**: {TASK_PRIMARY_METRIC[args.task]}",
        f"**Total runs in this project**: {len(runs)}",
        "",
    ]

    report_lines.append(render_summary(runs, args.task, args.runs_dir))
    report_lines.append("")
    report_lines.append(render_metrics_table(runs))
    report_lines.append("")
    report_lines.append(render_progression_bar_chart(runs))
    report_lines.append("")
    report_lines.append(render_run_history_section(runs, args.runs_dir))
    report_lines.append("")
    report_lines.append(render_insights(runs))

    args.output.write_text("\n".join(report_lines) + "\n")
    print(f"[generate_report] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
