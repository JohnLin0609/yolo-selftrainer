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


def detect_mode(project_dir: Path, events: list[dict]) -> tuple[str, str]:
    """Return (mode, detail) where mode ∈ {claude, agent, baseline}.

    `detail` is a short suffix shown in the report header (e.g.
    "random-search, seed=42" for baseline). Empty string for other modes.

    Detection is via the scaffolded orchestrator file, NOT by scanning
    events — a session with zero successful rounds still has a mode.
    """
    if (project_dir / "start_baseline.sh").exists():
        # Pull policy + seed from the most recent baseline-decision event.
        for ev in reversed(events):
            if ev.get("type") == "baseline_decision":
                policy = ev.get("policy", "?")
                seed = ev.get("seed", "?")
                return "baseline", f"{policy}, seed={seed}"
        return "baseline", "no decisions yet"
    if (project_dir / "start_agent.sh").exists():
        return "agent", ""
    return "claude", ""


def _hms(seconds: float) -> str:
    """H:MM:SS for the loop-cost table. Floors negatives to 0:00:00."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def render_loop_cost(events: list[dict], mode: str, max_rounds: int | None) -> str:
    """Mode-agnostic cost summary. Wall time comes from training_finished
    events; LLM cost is "$0.00 (no agent)" for baseline and "n/a (not
    tracked yet)" for claude/agent until cost parsing lands.
    """
    durations = [
        float(e.get("duration_sec") or 0.0)
        for e in events
        if e.get("type") == "training_finished"
    ]
    total_sec = sum(durations)
    n_runs = len(durations)
    avg_sec = (total_sec / n_runs) if n_runs else 0.0

    if mode == "baseline":
        llm_cost = "$0.00 (no agent)"
    else:
        llm_cost = "n/a (not tracked yet)"

    rounds_cell = f"{n_runs}" + (f" / {max_rounds}" if max_rounds else "")

    return "\n".join([
        "## Loop cost",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| LLM cost (USD) | {llm_cost} |",
        f"| Total training wall time | {_hms(total_sec)} |",
        f"| Avg per round | {_hms(avg_sec)} |",
        f"| Rounds completed | {rounds_cell} |",
    ])


def test_runs_by_name(events: list[dict]) -> dict[str, dict]:
    """Index test_metrics by run_name for joining with val (training) rows.

    This is OPERATOR-ONLY data — never feeds into prompts. The report
    surfaces it so the human can compare val vs test progression and
    spot val-overfitting that the agent (which never sees test) could
    not catch.
    """
    out: dict[str, dict] = {}
    for e in events:
        if e.get("type") != "test_metrics":
            continue
        rn = e.get("run_name")
        if rn:
            # If duplicates (re-run test eval), keep the latest by ts.
            existing = out.get(rn)
            if existing is None or e.get("ts", "") > existing.get("ts", ""):
                out[rn] = e
    return out


def _test_extras_value(test_event: dict | None, metric_key_substring: str):
    """Pull a single metric out of a test event's extras dict by substring.

    Returns None if no event, no extras, or no matching key.
    """
    if not test_event:
        return None
    extras = test_event.get("extras") or {}
    for k, v in extras.items():
        if metric_key_substring in k and isinstance(v, (int, float)):
            return v
    return None


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


def render_metrics_table(runs: list[dict], test_by_name: dict[str, dict] | None = None) -> str:
    """Per-round table of ALL evaluation metrics at the best epoch.

    When `test_by_name` (test_metrics indexed by run_name) is non-empty,
    appends three columns: test_mAP50, test_mAP50-95, Δ(val−test). The
    Δ column makes val-overfitting visible — large positive Δ means the
    model does much better on val than test, which is the canonical
    "val noise the agent fit to" failure mode.
    """
    if not runs:
        return ""
    test_by_name = test_by_name or {}
    has_test = bool(test_by_name)

    # Union of extras keys, preserving first-seen order
    keys: list[str] = []
    for r in runs:
        for k in (r.get("extras") or {}):
            if k not in keys:
                keys.append(k)
    labels = [LABEL_MAP.get(k, k) for k in keys]
    header = ["#", "run_name", "ep", "Δ_primary"] + labels
    if has_test:
        header += ["test_mAP50", "test_mAP50-95", "Δ(val−test)"]

    lines = [
        "## Per-round metrics at best epoch",
        "",
        "Every value is at the epoch where the primary metric peaked.",
        "`Δ_primary` shows the change in primary metric vs the previous run.",
        "`gap` = val_box_loss − train_box_loss (positive = overfitting).",
    ]
    if has_test:
        lines.append(
            "`Δ(val−test)` = val mAP50 − test mAP50. Large + means val is "
            "easier than the held-out set — agent may be fitting val noise."
        )
    lines += [
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
        if has_test:
            t = test_by_name.get(run_name)
            t_map50 = _test_extras_value(t, "mAP50(B)")
            # Distinguish mAP50 vs mAP50-95: the substring "mAP50-95" wins
            # on the longer match. Pull both explicitly.
            t_map50_95 = None
            if t is not None:
                for k, v in (t.get("extras") or {}).items():
                    if "mAP50-95" in k and isinstance(v, (int, float)):
                        t_map50_95 = v
                        break
                # And restrict t_map50 to the non-95 variant.
                for k, v in (t.get("extras") or {}).items():
                    if k.endswith("mAP50(B)") and isinstance(v, (int, float)):
                        t_map50 = v
                        break
            val_map50 = None
            for k, v in extras.items():
                if k.endswith("mAP50(B)") and isinstance(v, (int, float)):
                    val_map50 = v
                    break
            if val_map50 is not None and t_map50 is not None:
                diff = val_map50 - t_map50
                diff_str = f"{diff:+.4f}"
            else:
                diff_str = ""
            cells.append(fmt(t_map50) if t_map50 is not None else "")
            cells.append(fmt(t_map50_95) if t_map50_95 is not None else "")
            cells.append(diff_str)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_progression_bar_chart(runs: list[dict], test_by_name: dict[str, dict] | None = None) -> str:
    """ASCII bar chart of primary-metric progression across rounds.

    When test_by_name is non-empty, draws two bars per round:
      val  (filled █) — what the agent saw and optimized against
      test (open ░)   — what the agent never saw; the unbiased benchmark
    Divergence between the two bars at a glance flags val-overfitting.
    """
    if not runs:
        return ""
    test_by_name = test_by_name or {}
    values = [(r.get("round", i + 1),
               r.get("best_metric_value", 0.0) or 0.0,
               r.get("run_name", ""))
              for i, r in enumerate(runs)]
    # Normalize bars to the global peak across both val and test so the visual
    # comparison is honest (can't make test bars look longer by scaling alone).
    peak = max(v for _, v, _ in values)
    for _, _, rn in values:
        t = test_by_name.get(rn)
        if t:
            peak = max(peak, t.get("best_metric_value", 0.0) or 0.0)
    if peak == 0:
        peak = 1.0
    best_round = max(values, key=lambda x: x[1])[0]

    lines = ["## Primary metric progression", "", "```"]
    has_test = bool(test_by_name)
    for round_n, v, rn in values:
        bar_len = int(round((v / peak) * 50))
        marker = "  ← best (val)" if round_n == best_round else ""
        if has_test:
            lines.append(f"  Round {round_n:>2} val  | {'█' * bar_len} {fmt(v)}{marker}")
            t = test_by_name.get(rn)
            t_v = (t.get("best_metric_value", 0.0) or 0.0) if t else None
            if t_v is not None:
                t_bar = int(round((t_v / peak) * 50))
                lines.append(f"           test | {'░' * t_bar} {fmt(t_v)}")
            else:
                lines.append(f"           test | (no test eval for this run)")
        else:
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


def render_insights(runs: list[dict], test_by_name: dict[str, dict] | None = None) -> str:
    """Quick high-level observations the operator might miss."""
    if not runs:
        return ""
    test_by_name = test_by_name or {}
    first_v = runs[0].get("best_metric_value", 0.0) or 0.0
    best_v  = max(r.get("best_metric_value", 0.0) or 0.0 for r in runs)
    pct_gain = ((best_v - first_v) / first_v * 100) if first_v else 0.0

    lines = ["## Insights", ""]
    lines.append(f"- Total improvement: {fmt(first_v)} → {fmt(best_v)} "
                 f"({best_v - first_v:+.4f}, {pct_gain:+.1f}%)")

    # Overfitting watch (train↔val gap)
    last_gaps = [
        (r.get("round"), (r.get("extras") or {}).get("overfit_gap_box"))
        for r in runs[-3:]
        if (r.get("extras") or {}).get("overfit_gap_box") is not None
    ]
    if last_gaps:
        gaps = ", ".join(f"r{n}={fmt(g, 3)}" for n, g in last_gaps)
        lines.append(f"- Overfit gap (val_box−train_box) last {len(last_gaps)} runs: {gaps}")

    # Val↔test divergence (only shown when test eval ran)
    if test_by_name:
        divergences = []
        for r in runs:
            t = test_by_name.get(r.get("run_name", ""))
            if t is None:
                continue
            v_map = r.get("best_metric_value", 0.0) or 0.0
            t_map = t.get("best_metric_value", 0.0) or 0.0
            divergences.append((r.get("round"), v_map - t_map))
        if divergences:
            last_div = ", ".join(f"r{n}={d:+.3f}" for n, d in divergences[-3:])
            lines.append(
                f"- Val−test divergence last {min(len(divergences),3)} runs: {last_div}"
            )
            # Flag growing divergence — the canonical val-overfitting signal.
            if len(divergences) >= 2:
                trend = divergences[-1][1] - divergences[0][1]
                if trend > 0.02:
                    lines.append(
                        f"  ⚠️ divergence grew by {trend:+.3f} across the session — "
                        f"agent may be fitting val noise (val improves faster than test)"
                    )
    return "\n".join(lines)


def render_per_class_weaknesses(events: list[dict], project_dir: Path) -> str:
    """Per-class weakness section + agent's data-layer recommendations.

    Reads the most recent per_class_metrics event, ranks weak classes via
    diagnose_weak_classes, and reproduces (read-only) any
    `## Data-layer recommendations` block the agent wrote in
    next_instruction.md. The human reader sees both: the machine ranking
    and the agent's prose interpretation.
    """
    per_class_events = sorted(
        (e for e in events if e.get("type") == "per_class_metrics"),
        key=lambda e: e.get("ts", ""),
    )
    if not per_class_events:
        return ""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from diagnose_classes import diagnose_weak_classes
    except ImportError:
        return ""

    latest = per_class_events[-1]
    prior = per_class_events[:-1]
    per_class = latest.get("per_class") or {}
    confusion = latest.get("confusion") or []
    class_names = latest.get("class_names") or []
    if not per_class:
        return ""

    # Lightweight history for persistence detection — mirror build_prompt.py
    history: list[dict] = []
    for ev in prior:
        pc = ev.get("per_class") or {}
        if not pc:
            history.append({"worst": []})
            continue
        ranked = sorted(
            pc.items(),
            key=lambda kv: (
                float((kv[1] or {}).get("mAP50", 0.0) or 0.0),
                int((kv[1] or {}).get("support", 0) or 0),
                kv[0],
            ),
        )
        history.append({"worst": [{"class": ranked[0][0]}]})

    d = diagnose_weak_classes(
        per_class, confusion, class_names,
        history=history, persistent_n=3, metric="mAP50",
    )

    lines = ["## Per-class weaknesses"]
    lines.append(
        f"_Latest run: `{latest.get('run_name', '?')}` (round {latest.get('round', '?')})_"
    )
    lines.append("")
    if d["worst"]:
        lines.append("| Rank | Class | mAP50 | Support | Persistent |")
        lines.append("|---|---|---|---|---|")
        for i, w in enumerate(d["worst"], 1):
            mark = "**yes**" if w.get("persistent") else "no"
            lines.append(
                f"| {i} | `{w['class']}` | {fmt(w['score'])} | {w['support']} | {mark} |"
            )
    if d["confused_pairs"]:
        lines.append("")
        lines.append("**Most-confused pairs** (true → predicted):")
        for p in d["confused_pairs"]:
            lines.append(f"- `{p['a']}` → `{p['b']}`: {p['count']}")
    if d["recommend_data_review"]:
        flagged = ", ".join(f"`{c}`" for c in d["recommend_data_review"])
        lines.append("")
        lines.append("### ⚠️ Persistent weakness flagged")
        lines.append(
            f"Class(es) {flagged} have been worst for ≥3 consecutive rounds. "
            "Likely root cause is in the dataset (labeling noise, sample count, "
            "or augmentation fit). Below: the agent's data-layer recommendations "
            "from `next_instruction.md` (read-only — agent never modifies datasets/)."
        )

    # Pull the agent's `## Data-layer recommendations` block verbatim, if any.
    ni = project_dir / "next_instruction.md"
    if ni.exists():
        text = ni.read_text()
        marker = "## Data-layer recommendations"
        idx = text.find(marker)
        if idx >= 0:
            tail = text[idx:]
            # Cut at the next level-2 heading
            next_h = tail.find("\n## ", 1)
            block = tail if next_h < 0 else tail[:next_h]
            lines.append("")
            lines.append(block.rstrip())

    return "\n".join(lines)


def _read_max_rounds(project_dir: Path) -> int | None:
    """Look up MAX_ROUNDS from whichever start_*.sh exists. Returns None if
    not parseable — the Loop-cost table just omits the denominator then.
    """
    for fname in ("start_baseline.sh", "start_agent.sh", "start_claude.sh"):
        f = project_dir / fname
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if line.startswith("MAX_ROUNDS="):
                try:
                    return int(line.split("=", 1)[1].split()[0])
                except (ValueError, IndexError):
                    return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, required=True)
    ap.add_argument("--runs-dir",    type=Path, required=True)
    ap.add_argument("--task",        required=True, choices=list(TASK_PRIMARY_METRIC))
    ap.add_argument("--output",      type=Path, required=True)
    args = ap.parse_args()

    events = read_events(args.project_dir)
    runs   = metric_runs(events)
    test_by_name = test_runs_by_name(events)
    mode, mode_detail = detect_mode(args.project_dir, events)
    max_rounds = _read_max_rounds(args.project_dir)

    mode_label = f"{mode} ({mode_detail})" if mode_detail else mode

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "# Training Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Project**: `{args.project_dir.name}`",
        f"**Mode**: {mode_label}",
        f"**Task**: {args.task}",
        f"**Primary metric**: {TASK_PRIMARY_METRIC[args.task]}",
        f"**Total runs in this project**: {len(runs)}",
    ]
    if test_by_name:
        report_lines.append(
            f"**Held-out test evals**: {len(test_by_name)} (agent never saw these)"
        )
    report_lines.append("")

    report_lines.append(render_summary(runs, args.task, args.runs_dir))
    report_lines.append("")
    report_lines.append(render_loop_cost(events, mode, max_rounds))
    report_lines.append("")
    report_lines.append(render_metrics_table(runs, test_by_name))
    report_lines.append("")
    per_class_block = render_per_class_weaknesses(events, args.project_dir)
    if per_class_block:
        report_lines.append(per_class_block)
        report_lines.append("")
    report_lines.append(render_progression_bar_chart(runs, test_by_name))
    report_lines.append("")
    report_lines.append(render_run_history_section(runs, args.runs_dir))
    report_lines.append("")
    report_lines.append(render_insights(runs, test_by_name))

    args.output.write_text("\n".join(report_lines) + "\n")
    print(f"[generate_report] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
