#!/usr/bin/env python3
"""Event log for the YOLO self-trainer (Harness §九).

Why this exists:
  state was scattered across single-file mutables (round_counter,
  train_completed, last_run_name, consecutive_failures) that overwrite in
  place. A crash mid-write loses history, and reconstructing what happened
  required reading every per-round log dir.

  events.jsonl is append-only, sortable by `ts`, and uniquely identifies
  every state transition. The legacy state files (round_counter, etc.) stay
  as a cache for fast bash access — events.jsonl is the source of truth and
  the cache can be rebuilt from it (see `query current-round`).

Schema:
  Every line is a single JSON object with `ts` (ISO8601) + `type` + payload.
  The grammar is enforced at emit time so a typo in event_type fails loud
  instead of silently writing an unparseable event.

Event types (use kebab-case on the CLI; underscore in the JSON):
  round-started        round, session_id
  training-started     round, run_name, params (JSON dict)
  training-finished    round, run_name, exit_code, duration_sec
  training-metrics     round, run_name, best_metric_name, best_metric_value,
                       best_epoch, final_epoch, total_epochs, patience_triggered
  claude-started       round, session_id
  claude-finished      round, exit_code, duration_sec
  validation-failed    round, violations (JSON list)
  preflight-failed     round, reason
  halted               reason, details (JSON dict)
  session-resumed      round, note
  test-metrics         round, run_name, test_split_size, best_metric_name,
                       best_metric_value, best_epoch, extras (JSON dict)
                       OPERATOR-ONLY — never consumed by build_prompt.py.
                       See scripts/build_prompt.py AGENT_INVISIBLE_EVENT_TYPES.
  baseline-decision    round, policy, seed, params (JSON dict)
                       Emitted by start_baseline.sh — records the LLM-free
                       policy's choice for the round. Lets generate_report.py
                       distinguish baseline runs from agent runs.
  plateau-detected     round, n, threshold, improvement, best_recent, best_before
                       Emitted by train.sh when q_plateau_status reports
                       state=warn. Agent-visible: build_prompt.py injects an
                       orthogonal-strategy nudge into the next prompt while
                       this warning is active. Cleared implicitly by a later
                       training_metrics that improves by ≥ threshold.
  per-class-metrics    round, run_name, per_class (JSON dict), confusion
                       (JSON list[list[int]]), class_names (JSON list[str])
                       Emitted by per_class_metrics.py after a successful
                       training run. Agent-visible: build_prompt.py surfaces
                       a worst-classes + confused-pairs summary and (when
                       persistent) a "data-layer recommendation" callout.

Why we don't fancy schema-validate the payload:
  Harness §四 "fail loud" applies to safety; for an audit log we'd rather
  always record an event than refuse to write one because of a missing field.
  Querying code defends against malformed records (see read_all).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# Set of accepted event types. Source of truth — every emitter passes through
# this set so an unrecognized type fails loud.
EVENT_TYPES = {
    "round-started",
    "training-started",
    "training-finished",
    "training-metrics",
    "claude-started",
    "claude-finished",
    "validation-failed",
    "preflight-failed",
    "halted",
    "session-resumed",
    # OPERATOR-ONLY — agent must never see this. See build_prompt.py.
    "test-metrics",
    # Emitted by start_baseline.sh when the LLM-free policy picks params.
    "baseline-decision",
    # Emitted by train.sh's plateau circuit; consumed by build_prompt.py to
    # inject the orthogonal-strategy nudge. Agent-visible by design.
    "plateau-detected",
    # Emitted by per_class_metrics.py after each successful train. Carries
    # per-class P/R/mAP + the confusion matrix. Agent-visible — build_prompt.py
    # surfaces a worst-classes + confused-pairs summary.
    "per-class-metrics",
}


def now_iso() -> str:
    """ISO8601 with local-zone offset, second precision (sortable lexically)."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def events_path(project: Path) -> Path:
    return project / "events.jsonl"


def emit_event(project: Path, event_type: str, payload: dict) -> None:
    """Append a single event line. Atomic per call (single write())."""
    if event_type not in EVENT_TYPES:
        raise ValueError(
            f"unknown event type: {event_type!r} (allowed: {sorted(EVENT_TYPES)})"
        )
    # Store the JSON-friendly form (underscores) so downstream tooling
    # doesn't have to normalize on every read.
    event = {"ts": now_iso(), "type": event_type.replace("-", "_"), **payload}
    p = events_path(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, separators=(",", ":"), default=str) + "\n"
    # 'a' mode + a single write keeps each event atomic at the kernel level
    # (Linux append writes ≤ PIPE_BUF are atomic). No flock needed because
    # train.sh and start_claude.sh never run concurrently.
    with p.open("a") as f:
        f.write(line)


def read_all(project: Path) -> list[dict]:
    """Read and parse every event. Skips (with a warning) malformed lines."""
    p = events_path(project)
    if not p.exists():
        return []
    events: list[dict] = []
    with p.open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                # Loud warn (Harness §二): a malformed line means the audit
                # trail is broken at this point. Don't silently drop it.
                print(
                    f"[event.py] WARN: events.jsonl line {n} unparseable ({e}): {line[:120]!r}",
                    file=sys.stderr,
                )
    return events


# ─── Query operations ───────────────────────────────────────────────

def q_current_round(project: Path) -> int:
    """Highest round number across all events that carry one.

    Includes session_resumed so projects migrated from the legacy state model
    (round_counter file only) report the resumed value rather than 0.
    """
    events = read_all(project)
    rounds = [
        e.get("round")
        for e in events
        if isinstance(e.get("round"), int)
    ]
    return max(rounds, default=0)


def q_last_metrics(project: Path):
    events = read_all(project)
    for e in reversed(events):
        if e.get("type") == "training_metrics":
            return e
    return None


def q_consecutive_failures(project: Path) -> int:
    """Count failures since the most recent successful training_finished."""
    events = read_all(project)
    n = 0
    for e in reversed(events):
        t = e.get("type")
        if t == "training_finished":
            if e.get("exit_code") == 0:
                return n
            n += 1
        elif t in ("validation_failed", "preflight_failed"):
            n += 1
    return n


def q_runs_history(project: Path) -> list[dict]:
    return [e for e in read_all(project) if e.get("type") == "training_metrics"]


PLATEAU_N_DEFAULT = 3
PLATEAU_THRESHOLD_DEFAULT = 0.005
PLATEAU_M_DEFAULT = 2


def q_plateau_status(
    project: Path,
    n: int = PLATEAU_N_DEFAULT,
    threshold: float = PLATEAU_THRESHOLD_DEFAULT,
    m: int = PLATEAU_M_DEFAULT,
) -> dict:
    """Plateau circuit state — independent of the crash circuit breaker.

    Reads training_metrics + plateau_detected events from events.jsonl and
    decides whether the loop has stopped progressing on the primary metric.

    State machine (pure function of events):
      - insufficient: < n+1 successful runs yet
      - ok          : last-N improvement ≥ threshold, OR an active warning
                      was cleared by a subsequent ≥-threshold jump
      - warn        : last-N improvement < threshold and either no prior
                      warning OR warning still pending grace
      - halt        : warning previously emitted and ≥ m further runs since
                      it have not improved by threshold

    `action` tells callers what to do (decoupled from state for clarity):
      - none      : nothing to emit
      - emit-warn : emit a plateau-detected event (state just became warn)
      - halt      : write HALTED + emit halted --reason plateau

    Returns a JSON-serializable dict so the CLI can hand it to bash via
    `event.py query plateau-status`.
    """
    runs = sorted(
        (e for e in read_all(project) if e.get("type") == "training_metrics"),
        key=lambda e: e.get("ts", ""),
    )
    base = {
        "state": "insufficient",
        "improvement": None,
        "threshold": threshold,
        "n": n,
        "m": m,
        "best_recent": None,
        "best_before": None,
        "rounds_since_warn": 0,
        "action": "none",
    }
    if len(runs) < n + 1:
        return base

    events = read_all(project)
    warn_event = next(
        (e for e in reversed(events) if e.get("type") == "plateau_detected"),
        None,
    )

    def _values(rs: list[dict]) -> list[float]:
        out: list[float] = []
        for r in rs:
            v = r.get("best_metric_value")
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    warning_active = False
    if warn_event is not None:
        # Compare by round number, not ts: timestamps are second-precision
        # and sub-second emits collide, which would misclassify post-warn
        # runs as pre-warn. Round numbers are monotonic and meaningful.
        warn_round = warn_event.get("round")
        if not isinstance(warn_round, int):
            warn_round = -1
        runs_at_or_before = [r for r in runs if int(r.get("round") or 0) <= warn_round]
        runs_after        = [r for r in runs if int(r.get("round") or 0) >  warn_round]
        vals_before = _values(runs_at_or_before)
        vals_after = _values(runs_after)
        if vals_after and vals_before:
            jump = max(vals_after) - max(vals_before)
            if jump >= threshold:
                # Warning was implicitly cleared — fall through to fresh eval.
                warning_active = False
            else:
                warning_active = True
                base["best_before"] = max(vals_before)
                base["best_recent"] = max(vals_after) if vals_after else None
                base["improvement"] = (
                    (max(vals_after) - max(vals_before)) if vals_after else None
                )
                base["rounds_since_warn"] = len(vals_after)
                if len(vals_after) >= m:
                    base["state"] = "halt"
                    base["action"] = "halt"
                else:
                    base["state"] = "warn"
                    base["action"] = "none"
                return base
        else:
            # No runs after the warning yet — still warned, no grace consumed.
            warning_active = True
            base["best_before"] = max(vals_before) if vals_before else None
            base["best_recent"] = None
            base["improvement"] = None
            base["rounds_since_warn"] = 0
            base["state"] = "warn"
            base["action"] = "none"
            return base

    # No active warning — compute fresh from the last N runs.
    vals = _values(runs)
    best_recent = max(vals[-n:])
    best_before = max(vals[:-n]) if vals[:-n] else None
    base["best_recent"] = best_recent
    base["best_before"] = best_before
    if best_before is None:
        base["state"] = "ok"
        base["improvement"] = None
        return base
    improvement = best_recent - best_before
    base["improvement"] = improvement
    if improvement < threshold:
        base["state"] = "warn"
        base["action"] = "emit-warn"
    else:
        base["state"] = "ok"
    return base


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[event.py] WARN: {name}={raw!r} not int — using default {default}", file=sys.stderr)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[event.py] WARN: {name}={raw!r} not float — using default {default}", file=sys.stderr)
        return default


def q_metrics_table(project: Path) -> str:
    """Render a Markdown table of all evaluation metrics per round.

    Output columns: round, run_name, best_epoch, plus every metric key found
    in any run's `extras` dict (union — runs missing a key show as empty).
    Final column: Δ vs prev best (signed change in primary metric).
    """
    runs = q_runs_history(project)
    if not runs:
        return "(no training_metrics events yet)"

    # Collect all extras keys in encounter order (stable across runs)
    seen_keys: list[str] = []
    for r in runs:
        for k in (r.get("extras") or {}):
            if k not in seen_keys:
                seen_keys.append(k)

    # Friendlier short labels for common keys
    label_map = {
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
    labels = [label_map.get(k, k) for k in seen_keys]

    header = ["#", "run_name", "ep", "Δ"] + labels
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    prev_primary = None
    for r in runs:
        rn = r.get("run_name", "?")[-22:]  # trim long timestamps
        ep = r.get("best_epoch", "?")
        primary = r.get("best_metric_value", 0)
        delta = "—" if prev_primary is None else f"{primary - prev_primary:+.4f}"
        prev_primary = primary
        row = [
            str(r.get("round", "?")),
            f"`{rn}`",
            str(ep),
            delta,
        ]
        extras = r.get("extras") or {}
        for k in seen_keys:
            v = extras.get(k)
            row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ─── Metrics extraction from results.csv ────────────────────────────

TASK_PRIMARY = {
    "detect":   "mAP50(B)",
    "obb":      "mAP50(B)",
    "segment":  "mAP50(M)",
    "pose":     "mAP50(P)",
    "classify": "accuracy_top1",
}


# Header substrings we capture into `extras` at the best epoch. The match is
# substring-based so it works across ultralytics versions and across tasks
# (mAP50(B) for detect/obb, mAP50(M) for segment, etc).
EXTRA_METRIC_PATTERNS = [
    "precision", "recall",
    "mAP50",  # both mAP50(B) and mAP50-95(B) — disambiguated by header name
    "box_loss", "cls_loss", "dfl_loss",
    "accuracy_top",  # classify
    "kobj",          # pose / keypoint objectness loss
    "seg_loss",      # segment
]


def _find_col(header: list[str], needle: str) -> int | None:
    for i, h in enumerate(header):
        if needle in h:
            return i
    return None


def extract_metrics_from_run(run_dir: Path, task: str, round_num: int, run_name: str) -> dict | None:
    """Build a training_metrics payload by reading results.csv + args.yaml.

    Captures the best epoch (where the task's primary metric is highest) and
    grabs ALL evaluation columns at that epoch into the `extras` dict —
    Precision, Recall, mAP50, mAP50-95, plus train/val box/cls/dfl losses
    and the overfit gap. Downstream tooling (build_prompt.py, metrics-table
    query) consumes these to show per-round metric deltas.
    """
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"[event.py] WARN: no results.csv at {results_csv}", file=sys.stderr)
        return None

    primary_metric_name = TASK_PRIMARY.get(task)
    if primary_metric_name is None:
        print(f"[event.py] WARN: unknown task {task!r}", file=sys.stderr)
        return None

    with results_csv.open() as f:
        reader = csv.reader(f)
        try:
            raw_header = next(reader)
        except StopIteration:
            print(f"[event.py] WARN: empty results.csv at {results_csv}", file=sys.stderr)
            return None
        header = [h.strip() for h in raw_header]
        rows = [[v.strip() for v in row] for row in reader]

    if not rows:
        print(f"[event.py] WARN: results.csv has no data rows", file=sys.stderr)
        return None

    # Locate the primary metric column by header name (resilient to column
    # reordering between ultralytics versions).
    primary_col = _find_col(header, primary_metric_name)
    if primary_col is None:
        print(
            f"[event.py] WARN: primary metric {primary_metric_name!r} not found in header {header!r}",
            file=sys.stderr,
        )
        return None

    best_val = -1.0
    best_epoch = -1
    best_row: list[str] | None = None
    for row in rows:
        try:
            v = float(row[primary_col])
            if v > best_val:
                best_val = v
                best_epoch = int(float(row[0]))
                best_row = row
        except (ValueError, IndexError):
            continue

    try:
        final_epoch = int(float(rows[-1][0]))
    except (ValueError, IndexError):
        final_epoch = len(rows)

    # Parse configured epochs from args.yaml (line scan — args.yaml is YAML
    # but we only need one key and don't want to depend on pyyaml).
    args_yaml = run_dir / "args.yaml"
    total_epochs = None
    if args_yaml.exists():
        for line in args_yaml.read_text().splitlines():
            if line.startswith("epochs:"):
                try:
                    total_epochs = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
                break

    # Build extras: every metric/loss column at the best epoch.
    extras: dict[str, float] = {}
    if best_row is not None:
        for i, col_name in enumerate(header):
            if not any(pat in col_name for pat in EXTRA_METRIC_PATTERNS):
                continue
            try:
                extras[col_name] = round(float(best_row[i]), 6)
            except (ValueError, IndexError):
                continue
        # Derived: overfit gap = val_box_loss - train_box_loss
        # (positive = overfitting; near 0 = healthy fit)
        train_box = extras.get("train/box_loss")
        val_box   = extras.get("val/box_loss")
        if train_box is not None and val_box is not None:
            extras["overfit_gap_box"] = round(val_box - train_box, 6)

    return {
        "round": round_num,
        "run_name": run_name,
        "best_metric_name": primary_metric_name,
        "best_metric_value": round(best_val, 6),
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "total_epochs": total_epochs,
        "patience_triggered": (
            total_epochs is not None and final_epoch < total_epochs
        ),
        "extras": extras,
    }


# ─── CLI ─────────────────────────────────────────────────────────────

# Per-event-type required + optional arguments. Required ones are positional
# in the bash callers' minds; keep --flag form for clarity. The shape is
# enforced by build_payload below.
EVENT_FIELDS = {
    "round-started":     {"round": int,    "session_id": str},
    "training-started":  {"round": int,    "run_name": str,    "params_json": "json"},
    "training-finished": {"round": int,    "run_name": str,    "exit_code": int,
                          "duration_sec": float},
    "training-metrics":  {"round": int,    "run_name": str,    "best_metric_name": str,
                          "best_metric_value": float, "best_epoch": int,
                          "final_epoch": int, "total_epochs": int,
                          "patience_triggered": bool},
    "claude-started":    {"round": int,    "session_id": str},
    "claude-finished":   {"round": int,    "exit_code": int,   "duration_sec": float},
    "validation-failed": {"round": int,    "violations_json": "json"},
    "preflight-failed":  {"round": int,    "reason": str},
    "halted":            {"reason": str,   "details_json": "json"},
    "session-resumed":   {"round": int,    "note": str},
    # OPERATOR-ONLY — held-out test split metrics. Firewalled from prompts.
    "test-metrics":      {"round": int,    "run_name": str,    "test_split_size": int,
                          "best_metric_name": str, "best_metric_value": float,
                          "best_epoch": int, "extras_json": "json"},
    # LLM-free policy decision. `policy` is e.g. "defaults" or "random-search".
    "baseline-decision": {"round": int,    "policy": str,      "seed": int,
                          "params_json": "json"},
    # Plateau circuit warning. All values are pulled from q_plateau_status.
    "plateau-detected":  {"round": int,    "n": int,           "threshold": float,
                          "improvement": float, "best_recent": float, "best_before": float},
    # Per-class diagnostic carrying raw metrics for the round. build_prompt.py
    # diagnoses on read (so the persistence-window N can change at any time
    # without re-emitting). The "_json" suffix means build_payload parses the
    # value as JSON and stores it under the unsuffixed key (per_class etc.).
    "per-class-metrics": {"round": int,    "run_name": str,
                          "per_class_json":   "json",
                          "confusion_json":   "json",
                          "class_names_json": "json"},
}


def build_payload(event_type: str, args) -> dict:
    fields = EVENT_FIELDS[event_type]
    payload: dict = {}
    for field, kind in fields.items():
        raw = getattr(args, field.replace("-", "_"), None)
        if raw is None:
            continue
        if kind is int:
            payload[field] = int(raw)
        elif kind is float:
            payload[field] = float(raw)
        elif kind is bool:
            payload[field] = (str(raw).lower() in ("1", "true", "yes"))
        elif kind == "json":
            try:
                payload[field.removesuffix("_json")] = json.loads(raw)
            except json.JSONDecodeError:
                print(
                    f"[event.py] WARN: --{field.replace('_', '-')} value is not JSON; storing as string",
                    file=sys.stderr,
                )
                payload[field.removesuffix("_json")] = raw
        else:
            payload[field] = raw
    return payload


def make_emit_parser(sub):
    emit_parser = sub.add_parser("emit", help="append an event")
    type_sub = emit_parser.add_subparsers(dest="event_type", required=True)
    for event_type, fields in EVENT_FIELDS.items():
        sp = type_sub.add_parser(event_type)
        for field, kind in fields.items():
            flag = "--" + field.replace("_", "-")
            if kind is bool:
                sp.add_argument(flag, default="false")
            else:
                sp.add_argument(flag, required=True)
    return emit_parser


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO self-trainer event log")
    p.add_argument("project", type=Path, help="project directory")
    sub = p.add_subparsers(dest="cmd", required=True)

    make_emit_parser(sub)

    q = sub.add_parser("query", help="answer a question from the log")
    q.add_argument(
        "question",
        choices=["current-round", "last-metrics", "consecutive-failures",
                 "runs-history", "metrics-table", "plateau-status"],
    )
    # plateau-status overrides — env vars apply as defaults, CLI wins.
    q.add_argument("--n",         type=int,   default=None,
                   help="plateau-status: window size (default: env YOLO_TRAINER_PLATEAU_N or 3)")
    q.add_argument("--threshold", type=float, default=None,
                   help="plateau-status: improvement threshold (default: env YOLO_TRAINER_PLATEAU_THRESHOLD or 0.005)")
    q.add_argument("--m",         type=int,   default=None,
                   help="plateau-status: grace rounds after a warning (default: env YOLO_TRAINER_PLATEAU_M or 2)")

    em = sub.add_parser("extract-metrics", help="read results.csv and emit a training-metrics event")
    em.add_argument("--run-dir", type=Path, required=True)
    em.add_argument("--task", required=True, choices=list(TASK_PRIMARY))
    em.add_argument("--round", dest="round_num", type=int, required=True)
    em.add_argument("--run-name", required=True)

    args = p.parse_args()

    if args.cmd == "emit":
        payload = build_payload(args.event_type, args)
        emit_event(args.project, args.event_type, payload)
        return 0

    if args.cmd == "query":
        if args.question == "current-round":
            print(q_current_round(args.project))
        elif args.question == "last-metrics":
            m = q_last_metrics(args.project)
            print(json.dumps(m) if m else "null")
        elif args.question == "consecutive-failures":
            print(q_consecutive_failures(args.project))
        elif args.question == "runs-history":
            print(json.dumps(q_runs_history(args.project)))
        elif args.question == "metrics-table":
            print(q_metrics_table(args.project))
        elif args.question == "plateau-status":
            n = args.n         if args.n         is not None else _env_int("YOLO_TRAINER_PLATEAU_N", PLATEAU_N_DEFAULT)
            t = args.threshold if args.threshold is not None else _env_float("YOLO_TRAINER_PLATEAU_THRESHOLD", PLATEAU_THRESHOLD_DEFAULT)
            m = args.m         if args.m         is not None else _env_int("YOLO_TRAINER_PLATEAU_M", PLATEAU_M_DEFAULT)
            print(json.dumps(q_plateau_status(args.project, n=n, threshold=t, m=m)))
        return 0

    if args.cmd == "extract-metrics":
        payload = extract_metrics_from_run(args.run_dir, args.task, args.round_num, args.run_name)
        if payload is None:
            return 1
        emit_event(args.project, "training-metrics", payload)
        print(
            f"[event] training_metrics: best {payload['best_metric_name']}="
            f"{payload['best_metric_value']:.4f} at epoch {payload['best_epoch']} "
            f"(final {payload['final_epoch']}/{payload['total_epochs']}"
            f"{', patience triggered' if payload['patience_triggered'] else ''})"
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
