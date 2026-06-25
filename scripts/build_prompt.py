#!/usr/bin/env python3
"""Build the per-round Claude prompt from machine facts + Claude's prose.

Why this exists (Harness §六 + §八 + §10):
  Before P3, next_instruction.md (written by Claude) was the ONLY memory
  between rounds. Three problems:
    1. Claude's prose was the source of truth — a mis-recall propagated.
    2. The file grew unbounded; rounds 20+ would pay for the bloat of every
       earlier run sitting in the prompt.
    3. Same file served the human reading the log AND the next Claude. The
       two views need different filters (UI vs Model — §8).

  This script splits the prompt into three layers:
    (1) Machine facts pulled directly from events.jsonl + results.csv. Claude
        sees the same numbers a human would compute. No prose middleman.
    (2) Run history with the newest N runs in full and older runs collapsed
        to one line each (Harness §6.4.1 micro-compact — recent full text,
        old stuff cleared to a marker but never deleted).
    (3) Claude's free-form notes from the previous round. Labeled clearly as
        "may contain errors" so when prose disagrees with facts, Claude knows
        which to trust.

Why we don't always include all three:
  Cold start has no events yet. We delegate that case back to the bash
  cold-start prompt in start_claude.sh — building Claude's first-round
  instructions belongs in the orchestrator, not here.

Token budget:
  No fancy estimator. We just cap the prose section at MAX_PROSE_CHARS and
  strip auto-generated sections (the human reads them in the per-round log
  dir anyway). Tuned for Sonnet/Opus 200K windows; override via env var
  YOLO_TRAINER_PROSE_BUDGET if you run on a smaller model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


RECENT_RUNS_FULL = 3  # most recent N runs get full detail in the history section
MAX_PROSE_CHARS = int(os.environ.get("YOLO_TRAINER_PROSE_BUDGET", "12000"))

# Persistence window for per-class weakness detection. A class flagged as the
# worst class for `PERSIST_N` consecutive rounds (this round inclusive) gets
# the "Persistent weakness detected" callout in the prompt.
PERSIST_N = int(os.environ.get("YOLO_TRAINER_PERSIST_N", "3"))


# ─── FIREWALL: operator-only event types ────────────────────────────────
# These event types carry data the agent must NEVER see. Held-out test
# split metrics are the canonical example — exposing them would defeat the
# whole point of an unbiased post-hoc benchmark.
#
# The firewall has three independent layers:
#   1. This denylist constant (single source of truth).
#   2. load_events() drops these types unconditionally, so any downstream
#      `for e in events` loop already-blind to them.
#   3. main() scans the final assembled prompt for guard strings and aborts
#      with a non-zero exit if any leak past the first two layers.
#
# If you add a new operator-only event type, add it here AND list a couple
# of distinctive guard strings in _PROMPT_GUARD_TERMS below.
AGENT_INVISIBLE_EVENT_TYPES = frozenset({
    "test_metrics",
})

# Substrings that, if present in the final prompt, indicate a leak from
# the firewall. Conservative — anything that resembles a test-eval keyword
# is flagged. Update in lockstep with AGENT_INVISIBLE_EVENT_TYPES above.
_PROMPT_GUARD_TERMS = (
    "test_metrics",
    "test-metrics",
    "split=test",
    "test_mAP",
    "test mAP",
    "held-out test",
    "test split",
)


# Section headings in Claude-written next_instruction.md that we drop during
# micro-compaction. The machine-facts section above already contains this
# information verbatim, so keeping Claude's prose copy just wastes tokens.
PROSE_SECTIONS_TO_DROP = (
    "Run history & analysis",
    "Improvement trajectory",
)


def load_events(project: Path) -> list[dict]:
    p = project / "events.jsonl"
    if not p.exists():
        return []
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                # The event log reader (event.py read_all) warns loudly; here
                # we just skip — the prompt build should still succeed.
                continue
            # Firewall layer 2: drop operator-only events at the loading
            # boundary so no downstream code can accidentally surface them.
            if ev.get("type") in AGENT_INVISIBLE_EVENT_TYPES:
                continue
            out.append(ev)
    return out


def runs_in_order(events: list[dict]) -> list[dict]:
    return sorted(
        (e for e in events if e.get("type") == "training_metrics"),
        key=lambda e: e.get("ts", ""),
    )


def build_plateau_section(project: Path) -> str | None:
    """Read the plateau circuit state and, while a warning is active, return
    a high-visibility nudge that goes at the top of the prompt.

    Returns None when state is not warn (insufficient / ok / halt — none of
    those benefit from the nudge). Delegated to event.py via subprocess so
    env-var overrides (YOLO_TRAINER_PLATEAU_*) apply identically to train.sh.
    """
    import subprocess
    script = Path(__file__).resolve().parent / "event.py"
    try:
        out = subprocess.run(
            ["python3", str(script), str(project), "query", "plateau-status"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        st = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    if st.get("state") != "warn":
        return None

    n = st.get("n", 3)
    threshold = float(st.get("threshold") or 0.0)
    improvement = st.get("improvement")
    # When the warning is active and no after-warn runs exist yet, the
    # query reports improvement=null. Recover the delta from the warning
    # event itself so the prompt always shows a concrete number.
    if improvement is None:
        for ev in reversed(load_events(project)):
            if ev.get("type") == "plateau_detected":
                improvement = ev.get("improvement")
                break
    m = int(st.get("m") or 2)
    rounds_since = int(st.get("rounds_since_warn") or 0)
    # Grace left: emit-warn is being delivered this round → all m rounds still
    # ahead. Otherwise it's whatever m − rounds_since gives us, clamped ≥ 1
    # so the prompt never reads "0 round(s) before halt" while it's still warn.
    if st.get("action") == "emit-warn":
        grace_left = m
    else:
        grace_left = max(1, m - rounds_since)

    imp_str = "—" if improvement is None else f"{improvement:+.4f}"

    return (
        "## ⚠️ Plateau detected — switch to an orthogonal axis\n"
        "> Operator-injected: the framework noticed the primary metric stopped\n"
        "> moving over the last few rounds. This block is automatic; it is not\n"
        "> from prior Claude notes.\n"
        "\n"
        f"Last N={n} rounds' best primary metric improved by {imp_str} "
        f"(threshold {threshold:.4f}). Continued micro-adjustments along the\n"
        f"same direction will waste the remaining **{grace_left} round(s)** before\n"
        "the framework halts this session automatically.\n"
        "\n"
        "**DO NOT** this round:\n"
        "- Tweak LR / momentum / weight_decay further in the same range\n"
        "- Add another fraction-point of an augmentation you already touched\n"
        "- Re-run with the same model size hoping for a different result\n"
        "\n"
        "**MUST** try ONE orthogonal lever this round:\n"
        "1. **Augmentation regime change** — disable mosaic mid-train via\n"
        "   `close_mosaic`, swap to a different augmentation mix (drop the\n"
        "   dominant lever, raise a neglected one), or zero out an aug that\n"
        "   may be hurting.\n"
        "2. **Model size swap** — n→s or s→m if VRAM allows; or back off to\n"
        "   n if you suspect underfitting at higher capacity.\n"
        "3. **Re-examine the data** — is the val set noise-bounded? Is one\n"
        "   class under-represented? Are labels noisy in specific classes?\n"
        "   If so, STOP and write next_instruction.md describing the data\n"
        "   work needed; do not launch another training run.\n"
        "\n"
        f"If you again propose adjustments in the same direction as the last\n"
        f"N={n} runs, this round will be wasted and the chain will halt after\n"
        f"{grace_left} more such round(s)."
    )


def fmt_metric(value, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def build_facts_section(events: list[dict]) -> str:
    runs = runs_in_order(events)
    last_round = max(
        (e["round"] for e in events if isinstance(e.get("round"), int)),
        default=0,
    )

    # Consecutive failures since the most recent successful training.
    consec_fail = 0
    for e in reversed(events):
        t = e.get("type")
        if t == "training_finished":
            if e.get("exit_code") == 0:
                break
            consec_fail += 1
        elif t in ("validation_failed", "preflight_failed"):
            consec_fail += 1

    # Best run by best_metric_value (higher = better for all primary metrics).
    best = max(
        runs,
        key=lambda r: float(r.get("best_metric_value", -1.0) or -1.0),
        default=None,
    )

    # Recent trajectory — easier than asking Claude to scan the history table.
    trajectory = [r.get("best_metric_value") for r in runs[-10:]]

    lines = ["## Verified facts (machine-extracted from events.jsonl)"]
    lines.append("> These are not Claude's recollection. Trust these over any disagreement")
    lines.append("> with the free-form notes below.")
    lines.append("")
    lines.append(f"- Last round in log: {last_round}")
    lines.append(f"- Total successful training runs: {len(runs)}")
    lines.append(f"- Consecutive failures since last success: {consec_fail}")
    if best is not None:
        lines.append(
            f"- BEST so far: {best['best_metric_name']}={fmt_metric(best['best_metric_value'])} "
            f"at run `{best['run_name']}` epoch {best['best_epoch']} (round {best.get('round','?')})"
        )
    if trajectory:
        traj_str = ", ".join(fmt_metric(v, 3) for v in trajectory)
        lines.append(f"- Last {len(trajectory)} runs trajectory: [{traj_str}]")
    return "\n".join(lines)


def build_metrics_table_section(events: list[dict]) -> str:
    """Per-round table of ALL evaluation metrics at the best epoch.

    Sourced from training_metrics events' `extras` dict (populated by
    event.py extract_metrics_from_run). The Δ column tracks change in the
    primary metric vs the previous run — easy at-a-glance trajectory read.
    """
    runs = runs_in_order(events)
    if not runs:
        return ""

    # Stable union of extras keys across all runs
    seen_keys: list[str] = []
    for r in runs:
        for k in (r.get("extras") or {}):
            if k not in seen_keys:
                seen_keys.append(k)

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

    headers = ["#", "ep", "Δ"] + labels
    lines = [
        "## Per-round metrics at best epoch",
        "> Every value below is at the epoch where the primary metric peaked,",
        "> extracted from the run's results.csv (NOT Claude's recollection).",
        "> Δ = change vs previous run. gap = val_box_loss − train_box_loss",
        "> (positive = overfitting; near 0 = healthy).",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]

    prev_primary = None
    for r in runs:
        ep = r.get("best_epoch", "?")
        primary = r.get("best_metric_value", 0.0) or 0.0
        delta = "—" if prev_primary is None else f"{primary - prev_primary:+.4f}"
        prev_primary = primary
        cells = [str(r.get("round", "?")), str(ep), delta]
        extras = r.get("extras") or {}
        for k in seen_keys:
            v = extras.get(k)
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_class_events(events: list[dict]) -> list[dict]:
    """All per_class_metrics events in chronological order."""
    return sorted(
        (e for e in events if e.get("type") == "per_class_metrics"),
        key=lambda e: e.get("ts", ""),
    )


def _light_history_for_diagnose(prior_events: list[dict], metric: str) -> list[dict]:
    """Convert prior per_class_metrics events to the lightweight history
    shape `diagnose_weak_classes` expects (a list of past Diagnosis dicts).

    We only need each prior round's `worst[0].class` for the persistence
    test, so we don't re-run the full diagnose; we just compute the rank-1
    class. This keeps build_prompt fast even when events.jsonl has many
    rounds of per-class data.
    """
    history: list[dict] = []
    for ev in prior_events:
        pc = ev.get("per_class") or {}
        if not pc:
            history.append({"worst": []})
            continue
        ranked = sorted(
            pc.items(),
            key=lambda kv: (
                float((kv[1] or {}).get(metric, 0.0) or 0.0),
                int((kv[1] or {}).get("support", 0) or 0),
                kv[0],
            ),
        )
        history.append({"worst": [{"class": ranked[0][0]}]})
    return history


def build_per_class_section(events: list[dict]) -> str | None:
    """Render the per-class weakness block, or None if no data yet.

    Sources its data from the latest `per_class_metrics` event and the
    diagnose_weak_classes pure function. When the diagnosis flags any
    class as `persistent`, the section prepends a high-visibility callout
    telling the agent to write a `## Data-layer recommendations` block in
    next_instruction.md — and reminds it that the Bash guard rejects any
    write under `datasets/`.
    """
    # Import here, not at module top, so `import build_prompt` keeps working
    # in environments where diagnose_classes is somehow absent (defensive —
    # the file is shipped alongside this one).
    try:
        from diagnose_classes import diagnose_weak_classes
    except ImportError:
        return None

    series = _per_class_events(events)
    if not series:
        return None

    latest = series[-1]
    prior = series[:-1]
    per_class = latest.get("per_class") or {}
    confusion = latest.get("confusion") or []
    class_names = latest.get("class_names") or []
    if not per_class:
        return None

    metric = "mAP50"
    history = _light_history_for_diagnose(prior, metric=metric)
    d = diagnose_weak_classes(
        per_class, confusion, class_names,
        history=history,
        persistent_n=PERSIST_N,
        metric=metric,
    )

    lines = ["## Per-class diagnosis (from latest run's per_class_metrics event)"]
    lines.append("> Machine-extracted from `model.val()` on the run's `weights/best.pt`.")
    lines.append("> See `runs/<task>/<run_name>/confusion_matrix.png` for the full matrix.")
    lines.append("")
    lines.append(f"- Run: `{latest.get('run_name', '?')}` (round {latest.get('round', '?')})")
    lines.append(f"- Ranking metric: **{metric}** (lower = worse)")
    lines.append("")
    lines.append("### Worst classes")
    if d["worst"]:
        lines.append("| Rank | Class | mAP50 | support | persistent |")
        lines.append("|---|---|---|---|---|")
        for i, w in enumerate(d["worst"], 1):
            mark = "**yes**" if w.get("persistent") else "no"
            lines.append(
                f"| {i} | `{w['class']}` | {fmt_metric(w['score'])} "
                f"| {w['support']} | {mark} |"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("### Most-confused pairs (true → predicted)")
    if d["confused_pairs"]:
        for p in d["confused_pairs"]:
            lines.append(f"- `{p['a']}` → `{p['b']}`: {p['count']} mispredictions")
    else:
        lines.append("(no off-diagonal entries above the per-row noise floor)")

    if d["recommend_data_review"]:
        flagged = ", ".join(f"`{c}`" for c in d["recommend_data_review"])
        lines.append("")
        lines.append("### ⚠️ Persistent weakness detected")
        lines.append(
            f"> Class(es) flagged: {flagged} — worst for {PERSIST_N} consecutive rounds. "
            "The data layer is the likely root cause."
        )
        lines.append("")
        lines.append(
            "Write a `## Data-layer recommendations` block in next_instruction.md "
            "enumerating one or more of:"
        )
        lines.append("- **Suspected labeling noise** (audit annotations for this class)")
        lines.append("- **Insufficient samples** (collect more training examples for this class)")
        lines.append("- **Augmentation mismatch** (review aug parameters for this class type)")
        lines.append("")
        lines.append(
            "**DO NOT modify the `datasets/` directory directly** — the Bash guard "
            "rejects writes under `datasets/`. All data work is human-mediated; the "
            "agent's role is to surface the recommendation, not to act on it."
        )

    return "\n".join(lines)


def build_history_section(events: list[dict]) -> str:
    runs = runs_in_order(events)
    if not runs:
        return "## Run history\n(no completed runs in events.jsonl yet)"

    n = len(runs)
    full_idx = max(0, n - RECENT_RUNS_FULL)

    lines = [
        "## Run history",
        f"> Oldest {full_idx} runs are micro-compacted to one line each (Harness §6.4.1).",
        f"> The last {n - full_idx} runs have full detail. Read the run dir's results.csv",
        f"> if you need finer-grained metrics than what's shown here.",
        "",
    ]

    for i, r in enumerate(runs):
        run_no = i + 1
        run_name = r.get("run_name", "?")
        metric_name = r.get("best_metric_name", "?")
        metric_val = r.get("best_metric_value", -1.0)
        best_ep = r.get("best_epoch", "?")
        final_ep = r.get("final_epoch", "?")
        total_ep = r.get("total_epochs", "?")
        patience = " (patience triggered)" if r.get("patience_triggered") else ""

        if i < full_idx:
            lines.append(
                f"- Run #{run_no}: {metric_name}={fmt_metric(metric_val)} "
                f"@ep{best_ep}/{final_ep}  `{run_name}`"
            )
        else:
            lines.append(f"")
            lines.append(f"### Run #{run_no} — `{run_name}` (round {r.get('round','?')})")
            lines.append(
                f"  {metric_name}: {fmt_metric(metric_val)} at epoch {best_ep}"
            )
            lines.append(
                f"  Trained {final_ep}/{total_ep} epochs{patience}"
            )

    return "\n".join(lines)


def compact_prose(text: str) -> str:
    """Drop auto-generated sections that duplicate machine facts.

    The structure of next_instruction.md is loose markdown — sections start
    with `## <title>` and continue until the next `## ` or EOF. We strip
    sections whose title matches PROSE_SECTIONS_TO_DROP, then truncate if
    still over budget.
    """
    if not text:
        return text

    # Split on level-2 headings, preserving the heading text per segment.
    parts = re.split(r"(?m)^(?=##\s)", text)
    out_parts: list[str] = []
    for part in parts:
        first_line = part.lstrip("\n").split("\n", 1)[0]
        match = re.match(r"##\s+(.+?)\s*$", first_line)
        if match:
            title = match.group(1).strip()
            if any(drop in title for drop in PROSE_SECTIONS_TO_DROP):
                continue
        out_parts.append(part)
    compacted = "".join(out_parts)

    if len(compacted) > MAX_PROSE_CHARS:
        head = compacted[: MAX_PROSE_CHARS // 2]
        tail = compacted[-MAX_PROSE_CHARS // 2 :]
        compacted = (
            head
            + f"\n\n[... prose truncated: {len(compacted) - MAX_PROSE_CHARS} chars omitted from middle ...]\n\n"
            + tail
        )
    return compacted


def build_prose_section(project: Path) -> str | None:
    p = project / "next_instruction.md"
    if not p.exists():
        return None
    raw = p.read_text()
    return compact_prose(raw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", type=Path)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--max-rounds", type=int, required=True)
    args = ap.parse_args()

    events = load_events(args.project)
    has_runs = any(e.get("type") == "training_metrics" for e in events)

    sections: list[str] = [f"# Round {args.round} of {args.max_rounds}"]

    # Plateau nudge: high-visibility, comes BEFORE Verified facts so the
    # agent reads it before any history. None when the warning is not
    # currently active.
    plateau_section = build_plateau_section(args.project)
    if plateau_section:
        sections.append(plateau_section)

    if has_runs:
        sections.append(build_facts_section(events))
        per_class_section = build_per_class_section(events)
        if per_class_section:
            sections.append(per_class_section)
        sections.append(build_metrics_table_section(events))
        sections.append(build_history_section(events))
    else:
        sections.append(
            "## Status\nNo completed training runs in events.jsonl yet. "
            "Either this is a fresh project or the prior run failed before "
            "emitting training_metrics."
        )

    prose = build_prose_section(args.project)
    if prose:
        sections.append(
            "## Claude's notes from previous round (FREE-FORM — may contain errors)"
        )
        sections.append(
            "> Cross-check any number you reuse against the verified facts above."
        )
        sections.append(prose)

    # Per-round directives. The hardcoded "Round N-1 = mandatory Action K"
    # was intentionally removed in P5; convergence signal will drive that.
    if args.round >= args.max_rounds:
        sections.append(
            "## THIS IS THE FINAL ROUND\n"
            "Do NOT launch training. Instead:\n"
            "1. Verify the latest run's results from events.jsonl + the run dir's results.csv\n"
            "2. Write a comprehensive next_instruction.md summarizing all runs, best result,\n"
            "   best model path, and recommendations for the next session\n"
            "3. Exit"
        )

    final_prompt = "\n\n".join(sections)

    # Firewall layer 3: belt-and-braces — scan the assembled prompt for any
    # operator-only guard terms. Layers 1+2 should make this impossible;
    # this fires only on a future regression. Fail-loud (Harness §二):
    # exit non-zero so the orchestrator notices instead of silently shipping
    # a leaky prompt.
    leaks = [t for t in _PROMPT_GUARD_TERMS if t in final_prompt]
    if leaks:
        print(
            f"[build_prompt] FATAL: prompt contains operator-only term(s) {leaks!r} — "
            "this indicates a firewall regression. Refusing to emit.",
            file=sys.stderr,
        )
        return 2

    print(final_prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
