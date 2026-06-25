#!/usr/bin/env python3
"""Pure Markdown renderer for cross-provider benchmark results.

`render_comparison_table` consumes a list of ProviderRow dicts (as produced
by benchmark_aggregate.aggregate_from_events) and returns a Markdown table.
Sorted DESC by val mAP with None LAST and alphabetic tie-break — so the
top performer is always on the first data row regardless of input order.

Contract pinned by tests/unit/test_benchmark_render.py. Pure: no I/O.
"""
from __future__ import annotations

from typing import Any


# ─── Cell formatters ─────────────────────────────────────────────────

def _hms(seconds: int | float) -> str:
    """H:MM:SS for wall-time cells. Mirrors generate_report.py:_hms.

    Negative inputs floor to 0 so a corrupt duration doesn't render "—1:59:59".
    """
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_metric(v: float | None) -> str:
    """Four-decimal-place metric cell; None → em-dash (NOT the literal 'None')."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_cost(c: float | None) -> str:
    """`$0.00` for values below 0.005 (avoids scientific notation),
    `$X.XX` otherwise. None treated as 0."""
    try:
        f = float(c) if c is not None else 0.0
    except (TypeError, ValueError):
        f = 0.0
    if f < 0.005:
        return "$0.00"
    return f"${f:.2f}"


# ─── Sort key ────────────────────────────────────────────────────────

def _sort_key(row: dict) -> tuple:
    """val mAP DESC, None LAST, then provider alpha ASC.

    Encoding: (is_none, -val_or_0, provider).
      - is_none → True for None rows; bool sort puts False (non-None) first
      - -val pushes higher values forward in ascending sort
      - provider name breaks ties
    """
    val = row.get("final_best_val_mAP")
    return (val is None, -float(val) if val is not None else 0.0,
            str(row.get("provider", "")))


# ─── Main entry ──────────────────────────────────────────────────────

HEADER = "| Provider | Model | val mAP | test mAP | LLM $ | wall | rounds | halts |"
SEPARATOR = "|---|---|---|---|---|---|---|---|"
FOOTNOTE = (
    "_LLM $ = total cost across all rounds; wall = train + agent time; "
    "halts = circuit-breaker triggers._"
)
EMPTY_PLACEHOLDER = "_(no providers ran)_"


def render_comparison_table(rows: list[dict]) -> str:
    """Return a Markdown comparison table for a list of ProviderRow dicts.

    Always includes header + separator. When `rows` is empty, the body is a
    single italic placeholder line — keeps the document shape consistent for
    a reader who just runs the benchmark with no providers.

    When any row has non-empty `halted_reasons`, appends a `### Halt reasons`
    section listing each halted provider's reason chain. This is the
    human-facing handle on why a provider's row carries a non-zero halts
    cell — gives the operator something to grep for in events.jsonl.
    """
    out_lines: list[str] = [HEADER, SEPARATOR]

    if not rows:
        out_lines.append(EMPTY_PLACEHOLDER)
        out_lines.append("")
        out_lines.append(FOOTNOTE)
        return "\n".join(out_lines) + "\n"

    sorted_rows = sorted(rows, key=_sort_key)

    for row in sorted_rows:
        provider = str(row.get("provider", ""))
        model    = str(row.get("model", ""))
        val_cell  = _fmt_metric(row.get("final_best_val_mAP"))
        test_cell = _fmt_metric(row.get("final_test_mAP"))
        cost_cell = _fmt_cost(row.get("total_llm_cost_usd"))
        wall_cell = _hms(row.get("total_wall_sec", 0) or 0)
        rounds = int(row.get("rounds_completed", 0) or 0)
        trips  = int(row.get("circuit_breaker_trips", 0) or 0)
        out_lines.append(
            f"| {provider} | `{model}` | {val_cell} | {test_cell} "
            f"| {cost_cell} | {wall_cell} | {rounds} | {trips} |"
        )

    out_lines.append("")
    out_lines.append(FOOTNOTE)

    # Halt-reasons section — only when at least one row carries a non-empty
    # reasons list. Order follows the sort order above so it lines up with
    # the table visually.
    halted_rows = [r for r in sorted_rows if r.get("halted_reasons")]
    if halted_rows:
        out_lines.append("")
        out_lines.append("### Halt reasons")
        for r in halted_rows:
            reasons = ", ".join(str(x) for x in r.get("halted_reasons", []))
            out_lines.append(f"- `{r.get('provider', '')}`: {reasons}")

    return "\n".join(out_lines) + "\n"
