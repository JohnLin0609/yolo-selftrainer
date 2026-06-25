#!/usr/bin/env python3
"""Pure aggregator that turns one project's events.jsonl into a ProviderRow.

`aggregate_from_events` is the analytic core of the cross-provider benchmark
feature. It consumes a list of already-parsed event dicts (as written by
scripts/event.py) and produces the metrics that go into a benchmark
comparison table:

  - final_best_val_mAP   (best primary metric seen, NOT latest)
  - final_test_mAP       (most recent held-out test_metrics, if any)
  - total_llm_cost_usd   (sum across claude_finished events)
  - total_wall_sec       (sum across both training_finished + claude_finished)
  - circuit_breaker_trips + halted_reasons
  - rounds_completed     (max round in any training_metrics event)

Pure: no I/O, no global state. Identical inputs → identical outputs. The
contract is pinned exhaustively by tests/unit/test_benchmark_aggregate.py.
"""
from __future__ import annotations

from typing import Any, TypedDict


class ProviderRow(TypedDict):
    provider: str
    model:    str
    final_best_val_mAP:    float | None
    final_test_mAP:        float | None
    total_llm_cost_usd:    float
    total_wall_sec:        int
    circuit_breaker_trips: int
    halted_reasons:        list[str]
    rounds_completed:      int


def _ts_key(ev: dict) -> str:
    """Stable sort key — empty string sorts first, missing ts → first."""
    return str(ev.get("ts") or "")


def _as_float(v: Any) -> float:
    """Best-effort float; None / non-numeric → 0.0 (forward-compat for the
    missing-total_cost_usd case)."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def aggregate_from_events(
    events: list[dict],
    provider: str,
    model: str,
) -> ProviderRow:
    # Sort by ts so "latest test metric" + "halt order" are deterministic
    # regardless of input list order.
    ordered = sorted(events, key=_ts_key)

    training_metrics  = [e for e in ordered if e.get("type") == "training_metrics"]
    test_metrics      = [e for e in ordered if e.get("type") == "test_metrics"]
    training_finished = [e for e in ordered if e.get("type") == "training_finished"]
    claude_finished   = [e for e in ordered if e.get("type") == "claude_finished"]
    halted            = [e for e in ordered if e.get("type") == "halted"]

    # final_best_val_mAP = MAX of best_metric_value across training_metrics
    # (mirrors generate_report.py:render_summary which picks the run with the
    # highest best_metric_value, not the latest). None when no training has
    # produced metrics yet.
    val_values: list[float] = []
    for e in training_metrics:
        v = e.get("best_metric_value")
        if v is not None:
            try:
                val_values.append(float(v))
            except (TypeError, ValueError):
                pass
    final_best_val = max(val_values) if val_values else None

    # final_test_mAP = the LATEST test_metrics (sorted ts asc, take last).
    final_test = None
    if test_metrics:
        last_test = test_metrics[-1]
        tv = last_test.get("best_metric_value")
        if tv is not None:
            try:
                final_test = float(tv)
            except (TypeError, ValueError):
                final_test = None

    # Cost: sum total_cost_usd over claude_finished. Missing field → 0.0.
    total_cost = sum(_as_float(e.get("total_cost_usd")) for e in claude_finished)

    # Wall: sum duration_sec across BOTH training and claude finishes.
    # Float seconds rounded to int for clean table cells.
    wall = sum(
        _as_float(e.get("duration_sec"))
        for e in (training_finished + claude_finished)
    )
    total_wall_sec = int(round(wall))

    # Halts.
    trips = len(halted)
    reasons: list[str] = []
    for e in halted:
        reason = e.get("reason")
        if reason is None:
            reasons.append("")
        else:
            reasons.append(str(reason))

    # rounds_completed = max round in training_metrics (handles retries —
    # two training_metrics with the same round number still yield that round
    # as the count). 0 when no training_metrics exist.
    round_numbers: list[int] = []
    for e in training_metrics:
        r = e.get("round")
        if isinstance(r, int):
            round_numbers.append(r)
        else:
            try:
                round_numbers.append(int(r))
            except (TypeError, ValueError):
                continue
    rounds_completed = max(round_numbers) if round_numbers else 0

    return {
        "provider":              provider,
        "model":                 model,
        "final_best_val_mAP":    final_best_val,
        "final_test_mAP":        final_test,
        "total_llm_cost_usd":    total_cost,
        "total_wall_sec":        total_wall_sec,
        "circuit_breaker_trips": trips,
        "halted_reasons":        reasons,
        "rounds_completed":      rounds_completed,
    }
