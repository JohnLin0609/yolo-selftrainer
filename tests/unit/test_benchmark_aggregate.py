"""Unit tests for benchmark_aggregate.aggregate_from_events.

Pins the contract that the impl PR must satisfy. Skips cleanly until the
production module exists — once it does, the same tests run as PASSED
with no edits required here.
"""
from __future__ import annotations

import importlib

import pytest


try:
    benchmark_aggregate = importlib.import_module("benchmark_aggregate")
    _MOD_OK = True
except ImportError:
    benchmark_aggregate = None  # type: ignore[assignment]
    _MOD_OK = False


pytestmark = pytest.mark.skipif(
    not _MOD_OK,
    reason="benchmark_aggregate not yet implemented (see plan: cross-provider benchmark)",
)


# ─── Helpers ─────────────────────────────────────────────────────────

def _ts(sec: int) -> str:
    return f"2026-06-25T10:00:{sec:02d}+08:00"


def _tm(round_num: int, value: float, ts_sec: int) -> dict:
    return {
        "ts": _ts(ts_sec),
        "type": "training_metrics",
        "round": round_num,
        "run_name": f"r{round_num}",
        "best_metric_name": "mAP50(B)",
        "best_metric_value": value,
        "best_epoch": 50,
        "final_epoch": 100,
        "total_epochs": 100,
        "patience_triggered": False,
    }


def _tf(round_num: int, duration: int, ts_sec: int, exit_code: int = 0) -> dict:
    return {
        "ts": _ts(ts_sec),
        "type": "training_finished",
        "round": round_num,
        "run_name": f"r{round_num}",
        "exit_code": exit_code,
        "duration_sec": duration,
    }


def _cf(round_num: int, duration: int, ts_sec: int,
        cost: float | None = None) -> dict:
    ev = {
        "ts": _ts(ts_sec),
        "type": "claude_finished",
        "round": round_num,
        "exit_code": 0,
        "duration_sec": duration,
    }
    if cost is not None:
        ev["total_cost_usd"] = cost
    return ev


def _testm(round_num: int, value: float, ts_sec: int) -> dict:
    return {
        "ts": _ts(ts_sec),
        "type": "test_metrics",
        "round": round_num,
        "run_name": f"r{round_num}",
        "best_metric_name": "mAP50(B)",
        "best_metric_value": value,
        "best_epoch": -1,
        "test_split_size": 30,
        "extras": {},
    }


def _halt(reason: str, ts_sec: int) -> dict:
    return {
        "ts": _ts(ts_sec),
        "type": "halted",
        "reason": reason,
        "details": {},
    }


# ─── Case 1: normal multi-round happy path ───────────────────────────

def test_normal_run_aggregates_all_fields():
    events = [
        _tm(1, 0.40, 1),
        _tf(1, 300, 2),
        _cf(1, 45,  3, cost=0.12),
        _tm(2, 0.78, 4),
        _tf(2, 400, 5),
        _cf(2, 50,  6, cost=0.10),
        _testm(2, 0.71, 7),
    ]
    row = benchmark_aggregate.aggregate_from_events(events, "anthropic", "claude-opus-4-7")

    assert row["provider"] == "anthropic"
    assert row["model"]    == "claude-opus-4-7"
    assert row["final_best_val_mAP"] == pytest.approx(0.78)
    assert row["final_test_mAP"]     == pytest.approx(0.71)
    assert row["total_llm_cost_usd"] == pytest.approx(0.22)
    assert row["total_wall_sec"]     == 300 + 400 + 45 + 50
    assert row["circuit_breaker_trips"] == 0
    assert row["halted_reasons"] == []
    assert row["rounds_completed"] == 2


# ─── Case 2: best val is MAX, not latest ────────────────────────────

def test_final_best_val_map_is_max_across_history_not_latest():
    events = [
        _tm(1, 0.40, 1),
        _tm(2, 0.85, 2),   # peak
        _tm(3, 0.60, 3),   # regression — must NOT shadow the peak
    ]
    row = benchmark_aggregate.aggregate_from_events(events, "x", "y")
    assert row["final_best_val_mAP"] == pytest.approx(0.85), (
        f"expected MAX (0.85), got {row['final_best_val_mAP']} — likely picked latest"
    )


# ─── Case 3: empty events ────────────────────────────────────────────

def test_empty_events_returns_zeroed_row_without_raising():
    row = benchmark_aggregate.aggregate_from_events([], "gemini", "gemini-2.5-pro")
    assert row["provider"] == "gemini"
    assert row["model"]    == "gemini-2.5-pro"
    assert row["final_best_val_mAP"] is None
    assert row["final_test_mAP"]     is None
    assert row["total_llm_cost_usd"] == 0.0
    assert row["total_wall_sec"]     == 0
    assert row["circuit_breaker_trips"] == 0
    assert row["halted_reasons"]     == []
    assert row["rounds_completed"]   == 0


# ─── Case 4: no test_metrics ────────────────────────────────────────

def test_no_test_metrics_yields_none_test_map_but_keeps_other_fields():
    events = [_tm(1, 0.55, 1), _tf(1, 350, 2), _cf(1, 60, 3, cost=0.0)]
    row = benchmark_aggregate.aggregate_from_events(events, "gemini", "gemini-2.5-pro")
    assert row["final_test_mAP"] is None
    assert row["final_best_val_mAP"] == pytest.approx(0.55)
    assert row["total_wall_sec"] == 350 + 60


# ─── Case 5: missing total_cost_usd ─────────────────────────────────

def test_missing_total_cost_usd_treated_as_zero():
    events = [
        _tm(1, 0.55, 1),
        _cf(1, 60, 2, cost=None),   # no total_cost_usd field
        _cf(2, 70, 3, cost=0.03),
    ]
    row = benchmark_aggregate.aggregate_from_events(events, "x", "y")
    assert row["total_llm_cost_usd"] == pytest.approx(0.03), (
        "missing total_cost_usd must be treated as 0.0, not raise"
    )


# ─── Case 6: multiple halted events ─────────────────────────────────

def test_multiple_halted_events_counted_and_reasons_in_order():
    events = [
        _tm(1, 0.30, 1),
        _halt("preflight", 2),
        _tm(2, 0.32, 3),
        _halt("circuit-breaker-agent", 4),
    ]
    row = benchmark_aggregate.aggregate_from_events(events, "x", "y")
    assert row["circuit_breaker_trips"] == 2
    assert row["halted_reasons"] == ["preflight", "circuit-breaker-agent"]


# ─── Case 7: order-independence (events list not chronological) ─────

def test_aggregator_is_order_independent_on_input_list():
    chronological = [
        _tm(1, 0.40, 1),
        _testm(1, 0.30, 2),
        _tm(2, 0.78, 3),
        _testm(2, 0.71, 4),
    ]
    # Same events shuffled — latest test mAP must still resolve to ts=4's value.
    shuffled = [chronological[2], chronological[0], chronological[3], chronological[1]]
    row_a = benchmark_aggregate.aggregate_from_events(chronological, "x", "y")
    row_b = benchmark_aggregate.aggregate_from_events(shuffled, "x", "y")
    assert row_a["final_test_mAP"] == row_b["final_test_mAP"] == pytest.approx(0.71)
    assert row_a["final_best_val_mAP"] == row_b["final_best_val_mAP"] == pytest.approx(0.78)


# ─── Case 8: retried round (two training_metrics with same round) ───

def test_retry_same_round_keeps_max_val_and_max_round_count():
    events = [
        _tm(1, 0.40, 1),
        _tm(2, 0.55, 2),
        _tm(2, 0.78, 3),   # retry — same round number, higher value
    ]
    row = benchmark_aggregate.aggregate_from_events(events, "x", "y")
    assert row["rounds_completed"] == 2, (
        f"rounds_completed should be max round (2), got {row['rounds_completed']}"
    )
    assert row["final_best_val_mAP"] == pytest.approx(0.78)
