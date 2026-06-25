"""Step defs for cross_provider_benchmark.feature.

Both production modules (`benchmark_aggregate`, `benchmark_render`) ship in
a follow-up PR. Until they exist, `pytestmark` skips every scenario in
this file — so the suite stays exit 0 and the contract documented here
travels with the feature file as a frozen target for the impl PR.

When the impl lands, the modules import successfully → `_MODULES_OK` is
True → pytestmark stops skipping → these scenarios run and must pass.
No edits to this file are required at impl time.
"""
from __future__ import annotations

import importlib
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when


# ─── Lazy imports — skip cleanly until impl lands ────────────────────

try:
    benchmark_aggregate = importlib.import_module("benchmark_aggregate")
    benchmark_render = importlib.import_module("benchmark_render")
    _MODULES_OK = True
except ImportError:
    benchmark_aggregate = None  # type: ignore[assignment]
    benchmark_render = None  # type: ignore[assignment]
    _MODULES_OK = False


pytestmark = pytest.mark.skipif(
    not _MODULES_OK,
    reason=(
        "benchmark_aggregate / benchmark_render not yet implemented — "
        "see plan: cross-provider benchmark"
    ),
)


scenarios("../features/cross_provider_benchmark.feature")


# ─── Synthetic events.jsonl fixtures ──────────────────────────────────
# Keyed by name so the Gherkin reads like English. Each list is what
# the impl's aggregator should see (already-parsed event dicts).

def _ts(sec: int) -> str:
    """Stable ISO timestamps with second-precision ordering."""
    return f"2026-06-25T10:00:{sec:02d}+08:00"


def _happy_provider_events(
    val_a: float, val_b: float, test_value: float,
    cost_a: float, cost_b: float,
) -> list[dict]:
    """Two rounds of training + agent + a single test eval — the canonical
    happy-path shape we use in scenario 1."""
    return [
        {"ts": _ts(1), "type": "training_metrics",
         "round": 1, "run_name": "r1",
         "best_metric_name": "mAP50(B)", "best_metric_value": val_a,
         "best_epoch": 50, "final_epoch": 100, "total_epochs": 100,
         "patience_triggered": False},
        {"ts": _ts(2), "type": "training_finished",
         "round": 1, "run_name": "r1", "exit_code": 0, "duration_sec": 300},
        {"ts": _ts(3), "type": "claude_finished",
         "round": 1, "exit_code": 0,
         "duration_sec": 45, "total_cost_usd": cost_a},
        {"ts": _ts(4), "type": "training_metrics",
         "round": 2, "run_name": "r2",
         "best_metric_name": "mAP50(B)", "best_metric_value": val_b,
         "best_epoch": 50, "final_epoch": 100, "total_epochs": 100,
         "patience_triggered": False},
        {"ts": _ts(5), "type": "training_finished",
         "round": 2, "run_name": "r2", "exit_code": 0, "duration_sec": 400},
        {"ts": _ts(6), "type": "claude_finished",
         "round": 2, "exit_code": 0,
         "duration_sec": 50, "total_cost_usd": cost_b},
        {"ts": _ts(7), "type": "test_metrics",
         "round": 2, "run_name": "r2",
         "best_metric_name": "mAP50(B)", "best_metric_value": test_value,
         "best_epoch": -1, "test_split_size": 30, "extras": {}},
    ]


FIXTURES: dict[str, list[dict]] = {
    # Anthropic: best val=0.78, test=0.71, cost=0.22 total.
    "happy_anthropic": _happy_provider_events(0.40, 0.78, 0.71, 0.12, 0.10),
    # OpenAI: best val=0.62, test=0.58, cost=0.40 total.
    "happy_openai":    _happy_provider_events(0.45, 0.62, 0.58, 0.20, 0.20),
    # Gemini: best val=0.55, no test_metrics event, no total_cost_usd on
    # claude_finished — covers the partial-data fallback path.
    "no_test_no_cost_gemini": [
        {"ts": _ts(1), "type": "training_metrics",
         "round": 1, "run_name": "g1",
         "best_metric_name": "mAP50(B)", "best_metric_value": 0.55,
         "best_epoch": 50, "final_epoch": 100, "total_epochs": 100,
         "patience_triggered": False},
        {"ts": _ts(2), "type": "training_finished",
         "round": 1, "run_name": "g1", "exit_code": 0, "duration_sec": 350},
        {"ts": _ts(3), "type": "claude_finished",
         "round": 1, "exit_code": 0, "duration_sec": 60},
    ],
    # Ollama hits the circuit breaker on round 2; halted reason in trailing event.
    "halted_ollama": [
        {"ts": _ts(1), "type": "training_metrics",
         "round": 1, "run_name": "o1",
         "best_metric_name": "mAP50(B)", "best_metric_value": 0.30,
         "best_epoch": 30, "final_epoch": 60, "total_epochs": 100,
         "patience_triggered": True},
        {"ts": _ts(2), "type": "claude_finished",
         "round": 1, "exit_code": 0,
         "duration_sec": 80, "total_cost_usd": 0.0},
        {"ts": _ts(3), "type": "training_finished",
         "round": 2, "run_name": "o2", "exit_code": 1, "duration_sec": 10},
        {"ts": _ts(4), "type": "halted",
         "reason": "circuit-breaker-agent",
         "details": {"consecutive_failures": 3}},
    ],
    # Scenario 3 — fixed durations summing to 515 (200+250 + 30+35).
    "two_rounds_durations_anthropic": [
        {"ts": _ts(1), "type": "training_finished",
         "round": 1, "run_name": "r1", "exit_code": 0, "duration_sec": 200},
        {"ts": _ts(2), "type": "training_finished",
         "round": 2, "run_name": "r2", "exit_code": 0, "duration_sec": 250},
        {"ts": _ts(3), "type": "claude_finished",
         "round": 1, "exit_code": 0,
         "duration_sec": 30, "total_cost_usd": 0.05},
        {"ts": _ts(4), "type": "claude_finished",
         "round": 2, "exit_code": 0,
         "duration_sec": 35, "total_cost_usd": 0.05},
    ],
}


# ─── Per-scenario context ────────────────────────────────────────────

@pytest.fixture
def ctx(tmp_path) -> dict[str, Any]:
    return {
        "workspace":  tmp_path,
        "providers":  [],   # list of (provider, model, events)
        "rows":       [],   # list of ProviderRow dicts (post-aggregate)
        "report":     "",   # rendered Markdown
        "single_row": None, # for the math-only scenario
    }


# ─── Helpers used by Then steps ──────────────────────────────────────

def _data_row_lines(report: str) -> list[str]:
    """Return only data-row lines (excluding header + separator + footnotes)."""
    out: list[str] = []
    seen_separator = False
    for line in report.splitlines():
        s = line.strip()
        if not seen_separator:
            if s.startswith("|") and set(s) <= set("|-: "):
                seen_separator = True
            continue
        if not s.startswith("|"):
            break  # end of table
        out.append(line)
    return out


def _row_for_provider(report: str, provider: str) -> str:
    for line in _data_row_lines(report):
        # Provider name appears between the first two `|`s; backtick-wrapped is OK
        first_cell = line.split("|", 2)[1].strip().strip("`")
        if first_cell == provider:
            return line
    raise AssertionError(
        f"no data row for provider {provider!r} in report:\n{report}"
    )


# ─── Background ──────────────────────────────────────────────────────

@given("a temporary benchmark workspace")
def given_workspace(ctx):
    # tmp_path already provided by pytest; nothing else to set up.
    pass


# ─── Given ───────────────────────────────────────────────────────────

@given(parsers.parse(
    'a benchmarked provider "{provider}" model "{model}" with fixture "{fix}"'
))
def given_provider_with_fixture(ctx, provider, model, fix):
    assert fix in FIXTURES, f"unknown fixture name: {fix!r}"
    ctx["providers"].append((provider, model, list(FIXTURES[fix])))


# ─── When ────────────────────────────────────────────────────────────

@when("the benchmark report is rendered")
def when_render(ctx):
    ctx["rows"] = [
        benchmark_aggregate.aggregate_from_events(events, provider, model)
        for provider, model, events in ctx["providers"]
    ]
    ctx["report"] = benchmark_render.render_comparison_table(ctx["rows"])


@when("aggregate_from_events is called for that provider")
def when_aggregate_single(ctx):
    provider, model, events = ctx["providers"][0]
    ctx["single_row"] = benchmark_aggregate.aggregate_from_events(
        events, provider, model
    )


# ─── Then (scenario 1) ───────────────────────────────────────────────

@then(parsers.parse('the report contains exactly {n:d} data rows'))
def then_n_data_rows(ctx, n):
    rows = _data_row_lines(ctx["report"])
    assert len(rows) == n, (
        f"expected {n} data rows, got {len(rows)}:\n{ctx['report']}"
    )


@then(parsers.parse('the data rows appear in this provider order: {order}'))
def then_provider_order(ctx, order):
    expected = [s.strip() for s in order.split(",")]
    actual: list[str] = []
    for line in _data_row_lines(ctx["report"]):
        first_cell = line.split("|", 2)[1].strip().strip("`")
        actual.append(first_cell)
    assert actual == expected, (
        f"data row order {actual} != expected {expected}\n\n{ctx['report']}"
    )


@then(parsers.parse(
    'the row for provider "{provider}" has val mAP cell "{val}"'
))
def then_val_cell(ctx, provider, val):
    row = _row_for_provider(ctx["report"], provider)
    assert val in row, (
        f"expected val mAP cell {val!r} in row, got:\n{row}"
    )


@then(parsers.parse(
    'the row for provider "{provider}" has test mAP cell "{val}"'
))
def then_test_cell(ctx, provider, val):
    row = _row_for_provider(ctx["report"], provider)
    assert val in row, (
        f"expected test mAP cell {val!r} in row, got:\n{row}"
    )


@then(parsers.parse(
    'the row for provider "{provider}" has test mAP cell em-dash'
))
def then_test_cell_dash(ctx, provider):
    row = _row_for_provider(ctx["report"], provider)
    assert "—" in row, (
        f"expected em-dash (—) test mAP cell in row, got:\n{row}"
    )
    # Defensive: must not render the Python literal "None"
    assert "None" not in row, (
        f"row rendered the literal string 'None' — should be em-dash:\n{row}"
    )


@then(parsers.parse(
    'the row for provider "{provider}" has LLM cost cell "{val}"'
))
def then_cost_cell(ctx, provider, val):
    row = _row_for_provider(ctx["report"], provider)
    assert val in row, (
        f"expected LLM cost cell {val!r} in row, got:\n{row}"
    )


@then(parsers.parse('every row has halts cell "{val}"'))
def then_every_row_halts(ctx, val):
    for line in _data_row_lines(ctx["report"]):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # halts is the last column per the renderer contract
        assert cells[-1] == val, (
            f"row halts cell {cells[-1]!r} != {val!r}\nrow: {line}"
        )


@then('the report includes the footnote line explaining the columns')
def then_footnote(ctx):
    rep = ctx["report"]
    assert "LLM $" in rep and "wall" in rep and "halts" in rep, (
        f"footnote missing column-explanation keywords:\n{rep}"
    )


# ─── Then (scenario 2) ───────────────────────────────────────────────

@then(parsers.parse(
    'the row for provider "{provider}" has halts cell "{val}"'
))
def then_halts_cell(ctx, provider, val):
    row = _row_for_provider(ctx["report"], provider)
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    assert cells[-1] == val, (
        f"halts cell for {provider!r} = {cells[-1]!r}, expected {val!r}\n"
        f"row: {row}"
    )


@then(parsers.parse(
    'the report contains a halt-reasons section that names "{reason}" '
    'for "{provider}"'
))
def then_halt_section(ctx, reason, provider):
    rep = ctx["report"]
    assert "Halt reasons" in rep or "halt reasons" in rep.lower(), (
        f"expected a halt-reasons section header, got:\n{rep}"
    )
    # The provider AND reason must both appear after the section header
    idx = rep.lower().find("halt reasons")
    tail = rep[idx:]
    assert provider in tail, (
        f"provider {provider!r} missing from halt-reasons section:\n{tail}"
    )
    assert reason in tail, (
        f"reason {reason!r} missing from halt-reasons section:\n{tail}"
    )


# ─── Then (scenario 3) ───────────────────────────────────────────────

@then(parsers.parse(
    "the resulting row's total_wall_sec equals {expected:d}"
))
def then_total_wall(ctx, expected):
    assert ctx["single_row"] is not None, "aggregate_from_events was not called"
    actual = ctx["single_row"]["total_wall_sec"]
    assert actual == expected, (
        f"total_wall_sec {actual} != expected {expected} "
        f"(row: {ctx['single_row']})"
    )


@then(parsers.parse(
    "the resulting row's circuit_breaker_trips equals {expected:d}"
))
def then_trips(ctx, expected):
    assert ctx["single_row"] is not None
    actual = ctx["single_row"]["circuit_breaker_trips"]
    assert actual == expected, (
        f"circuit_breaker_trips {actual} != expected {expected} "
        f"(row: {ctx['single_row']})"
    )
