"""Unit tests for benchmark_render.render_comparison_table.

Pins the Markdown shape so the impl PR's renderer satisfies a precise
contract. Skips cleanly until the production module exists.
"""
from __future__ import annotations

import importlib

import pytest


try:
    benchmark_render = importlib.import_module("benchmark_render")
    _MOD_OK = True
except ImportError:
    benchmark_render = None  # type: ignore[assignment]
    _MOD_OK = False


pytestmark = pytest.mark.skipif(
    not _MOD_OK,
    reason="benchmark_render not yet implemented (see plan: cross-provider benchmark)",
)


# ─── Helpers ─────────────────────────────────────────────────────────

def _row(provider: str, model: str, *,
         val_map: float | None = 0.50,
         test_map: float | None = 0.40,
         cost: float = 0.10,
         wall_sec: int = 600,
         rounds: int = 1,
         trips: int = 0,
         halted_reasons: list[str] | None = None) -> dict:
    return {
        "provider": provider,
        "model":    model,
        "final_best_val_mAP":  val_map,
        "final_test_mAP":      test_map,
        "total_llm_cost_usd":  cost,
        "total_wall_sec":      wall_sec,
        "circuit_breaker_trips": trips,
        "halted_reasons":      halted_reasons or [],
        "rounds_completed":    rounds,
    }


def _data_row_lines(report: str) -> list[str]:
    out: list[str] = []
    seen_sep = False
    for line in report.splitlines():
        s = line.strip()
        if not seen_sep:
            if s.startswith("|") and set(s) <= set("|-: "):
                seen_sep = True
            continue
        if not s.startswith("|"):
            break
        out.append(line)
    return out


def _row_for(report: str, provider: str) -> str:
    for line in _data_row_lines(report):
        first = line.split("|", 2)[1].strip().strip("`")
        if first == provider:
            return line
    raise AssertionError(f"no row for {provider!r} in:\n{report}")


# ─── Case 1: three rows sort by val DESC ─────────────────────────────

def test_three_rows_sort_descending_by_val_map():
    rows = [
        _row("openai",    "gpt-4o",          val_map=0.62, test_map=0.58, cost=0.40, wall_sec=900, trips=0),
        _row("anthropic", "claude-opus-4-7", val_map=0.78, test_map=0.71, cost=0.22, wall_sec=800, trips=0),
        _row("gemini",    "gemini-2.5-pro",  val_map=0.55, test_map=None, cost=0.0,  wall_sec=410, trips=0),
    ]
    out = benchmark_render.render_comparison_table(rows)
    data_lines = _data_row_lines(out)
    assert len(data_lines) == 3
    order = [l.split("|", 2)[1].strip().strip("`") for l in data_lines]
    assert order == ["anthropic", "openai", "gemini"], f"sort order wrong: {order}"
    # Footnote present
    assert "LLM $" in out and "wall" in out


# ─── Case 2: rows with None val mAP sort to bottom ──────────────────

def test_none_val_map_rows_sort_to_bottom():
    rows = [
        _row("a", "m", val_map=None),
        _row("b", "m", val_map=0.60),
        _row("c", "m", val_map=0.40),
        _row("d", "m", val_map=None),
    ]
    out = benchmark_render.render_comparison_table(rows)
    order = [
        l.split("|", 2)[1].strip().strip("`")
        for l in _data_row_lines(out)
    ]
    # Non-None first by value DESC, then None providers (alpha tie-break)
    assert order[0] == "b"
    assert order[1] == "c"
    assert set(order[2:]) == {"a", "d"}, f"None providers should be last: {order}"


# ─── Case 3: empty rows ──────────────────────────────────────────────

def test_empty_rows_yields_placeholder_not_broken_markdown():
    out = benchmark_render.render_comparison_table([])
    assert "(no providers ran)" in out, (
        f"expected placeholder line for empty input, got:\n{out}"
    )
    # Must not have data rows
    assert _data_row_lines(out) == []


# ─── Case 4: ties on val mAP break by provider alphabetic ───────────

def test_tie_on_val_map_broken_by_provider_name_alphabetic():
    rows = [
        _row("zulu",   "m", val_map=0.50),
        _row("alpha",  "m", val_map=0.50),
        _row("mike",   "m", val_map=0.50),
    ]
    out = benchmark_render.render_comparison_table(rows)
    order = [
        l.split("|", 2)[1].strip().strip("`")
        for l in _data_row_lines(out)
    ]
    assert order == ["alpha", "mike", "zulu"], f"alpha tie-break failed: {order}"


# ─── Case 5: None metric cells render as em-dash ────────────────────

def test_none_metric_cells_render_as_em_dash_not_literal_none():
    rows = [_row("x", "m", val_map=None, test_map=None)]
    out = benchmark_render.render_comparison_table(rows)
    line = _row_for(out, "x")
    assert "—" in line, f"expected em-dash, got: {line}"
    assert "None" not in line, f"row leaked literal 'None': {line}"


# ─── Case 6: cost formatting ─────────────────────────────────────────

def test_cost_formatting_zero_tiny_normal_and_large():
    rows = [
        _row("zero",   "m", val_map=0.10, cost=0.0),
        _row("tiny",   "m", val_map=0.20, cost=0.003),
        _row("normal", "m", val_map=0.30, cost=0.123),
        _row("large",  "m", val_map=0.40, cost=12.5),
    ]
    out = benchmark_render.render_comparison_table(rows)
    assert "$0.00" in _row_for(out, "zero")
    assert "$0.00" in _row_for(out, "tiny"),  "values below 0.005 should round to $0.00"
    assert "$0.12" in _row_for(out, "normal")
    assert "$12.50" in _row_for(out, "large")


# ─── Case 7: wall-time formatting H:MM:SS ───────────────────────────

def test_wall_time_formatting_zero_and_hours():
    rows = [
        _row("z",    "m", val_map=0.10, wall_sec=0),
        _row("hour", "m", val_map=0.20, wall_sec=3661),  # 1:01:01
    ]
    out = benchmark_render.render_comparison_table(rows)
    assert "0:00:00" in _row_for(out, "z")
    assert "1:01:01" in _row_for(out, "hour")


# ─── Case 8: footnote line explains the columns ─────────────────────

def test_footnote_line_explains_units():
    rows = [_row("x", "m", val_map=0.50)]
    out = benchmark_render.render_comparison_table(rows)
    # The footnote must reference at least the cost, wall, and halts cols by name.
    assert "LLM $" in out, "footnote missing LLM $ column note"
    assert "wall" in out, "footnote missing wall column note"
    assert "halts" in out, "footnote missing halts column note"


# ─── Case 9: halt reasons surface in a footnotes section ────────────

def test_halted_provider_reasons_appear_in_a_section():
    rows = [
        _row("ollama", "qwen2.5:32b", val_map=0.30, trips=1,
             halted_reasons=["circuit-breaker-agent"]),
    ]
    out = benchmark_render.render_comparison_table(rows)
    # Some "halt reasons" section under the table
    assert "halt" in out.lower() and "reason" in out.lower(), (
        f"expected a halt-reasons section, got:\n{out}"
    )
    idx = out.lower().find("halt")
    tail = out[idx:]
    assert "ollama" in tail
    assert "circuit-breaker-agent" in tail
