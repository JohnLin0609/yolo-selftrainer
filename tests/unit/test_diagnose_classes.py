"""Unit tests for diagnose_weak_classes — the pure analytic core.

Locks the contract from the per-class-diagnostics plan:
  - ranking is ascending by metric (lower = worse)
  - tie-break: lower support first, then alphabetic
  - persistence: class appears in worst[0] slot for N consecutive rounds
  - recommend_data_review = persistent worst[0] classes
  - confused_pairs: off-diagonal entries above per-row noise floor,
                    sorted desc by count, top-3

These tests run without ultralytics — the function is dependency-free by
design so it stays cheap to test.
"""
from __future__ import annotations

import pytest

from diagnose_classes import diagnose_weak_classes


# ─── Fixtures (small, scenario-tailored) ──────────────────────────────

def _mk_class(p: float, r: float, map50: float, map5095: float, support: int) -> dict:
    return {"P": p, "R": r, "mAP50": map50, "mAP50_95": map5095, "support": support}


# ─── Case 1: normal ranking ──────────────────────────────────────────

def test_normal_case_ranks_ascending_by_metric():
    per_class = {
        "dent":     _mk_class(0.92, 0.90, 0.88, 0.70, 120),
        "scratch":  _mk_class(0.45, 0.30, 0.12, 0.08, 60),
        "rust":     _mk_class(0.80, 0.75, 0.72, 0.55, 90),
        "crack":    _mk_class(0.85, 0.82, 0.79, 0.60, 80),
        "smudge":   _mk_class(0.50, 0.45, 0.35, 0.20, 40),
    }
    confusion = [[10] * 5 for _ in range(5)]  # irrelevant for this case
    class_names = list(per_class.keys())

    d = diagnose_weak_classes(per_class, confusion, class_names, history=None)

    assert [w["class"] for w in d["worst"]] == ["scratch", "smudge", "rust"]
    assert all(w["persistent"] is False for w in d["worst"])
    assert d["recommend_data_review"] == []


# ─── Case 2: ties resolve by lower support, then alphabetic ──────────

def test_tie_break_by_lower_support_then_alphabetic():
    per_class = {
        # All three share mAP50=0.40 — must be tie-broken
        "alpha":   _mk_class(0.5, 0.5, 0.40, 0.20, 100),  # higher support → later
        "bravo":   _mk_class(0.5, 0.5, 0.40, 0.20,  50),  # lower support → earlier
        "charlie": _mk_class(0.5, 0.5, 0.40, 0.20,  50),  # same support → alpha-after-bravo
        "dust":    _mk_class(0.5, 0.5, 0.99, 0.90,  10),  # not tied, best
    }
    confusion = [[1] * 4 for _ in range(4)]
    class_names = list(per_class.keys())

    d = diagnose_weak_classes(per_class, confusion, class_names, history=None)

    names = [w["class"] for w in d["worst"]]
    assert names == ["bravo", "charlie", "alpha"], (
        f"expected support-asc then alpha tie-break, got {names}"
    )


# ─── Case 3: empty input ─────────────────────────────────────────────

def test_empty_input_returns_empty_diagnosis():
    d = diagnose_weak_classes(per_class={}, confusion=[], class_names=[], history=None)
    assert d == {"worst": [], "confused_pairs": [], "recommend_data_review": []}


# ─── Case 4: single class ────────────────────────────────────────────

def test_single_class_has_no_confused_pairs_and_no_persistence():
    per_class = {"only": _mk_class(0.60, 0.55, 0.42, 0.30, 75)}
    confusion = [[75]]  # 1×1; pure diagonal
    class_names = ["only"]

    d = diagnose_weak_classes(per_class, confusion, class_names, history=None)

    assert len(d["worst"]) == 1
    assert d["worst"][0]["class"] == "only"
    assert d["worst"][0]["persistent"] is False
    assert d["confused_pairs"] == []
    assert d["recommend_data_review"] == []


# ─── Case 5: persistence triggers across N rounds ────────────────────

def test_persistent_weakness_across_n_rounds_flags_recommendation():
    per_class = {
        "scratch": _mk_class(0.40, 0.30, 0.15, 0.10, 50),
        "dent":    _mk_class(0.85, 0.82, 0.78, 0.60, 90),
    }
    confusion = [[40, 5], [3, 80]]
    class_names = ["scratch", "dent"]
    # Two prior rounds, each with scratch as worst[0].
    history = [
        {"worst": [{"class": "scratch", "score": 0.20}]},
        {"worst": [{"class": "scratch", "score": 0.18}]},
    ]

    d = diagnose_weak_classes(
        per_class, confusion, class_names,
        history=history, persistent_n=3,
    )

    assert d["worst"][0]["class"] == "scratch"
    assert d["worst"][0]["persistent"] is True
    assert d["recommend_data_review"] == ["scratch"]


# ─── Case 6: persistence resets if a different class displaces ───────

def test_persistence_resets_when_worst_class_changes():
    per_class = {
        "scratch": _mk_class(0.40, 0.30, 0.25, 0.15, 50),  # improved — no longer worst
        "rust":    _mk_class(0.30, 0.25, 0.10, 0.05, 80),  # new worst
        "dent":    _mk_class(0.85, 0.82, 0.78, 0.60, 90),
    }
    confusion = [[40, 5, 2], [3, 70, 5], [1, 2, 80]]
    class_names = ["scratch", "rust", "dent"]
    # Prior rounds had scratch worst, but this round rust takes over → reset.
    history = [
        {"worst": [{"class": "scratch", "score": 0.20}]},
        {"worst": [{"class": "scratch", "score": 0.18}]},
    ]

    d = diagnose_weak_classes(
        per_class, confusion, class_names,
        history=history, persistent_n=3,
    )

    assert d["worst"][0]["class"] == "rust"
    assert d["worst"][0]["persistent"] is False, (
        "rust has only been worst this round; persistence must NOT carry over from scratch"
    )
    assert d["recommend_data_review"] == []


# ─── Case 7: confused-pair extraction from off-diagonal entries ──────

def test_confused_pairs_extracted_above_noise_floor_and_sorted_desc():
    per_class = {
        "scratch": _mk_class(0.40, 0.30, 0.20, 0.10, 100),  # threshold = 10
        "dent":    _mk_class(0.85, 0.80, 0.78, 0.60, 100),  # threshold = 10
        "rust":    _mk_class(0.60, 0.55, 0.50, 0.30,  50),  # threshold = 5
    }
    # rows = true class, cols = predicted class
    confusion = [
        # true: scratch     | predicted: scratch dent rust
        [60, 30, 8],   # 30 scratch→dent (above 10), 8 scratch→rust (below 10 — dropped)
        # true: dent
        [12,  85, 2],  # 12 dent→scratch (above 10), 2 dent→rust (below 10 — dropped)
        # true: rust
        [3,  6, 40],   # 3 rust→scratch (below 5 — dropped), 6 rust→dent (above 5)
    ]
    class_names = ["scratch", "dent", "rust"]

    d = diagnose_weak_classes(per_class, confusion, class_names, history=None)

    # Three pairs survive threshold: scratch↔dent (30 + 12 separately) and rust→dent (6).
    # Top-3 sorted desc by count:
    counts = [p["count"] for p in d["confused_pairs"]]
    assert counts == sorted(counts, reverse=True), f"not sorted desc: {counts}"
    assert d["confused_pairs"][0] == {"a": "scratch", "b": "dent", "count": 30}
    assert {"a": "dent", "b": "scratch", "count": 12} in d["confused_pairs"]
    # The 8 (scratch→rust, below threshold 10) and 3 (rust→scratch, below 5) must NOT appear.
    for p in d["confused_pairs"]:
        assert not (p["a"] == "scratch" and p["b"] == "rust" and p["count"] == 8)
        assert not (p["a"] == "rust" and p["b"] == "scratch" and p["count"] == 3)
