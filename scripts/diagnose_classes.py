#!/usr/bin/env python3
"""Pure diagnostic over per-class metrics — no I/O, no YOLO dependency.

`diagnose_weak_classes` is the analytic core of the per-class-diagnostics
feature. It takes already-parsed per-class metrics + a confusion matrix +
optional history of prior rounds' diagnoses, and returns a structured
Diagnosis dict describing:

  - worst        : ranked weak classes (ascending by `metric`, tie-broken
                   by lower support then alphabetic class name)
  - confused_pairs: top-3 off-diagonal entries from the confusion matrix,
                   thresholded against per-class support to ignore noise
  - recommend_data_review: subset of `worst[*].class` flagged `persistent`
                   — i.e., the same class has been `worst[0]` for
                   `persistent_n` consecutive rounds (this round inclusive)

The function is the inner kernel called by `build_prompt.py` and (in
principle) by `generate_report.py`. Isolating it here means unit tests can
exercise the contract without needing ultralytics, real run dirs, or the
event store — all of which are I/O-heavy and slow.
"""
from __future__ import annotations

from typing import Any


ClassMetrics = dict[str, float | int]
Diagnosis = dict[str, Any]


def _ranked_classes(
    per_class: dict[str, ClassMetrics],
    metric: str,
) -> list[dict]:
    """Sort by (score asc, support asc, name asc).

    Lower score = worse. The tie-break (lower support first) reflects that
    a struggling class with few examples is more diagnostic of a data
    problem than the same score on a well-populated class.
    """
    entries: list[dict] = []
    for name, m in per_class.items():
        score = float(m.get(metric, 0.0) or 0.0)
        support = int(m.get("support", 0) or 0)
        entries.append({"class": name, "score": score, "support": support})
    entries.sort(key=lambda e: (e["score"], e["support"], e["class"]))
    return entries


def _is_persistent(
    current_worst_class: str | None,
    history: list[dict] | None,
    persistent_n: int,
) -> bool:
    """True iff `current_worst_class` has been `worst[0].class` for the last
    `persistent_n - 1` historical entries AND is current worst now.

    History entries are prior Diagnosis dicts. Looking only at the `[0]`
    slot keeps the rule sharp: a class drifting from #2 to #1 doesn't fire
    the persistent flag — only consistent #1-worst does. Persistence
    resets the moment a different class displaces it (we look at the
    tail-N, not "ever in worst[0]").

    `persistent_n == 1` collapses to "always persistent" — we treat it as
    "this round alone counts", consistent with the rule "appears in the
    last N consecutive worst[0] slots including this one".
    """
    if current_worst_class is None:
        return False
    if persistent_n <= 1:
        return True
    prior_needed = persistent_n - 1
    if not history or len(history) < prior_needed:
        return False
    tail = history[-prior_needed:]
    for prior in tail:
        worst = prior.get("worst") or []
        if not worst:
            return False
        if worst[0].get("class") != current_worst_class:
            return False
    return True


def _extract_confused_pairs(
    per_class: dict[str, ClassMetrics],
    confusion: list[list[int]],
    class_names: list[str],
    top_k: int = 3,
) -> list[dict]:
    """Off-diagonal counts above the per-row noise floor, sorted desc.

    `confusion[i][j]` = times true class i was predicted as class j
    (i != j). Threshold per row = max(1, support[true_i] // 10) — drops
    spurious 1-2 misses on heavily-populated classes while keeping
    visibility on misses in small classes.
    """
    if not confusion or not class_names:
        return []
    pairs: list[dict] = []
    nc = len(class_names)
    rows = min(nc, len(confusion))
    for i in range(rows):
        row = confusion[i]
        if not row:
            continue
        true_name = class_names[i]
        support_true = int((per_class.get(true_name) or {}).get("support", 0) or 0)
        threshold = max(1, support_true // 10)
        cols = min(nc, len(row))
        for j in range(cols):
            if i == j:
                continue
            try:
                count = int(row[j])
            except (TypeError, ValueError):
                continue
            if count >= threshold:
                pairs.append({"a": true_name, "b": class_names[j], "count": count})
    pairs.sort(key=lambda p: (-p["count"], p["a"], p["b"]))
    return pairs[:top_k]


def diagnose_weak_classes(
    per_class: dict[str, ClassMetrics],
    confusion: list[list[int]],
    class_names: list[str],
    history: list[dict] | None = None,
    *,
    worst_k: int = 3,
    persistent_n: int = 3,
    metric: str = "mAP50",
) -> Diagnosis:
    """Return a Diagnosis dict — see module docstring for shape.

    Pure: identical inputs → identical outputs. No I/O, no global state.
    """
    if not per_class:
        return {"worst": [], "confused_pairs": [], "recommend_data_review": []}

    ranked = _ranked_classes(per_class, metric)
    worst = ranked[:max(0, worst_k)]
    worst_class = worst[0]["class"] if worst else None

    persistent = _is_persistent(worst_class, history, persistent_n)
    for i, entry in enumerate(worst):
        entry["persistent"] = bool(i == 0 and persistent)

    recommend = [e["class"] for e in worst if e.get("persistent")]
    pairs = _extract_confused_pairs(per_class, confusion, class_names)

    return {
        "worst": worst,
        "confused_pairs": pairs,
        "recommend_data_review": recommend,
    }
