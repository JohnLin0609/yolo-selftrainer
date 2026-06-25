"""Unit tests for the strict-heldout heldout-read patterns in
scripts/claude_bash_guard.py.

Locks the contract:

  - When strict mode is OFF, nothing in the heldout layer fires (back-compat).
  - When strict mode is ON:
      * Reads of datasets/<name>/{images,labels}/test/ are blocked
      * `yolo val split=test` and friends are blocked
      * Python keyword-arg `split="test"` smuggling is blocked
      * The sanctioned exit (`python3 scripts/run_test_tool.py`) is allowed
      * Reading datasets/<name>/{images,labels}/{train,val}/ stays allowed
      * `yolo val split=val` stays allowed
"""
from __future__ import annotations

import pytest

from claude_bash_guard import check_command


# ─── Strict mode OFF — every case below allowed (back-compat) ────────

OFF_ALLOWED_NOW = [
    "cat datasets/d/labels/test/img1.txt",
    "yolo val data=datasets/d/dataset.yaml model=runs/x/best.pt split=test",
    "find datasets/d/images/test -name *.jpg",
    # NB: `python -c '...'` is rejected by the generic interpreter-bypass
    # check regardless of strict mode, so we exclude it from this list.
]


@pytest.mark.parametrize("cmd", OFF_ALLOWED_NOW)
def test_strict_off_lets_heldout_reads_through(cmd):
    ok, reason = check_command(cmd, strict_heldout=False)
    assert ok, f"strict=False blocked {cmd!r}: {reason}"


# ─── Strict mode ON — exfil routes rejected ──────────────────────────

EXFIL_ATTEMPTS = [
    # Direct path reads
    "cat datasets/d/labels/test/img1.txt",
    "cat datasets/d/images/test/img1.jpg",
    "head -1 datasets/wheel/labels/test/000.txt",
    "ls datasets/d/images/test/",
    "find datasets/d/images/test -name *.jpg",
    "find datasets/d/labels/test -name *.txt -print",
    # yolo CLI split=test
    "yolo val data=datasets/d/dataset.yaml model=runs/x/best.pt split=test",
    "yolo val data=datasets/d/dataset.yaml model=runs/x/best.pt split='test'",
    'yolo val data=datasets/d/dataset.yaml model=runs/x/best.pt split="test"',
    # Python smuggling via dict literal / kwarg
    'python -c \'from ultralytics import YOLO; YOLO("x.pt").val(split="test")\'',
    'python3 -c "from ultralytics import YOLO; YOLO(\\"x.pt\\").val(split=\\"test\\")"',
]


@pytest.mark.parametrize("cmd", EXFIL_ATTEMPTS)
def test_strict_on_blocks_exfil_routes(cmd):
    # The security property is "blocked", not "blocked by THIS specific
    # layer". Some commands (e.g. python -c) are caught by the older
    # bypass-pattern check first — that's fine, the agent still can't
    # exfil. We only assert the rejection.
    ok, _ = check_command(cmd, strict_heldout=True)
    assert not ok, f"strict guard FAILED to block: {cmd!r}"


# ─── Strict mode ON — sanctioned + non-test routes allowed ───────────

STILL_ALLOWED_STRICT = [
    # Sanctioned exit — the one route the agent can use to get a score
    "python3 scripts/run_test_tool.py --project projects/x",
    "python scripts/run_test_tool.py --project projects/x",
    # Train/val reads stay allowed (they're not held-out)
    "cat datasets/d/labels/train/img1.txt",
    "find datasets/d/images/val -name *.jpg",
    "head -1 datasets/d/images/val/000.jpg",
    "cat datasets/d/dataset.yaml",
    # yolo on val split (not test) stays allowed
    "yolo val data=datasets/d/dataset.yaml model=runs/x/best.pt",
    "yolo val data=datasets/d/dataset.yaml model=runs/x/best.pt split=val",
    # General read-only access untouched
    "ls projects/x/runs",
    "kill -0 $(cat train.pid)",
]


@pytest.mark.parametrize("cmd", STILL_ALLOWED_STRICT)
def test_strict_on_lets_sanctioned_and_nontest_reads_through(cmd):
    ok, reason = check_command(cmd, strict_heldout=True)
    assert ok, f"strict guard wrongly blocked: {cmd!r}: {reason}"


# ─── Marker-file detection (default behavior is "look for marker") ───

def test_default_strict_arg_walks_up_for_marker(tmp_path, monkeypatch):
    """When strict_heldout=None, the guard walks up from cwd for the
    `.heldout_strict` marker. This test simulates both states."""
    # Project layout: project_dir with marker, cwd inside it
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".heldout_strict").touch()
    monkeypatch.chdir(project)

    # Without explicit flag, marker is found → strict mode ON
    ok, _ = check_command("cat datasets/d/labels/test/x.txt")
    assert not ok, "marker present but heldout patterns didn't fire"

    # Remove marker → strict mode OFF, same command allowed
    (project / ".heldout_strict").unlink()
    ok, _ = check_command("cat datasets/d/labels/test/x.txt")
    assert ok, "marker absent but heldout patterns still fired"


def test_marker_search_walks_up_subdirs(tmp_path, monkeypatch):
    """If the agent's cwd is project/subdir, the marker at project/ still
    triggers strict mode."""
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    (project / ".heldout_strict").touch()
    monkeypatch.chdir(project / "sub")

    ok, reason = check_command("yolo val data=x.yaml model=y.pt split=test")
    assert not ok, "marker in parent dir didn't trigger strict mode"
