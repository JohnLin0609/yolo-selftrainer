"""Shared fixtures + collection hooks for the test suite.

Two responsibilities live here:

1. Make `scripts/` importable without touching PYTHONPATH on the CLI.
2. Provide the `bash_guard` fixture — a callable that invokes
   `scripts/claude_bash_guard.py` as a subprocess (the same way the
   PreToolUse hook does in production). Tests assert exit codes + stderr
   rather than poking at internals, so the wire-level behavior is what
   gets locked in.

The `pytest_collection_modifyitems` hook below auto-xfails every test
inside `tests/features/bypass_attempts.feature` and every test marked
with `bypass_pending`, AND auto-skips every test whose path is under
`tests/features/sandbox_isolation.feature` when `scripts/sandbox` does
not yet exist. This keeps the test suite green on `main` while the
contract is documented; the implementation PRs delete the corresponding
hook entries (or simply create `scripts/sandbox.py`) to flip behavior
without changing the .feature files.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
GUARD_PATH = REPO_ROOT / "scripts" / "claude_bash_guard.py"

# Make `import param_bounds`, `import claude_bash_guard`, etc. work from
# tests without each test fiddling with sys.path.
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ─── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def guard_path() -> Path:
    return GUARD_PATH


@pytest.fixture
def bash_guard(guard_path):
    """Return a callable: bash_guard(cmd) -> CompletedProcess.

    Same wire format as the Claude CLI PreToolUse hook: stdin gets a JSON
    payload {"tool_input": {"command": cmd}}; the guard exits 0 (allow)
    or 2 (deny + stderr reason).
    """
    def _run(cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["python3", str(guard_path)],
            input=json.dumps({"tool_input": {"command": cmd}}),
            capture_output=True,
            text=True,
            timeout=10,
        )
    return _run


# ─── Sandbox runtime presence (controls skipif on sandbox scenarios) ──

def _sandbox_available() -> bool:
    """Returns True once scripts/sandbox is importable.

    The implementation PR that lands the sandbox runtime creates
    `scripts/sandbox.py` (or a package) with a callable named
    `run_in_sandbox`. Until then this returns False and the sandbox
    BDD feature is skipped.
    """
    spec = importlib.util.find_spec("sandbox")
    if spec is None:
        return False
    try:
        mod = importlib.import_module("sandbox")
    except Exception:
        return False
    return hasattr(mod, "run_in_sandbox")


SANDBOX_AVAILABLE = _sandbox_available()


# ─── Collection-time marking ─────────────────────────────────────────

# pytest-bdd generates test functions inside the `test_*.py` step file,
# so the nodeid path carries the .py filename (NOT the .feature). Match
# on both for robustness — manual unit-test runners may use the .feature
# extension; pytest-bdd-generated ones use the .py extension.
_BYPASS_MARKERS = ("bypass_attempts.feature", "test_bypass_attempts.py")
_SANDBOX_MARKERS = ("sandbox_isolation.feature", "test_sandbox_isolation.py")


def pytest_collection_modifyitems(config, items):
    """Apply xfail / skip markers based on file path + test name.

    Why here vs. inline @pytest.mark on the scenario function:
      pytest-bdd auto-generates one test per scenario; reaching into the
      generator to add markers is verbose. A collection hook is cleaner
      and the convention lives in ONE place future contributors can find.
    """
    bypass_xfail = pytest.mark.xfail(
        reason=(
            "Boundary 1 hardening pending — see "
            "tests/README.md \"Conventions\". Removed when the predicate "
            "is hardened (approach 2 allow-list) or sandbox lands "
            "(approach 1)."
        ),
        strict=False,
    )
    sandbox_skip = pytest.mark.skipif(
        not SANDBOX_AVAILABLE,
        reason=(
            "scripts/sandbox runtime not present — sandbox isolation "
            "scenarios document the contract for the next PR. They "
            "auto-run once scripts/sandbox.run_in_sandbox is importable."
        ),
    )
    for item in items:
        nodeid = item.nodeid
        # Only xfail the "this command must be rejected" scenarios — the
        # "must still be allowed" scenarios in the same file are NOT
        # bypasses and should pass cleanly today. Convention: bypass
        # scenarios start their name with "rejects_" (Gherkin
        # "Scenario: rejects …" → pytest "test_rejects_…").
        is_bypass_scenario = (
            any(m in nodeid for m in _BYPASS_MARKERS)
            and "test_rejects_" in nodeid
        )
        if is_bypass_scenario or item.get_closest_marker("bypass_pending"):
            item.add_marker(bypass_xfail)
        if any(m in nodeid for m in _SANDBOX_MARKERS):
            item.add_marker(sandbox_skip)


def pytest_configure(config):
    """Register custom markers so `pytest --strict-markers` (or just a
    helpful run with -W error::pytest.PytestUnknownMarkWarning) doesn't
    complain."""
    config.addinivalue_line(
        "markers",
        "bypass_pending: test exercises a Bash-guard bypass that the current "
        "predicate misses; auto-xfailed by conftest until hardening lands.",
    )
