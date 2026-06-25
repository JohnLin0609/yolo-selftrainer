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
    """Returns True iff the sandbox module imports AND a backend runtime
    (currently `bwrap`) is on PATH. Both checks matter: a contributor
    might delete `scripts/sandbox.py`, or run the suite on a host without
    bubblewrap installed (e.g., CI containers, macOS dev machines). In
    either case the sandbox BDD scenarios skip gracefully.
    """
    spec = importlib.util.find_spec("sandbox")
    if spec is None:
        return False
    try:
        mod = importlib.import_module("sandbox")
    except Exception:
        return False
    if not hasattr(mod, "run_in_sandbox"):
        return False
    is_available = getattr(mod, "is_available", None)
    if callable(is_available):
        try:
            return bool(is_available())
        except Exception:
            return False
    # Module exists, no is_available helper — assume capable.
    return True


SANDBOX_AVAILABLE = _sandbox_available()


# ─── Collection-time marking ─────────────────────────────────────────

# pytest-bdd generates test functions inside the `test_*.py` step file,
# so the nodeid path carries the .py filename (NOT the .feature). Match
# on both for robustness.
_SANDBOX_MARKERS = ("sandbox_isolation.feature", "test_sandbox_isolation.py")


def pytest_collection_modifyitems(config, items):
    """Apply xfail / skip markers based on file path + test name.

    Why here vs. inline @pytest.mark on the scenario function:
      pytest-bdd auto-generates one test per scenario; reaching into the
      generator to add markers is verbose. A collection hook is cleaner
      and the convention lives in ONE place future contributors can find.
    """
    # The Boundary 1 predicate is hardened (commit-followup to dc0854d): the
    # bypass-rejection patterns + bwrap sandbox runtime are both shipped.
    # Bypass tests now PASS on their own; no auto-xfail marker is applied.
    # Sandbox tests still skip when bwrap (or scripts/sandbox.py) is missing,
    # so the suite stays green on hosts without the runtime.
    sandbox_skip = pytest.mark.skipif(
        not SANDBOX_AVAILABLE,
        reason=(
            "sandbox runtime unavailable — install `bubblewrap` (bwrap) and "
            "ensure scripts/sandbox.py is importable. The sandbox-isolation "
            "scenarios document the OS-level contract; on a capable host "
            "they run automatically."
        ),
    )
    for item in items:
        nodeid = item.nodeid
        # The bypass_pending marker is still recognized so a future
        # contributor can mark a newly-discovered bypass as expected-fail
        # without scattering inline xfail decorators. When the marker is
        # present we still auto-xfail; otherwise nothing happens.
        if item.get_closest_marker("bypass_pending"):
            item.add_marker(pytest.mark.xfail(
                reason="bypass marked pending — see tests/README.md",
                strict=False,
            ))
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
