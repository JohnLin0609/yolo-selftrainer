# Tests

## How to run

```bash
# All tests (unit + BDD)
.venv/bin/pytest tests/ -v

# Just the unit tests (fast — no subprocesses):
.venv/bin/pytest tests/unit/ -v

# Just the BDD scenarios:
.venv/bin/pytest tests/features/ -v
```

`scripts/setup_env.sh` installs `pytest` and `pytest-bdd` into the venv;
no other test deps.

## Layout

```
tests/
├── README.md                              # this file
├── conftest.py                            # fixtures + xfail/skip markers
├── unit/
│   └── test_claude_bash_guard.py          # predicate unit tests
├── features/
│   ├── bypass_attempts.feature            # guard predicate scenarios
│   └── sandbox_isolation.feature          # OS-isolation scenarios (skipped)
└── steps/
    ├── bypass_attempts_steps.py           # step defs — subprocess-invoke guard
    └── sandbox_isolation_steps.py         # step defs — lazy-import sandbox runner
```

## Conventions

### bypass_pending marker (for newly-discovered bypasses)

The predicate (`scripts/claude_bash_guard.py::_BYPASS_PATTERNS`) is
hardened: bypass tests in `test_bypass_attempts_blocked` and the
scenarios in `bypass_attempts.feature` pass cleanly on `main`. If you
discover a NEW bypass that the predicate doesn't yet catch, add it to
`BYPASS_ATTEMPTS` with `@pytest.mark.bypass_pending` — `conftest.py`
auto-xfails any test carrying that marker, so the suite stays green
while you (or a follow-up PR) extend `_BYPASS_PATTERNS` to handle it.

`strict=False` on the marker so once the predicate catches it, the test
silently flips from XFAIL to XPASS; the PR that hardens the pattern is
responsible for removing the `bypass_pending` marker from the test row.

### skipif on sandbox scenarios

`tests/features/sandbox_isolation.feature` exercises the bwrap-based
sandbox runtime in `scripts/sandbox.py`. On hosts WITH `bubblewrap`
(`apt install bubblewrap`) the scenarios run and assert OS-level
isolation: command writes can't escape the project, sibling projects
are unreadable, network is denied. On hosts without `bwrap` they skip
gracefully — `conftest.py::_sandbox_available()` checks both that the
module imports AND that `sandbox.is_available()` returns True.

`tests/steps/test_sandbox_isolation.py` uses **lazy imports** inside
each step function so pytest-bdd can collect the file even when the
module / runtime is absent. Module-level imports would crash collection.

### Why both unit tests and BDD scenarios

They lock the same contract from two angles:

- **Unit tests** target `check_command(cmd)` directly — fast, no
  subprocesses. Good for tight TDD on the predicate itself.
- **BDD scenarios** invoke the guard the same way the Claude CLI's
  PreToolUse hook does (subprocess + JSON stdin + exit code + stderr).
  Catches integration-level regressions (e.g., a refactor that breaks
  the JSON-input parser without changing `check_command`).

The sandbox feature is BDD-only — it exercises the OS-level isolation
contract end-to-end (real bwrap, real mount layout, real subprocesses).
Unit-testing the bwrap argv builder directly would only re-test what
bwrap itself already enforces.

## Exit-code contract

`.venv/bin/pytest tests/ -q` must return exit code **0** on `main`. On a
host with `bubblewrap` installed the expected summary is `103 passed`;
without bwrap the 7 sandbox-isolation scenarios skip, summary `96
passed, 7 skipped`. Either way, exit 0.

If a `bypass_pending`-marked test ever appears (a newly-discovered
bypass), the summary will also include an `xfailed` count — still
exit 0.
