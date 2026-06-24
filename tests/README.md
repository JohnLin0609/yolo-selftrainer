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

### xfail on the bypass scenarios

Today's Bash guard (`scripts/claude_bash_guard.py`) is a denylist of
command **heads**. Several straightforward bypasses (`python3 -c`,
`bash -c`, `find -delete`, heredocs, command substitution) slip through.
The tests/feature scenarios that exercise these bypasses are marked
`xfail` via `conftest.py::pytest_collection_modifyitems` keyed on the
filename `bypass_attempts.feature` plus a separate per-test marker for
the matching unit tests.

When the predicate is hardened (allow-list or sandbox-residual), those
tests **XPASS**. The implementation PR strips the xfail marker in
`conftest.py` in the same diff that adds the predicate logic.

`strict=False` is deliberate — strict-xfail would FAIL on XPASS, which
defeats the "tests turn green when impl lands" workflow. The trade-off:
an XPASS goes silently green; the impl PR's reviewer is responsible for
removing the marker.

### skipif on sandbox scenarios

`tests/features/sandbox_isolation.feature` describes the contract the
**future** sandbox runtime must satisfy: command writes don't escape
the project; network is denied; sibling projects unreadable. The
runtime doesn't exist yet, so `conftest.py` auto-skips every scenario
in that feature file unless `scripts/sandbox.run_in_sandbox` is
importable. Once the implementation PR adds `scripts/sandbox.py`, the
scenarios run automatically — no test-file changes needed.

`tests/steps/sandbox_isolation_steps.py` uses **lazy imports** inside
each step function so pytest-bdd can collect the file even when the
module is absent. Module-level imports would crash collection.

### Why both unit tests and BDD scenarios

They lock the same contract from two angles:

- **Unit tests** target `check_command(cmd)` directly — fast, no
  subprocesses. Good for tight TDD on the predicate itself.
- **BDD scenarios** invoke the guard the same way the Claude CLI's
  PreToolUse hook does (subprocess + JSON stdin + exit code + stderr).
  Catches integration-level regressions (e.g., a refactor that breaks
  the JSON-input parser without changing `check_command`).

When the allow-list implementation lands, both sets un-xfail in lockstep
(the impl PR diff touches one conftest.py marker block).

## Running just the failing-on-main contract

```bash
# See exactly which gaps the current guard has:
.venv/bin/pytest tests/ -v -m "" --runxfail
```

`--runxfail` disables the xfail decorator, so the bypass tests fail
loudly. Useful for confirming "these are the bypasses we know about" in
a security review.

## Exit-code contract

`.venv/bin/pytest tests/ -q` must return exit code **0** on `main`. The
xfail + skipped counters appear in the summary line; neither counts as
failure. This keeps CI green while the contract is documented but
unimplemented.
