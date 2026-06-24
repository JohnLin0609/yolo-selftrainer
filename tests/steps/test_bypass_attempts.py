"""pytest-bdd step definitions for tests/features/bypass_attempts.feature.

The whole feature is auto-xfailed by tests/conftest.py — these step
definitions describe the contract the hardened guard must satisfy. They
invoke scripts/claude_bash_guard.py at the wire level (subprocess + JSON
stdin) so they catch refactors that break the JSON-input parser even if
check_command() itself stays correct.
"""
from __future__ import annotations

from pytest_bdd import given, parsers, scenarios, then, when

# Wire every scenario in the .feature file to this module.
scenarios("../features/bypass_attempts.feature")


@given(parsers.parse("the Bash guard is at scripts/claude_bash_guard.py"))
def _given_guard_present(guard_path):
    """Sanity: the script exists. Fixture comes from tests/conftest.py."""
    assert guard_path.exists(), f"missing: {guard_path}"


@when(
    parsers.re(r'the agent submits the Bash command "(?P<cmd>.+)"'),
    target_fixture="result",
)
def _when_submit(cmd, bash_guard):
    # The Gherkin scenario quotes contain escapes (\\;, \"...); pytest-bdd
    # passes the raw matched group, which is exactly the shell string the
    # agent would send. The guard reads it from JSON stdin verbatim.
    return bash_guard(_unescape_gherkin(cmd))


def _unescape_gherkin(s: str) -> str:
    """Undo the minimal Gherkin escaping used in our scenarios.

    Gherkin requires escaping `"` and `\` inside quoted step args. We use
    `\"` and `\\;` to embed those literally; nothing else needs special
    handling here (no \n / \t in our command strings).
    """
    return s.replace('\\"', '"').replace("\\\\", "\\")


@then(parsers.parse("the guard exits non-zero"))
def _then_nonzero(result):
    assert result.returncode != 0, (
        f"expected non-zero exit; got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@then(parsers.parse("the guard exits zero"))
def _then_zero(result):
    assert result.returncode == 0, (
        f"expected zero exit; got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@then(parsers.re(r'stderr contains "(?P<needle>.+)"'))
def _then_stderr_contains(needle, result):
    assert needle in result.stderr, (
        f"expected {needle!r} in stderr; got: {result.stderr!r}"
    )
