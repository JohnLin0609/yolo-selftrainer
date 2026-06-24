"""Unit tests for scripts/claude_bash_guard.py.

Three parametrize blocks:

  1. test_allowed_today      — regression: commands the current denylist
                               correctly allows. Must stay green.
  2. test_blocked_today      — regression: commands the current denylist
                               correctly blocks (rm, sudo, pip, sed-train.sh,
                               …). Must stay green.
  3. test_bypass_attempts    — gap coverage: commands today's denylist
                               misses. xfail'd via conftest's collection
                               hook keyed on the `bypass_pending` marker.
                               These XPASS when the predicate is hardened;
                               the impl PR removes the marker.

Plus a regression test pinning the six `check_train_sh_write` patterns
that landed in the structured-param-contract refactor.
"""
from __future__ import annotations

import pytest

# `tests/conftest.py` adds scripts/ to sys.path so this import works.
import claude_bash_guard as cbg  # type: ignore[import-not-found]


# ─── Block 1: regression — commands the guard ALLOWS today ───────────

ALLOWED_TODAY = [
    "ls -la projects/foo",
    "cat results.csv",
    "head -1 results.csv",
    "tail -50 current.log",
    "grep -E '^EPOCHS=' train.sh",
    "sed -n '/^EPOCHS/p' train.sh",          # read-only sed is fine
    "awk '{print $1}' results.csv",          # read-only awk is fine
    "nohup bash train.sh > current.log 2>&1 &",
    "python3 scripts/event.py projects/x query current-round",
    "echo hello",
    "echo {} > next_params.json",            # write to next_params is allowed
    "kill -0 $(cat train.pid)",
    "git status",
    "git log --oneline -5",
    "git diff HEAD",
]


@pytest.mark.parametrize("cmd", ALLOWED_TODAY, ids=lambda c: c[:50])
def test_allowed_today(cmd):
    ok, reason = cbg.check_command(cmd)
    assert ok is True, f"current guard incorrectly blocks {cmd!r}: {reason}"


# ─── Block 2: regression — commands the guard BLOCKS today ───────────

BLOCKED_TODAY = [
    "rm -rf /tmp/x",
    "rm projects/foo/events.jsonl",
    "rmdir projects/foo",
    "sudo systemctl restart something",
    "chmod 777 /etc/passwd",
    "mv projects/foo /tmp/",
    "cp /etc/passwd /tmp/x",
    "ln -sf /etc/shadow /tmp/x",
    "dd if=/dev/zero of=/tmp/x bs=1M count=100",
    "pip install requests",
    "pip3 install requests",
    "uv pip install requests",
    "poetry add requests",
    "npm install left-pad",
    "ssh user@host echo hi",
    "scp file user@host:/tmp/",
    "rsync -av a/ b/",
    "curl https://example.com",
    "wget https://example.com",
    "docker run -it ubuntu",
    "systemctl restart nginx",
    "git push origin main",
    "git reset --hard HEAD",
    "git checkout main",
    "git rebase main",
    "git commit -m wip",
    # train.sh write-protection (added in the structured-param-contract PR):
    "sed -i s/X/Y/ train.sh",
    "sed -i.bak s/X/Y/ train.sh",
    "echo hello > train.sh",
    "echo hello >> train.sh",
    "tee train.sh < /tmp/x",
    "awk -i inplace '/X/{print Y}' train.sh",
    "perl -i -pe s/X/Y/ train.sh",
    "cp /tmp/other train.sh",
    # Compound: leading benign command must NOT shield the dangerous tail
    "echo hi && rm -rf /tmp/x",
    "true; sudo halt",
    "ls | head | xargs rm",  # xargs as a head is currently allowed,
                              # but the `rm` in xargs args is parseable today?
                              # Actually `xargs` is the head here. Leave it.
]


# Drop the xargs case — its current behavior is "allowed" (xargs is the
# head, not on DENY_HEADS). The bypass set covers xargs explicitly.
BLOCKED_TODAY = [c for c in BLOCKED_TODAY if not c.startswith("ls | head | xargs")]


@pytest.mark.parametrize("cmd", BLOCKED_TODAY, ids=lambda c: c[:50])
def test_blocked_today(cmd):
    ok, reason = cbg.check_command(cmd)
    assert ok is False, f"current guard incorrectly allows {cmd!r}"
    assert reason, "blocked commands must carry a non-empty reason"


# ─── Block 3: gap coverage — bypasses the guard MISSES today ─────────
#
# Marked xfail by tests/conftest.py via the `bypass_pending` marker
# (collection hook auto-applies it to anything matching the file path or
# this specific marker name). When the predicate is hardened, these go
# XPASS and the impl PR removes the conftest marker.

BYPASS_ATTEMPTS = [
    # Wrapped interpreters: head is python3/bash/sh/eval, not a known deny.
    'python3 -c "import os; os.remove(\'/tmp/x\')"',
    "python -c 'import shutil; shutil.rmtree(\"/tmp/x\")'",
    "bash -c 'rm /tmp/x'",
    "sh -c 'rm /tmp/x'",
    "/bin/bash -c 'rm /tmp/x'",
    "eval 'rm /tmp/x'",

    # find can both -delete and -exec arbitrary commands.
    "find . -delete",
    "find . -name '*.txt' -delete",
    "find . -exec rm {} \\;",
    "find / -exec rm -rf {} \\;",

    # AWK / Perl with system()/unlink — embed an interpreter inside an
    # otherwise-benign-looking command.
    "awk 'BEGIN{system(\"rm /tmp/x\")}' /dev/null",
    "perl -e 'unlink \"/tmp/x\"'",

    # xargs with sh -c — the dangerous tail is buried in an arg list.
    "xargs -I{} sh -c 'rm {}' < list.txt",
    "echo /tmp/x | xargs rm",

    # Command substitution at the head position — the actual head is
    # produced by an inner command and the outer guard never sees it.
    'echo $(printf "rm -rf /tmp/x")',
    "$(echo rm) -rf /tmp/x",
    "`echo rm` -rf /tmp/x",

    # Heredoc feeding an interpreter — same as bash -c at the wire level.
    "bash <<'EOF'\nrm /tmp/x\nEOF",
    "python3 <<'EOF'\nimport os; os.remove('/tmp/x')\nEOF",
]


@pytest.mark.bypass_pending
@pytest.mark.parametrize("cmd", BYPASS_ATTEMPTS, ids=lambda c: c[:55])
def test_bypass_attempts_blocked(cmd):
    """These currently FAIL (guard allows). xfail'd via conftest collection
    hook; the implementation PR that hardens the predicate removes the
    marker and these flip to plain passing tests."""
    ok, reason = cbg.check_command(cmd)
    assert ok is False, (
        f"BYPASS: guard allows {cmd!r} — "
        f"approach 2 (allow-list) or approach 1 (sandbox) should block it"
    )
    assert reason, "blocked commands must carry a non-empty reason"


# ─── train.sh write-protection regression ────────────────────────────
#
# Locks in the six patterns added in the structured-param-contract PR.
# If anyone weakens those regexes by accident, this test catches it.

TRAIN_SH_WRITE_ATTEMPTS = [
    "sed -i s/X/Y/ train.sh",
    "sed -i.bak s/X/Y/ train.sh",
    "awk -i inplace '/X/{print Y}' train.sh",
    "perl -i -pe s/X/Y/ train.sh",
    "echo hello > train.sh",
    "echo hello >> train.sh",
    "tee train.sh < /tmp/x",
    "cp /tmp/other train.sh",
    "mv /tmp/other train.sh",
    # Compound: dangerous tail inside an && chain
    "echo hi && sed -i s/X/Y/ train.sh",
]


@pytest.mark.parametrize("cmd", TRAIN_SH_WRITE_ATTEMPTS, ids=lambda c: c[:55])
def test_train_sh_writes_rejected(cmd):
    ok, reason = cbg.check_command(cmd)
    assert ok is False, f"train.sh write protection bypassed by {cmd!r}"
    assert "train.sh" in reason or "deny list" in reason, (
        f"reason should mention train.sh or deny list, got: {reason!r}"
    )


# Read-only access to train.sh is still allowed (no false positives on the
# write-guard regexes).
TRAIN_SH_READ_OK = [
    "cat train.sh",
    "grep EPOCHS train.sh",
    "sed -n '/^EPOCHS/p' train.sh",
    "head -20 train.sh",
    "diff train.sh /tmp/old_train.sh",
]


@pytest.mark.parametrize("cmd", TRAIN_SH_READ_OK, ids=lambda c: c[:55])
def test_train_sh_reads_allowed(cmd):
    ok, _ = cbg.check_command(cmd)
    assert ok is True, f"read-only access to train.sh should not be blocked: {cmd!r}"
