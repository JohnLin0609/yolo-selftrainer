#!/usr/bin/env python3
"""PreToolUse hook: validate Bash commands against a deny list.

Why this exists (Harness §5.1, §4.6):
  start_claude.sh runs Claude with --dangerously-skip-permissions so the
  autonomous loop has no UI prompts. That removes Claude Code's built-in
  permission layer, so we restore the safety boundary with a hook that runs
  BEFORE every Bash call. Hooks fire regardless of permission mode.

  Compound commands are split on logical operators (&&, ||, ;, |, &) and the
  head token of EACH subcommand is checked against the deny list. A compound
  like `echo hi && rm -rf /` would not hit `rm` if we matched the whole
  command string — splitting first is the SECURITY-critical step (mirrors
  easy-agent splitCommand.ts).

Why the splitter is intentionally minimal:
  We respect single/double quotes but NOT subshells $(...), heredocs, or
  escapes inside double quotes. The worst case is "we miss a cleverly-quoted
  dangerous subcommand" — the playbook tells Claude to use plain sed/grep/
  nohup, so exotic shell constructs already deviate from intended behavior
  and warrant human review on their own.
"""
import json
import re
import shlex
import sys


# Commands blocked unconditionally. Mirrors easy-agent DANGEROUS_BASH_PREFIXES
# plus things specific to this framework (pip would break the venv, mv could
# move dataset files, etc.). Be liberal here — false positives just nudge the
# operator to broaden the playbook; false negatives can wreck a 12-hour run.
DENY_HEADS = {
    "rm", "rmdir",
    "sudo", "doas", "su",
    "chmod", "chown", "chgrp",
    "mv", "cp", "ln",
    "dd",
    "mkfs", "mkfs.ext4", "mkfs.fat", "mkfs.xfs",
    "shutdown", "reboot", "halt", "poweroff", "init",
    "pip", "pip3", "uv", "poetry", "conda",
    "npm", "yarn", "pnpm",
    "ssh", "scp", "sftp", "rsync",
    "curl", "wget",
    "docker", "podman", "kubectl",
    "systemctl", "service",
}

# Git subcommands that mutate shared state. `git status/log/diff/show` are fine.
DENY_GIT_SUBCOMMANDS = {
    "push", "reset", "clean", "checkout", "rebase",
    "commit", "merge", "stash", "tag", "branch",
    "remote", "config", "filter-branch",
}

# The agent's hyperparameter contract is `next_params.json`. Any attempt to
# write to train.sh (sed -i, awk -i inplace, redirect, tee) bypasses
# apply_params.py + param_bounds.py and reopens the same trust hole the
# structured contract closed. Read-only inspection (cat, grep, sed -n) is
# still allowed.
_TRAIN_SH_WRITE_PATTERNS = [
    # `sed -i ... train.sh` (-i may carry an optional backup suffix like -i.bak)
    re.compile(r"\bsed\b[^|;&]*\s-i(\.[A-Za-z]+)?\b[^|;&]*\btrain\.sh\b"),
    # `awk -i inplace ... train.sh`
    re.compile(r"\bawk\b[^|;&]*\s-i\s+inplace\b[^|;&]*\btrain\.sh\b"),
    # `perl -i ... train.sh`
    re.compile(r"\bperl\b[^|;&]*\s-i(\.[A-Za-z]+)?\b[^|;&]*\btrain\.sh\b"),
    # `> train.sh` / `>> train.sh` redirect (anywhere in the subcommand)
    re.compile(r">>?\s*([^\s|;&'\"]+/)?train\.sh\b"),
    # `tee train.sh` / `tee -a train.sh`
    re.compile(r"\btee\b[^|;&]*\s([^\s|;&]+/)?train\.sh\b"),
    # `cp X train.sh` / `mv X train.sh` — already blocked by DENY_HEADS but
    # belt-and-braces in case future relaxation enables them.
    re.compile(r"\b(cp|mv|install)\b[^|;&]*\s([^\s|;&]+/)?train\.sh\b"),
]


def check_train_sh_write(subcommand: str) -> tuple[bool, str]:
    """Return (allowed, reason). False = subcommand writes to train.sh.

    Designed to be coarse: a few false positives are acceptable since the
    agent has next_params.json as the supported channel. The matched
    pattern is included in the reason so the agent can re-plan.
    """
    for pat in _TRAIN_SH_WRITE_PATTERNS:
        if pat.search(subcommand):
            return False, (
                f"writes to train.sh are blocked — set hyperparameters via "
                f"next_params.json instead (matched: {pat.pattern!r})"
            )
    return True, ""


# The dataset is operator-curated. The agent's role on data weakness is to
# WRITE A RECOMMENDATION (next_instruction.md `## Data-layer recommendations`),
# not to silently relabel / delete / mutate dataset files. Read-only access
# (`cat`, `grep`, `find -print`, `sed -n`) stays allowed; only in-place
# mutators and writes targeting the `datasets/` subtree are rejected.
#
# Same pattern shape as _TRAIN_SH_WRITE_PATTERNS — coarse, false-positive
# tolerant. cp/mv/install are already in DENY_HEADS so we don't repeat them.
_DATASETS_WRITE_PATTERNS = [
    # `sed -i ... datasets/...` (-i may carry an optional backup suffix)
    re.compile(r"\bsed\b[^|;&]*\s-i(\.[A-Za-z]+)?\b[^|;&]*\bdatasets/"),
    # `awk -i inplace ... datasets/...`
    re.compile(r"\bawk\b[^|;&]*\s-i\s+inplace\b[^|;&]*\bdatasets/"),
    # `perl -i ... datasets/...`
    re.compile(r"\bperl\b[^|;&]*\s-i(\.[A-Za-z]+)?\b[^|;&]*\bdatasets/"),
    # `> datasets/...` / `>> datasets/...` — redirect anywhere whose target
    # contains the datasets/ segment (catches `> ./datasets/foo`, `>> a/datasets/b`)
    re.compile(r">>?\s*\S*\bdatasets/"),
    # `tee datasets/...` / `tee -a datasets/...`
    re.compile(r"\btee\b[^|;&]*\s\S*\bdatasets/"),
]


def check_datasets_write(subcommand: str) -> tuple[bool, str]:
    """Return (allowed, reason). False = subcommand writes under datasets/.

    Read-only inspection (cat/grep/find/sed -n/awk read-only) is unaffected
    — only in-place mutators and redirect/tee targeting datasets/ are caught
    here. cp/mv/install are blocked unconditionally via DENY_HEADS already.
    """
    for pat in _DATASETS_WRITE_PATTERNS:
        if pat.search(subcommand):
            return False, (
                f"writes under datasets/ are blocked — surface data-layer "
                f"issues as recommendations in next_instruction.md under "
                f"`## Data-layer recommendations` instead "
                f"(matched: {pat.pattern!r})"
            )
    return True, ""


# Known bypass patterns for the head-token denylist. Each entry is
# (compiled_regex, human-readable reason). The patterns are tight enough
# to avoid false positives on legitimate idioms (read-only sed/awk, python3
# with a script arg, mid-args $(...) like `kill -0 $(cat train.pid)`).
#
# Defense-in-depth alongside Boundary 2 (param contract) and Boundary 3
# (circuit breaker). In run_agent.py mode the sandbox runtime adds OS-level
# isolation as well — see scripts/sandbox.py and docs/architecture.md.
_BYPASS_PATTERNS = [
    # -c / --command on any interpreter: payload is arbitrary code.
    # The interpreter token may carry a /path/to/ prefix (/bin/bash etc.).
    (re.compile(r"\b(?:/[^/\s]+/)*(?:bash|sh|dash|zsh|python3?|node|ruby|perl)\b[^|;&]*\s(?:-[a-zA-Z]*c|--command)\b"),
     "interpreter -c: arbitrary code execution"),
    # eval at word boundary
    (re.compile(r"(?<![\w/.-])eval\b"),
     "eval: arbitrary code execution"),
    # find primitives that delete or spawn commands
    (re.compile(r"\bfind\b[^|;&]*\s-delete\b"),
     "find -delete: erases without invoking rm"),
    (re.compile(r"\bfind\b[^|;&]*\s-(?:exec|execdir|ok|okdir)\b"),
     "find -exec/-execdir/-ok: arbitrary command spawn"),
    # awk / perl embedded shell
    (re.compile(r"\bawk\b[^|;&]*['\"][^'\"]*\bsystem\s*\("),
     "awk system(): arbitrary command spawn"),
    (re.compile(r"\bperl\b[^|;&]*\s-[a-zA-Z]*e\b"),
     "perl -e: arbitrary code execution"),
    # xargs feeding a wrapper interpreter
    (re.compile(r"\bxargs\b[^|;&]*\s-[Ii]\b[^|;&]*\s(?:bash|sh|python3?|dash|zsh)\b"),
     "xargs -I … sh|bash|python: arbitrary command spawn"),
    # xargs piped to a deny-listed command — the bypass is the wrapper, not
    # xargs itself
    (re.compile(r"\bxargs\b\s+(?:-[a-zA-Z0-9]+(?:=\S+)?\s+)*(?:rm|rmdir|sudo|mv|cp|chmod|chown|dd|kill)\b"),
     "xargs to a deny-listed command"),
    # Heredoc redirected INTO an interpreter (bash <<EOF, python3 <<'EOF', …).
    # The heredoc body is arbitrary code the interpreter will run.
    (re.compile(r"\b(?:bash|sh|dash|zsh|python3?|perl|ruby|node)\b[^|;&]*\s<<-?\s*['\"]?\w+"),
     "heredoc into interpreter: arbitrary code execution"),
    # Command substitution AT THE HEAD POSITION (start of subcommand, after
    # optional env-var assignments + whitespace). Substitution mid-args is
    # allowed — `kill -0 $(cat train.pid)` is a legitimate idiom.
    (re.compile(r"^\s*(?:[A-Z_][A-Z0-9_]*=\S*\s+)*\$\("),
     "command substitution at head: actual command obscured from validator"),
    (re.compile(r"^\s*(?:[A-Z_][A-Z0-9_]*=\S*\s+)*`"),
     "backtick command substitution at head: actual command obscured"),
]


def check_bypass_patterns(subcommand: str) -> tuple[bool, str]:
    """Return (allowed, reason). False = subcommand matches a known bypass.

    Runs between check_train_sh_write and the head-token deny check, so
    rejections here surface BEFORE the head-token gets to decide.
    """
    for pat, reason in _BYPASS_PATTERNS:
        if pat.search(subcommand):
            return False, reason
    return True, ""


def split_subcommands(command: str):
    """Split on logical operators while respecting single + double quotes."""
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    in_single = False
    in_double = False
    OPERATORS = ("&&", "||", ";", "|", "&")
    while i < len(command):
        c = command[i]
        if c == "'" and not in_double:
            in_single = not in_single
            buf.append(c)
            i += 1
            continue
        if c == '"' and not in_single:
            in_double = not in_double
            buf.append(c)
            i += 1
            continue
        if not in_single and not in_double:
            matched = None
            for op in OPERATORS:
                if command[i:i + len(op)] == op:
                    matched = op
                    break
            if matched:
                parts.append("".join(buf).strip())
                buf = []
                i += len(matched)
                continue
        buf.append(c)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def head_token(subcommand: str) -> tuple[str, list[str]]:
    """Return (command_name, all_tokens) skipping leading env-var assignments.

    Handles patterns like `FOO=1 BAR=baz cmd args`. Returns ("", []) on parse
    failure so callers can decide whether to allow or block.
    """
    try:
        tokens = shlex.split(subcommand)
    except ValueError:
        return "", []
    idx = 0
    while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("="):
        idx += 1
    if idx >= len(tokens):
        return "", tokens
    return tokens[idx], tokens[idx:]


def check_command(command: str) -> tuple[bool, str]:
    """Return (allowed, reason). reason is empty when allowed."""
    for sub in split_subcommands(command):
        # Check the train.sh-write patterns BEFORE the head-token deny list.
        # This catches `bash -c "sed -i ... train.sh"`-style wrappers where
        # the head token is something benign.
        ok, reason = check_train_sh_write(sub)
        if not ok:
            return False, f"{reason} (subcommand: {sub!r})"
        # Datasets/ are operator-curated. Agent surfaces recommendations,
        # not writes. Runs immediately after train.sh-write because both
        # are "trusted-asset" guards; ordering vs bypass patterns is
        # irrelevant since they're disjoint.
        ok, reason = check_datasets_write(sub)
        if not ok:
            return False, f"{reason} (subcommand: {sub!r})"
        # Bypass patterns: interpreter -c, eval, find -delete, awk system(),
        # heredoc-to-interpreter, head-position substitution, etc. Runs
        # AFTER train-sh-write but BEFORE the head-token deny so a wrapper
        # like `bash -c rm /x` is rejected at this layer, not the next.
        ok, reason = check_bypass_patterns(sub)
        if not ok:
            return False, f"{reason} (subcommand: {sub!r})"
        head, tokens = head_token(sub)
        if not head:
            continue
        if head in DENY_HEADS:
            return False, f"command '{head}' is on the deny list (subcommand: {sub!r})"
        if head == "git" and len(tokens) >= 2 and tokens[1] in DENY_GIT_SUBCOMMANDS:
            return False, f"'git {tokens[1]}' is on the deny list (subcommand: {sub!r})"
    return True, ""


def main():
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        # Fail-loud (Harness §二): if we can't parse the hook input we don't
        # know what we're allowing. Block.
        print(f"claude_bash_guard: cannot parse hook input: {e}", file=sys.stderr)
        sys.exit(2)

    command = payload.get("tool_input", {}).get("command", "")
    if not isinstance(command, str) or not command.strip():
        # Defensive: empty Bash call — let it through (it'll no-op).
        sys.exit(0)

    ok, reason = check_command(command)
    if ok:
        sys.exit(0)

    # exit 2 + stderr is shown to the model so it can recover / re-plan.
    print(
        f"BLOCKED by yolo_selftrainer safety guard: {reason}\n"
        f"This guard runs because the autonomous loop uses "
        f"--dangerously-skip-permissions. If this command is genuinely needed, "
        f"a human must edit scripts/claude_bash_guard.py to allow it.",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
