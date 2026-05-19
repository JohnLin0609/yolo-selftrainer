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
