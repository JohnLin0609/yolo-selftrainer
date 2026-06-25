#!/usr/bin/env python3
"""OS-level Bash sandbox via bubblewrap (`bwrap`).

Boundary 1's predicate (`scripts/claude_bash_guard.py`) is a string-matching
denylist + bypass-rejection layer. It catches the obvious bypasses but
cannot offer hard guarantees. This module is the OS-level companion: even
if a future bypass slips past the predicate, the sandbox confines side
effects to `project_dir` and prevents network access.

Used by `scripts/run_agent.py` for the Bash tool dispatch in agent mode.
Not wired into claude CLI mode (claude controls its own Bash execution;
sandboxing that path would require launching claude itself inside bwrap,
which is a separate piece of work).

Backend: bubblewrap only (cf. docs/architecture.md "Boundary 1"). bwrap
ships in the `bubblewrap` package on Debian/Ubuntu and is present on
every common CI runner. If unavailable, `is_available()` returns False
and the sandbox-isolation tests skip gracefully.

The mount layout pins what is and isn't writable:
  /                      read-only (binaries, libs, /etc/passwd for uid lookups)
  {framework_root}       read-only (source code; agent can't modify it)
  {project_dir}          read-write (the agent's scratch space)
  /tmp                   tmpfs (in-sandbox; host /tmp untouched)
  /proc, /dev            standard bwrap fixtures
Plus:
  --unshare-net          no network
  --unshare-pid          own pid namespace (cannot kill host processes)
  --die-with-parent      cleanup if the python parent dies
  --new-session          detach from controlling terminal
  --chdir {project_dir}  command starts in the project root
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Final


# Default timeout in seconds for a single sandbox command. The agent can
# launch nohup'd training jobs that exit quickly from this wrapper's POV
# (they detach), so this only bounds synchronous tool calls.
DEFAULT_TIMEOUT_S: Final[float] = 60.0


@lru_cache(maxsize=1)
def _backend_path() -> str | None:
    """Locate the sandbox backend on PATH. Cached — runs once per process.

    Currently only bubblewrap (`bwrap`). Returning None signals that the
    sandbox is unavailable; callers should fall back or skip.
    """
    return shutil.which("bwrap")


def is_available() -> bool:
    """True iff the sandbox runtime can be used right now."""
    return _backend_path() is not None


def _build_argv(
    project_dir: Path,
    command: str,
    framework_root: Path,
) -> list[str]:
    """Compose the bwrap argv. Extracted for unit-testability."""
    bwrap = _backend_path()
    if bwrap is None:
        # Defensive — callers should have checked is_available() first.
        raise RuntimeError(
            "no sandbox backend available; install `bubblewrap` (apt-get "
            "install bubblewrap) or check is_available() before calling "
            "run_in_sandbox()"
        )
    project_dir = Path(project_dir).resolve()
    framework_root = Path(framework_root).resolve()
    # bwrap processes mount operations in argv order. The tmpfs on /tmp
    # MUST land before any --bind whose source path lives under /tmp
    # (tests use tmp_path for both project_dir and framework_root). If
    # we tmpfs /tmp AFTER the binds, the tmpfs overlays our project mount
    # and `--chdir project_dir` fails with "No such file or directory".
    argv: list[str] = [
        bwrap,
        # Default everything read-only — explicit binds override below.
        "--ro-bind", "/", "/",
        # In-sandbox /tmp is a tmpfs: writes to /tmp/anything by the
        # in-sandbox command go to ephemeral storage. Comes BEFORE any
        # bind whose path is rooted at /tmp (see comment above).
        "--tmpfs", "/tmp",
        # Standard bwrap-managed fixtures (NOT host /dev / /proc).
        "--dev", "/dev",
        "--proc", "/proc",
        # Self-document the framework as read-only (under the blanket
        # `/` mount this is redundant; explicit so future tightening of
        # the root mount doesn't break the contract).
        "--ro-bind", str(framework_root), str(framework_root),
        # The agent's scratch dir is the ONLY writable real path. Layers
        # on top of the tmpfs when project_dir is under /tmp (tests).
        "--bind", str(project_dir), str(project_dir),
        # Isolation flags.
        "--unshare-net",
        "--unshare-pid",
        "--die-with-parent",
        "--new-session",
        "--chdir", str(project_dir),
        # /bin/sh -c carries the agent's command. /bin/sh is bound by the
        # `/` read-only mount; no separate setup needed.
        "/bin/sh", "-c", command,
    ]
    return argv


def run_in_sandbox(
    project_dir: Path,
    command: str,
    framework_root: Path,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    """Execute `command` inside a bwrap sandbox. Returns CompletedProcess.

    Side effects from `command` are confined to `project_dir`, the
    sandbox-private `/tmp`, and (within the namespace) `/proc` and `/dev`.
    Attempts to touch anything else either fail (read-only mount) or land
    in the tmpfs that disappears when bwrap exits.

    Raises RuntimeError if the sandbox backend is not available — callers
    should check is_available() first or be prepared to fall back to
    direct host execution.
    """
    argv = _build_argv(project_dir, command, framework_root)
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


if __name__ == "__main__":
    # Convenience CLI: `python3 scripts/sandbox.py --check` reports
    # backend + version. Useful for verifying a host before wiring the
    # sandbox into a session.
    import sys
    if "--check" in sys.argv:
        path = _backend_path()
        if path is None:
            print("sandbox: NOT AVAILABLE (no bwrap on PATH)")
            sys.exit(1)
        v = subprocess.run([path, "--version"], capture_output=True, text=True)
        print(f"sandbox: bubblewrap at {path}")
        print(f"         {v.stdout.strip() or '(no version output)'}")
        sys.exit(0)
    print(__doc__)
