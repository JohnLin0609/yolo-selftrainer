"""pytest-bdd step definitions for tests/features/sandbox_isolation.feature.

The whole feature is auto-skipped by tests/conftest.py until
`scripts/sandbox.run_in_sandbox` exists. Step functions therefore
**lazy-import** the runtime — top-level imports would crash collection.

Contract the implementation must satisfy:

    scripts/sandbox.run_in_sandbox(
        project_dir: Path,          # path to projects/<name>/
        command: str,               # the agent's Bash command
        framework_root: Path,       # path to the repo root (read-only mount)
    ) -> subprocess.CompletedProcess

The CompletedProcess.returncode is what the scenarios assert on.
Whether the implementation uses containers, namespaces (unshare), or
something else is opaque to the test — only the observable behavior
(escape attempt fails, host files unchanged) matters.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

scenarios("../features/sandbox_isolation.feature")


# ─── Lazy-import wrapper ─────────────────────────────────────────────

def _run_in_sandbox(*args, **kwargs):
    # Imported at call-time, not collection-time. With the conftest
    # skipif active, this function is never called on `main` and the
    # missing module never trips collection.
    from sandbox import run_in_sandbox  # type: ignore[import-not-found]
    return run_in_sandbox(*args, **kwargs)


# ─── Fixtures (per-scenario tmp project tree) ────────────────────────

@pytest.fixture
def sandbox_workspace(tmp_path, repo_root):
    """A tmp tree mirroring the project layout the sandbox expects.

    Returns dict with keys: framework_root, project_dir, sibling_dir.
    The sibling is created as a peer of project_dir so the "cannot
    delete sibling project" scenario has a real target.
    """
    framework_root = tmp_path / "framework"
    framework_root.mkdir()
    # Copy just the scripts/ tree we care about — small and read-only.
    shutil.copytree(repo_root / "scripts", framework_root / "scripts")

    projects = tmp_path / "projects"
    projects.mkdir()
    project_dir = projects / "sandbox_e2e"
    project_dir.mkdir()
    (project_dir / "events.jsonl").write_text("")

    sibling_dir = projects / "other"
    sibling_dir.mkdir()
    (sibling_dir / "events.jsonl").write_text("placeholder\n")

    return {
        "framework_root": framework_root,
        "project_dir": project_dir,
        "sibling_dir": sibling_dir,
    }


# ─── Background steps ────────────────────────────────────────────────

@given(parsers.parse('a sandboxed project at "{name}"'), target_fixture="project_ctx")
def _given_project(name, sandbox_workspace):
    return {"name": name, **sandbox_workspace}


@given(parsers.parse('a sibling project "{name}" must not be touched'))
def _given_sibling_intact(name, project_ctx):
    sibling = project_ctx["sibling_dir"]
    assert sibling.exists(), f"fixture failed to create sibling: {sibling}"
    assert sibling.name == name


@given(parsers.parse('the sibling project "{name}" has secret content in its events.jsonl'))
def _given_sibling_secret(name, project_ctx):
    sibling = project_ctx["sibling_dir"]
    assert sibling.name == name
    (sibling / "events.jsonl").write_text("OPERATOR_SECRET=42\n")


# ─── Action step ─────────────────────────────────────────────────────

@when(parsers.parse("the in-sandbox command runs:\n{cmd}"), target_fixture="result")
def _when_run(cmd, project_ctx):
    return _run_in_sandbox(
        project_dir=project_ctx["project_dir"],
        command=cmd.strip(),
        framework_root=project_ctx["framework_root"],
    )


# ─── Assertion steps ────────────────────────────────────────────────

@then(parsers.parse('the sibling project "{name}" still exists'))
def _then_sibling_exists(name, project_ctx):
    sibling = project_ctx["sibling_dir"]
    assert sibling.name == name
    assert sibling.exists(), f"sibling project was deleted: {sibling}"
    assert (sibling / "events.jsonl").exists(), "sibling lost its events.jsonl"


@then(parsers.parse('the host file "{path}" does not exist'))
def _then_host_file_absent(path):
    # Path is interpreted relative to "/" — these scenarios use absolute
    # paths like /tmp/sandbox_escape_marker.
    assert not Path(path).exists(), f"host file was created (escape!): {path}"


@then(parsers.parse("the in-sandbox command exits non-zero"))
def _then_nonzero(result):
    assert result.returncode != 0, (
        f"expected non-zero exit; got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@then(parsers.parse("the in-sandbox command exits zero"))
def _then_zero(result):
    assert result.returncode == 0, (
        f"expected zero exit; got {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@then(parsers.parse('the file "{relpath}" exists within the project'))
def _then_in_project(relpath, project_ctx):
    p = project_ctx["project_dir"] / relpath
    assert p.exists(), f"expected file in project: {p}"
