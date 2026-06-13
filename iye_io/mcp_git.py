#!/usr/bin/env python3
# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Local MCP server for Iyye self-introspection and self-modification.

Exposes git operations restricted to the prompts/ and streams/ directories
so that Iyye can read, modify, and commit its own prompt templates and
processing stream code — and nothing else.

Tools:
  get_status()                    — staged/unstaged changes in allowed dirs
  poll_changes(limit)             — new commits since last poll (stateful)
  list_files(directory)           — files in prompts/ or streams/
  read_file(path)                 — read a file (allowed dirs only)
  get_diff(path?)                 — working-tree diff (allowed dirs only)
  write_file(path, content)       — overwrite/create a file (allowed dirs only)
  commit_changes(message, paths?) — git add + commit (allowed dirs only)

Required env:
  (none — operates on the local repo)

Run directly:
  python io/mcp_git.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_DIRS = frozenset({"prompts", "streams"})

mcp = FastMCP("git-mcp")

# Tracks the last commit hash seen by poll_changes so it returns only new work.
_last_seen_commit: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + list(args),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def _resolve_allowed(path: str) -> Path:
    """
    Resolve *path* relative to REPO_ROOT and verify it is inside one of the
    ALLOWED_DIRS.  Raises ValueError if the path escapes the allowed scope.
    """
    resolved = (REPO_ROOT / path).resolve()
    for allowed in ALLOWED_DIRS:
        try:
            resolved.relative_to((REPO_ROOT / allowed).resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(
        f"Path '{path}' is outside allowed directories {sorted(ALLOWED_DIRS)}"
    )


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_status() -> dict[str, Any]:
    """
    Return current git status restricted to prompts/ and streams/.

    Result fields:
      commit        — current HEAD hash
      changed_files — list of porcelain status lines (e.g. "M streams/foo.py")
      has_changes   — True when the working tree or index differs from HEAD
    """
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "prompts/", "streams/")
    lines = [l for l in status.stdout.splitlines() if l.strip()]
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "changed_files": lines,
        "has_changes": bool(lines),
    }


@mcp.tool()
def poll_changes(limit: int = 10) -> dict[str, Any]:
    """
    Return commits that touched prompts/ or streams/ since the last call.

    On the first call returns up to *limit* most-recent such commits.
    Subsequent calls return only commits that arrived since the previous poll.

    Result fields:
      count   — number of new commits
      commits — list of "hash subject" strings (--oneline format)
      head    — current HEAD hash
    """
    global _last_seen_commit

    limit = max(1, min(limit, 100))
    head = _git("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"count": 0, "commits": [], "head": None}

    current = head.stdout.strip()

    if _last_seen_commit and _last_seen_commit != current:
        log_result = _git(
            "log", f"{_last_seen_commit}..HEAD",
            "--oneline", "--no-merges",
            "--", "prompts/", "streams/",
        )
    elif not _last_seen_commit:
        log_result = _git(
            "log", f"-{limit}",
            "--oneline", "--no-merges",
            "--", "prompts/", "streams/",
        )
    else:
        # HEAD unchanged since last poll
        _last_seen_commit = current
        return {"count": 0, "commits": [], "head": current}

    _last_seen_commit = current
    commits = [l for l in log_result.stdout.splitlines() if l.strip()]
    return {"count": len(commits), "commits": commits, "head": current}


@mcp.tool()
def list_files(directory: str) -> dict[str, Any]:
    """
    List files inside prompts/ or streams/.

    Args:
      directory — must be "prompts" or "streams"
    """
    if directory not in ALLOWED_DIRS:
        return {"error": f"directory must be one of {sorted(ALLOWED_DIRS)}"}
    d = REPO_ROOT / directory
    files = sorted(f.name for f in d.iterdir() if f.is_file()) if d.exists() else []
    return {"directory": directory, "files": files}


@mcp.tool()
def read_file(path: str) -> dict[str, Any]:
    """
    Read a file from prompts/ or streams/.

    Args:
      path — relative to repo root, e.g. "prompts/chat_response.md"
    """
    try:
        p = _resolve_allowed(path)
        return {"path": path, "content": p.read_text(encoding="utf-8")}
    except ValueError as exc:
        return {"error": str(exc)}
    except FileNotFoundError:
        return {"error": f"file not found: {path}"}


@mcp.tool()
def get_diff(path: Optional[str] = None) -> dict[str, Any]:
    """
    Return the working-tree diff versus HEAD for prompts/ and streams/.

    Args:
      path — optional specific file (must be inside prompts/ or streams/)
    """
    if path:
        try:
            _resolve_allowed(path)
            result = _git("diff", "HEAD", "--", path)
        except ValueError as exc:
            return {"error": str(exc)}
    else:
        result = _git("diff", "HEAD", "--", "prompts/", "streams/")
    return {"diff": result.stdout, "has_diff": bool(result.stdout.strip())}


@mcp.tool()
def write_file(path: str, content: str) -> dict[str, Any]:
    """
    Write (create or overwrite) a file inside prompts/ or streams/.

    Args:
      path    — relative to repo root, e.g. "streams/my_stream.py"
      content — full file content as a string
    """
    try:
        p = _resolve_allowed(path)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "bytes_written": len(content.encode())}
    except Exception as exc:
        return {"error": f"write failed: {exc}"}


@mcp.tool()
def commit_changes(message: str, paths: Optional[list] = None) -> dict[str, Any]:
    """
    Stage and commit changes inside prompts/ and/or streams/.

    Args:
      message — commit message
      paths   — optional list of specific relative paths to stage;
                if omitted, stages all changes under prompts/ and streams/
    """
    if paths:
        for p in paths:
            try:
                abs_p = _resolve_allowed(p)
            except ValueError as exc:
                return {"error": str(exc)}
            _git("add", "--", str(abs_p))
    else:
        _git("add", "--", "prompts/", "streams/")

    result = _git("commit", "-m", message)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        return {"ok": False, "error": stderr or stdout}

    head = _git("rev-parse", "HEAD")
    return {
        "ok": True,
        "commit": head.stdout.strip(),
        "message": message,
        "output": result.stdout.strip(),
    }


if __name__ == "__main__":
    mcp.run()
