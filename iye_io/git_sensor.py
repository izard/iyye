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
Git Sensor - Monitors Iyye's own prompts/ and streams/ directories via MCP.

HLD: "MCP client connected to local git server that contains IYYE source code."

Polls io/mcp_git.py for new commits that touched prompts/ or streams/ and
pushes one queue item per poll cycle when changes are detected.  The full
MCP tool set (read_file, write_file, commit_changes, …) is accessible via
self.mcp_client.call_tool() for use by conscious streams that want to
perform self-modification.
"""

import logging
from typing import Any, Dict

from iyye_base import PROJECT_ROOT, BaseSensorQueue
from mcp_client import MCPStdioClient, MCPSensorWrapper

log = logging.getLogger("Iyye.Sensors.Git")

_VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
_MCP_SCRIPT = str(PROJECT_ROOT / "iyye_io" / "mcp_git.py")


class GitSensor(MCPSensorWrapper, BaseSensorQueue):
    """
    Detects changes to prompts/ and streams/ via the local git MCP server.

    Each poll calls the MCP tool `poll_changes` which returns only commits
    that arrived since the previous call.  A non-empty result is pushed to
    the queue so that processing streams can react to self-modifications.

    The full MCP interface (read_file, write_file, commit_changes, get_diff,
    list_files) is available through self.mcp_client.call_tool() for
    conscious streams that perform prompt or stream self-editing.
    """

    def __init__(
        self,
        mcp_script_path: str = _MCP_SCRIPT,
        poll_interval: float = 10.0,
    ):
        mcp_client = MCPStdioClient(
            command=_VENV_PYTHON,
            args=[mcp_script_path],
        )

        startup_ok = mcp_client.start()
        if not startup_ok:
            log.error("Failed to start git MCP server — sensor will be inactive")

        super().__init__(
            name="GitSensor",
            mcp_client=mcp_client,
            poll_tool="poll_changes",
            poll_args={"limit": 10},
            poll_interval=poll_interval,
            maxlen=1_000,
        )

        self._startup_failed = not startup_ok

    # ------------------------------------------------------------------
    # Override MCPSensorWrapper.poll to only push when there are new commits
    # ------------------------------------------------------------------

    # Commit prefixes authored by Iyye itself — ignore to avoid a feedback
    # loop where StreamFactory commits trigger GitSensor which triggers a
    # generated git-curiosity stream which reacts to its own auto-generated
    # code as though it were a meaningful external event.
    _SELF_AUTHORED_PREFIXES = (
        "StreamFactory:",
        "auto-generated",
    )

    def poll(self) -> None:
        """Push a queue item only when new commits touching prompts/ or streams/ appear."""
        if self._startup_failed:
            return

        import time
        now = time.time()
        if now - self._last_poll < self.poll_interval:
            return
        self._last_poll = now

        result = self.mcp_client.call_tool("poll_changes", {"limit": 10})
        if not result or "error" in result:
            if result:
                log.warning("Git MCP poll error: %s", result["error"])
            return

        if result.get("count", 0) > 0:
            # Filter out self-authored commits to break the feedback loop.
            raw_commits = result.get("commits", [])
            external = [
                c for c in raw_commits
                if not self._is_self_authored(c)
            ]
            if not external:
                log.debug(
                    "GitSensor: %d commit(s) all self-authored — skipped",
                    len(raw_commits),
                )
                return
            result = {**result, "commits": external, "count": len(external)}
            log.info(
                "GitSensor: %d new commit(s) in prompts/ or streams/: %s",
                result["count"],
                result["commits"],
            )
            self.push(result)

    @classmethod
    def _is_self_authored(cls, commit_line: str) -> bool:
        """True if this commit message was authored by Iyye itself."""
        # Commit lines look like "7bf1027 StreamFactory: auto-generated ..."
        # Strip the leading hash to get the message.
        parts = commit_line.split(None, 1)
        msg = parts[1] if len(parts) > 1 else commit_line
        return any(msg.startswith(p) for p in cls._SELF_AUTHORED_PREFIXES)

    # ------------------------------------------------------------------
    # Convenience wrappers for self-modification tools
    # ------------------------------------------------------------------

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file from prompts/ or streams/."""
        return self.mcp_client.call_tool("read_file", {"path": path})

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write/create a file inside prompts/ or streams/."""
        return self.mcp_client.call_tool("write_file", {"path": path, "content": content})

    def commit_changes(self, message: str, paths: list = None) -> Dict[str, Any]:
        """Stage and commit changes inside prompts/ and/or streams/."""
        args: Dict[str, Any] = {"message": message}
        if paths:
            args["paths"] = paths
        return self.mcp_client.call_tool("commit_changes", args)

    def list_files(self, directory: str) -> Dict[str, Any]:
        """List files in prompts/ or streams/."""
        return self.mcp_client.call_tool("list_files", {"directory": directory})

    def get_diff(self, path: str = None) -> Dict[str, Any]:
        """Get working-tree diff for prompts/ and streams/."""
        args = {"path": path} if path else {}
        return self.mcp_client.call_tool("get_diff", args)

    def stop_collection(self) -> None:
        if hasattr(self, "mcp_client") and self.mcp_client:
            self.mcp_client.stop()
        log.info("GitSensor stopped")
