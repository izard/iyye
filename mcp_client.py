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
# mcp_client.py - COMPLETE REWRITE
#!/usr/bin/env python3
"""
MCP Client wrapper for sensor/actuator communication.
HLD: "Sensor inputs are available as local MCP stdio clients."

CORRECTED: MCP stdio uses newline-delimited JSON-RPC, not Content-Length framing.
"""

import asyncio
import json
import os
import select
import subprocess
import time
from typing import Any, Dict, Optional, List
from collections import deque
from datetime import datetime
import logging

# Import base classes from main orchestrator
from iyye_base import BaseSensorQueue

log = logging.getLogger("Iyye.MCP")


class MCPStdioClient:
    """
    Wrapper for MCP stdio-based communication with external services.
    Implements MCP JSON-RPC protocol with NEWLINE delimitation (not Content-Length).
    
    Per MCP spec 2024-11-05:
    - Messages are delimited by newlines
    - Messages MUST NOT contain embedded newlines
    - Server MAY write to stderr for logging
    """
    
    def __init__(self, command: str, args: list = None, env: dict = None,
                 timeout: float = 5.0):
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self._process: Optional[subprocess.Popen] = None
        self._initialized = False
        self._request_id = 0
        self._protocol_version = "2024-11-05"
        
    def start(self) -> bool:
        """Start the MCP server process."""
        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Discard — never let a full pipe block the child
                env={**os.environ, **self.env},
                bufsize=1,  # Line buffered for newline delimitation
                text=True,  # Use text mode for newline handling
            )
            log.info("Started MCP server: %s", self.command)

            # Initialize MCP connection; only mark initialized on success.
            result = self._initialize_mcp()
            if not result:
                log.error("MCP handshake failed for %s — marking as not initialized", self.command)
                self._initialized = False
                return False

            self._initialized = True
            return True
        except Exception as e:
            log.error("Failed to start MCP server %s: %s", self.command, e)
            self._initialized = False
            return False
    
    def _initialize_mcp(self) -> Dict[str, Any]:
        """Send MCP initialization request and wait for response."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": self._protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "iyye", "version": "1.0.0"}
            }
        }
        self._send_message(init_request)
        
        # Read initialization response
        response = self._read_message()
        if response and "result" in response:
            log.info("MCP initialized with protocol version: %s", 
                    response["result"].get("protocolVersion", "unknown"))
            
            # Send initialized notification
            self._send_message({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })
            
            return response["result"]
        
        log.error("MCP initialization failed")
        return {}
        
    def _send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON-RPC message with newline delimitation."""
        if not self._process:
            raise RuntimeError("MCP server not started")
        
        content = json.dumps(message)
        # MCP stdio uses newline delimitation (NOT Content-Length)
        self._process.stdin.write(content + "\n")
        self._process.stdin.flush()
        
    def _is_process_alive(self) -> bool:
        """Return True if the subprocess is still running."""
        return self._process is not None and self._process.poll() is None

    # How long to wait for one JSON-RPC response before declaring the child hung.
    _READ_TIMEOUT = 15.0  # seconds

    def _read_message(self) -> Optional[Dict[str, Any]]:
        """Read one JSON-RPC response line with a hard timeout.

        Uses select() to avoid blocking forever on an alive-but-stuck child.
        On timeout or EOF the client is marked uninitialized so the next poll
        triggers a restart.
        """
        if not self._process:
            raise RuntimeError("MCP server not started")

        try:
            ready, _, _ = select.select(
                [self._process.stdout], [], [], self._READ_TIMEOUT
            )
            if not ready:
                log.warning(
                    "MCP server %s did not respond within %.0fs — marking dead",
                    self.command, self._READ_TIMEOUT,
                )
                self._initialized = False
                return None

            line = self._process.stdout.readline()
            if not line:
                # EOF — subprocess exited.
                log.warning("MCP server process exited (EOF on stdout)")
                self._initialized = False
                return None
            return json.loads(line.strip())
        except json.JSONDecodeError as e:
            log.error("Failed to parse MCP message: %s", e)
            return None

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool and return the result.
        Implements MCP JSON-RPC protocol.
        """
        if not self._initialized:
            raise RuntimeError("MCP server not initialized")
        
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            # Send request
            self._send_message(request)
            
            # Read response
            response = self._read_message()
            
            if response is None:
                return {"error": "No response from MCP server"}
            
            if "error" in response:
                log.error("MCP tool error: %s", response["error"])
                return {"error": response["error"]}

            result = response.get("result", {})

            # FastMCP wraps the tool's return value inside
            # result["content"][0]["text"] as a JSON string.
            # Unwrap it here so every caller receives the actual tool data.
            if isinstance(result, dict):
                if result.get("isError"):
                    content = result.get("content") or []
                    err_text = (content[0].get("text", "tool error")
                                if content else "tool error")
                    log.error("MCP tool %s isError: %s", tool_name, err_text[:200])
                    return {"error": err_text}
                content = result.get("content") or []
                if (content and isinstance(content[0], dict)
                        and content[0].get("type") == "text"):
                    try:
                        return json.loads(content[0]["text"])
                    except (json.JSONDecodeError, TypeError):
                        return {"text": content[0]["text"]}

            return result
            
        except Exception as e:
            log.error("MCP call failed: %s", e)
            return {"error": str(e)}
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/list",
            "params": {}
        }
        
        try:
            self._send_message(request)
            response = self._read_message()
            if response and "tools" in response.get("result", {}):
                return response["result"]["tools"]
        except Exception as e:
            log.error("Failed to list tools: %s", e)
        
        return []
    
    def stop(self) -> None:
        """Stop the MCP server process."""
        if self._process:
            # Send shutdown notification
            shutdown = {
                "jsonrpc": "2.0",
                "method": "notifications/shutdown"
            }
            try:
                self._send_message(shutdown)
            except Exception:
                pass
            
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
            self._initialized = False
            log.info("Stopped MCP server: %s", self.command)


class MCPSensorWrapper:
    """
    Mixin that adds MCP polling to a BaseSensorQueue subclass.
    Concrete sensors should inherit (MCPSensorWrapper, BaseSensorQueue).
    HLD: Sensors are MCP stdio clients.
    """

    def __init__(self, name: str, mcp_client: MCPStdioClient,
                 poll_tool: str, poll_args: dict = None,
                 poll_interval: float = 1.0, maxlen: int = 10_000, **kwargs):
        super().__init__(name=name, maxlen=maxlen)
        self.mcp_client = mcp_client
        self.poll_tool = poll_tool
        self.poll_args = poll_args or {}
        self.poll_interval = poll_interval
        self._last_poll = 0
        self._consecutive_errors = 0

    # After this many consecutive failures restart even if the process is alive.
    _MAX_ERRORS_BEFORE_RESTART = 3

    def _restart_mcp(self) -> bool:
        """Kill the current subprocess (if any) and start a fresh one."""
        try:
            self.mcp_client.stop()
        except Exception:
            pass
        ok = self.mcp_client.start()
        if ok:
            log.info("MCP server for %s restarted successfully", self.name)
            self._consecutive_errors = 0
        else:
            log.warning("MCP server restart failed for %s", self.name)
        return ok

    def poll(self) -> None:
        """Poll MCP server and push results to queue."""
        # Restart if the subprocess is not initialized (dead or timed-out).
        if not getattr(self.mcp_client, '_initialized', False):
            self._restart_mcp()
            return  # give it one tick before polling

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_poll < self.poll_interval:
            return
        self._last_poll = current_time

        try:
            result = self.mcp_client.call_tool(self.poll_tool, self.poll_args)
        except RuntimeError:
            # _initialized was flipped to False by _read_message (EOF or timeout)
            self._consecutive_errors += 1
            log.warning("MCP server %s unresponsive (%d/%d)",
                        self.name, self._consecutive_errors, self._MAX_ERRORS_BEFORE_RESTART)
            return

        if result and "error" not in result:
            self._consecutive_errors = 0
            if isinstance(result, list):
                for item in result:
                    self.push(item)
            else:
                self.push(result)
        elif result and "error" in result:
            self._consecutive_errors += 1
            log.warning("MCP sensor %s poll error (%d/%d): %s",
                        self.name, self._consecutive_errors,
                        self._MAX_ERRORS_BEFORE_RESTART, result["error"])
            # Restart after too many consecutive errors, even if process is alive —
            # it may be alive but stuck (e.g., an internal deadlock or Telegram error loop).
            if self._consecutive_errors >= self._MAX_ERRORS_BEFORE_RESTART:
                log.warning("MCP sensor %s: too many errors, forcing restart", self.name)
                self.mcp_client._initialized = False  # triggers restart on next poll



