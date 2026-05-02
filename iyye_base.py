#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
Iyye Base Classes - shared by main orchestrator, streams, and IO modules.
"""

import re
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

log = logging.getLogger("Iyye")

PROJECT_ROOT = Path(__file__).parent.resolve()

_IO_HISTORY_ROOT = PROJECT_ROOT / "io_history"
_STREAMS_HISTORY_ROOT = PROJECT_ROOT / "streams_history"


class BaseSensorQueue(deque):
    """
    Minimal wrapper around ``deque`` used for all input queues.
    """

    def __init__(self, name: str, maxlen: int = 10_000):
        self.name = name
        self._history_dir: Path | None = None
        super().__init__([], maxlen=maxlen)

    def _get_history_dir(self) -> Path:
        if self._history_dir is None:
            self._history_dir = _IO_HISTORY_ROOT / self.name
            self._history_dir.mkdir(parents=True, exist_ok=True)
        return self._history_dir

    def push(self, payload: Any) -> None:
        """Append a new datum with timestamp and persist to io_history."""
        ts = datetime.now(timezone.utc).isoformat()
        entry = {"ts": ts, "data": payload}
        self.append(entry)
        log.debug("Sensor %s – pushed %r", self.name, payload)
        try:
            day = ts[:10]  # YYYY-MM-DD
            log_file = self._get_history_dir() / f"{day}.txt"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"{ts}\t{payload}\n")
        except Exception as exc:
            log.warning("Sensor %s – failed to write io_history: %s", self.name, exc)

    def pop_all(self) -> List[Dict[str, Any]]:
        """Return **all** queued items and empty the queue."""
        items = list(self)
        self.clear()
        return items


class BaseActuator:
    """
    Very small contract for all output devices (actuators).

    Concrete actuators implement ``_do_actuate(payload: str)`` to deliver
    a message to their output device.  The public ``actuate()`` method adds
    guardrails (dedup + raw-data suppression) before calling ``_do_actuate()``.
    ``send()`` is an alias for ``actuate()``.

    Built-in dedup: identical messages are suppressed for ``_DEDUP_WINDOW``
    seconds (default 120).  Subclasses can override the window or set it to 0
    to disable.
    """

    _DEDUP_WINDOW: float = 120.0  # seconds

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    # Payloads matching this pattern are almost certainly raw data structures
    # (memory search results, JSON blobs) that should never reach the user.
    _RAW_DATA_MARKER = re.compile(
        r"'(?:id|provenance|time_frame|metadata)':\s", re.IGNORECASE
    )

    def _do_actuate(self, payload: str) -> Any:
        """Deliver *payload* to the output device.  Subclasses **must** override."""
        raise NotImplementedError

    def actuate(self, payload: str) -> Any:
        """Send *payload* through guardrails (dedup + raw-data suppression),
        then deliver via ``_do_actuate()``.

        All callers — trusted streams, brain, LLM-generated code — get the
        same protection automatically.
        """
        # Reject payloads that are clearly raw data dumps
        if self._RAW_DATA_MARKER.search(str(payload)):
            import logging
            logging.getLogger("Iyye").warning(
                "Actuator %s: suppressed raw-data payload (%d chars)",
                getattr(self, 'name', type(self).__name__), len(str(payload)),
            )
            return None
        now = time.monotonic()
        recent = getattr(self, '_recent_payloads', None)
        if recent is None:
            self._recent_payloads: dict[str, float] = {}
            recent = self._recent_payloads
        window = self._DEDUP_WINDOW
        if window > 0:
            last_sent = recent.get(payload)
            if last_sent is not None and now - last_sent < window:
                return None  # suppressed duplicate
        recent[payload] = now
        # Evict stale entries to bound memory
        if len(recent) > 200:
            cutoff = now - window
            self._recent_payloads = {k: v for k, v in recent.items() if v > cutoff}
        return self._do_actuate(payload)

    def send(self, payload: str) -> Any:
        """Alias for :meth:`actuate` — kept for backward compatibility."""
        return self.actuate(payload)

    def is_safe_to_stop(self) -> bool:
        """Check if actuator can be safely stopped."""
        return True

    def graceful_stop(self) -> None:
        """Gracefully stop the actuator."""
        pass


class ProcessingStream(ABC):
    """
    Base class for all processing streams (conscious and subconscious).

    HLD: "Execution stream is a python code that has history of inputs,
    including the latest input in progress/being processed, history of
    outputs and current state."
    """

    def __init__(self, name: str = "unnamed_stream"):
        self.name = name
        self.priority: int = 1
        self.input_history: List[Dict[str, Any]] = []
        self.output_history: List[Dict[str, Any]] = []
        self.activity_log: List[str] = []
        self.is_conscious: bool = False
        self.alignment_scores: Dict[str, float] = {}
        self._checkpoint: int = 0
        self._in_critical_section: bool = False
        self._stop_requested: bool = False
        self._can_be_conscious: bool = True  # Most streams can be conscious
        self._current_input: Optional[Dict[str, Any]] = None
        self._state: Dict[str, Any] = {}  # Stream-specific state
        self._checkpoint_pause_requested: bool = False
        self.urgency: float = 0.0
        self._last_conscious_tick: int = 0  # Required for attention stream

    def begin_critical_section(self) -> None:
        """Mark start of non-interruptible code block."""
        self._in_critical_section = True

    def end_critical_section(self) -> None:
        """Mark end of non-interruptible code block."""
        self._in_critical_section = False
        self.checkpoint()

    def request_stop(self) -> None:
        """Request stream to stop at next safe point."""
        self._stop_requested = True

    def can_stop_safely(self) -> bool:
        """Return True if the stream can be stopped externally right now.
        Only blocked by an active critical section; _stop_requested is a signal
        *to* the stream and has no bearing on whether it is safe to stop it."""
        return not self._in_critical_section

    def request_checkpoint_pause(self) -> None:
        """Request stream to pause at next checkpoint for external inspection."""
        self._checkpoint_pause_requested = True

    def resume_from_checkpoint_pause(self) -> None:
        """Allow stream to resume after checkpoint pause."""
        self._checkpoint_pause_requested = False

    def checkpoint(self) -> int:
        """
        Create a safe checkpoint for cooperative multitasking stop.
        HLD: "It runs in a mode similar to cooperative multitasking,
        permitting external stops at safe checkpoints."
        """
        self._checkpoint += 1

        # Pause for external inspection if requested
        while getattr(self, '_checkpoint_pause_requested', False):
            time.sleep(0.001)  # Brief yield

        if self._stop_requested and not self._in_critical_section:
            raise StopIteration("Stream stop requested at checkpoint")
        return self._checkpoint

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Optional[Any]:
        """
        Process inputs and optionally produce output.
        Must checkpoint safely at regular intervals.

        Args:
            context: Contains sensors_data, streams, memory, adenosine, etc.

        Returns:
            Optional result or command
        """
        ...

    def add_to_log(self, action: str) -> None:
        """Record an action/thought to the activity log."""
        ts = datetime.now(timezone.utc).isoformat()
        entry = f"[{ts}] {action}"
        self.activity_log.append(entry)
        # Keep log bounded
        if len(self.activity_log) > 1000:
            self.activity_log = self.activity_log[-500:]
        log.debug("Stream %s: %s", self.name, action[:100])
        try:
            day = ts[:10]
            history_dir = _STREAMS_HISTORY_ROOT / self.name
            history_dir.mkdir(parents=True, exist_ok=True)
            log_file = history_dir / f"{day}.txt"
            with log_file.open("a", encoding="utf-8") as f:
                f.write(f"{entry}\n")
        except Exception as exc:
            import traceback as _tb
            log.warning(
                "Stream %s – failed to write streams_history (path=%s): %s\n%s",
                self.name,
                str(_STREAMS_HISTORY_ROOT / self.name),
                exc,
                _tb.format_exc(),
            )

    def add_input(self, input_data: Any, source: str = "unknown") -> None:
        """Add an input to history."""
        entry = {
            "data": input_data,
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.input_history.append(entry)
        self._current_input = entry
        # Keep history bounded
        if len(self.input_history) > 100:
            self.input_history = self.input_history[-50:]

    def add_output(self, output_data: Any, target: str = "unknown") -> None:
        """Add an output to history."""
        entry = {
            "data": output_data,
            "target": target,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.output_history.append(entry)
        # Keep history bounded
        if len(self.output_history) > 100:
            self.output_history = self.output_history[-50:]

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for persistence."""
        return {
            "name": self.name,
            "priority": self.priority,
            "is_conscious": self.is_conscious,
            "alignment_scores": self.alignment_scores,
            "checkpoint": self._checkpoint,
            "state": self._state,
            "recent_log": self.activity_log[-20:],
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from persistence."""
        self.priority = state.get("priority", self.priority)
        self._state = state.get("state", {})


__all__ = ['PROJECT_ROOT', 'BaseSensorQueue', 'BaseActuator', 'ProcessingStream']
