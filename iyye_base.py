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
# -*- coding: utf-8 -*-
"""
Iyye Base Classes - shared by main orchestrator, streams, and IO modules.
"""

import re
import time
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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
        """Return **all** currently queued items and empty the queue.

        Drains via repeated ``popleft()`` (atomic under the GIL) instead of
        ``list(self)`` + ``clear()``: a background collection thread (e.g.
        TelegramSensor's poll loop) can ``push()`` concurrently, and the old
        snapshot-then-clear would silently discard any item appended between the
        two calls.  Here each item is either returned or left for the next
        drain — never dropped."""
        items: List[Dict[str, Any]] = []
        while True:
            try:
                items.append(self.popleft())
            except IndexError:
                break
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


@dataclass
class SleepPhase:
    """One unit of asleep-state housekeeping, owned by a stream/unit.

    ``run(brain)`` performs (a slice of) the phase's work and returns True when
    the phase is finished for this sleep cycle, False to be called again next
    tick (e.g. batched replay).  ``order`` sets the position in the brain's
    sleep pipeline (lower runs earlier)."""
    name: str
    run: Callable[[Any], bool]
    order: int = 100


@dataclass(frozen=True)
class StreamView:
    """Immutable read-only snapshot of a stream for cross-stream queries.

    The communication contract for *inspecting* peers: attention/alignment/
    factory consume views via ``brain.stream_views()`` instead of holding raw
    mutable stream objects, so a stream's internals can change without
    breaking its observers, and observers cannot mutate peers."""
    name: str
    priority: int
    is_conscious: bool
    can_be_conscious: bool
    urgency: float
    alignment_scores: Dict[str, float]
    pending: int                      # unprocessed messages (e.g. chat)
    plan_remaining: Optional[int]     # plan steps left, or None if not planned
    in_critical_section: bool
    last_conscious_tick: int
    is_generated: bool                # backed by an LLM-generated source file
    recent_activity: Tuple[str, ...]
    recent_outputs: Tuple[str, ...]


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
        # Number of activity_log entries that have been trimmed away from the
        # front (see add_to_log).  activity_log[0] therefore corresponds to
        # absolute sequence number ``_log_dropped`` in the never-trimmed log,
        # and ``_log_dropped + len(activity_log)`` is the total ever appended.
        # Consumers that remember "how many entries I have read" must use
        # get_log_since() so a trim cannot strand their cursor past the end.
        self._log_dropped: int = 0
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
        # HLD: "all subconscious streams pause" during winding-down.  Streams
        # that spawn background threads (alignment LLM scoring, LLM start/stop
        # subprocesses) must NOT launch new work while paused, and must drain
        # in-flight threads via settle().  The default settle() joins anything
        # appended to _background_threads — subclasses can override for richer
        # cleanup (e.g. ToM flushing dirty contacts via on_pause()).
        self._paused: bool = False
        self._background_threads: List[threading.Thread] = []

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

    # ------------------------------------------------------------------
    # Pause/settle protocol (used during WINDING_DOWN → ASLEEP)
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Stop initiating new background work.

        Subclasses with background threads (alignment scoring, LLM start/stop)
        check ``self._paused`` before spawning new work.  Cheap to call;
        idempotent.
        """
        self._paused = True

    def resume(self) -> None:
        """Re-enable background work after wakeup."""
        self._paused = False

    def settle(self, timeout_s: float = 5.0) -> bool:
        """Wait for in-flight background threads to complete.

        Default implementation joins every thread appended to
        ``self._background_threads`` with a shared deadline.  Returns True
        if all threads finished, False on timeout.  Subclasses that need
        richer settling (e.g. draining pending results into cache) should
        override and call ``super().settle()`` first.
        """
        deadline = time.monotonic() + timeout_s
        for t in list(self._background_threads):
            if t.is_alive():
                remaining = max(0.0, deadline - time.monotonic())
                t.join(timeout=remaining)
        self._background_threads = [
            t for t in self._background_threads if t.is_alive()
        ]
        return not self._background_threads

    def on_pause(self) -> None:
        """Hook called once after settle() to flush persistent in-memory state.

        Default no-op.  Theory-of-Mind overrides this to flush dirty contacts
        so the special-case ToM flush in the brain's sleep transition isn't
        needed.
        """
        pass

    def sleep_phases(self) -> List["SleepPhase"]:
        """Sleep-housekeeping phases this stream owns (default none).

        HLD: "Asleep state runs only 'housekeeping' subconscious execution
        streams."  A stream returns ordered :class:`SleepPhase` units here; the
        brain's sleep scheduler runs them in priority order while asleep, so the
        *work* is owned by the stream and only the *sequencing* by the brain.
        """
        return []

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
        # Shadow-journal this activity line as a stream_activity event so the
        # event journal is the canonical interleaved record (Phase 1).  The
        # brain attaches its journal to each stream; guarded + best-effort so
        # a stream without a brain (or a journal hiccup) never breaks logging.
        _brain = getattr(self, 'brain', None)
        _journal = getattr(_brain, 'journal', None) if _brain is not None else None
        if _journal is not None:
            try:
                # Carry current alignment scores so sleep's what-if analysis can
                # find close-call decisions straight from the journal (only when
                # scored, to avoid bloating every line with an empty dict).
                _scores = getattr(self, 'alignment_scores', None) or None
                _journal.append('stream_activity', stream=self.name,
                                text=action, alignment_scores=_scores)
            except Exception:
                pass
        # Keep log bounded.  Track how many entries we drop so index-based
        # consumers (replay capture, STM extraction) can detect the
        # compaction and resume from the earliest retained entry instead of
        # silently stranding their cursor past the new, shorter end.
        if len(self.activity_log) > 1000:
            dropped_now = len(self.activity_log) - 500
            self.activity_log = self.activity_log[-500:]
            self._log_dropped += dropped_now
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

    def activity_log_total(self) -> int:
        """Total number of activity_log entries ever appended.

        Counts entries trimmed away by add_to_log as well as those still
        retained, so it is monotonic for the life of the stream and usable
        as an absolute cursor space by get_log_since()."""
        return self._log_dropped + len(self.activity_log)

    def get_log_since(self, seen: int) -> Tuple[List[str], int, int]:
        """Return new activity-log entries for a consumer that last read *seen*.

        *seen* is an absolute count in the same space as activity_log_total()
        (i.e. the value returned as ``new_cursor`` by a previous call).

        Returns ``(entries, lost, new_cursor)`` where:
        - ``entries``    — log lines the consumer has not seen yet (a copy).
        - ``lost``       — number of entries that were trimmed away before the
                           consumer could read them (0 in the normal case).
                           Non-zero only for a stream so busy that add_to_log
                           compacted past the consumer's cursor between reads.
        - ``new_cursor`` — value to pass as *seen* on the next call.

        When ``lost`` > 0 the entries resume from the earliest retained line,
        so capture continues instead of cutting off forever."""
        dropped = self._log_dropped
        total = dropped + len(self.activity_log)
        # A cursor ahead of total can only happen if the log was replaced
        # wholesale; clamp so we never index with a negative offset.
        if seen >= total:
            return [], 0, total
        if seen < dropped:
            # Compaction dropped entries this consumer had not read yet.
            return list(self.activity_log), dropped - seen, total
        return list(self.activity_log[seen - dropped:]), 0, total

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

    def to_view(self) -> "StreamView":
        """Return an immutable snapshot of this stream for peer inspection.

        Each stream decides what it exposes; observers read the view rather
        than reaching into mutable internals (see StreamView)."""
        plan_steps = getattr(self, '_plan_steps', None)
        plan_remaining = (
            max(0, len(plan_steps) - getattr(self, '_current_step', 0))
            if plan_steps is not None else None
        )
        return StreamView(
            name=self.name,
            priority=getattr(self, 'priority', 1),
            is_conscious=bool(self.is_conscious),
            can_be_conscious=bool(getattr(self, '_can_be_conscious', True)),
            urgency=float(getattr(self, 'urgency', 0.0)),
            alignment_scores=dict(getattr(self, 'alignment_scores', {}) or {}),
            pending=len(getattr(self, '_pending_messages', []) or []),
            plan_remaining=plan_remaining,
            in_critical_section=bool(getattr(self, '_in_critical_section', False)),
            last_conscious_tick=int(getattr(self, '_last_conscious_tick', 0)),
            is_generated=bool(getattr(self, '_source_file', None)),
            recent_activity=tuple(self.activity_log[-20:]),
            recent_outputs=tuple(
                str(o.get('data', '') if isinstance(o, dict) else o)
                for o in (getattr(self, 'output_history', []) or [])[-5:]
            ),
        )

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


__all__ = ['PROJECT_ROOT', 'BaseSensorQueue', 'BaseActuator', 'ProcessingStream',
           'StreamView', 'SleepPhase']
