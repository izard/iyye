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
# streams/stream_factory.py
#!/usr/bin/env python3
"""
Stream Factory - Creates new processing streams based on high-alignment streams.
"""

import ast
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
import importlib.util
import logging
import os
import re
import sys

from iyye_base import PROJECT_ROOT, ProcessingStream

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")

# ------------------------------------------------------------------
# Safety validation for LLM-generated stream code
# ------------------------------------------------------------------

# Modules that could allow arbitrary OS access, network calls, or process control.
_BLOCKED_IMPORTS = frozenset({
    'subprocess', 'shutil', 'socket', 'http', 'urllib',
    'requests', 'ctypes', 'signal', 'multiprocessing',
})

# Built-in calls that execute arbitrary code or access internals.
_BLOCKED_CALLS = frozenset({
    'eval', 'exec', 'compile', '__import__', 'open', 'input',
})

# os.* methods that run commands or delete files.
_BLOCKED_OS_ATTRS = frozenset({
    'system', 'popen', 'exec', 'execvp', 'execve', 'execl',
    'remove', 'unlink', 'rmdir', 'rename',
})


# Methods that must not be overridden — the base class implementations
# handle cooperative stop, persistent logging, and timestamped history.
_FORBIDDEN_OVERRIDES = frozenset({
    'checkpoint', 'add_to_log', 'add_input', 'add_output',
})


def _validate_code_safety(code: str, sensor_key: Optional[str] = None) -> Optional[str]:
    """Return an error message if generated code is unsafe or useless, else None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Syntax error: {exc}"

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in _BLOCKED_IMPORTS:
                    return f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in _BLOCKED_IMPORTS:
                return f"Blocked import: {node.module}"

        # Check dangerous function calls
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BLOCKED_CALLS:
                return f"Blocked call: {func.id}()"
            if (isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'os'
                    and func.attr in _BLOCKED_OS_ATTRS):
                return f"Blocked call: os.{func.attr}()"

    # Check for forbidden method overrides and is_conscious assignment
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                # Reject overrides of base-class methods
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name in _FORBIDDEN_OVERRIDES:
                        return (f"Overrides base method '{item.name}' — "
                                f"call self.{item.name}() instead of redefining it")

    # Reject is_conscious assignment anywhere (self.is_conscious or bare is_conscious)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                attr_name = None
                if isinstance(target, ast.Attribute):
                    attr_name = target.attr
                elif isinstance(target, ast.Name):
                    attr_name = target.id
                if attr_name == 'is_conscious':
                    return "Sets is_conscious directly — consciousness is managed by the attention stream"

    # Reject use of restricted actuators — only web_chat_actuator is allowed.
    _RESTRICTED_ACTUATORS = {'TelegramActuator', 'tts_actuator'}
    for tok in _RESTRICTED_ACTUATORS:
        if tok in code:
            return f"References restricted actuator '{tok}' — generated streams may only use web_chat_actuator"

    # --- Usefulness checks ---

    # Sensor streams must actually read their sensor key.
    # Accept both direct (sensors_data.get) and subscript (context['sensors_data'].get) forms.
    if sensor_key:
        _sk_re = re.compile(
            r"""sensors_data[^.]*\.get\s*\(\s*['"]"""
            + re.escape(sensor_key)
            + r"""['"]"""
            r"""|sensors_data[^[]*\[\s*['"]"""
            + re.escape(sensor_key)
            + r"""['"]\s*\]"""
        )
        if not _sk_re.search(code):
            return (f"Sensor stream does not read sensors_data.get('{sensor_key}', []) "
                    f"— it would never process its input")

    # Sensor streams must call self.add_input() and self.add_output().
    if sensor_key:
        if 'add_input(' not in code:
            return "Sensor stream never calls self.add_input() — inputs won't be recorded"
        if 'add_output(' not in code:
            return "Sensor stream never calls self.add_output() — results won't be recorded"

    # Sensor and goal streams must NOT touch actuators — they process silently.
    if sensor_key:
        if 'actuator' in code.lower():
            return "Sensor stream references actuators — sensor handlers must process data silently"

    # Reject vague LTM search queries that produce useless results.
    _VAGUE_QUERY_RE = re.compile(
        r"""\.search\s*\(\s*['"]"""
        r"""(interesting|concept|knowledge|fact|general|information|data|stuff|things)"""
        r"""['"]\s*\)""",
        re.IGNORECASE,
    )
    m = _VAGUE_QUERY_RE.search(code)
    if m:
        return f"Uses vague memory search term '{m.group(1)}' — use specific, targeted queries"

    return None


# ------------------------------------------------------------------
# Goal coverage registry — prevents duplicate goal streams
# ------------------------------------------------------------------

def _plan_fingerprint(steps: List[Dict]) -> str:
    """Canonicalize plan steps and produce a short hash for dedup."""
    canonical = []
    for step in steps:
        desc = re.sub(r'\s+', ' ', step.get('description', '').lower().strip())
        step_type = step.get('type', 'unknown')
        canonical.append(f"{step_type}:{desc}")
    canonical.sort()
    return hashlib.sha256("|".join(canonical).encode()).hexdigest()[:16]


class _CoverageEntry:
    """Tracks one generated stream's lifecycle for dedup and evaluation."""

    __slots__ = (
        'key', 'stream_name', 'status', 'plan_hash',
        'created_tick', 'completed_tick', 'usefulness',
        'evidence_ids', 'vague_output_count', 'total_output_count',
    )

    def __init__(
        self,
        key: Tuple[str, str],
        stream_name: str,
        plan_hash: str,
        created_tick: int = 0,
        evidence_ids: Optional[List[str]] = None,
    ) -> None:
        self.key = key
        self.stream_name = stream_name
        self.status = "active"          # active | completed | failed
        self.plan_hash = plan_hash
        self.created_tick = created_tick
        self.completed_tick: Optional[int] = None
        self.usefulness = 0.5
        self.evidence_ids = evidence_ids or []
        self.vague_output_count = 0
        self.total_output_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': list(self.key),
            'stream_name': self.stream_name,
            'status': self.status,
            'plan_hash': self.plan_hash,
            'created_tick': self.created_tick,
            'completed_tick': self.completed_tick,
            'usefulness': self.usefulness,
            'evidence_ids': self.evidence_ids,
            'vague_output_count': self.vague_output_count,
            'total_output_count': self.total_output_count,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_CoverageEntry":
        e = cls(
            key=tuple(d['key']),
            stream_name=d['stream_name'],
            plan_hash=d.get('plan_hash', ''),
            created_tick=d.get('created_tick', 0),
            evidence_ids=d.get('evidence_ids', []),
        )
        e.status = d.get('status', 'completed')
        e.completed_tick = d.get('completed_tick')
        e.usefulness = d.get('usefulness', 0.5)
        e.vague_output_count = d.get('vague_output_count', 0)
        e.total_output_count = d.get('total_output_count', 0)
        return e


class GoalCoverageRegistry:
    """Tracks active and recently-retired generated streams per coverage key.

    A coverage key is a tuple like ``("goal", "agency")`` or
    ``("sensor", "hardware_sensor")``.  The registry prevents duplicate
    stream creation and provides a novelty gate that rejects plans whose
    fingerprint matches a recent failure.
    """

    _HISTORY_CAP = 50

    def __init__(self) -> None:
        self._active: Dict[Tuple[str, str], _CoverageEntry] = {}
        self._history: List[_CoverageEntry] = []

    # -- queries --

    def is_covered(self, key: Tuple[str, str]) -> bool:
        entry = self._active.get(key)
        return entry is not None and entry.status == "active"

    def find_active(self, key: Tuple[str, str]) -> Optional[_CoverageEntry]:
        entry = self._active.get(key)
        if entry is not None and entry.status == "active":
            return entry
        return None

    def was_recently_covered(
        self, key: Tuple[str, str], current_tick: int, lookback: int = 200,
    ) -> bool:
        """True if *key* is active OR was handled within *lookback* ticks."""
        if self.is_covered(key):
            return True
        cutoff = current_tick - lookback
        return any(
            e.key == key and (e.completed_tick or e.created_tick) >= cutoff
            for e in self._history
        )

    def has_recent_duplicate(
        self, key: Tuple[str, str], plan_hash: str, lookback: int = 200,
        current_tick: int = 0,
    ) -> bool:
        """True if a stream with the same key+hash completed or failed recently."""
        cutoff = current_tick - lookback
        for e in self._history:
            if e.key == key and e.plan_hash == plan_hash:
                if (e.completed_tick or e.created_tick) >= cutoff:
                    return True
        return False

    # -- mutations --

    def register(
        self, key: Tuple[str, str], stream_name: str,
        plan_hash: str, created_tick: int = 0,
        evidence_ids: Optional[List[str]] = None,
    ) -> _CoverageEntry:
        entry = _CoverageEntry(
            key=key, stream_name=stream_name, plan_hash=plan_hash,
            created_tick=created_tick, evidence_ids=evidence_ids,
        )
        self._active[key] = entry
        return entry

    def mark_completed(
        self, stream_name: str, usefulness: float, tick: int,
        vague_count: int = 0, total_count: int = 0,
    ) -> None:
        for key, entry in list(self._active.items()):
            if entry.stream_name == stream_name:
                entry.status = "failed" if usefulness < 0.3 else "completed"
                entry.usefulness = usefulness
                entry.completed_tick = tick
                entry.vague_output_count = vague_count
                entry.total_output_count = total_count
                self._history.append(entry)
                if len(self._history) > self._HISTORY_CAP:
                    self._history.pop(0)
                del self._active[key]
                return

    def remove_stale(self, live_stream_names: set) -> None:
        """Mark entries whose stream no longer exists as failed."""
        for key, entry in list(self._active.items()):
            if entry.status == "active" and entry.stream_name not in live_stream_names:
                entry.status = "failed"
                entry.usefulness = 0.0
                entry.completed_tick = entry.created_tick
                self._history.append(entry)
                if len(self._history) > self._HISTORY_CAP:
                    self._history.pop(0)
                del self._active[key]

    # -- serialization --

    def to_dict(self) -> Dict[str, Any]:
        return {
            'active': {"|".join(k): e.to_dict() for k, e in self._active.items()},
            'history': [e.to_dict() for e in self._history],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoalCoverageRegistry":
        reg = cls()
        for k_str, ed in data.get('active', {}).items():
            entry = _CoverageEntry.from_dict(ed)
            reg._active[entry.key] = entry
        for ed in data.get('history', []):
            reg._history.append(_CoverageEntry.from_dict(ed))
        return reg


class PlannedContinuationStream(ProcessingStream):
    """Stream created with a specific processing plan.

    Each step carries a ``type`` (learning, action, social, maintenance) and
    optionally an ``input`` payload.  The stream calls the LLM to actually
    process the step, writes discovered facts to STM, and records meaningful
    output — making it a genuine fallback when LLM codegen fails rather than
    a no-op placeholder.
    """

    # Instances are created on-demand by StreamFactory, not by the stream loader.
    _factory_created: bool = True

    # Prompt fragments keyed by step type.
    _TYPE_INSTRUCTIONS = {
        "learning": (
            "You are Iyye's curiosity subsystem. Analyse the input below and "
            "extract any new facts, questions worth pursuing, or knowledge gaps. "
            "Reply with a short summary of what you learned and 1-3 bullet-point "
            "facts worth remembering."
        ),
        "action": (
            "You are Iyye's agency subsystem. Given the input below, suggest a "
            "concrete action Iyye could take to affect the outside world. Reply "
            "with what you would do and why."
        ),
        "social": (
            "You are Iyye's social subsystem. Given the input below, draft a "
            "brief, friendly message or reflection about the social interaction. "
            "Reply with your observation and any suggested follow-up."
        ),
        "maintenance": (
            "You are Iyye's self-preservation subsystem. Given the system status "
            "below, identify any stability risks and recommend mitigations. "
            "Reply with a short assessment."
        ),
    }
    _DEFAULT_INSTRUCTION = (
        "You are Iyye, an AI with human-like traits. Process the input below "
        "and reply with a short, useful summary or observation."
    )

    # Patterns that indicate vague, ungrounded LLM output.
    _VAGUE_OUTPUT_RE = re.compile(
        r'\bno\s+(?:concrete|specific|actionable)\b'
        r'|\bcould\s+(?:potentially|theoretically)\b'
        r'|\bgeneral\s+(?:observation|assessment)\b'
        r'|\btake\s+action\s+on\s+pending\b'
        r'|\bengage\s+with\s+(?:user|social)\b'
        r'|\bno\s+(?:new|relevant)\s+(?:information|data)\b',
        re.IGNORECASE,
    )

    def __init__(self, source_name: str, plan: Dict[str, Any], stream_id: int):
        super().__init__(name=f"plan_{source_name}_{stream_id}")
        self.source_name = source_name
        self.plan = plan
        self.priority = plan.get('priority', 2)
        self._plan_steps = plan.get('steps', [])
        self._current_step = 0
        self._vague_output_count = 0
        self.brain = None  # Set by _create_planned_stream after instantiation

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _get_llm(self):
        router = getattr(self.brain, 'llm_router', None)
        if router is not None:
            return router.get_client(role="fast", no_think=True, max_tokens=512)
        from llm_client import LLMClient
        return LLMClient(no_think=True, max_tokens=512)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Any:
        if self._current_step >= len(self._plan_steps):
            usefulness = self._evaluate_usefulness()
            self.add_to_log(
                f"Plan completed for {self.source_name}, "
                f"usefulness={usefulness:.2f}, retiring stream"
            )
            brain = getattr(self, 'brain', None)
            if brain is not None:
                # Report back to StreamFactory's coverage registry.
                brain.post_message("stream_factory", {
                    "action": "stream_completed",
                    "stream_name": self.name,
                    "source_name": self.source_name,
                    "usefulness": usefulness,
                    "total_outputs": len(self.output_history),
                    "vague_outputs": self._vague_output_count,
                })
                try:
                    brain.streams.remove(self)
                except ValueError:
                    pass
            return {'action': 'plan_complete', 'source': self.source_name,
                    'usefulness': usefulness}

        step = self._plan_steps[self._current_step]
        desc = step.get('description', 'unknown')
        self.add_to_log(f"Step {self._current_step + 1}/{len(self._plan_steps)}: {desc}")

        # Cooperative checkpoint before the potentially blocking LLM call.
        self.checkpoint()

        result_text = self._execute_step(step, context)
        self._current_step += 1

        if result_text:
            if self._VAGUE_OUTPUT_RE.search(result_text):
                self._vague_output_count += 1
            self.add_output({'data': result_text, 'step': self._current_step},
                            target=self.source_name)
            self._store_facts(result_text)

        self.checkpoint()

        return {
            'source': self.source_name,
            'action': 'continuing',
            'step': self._current_step,
            'total_steps': len(self._plan_steps),
            'result': (result_text or '')[:200],
        }

    def _evaluate_usefulness(self) -> float:
        """Score 0.0-1.0: what fraction of outputs were concrete (not vague)."""
        total = len(self.output_history)
        if total == 0:
            return 0.0
        concrete = total - self._vague_output_count
        return min(1.0, max(0.0, concrete / total))

    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Run a single plan step through the LLM and return its response."""
        step_type = step.get('type', '')
        instruction = self._TYPE_INSTRUCTIONS.get(step_type, self._DEFAULT_INSTRUCTION)

        # Build user prompt from whatever context is available for this step.
        parts = [f"Goal: {self.plan.get('primary_goal', 'general')}"]
        parts.append(f"Step: {step.get('description', '')}")

        inp = step.get('input')
        if inp is not None:
            parts.append(f"Input:\n{str(inp)[:800]}")
        elif step_type == 'maintenance':
            # Feed system state for maintenance steps.
            sr = context.get('self_reflection_state') or getattr(
                self.brain, '_self_reflection_snapshot', None
            )
            if sr:
                parts.append(f"System state: {sr}")
        user_prompt = "\n\n".join(parts)

        try:
            llm = self._get_llm()
            return llm.complete(user_prompt, system_prompt=instruction)
        except Exception as exc:
            log.warning("PlannedContinuationStream LLM failed: %s", exc)
            return ""

    # Step types whose LLM output is an internal recommendation, not a
    # fact about the world.  These are stored as 'session' so they are
    # available during the current wake cycle but never promoted to LTM.
    _RECOMMENDATION_TYPES = frozenset({'action', 'social', 'maintenance'})

    def _store_facts(self, text: str) -> None:
        """Push discovered facts into STM.

        Only 'learning' steps produce genuine world-facts (time_frame='today').
        Action / social / maintenance recommendations are internal plans, not
        durable knowledge — stored as 'session' so they are never promoted.
        """
        stm = getattr(self.brain, 'stm', None)
        if stm is None:
            return
        step_type = ''
        if self._current_step > 0 and self._current_step <= len(self._plan_steps):
            step_type = self._plan_steps[self._current_step - 1].get('type', '')
        is_recommendation = step_type in self._RECOMMENDATION_TYPES
        try:
            stm.add_fact(
                text=text[:500],
                confidence=0.5,
                provenance=self.name,
                time_frame='session' if is_recommendation else 'today',
            )
        except Exception as exc:
            log.debug("PlannedContinuationStream: STM write failed: %s", exc)

class StreamFactory(ProcessingStream):
    """
    Creates new streams to pursue high-alignment opportunities.
    Never becomes conscious (HLD requirement).
    """

    MAX_STREAMS = 20

    def __init__(self, brain: "IyyeBrain"):
        super().__init__(name="stream_factory")
        self.brain = brain
        self.priority = 0
        self._can_be_conscious = False
        self._stream_counter = 0  # For unique stream IDs
        self._created_streams: List[str] = []
        self._tick_count = 0
        self._CODEGEN_INTERVAL = 50  # only attempt LLM codegen every N ticks
        # Queued suggestions from self-reflection (processed on codegen ticks).
        self._pending_suggestions: List[Dict[str, Any]] = []
        # Buffer for non-chat sensor payloads that no stream handles yet.
        # Filled every tick; drained when a handling stream is created.
        self._unhandled_sensor_buffer: Dict[str, List[Any]] = {}
        self._SENSOR_BUFFER_CAP = 20  # max payloads kept per sensor
        # Sensors for which a handling stream has already been created
        # *in this session*.  Populated by _create_suggested_stream() when a
        # stream is actually created and registered, and by restore_state()
        # which carries forward coverage from the previous session.
        #
        # Previously this was seeded by scanning llm_suggested_* filenames on
        # disk, but main_loop intentionally skips loading those files.  That
        # caused stale files to suppress sensor coverage with no running stream
        # to process the data.
        self._covered_sensors: set = set()
        # Goal coverage registry — prevents duplicate goal streams.
        self._goal_registry = GoalCoverageRegistry()

    def _is_sensor_covered(self, sensor_lower: str) -> bool:
        """Check if a sensor is already covered, handling truncated fragments
        from restart inference (e.g. 'hardware_s' covers 'hardware_sensor')."""
        if sensor_lower in self._covered_sensors:
            return True
        # Check if any stored fragment is a prefix of the sensor name
        # (handles truncation from the 20-char safe_name limit).
        return any(sensor_lower.startswith(frag) for frag in self._covered_sensors)

    # Never becomes conscious
    @property
    def is_conscious(self) -> bool:
        return False

    @is_conscious.setter
    def is_conscious(self, value: bool) -> None:
        pass
        
 
    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze sensor input queues and high-alignment streams, create new streams.

        HLD: "creates new processing streams by looking at latest inputs queues,
        and at existing streams with highest alignment scores and creating a plan
        how these streams can process further."
        """
        self._tick_count += 1
        streams = context.get('streams', self.brain.streams)

        # Drain mailbox — handle stream creation suggestions from
        # self-reflection and other streams.
        for msg in self.brain.drain_messages("stream_factory"):
            self._handle_message(msg)

        # Consume research tasks queued by UserChatStream and create WebResearchStreams.
        # Always processed regardless of tick count or stream cap.
        # Note: do NOT return early here — _create_streams_from_inputs must also run
        # so that chat messages arriving in the same tick are not dropped.
        research_result = self._create_research_streams()

        # Chat inputs are always processed regardless of stream count — sensor
        # messages were already drained from the queue by pop_all() this tick,
        # so if we skip them here they are permanently lost.
        input_result = self._create_streams_from_inputs(context)

        # Buffer non-chat sensor data that no stream handles yet.
        # Must run every tick *before* any early return so payloads popped by
        # run_once() are preserved even when chat/research was also handled.
        self._buffer_unhandled_sensors(context)

        if input_result or research_result:
            return input_result or research_result

        # Clean stale registry entries for streams that disappeared without
        # sending a completion message (e.g. removed by brain shutdown).
        live_names = {s.name for s in self.brain.streams}
        self._goal_registry.remove_stale(live_names)

        # Prune idle speculative streams (PlannedContinuationStream and llm_gen_*)
        # before checking the cap, so zombie streams don't block future creation.
        self._prune_idle_streams()

        # Only attempt speculative/exploratory codegen every _CODEGEN_INTERVAL ticks.
        if self._tick_count % self._CODEGEN_INTERVAL != 1:
            return None

        # Only apply MAX_STREAMS cap to speculative/exploratory stream creation.
        if len(streams) >= self.MAX_STREAMS:
            self.add_to_log(f"Max streams ({self.MAX_STREAMS}) reached, skipping speculative creation")
            return None

        # --- Process queued suggestions from self-reflection ---
        suggestion_result = self._process_pending_suggestions(context)
        if suggestion_result:
            return suggestion_result

        # --- HLD: "looks at latest inputs queues" — proactive sensor scan ---
        sensor_result = self._create_streams_from_sensor_gaps(context)
        if sensor_result:
            return sensor_result

        # --- HLD: look at existing streams with highest alignment scores ---
        scored = []
        for stream in streams:
            if self._is_self(stream):
                continue
            alignment = getattr(stream, 'alignment_scores', {})
            if alignment:
                total = sum(alignment.values())
                scored.append((stream, total))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        top_stream, top_score = scored[0]

        result = None

        if top_score > 2.5:
            if self._is_stream_stalled(top_stream):
                result = self._create_continuation_stream(top_stream)
            elif top_score > 3.0:
                result = self._create_exploratory_stream(top_stream)

        return result

    # Substrings that identify chat/text sensors (case-insensitive).
    # Hardware/git sensors are excluded by not matching any of these.
    _CHAT_KEYWORDS = ('chat', 'telegram', 'microphone', 'whisper', 'message')
    _SYSTEM_KEYWORDS = ('cpu', 'gpu', 'memory', 'disk', 'hardware', '_hw_', 'git')

    @classmethod
    def _is_chat_sensor(cls, name: str) -> bool:
        n = name.lower()
        if any(kw in n for kw in cls._SYSTEM_KEYWORDS):
            return False
        return any(kw in n for kw in cls._CHAT_KEYWORDS)

    def _refine_existing_stream(
        self, entry: _CoverageEntry, suggestion: Dict[str, Any],
    ) -> None:
        """Feed new evidence into an existing goal stream instead of creating N+1."""
        target = None
        for s in self.brain.streams:
            if s.name == entry.stream_name:
                target = s
                break
        if target is None:
            # Stream was pruned but registry not yet updated.
            self._goal_registry.mark_completed(
                entry.stream_name, usefulness=0.0, tick=self._tick_count,
            )
            return
        evidence_texts = suggestion.get("evidence_texts", [])
        if evidence_texts and isinstance(target, PlannedContinuationStream):
            step_type = target._plan_steps[-1].get('type', 'learning') \
                if target._plan_steps else 'learning'
            target._plan_steps.append({
                'step': len(target._plan_steps),
                'description': f"Process new evidence: {evidence_texts[0][:80]}",
                'type': step_type,
                'input': "\n".join(evidence_texts),
            })
            self.add_to_log(
                f"Refined '{entry.stream_name}' with new evidence "
                f"(now {len(target._plan_steps)} steps)"
            )
        entry.evidence_ids.extend(suggestion.get("evidence_ids", []))

    def _create_streams_from_inputs(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create or update UserChatStream instances for sensor queues that have new messages.

        HLD: "looks at latest inputs queues"

        Sensor names come from the dynamic loader and use class names
        (e.g. 'TelegramSensor', 'CameraSensor') or manual keys ('web_chat').
        Matching is done by substring rather than exact name.

        Multi-contact sensors (Telegram) are keyed per chat_id so that
        different users get separate conversation streams and histories.
        """
        from streams.user_chat_stream import UserChatStream

        sensors_data: Dict[str, List] = context.get('sensors_data', {})
        if not sensors_data:
            return None

        # Build lookup: stream name → existing UserChatStream
        existing_chat: Dict[str, UserChatStream] = {
            s.name: s
            for s in self.brain.streams
            if isinstance(s, UserChatStream)
        }
        created = []
        fed = []

        for sensor_name, messages in sensors_data.items():
            if not messages:
                continue
            if not self._is_chat_sensor(sensor_name):
                continue

            messages = self._expand_mcp_batches(messages)
            if not messages:
                continue

            # Group messages by contact so multi-user sensors (Telegram)
            # don't bleed conversations into each other.
            groups = self._group_by_contact(messages, sensor_name)

            for contact_key, contact_msgs in groups.items():
                stream_name = f"chat_{contact_key}"

                if stream_name in existing_chat:
                    existing_chat[stream_name]._pending_messages.extend(contact_msgs)
                    for msg in contact_msgs:
                        existing_chat[stream_name].add_input(
                            msg if not isinstance(msg, dict) else msg.get('data', msg),
                            source=sensor_name,
                        )
                    self.add_to_log(
                        f"Fed {len(contact_msgs)} message(s) to existing '{stream_name}'"
                    )
                    fed.append(stream_name)
                else:
                    new_stream = UserChatStream(
                        name=stream_name,
                        messages=list(contact_msgs),
                        brain=self.brain,
                        sensor_name=sensor_name,
                    )
                    self.brain.streams.append(new_stream)
                    self._created_streams.append(stream_name)
                    existing_chat[stream_name] = new_stream
                    self.add_to_log(
                        f"Created UserChatStream '{stream_name}' "
                        f"with {len(contact_msgs)} message(s)"
                    )
                    created.append(stream_name)

        if created or fed:
            return {'action': 'handled_chat_inputs', 'created': created, 'fed': fed}
        return None

    @staticmethod
    def _group_by_contact(messages: list, sensor_name: str) -> Dict[str, list]:
        """Group messages by contact key.

        Telegram messages carry a ``chat_id`` field — each distinct chat_id
        gets its own group (and therefore its own UserChatStream).  Messages
        without a chat_id (web_chat, microphone) are grouped under the
        sensor name as before.
        """
        groups: Dict[str, list] = {}
        sensor_lower = sensor_name.lower()
        for msg in messages:
            if isinstance(msg, dict) and msg.get('chat_id') is not None:
                key = f"{sensor_lower}_{msg['chat_id']}"
            else:
                key = sensor_lower
            groups.setdefault(key, []).append(msg)
        return groups
    
    def _create_research_streams(self) -> Optional[Dict[str, Any]]:
        """
        Drain brain._pending_research_tasks and create a WebResearchStream for each.
        Called every tick before chat input handling so research follows quickly
        after UserChatStream queues a task.
        """
        pending = getattr(self.brain, '_pending_research_tasks', None)
        if not pending:
            return None

        from streams.web_research_stream import WebResearchStream

        tasks, self.brain._pending_research_tasks = list(pending), []
        created = []
        for task in tasks:
            stream = WebResearchStream(task=task, brain=self.brain)
            self.brain.streams.append(stream)
            label = task.get('query') or task.get('url', '')[:50]
            self.add_to_log(
                f"Created WebResearchStream for {task.get('type')!r} task: {label}"
            )
            created.append(stream.name)

        return {'action': 'created_research_streams', 'count': len(created), 'streams': created}

    # ------------------------------------------------------------------
    # Mailbox: handle suggestions from self-reflection and other streams
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "suggest_stream":
            self._pending_suggestions.append(msg)
            log.debug("StreamFactory: queued suggestion: %s", msg.get("reason", "?"))
        elif action == "stream_completed":
            self._handle_stream_completion(msg)
        else:
            log.debug("StreamFactory: unknown message action %r", action)

    def _handle_stream_completion(self, msg: Dict[str, Any]) -> None:
        """Update coverage registry when a planned stream finishes."""
        stream_name = msg.get("stream_name", "")
        usefulness = msg.get("usefulness", 0.0)
        vague = msg.get("vague_outputs", 0)
        total = msg.get("total_outputs", 0)
        self._goal_registry.mark_completed(
            stream_name, usefulness, tick=self._tick_count,
            vague_count=vague, total_count=total,
        )
        if usefulness < 0.3:
            self.add_to_log(
                f"Registry: '{stream_name}' completed with low usefulness "
                f"({usefulness:.2f}) — marked as failed"
            )
        else:
            self.add_to_log(
                f"Registry: '{stream_name}' completed "
                f"(usefulness={usefulness:.2f})"
            )

    def _process_pending_suggestions(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process stream-creation suggestions queued from self-reflection.

        Called on codegen ticks (every _CODEGEN_INTERVAL) when under MAX_STREAMS.
        Deduplicates against existing streams and recently created ones.
        """
        if not self._pending_suggestions:
            return None

        streams = context.get('streams', self.brain.streams)

        result = None
        # Process one suggestion per codegen tick to avoid burst creation.
        while self._pending_suggestions:
            suggestion = self._pending_suggestions.pop(0)
            sensor = suggestion.get("sensor")
            goal = suggestion.get("goal", "curiosity")
            reason = suggestion.get("reason", "")

            # Dedup: skip if a stream for this sensor already exists
            # (either via successful codegen in _covered_sensors, or via a
            # planned fallback tracked in the coverage registry — active or
            # recently completed).
            if sensor:
                sensor_key = ("sensor", sensor.lower())
                if self._is_sensor_covered(sensor.lower()):
                    log.debug("StreamFactory: skipping suggestion — "
                              "stream for sensor '%s' already exists", sensor)
                    continue
                active_entry = self._goal_registry.find_active(sensor_key)
                if active_entry is not None:
                    log.debug(
                        "StreamFactory: sensor '%s' already covered by "
                        "planned fallback '%s'",
                        sensor, active_entry.stream_name,
                    )
                    self._refine_existing_stream(active_entry, suggestion)
                    continue
                if self._goal_registry.was_recently_covered(
                    sensor_key, self._tick_count,
                ):
                    log.debug("StreamFactory: sensor '%s' recently handled "
                              "— skipping", sensor)
                    continue

            # Dedup: skip if a goal handler already exists (active or recent).
            if not sensor and goal:
                goal_key = ("goal", goal)
                active_entry = self._goal_registry.find_active(goal_key)
                if active_entry is not None:
                    log.debug(
                        "StreamFactory: goal '%s' already covered by '%s' "
                        "— routing as refinement",
                        goal, active_entry.stream_name,
                    )
                    self._refine_existing_stream(active_entry, suggestion)
                    continue
                if self._goal_registry.was_recently_covered(
                    goal_key, self._tick_count,
                ):
                    log.debug("StreamFactory: goal '%s' recently handled "
                              "— skipping", goal)
                    continue

            if len(streams) >= self.MAX_STREAMS:
                log.debug("StreamFactory: skipping suggestion — at MAX_STREAMS")
                self._pending_suggestions.clear()
                break

            self.add_to_log(f"Processing suggestion: {reason}")
            result = self._create_suggested_stream(
                sensor, goal, context,
                evidence_ids=suggestion.get("evidence_ids"),
                evidence_texts=suggestion.get("evidence_texts"),
            )
            break  # one per tick

        return result

    def _create_suggested_stream(
        self,
        sensor: Optional[str],
        goal: str,
        context: Dict[str, Any],
        goal_question: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        evidence_texts: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a stream from a self-reflection suggestion.

        Sensor-driven suggestions use LLM codegen with the sensor-specific
        prompt.  Goal-driven suggestions (sensor=None) always use
        PlannedContinuationStream — LLM codegen for goal-gap streams
        produces mostly noise that STM/replay discards.

        A novelty gate rejects planned streams whose fingerprint matches
        a recent failure in the GoalCoverageRegistry.
        """
        source_name = f"suggested_{sensor.lower()}" if sensor else f"suggested_{goal}"

        # Gather recent inputs from the unhandled sensor buffer (accumulated
        # across ticks) rather than the single-tick sensors_data, which is
        # likely empty on codegen ticks for infrequent sensors.
        recent_inputs: List[Any] = []
        if sensor:
            buffered = self._unhandled_sensor_buffer.get(sensor, [])
            if not buffered:
                # Fallback: check current tick's sensors_data.
                for q_name, payloads in context.get('sensors_data', {}).items():
                    if sensor.lower() in q_name.lower() and payloads:
                        buffered = payloads
                        break
            if buffered:
                recent_inputs = [
                    item.get('data', item) if isinstance(item, dict) else item
                    for item in buffered[-5:]
                ]

        # Gather alignment context from the highest-scoring stream for this goal.
        alignment: Dict[str, float] = {}
        for s in self.brain.streams:
            scores = getattr(s, 'alignment_scores', {})
            if scores.get(goal, 0) > alignment.get(goal, 0):
                alignment = dict(scores)

        new_stream = None
        method = 'planned_fallback'

        if sensor:
            # Sensor-driven: try LLM codegen with the sensor-specific prompt.
            new_stream = self._generate_stream_code(
                source_name=source_name,
                primary_goal=goal,
                recent_inputs=recent_inputs,
                alignment=alignment or {goal: 0.0},
                sensor_key=sensor,
            )
            if new_stream is not None:
                method = 'llm_codegen'

        # Goal-driven OR sensor codegen failed: use PlannedContinuationStream.
        if new_stream is None:
            plan = self._create_processing_plan(
                source_name, goal, recent_inputs,
                evidence_texts=evidence_texts,
            )

            # Novelty gate: reject if same plan was recently useless/failed.
            cov_key = (
                ("sensor", sensor.lower()) if sensor
                else ("goal", goal)
            )
            plan_hash = _plan_fingerprint(plan.get('steps', []))
            if self._goal_registry.has_recent_duplicate(
                cov_key, plan_hash, current_tick=self._tick_count,
            ):
                self.add_to_log(
                    f"Novelty gate: plan for {cov_key} matches recent "
                    f"failure (hash {plan_hash[:8]}) — skipping"
                )
                return None

            new_stream = self._create_planned_stream(source_name, plan)
            method = 'planned_fallback'

            # Register in coverage registry so duplicates are blocked.
            self._goal_registry.register(
                key=cov_key,
                stream_name=new_stream.name,
                plan_hash=plan_hash,
                created_tick=self._tick_count,
                evidence_ids=evidence_ids,
            )

        self.brain.streams.append(new_stream)
        self._created_streams.append(new_stream.name)
        if sensor and method == 'llm_codegen':
            self._covered_sensors.add(sensor.lower())
            # Replay buffered payloads into the next tick's sensors_data so
            # the new stream's first execute() sees the data that triggered
            # its creation.  Uses the same _pending_factory_replay stash that
            # _awake_actions merges, mirroring _pending_interrupt_data.
            buffered = self._unhandled_sensor_buffer.get(sensor, [])
            if buffered:
                replay = getattr(self.brain, '_pending_factory_replay', None)
                if replay is None:
                    replay = {}
                    self.brain._pending_factory_replay = replay
                replay.setdefault(sensor, []).extend(buffered)
        self._drain_for_creation()
        self.add_to_log(
            f"Created suggested stream '{new_stream.name}' [{method}] "
            f"for goal '{goal}'"
            + (f" from sensor '{sensor}'" if sensor else "")
        )
        return {
            'action': 'created_suggested',
            'new_stream': new_stream.name,
            'goal': goal,
            'sensor': sensor,
            'method': method,
        }

    # ------------------------------------------------------------------
    # Non-chat sensor buffering & gap detection
    # ------------------------------------------------------------------

    def _buffer_unhandled_sensors(self, context: Dict[str, Any]) -> None:
        """Accumulate non-chat sensor payloads that no stream handles yet.

        Runs every tick so that data popped from queues by run_once() is
        preserved until a handling stream is created on a codegen tick.
        Once a handling stream exists for a sensor, its buffer is cleared
        automatically (the coverage check stops matching).
        """
        sensors_data: Dict[str, List] = context.get('sensors_data', {})

        for sensor_name, payloads in sensors_data.items():
            if not payloads:
                continue
            if self._is_chat_sensor(sensor_name):
                continue
            sensor_lower = sensor_name.lower()

            # Already covered by a created stream — don't buffer.
            if self._is_sensor_covered(sensor_lower):
                self._unhandled_sensor_buffer.pop(sensor_name, None)
                continue

            buf = self._unhandled_sensor_buffer.setdefault(sensor_name, [])
            buf.extend(payloads)
            # Cap to prevent unbounded growth.
            if len(buf) > self._SENSOR_BUFFER_CAP:
                self._unhandled_sensor_buffer[sensor_name] = buf[-self._SENSOR_BUFFER_CAP:]

    def _create_streams_from_sensor_gaps(
        self, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        HLD: "looks at latest inputs queues … and creates new processing streams."

        Creates one exploratory stream per codegen tick when the buffer
        contains unhandled non-chat sensor data.  Reads from the accumulated
        buffer (filled every tick) rather than the single-tick sensors_data,
        so payloads that arrived between codegen ticks are not lost.
        """
        for sensor_name, buffered in list(self._unhandled_sensor_buffer.items()):
            if not buffered:
                continue
            sensor_lower = sensor_name.lower()
            if self._is_sensor_covered(sensor_lower):
                continue
            # Also check the coverage registry — a planned fallback from a
            # previous codegen failure already handles this sensor, or one
            # recently completed (avoid re-creating the same handler).
            sensor_key = ("sensor", sensor_lower)
            if self._goal_registry.was_recently_covered(
                sensor_key, self._tick_count,
            ):
                continue

            self.add_to_log(
                f"Sensor gap: '{sensor_name}' has {len(buffered)} buffered "
                f"item(s), no handling stream — creating exploratory stream"
            )
            result = self._create_suggested_stream(
                sensor=sensor_name, goal="curiosity", context=context
            )
            # Clear buffer after creation so the stream starts fresh.
            self._unhandled_sensor_buffer.pop(sensor_name, None)
            return result

        return None

    @staticmethod
    def _expand_mcp_batches(raw_messages: list) -> list:
        """
        Expand MCP batch payloads into individual single-message items.

        FastMCP unwraps tool results from content[0]["text"], so after
        call_tool() the item is the direct tool dict:
            {"count": N, "messages": [{"text": "...", "chat_id": ..., ...}, ...]}
        Pushing the whole batch as one queue item means only the first message
        would ever be processed.  Split into one item per message.
        """
        expanded = []
        for item in raw_messages:
            if not isinstance(item, dict):
                expanded.append(item)
                continue
            # Direct tool-result format: top-level "messages" list
            msgs = item.get('messages')
            if isinstance(msgs, list):
                for msg in msgs:
                    if isinstance(msg, dict):
                        expanded.append(msg)
                continue
            # Legacy structuredContent wrapper (kept for backward compatibility)
            sc = item.get('structuredContent', {})
            if isinstance(sc, dict) and isinstance(sc.get('messages'), list):
                for msg in sc['messages']:
                    if isinstance(msg, dict):
                        expanded.append(msg)
                continue
            expanded.append(item)
        return expanded

    def _generate_stream_code(
        self,
        source_name: str,
        primary_goal: Optional[str],
        recent_inputs: List[Any],
        alignment: Dict[str, float],
        sensor_key: Optional[str] = None,
        goal_question: Optional[str] = None,
    ):
        """
        Ask the LLM to write a ProcessingStream subclass, save it to streams/
        for debugging, load it dynamically, and return an instance.  Returns
        None on any failure so callers can fall back to PlannedContinuationStream.

        Uses separate prompts for sensor-handler vs goal-exploration streams.
        """
        from iyye_base import ProcessingStream as _PS
        import json as _json

        self._stream_counter += 1
        stream_id = self._stream_counter
        safe_name = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")[:20]
        class_name = (
            "Llm"
            + "".join(p.capitalize() for p in safe_name.split("_") if p)
            + f"Stream{stream_id}"
        )
        file_rel = f"streams/llm_{safe_name}_{stream_id}.py"

        alignment_text = ", ".join(
            f"{g}={v:.2f}" for g, v in sorted(alignment.items(), key=lambda x: -x[1])
        ) or "(none)"

        # --- Build prompt variables based on stream type ---
        try:
            router = getattr(getattr(self, 'brain', None), 'llm_router', None)
            if router is not None:
                llm = router.get_client(role="codegen")
            else:
                from llm_client import LLMClient
                llm = LLMClient()

            if sensor_key:
                # --- Sensor-handler stream: rich grounding with full payload ---
                buffered = self._unhandled_sensor_buffer.get(sensor_key, [])
                sample = buffered[-1] if buffered else (
                    recent_inputs[-1] if recent_inputs else {}
                )
                try:
                    payload_sample = _json.dumps(sample, indent=2, default=str)[:1500]
                except (TypeError, ValueError):
                    payload_sample = str(sample)[:1500]

                raw = llm.complete_from_file(
                    "stream_codegen_sensor",
                    source_name=source_name,
                    primary_goal=primary_goal or "curiosity",
                    alignment_scores=alignment_text,
                    sensor_key=sensor_key,
                    payload_sample=payload_sample,
                    payload_count=str(len(buffered)),
                    class_name=class_name,
                    file_name=os.path.basename(file_rel),
                )
            else:
                # --- Goal-exploration stream: specific question + STM context ---
                question = goal_question or (
                    f"What concrete action could Iyye take to improve "
                    f"its {primary_goal or 'curiosity'} alignment score?"
                )
                stm = getattr(self.brain, 'stm', None)
                stm_facts = []
                if stm:
                    for f in stm.get_recent(10):
                        stm_facts.append(f"- [{f.get('time_frame','?')}] {f.get('text','')[:120]}")
                stm_context = "\n".join(stm_facts) or "(no recent facts)"

                raw = llm.complete_from_file(
                    "stream_codegen_goal",
                    primary_goal=primary_goal or "curiosity",
                    alignment_scores=alignment_text,
                    question=question,
                    stm_context=stm_context,
                    class_name=class_name,
                    file_name=os.path.basename(file_rel),
                )
        except Exception as exc:
            log.warning("StreamFactory LLM codegen call failed: %s", exc)
            return None

        # --- Extract code block ---
        match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if match:
            code = match.group(1).strip()
        elif "class " in raw:
            code = raw.strip()
        else:
            log.warning("StreamFactory codegen: no Python code block in LLM response")
            return None

        # --- Syntax check ---
        try:
            compile(code, os.path.basename(file_rel), "exec")
        except SyntaxError as exc:
            log.warning("StreamFactory codegen: syntax error in generated code: %s", exc)
            return None

        # --- Safety + usefulness check ---
        safety_err = _validate_code_safety(code, sensor_key=sensor_key)
        if safety_err is not None:
            log.warning("StreamFactory codegen: rejected code: %s", safety_err)
            self.add_to_log(f"Rejected generated code for {class_name}: {safety_err}")
            return None

        # --- Persist to disk for debugging (not committed to git) ---
        abs_path = str(PROJECT_ROOT / file_rel)
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            self.add_to_log(f"Saved generated stream {class_name} to {abs_path}")
        except Exception as exc:
            log.warning("StreamFactory codegen: write failed: %s", exc)
            return None

        # --- Dynamic load ---
        mod_name = f"llm_gen_{safe_name}_{stream_id}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, abs_path)
            if spec is None or spec.loader is None:
                log.warning("StreamFactory codegen: cannot create module spec for %s", abs_path)
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("StreamFactory codegen: module load failed: %s", exc)
            return None

        # --- Find and instantiate subclass ---
        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, _PS)
                and obj is not _PS
            ):
                try:
                    instance = obj()
                    # Ensure attributes required by AttentionStream / AlignmentStream
                    for attr, default in (
                        ("urgency", 0.0),
                        ("_last_conscious_tick", 0),
                        ("alignment_scores", {}),
                        ("_can_be_conscious", True),
                    ):
                        if not hasattr(instance, attr):
                            setattr(instance, attr, default)
                    instance.brain = self.brain
                    instance._source_file = abs_path
                    self.add_to_log(
                        f"Loaded LLM-generated class {obj.__name__} from {file_rel}"
                    )
                    return instance
                except Exception as exc:
                    log.warning(
                        "StreamFactory codegen: instantiation of %s failed: %s",
                        obj.__name__,
                        exc,
                    )

        log.warning("StreamFactory codegen: no ProcessingStream subclass found in generated code")
        return None

    # Names of streams that must never be pruned.
    _PROTECTED_NAMES = frozenset({
        'attention_stream', 'alignment_stream', 'stream_factory',
        'self_reflection', 'adenosine_stream', 'stm_update',
        'llm_management', 'theory_of_mind',
    })

    # Continuous (zero-input) LLM-generated streams are pruned once their
    # activity log exceeds this threshold.  Each execute() call typically adds
    # 1-3 log entries, so 50 entries ≈ 20-50 ticks of runtime — long enough
    # to do useful work, short enough to prevent infinite accumulation.
    _MAX_CONTINUOUS_LOG = 50

    def _prune_idle_streams(self) -> None:
        """
        Remove speculative streams that have finished all their work.

        A stream is considered idle when:
        - Its name is not in the protected set (not a special subconscious stream)
        - It is not the current conscious stream
        - It has no pending messages (_pending_messages is empty or absent)
        - Request/response streams (inputs > 0): all inputs have a matching output
        - Continuous streams (inputs == 0, LLM-generated): activity log exceeds
          _MAX_CONTINUOUS_LOG — they have had enough time to do useful work
        """
        current_conscious = getattr(self.brain, '_current_conscious', None)
        to_remove = []
        for stream in self.brain.streams:
            if stream is self:
                continue
            if stream.name in self._PROTECTED_NAMES:
                continue
            if stream is current_conscious:
                continue
            pending = len(getattr(stream, '_pending_messages', []))
            if pending > 0:
                continue
            inputs = len(getattr(stream, 'input_history', []))
            outputs = len(getattr(stream, 'output_history', []))
            # Request/response streams: done when outputs >= inputs
            if inputs > 0 and outputs >= inputs:
                to_remove.append(stream)
                continue
            # Continuous streams (zero inputs): prune after they have been
            # running long enough.  These are LLM-generated streams that
            # execute every tick; without this cap they survive indefinitely.
            if inputs == 0:
                log_len = len(getattr(stream, 'activity_log', []))
                if log_len > self._MAX_CONTINUOUS_LOG:
                    to_remove.append(stream)

        for stream in to_remove:
            try:
                self.brain.streams.remove(stream)
            except ValueError:
                continue
            # Notify the coverage registry that this stream was pruned.
            self._goal_registry.mark_completed(
                stream.name, usefulness=0.0, tick=self._tick_count,
            )
            # Delete the source file so it isn't reloaded on restart.
            src = getattr(stream, '_source_file', None)
            if src:
                try:
                    os.remove(src)
                    self.add_to_log(f"Pruned idle stream '{stream.name}' and deleted {src}")
                except OSError as exc:
                    log.warning("Failed to delete pruned stream file %s: %s", src, exc)
                    self.add_to_log(f"Pruned idle stream '{stream.name}' (file delete failed)")
            else:
                self.add_to_log(f"Pruned idle stream '{stream.name}'")

    def _drain_for_creation(self) -> None:
        """HLD: adenosine depletes on creating new processing streams."""
        adenosine = getattr(self.brain, 'adenosine', None)
        if adenosine is not None:
            adenosine.drain_activity("stream_create")

    def _is_self(self, stream) -> bool:
        """Check if stream is self."""
        return stream is self or (hasattr(stream, 'name') and stream.name == self.name)
    
    def _is_stream_stalled(self, stream) -> bool:
        """Check if a stream has pending work but isn't progressing."""
        inputs = len(getattr(stream, 'input_history', []))
        outputs = len(getattr(stream, 'output_history', []))
        return inputs > outputs + 2
    
    def _create_continuation_stream(self, source) -> Optional[Dict[str, Any]]:
        """
        Create a new stream that continues the work of source stream.
        Always uses PlannedContinuationStream — continuation streams have
        concrete pending inputs that are better handled step-by-step than
        by generating a free-running Python class.
        """
        source_alignment = getattr(source, 'alignment_scores', {})
        recent_inputs  = getattr(source, 'input_history',  [])[-5:]
        primary_goal   = (
            max(source_alignment.items(), key=lambda x: x[1])[0]
            if source_alignment else None
        )

        plan = self._create_processing_plan(
            source.name, primary_goal, recent_inputs,
        )
        new_stream = self._create_planned_stream(source.name, plan)
        method = 'planned_fallback'

        self.brain.streams.append(new_stream)
        self._created_streams.append(new_stream.name)
        self._drain_for_creation()
        self.add_to_log(
            f"Created continuation stream '{new_stream.name}' [{method}]"
            f" for goal '{primary_goal}'"
        )
        return {
            'action': 'created',
            'new_stream': new_stream.name,
            'source': source.name,
            'primary_goal': primary_goal,
            'method': method,
        }
    
    def _create_exploratory_stream(self, source) -> Optional[Dict[str, Any]]:
        """
        Create an exploratory stream based on high-alignment source.
        Always uses PlannedContinuationStream — exploratory streams work
        better with explicit steps than with free-running generated code.
        """
        source_alignment = getattr(source, 'alignment_scores', {})
        primary_goal = (
            max(source_alignment.items(), key=lambda x: x[1])[0]
            if source_alignment else 'curiosity'
        )
        recent_inputs  = getattr(source, 'input_history',  [])[-5:]

        plan = {
            'source': source.name,
            'primary_goal': primary_goal,
            'priority': 4,
            'steps': [
                {'step': 0, 'description': f'Explore {primary_goal} opportunities'},
                {'step': 1, 'description': 'Gather new information'},
                {'step': 2, 'description': 'Evaluate findings'},
            ],
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        new_stream = self._create_planned_stream(f"explore_{source.name}", plan)
        method = 'planned_fallback'

        self.brain.streams.append(new_stream)
        self._created_streams.append(new_stream.name)
        self._drain_for_creation()
        self.add_to_log(
            f"Created exploratory stream '{new_stream.name}' [{method}]"
            f" for goal '{primary_goal}'"
        )
        return {
            'action': 'created_exploratory',
            'new_stream': new_stream.name,
            'source': source.name,
            'primary_goal': primary_goal,
            'method': method,
        }
    
    def _create_planned_stream(self, source_name: str, plan: Dict[str, Any]) -> "ProcessingStream":
        """Create a PlannedContinuationStream instance with unique ID."""
        self._stream_counter += 1
        new_stream = PlannedContinuationStream(source_name, plan, self._stream_counter)
        new_stream.brain = self.brain  # needed for self-retirement on completion
        return new_stream
    
    _GOAL_STEP_TYPES: Dict[str, str] = {
        'curiosity': 'learning',
        'agency': 'action',
        'social': 'social',
        'self_preservation': 'maintenance',
    }

    def _create_processing_plan(
        self,
        source_name: str,
        primary_goal: Optional[str],
        pending_inputs: List[Any],
        evidence_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a processing plan for continuation.

        When *evidence_texts* are supplied, each piece of evidence becomes
        a concrete plan step with its text as input — replacing the generic
        "Take action on pending outputs" templates.  Plans without evidence
        AND without pending inputs get a single generic step as a last resort.
        """
        steps: List[Dict[str, Any]] = []

        for i, inp in enumerate(pending_inputs[:3]):
            steps.append({
                'step': i,
                'description': f"Process pending input: {str(inp)[:50]}",
                'input': inp,
            })

        step_type = self._GOAL_STEP_TYPES.get(primary_goal or '', 'learning')

        # Evidence-bound steps: each piece of evidence becomes a concrete step.
        if evidence_texts:
            for ev_text in evidence_texts[:3]:
                steps.append({
                    'step': len(steps),
                    'description': f"Analyse: {ev_text[:80]}",
                    'type': step_type,
                    'input': ev_text,
                })
        elif not pending_inputs:
            # Last-resort generic template — should be rare after evidence
            # gating is enforced in self_reflection.
            _GENERIC_DESC = {
                'curiosity': 'Investigate and learn from new information',
                'agency': 'Take action on pending outputs',
                'social': 'Engage with user or social context',
                'self_preservation': 'Ensure system stability and resource management',
            }
            steps.append({
                'step': len(steps),
                'description': _GENERIC_DESC.get(primary_goal or '', 'Process available data'),
                'type': step_type,
            })

        return {
            'source': source_name,
            'primary_goal': primary_goal,
            'priority': 3 if primary_goal in ['self_preservation', 'social'] else 2,
            'steps': steps,
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
 
    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['created_streams'] = self._created_streams
        state['covered_sensors'] = list(self._covered_sensors)
        state['goal_registry'] = self._goal_registry.to_dict()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self._created_streams = state.get('created_streams', [])
        self._covered_sensors = set(state.get('covered_sensors', []))
        registry_data = state.get('goal_registry')
        if registry_data:
            self._goal_registry = GoalCoverageRegistry.from_dict(registry_data)

