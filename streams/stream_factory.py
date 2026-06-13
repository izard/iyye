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
from capabilities import ScopedBrain
from llm_scheduler import LLMCall, LLMConsumerMixin

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

# Capability boundary (least privilege for generated code).  Attribute names a
# generated stream must never reach: the raw brain and the privileged surfaces
# it would expose, plus direct long-term-memory writes (memory is read-only for
# generated streams; STM writes go through the scoped context['stm']/cap).
_BLOCKED_CAP_ATTRS = frozenset({
    'brain',                         # raw brain → everything below + more
    'post_message', 'drain_messages',
    'llm_router', '_tom_stream',
    'set_contact_trusted',
    'store_fact', 'delete_fact',     # LTM writes
})
# context[...] keys whose raw handle is withheld from generated streams (no
# cross-stream access, no direct actuators — use context['cap'].emit instead).
_BLOCKED_CONTEXT_KEYS = frozenset({'actuators', 'streams'})


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

        # Capability boundary: deny raw brain / privileged attribute access.
        elif isinstance(node, ast.Attribute) and node.attr in _BLOCKED_CAP_ATTRS:
            return (f"Capability denied: '{node.attr}' is not accessible to a "
                    f"generated stream — use context['cap'] / context['stm']")

        # Deny pulling raw broad handles (all actuators / all streams) out of
        # the context dict.
        elif isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Index):          # py<3.9 compatibility
                sl = sl.value
            key = sl.value if isinstance(sl, ast.Constant) else None
            if key in _BLOCKED_CONTEXT_KEYS:
                return (f"Capability denied: context['{key}'] is withheld from "
                        f"generated streams (no cross-stream or direct-actuator "
                        f"access; use context['cap'].emit for output)")

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


# ---------------------------------------------------------------------------
# Generated-stream lifecycle thresholds (the "usefulness tests")
# ---------------------------------------------------------------------------
# A generated stream is a *candidate* until it proves itself over several
# awake cycles, at which point it *graduates* into a durable spec that is
# reloaded on restart and allowed to write promotable facts.  These constants
# define the graduation/demotion bar; see _CoverageEntry.decide_stage.
_EMA_ALPHA            = 0.5    # weight of the latest cycle in the usefulness EMA
_GRAD_MIN_CYCLES      = 2      # must survive at least this many awake cycles
_GRAD_MIN_EMA         = 0.6    # sustained usefulness bar
_GRAD_MIN_INPUTS      = 3      # OR handled this many real inputs (impact signal)
_GRAD_MAX_VAGUE_RATIO = 0.5    # reject if most outputs were vague/no-op
_CAND_MAX_CYCLES      = 3      # candidate that hasn't graduated by now is dropped
_CAND_FAIL_EMA        = 0.2    # candidate whose EMA falls this low is dropped early
_DEGRADE_EMA          = 0.3    # graduated stream below this is deprecated
_MAX_GRADUATED        = 12     # cap on durable graduated specs (deprecate lowest over cap)
# Phase 3 — refinement / rollback.
_REFINE_COOLDOWN_TICKS = 150   # min ticks between refinements of the same spec
_ROLLBACK_WINDOW       = 2     # cycles to observe a refined version before judging
_ROLLBACK_MARGIN       = 0.05  # new EMA must be within this of the old to be kept

# Durable code store for graduated generated streams.  Kept separate from the
# disposable ``streams/llm_*.py`` debug artifacts (which main_loop deletes on
# boot) so a proven stream survives restarts and is reloaded from here.
_GENERATED_DIR = PROJECT_ROOT / "streams" / "generated"


class _CoverageEntry:
    """Durable spec + lifecycle record for one generated stream.

    Two orthogonal axes are tracked:

    * ``status`` — *liveness* for dedup/coverage: active | completed | failed.
      Unchanged semantics; the coverage queries (is_covered, find_active,
      active_sensor_names) still key off ``status == "active"``.
    * ``stage`` — *maturity* on the candidate→graduated ladder:
      candidate | graduated | deprecated.  A graduated spec is durable: its
      code is kept on disk, reloaded on restart, and allowed to write
      promotable STM facts.

    Per-cycle usefulness signals (EMA, cycles observed, inputs handled, facts
    that survived replay to LTM, conscious promotions) accumulate across awake
    cycles so graduation reflects sustained value, not a single lucky tick.
    """

    __slots__ = (
        'key', 'stream_name', 'status', 'plan_hash',
        'created_tick', 'completed_tick', 'usefulness',
        'evidence_ids', 'vague_output_count', 'total_output_count',
        # --- durable spec / lifecycle fields ---
        'stage', 'kind', 'version', 'code_path', 'code_hash',
        'usefulness_ema', 'cycles_observed', 'inputs_handled',
        'facts_to_ltm', 'conscious_promotions',
        'prev_code_path', 'prev_ema',
    )

    def __init__(
        self,
        key: Tuple[str, str],
        stream_name: str,
        plan_hash: str,
        created_tick: int = 0,
        evidence_ids: Optional[List[str]] = None,
        kind: str = "plan",
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
        # Lifecycle / durable spec.
        self.stage = "candidate"        # candidate | graduated | deprecated
        self.kind = kind                # plan | codegen
        self.version = 1
        self.code_path: Optional[str] = None   # durable copy for codegen specs
        self.code_hash: Optional[str] = None
        self.usefulness_ema = 0.5
        self.cycles_observed = 0
        self.inputs_handled = 0
        self.facts_to_ltm = 0
        self.conscious_promotions = 0
        # Retained predecessor for refinement rollback (Phase 3).
        self.prev_code_path: Optional[str] = None
        self.prev_ema: Optional[float] = None

    # -- per-cycle evaluation --

    def record_cycle(
        self, cycle_usefulness: float, inputs_handled: int = 0,
        facts_to_ltm: int = 0, conscious_promotions: int = 0,
        vague_count: int = 0, total_count: int = 0,
    ) -> None:
        """Fold one awake cycle's signals into the spec's running stats."""
        self.inputs_handled += inputs_handled
        self.facts_to_ltm += facts_to_ltm
        self.conscious_promotions += conscious_promotions
        self.vague_output_count += vague_count
        self.total_output_count += total_count
        self.usefulness = cycle_usefulness
        # Seed the EMA from the first observed cycle rather than blending with
        # the arbitrary 0.5 prior — keeps early graduation/demotion decisions
        # honest instead of dragging them toward 0.5 for several cycles.
        if self.cycles_observed == 0:
            self.usefulness_ema = round(cycle_usefulness, 3)
        else:
            self.usefulness_ema = round(
                _EMA_ALPHA * cycle_usefulness
                + (1 - _EMA_ALPHA) * self.usefulness_ema,
                3,
            )
        self.cycles_observed += 1

    def _vague_ratio(self) -> float:
        if self.total_output_count <= 0:
            return 0.0
        return self.vague_output_count / self.total_output_count

    def decide_stage(self) -> str:
        """Return the stage this spec should be in given its accumulated stats.

        Pure function of the spec's counters — no side effects — so it is
        unit-testable.  The caller applies the returned stage and any
        on-disk/runtime consequences.
        """
        if self.stage == "deprecated":
            return "deprecated"

        graduated_now = self.stage == "graduated"

        # Demote a graduated spec whose sustained usefulness has decayed.
        if graduated_now:
            if self.usefulness_ema < _DEGRADE_EMA:
                return "deprecated"
            return "graduated"

        # Candidate evaluation.
        meets_impact = (
            self.inputs_handled >= _GRAD_MIN_INPUTS
            or self.facts_to_ltm >= 1
            or self.conscious_promotions >= 1
        )
        if (
            self.cycles_observed >= _GRAD_MIN_CYCLES
            and self.usefulness_ema >= _GRAD_MIN_EMA
            and meets_impact
            and self._vague_ratio() <= _GRAD_MAX_VAGUE_RATIO
        ):
            return "graduated"
        # Drop a candidate that is clearly unhelpful or has run out of chances.
        # Require ≥2 cycles before EMA-based failure so a stream that simply
        # had no input on its first cycle isn't killed prematurely; the hard
        # _CAND_MAX_CYCLES timeout still bounds total probation.
        if (
            (self.usefulness_ema < _CAND_FAIL_EMA and self.cycles_observed >= 2)
            or self.cycles_observed >= _CAND_MAX_CYCLES
        ):
            return "deprecated"
        return "candidate"

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
            'stage': self.stage,
            'kind': self.kind,
            'version': self.version,
            'code_path': self.code_path,
            'code_hash': self.code_hash,
            'usefulness_ema': self.usefulness_ema,
            'cycles_observed': self.cycles_observed,
            'inputs_handled': self.inputs_handled,
            'facts_to_ltm': self.facts_to_ltm,
            'conscious_promotions': self.conscious_promotions,
            'prev_code_path': self.prev_code_path,
            'prev_ema': self.prev_ema,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_CoverageEntry":
        e = cls(
            key=tuple(d['key']),
            stream_name=d['stream_name'],
            plan_hash=d.get('plan_hash', ''),
            created_tick=d.get('created_tick', 0),
            evidence_ids=d.get('evidence_ids', []),
            kind=d.get('kind', 'plan'),
        )
        e.status = d.get('status', 'completed')
        e.completed_tick = d.get('completed_tick')
        e.usefulness = d.get('usefulness', 0.5)
        e.vague_output_count = d.get('vague_output_count', 0)
        e.total_output_count = d.get('total_output_count', 0)
        e.stage = d.get('stage', 'candidate')
        e.version = d.get('version', 1)
        e.code_path = d.get('code_path')
        e.code_hash = d.get('code_hash')
        e.usefulness_ema = d.get('usefulness_ema', e.usefulness)
        e.cycles_observed = d.get('cycles_observed', 0)
        e.inputs_handled = d.get('inputs_handled', 0)
        e.facts_to_ltm = d.get('facts_to_ltm', 0)
        e.conscious_promotions = d.get('conscious_promotions', 0)
        e.prev_code_path = d.get('prev_code_path')
        e.prev_ema = d.get('prev_ema')
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
        kind: str = "plan",
    ) -> _CoverageEntry:
        entry = _CoverageEntry(
            key=key, stream_name=stream_name, plan_hash=plan_hash,
            created_tick=created_tick, evidence_ids=evidence_ids, kind=kind,
        )
        self._active[key] = entry
        return entry

    def add_entry(self, entry: _CoverageEntry) -> None:
        """Re-insert a fully-formed entry as the active handler for its key.

        Used when reloading a graduated spec on restart: the durable record
        comes back from history and becomes the live coverage entry again."""
        entry.status = "active"
        self._active[entry.key] = entry

    def find_by_name(self, stream_name: str) -> Optional[_CoverageEntry]:
        """Return the entry for *stream_name*, preferring the active one.

        Falls back to the most recent history entry so a spec's durable stats
        and stage survive after its instance was retired."""
        for entry in self._active.values():
            if entry.stream_name == stream_name:
                return entry
        for entry in reversed(self._history):
            if entry.stream_name == stream_name:
                return entry
        return None

    def all_entries(self) -> List[_CoverageEntry]:
        """All known entries (active first, then history), newest history last."""
        return list(self._active.values()) + list(self._history)

    def graduated_entries(self) -> List[_CoverageEntry]:
        """Durable graduated specs (deduped by stream_name, active preferred)."""
        seen: set = set()
        out: List[_CoverageEntry] = []
        for entry in self.all_entries():
            if entry.stage != "graduated" or entry.stream_name in seen:
                continue
            seen.add(entry.stream_name)
            out.append(entry)
        return out

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

    def active_sensor_names(self) -> set:
        """Return the set of sensor names that still have an active entry."""
        return {
            key[1] for key, e in self._active.items()
            if key[0] == "sensor" and e.status == "active"
        }

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


class PlannedContinuationStream(LLMConsumerMixin, ProcessingStream):
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
        r'|\bno\s+(?:new|relevant)\s+(?:information|data)\b'
        # Restated summaries / speculative prose without new information:
        r'|\b(?:appears?\s+to\s+be|seems?\s+to\s+be)\s+(?:running|operating|functioning)\s+(?:normally|well|fine)\b'
        r'|\ball\s+(?:systems?|metrics?|values?)\s+(?:are\s+)?(?:within|in)\s+(?:normal|acceptable|expected)\b'
        r'|\bnothing\s+(?:unusual|abnormal|alarming|noteworthy)\b'
        r'|\bcontinue\s+(?:monitoring|observing|tracking)\b'
        r'|\bworth\s+(?:monitoring|watching|keeping\s+an\s+eye)\b'
        r'|\bno\s+(?:immediate|urgent|critical)\s+(?:action|concern|issue|risk)\b',
        re.IGNORECASE,
    )

    # An output must reference at least one concrete data point (number,
    # percentage, identifier, path, etc.) from the input to count as
    # grounded rather than generic prose.
    _GROUNDED_RE = re.compile(
        r'\d+\.?\d*\s*%'          # percentage
        r'|\b\d{2,}\b'            # multi-digit number
        r'|/[\w./]+'              # file path
        r'|\b[A-Z][a-z]+[A-Z]\w*\b'  # CamelCase identifier
        r'|\b\w+_\w+\b'          # snake_case identifier
    )

    # Sensor-bound streams wait this many idle ticks (no new data) before
    # retiring.  Keeps the stream alive between bursts of sensor payloads
    # without holding it indefinitely when the sensor goes silent.
    _MAX_IDLE_TICKS = 100

    def __init__(self, source_name: str, plan: Dict[str, Any], stream_id: int):
        super().__init__(name=f"plan_{source_name}_{stream_id}")
        self.source_name = source_name
        self.plan = plan
        self.priority = plan.get('priority', 2)
        self._plan_steps = plan.get('steps', [])
        self._current_step = 0
        self._vague_output_count = 0
        self.brain = None  # Set by _create_planned_stream after instantiation
        # When set, this stream is sensor-bound and will recycle its plan
        # when new payloads arrive instead of retiring after one pass.
        self._sensor_key: Optional[str] = None
        self._idle_ticks = 0  # ticks since last new data (sensor-bound only)

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Any:
        # Apply a finished step (its LLM call ran on a scheduler worker).
        result = self._llm_poll()
        if result is not None:
            return self._on_step_result(result)
        # A step is still running — wait.
        if self._llm_busy():
            return None
        # Idle: either the plan is done, or submit the current step.
        if self._current_step >= len(self._plan_steps):
            return self._on_plan_exhausted()

        step = self._plan_steps[self._current_step]
        desc = step.get('description', 'unknown')
        self.add_to_log(f"Step {self._current_step + 1}/{len(self._plan_steps)}: {desc}")
        try:
            self.checkpoint()
        except StopIteration:
            return None

        user_prompt, instruction = self._build_step_prompt(step, context)
        submitted = self._llm_submit(
            role="fast", kind="planned",
            call=LLMCall.prompt(user_prompt, system_prompt=instruction),
            client_kwargs={"no_think": True, "max_tokens": 512},
        )
        if not submitted:
            return None  # scheduler paused/busy — retry next tick
        return {'source': self.source_name, 'action': 'step_submitted',
                'step': self._current_step}

    def _on_step_result(self, result) -> Any:
        """Apply a finished step's output, then advance to the next step."""
        if result.discarded:
            return None  # step lost (cycle rotated) — retry it next tick
        result_text = result.text if result.ok else ""
        self._current_step += 1
        if result_text:
            if (self._VAGUE_OUTPUT_RE.search(result_text)
                    or not self._GROUNDED_RE.search(result_text)):
                self._vague_output_count += 1
            self.add_output({'data': result_text, 'step': self._current_step},
                            target=self.source_name)
            self._store_facts(result_text)
        try:
            self.checkpoint()
        except StopIteration:
            pass
        return {
            'source': self.source_name,
            'action': 'continuing',
            'step': self._current_step,
            'total_steps': len(self._plan_steps),
            'result': (result_text or '')[:200],
        }

    def _on_plan_exhausted(self) -> Any:
        """All steps done: recycle (sensor-bound) or evaluate + retire."""
        # Sensor-bound streams recycle: check for new buffered payloads and
        # rebuild plan steps instead of retiring after one pass.
        if self._sensor_key and self._try_recycle():
            return {'action': 'recycled', 'source': self.source_name,
                    'idle_ticks': self._idle_ticks}

        usefulness = self._evaluate_usefulness()
        self.add_to_log(
            f"Plan completed for {self.source_name}, "
            f"usefulness={usefulness:.2f}, retiring stream"
        )
        brain = getattr(self, 'brain', None)
        if brain is not None:
            # Report back to StreamFactory's coverage registry.
            from messaging import Messages
            brain.post_message("stream_factory", Messages.stream_completed(
                stream_name=self.name,
                usefulness=usefulness,
                total_outputs=len(self.output_history),
                vague_outputs=self._vague_output_count,
            ))
            # Long-term-plan step executor: report completion to the planner
            # so it can advance the durable plan (see streams/planner_stream).
            plan_ref = getattr(self, '_plan_ref', None)
            if plan_ref:
                # add_output wraps step results: history entry 'data' is the
                # {'data': result_text, 'step': n} dict from _on_step_result.
                last = ''
                for o in reversed(self.output_history):
                    inner = o.get('data') if isinstance(o, dict) else None
                    if isinstance(inner, dict):
                        inner = inner.get('data', '')
                    if inner:
                        last = str(inner)
                        break
                brain.post_message("planner", Messages.plan_step_done(
                    plan_id=plan_ref.get('plan_id', ''),
                    step_index=int(plan_ref.get('step_index', -1)),
                    usefulness=usefulness,
                    summary=last[:300],
                ))
            try:
                self.request_retire("plan exhausted")
                brain.streams.remove(self)
            except ValueError:
                pass
        return {'action': 'plan_complete', 'source': self.source_name,
                'usefulness': usefulness}

    def _try_recycle(self) -> bool:
        """Check for new sensor payloads and rebuild plan steps.

        Returns True if the stream should stay alive (either new steps were
        created or the idle budget hasn't been exhausted), False when the
        stream should retire.
        """
        brain = getattr(self, 'brain', None)
        if brain is None:
            return False

        # Find the StreamFactory to access the unhandled sensor buffer.
        factory = None
        for s in brain.streams:
            if s.name == 'stream_factory':
                factory = s
                break
        if factory is None:
            return False

        buf = factory._unhandled_sensor_buffer.get(self._sensor_key, [])
        if buf:
            # New data arrived — rebuild plan steps from buffered payloads.
            self._idle_ticks = 0
            new_inputs = [
                item.get('data', item) if isinstance(item, dict) else item
                for item in buf[-5:]
            ]
            step_type = factory._GOAL_STEP_TYPES.get(
                self.plan.get('primary_goal', ''), 'learning',
            )
            self._plan_steps = [
                {
                    'step': i,
                    'description': f"Process sensor payload: {str(inp)[:50]}",
                    'type': step_type,
                    'input': inp,
                }
                for i, inp in enumerate(new_inputs)
            ]
            self._current_step = 0
            factory._unhandled_sensor_buffer.pop(self._sensor_key, None)
            self.add_to_log(
                f"Recycled: {len(self._plan_steps)} new steps from "
                f"{self._sensor_key} buffer"
            )
            return True

        # No new data — count idle ticks.
        self._idle_ticks += 1
        if self._idle_ticks < self._MAX_IDLE_TICKS:
            return True
        self.add_to_log(
            f"Sensor-bound stream idle for {self._idle_ticks} ticks, retiring"
        )
        return False

    def _evaluate_usefulness(self) -> float:
        """Score 0.0-1.0: what fraction of outputs were concrete (not vague)."""
        total = len(self.output_history)
        if total == 0:
            return 0.0
        concrete = total - self._vague_output_count
        return min(1.0, max(0.0, concrete / total))

    def _build_step_prompt(self, step: Dict[str, Any], context: Dict[str, Any]):
        """Build ``(user_prompt, system_instruction)`` for a plan step."""
        step_type = step.get('type', '')
        instruction = self._TYPE_INSTRUCTIONS.get(step_type, self._DEFAULT_INSTRUCTION)

        parts = [f"Goal: {self.plan.get('primary_goal', 'general')}"]
        parts.append(f"Step: {step.get('description', '')}")

        inp = step.get('input')
        if inp is not None:
            parts.append(f"Input:\n{str(inp)[:800]}")
        elif step_type == 'maintenance':
            # Feed system state for maintenance steps.
            sr = (context.get('self_reflection_state')
                  or self.brain.self_reflection_snapshot())
            if sr:
                parts.append(f"System state: {sr}")
        return "\n\n".join(parts), instruction

    def _store_facts(self, text: str) -> None:
        """Push discovered facts into STM.

        All planned-stream facts are stored as 'session' — LTM replay
        already rejects facts whose provenance matches planned/generated
        stream prefixes, so marking them 'today' just creates STM noise
        that is guaranteed to be discarded during promotion.
        """
        stm = getattr(self.brain, 'stm', None)
        if stm is None:
            return
        try:
            stm.add_fact(
                text=text[:500],
                confidence=0.5,
                provenance=self.name,
                time_frame='session',
            )
        except Exception as exc:
            log.debug("PlannedContinuationStream: STM write failed: %s", exc)

class StreamFactory(LLMConsumerMixin, ProcessingStream):
    """
    Creates new streams to pursue high-alignment opportunities.
    Never becomes conscious (HLD requirement).

    LLM codegen (create / convert / refine) goes through the async scheduler so
    a 30–60s code-generation call never blocks the main loop.  At most one
    codegen is in flight at a time (the scheduler's one-job-per-stream rule);
    while it runs, the factory starts no new stream creation, which keeps the
    coverage/dedup invariants exactly as in the synchronous version.
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
        # Goal coverage registry — prevents duplicate goal streams and now
        # also stores the durable lifecycle/spec record for each generated
        # stream (candidate → graduated → deprecated).
        self._goal_registry = GoalCoverageRegistry()
        # Graduated generated streams are reloaded from the durable store on
        # the first awake tick after a restart (guarded so it runs once).
        self._graduated_loaded = False
        # Per-spec tick of the last code refinement, to throttle re-refining.
        self._last_refine_tick: Dict[str, int] = {}
        # The single codegen op in flight on the scheduler (create | convert |
        # refine), or None.  Applied by _pump_codegen() at the top of execute().
        self._codegen_op: Optional[Dict[str, Any]] = None

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

        # Apply a finished codegen (its LLM call ran on a scheduler worker).
        self._pump_codegen()

        # Reload durable graduated streams once per process (after restore_state
        # has rebuilt the registry).  These survive restarts unlike candidate
        # streams whose files are deleted on boot.
        if not self._graduated_loaded:
            self._graduated_loaded = True
            self._reload_graduated_streams()

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

        # Reconcile _covered_sensors with the registry: drop sensors whose
        # handler stream was pruned or retired so new payloads are not silently
        # discarded (see bug: sensor coverage can outlive the handler).
        self._covered_sensors &= self._goal_registry.active_sensor_names()

        # Only attempt speculative/exploratory codegen every _CODEGEN_INTERVAL ticks.
        if self._tick_count % self._CODEGEN_INTERVAL != 1:
            return None

        # A codegen is already in flight — start no new stream creation this
        # tick.  This keeps the coverage/dedup invariants identical to the
        # synchronous version (nothing else mutates coverage while we wait).
        if self._llm_busy():
            return None

        # Upgrade any graduated *plan* handler into durable codegen (a swap,
        # not a new stream, so it runs even at the stream cap).
        self._convert_graduated_plans()

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
        # Codegen specs are refined by revising their *code* (evolve the
        # handler) rather than appending plan steps — this is the core
        # "modify the existing stream before creating another" behavior.
        if entry.kind == "codegen":
            entry.evidence_ids.extend(suggestion.get("evidence_ids", []))
            self._refine_stream_code(entry, target, suggestion)
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
        elif action == "create_for_plan":
            self._handle_create_for_plan(msg)
        elif action == "chat_follow_up":
            self._handle_chat_follow_up(msg)
        else:
            log.debug("StreamFactory: unknown message action %r", action)

    def _handle_chat_follow_up(self, msg: Dict[str, Any]) -> None:
        """Route a promise-repair follow-up into the contact's chat stream.

        HLD (STM section): an aged promise with no matching delivery triggers
        a follow-up input back into the conversation it came from.  The
        factory owns chat-stream creation, so it either feeds the live stream
        or recreates one (chat streams are pruned once idle).  The synthetic
        input goes through the stream's normal pipeline — trust is
        re-evaluated for the contact, tools are gated by their capability
        tier, replies route through the usual actuator.
        """
        stream_name = (msg.get("stream_name") or "").strip()
        text = (msg.get("text") or "").strip()
        contact = (msg.get("contact") or "").strip()
        if not stream_name.startswith("chat_") or not text:
            log.warning("StreamFactory: malformed chat_follow_up — ignored")
            return
        from streams.user_chat_stream import UserChatStream

        # chat_<sensor>_<chat_id> for telegram-style contacts; chat_<sensor>
        # for single-user channels (web chat).
        rest = stream_name[len("chat_"):]
        m = re.match(r"^(.*?)(?:_(\d+))?$", rest)
        sensor_name = m.group(1) if m else rest
        chat_id = int(m.group(2)) if (m and m.group(2)) else None
        if chat_id is not None:
            synthetic: Any = {"text": text, "chat_id": chat_id,
                              "user_id": chat_id, "_follow_up": True}
            if contact:
                synthetic["first_name"] = contact
        else:
            synthetic = text  # plain-string input (web chat format)

        existing = next(
            (s for s in self.brain.streams
             if s.name == stream_name and isinstance(s, UserChatStream)),
            None,
        )
        if existing is not None:
            existing._pending_messages.append(synthetic)
            existing.add_input(text, source="promise_follow_up")
            self.add_to_log(
                f"Fed promise follow-up to existing '{stream_name}'")
            return
        new_stream = UserChatStream(
            name=stream_name, messages=[synthetic],
            brain=self.brain, sensor_name=sensor_name,
        )
        self.brain.streams.append(new_stream)
        self._created_streams.append(stream_name)
        self._drain_for_creation()
        self.add_to_log(
            f"Created '{stream_name}' for promise follow-up"
            + (f" to {contact}" if contact else ""))

    def _handle_create_for_plan(self, msg: Dict[str, Any]) -> None:
        """Install an executor for one long-term-plan step (from the planner).

        The planner already decomposed the step into the concrete plan-dict
        shape PlannedContinuationStream executes, so this skips codegen and
        the novelty gate — long term plan steps are owner/planner-curated,
        not speculative coverage work."""
        plan = msg.get("plan") or {}
        plan_ref = msg.get("plan_ref") or {}
        if not plan.get("steps") or not plan_ref.get("plan_id"):
            log.warning("StreamFactory: malformed create_for_plan — ignored")
            return
        new_stream = self._create_planned_stream(
            plan.get("source", "planner"), plan)
        new_stream._plan_ref = plan_ref
        # Deadline pressure rides on the executor (see planner _dispatch_step):
        # attention weighs StreamView.urgency, so a due plan step competes for
        # consciousness through the stream doing the work.
        try:
            new_stream.urgency = float(plan.get("urgency", 0.0))
        except (TypeError, ValueError):
            pass
        self.brain.streams.append(new_stream)
        self._created_streams.append(new_stream.name)
        self._drain_for_creation()
        self.add_to_log(
            f"Created plan-step executor '{new_stream.name}' for "
            f"'{plan_ref.get('plan_id')}' step {plan_ref.get('step_index')}"
        )

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

        cov_key = (
            ("sensor", sensor.lower()) if sensor
            else ("goal", goal)
        )

        if sensor:
            # Sensor-driven: submit LLM codegen (async).  Capture the buffered
            # payloads now so the replay at apply-time is independent of buffer
            # timing.  On a successful submit, finalization is deferred to
            # _apply_codegen_result; on failure we fall through to planned.
            submitted = self._submit_codegen(
                op="create",
                source_name=source_name,
                primary_goal=goal,
                recent_inputs=recent_inputs,
                alignment=alignment or {goal: 0.0},
                sensor_key=sensor,
                apply_ctx={
                    "sensor": sensor, "goal": goal, "source_name": source_name,
                    "recent_inputs": recent_inputs, "cov_key": cov_key,
                    "evidence_ids": evidence_ids, "evidence_texts": evidence_texts,
                    "buffered_for_replay": list(
                        self._unhandled_sensor_buffer.get(sensor, [])
                    ),
                },
            )
            if submitted:
                self.add_to_log(
                    f"Submitted codegen for sensor '{sensor}' (goal '{goal}')"
                )
                return {'action': 'codegen_pending', 'sensor': sensor, 'goal': goal}
            # Scheduler unavailable — fall through to the planned fallback.

        # Goal-driven OR sensor codegen couldn't be submitted: planned stream.
        return self._create_planned_and_finalize(
            source_name, goal, recent_inputs, evidence_texts, evidence_ids,
            cov_key, sensor,
        )

    def _finalize_codegen_stream(
        self, new_stream, cov_key, sensor, goal, evidence_ids, buffered_for_replay,
    ) -> Optional[Dict[str, Any]]:
        """Register and install a freshly-generated codegen stream."""
        self._goal_registry.register(
            key=cov_key, stream_name=new_stream.name, plan_hash=new_stream.name,
            created_tick=self._tick_count, evidence_ids=evidence_ids, kind='codegen',
        )
        self.brain.streams.append(new_stream)
        self._created_streams.append(new_stream.name)
        if sensor:
            self._covered_sensors.add(sensor.lower())
            # Replay the payloads that triggered creation into the next tick's
            # sensors_data so the new stream's first execute() sees them.
            buffered = buffered_for_replay or self._unhandled_sensor_buffer.get(sensor, [])
            if buffered:
                replay = getattr(self.brain, '_pending_factory_replay', None)
                if replay is None:
                    replay = {}
                    self.brain._pending_factory_replay = replay
                replay.setdefault(sensor, []).extend(buffered)
        self._drain_for_creation()
        self.add_to_log(
            f"Created suggested stream '{new_stream.name}' [llm_codegen] "
            f"for goal '{goal}'" + (f" from sensor '{sensor}'" if sensor else "")
        )
        return {
            'action': 'created_suggested', 'new_stream': new_stream.name,
            'goal': goal, 'sensor': sensor, 'method': 'llm_codegen',
        }

    def _create_planned_and_finalize(
        self, source_name, goal, recent_inputs, evidence_texts, evidence_ids,
        cov_key, sensor,
    ) -> Optional[Dict[str, Any]]:
        """Build, gate, register and install a PlannedContinuationStream."""
        plan = self._create_processing_plan(
            source_name, goal, recent_inputs, evidence_texts=evidence_texts,
        )
        # Novelty gate: reject if the same plan was recently useless/failed.
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
        # Sensor-bound: mark continuous so it recycles on new payloads.
        if sensor:
            new_stream._sensor_key = sensor

        self._goal_registry.register(
            key=cov_key, stream_name=new_stream.name, plan_hash=plan_hash,
            created_tick=self._tick_count, evidence_ids=evidence_ids, kind='plan',
        )
        self.brain.streams.append(new_stream)
        self._created_streams.append(new_stream.name)
        self._drain_for_creation()
        self.add_to_log(
            f"Created suggested stream '{new_stream.name}' [planned_fallback] "
            f"for goal '{goal}'" + (f" from sensor '{sensor}'" if sensor else "")
        )
        return {
            'action': 'created_suggested', 'new_stream': new_stream.name,
            'goal': goal, 'sensor': sensor, 'method': 'planned_fallback',
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
            # Check if an active handler is registered (defence-in-depth
            # alongside _is_sensor_covered).  History entries are
            # intentionally NOT checked here — if buffered data exists and
            # no live handler is processing it, a new stream should be
            # created regardless of how recently the previous one retired.
            sensor_key = ("sensor", sensor_lower)
            if self._goal_registry.is_covered(sensor_key):
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

    def _submit_codegen(
        self,
        op: str,
        source_name: str,
        primary_goal: Optional[str],
        recent_inputs: List[Any],
        alignment: Dict[str, float],
        sensor_key: Optional[str] = None,
        goal_question: Optional[str] = None,
        apply_ctx: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Build the codegen prompt and submit it to the async scheduler.

        On success stashes ``self._codegen_op`` (parsed by _apply_codegen_result
        on a later tick).  Returns True if submitted, False otherwise.  Uses
        separate prompts for sensor-handler vs goal-exploration streams.
        """
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
        mod_name = f"llm_gen_{safe_name}_{stream_id}"

        alignment_text = ", ".join(
            f"{g}={v:.2f}" for g, v in sorted(alignment.items(), key=lambda x: -x[1])
        ) or "(none)"

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
            prompt_name = "stream_codegen_sensor"
            variables = dict(
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
            prompt_name = "stream_codegen_goal"
            variables = dict(
                primary_goal=primary_goal or "curiosity",
                alignment_scores=alignment_text,
                question=question,
                stm_context=stm_context,
                class_name=class_name,
                file_name=os.path.basename(file_rel),
            )

        submitted = self._llm_submit(
            role="codegen", kind="codegen",
            call=LLMCall.from_file(prompt_name, **variables),
        )
        if not submitted:
            return False
        self._codegen_op = {
            "op": op, "class_name": class_name, "file_rel": file_rel,
            "mod_name": mod_name, "sensor_key": sensor_key,
            **(apply_ctx or {}),
        }
        return True

    def _parse_codegen(
        self, raw: str, class_name: str, file_rel: str, mod_name: str,
        sensor_key: Optional[str],
    ):
        """Parse a codegen LLM response into a loaded stream instance, or None.

        Extracts the code block, syntax- and safety-checks it, writes it to the
        debug file, loads the module, and instantiates the subclass.
        """
        from iyye_base import ProcessingStream as _PS

        match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if match:
            code = match.group(1).strip()
        elif "class " in raw:
            code = raw.strip()
        else:
            log.warning("StreamFactory codegen: no Python code block in LLM response")
            return None

        try:
            compile(code, os.path.basename(file_rel), "exec")
        except SyntaxError as exc:
            log.warning("StreamFactory codegen: syntax error in generated code: %s", exc)
            return None

        safety_err = _validate_code_safety(code, sensor_key=sensor_key)
        if safety_err is not None:
            log.warning("StreamFactory codegen: rejected code: %s", safety_err)
            self.add_to_log(f"Rejected generated code for {class_name}: {safety_err}")
            return None

        abs_path = str(PROJECT_ROOT / file_rel)
        try:
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as fh:
                fh.write(code)
            self.add_to_log(f"Saved generated stream {class_name} to {abs_path}")
        except Exception as exc:
            log.warning("StreamFactory codegen: write failed: %s", exc)
            return None

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

        for obj in vars(module).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, _PS)
                and obj is not _PS
            ):
                try:
                    instance = obj()
                    for attr, default in (
                        ("urgency", 0.0),
                        ("_last_conscious_tick", 0),
                        ("alignment_scores", {}),
                        ("_can_be_conscious", True),
                    ):
                        if not hasattr(instance, attr):
                            setattr(instance, attr, default)
                    # Least privilege: generated code gets a scoped brain façade.
                    instance.brain = ScopedBrain(self.brain)
                    instance._source_file = abs_path
                    instance._cap_sensors = {sensor_key} if sensor_key else set()
                    self.add_to_log(
                        f"Loaded LLM-generated class {obj.__name__} from {file_rel}"
                    )
                    return instance
                except Exception as exc:
                    log.warning(
                        "StreamFactory codegen: instantiation of %s failed: %s",
                        obj.__name__, exc,
                    )

        log.warning("StreamFactory codegen: no ProcessingStream subclass found in generated code")
        return None

    # ------------------------------------------------------------------
    # Async codegen result application
    # ------------------------------------------------------------------

    def _pump_codegen(self) -> None:
        """Apply a finished codegen result, if one is ready."""
        if self._codegen_op is None:
            return
        result = self._llm_poll()
        if result is not None:
            self._apply_codegen_result(result)

    def _apply_codegen_result(self, result) -> None:
        """Dispatch a finished codegen to its op-specific handler."""
        op = self._codegen_op or {}
        self._codegen_op = None
        kind = op.get("op")
        instance = None
        if not result.discarded and result.ok and result.text:
            instance = self._parse_codegen(
                result.text, op.get("class_name", ""), op.get("file_rel", ""),
                op.get("mod_name", ""), op.get("sensor_key"),
            )
        if kind == "create":
            if instance is not None:
                self._finalize_codegen_stream(
                    instance, op["cov_key"], op.get("sensor"),
                    op.get("goal"), op.get("evidence_ids"),
                    op.get("buffered_for_replay"),
                )
            else:
                # Codegen failed/discarded → planned fallback.
                self._create_planned_and_finalize(
                    op["source_name"], op.get("goal"), op.get("recent_inputs", []),
                    op.get("evidence_texts"), op.get("evidence_ids"),
                    op["cov_key"], op.get("sensor"),
                )
        elif kind == "convert":
            self._apply_convert_result(instance, op)
        elif kind == "refine":
            self._apply_refine_result(result, op)

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

        Safety — "is it safe to remove without losing work?" — is the brain's
        single liveness contract (``brain.is_reapable``): it rejects the
        conscious stream, protected infrastructure streams, and any stream
        that is AWAITING (in-flight LLM job or open multi-stage turn) or
        ACTIVE (pending messages / unanswered inputs).  This method only adds
        the factory's *policy* on which reapable streams to actually retire:
        graduated and sensor-bound streams are kept (they manage their own
        lifecycle); continuous LLM-generated streams are retired only after
        enough runtime (_MAX_CONTINUOUS_LOG).
        """
        to_remove = []
        for stream in self.brain.streams:
            if stream is self:
                continue
            # Universal safety — one predicate, shared by every reaping actor.
            if not self.brain.is_reapable(stream):
                continue
            # Graduated streams are durable, continuous handlers — never prune
            # them here (their code lives in the persistent store).
            entry = self._goal_registry.find_by_name(stream.name)
            if entry is not None and entry.stage == "graduated":
                continue
            # Sensor-bound planned streams manage their own lifecycle via
            # _try_recycle / _MAX_IDLE_TICKS — don't prune them here.
            if getattr(stream, '_sensor_key', None):
                continue
            inputs = len(getattr(stream, 'input_history', []))
            # Request/response streams: reapable already implies all inputs are
            # answered, so retire them.
            if inputs > 0:
                to_remove.append(stream)
                continue
            # Continuous streams (zero inputs): retire only after enough
            # runtime, else they would churn every tick.
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
 
    # ------------------------------------------------------------------
    # Generated-stream lifecycle: candidate → graduated → deprecated
    # ------------------------------------------------------------------

    def on_pause(self) -> None:
        """End-of-cycle hook (run during the brain's wind-down settle).

        Folds this awake cycle's signals into each tracked generated stream's
        durable spec and applies graduation / demotion.  Running it here means
        graduation requires surviving multiple sleep cycles, not one lucky tick.
        """
        try:
            self._evaluate_generated_streams()
        except Exception as exc:
            log.warning("StreamFactory: generated-stream evaluation failed: %s", exc)

    @staticmethod
    def _code_hash(code: str) -> str:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()

    def _tracked_instances(self) -> List[Tuple[Any, "_CoverageEntry"]]:
        """Live generated stream instances paired with their registry entry."""
        pairs: List[Tuple[Any, _CoverageEntry]] = []
        for s in list(self.brain.streams):
            if self._is_self(s) or s.name in self._PROTECTED_NAMES:
                continue
            entry = self._goal_registry.find_by_name(s.name)
            if entry is not None:
                pairs.append((s, entry))
        return pairs

    def _cycle_usefulness(self, instance) -> float:
        """Best-effort 0-1 usefulness for one stream over the cycle."""
        evaluator = getattr(instance, "_evaluate_usefulness", None)
        if callable(evaluator):
            try:
                return float(max(0.0, min(1.0, evaluator())))
            except Exception:
                pass
        align = getattr(instance, "alignment_scores", {}) or {}
        score = (sum(align.values()) / len(align)) if align else 0.0
        if getattr(instance, "output_history", []):
            score += 0.2
        if instance.name in getattr(self.brain, "_was_conscious_streams", set()):
            score += 0.15
        return float(max(0.0, min(1.0, score)))

    def _collect_cycle_signals(self, instance, entry) -> Dict[str, int]:
        """Per-cycle deltas for *instance* via counters stashed on it."""
        cur_in = len(getattr(instance, "input_history", []))
        cur_out = len(getattr(instance, "output_history", []))
        last_in = getattr(instance, "_spec_seen_inputs", 0)
        last_out = getattr(instance, "_spec_seen_outputs", 0)
        instance._spec_seen_inputs = cur_in
        instance._spec_seen_outputs = cur_out
        vague = 0
        for o in getattr(instance, "output_history", [])[last_out:]:
            data = o.get("data", "") if isinstance(o, dict) else o
            if len(str(data).strip()) < 15:
                vague += 1
        credit = getattr(self.brain, "_graduated_fact_credit", {}) or {}
        return {
            "inputs": max(0, cur_in - last_in),
            "outputs": max(0, cur_out - last_out),
            "vague": vague,
            "facts": int(credit.get(instance.name, 0)),
            "conscious": 1 if instance.name in getattr(
                self.brain, "_was_conscious_streams", set()) else 0,
        }

    def _evaluate_generated_streams(self) -> None:
        changed = False
        for instance, entry in self._tracked_instances():
            # Judge a recent refinement before folding in more signal: if the
            # new version regressed against its predecessor, swap the old code
            # back in and re-probate it next cycle.
            if self._maybe_rollback(instance, entry):
                changed = True
                continue
            sig = self._collect_cycle_signals(instance, entry)
            entry.record_cycle(
                cycle_usefulness=self._cycle_usefulness(instance),
                inputs_handled=sig["inputs"],
                facts_to_ltm=sig["facts"],
                conscious_promotions=sig["conscious"],
                vague_count=sig["vague"],
                total_count=sig["outputs"],
            )
            new_stage = entry.decide_stage()
            if new_stage == entry.stage:
                continue
            if new_stage == "graduated":
                changed = self._graduate(instance, entry) or changed
            else:  # deprecated
                entry.stage = "deprecated"
                self.add_to_log(
                    f"Lifecycle: '{entry.stream_name}' deprecated "
                    f"(ema={entry.usefulness_ema:.2f}, cycles={entry.cycles_observed})"
                )
                changed = True

        # Enforce the graduated cap: deprecate the lowest-EMA specs over limit.
        grads = self._goal_registry.graduated_entries()
        if len(grads) > _MAX_GRADUATED:
            grads.sort(key=lambda e: e.usefulness_ema)
            for e in grads[:len(grads) - _MAX_GRADUATED]:
                e.stage = "deprecated"
                self.add_to_log(f"Lifecycle: '{e.stream_name}' deprecated (over cap)")
            changed = True

        # Per-cycle LTM credit has been folded into the specs; reset it.
        if hasattr(self.brain, "_graduated_fact_credit"):
            self.brain._graduated_fact_credit = {}
        if changed:
            self._refresh_graduated_names()

    def _graduate(self, instance, entry: "_CoverageEntry") -> bool:
        """Promote a candidate to a durable graduated spec, persisting code."""
        if entry.kind != "codegen":
            # Plans carry no reloadable code; mark graduated for status/learning
            # but they won't survive restart until Phase 3 converts them.
            entry.stage = "graduated"
            self.add_to_log(
                f"Lifecycle: plan stream '{entry.stream_name}' graduated "
                f"(no durable code; ema={entry.usefulness_ema:.2f})"
            )
            return True
        src = getattr(instance, "_source_file", None)
        if not src or not os.path.exists(src):
            log.warning("Cannot graduate %s: source file missing", entry.stream_name)
            return False
        try:
            with open(src, "r", encoding="utf-8") as fh:
                code = fh.read()
        except OSError as exc:
            log.warning("Cannot read source for %s: %s", entry.stream_name, exc)
            return False
        if not self._persist_graduated_code(entry, code):
            return False
        entry.stage = "graduated"
        try:
            instance._source_file = entry.code_path
        except Exception:
            pass
        self.add_to_log(
            f"Lifecycle: '{entry.stream_name}' GRADUATED v{entry.version} "
            f"(ema={entry.usefulness_ema:.2f}, cycles={entry.cycles_observed}, "
            f"inputs={entry.inputs_handled}, facts_ltm={entry.facts_to_ltm})"
        )
        return True

    def _persist_graduated_code(self, entry: "_CoverageEntry", code: str) -> bool:
        """Write *code* to the durable store; record its path + hash.

        Re-validates safety first — never durably persist unsafe code."""
        safety_err = _validate_code_safety(code)
        if safety_err is not None:
            log.warning("Refusing to graduate %s: %s", entry.stream_name, safety_err)
            return False
        try:
            _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^a-z0-9_]+", "_", entry.stream_name.lower()).strip("_")
            dest = _GENERATED_DIR / f"{safe}.py"
            dest.write_text(code, encoding="utf-8")
            entry.code_path = str(dest)
            entry.code_hash = self._code_hash(code)
            return True
        except OSError as exc:
            log.warning("Failed to persist graduated code for %s: %s",
                        entry.stream_name, exc)
            return False

    def _refresh_graduated_names(self) -> None:
        """Publish graduated stream names on the brain so STM-wrapper selection
        (Phase 2) can grant them durable-fact permissions."""
        try:
            self.brain._graduated_stream_names = {
                e.stream_name for e in self._goal_registry.graduated_entries()
            }
        except Exception:
            pass

    def _reload_graduated_streams(self) -> None:
        """Recreate graduated stream instances from the durable store on boot."""
        live = {s.name for s in self.brain.streams}
        reloaded = 0
        for entry in self._goal_registry.graduated_entries():
            if entry.stream_name in live:
                continue
            path = entry.code_path
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    code = fh.read()
            except OSError:
                continue
            # Durable code is persistent attack surface: verify the recorded
            # hash and re-run the safety validator before executing it.
            if entry.code_hash and self._code_hash(code) != entry.code_hash:
                log.warning("Graduated %s: code hash mismatch — skipping reload",
                            entry.stream_name)
                continue
            if _validate_code_safety(code) is not None:
                log.warning("Graduated %s: failed safety re-check — skipping reload",
                            entry.stream_name)
                continue
            instance = self._load_stream_from_file(path)
            if instance is None:
                continue
            instance._cap_sensors = (
                {entry.key[1]} if entry.key[0] == 'sensor' else set()
            )
            self.brain.streams.append(instance)
            self._goal_registry.add_entry(entry)
            live.add(instance.name)
            reloaded += 1
            self.add_to_log(
                f"Reloaded graduated stream '{entry.stream_name}' from {path}"
            )
        if reloaded:
            log.info("StreamFactory: reloaded %d graduated stream(s)", reloaded)
        self._refresh_graduated_names()

    def _load_stream_from_file(self, path: str):
        """Load a ProcessingStream subclass from a durable generated file."""
        from iyye_base import ProcessingStream as _PS
        mod_name = "llm_graduated_" + re.sub(
            r"[^a-z0-9_]+", "_", os.path.basename(path)[:-3].lower()
        )
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except Exception as exc:
            log.warning("Failed to load graduated file %s: %s", path, exc)
            return None
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, _PS) and obj is not _PS:
                try:
                    instance = obj()
                    for attr, default in (
                        ("urgency", 0.0), ("_last_conscious_tick", 0),
                        ("alignment_scores", {}), ("_can_be_conscious", True),
                    ):
                        if not hasattr(instance, attr):
                            setattr(instance, attr, default)
                    # Least privilege: generated code gets a scoped brain
                    # façade (journal only), never the raw brain.
                    instance.brain = ScopedBrain(self.brain)
                    instance._source_file = path
                    return instance
                except Exception as exc:
                    log.warning("Instantiation of graduated %s failed: %s",
                                obj.__name__, exc)
        return None

    # ------------------------------------------------------------------
    # Phase 3 — versioned code refinement + rollback
    # ------------------------------------------------------------------

    def _extract_code_block(self, raw: str) -> Optional[str]:
        """Pull a python code block from an LLM response (mirrors codegen)."""
        match = re.search(r"```python\s*(.*?)```", raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "class " in raw:
            return raw.strip()
        return None

    def _write_code_file(self, path: str, code: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(code)
            return True
        except OSError as exc:
            log.warning("StreamFactory: failed to write %s: %s", path, exc)
            return False

    def _hot_swap_stream(self, old, new) -> None:
        """Replace *old* with *new* in brain.streams, preserving position."""
        try:
            idx = self.brain.streams.index(old)
            self.brain.streams[idx] = new
        except ValueError:
            self.brain.streams.append(new)

    def _refine_stream_code(
        self, entry: "_CoverageEntry", target, suggestion: Dict[str, Any],
    ) -> None:
        """Revise an existing codegen stream's source in place (new version).

        Generates a replacement subclass from the current code plus observed
        issues, validates it, and hot-swaps it in while retaining the prior
        version on disk for rollback.  Evolving the handler this way avoids
        spawning yet another sibling stream for a recurring sensor/goal.
        """
        # Never disturb a stream that is conscious or in a critical section.
        if (getattr(target, '_in_critical_section', False)
                or getattr(self.brain, '_current_conscious', None) is target):
            return
        last = self._last_refine_tick.get(entry.stream_name, -10 ** 9)
        if self._tick_count - last < _REFINE_COOLDOWN_TICKS:
            return
        src_path = entry.code_path or getattr(target, '_source_file', None)
        if not src_path or not os.path.exists(src_path):
            return
        try:
            with open(src_path, "r", encoding="utf-8") as fh:
                current_code = fh.read()
        except OSError:
            return

        issues = [
            f"- Sustained usefulness EMA is {entry.usefulness_ema:.2f} over "
            f"{entry.cycles_observed} cycle(s) — below the bar."
        ]
        if entry.total_output_count:
            vr = entry.vague_output_count / entry.total_output_count
            if vr > 0.3:
                issues.append(f"- {vr * 100:.0f}% of outputs were vague or empty.")
        for ev in (suggestion.get("evidence_texts") or [])[:3]:
            issues.append(f"- New input to handle better: {str(ev)[:120]}")

        import json as _json
        sensor = entry.key[1] if entry.key[0] == "sensor" else ""
        buffered = self._unhandled_sensor_buffer.get(sensor, []) if sensor else []
        try:
            payload_sample = _json.dumps(
                buffered[-1] if buffered else {}, indent=2, default=str
            )[:1500]
        except (TypeError, ValueError):
            payload_sample = "{}"

        new_class = re.sub(r"[^A-Za-z0-9]", "", entry.stream_name) + f"V{entry.version + 1}"
        # Count the attempt against the cooldown up front so a persistently bad
        # LLM response can't be retried every tick.
        self._last_refine_tick[entry.stream_name] = self._tick_count
        submitted = self._llm_submit(
            role="codegen", kind="refine",
            call=LLMCall.from_file(
                "stream_refine",
                class_name=new_class,
                sensor_key=sensor,
                payload_sample=payload_sample,
                issues="\n".join(issues),
                current_code=current_code,
            ),
        )
        if not submitted:
            return
        self._codegen_op = {
            "op": "refine", "entry": entry, "target_name": entry.stream_name,
            "sensor": sensor, "current_code": current_code, "n_issues": len(issues),
        }

    def _apply_refine_result(self, result, op: Dict[str, Any]) -> None:
        """Install a refined stream version produced by the scheduler, retaining
        the prior version on disk for rollback."""
        entry = op["entry"]
        sensor = op.get("sensor", "")
        current_code = op.get("current_code", "")
        target = next(
            (s for s in self.brain.streams if s.name == op["target_name"]), None
        )
        if target is None:
            return
        if (getattr(target, '_in_critical_section', False)
                or getattr(self.brain, '_current_conscious', None) is target):
            return
        if result.discarded or not result.ok or not result.text:
            return

        code = self._extract_code_block(result.text)
        if code is None:
            log.warning("Refine %s: no code block in response", entry.stream_name)
            return
        try:
            compile(code, f"refine_{entry.stream_name}", "exec")
        except SyntaxError as exc:
            log.warning("Refine %s: syntax error: %s", entry.stream_name, exc)
            return
        safety_err = _validate_code_safety(code, sensor_key=sensor or None)
        if safety_err is not None:
            log.warning("Refine %s: rejected unsafe code: %s",
                        entry.stream_name, safety_err)
            return

        # Retain the current version for rollback.
        safe = re.sub(r"[^a-z0-9_]+", "_", entry.stream_name.lower()).strip("_")
        _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        prev_path = str(_GENERATED_DIR / f"{safe}.v{entry.version}.py")
        if not self._write_code_file(prev_path, current_code):
            return

        # Write the new version: graduated specs overwrite their durable file;
        # candidates get a fresh debug file.
        if entry.stage == "graduated" and entry.code_path:
            new_path = entry.code_path
        else:
            self._stream_counter += 1
            new_path = str(PROJECT_ROOT / f"streams/llm_{safe}_{self._stream_counter}.py")
        if not self._write_code_file(new_path, code):
            return

        new_inst = self._load_stream_from_file(new_path)
        if new_inst is None:
            return
        # Preserve identity so the registry entry keeps mapping to this stream.
        new_inst.name = entry.stream_name
        new_inst._source_file = new_path
        new_inst._cap_sensors = (
            {entry.key[1]} if entry.key[0] == 'sensor' else set()
        )
        self._hot_swap_stream(target, new_inst)

        entry.prev_code_path = prev_path
        entry.prev_ema = entry.usefulness_ema
        entry.version += 1
        entry.cycles_observed = 0      # re-probate the new version
        if entry.stage == "graduated":
            entry.code_path = new_path
            entry.code_hash = self._code_hash(code)
        self.add_to_log(
            f"Refined '{entry.stream_name}' → v{entry.version} "
            f"({op.get('n_issues', 0)} issue(s); prev ema={entry.prev_ema:.2f})"
        )

    def _convert_graduated_plans(self) -> None:
        """Upgrade a graduated *plan* stream into durable codegen.

        A plan that proved useful should become a real, refinable code handler
        instead of staying a step-list forever.  Only sensor-bound plans are
        converted (goal codegen tends to produce noise).  One attempt per tick,
        throttled per spec; a swap so it doesn't change the stream count."""
        for entry in self._goal_registry.graduated_entries():
            if entry.kind != "plan" or entry.code_path or entry.key[0] != "sensor":
                continue
            last = self._last_refine_tick.get(entry.stream_name, -10 ** 9)
            if self._tick_count - last < _REFINE_COOLDOWN_TICKS:
                continue
            target = next(
                (s for s in self.brain.streams if s.name == entry.stream_name), None
            )
            if target is None:
                continue
            if (getattr(target, '_in_critical_section', False)
                    or getattr(self.brain, '_current_conscious', None) is target):
                continue
            sensor = entry.key[1]
            buffered = self._unhandled_sensor_buffer.get(sensor, [])
            recent_inputs = [
                (it.get('data', it) if isinstance(it, dict) else it)
                for it in buffered[-5:]
            ]
            alignment: Dict[str, float] = {}
            for s in self.brain.streams:
                for g, v in (getattr(s, 'alignment_scores', {}) or {}).items():
                    if v > alignment.get(g, 0):
                        alignment[g] = v
            self._last_refine_tick[entry.stream_name] = self._tick_count
            # Submit codegen async; _apply_convert_result hot-swaps on result.
            self._submit_codegen(
                op="convert",
                source_name=f"graduated_{sensor}",
                primary_goal=None,
                recent_inputs=recent_inputs,
                alignment=alignment,
                sensor_key=sensor,
                apply_ctx={"entry": entry, "sensor": sensor},
            )
            return  # one conversion attempt per tick

    def _apply_convert_result(self, instance, op: Dict[str, Any]) -> None:
        """Install a converted plan→codegen stream produced by the scheduler."""
        if instance is None:
            return
        entry = op["entry"]
        target = next(
            (s for s in self.brain.streams if s.name == entry.stream_name), None
        )
        if target is None:
            return
        instance.name = entry.stream_name
        self._hot_swap_stream(target, instance)
        entry.kind = "codegen"
        try:
            with open(instance._source_file, "r", encoding="utf-8") as fh:
                self._persist_graduated_code(entry, fh.read())
            instance._source_file = entry.code_path
        except OSError:
            pass
        self.add_to_log(
            f"Converted graduated plan '{entry.stream_name}' to durable codegen"
        )

    def _maybe_rollback(self, instance, entry: "_CoverageEntry") -> bool:
        """Roll back to the retained version if a refinement underperformed.

        Returns True if a rollback (or acceptance) decision was made this call."""
        if entry.prev_ema is None or not entry.prev_code_path:
            return False
        if entry.cycles_observed < _ROLLBACK_WINDOW:
            return False  # give the new version time to prove itself
        if entry.usefulness_ema + _ROLLBACK_MARGIN >= entry.prev_ema:
            # New version is at least as good — accept it, clear rollback state.
            entry.prev_ema = None
            entry.prev_code_path = None
            return False
        # Regression: restore the retained predecessor.
        if os.path.exists(entry.prev_code_path):
            restored = self._load_stream_from_file(entry.prev_code_path)
            if restored is not None:
                restored.name = entry.stream_name
                restored._source_file = entry.prev_code_path
                restored._cap_sensors = (
                    {entry.key[1]} if entry.key[0] == 'sensor' else set()
                )
                self._hot_swap_stream(instance, restored)
                if entry.stage == "graduated":
                    try:
                        with open(entry.prev_code_path, "r", encoding="utf-8") as fh:
                            self._persist_graduated_code(entry, fh.read())
                    except OSError:
                        pass
                entry.usefulness_ema = entry.prev_ema
                entry.version += 1
                self.add_to_log(
                    f"Rolled back '{entry.stream_name}' → v{entry.version}: "
                    f"refinement ema {entry.usefulness_ema:.2f} < prior "
                    f"{entry.prev_ema:.2f}"
                )
        entry.prev_ema = None
        entry.prev_code_path = None
        entry.cycles_observed = 0
        return True

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['created_streams'] = self._created_streams
        state['covered_sensors'] = list(self._covered_sensors)
        state['goal_registry'] = self._goal_registry.to_dict()
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self._created_streams = state.get('created_streams', [])
        # _covered_sensors is intentionally NOT restored — it is a runtime
        # cache that must match live streams.  On restart no codegen streams
        # are loaded (stale files are deleted), so restoring stale coverage
        # would block sensor buffering until the first reconciliation tick.
        # The goal registry is the authoritative source of coverage state.
        registry_data = state.get('goal_registry')
        if registry_data:
            self._goal_registry = GoalCoverageRegistry.from_dict(registry_data)

