#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Iyye – high‑level orchestrator.
"""

# --------------------------------------------------------------------------- #
# Imports – keep them lightweight; heavy deps are imported lazily in stubs.
# --------------------------------------------------------------------------- #
import os
import re
import json
import time
import atexit
import logging
import multiprocessing as mp
from datetime import datetime, timezone
from enum import Enum, auto
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Callable, Optional

from iyye_base import PROJECT_ROOT, BaseSensorQueue, BaseActuator, ProcessingStream

# Matches text that describes an ephemeral system metric snapshot.
# Facts matching this regex are never promoted to LTM.
_EPHEMERAL_METRIC_RE = re.compile(
    r'\b(?:cpu|memory|mem|disk|ram|adenosine)\b.{0,40}\b\d+\.?\d*\s*%'
    r'|\b\d+\.?\d*\s*%.{0,40}\b(?:cpu|memory|mem|disk|ram)\b'
    r'|\badenosine\s+(?:level|registers).{0,40}\b\d+\.?\d*\b'
    r'|\b(?:cpu|memory|mem)\s+(?:usage|utilization|load|level)\b',
    re.IGNORECASE,
)

# Matches LTM-unworthy content: LLM null-result placeholders, system status
# sentences, vague cognitive/operational state descriptions, and LLM
# chain-of-thought / reasoning artefacts that carry no durable information
# about the user or the world.
_LTM_NOISE_RE = re.compile(
    # LLM "nothing to report" placeholders (often wrapped in parens/brackets/backticks)
    r'^\s*[\(\[]?no facts?\b'
    r'|^\s*[\(\[]?nothing\s+(?:to\s+)?(?:store|report|note|extract)'
    r'|^\s*[\(\[]?no\s+(?:new\s+)?(?:information|data|content|entries?)\b'
    r'|^\s*[\(\[]?empty(?:\b|_)'                       # "empty output", "Empty response", "empty_response", etc.
    r'|^\s*```\s*empty\s*```'                         # markdown-wrapped "empty"
    r'|\bno\s+(?:facts?|information)\s+worth\b'       # "no facts worth storing" mid-sentence
    r'|\bno\s+(?:significant\s+)?long-term\s+memory\b'
    # System / LLM operational status sentences
    r'|\bllm\s+is\s+(?:operational|running|available|online|healthy)\b'
    r'|\bllm\s+(?:server\s+)?(?:status|health)\b'
    r'|\bmodel\s+is\s+(?:loaded|running|available|operational)\b'
    # Vague cognitive/energy state — not useful facts about the world
    r'|\b(?:cognitive|mental)\s+state\b'
    r'|\b(?:high\s+)?alertness\b'
    r'|\benergy\s+levels?\s+(?:are|is)\b'
    r'|\boptimal\s+(?:energy|processing|performance)\b'
    r'|\bfully\s+(?:awake|alert|operational|active)\b'
    # Vague system-status sentences that say nothing about the world
    r'|\bstatus\s+remains?\s+nominal\b'
    r'|\b(?:system|module)\s+(?:status\s+)?(?:is\s+)?(?:active|nominal|stable|healthy|functioning)\b'
    r'|\bno\s+(?:significant|notable|meaningful)\s+(?:change|event|update)\b'
    r'|\boperating\s+(?:normally|within\s+(?:normal|expected))\b'
    # Internal stream operational noise — never a fact about the world
    r'|\bcurrent\s+(?:operational\s+)?goal\s+is\b'
    r'|\bcuriosity\s+(?:was\s+)?(?:fulfilled|triggered|satisfied)\b'
    r'|\bsent\s+(?:proactive|hardware|agency)\b'
    r'|\b(?:proactive|curiosity)\s+(?:message|inquiry|follow-?up|outreach)\s+(?:was\s+)?sent\b'
    r'|\bproactive\s+(?:social\s+)?(?:follow-?up|outreach)\b'
    r'|\bsocial\s+follow-?up\s+(?:prompt|was\s+sent)\b'
    r'|\bsynthesized\s+(?:social|proactive)\b'
    r'|\bagency\s+(?:opportunity|assertion)\s+(?:was\s+)?(?:identified|executed)\b'
    r'|\bagency-driven\s+suggestions?\b'
    r'|\bhardware\s+suggest(?:ion|ed|s)\b.{0,20}\b(?:sent|identified)\b'
    r'|\bcurious\s+observations?\s+were\s+promoted\b'
    r'|\b(?:proactive\s+)?curiosity\s+prompts?\s+(?:was\s+|were\s+)?sent\b'
    r'|\bfollow-?up\s+prompts?\s+(?:were\s+)?synthesized\b'
    # Energy / sleep-cycle noise from self-preservation streams
    r'|\bcritical\s+energy\s+levels?\b'
    r'|\bpreparing\s+for\s+(?:a\s+)?(?:maintenance|sleep)\b'
    r'|\benergy\s+levels?\s+(?:may|might|could)\b'
    r'|\binitiating\s+(?:maintenance|sleep|rest)\b'
    r'|\bscanning\s+\d+\s+stream'
    # LLM chain-of-thought / reasoning artefacts
    r'|<\|thought\|>'
    r'|<\|/thought\|>'
    r'|^\s*\d+\.\s+\*\*'             # numbered bold steps: "1. **Analyze...**"
    r'|^\s*\*\*(?:Analyze|Step|Rule|Output|Summary|Conclusion)\b'
    r'|^\s*Rule:\s'                   # rule recitation
    r'|^\s*(?:Let me|I need to|I will|I should)\s'  # self-narration
    r'|\bmemory\s+consolidation\s+module\b'          # prompt echo
    r'|\bthe user wants me to\b'
    # LLM meta-commentary about having nothing to output
    r'|^\s*[\(\*`].*\bno\s+facts\s+(?:were|are)\s+found\b'
    r'|\boutput[\s_]+(?:should[\s_]+be[\s_]+)?(?:empty|nothing)\b'
    r'|\bempty[\s_]+response\b'
    r'|^empty_response$'                                 # bare token from LLM
    r'|\bno[\s_]+facts[\s_]+(?:were[\s_]+)?(?:found|extracted)\b'
    r'|\bnothing[\s_]+to[\s_]+extract\b'
    r'|\bself-correction\b'
    r'|^\s*\(Note:'
    r'|^\s*\(Actual\s+Output\)'
    r'|^\s*```\s*\(Note:'
    # Planned-stream advice / action-recommendation pseudo-facts
    r'|^\s*\*\*Action:\*\*'
    r'|^\s*\*\*Why:\*\*'
    r'|\blatent\s+agency\b'
    r'|\bconvert\s+latent\s+\w+\s+into\b'
    r'|\bexecute\s+all\s+(?:queued|drafted|pending|scheduled|finalized)\b'
    r'|\bwithout\s+further\s+deliberation\b'
    r'|\btransition\s+from\s+(?:internal\s+)?processing\s+to\s+external\s+impact\b'
    r'|\bconvert(?:ing)?\s+thought\s+into\s+(?:real|measurable|tangible)\b'
    r'|\bpotentiality\s+to\s+actuality\b',
    re.IGNORECASE,
)

# Streams whose output is never sent to the LLM for fact extraction during
# the sleep-phase replay.  Chat / telegram are skipped by keyword match.
_REPLAY_SKIP_LLM = frozenset({
    'attention_stream', 'alignment_stream', 'stream_factory',
    'adenosine_stream', 'stm_update', 'llm_management',
    'self_reflection',
})

# Prefix match: any stream whose name starts with one of these prefixes is
# also skipped for LLM fact extraction during replay.  LLM-generated streams
# log only internal operational noise ("curiosity fulfilled", "system active")
# that is never a fact about the world.
_REPLAY_SKIP_PREFIXES = (
    'llmsuggested',     # matches LlmsuggestedAgencystream3, etc.
    'llm_suggested',    # matches llm_suggested_hardware_stream, etc.
    'llmexplore',       # matches LlmExploreSocialFollowUp, etc.
    'explore_',         # matches explore_social_followup_stream, etc.
    'suggested_',       # matches suggested_self_preservation_monitor, etc.
    'plan_suggested',   # PlannedContinuationStream fallback streams
    'research_',        # WebResearchStream — result already sent to user
    'hardware_',        # matches hardware_suggestion_curiosity, etc.
)

# LLM-generated streams sometimes use custom names that don't start with
# any skip prefix but still contain telltale keywords.  These are checked
# as substring containment (not prefix match) as a safety net.
_REPLAY_SKIP_KEYWORDS = ('_suggestion_', '_suggested_')

# --------------------------------------------------------------------------- #
# Logging configuration – easy to turn on/off from the command line.
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
log = logging.getLogger("Iyye")

def _stamp(entry: Dict[str, Any]) -> Any:
    """Unwrap a sensor queue entry, injecting the timestamp into dict payloads.

    Sensor queues store ``{"ts": ..., "data": <payload>}``.  If the payload
    is a dict it gets an ``_ts`` key so downstream streams can see when the
    item was enqueued (HLD: "along with timestamps").  String payloads are
    returned as-is to keep existing consumers working.
    """
    data = entry.get("data", entry)
    ts = entry.get("ts")
    if isinstance(data, dict) and ts is not None:
        data.setdefault("_ts", ts)
    return data


# --------------------------------------------------------------------------- #
# 1 High‑level state machine
# --------------------------------------------------------------------------- #
class MindState(Enum):
    """Four logical states from the HLD."""
    ASLEEP        = auto()
    WAKING_UP     = auto()
    AWAKE         = auto()
    WINDING_DOWN  = auto()

# --------------------------------------------------------------------------- #
# 3 Long‑term memory client & short‑term memory
# --------------------------------------------------------------------------- #
from iyye_io.memory_mcp_client import MemoryClient
from iyye_io.short_term_memory import ShortTermMemory

import sys
import importlib.util

# Fix double-import: when run as __main__, register ourselves under our
# module name so that 'from main_loop import X' in stream/io files gets
# the same class objects as the running script, not a second copy.
# Without this, enum comparisons like `brain.state == MindState.AWAKE`
# silently fail because each side holds a different class instance.
if __name__ == '__main__':
    sys.modules.setdefault('main_loop', sys.modules['__main__'])

# --------------------------------------------------------------------------- #
# 4 Brain – orchestrates sensors, memory and streams
# --------------------------------------------------------------------------- #
class IyyeBrain:
    """
    High‑level orchestrator.

    *   Instantiates the always‑on ``web_chat`` sensor.
    *   Dynamically discovers additional IO sensors (./io) and processing
        streams (./streams).
    """
    @property
    def adenosine(self) -> "AdenosineStream":
        """Get adenosine stream, creating default if needed."""
        if self._adenosine_stream is None:
            from streams.adenosine_stream import AdenosineStream
            self._adenosine_stream = AdenosineStream(self)
            # Ensure adenosine stream is registered in streams list for execution
            if hasattr(self, 'streams') and self._adenosine_stream not in self.streams:
                self.streams.append(self._adenosine_stream)
                log.info("Auto-registered adenosine_stream")
        return self._adenosine_stream

    def __init__(self):
        # ------------------------------------------------------------------- #
        # Sensors – web chat is created locally; UI can attach to it later.
        # ------------------------------------------------------------------- #
        self.sensors: Dict[str, BaseSensorQueue] = {}
        self.sensors["web_chat"] = BaseSensorQueue("web_chat")
        self.actuators: Dict[str, BaseActuator] = {}
        try:
            from web_chat_2 import attach_sensor
            # give Flask UI direct access to the queue (optional)
            attach_sensor(self.sensors["web_chat"])
        except Exception as exc:  # pragma: no‑cover – UI optional
            log.warning(
                "Could not register web chat sensor with Flask UI: %s", exc
            )

        root_dir = os.path.dirname(os.path.abspath(__file__))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        # Pre-import base modules so they're available in sys.modules
        import iyye_base
        import mcp_client

        self._discover_io()

        # ------------------------------------------------------------------- #
        # Memory (real stub, not the dummy class that was previously inlined)
        # ------------------------------------------------------------------- #
        self.memory = MemoryClient()
        self._clean_polluted_ltm()

        # ------------------------------------------------------------------- #
        # Short-term memory (structured fact store, in-memory + daily JSONL)
        # ------------------------------------------------------------------- #
        self.stm = ShortTermMemory()

        # ------------------------------------------------------------------- #
        # Processing streams – loaded dynamically.
        # ------------------------------------------------------------------- #
        self.streams: List[ProcessingStream] = []
        self._adenosine_stream: Optional["AdenosineStream"] = None
        self._load_streams()

        # ------------------------------------------------------------------- #
        # State & housekeeping
        # ------------------------------------------------------------------- #
        self.state: MindState = MindState.ASLEEP
        self.iyye_day: int = self._load_iyye_state().get("iyye_day", 0)
        atexit.register(self.shutdown)
        self._current_conscious: Optional[ProcessingStream] = None
        self._attention_stream: Optional[ProcessingStream] = None
        self._waking_interrupted: bool = False
        self._waking_up_tick: int = 0
        self.last_conscious_log: List[Dict[str, Any]] = self._load_last_cycle()
        self.winding_down_started: bool = False
        self._wakeup_reason: Optional[str] = None
        # Lightweight inter-stream mailbox.  Any stream can post a message
        # addressed to another stream by name; the recipient drains its
        # mailbox at the start of its execute() tick.
        self._mailboxes: Dict[str, List[Dict[str, Any]]] = {}
        # Sleep-phase tracking — reset in _enter_asleep, consumed in _asleep_actions
        self._sleep_did_system_check: bool = False
        self._sleep_did_stm_flush: bool = False
        self._sleep_did_replay: bool = False
        self._sleep_prewarm_sent: bool = False
        # HLD: "dreaming" replay is skipped on the very first sleep of this process run.
        self._is_first_sleep: bool = True
        # Cursor may have been restored from last_cycle.json by _load_last_cycle;
        # fall back to 0 if not (first run or old-format file).
        self._replay_cursor: int = getattr(self, '_replay_cursor', 0)

    _LAST_CYCLE_PATH = PROJECT_ROOT / "last_cycle.json"
    _IYYE_STATE_PATH = PROJECT_ROOT / "iyye_state.json"

    def _load_iyye_state(self) -> Dict[str, Any]:
        """Load persisted brain state (iyye_day, etc.) from disk."""
        try:
            if self._IYYE_STATE_PATH.exists():
                return json.loads(self._IYYE_STATE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not load iyye_state.json: %s", exc)
        return {}

    def _save_iyye_state(self) -> None:
        """Persist brain state counters and stream_factory registry to disk."""
        try:
            data: Dict[str, Any] = {"iyye_day": self.iyye_day}
            # Persist stream_factory's goal coverage registry so it
            # survives restarts — prevents duplicate goal streams.
            factory = next(
                (s for s in self.streams if s.name == 'stream_factory'), None
            )
            if factory is not None and hasattr(factory, 'get_state'):
                data["stream_factory"] = factory.get_state()
            self._IYYE_STATE_PATH.write_text(
                json.dumps(data, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.warning("Could not save iyye_state.json: %s", exc)

    def _load_last_cycle(self) -> List[Dict[str, Any]]:
        """Load the previous awake cycle's log and replay cursor from disk."""
        try:
            if not self._LAST_CYCLE_PATH.exists():
                return []
            data = json.loads(self._LAST_CYCLE_PATH.read_text(encoding="utf-8"))

            # New format: {"entries": [...], "cursor": N}
            # Old format: plain list (backward compat)
            if isinstance(data, dict):
                raw = data.get("entries", [])
                saved_cursor = int(data.get("cursor", 0))
            elif isinstance(data, list):
                raw = data
                saved_cursor = 0
            else:
                log.warning("last_cycle.json: unrecognised format — ignoring")
                return []

            valid = [e for e in raw if isinstance(e, dict) and 'result' in e]
            dropped = len(raw) - len(valid)
            if dropped:
                log.warning("last_cycle.json: dropped %d malformed entries", dropped)

            # Restore cursor so replay resumes where it left off across restarts.
            self._replay_cursor = min(saved_cursor, len(valid))
            log.info("Loaded last_conscious_log from %s (%d entries, cursor=%d)",
                     self._LAST_CYCLE_PATH, len(valid), self._replay_cursor)
            return valid
        except Exception as exc:
            log.warning("Could not load last cycle log: %s", exc)
        return []

    def _save_last_cycle(self) -> None:
        """Persist the current awake cycle's log and replay cursor to disk.

        No entries are trimmed — HLD requires "replaying full conscious stream
        from last awake cycle memory".  Replay is already incremental (batches
        of _REPLAY_BATCH per sleep tick) so large logs don't cause a spike.
        """
        entries = self.last_conscious_log
        cursor = getattr(self, '_replay_cursor', 0)

        if not entries:
            # Replay completed (or no awake cycle yet) — remove the file so a
            # restart doesn't re-replay the same cycle.
            try:
                if self._LAST_CYCLE_PATH.exists():
                    self._LAST_CYCLE_PATH.unlink()
                    log.info("Cleared last_cycle.json (replay completed)")
            except Exception as exc:
                log.warning("Could not remove last_cycle.json: %s", exc)
            return
        try:
            self._LAST_CYCLE_PATH.write_text(
                json.dumps({"entries": entries, "cursor": cursor},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info("Saved last_conscious_log to %s (%d entries, cursor=%d)",
                     self._LAST_CYCLE_PATH, len(entries), cursor)
        except Exception as exc:
            log.warning("Could not save last cycle log: %s", exc)

    # --------------------------------------------------------------- #
    # One-time LTM cleanup (removes polluted planned-stream pseudo-facts)
    # --------------------------------------------------------------- #

    _LTM_POLLUTION_RE = re.compile(
        r'^\s*\*\*Action:\*\*'
        r'|^\s*\*\*Why:\*\*'
        r'|\blatent\s+agency\b'
        r'|\bconvert\s+latent\s+\w+\s+into\b'
        r'|\bexecute\s+all\s+(?:queued|drafted|pending|scheduled|finalized)\b'
        r'|\bwithout\s+further\s+deliberation\b'
        r'|\btransition\s+from\s+(?:internal\s+)?processing\s+to\s+external\s+impact\b'
        r'|\bconvert(?:ing)?\s+thought\s+into\s+(?:real|measurable|tangible)\b'
        r'|\bpotentiality\s+to\s+actuality\b',
        re.IGNORECASE,
    )

    def _clean_polluted_ltm(self) -> None:
        """Remove planned-stream pseudo-facts that leaked into LTM.

        Runs once at startup.  Targets rows where:
        - source starts with plan_suggested_ / suggested_ / llm_suggested_
        - OR text matches advice-shaped patterns (Action/Why, latent agency, etc.)
        """
        try:
            df = self.memory._tbl.to_pandas()
        except Exception:
            return
        to_delete = []
        for _, row in df.iterrows():
            source = str(row.get('source', ''))
            text = str(row.get('text', ''))
            src_lower = source.lower()
            if (any(src_lower.startswith(p) for p in _REPLAY_SKIP_PREFIXES)
                    or self._LTM_POLLUTION_RE.search(text)):
                to_delete.append(row['id'])
        if to_delete:
            for fid in to_delete:
                try:
                    self.memory.delete_fact(fid)
                except Exception:
                    pass
            log.info("LTM cleanup: removed %d polluted pseudo-fact(s)", len(to_delete))

    # --------------------------------------------------------------- #
    # Dynamic loading helpers
    # --------------------------------------------------------------- #

    def _discover_io(self) -> None:
        """Search ./iyye_io/ for .py files and instantiate subclasses of BaseSensorQueue."""
        io_dir = PROJECT_ROOT / "iyye_io"
        if not io_dir.is_dir():
            log.warning("Directory 'iyye_io' not found – skipping dynamic sensors.")
            return

        for fname in os.listdir(io_dir):
            if not (fname.endswith(".py") and fname != "__init__.py"):
                continue
            mod_name = fname[:-3]
            # Never let a file inside iyye_io/ shadow a module that was already
            # imported from the project root (e.g. mcp_client, iyye_base).
            # Such duplicates cause hard-to-trace import aliasing bugs.
            if mod_name in sys.modules:
                log.debug("_discover_io: skipping %s — '%s' already in sys.modules",
                          fname, mod_name)
                continue
            file_path = str(io_dir / fname)
            spec = importlib.util.spec_from_file_location(mod_name, file_path)
            if spec is None or spec.loader is None:
                log.warning("Cannot create loader for %s – skipping.", fname)
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            try:  # load the file
                spec.loader.exec_module(module)   # type: ignore[arg-type]
                for obj_name, obj in vars(module).items():
                    if isinstance(obj, type):
                        # --------------------------------------------------- #
                        # Sensors
                        # --------------------------------------------------- #
                        if issubclass(obj, BaseSensorQueue) and obj is not BaseSensorQueue:
                            instance = obj()          # default ctor – cheap objects
                            key = getattr(instance, 'name', obj_name)
                            self.sensors[key] = instance
                            # Some sensors use a background-thread model (start_collection)
                            # rather than poll(); start them immediately after instantiation.
                            if callable(getattr(instance, 'start_collection', None)):
                                try:
                                    instance.start_collection()
                                    log.info("Started background collection for %s", key)
                                except Exception as exc:
                                    log.warning("start_collection failed for %s: %s", key, exc)
                            log.info("Loaded sensor %s from %s", key, fname)

                        # --------------------------------------------------- #
                        # Actuators
                        # --------------------------------------------------- #
                        elif issubclass(obj, BaseActuator) and obj is not BaseActuator:
                            instance = obj()
                            key = getattr(instance, 'name', obj_name)
                            self.actuators[key] = instance
                            log.info("Loaded actuator %s from %s", key, fname)

            except Exception as exc:  # pragma: no‑cover
                log.error("Failed to load IO sensor %s – %s", fname, exc)

    def _load_streams(self) -> None:
        """Search ./streams/ for .py files and instantiate subclasses of ProcessingStream.

        LLM-generated streams (files starting with 'llm_') are always loaded
        regardless of _factory_created, because they use no-arg constructors and
        must survive restarts.  The _factory_created guard only applies to shipped
        streams like UserChatStream / PlannedContinuationStream that require
        constructor arguments.
        """
        streams_dir = PROJECT_ROOT / "streams"
        if not streams_dir.is_dir():
            log.warning("Directory 'streams' not found – no streams loaded.")
            return

        for fname in sorted(os.listdir(streams_dir)):
            if not (fname.endswith(".py") and fname != "__init__.py"):
                continue
            # LLM-suggested streams are short-lived and recreated on demand
            # by StreamFactory when self-reflection generates fresh goals.
            # Reloading stale ones from a previous session re-introduces
            # streams that should have been pruned.
            if fname.startswith("llm_suggested_"):
                log.debug("Skipping stale LLM-suggested stream file: %s", fname)
                continue
            is_llm_generated = fname.startswith("llm_")
            mod_name = fname[:-3]
            file_path = str(streams_dir / fname)
            spec = importlib.util.spec_from_file_location(mod_name, file_path)
            if spec is None or spec.loader is None:
                log.warning("Cannot create loader for %s – skipping.", fname)
                continue

            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            try:
                spec.loader.exec_module(module)   # type: ignore[arg-type]
                for obj in vars(module).values():
                    if not (
                        isinstance(obj, type)
                        and issubclass(obj, ProcessingStream)
                        and obj is not ProcessingStream
                    ):
                        continue
                    # Skip shipped streams that require constructor arguments
                    # (e.g. subconscious streams needing brain); they are
                    # instantiated explicitly in _start_subconscious_streams().
                    # LLM-generated files always use no-arg constructors.
                    if not is_llm_generated:
                        import inspect
                        try:
                            sig = inspect.signature(obj.__init__)
                            required = [
                                p for p in sig.parameters.values()
                                if p.name != "self"
                                and p.default is inspect.Parameter.empty
                            ]
                            if required:
                                log.debug(
                                    "Skipping %s from %s — requires args: %s",
                                    obj.__name__, fname,
                                    [p.name for p in required],
                                )
                                continue
                        except (ValueError, TypeError):
                            pass
                    try:
                        instance = obj()
                    except TypeError as exc:
                        log.debug(
                            "Skipping %s from %s — constructor error: %s",
                            obj.__name__, fname, exc,
                        )
                        continue
                    # Inject brain reference so the stream can access memory/sensors.
                    instance.brain = self
                    if is_llm_generated:
                        instance._source_file = file_path
                    self.streams.append(instance)
                    log.info("Loaded processing stream %s from %s", obj.__name__, fname)
            except Exception as exc:
                log.error("Failed to load processing stream %s – %s", fname, exc)

    def _start_subconscious_streams(self) -> None:
        """Initialize all special subconscious streams on wakeup."""
        # Import here to avoid circular dependency
        from streams.attention_stream import AttentionStream
        from streams.alignment_stream import AlignmentStream
        from streams.stream_factory import StreamFactory
        from streams.self_reflection_stream import SelfReflectionStream
        from streams.adenosine_stream import AdenosineStream
        from streams.stm_update_stream import StmUpdateStream
        from streams.llm_management_stream import LlmManagementStream
        from streams.theory_of_mind_stream import TheoryOfMindStream

        # Create special streams if not already present
        special_names = {'attention_stream', 'alignment_stream',
                        'stream_factory', 'self_reflection', 'adenosine_stream',
                        'stm_update', 'llm_management', 'theory_of_mind'}
        existing_names = {s.name for s in self.streams}

        for name in special_names - existing_names:
            if name == 'attention_stream':
                stream = AttentionStream(self)
                self._attention_stream = stream
            elif name == 'alignment_stream':
                stream = AlignmentStream(self)
            elif name == 'stream_factory':
                stream = StreamFactory(self)
                # Restore persisted registry so goal coverage survives restarts.
                saved = self._load_iyye_state().get("stream_factory")
                if saved:
                    stream.restore_state(saved)
            elif name == 'self_reflection':
                stream = SelfReflectionStream(self)
            elif name == 'adenosine_stream':
                stream = AdenosineStream(self)
                self._adenosine_stream = stream
            elif name == 'stm_update':
                stream = StmUpdateStream(self)
            elif name == 'llm_management':
                stream = LlmManagementStream(self)
            elif name == 'theory_of_mind':
                stream = TheoryOfMindStream(self)
            else:
                continue

            self.streams.append(stream)
            log.info("Started subconscious stream: %s", name)

    # --------------------------------------------------------------- #
    # Core tick loop
    # --------------------------------------------------------------- #
    def run_once(self) -> None:
        """Execute a single logical tick."""
        self._tick_counter = getattr(self, "_tick_counter", 0) + 1
        log.debug(
            "=== Tick %d – state=%s ===",
            self._tick_counter,
            self.state.name,
        )

        # ------------------------------------------------------------------- #
        # Gather sensor payloads (each returns a list of raw data items)
        # ------------------------------------------------------------------- #
        sensors_data: Dict[str, List[Any]] = {}
        for name, q in self.sensors.items():
            if callable(getattr(q, "poll", None)):
                try:
                    q.poll()
                except Exception as exc:
                    log.warning("Sensor %s poll error: %s", name, exc)
            payloads = q.pop_all()
            if payloads:
                sensors_data[name] = [_stamp(p) for p in payloads]

        # ------------------------------------------------------------------- #
        # Dispatch based on current state
        # ------------------------------------------------------------------- #
        if self.state == MindState.ASLEEP:
            self._asleep_actions(sensors_data)
        elif self.state == MindState.WAKING_UP:
            self._waking_up_tick = getattr(self, "_waking_up_tick", 0) + 1
            # Accumulate sensor data arriving during WAKING_UP so the first
            # AWAKE tick can process it rather than silently dropping it.
            if sensors_data:
                acc = getattr(self, '_waking_up_sensors_data', {})
                for _n, _items in sensors_data.items():
                    acc.setdefault(_n, [])
                    acc[_n].extend(_items)
                self._waking_up_sensors_data = acc
            self._waking_up_actions()
            # Transition happens inside _waking_up_actions when ready
        elif self.state == MindState.AWAKE:
            cmds = self._awake_actions(sensors_data)
            # Emit a generic debug line to every registered actuator.
            self._debug_to_actuators(
                f"[DEBUG] Tick {self._tick_counter} completed – state={self.state.name}"
            )
        elif self.state == MindState.WINDING_DOWN:
            # Accumulate any sensor data that arrives while winding down so it
            # is not permanently lost.  Sensors are popped above (the MCP server
            # advances last_update_id on poll), so we must save the data here or
            # it will be gone before the brain wakes up again.
            if sensors_data:
                acc = getattr(self, '_winding_down_sensors', {})
                for _n, _items in sensors_data.items():
                    acc.setdefault(_n, [])
                    acc[_n].extend(_items)
                self._winding_down_sensors = acc
            self._winding_down_actions()

    # --------------------------------------------------------------- #
    # State‑specific helpers
    # --------------------------------------------------------------- #
    def _waking_up_actions(self) -> None:
        """
        HLD: "Waking up state starts all subcon execution streams, selects 
        conscious stream to be either one reflecting over high priority input 
        that woke up the system in case of wake up caused by interrupt, or 
        self-reflection subconscious execution stream on 'fully rested' wakeup."
        """
        # Start all subconscious streams (first time only)
        if not getattr(self, '_subconscious_started', False):
            self._start_subconscious_streams()
            self._subconscious_started = True
            log.info("Started subconscious streams")
            # Ensure LLM is healthy before any stream can make a request.
            llm_mgmt = next(
                (s for s in self.streams if s.name == 'llm_management'), None
            )
            if llm_mgmt is not None:
                llm_mgmt.ensure_running()
            # Do not return
        
        # HLD: "selects conscious stream to be either one reflecting over high priority
        # input that woke up the system … or self-reflection on fully rested wakeup."
        if getattr(self, "_waking_interrupted", False):
            # Drive StreamFactory with the preserved interrupt payload so that the
            # UserChatStream exists before we select the conscious stream.
            pending = getattr(self, '_pending_interrupt_data', None)
            if pending:
                factory = next(
                    (s for s in self.streams if s.name == 'stream_factory'), None
                )
                if factory is not None:
                    try:
                        factory.execute({
                            'sensors_data': pending,
                            'streams': self.streams,
                            'memory': self.memory,
                            'actuators': self.actuators,
                            'adenosine': self.adenosine.level,
                            'tick_counter': 0,
                        })
                        # Only clear after success; on failure _awake_actions
                        # will merge the data back into the first awake tick.
                        self._pending_interrupt_data = None
                    except Exception as exc:
                        log.warning("StreamFactory pre-run during wakeup failed: %s", exc)
            self._select_conscious_for_interrupt()
        else:
            self._select_conscious_self_reflection()

        # Transition to awake
        self._enter_awake()        
    
    def _select_conscious_for_interrupt(self) -> None:
        """Select conscious stream based on interrupting input.

        Uses two signals to detect *actual pending work*, not historical activity:
        1. _pending_messages — UserChatStream's unprocessed message queue (direct).
        2. len(input_history) - len(output_history) — generic unprocessed-input
           proxy valid for any ProcessingStream subclass.

        input_history alone is append-only and always non-empty after the first
        message; using it without comparing to output_history would pick an old
        exhausted stream over the freshly-created one.
        """
        _SPECIAL = {'self_reflection', 'attention_stream', 'alignment_stream',
                    'stream_factory', 'adenosine_stream', 'llm_management'}
        candidates = []
        for stream in self.streams:
            if stream.name in _SPECIAL:
                continue
            # Direct pending-work count (UserChatStream and similar)
            pending_msgs = len(getattr(stream, '_pending_messages', []))
            # Generic proxy: inputs not yet answered
            unprocessed = max(
                0,
                len(getattr(stream, 'input_history', []))
                - len(getattr(stream, 'output_history', []))
            )
            score = max(pending_msgs, unprocessed)
            if score > 0:
                candidates.append((score, getattr(stream, 'priority', 1), stream))

        if candidates:
            # Highest pending count first; break ties by priority
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            stream = candidates[0][2]
            self._current_conscious = stream
            stream.is_conscious = True
            self._was_conscious_streams.add(stream.name)
            log.info("Selected %s for interrupt handling (pending_score=%d)",
                     stream.name, candidates[0][0])
            return

        # No stream has pending work — fall back to self-reflection
        self._select_conscious_self_reflection()
    
    def _select_conscious_self_reflection(self) -> None:
        """Select self-reflection stream as conscious (natural wakeup)."""
        for stream in self.streams:
            if stream.name == "self_reflection":
                self._current_conscious = stream
                stream.is_conscious = True
                log.info("Selected self_reflection as conscious stream")
                return
        
        # Fallback to first available stream
        if self.streams:
            self._current_conscious = self.streams[0]
            self._current_conscious.is_conscious = True

    def _awake_actions(self, sensors_data: Dict[str, List[Any]]) -> Optional[List[Any]]:
        """Main awake state processing."""
        if not self.streams:
            log.warning("No active streams!")
            return None

        # Merge sensor data collected during WAKING_UP (would otherwise be dropped).
        waking_acc = getattr(self, '_waking_up_sensors_data', None)
        if waking_acc:
            for name, items in waking_acc.items():
                if items:
                    sensors_data.setdefault(name, [])
                    sensors_data[name] = items + sensors_data[name]
            self._waking_up_sensors_data = {}

        # Merge any sensor data that was captured during the interrupt wakeup tick.
        # That data was popped from queues before the interrupt was detected and would
        # otherwise be lost; injecting it here ensures the first conscious tick sees it.
        pending = getattr(self, '_pending_interrupt_data', None)
        if pending:
            for name, items in pending.items():
                if items:
                    sensors_data.setdefault(name, [])
                    sensors_data[name] = items + sensors_data[name]
            self._pending_interrupt_data = None

        # Replay sensor payloads that were buffered by StreamFactory before
        # creating a new stream.  Without this, the stream that was shaped
        # around this data would never see it because pop_all() already
        # consumed it ticks ago.
        factory_replay = getattr(self, '_pending_factory_replay', None)
        if factory_replay:
            for name, items in factory_replay.items():
                if items:
                    sensors_data.setdefault(name, [])
                    sensors_data[name] = items + sensors_data[name]
            self._pending_factory_replay = None

        context = {
            'sensors_data': sensors_data,
            'streams': self.streams,
            'memory': self.memory,
            'stm': self.stm,
            'current_conscious': getattr(self, '_current_conscious', None),
            'adenosine': self.adenosine.level,
            'tick_counter': getattr(self, '_tick_counter', 0),
            'actuators': self.actuators,
            # Self-reflection snapshot from the previous tick — always one tick
            # stale but practically "current".  UserChatStream uses this to build
            # a rich system_state string for the LLM prompt.
            'self_reflection_state': getattr(self, '_self_reflection_snapshot', None),
        }

        # Run subconscious streams.
        # HLD: all streams execute every tick; "conscious" is a focus/priority
        # designation, not an exclusive execution gate.  The attention stream
        # decides which one is the focused (conscious) stream, but the others
        # still run as background processes.
        current_conscious = getattr(self, '_current_conscious', None)
        subconscious_results = []
        _special_names = {'attention_stream', 'alignment_stream',
                          'stream_factory', 'adenosine_stream', 'stm_update'}
        for stream in self.streams:
            if stream is current_conscious:
                continue
            # Skip special streams that run separately in their own loop below
            if stream.name in _special_names:
                continue
            try:
                result = stream.execute(context)
                if result:
                    subconscious_results.append(result)
                # Record subconscious stream activity for sleep replay so
                # insights from non-conscious streams are not lost.
                # Housekeeping streams in _REPLAY_SKIP_LLM are excluded
                # at replay time, so logging them here is harmless but
                # we skip them anyway to keep the log focused.
                if stream.name not in _REPLAY_SKIP_LLM:
                    self.add_to_stream_log(stream, result)
            except StopIteration:
                log.info("Stream %s stopped at checkpoint", stream.name)
            except Exception as exc:
                log.error("Subconscious stream %s error: %s", stream.name, exc)

        # Select conscious stream if none assigned (must happen before executing it)
        if current_conscious is None and self.streams:
            candidates = [s for s in self.streams
                         if getattr(s, '_can_be_conscious', True)]
            if candidates:
                candidates.sort(key=lambda s: -getattr(s, 'priority', 1))
                current_conscious = candidates[0]
                self._current_conscious = current_conscious
                self._was_conscious_streams.add(current_conscious.name)
                log.info("Selected %s as conscious stream", current_conscious.name)

        # Run conscious stream BEFORE special streams so that chat/task responses
        # are sent immediately without being blocked by the alignment LLM call.
        result = None
        if current_conscious:
            current_conscious._last_conscious_tick = getattr(self, '_tick_counter', 0)
            try:
                result = current_conscious.execute(context)
                self.add_to_stream_log(current_conscious, result)
                # The stream may have retired itself (removed from self.streams)
                # during execute().  Clear the conscious pointer so the next tick
                # picks a fresh candidate instead of holding a dead reference.
                if current_conscious not in self.streams:
                    log.info("Conscious stream %s retired itself", current_conscious.name)
                    current_conscious.is_conscious = False
                    self._current_conscious = None
            except StopIteration:
                log.info("Conscious stream %s stopped at checkpoint",
                    current_conscious.name)
                self._current_conscious = None
            except Exception as exc:
                log.error("Conscious stream %s error: %s", current_conscious.name, exc)

        # Run special subconscious streams in a fixed order so that factory
        # creates streams BEFORE attention_stream decides which to promote.
        # alignment → stm_update → factory → adenosine → attention
        _special_order = ['alignment_stream', 'stm_update', 'stream_factory',
                          'adenosine_stream', 'attention_stream']
        _special_map = {s.name: s for s in self.streams if s.name in _special_order}
        for sp_name in _special_order:
            stream = _special_map.get(sp_name)
            if stream is None:
                continue
            try:
                result_sp = stream.execute(context)
                if result_sp:
                    subconscious_results.append(result_sp)
            except Exception as exc:
                log.error("Special stream %s error: %s", sp_name, exc)

        # Check attention stream result for stream swap (from subconscious_results)
        attention_result = None
        for r in subconscious_results:
            if isinstance(r, dict) and 'promote' in r:
                attention_result = r
                break
        
        if attention_result and attention_result.get('promote'):
            old_conscious = self._current_conscious
            if old_conscious:
                old_conscious.is_conscious = False  # Properly demote
                old_conscious._last_conscious_tick = getattr(self, '_tick_counter', 0)
            self._current_conscious = attention_result['promote']
            self._current_conscious.is_conscious = True
            self._current_conscious._last_conscious_tick = getattr(self, '_tick_counter', 0)
            self._was_conscious_streams.add(self._current_conscious.name)
            log.info("Attention: promoting %s (demoting %s)",
                    self._current_conscious.name,
                    old_conscious.name if old_conscious else "none")
            # HLD: adenosine depletes on changing consciousness focus.
            if hasattr(self, 'adenosine'):
                self.adenosine.drain_activity("consciousness_switch")

        # Periodic state save — covers crash during awake ticks so the goal
        # coverage registry is not lost.  Every 100 ticks ≈ ~2 minutes.
        _awake_ticks = getattr(self, '_awake_tick_count', 0) + 1
        self._awake_tick_count = _awake_ticks
        if _awake_ticks % 100 == 0:
            self._save_iyye_state()

        # Check for wind-down trigger from adenosine.
        # Enforce a minimum of 10 ticks awake so a freshly woken brain always has
        # time to finish the current reply exchange before sleeping again.
        _MIN_AWAKE_TICKS = 10
        if self.adenosine.is_depleted() and _awake_ticks >= _MIN_AWAKE_TICKS:
            log.info("Adenosine depleted - initiating wind-down (awake %d ticks)", _awake_ticks)
            self.state = MindState.WINDING_DOWN
            self.winding_down_started = False

        return [result] if result is not None else subconscious_results

    def _check_system_state(self) -> Dict[str, Any]:
        """
        HLD: "It starts with checking the Iyye system state: inputs, actuators, 
        memory, hardware it runs on, etc."
        """
        import psutil
    
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'sensors': {
                name: {
                    'queue_size': len(q),
                    'maxlen': q.maxlen if hasattr(q, 'maxlen') else None,
                }
                for name, q in self.sensors.items()
            },
            'actuators': list(self.actuators.keys()),
            'memory_facts': self.memory.count(),
            'active_streams': len(self.streams),
            'conscious_stream': getattr(self._current_conscious, 'name', None) 
                                if hasattr(self, '_current_conscious') else None,
            'hardware': {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
            },
            'adenosine': self.adenosine.level,
        }
    
        log.info("System state check: %d sensors, %d actuators, %d streams, "
                "CPU=%.1f%%, Mem=%.1f%%",
                len(self.sensors), len(self.actuators), len(self.streams),
                state['hardware']['cpu_percent'], state['hardware']['memory_percent'])

        self._write_system_description(state)
        return state

    def _write_system_description(self, state: Dict[str, Any]) -> None:
        """
        HLD: "After checking the system, markdown file is created that describes
        the system to be used by awake execution streams."
        """
        hw = state['hardware']
        ts = state['timestamp']

        sensors_md = "\n".join(
            f"- **{name}**: queue {info['queue_size']} items"
            for name, info in state['sensors'].items()
        ) or "_(none)_"

        actuators_md = "\n".join(
            f"- **{name}**" for name in state['actuators']
        ) or "_(none)_"

        current_conscious_name = getattr(self._current_conscious, 'name', None) \
            if hasattr(self, '_current_conscious') else None
        streams_md = "\n".join(
            f"- **{s.name}** (priority={s.priority},"
            f" can_be_conscious={getattr(s, '_can_be_conscious', False)},"
            f" is_conscious={s.name == current_conscious_name})"
            for s in self.streams
        ) or "_(none)_"

        conscious_name = state.get('conscious_stream') or "_(none)_"

        lines = [
            "# Iyye System Description",
            f"_Generated: {ts} UTC_",
            "",
            "## Hardware",
            f"| Resource | Usage |",
            f"|----------|-------|",
            f"| CPU      | {hw['cpu_percent']:.1f}% |",
            f"| Memory   | {hw['memory_percent']:.1f}% |",
            f"| Disk     | {hw['disk_percent']:.1f}% |",
            "",
            "## Sensors",
            sensors_md,
            "",
            "## Actuators",
            actuators_md,
            "",
            "## Execution Streams",
            f"Active: {state['active_streams']}  |  Conscious: {conscious_name}",
            "",
            streams_md,
            "",
            "## Long-term Memory",
            f"Stored facts: {state['memory_facts']}",
            "",
            "## Adenosine",
            f"Level: {state['adenosine']:.3f} / {self.adenosine.MAX:.1f}",
            "",
        ]

        md_path = PROJECT_ROOT / "system_description.md"
        try:
            md_path.write_text("\n".join(lines), encoding="utf-8")
            log.info("System description written to %s", md_path)
        except Exception as exc:
            log.warning("Failed to write system description: %s", exc)

    def add_to_stream_log(self, stream: ProcessingStream, result: Any) -> None:
        """Record stream activity for later replay during sleep.

        Called for conscious and eligible subconscious streams so that
        the dreaming phase can extract facts from all meaningful work,
        not just the focused stream.
        """
        if not hasattr(self, 'last_conscious_log'):
            self.last_conscious_log = []

        # Use the stream's full activity_log (human-readable, includes all user
        # messages and responses).  We track how many entries we have already
        # snapshotted for this stream so each tick only appends new lines rather
        # than re-capturing a fixed tail — ensuring no entries are ever dropped.
        seen_key = f"_stream_log_seen_{stream.name}"
        seen = getattr(self, seen_key, 0)
        activity = getattr(stream, 'activity_log', [])
        new_entries = activity[seen:]
        setattr(self, seen_key, len(activity))

        if new_entries:
            text = '\n'.join(new_entries)
        elif result is not None:
            text = str(result)[:500]
        else:
            return  # nothing new to record

        self.last_conscious_log.append({
            'stream': stream.name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'result': text,
            'adenosine': self.adenosine.level,
            'alignment_scores': getattr(stream, 'alignment_scores', {}),
        })

    def _winding_down_actions(self) -> None:
        """
        HLD: "In winding down state, actuators processing stop when it is safe,
        conscious processing stream stops, all subconscious streams pause,
        and system is transferred to asleep state."
        """
        log.info("Winding down...")

        if not getattr(self, 'winding_down_started', False):
            self.winding_down_started = True

            # HLD: "actuators processing stops when it is safe"
            unsafe_actuators = []
            for name, actuator in self.actuators.items():
                if hasattr(actuator, 'is_safe_to_stop') and not actuator.is_safe_to_stop():
                    unsafe_actuators.append(name)
                    log.info("Actuator %s not safe to stop yet", name)

            if unsafe_actuators:
                return

            # HLD: "conscious processing stream stops" — request an explicit
            # stop so it can abort in-progress work at its next checkpoint.
            # HLD: "all subconscious streams pause" — subconscious streams are
            # paused implicitly: the state machine stops calling execute() once
            # the brain leaves AWAKE.  Their state (activity logs, alignment
            # scores, cursors) is preserved across the sleep cycle.
            if self._current_conscious is not None:
                self._current_conscious.request_stop()
                log.info("Requested conscious stream %s to stop",
                         self._current_conscious.name)

        # Wait for conscious stream to finish any in-progress work.
        current_conscious = self._current_conscious
        if current_conscious is not None:
            if not current_conscious.can_stop_safely():
                log.debug("Waiting for conscious stream to reach safe checkpoint...")
                return
            current_conscious.is_conscious = False
            self._current_conscious = None

        self.winding_down_started = False
        # HLD odd req: push "starting sleep" while actuators are still alive.
        self._push_to_web_chat("starting sleep")
        self._stop_actuators()
        self._enter_asleep()
 
    def _stop_actuators(self) -> None:
        """Gracefully stop all actuators.  HLD: 'no actuators are running' during sleep."""
        for name, actuator in self.actuators.items():
            try:
                if hasattr(actuator, 'graceful_stop'):
                    actuator.graceful_stop()
                elif hasattr(actuator, 'stop'):
                    actuator.stop()
                log.debug("Stopped actuator: %s", name)
            except Exception as e:
                log.error("Error stopping actuator %s: %s", name, e)

    def _restart_actuators(self) -> None:
        """Restart actuators that were stopped during sleep.

        Calls initialize() for MCP-based actuators (which clear
        _initialized in graceful_stop), then start() if available.
        """
        for name, actuator in self.actuators.items():
            try:
                if callable(getattr(actuator, 'initialize', None)):
                    actuator.initialize()
                if callable(getattr(actuator, 'start', None)):
                    actuator.start()
                log.debug("Restarted actuator: %s", name)
            except Exception as e:
                log.error("Error restarting actuator %s: %s", name, e)

    # --------------------------------------------------------------- #
    # State transition helpers
    # --------------------------------------------------------------- #
    def _enter_awake(self) -> None:
        # Iyye day counts completed sleep-wake cycles (natural wakeups only).
        # Interrupted wakeups (caused by a sensor message during sleep) are
        # brief and do not constitute a new "day".
        if not getattr(self, '_waking_interrupted', False):
            self.iyye_day += 1
            self._save_iyye_state()
        self._awake_tick_count = 0
        self._was_conscious_streams: set = set()
        log.info("Transition → AWAKE (Iyye day %d, interrupted=%s)",
                 self.iyye_day, self._waking_interrupted)
        self.state = MindState.AWAKE

    def _enter_asleep(self) -> None:
        """Return to ASLEEP state and prepare streams for the next wake cycle."""
        self._save_iyye_state()  # persist registry before sleeping
        REFILL_RATE = 0.05
        if hasattr(self.adenosine, "level"):
            self.adenosine.level = min(
                self.adenosine.MAX,
                self.adenosine.level + 5 * REFILL_RATE,
            )

        # Clear the conscious stream's stop flag (set during WINDING_DOWN) and
        # any self-set stop flags from the awake cycle.  Subconscious streams
        # were never stopped — only paused by not being executed.
        for stream in self.streams:
            stream._stop_requested = False
        # Reset so _start_subconscious_streams runs again on next WAKING_UP
        # (creates only missing streams; existing ones keep their state).
        self._subconscious_started = False
        # Reset sleep-phase flags so each new sleep cycle runs all phases.
        # Do NOT reset _replay_cursor — replay resumes where it left off so that
        # a restart or an interrupt wakeup doesn't throw away already-done work.
        self._sleep_did_system_check = False
        self._sleep_did_stm_flush = False
        self._sleep_did_replay = False
        self._sleep_prewarm_sent = False

        # If we're re-entering sleep after an interrupted wakeup that added
        # new awake-cycle entries, the STM fact snapshot from the previous
        # replay is stale — it was taken before the awake cycle created new
        # facts.  Delete the snapshot so _replay_batch rebuilds it from the
        # current STM.  Keep _replay_processed_stm so already-promoted facts
        # are not re-promoted.  _replay_stm_cursor resets to 0 since the
        # new sorted list has a different shape.
        if hasattr(self, '_replay_stm_sorted'):
            del self._replay_stm_sorted
            self._replay_stm_cursor = 0

        # Flush Theory-of-Mind contacts so interactions posted in the last
        # tick (after ToM's final execute()) are not lost.
        tom = getattr(self, '_tom_stream', None)
        if tom is not None and callable(getattr(tom, 'flush', None)):
            tom.flush()

        # Persist the awake cycle log before the replay phase clears it,
        # so the dreaming/replay phase works correctly after a restart.
        self._save_last_cycle()

        # "starting sleep" was already pushed by _winding_down_actions before
        # actuators were stopped.
        log.info("Sleeping… Adenosine refilled to %.3f", self.adenosine.level)
        self.state = MindState.ASLEEP

    # --------------------------------------------------------------- #
    # Actuator handling helpers
    # --------------------------------------------------------------- #

    def _actuate(self, payload: str) -> None:
        """
        Forward *payload* to **all** registered actuators.

        A failure in one actuator does not stop the others – it is merely logged.
        """
        for name, act in self.actuators.items():
            try:
                act.actuate(payload)
            except Exception as exc:                     # pragma: no‑cover
                log.error("Actuator %s failed: %s", name, exc)

    def _push_to_web_chat(self, text: str) -> None:
        """Send *text* to the web-chat actuator unconditionally (used for system status messages)."""
        for name, act in self.actuators.items():
            if 'web' in name.lower() or 'chat' in name.lower():
                try:
                    act.actuate(text)
                except Exception as exc:
                    log.warning("_push_to_web_chat via %s failed: %s", name, exc)
                return

    def _debug_to_actuators(self, text: str) -> None:
        """
        Unified debug output – sends *text* to every actuator (console,
        web‑chat UI …).  Emitted only when ``_chat_debug_enabled`` is True.
        """
        if not getattr(self, "_chat_debug_enabled", False):
            return
        self._actuate(text)

    # --------------------------------------------------------------- #
    # Inter-stream mailbox
    # --------------------------------------------------------------- #
    def post_message(self, target: str, message: Dict[str, Any]) -> None:
        """Post *message* to the mailbox of the stream named *target*."""
        self._mailboxes.setdefault(target, []).append(message)

    def drain_messages(self, target: str) -> List[Dict[str, Any]]:
        """Return and clear all pending messages for *target*."""
        return self._mailboxes.pop(target, [])

    # --------------------------------------------------------------- #
    # Graceful shutdown
    # --------------------------------------------------------------- #
    def shutdown(self) -> None:
        log.info("Shutting down IyyeBrain")
        # Flush Theory-of-Mind contacts before anything else — catches
        # interactions posted after the last ToM execute() or during a
        # crash/exit that skipped the normal WINDING_DOWN path.
        tom = getattr(self, '_tom_stream', None)
        if tom is not None and callable(getattr(tom, 'flush', None)):
            tom.flush()
        # Persist the awake-cycle log so the next sleep phase can replay it,
        # even when the process exits via timeout or KeyboardInterrupt rather
        # than going through the full WINDING_DOWN → ASLEEP sequence.
        self._save_last_cycle()
        self._save_iyye_state()
        # Stop background-thread sensors so their executor threads don't keep
        # the process alive after the main loop exits.
        for name, q in self.sensors.items():
            if callable(getattr(q, 'stop_collection', None)):
                try:
                    q.stop_collection()
                    log.debug("Stopped background sensor: %s", name)
                except Exception as exc:
                    log.warning("Error stopping sensor %s: %s", name, exc)
        try:
            self.memory.close()
        except Exception:  # pragma: no‑cover – defensive
            pass

    def _check_wakeup_triggers(self, sensors_data: Dict[str, List[Any]]) -> bool:
        """
        HLD: "Each important input source has a simple in-sleep processing routine 
        that checks if latest input item should force wakeup."
        """
        # Web chat messages always trigger wakeup
        if "web_chat" in sensors_data and sensors_data["web_chat"]:
            for msg in sensors_data["web_chat"]:
                # Check for urgent keywords
                if isinstance(msg, str):
                    msg_lower = msg.lower()
                    urgent_keywords = ['urgent', 'emergency', 'help', 'important', 'wake']
                    if any(kw in msg_lower for kw in urgent_keywords):
                        self._wakeup_reason = f"urgent web_chat message: {msg[:50]}"
                        log.info("Wakeup triggered by urgent web_chat input")
                        return True
            # Non-urgent messages still trigger wakeup but with lower priority
            log.info("Wakeup triggered by web_chat input")
            return True

        # Check microphone sensor for wake words
        if "microphone_sensor" in sensors_data and sensors_data["microphone_sensor"]:
            for transcription in sensors_data["microphone_sensor"]:
                if isinstance(transcription, dict):
                    text = transcription.get("text", "").lower()
                    # Check for wake words
                    wake_words = ['iyye', 'hey iyye', 'wake up']
                    if any(ww in text for ww in wake_words):
                        self._wakeup_reason = f"wake word detected: {text[:50]}"
                        log.info("Wakeup triggered by wake word")
                        return True
    
        # Check hardware sensor for critical conditions.
        for key in sensors_data:
            if 'hardware' in key.lower():
                for reading in sensors_data[key]:
                    if not isinstance(reading, dict):
                        continue
                    cpu = reading.get("cpu_percent", reading.get("usage_cpu_percent", 0))
                    mem = reading.get("memory_percent", reading.get("mem_usage", 0))
                    if cpu > 95:
                        self._wakeup_reason = "critical CPU usage"
                        log.info("Wakeup triggered by hardware alert (CPU %.1f%%)", cpu)
                        return True
                    if mem > 95:
                        self._wakeup_reason = "critical memory usage"
                        log.info("Wakeup triggered by hardware alert (mem %.1f%%)", mem)
                        return True
    
        # Telegram messages trigger wakeup on any real message.
        # New direct sensor format: each item is a message dict with 'update_id'.
        # Legacy MCP batch format: each item is {'count': N, 'messages': [...]}.
        for key in sensors_data:
            if 'telegram' not in key.lower():
                continue
            for item in sensors_data[key]:
                if isinstance(item, dict):
                    if (item.get('count', 0) > 0 or item.get('messages')
                            or 'update_id' in item):
                        self._wakeup_reason = f"telegram message on {key}"
                        log.info("Wakeup triggered by Telegram input (%s)", key)
                        return True
                elif item:  # plain non-empty string / other truthy value
                    self._wakeup_reason = f"telegram message on {key}"
                    log.info("Wakeup triggered by Telegram input (%s)", key)
                    return True

        # Don't check git sensor for code changes, only self can write to git

        return False

    def _is_high_priority(self, sensor_name: str, data: List[Any]) -> bool:
        """Determine if sensor data warrants immediate wakeup."""
        # Stub - implement based on sensor type and content analysis
        if "hardware" in sensor_name:
            # Hardware alerts might need attention
            for item in data:
                if isinstance(item, dict):
                    if item.get("usage_cpu_percent", 0) > 90:
                        return True
        return False

    def _asleep_actions(self, sensors_data: Dict[str, List[Any]]) -> None:
        """
        HLD: "Asleep state runs only 'housekeeping' subconscious tasks and
        checks some sensor inputs."
        """
        # Replenish adenosine first so the level is current for all decisions this tick.
        # HLD: "fully replenished adenosine, or partially/proportionally replenished
        # if waking up because of high priority input interruption."
        self.adenosine.replenish(0.05)

        # Re-inject any sensor data that was accumulated during WINDING_DOWN.
        # Those payloads were already popped from queues (and acknowledged by the
        # MCP server), so we stash them for the first awake tick — but do NOT
        # let them trigger a wakeup interrupt.  They were already known about
        # before sleep began; treating them as "new" would cause every sleep
        # cycle to be interrupted (day counter never advances).
        saved = getattr(self, '_winding_down_sensors', None)
        if saved:
            self._pending_interrupt_data = saved
            self._winding_down_sensors = {}

        # Phase 1: system check — runs exactly once per sleep cycle.
        # Must run BEFORE the wakeup-trigger check so that system_description.md
        # is always current when the first awake stream executes, even on an
        # interrupt wakeup that skips the rest of the sleep phases.
        if not self._sleep_did_system_check:
            self._check_system_state()
            self._sleep_did_system_check = True
            # Do not return here — fall through to the wakeup-trigger check so
            # that an interrupt arriving on the very same tick is not delayed by
            # one extra sleep tick.

        # Phase 1b: flush StmUpdateStream so chat/telegram entries that were
        # buffered but not yet extracted (due to LLM_INTERVAL throttle) are
        # written to STM before replay promotes facts to LTM.  Without this,
        # a wind-down that arrives mid-throttle loses user facts entirely —
        # replay skips chat stream text and relies on STM facts existing.
        if not self._sleep_did_stm_flush:
            stm_stream = next(
                (s for s in self.streams if s.name == 'stm_update'), None,
            )
            if stm_stream is not None and callable(getattr(stm_stream, 'flush', None)):
                stm_stream.flush()
            self._sleep_did_stm_flush = True

        # HLD: "Each important input source has a simple in-sleep processing routine
        # that checks if latest input item should force wakeup."
        if self._check_wakeup_triggers(sensors_data):
            # Adenosine is at whatever partial level sleep has replenished so far —
            # proportional to time slept, as required by the HLD.
            # Preserve the triggering sensor data: it was already popped from the
            # queues by pop_all() and will be gone on the next tick.  Stash it so
            # _awake_actions() can inject it into the first conscious tick.
            # Merge with any data already stashed from WINDING_DOWN.
            existing = getattr(self, '_pending_interrupt_data', None) or {}
            for _n, _items in sensors_data.items():
                existing.setdefault(_n, [])
                existing[_n].extend(_items)
            self._pending_interrupt_data = existing
            self._enter_waking_up(interrupted=True)
            return

        # Prewarm: request the stm/fast model before replay starts so
        # fact extraction has a dedicated fast model available.
        if not self._sleep_did_replay and not getattr(self, '_sleep_prewarm_sent', False):
            self._sleep_prewarm_sent = True
            router = getattr(self, 'llm_router', None)
            if router is not None:
                hp = router._healthy_ports
                stm_model = router._find_model("stm")
                if stm_model and hp is not None and stm_model["port"] not in hp:
                    self.post_message("llm_management", {
                        "action": "ensure_role",
                        "role": "stm",
                        "model_name": stm_model["name"],
                        "task": {
                            "prompt_tokens": 800,
                            "expected_output_tokens": 200,
                            "quality_need": 0.3,
                            "latency_budget_s": 15,
                            "urgency": 0.6,
                        },
                        "reason": "prewarm for sleep replay fact extraction",
                    })

        # Phase 2: replay ("dreaming") — HLD: skipped on the very first sleep of
        # this process run.  Spreads LLM extraction over multiple sleep ticks.
        if self._is_first_sleep:
            self._sleep_did_replay = True   # mark done so we fall through to wakeup
            self._is_first_sleep = False    # only skip once per process lifetime

        _REPLAY_BATCH = 3  # LLM calls per sleep tick
        if not self._sleep_did_replay:
            replay_log = getattr(self, 'last_conscious_log', []) or []
            cursor = getattr(self, '_replay_cursor', 0)
            if replay_log and cursor < len(replay_log):
                batch = replay_log[cursor: cursor + _REPLAY_BATCH]
                self._replay_batch(batch)
                self._replay_cursor = cursor + len(batch)
                if self._replay_cursor >= len(replay_log):
                    # Final batch — clean up and mark replay done.
                    self._replay_finish()
                    self._sleep_did_replay = True
                    self._replay_cursor = 0
                    # Durably clear the cycle file so a restart doesn't
                    # re-replay the same (now empty) cycle.
                    self._save_last_cycle()
            else:
                # No log or already exhausted.
                self._sleep_did_replay = True
                self._replay_cursor = 0
            return  # give this tick to replay

        # HLD: "keeps deleting the processed part of short term memory from last
        # awake cycle".  Trim both in-memory queues and on-disk STM files.
        for name, q in self.sensors.items():
            if len(q) > 1000:
                while len(q) > 500:
                    q.popleft()
                log.debug("Trimmed sensor queue: %s", name)
        self._cleanup_stm_files(keep_days=3)

        # Both phases complete — fill adenosine to MAX and wake naturally.
        # Phase 1 (system check) always runs; Phase 2 (dreaming/replay) runs only when
        # there is a previous-day log.  First sleep skips Phase 2 per HLD.
        log.info("Sleep phases complete — adenosine filled, natural wakeup")
        self.adenosine.level = self.adenosine.MAX
        self._enter_waking_up(interrupted=False)

    def _enter_waking_up(self, interrupted: bool = False) -> None:
        """
        HLD: "Waking up state starts all subcon execution streams, selects
        conscious stream..."

        Args:
            interrupted: True if waking due to high-priority input,
                        False if natural wakeup after rest.
        """
        # Restart actuators that were stopped during sleep so the "waking up"
        # message (and any subsequent actuator use) actually reaches the user.
        self._restart_actuators()

        # HLD: "Before transferring to 'waking up', it pushes a message 'waking up day %day'"
        # iyye_day is incremented in _enter_awake() for natural wakeups; preview the next value.
        next_day = self.iyye_day + (0 if interrupted else 1)
        self._push_to_web_chat(f"waking up day {next_day}")
        log.info("Transition → WAKING_UP (interrupted=%s)", interrupted)
        self.state = MindState.WAKING_UP
        self._waking_interrupted = interrupted
        self._waking_up_tick = 0

        # _start_subconscious_streams() is called in _waking_up_actions()

    def _cleanup_stm_files(self, keep_days: int = 3) -> None:
        """
        HLD: "keeps deleting the processed part of short term memory from last
        awake cycle (can also compact)."

        Deletes day-log files older than `keep_days` from both io_history/ and
        streams_history/.  Files are named YYYY-MM-DD.txt; any file whose stem
        sorts below the cutoff date string is removed.
        """
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=keep_days)).isoformat()
        deleted = 0
        for root_name in ("io_history", "streams_history"):
            root = PROJECT_ROOT / root_name
            if not root.exists():
                continue
            for subdir in root.iterdir():
                if not subdir.is_dir():
                    continue
                for fpath in list(subdir.iterdir()):
                    if fpath.suffix == ".txt" and fpath.stem < cutoff:
                        try:
                            fpath.unlink()
                            deleted += 1
                            log.debug("Deleted old STM file: %s", fpath)
                        except Exception as exc:
                            log.warning("Could not delete STM file %s: %s", fpath, exc)
        if deleted:
            log.info("STM cleanup: deleted %d file(s) older than %s", deleted, cutoff)

    def _fine_tune_dnns(self, discovered_facts: List[Dict[str, Any]]) -> None:
        """
        HLD: "Also uses the discovered data for light fine-tuning of relevant DNNs weights"
        
        FIXME: Implement actual fine-tuning when DNN models are available.
        Currently stores training metadata for future implementation.
        """
        if not discovered_facts:
            return
    
        log.info("DNN fine-tuning: processing %d discovered facts", len(discovered_facts))
    
        # Group facts by type for targeted fine-tuning
        facts_by_type = {}
        for fact in discovered_facts:
            metadata = fact.get('metadata', {})
            fact_type = metadata.get('type', 'general')
            if fact_type not in facts_by_type:
                facts_by_type[fact_type] = []
            facts_by_type[fact_type].append(fact)
        
        # TODO: Implement actual fine-tuning using LoRA/QLoRA when models available

    def _replay_batch(self, entries: List[Dict[str, Any]]) -> None:
        """
        Process one batch of stream-log entries during sleep replay.
        Called from _asleep_actions a few entries at a time to avoid blocking
        the main loop for the full duration of an LLM-heavy replay.

        HLD: "reads temporally associated facts from short term memory" —
        STM facts are sorted by timestamp and matched to log entries by
        chronological proximity (not by provenance string).

        HLD: "keeps deleting the processed part of short term memory from
        last awake cycle" — promoted STM facts are removed at the end of
        every batch (not deferred to _replay_finish) so an interrupted
        replay never re-promotes facts that were already committed to LTM.
        """
        stm = getattr(self, 'stm', None)

        # Build (or reuse) the sorted STM fact list for this replay cycle.
        # After an interrupted wakeup the snapshot is deleted (but
        # _replay_processed_stm is kept) so we rebuild with fresh STM
        # facts while still skipping already-promoted ones.
        if not hasattr(self, '_replay_stm_sorted'):
            stm_facts = stm.get_all_today() if stm is not None else []
            self._replay_stm_sorted: List[Dict[str, Any]] = sorted(
                stm_facts, key=lambda f: f.get('timestamp', ''),
            )
            self._replay_stm_cursor: int = 0
            if not hasattr(self, '_replay_processed_stm'):
                self._replay_processed_stm: set = set()
            if not hasattr(self, '_replay_discovered'):
                self._replay_discovered: List[Dict[str, Any]] = []
            log.info(
                "Sleep replay %s: %d total entries, %d STM facts, %d already promoted",
                "resuming" if self._replay_processed_stm else "starting",
                len(getattr(self, 'last_conscious_log', [])),
                len(stm_facts),
                len(self._replay_processed_stm),
            )

        sorted_facts = self._replay_stm_sorted
        cursor = self._replay_stm_cursor
        processed_stm_ids = self._replay_processed_stm
        discovered_facts = self._replay_discovered

        # IDs promoted in THIS batch — deleted from STM at the end.
        batch_promoted_ids: List[str] = []

        for entry in entries:
            stream_name = entry.get('stream', 'unknown')
            entry_ts = entry.get('timestamp', '')

            # 1. Promote temporally associated STM facts → LTM.
            # Advance the cursor through facts whose timestamps are ≤ this
            # entry's timestamp, pairing them with the closest log entry.
            while cursor < len(sorted_facts):
                fact = sorted_facts[cursor]
                if fact.get('timestamp', '') > entry_ts:
                    break
                cursor += 1
                if fact['id'] not in processed_stm_ids:
                    fact_id = self._promote_stm_to_ltm(fact, entry)
                    if fact_id:
                        processed_stm_ids.add(fact['id'])
                        batch_promoted_ids.append(fact['id'])
                        discovered_facts.append({'id': fact_id, 'text': fact['text']})

            # 2. LLM fact extraction — skip for chat/telegram (user messages are
            #    not useful LTM material) and for subconscious bookkeeping streams
            #    that log only housekeeping output.
            sn_lower = stream_name.lower()
            skip_llm = (
                any(kw in sn_lower for kw in ('chat', 'telegram'))
                or stream_name in _REPLAY_SKIP_LLM
                or any(sn_lower.startswith(p) for p in _REPLAY_SKIP_PREFIXES)
                or any(kw in sn_lower for kw in _REPLAY_SKIP_KEYWORDS)
            )
            text = entry.get('result', '')
            # Also skip if ALL lines are metric snapshots — saves an LLM call
            # for the common case where self_reflection logged only "[Day N] CPU=…".
            if text and not skip_llm:
                meaningful_lines = [
                    ln for ln in text.splitlines()
                    if ln.strip() and not _EPHEMERAL_METRIC_RE.search(ln)
                ]
                if not meaningful_lines:
                    skip_llm = True
                else:
                    text = '\n'.join(meaningful_lines)
            if text and not skip_llm:
                key_facts = self._extract_key_facts(text, stream_name=stream_name)
                for fact in key_facts:
                    if _EPHEMERAL_METRIC_RE.search(fact):
                        continue  # never store transient metric snapshots in LTM
                    if _LTM_NOISE_RE.search(fact):
                        continue  # skip placeholders and system-status sentences
                    stored_id = self.memory.store_fact(
                        text=fact,
                        confidence=0.7,
                        source=stream_name,
                        provenance=(
                            f"Extracted during sleep replay from '{stream_name}'"
                            f" at {entry.get('timestamp')}"
                        ),
                        time_frame='permanent',
                    )
                    discovered_facts.append({'id': stored_id, 'text': fact})

        # Persist cursor position for the next batch.
        self._replay_stm_cursor = cursor

        # Progressive STM cleanup: delete facts promoted in this batch
        # immediately so an interrupt wakeup can't re-promote them.
        if stm is not None and batch_promoted_ids:
            stm.remove_by_ids(batch_promoted_ids)
            log.debug("Replay batch: promoted and removed %d STM fact(s)",
                      len(batch_promoted_ids))

    def _replay_finish(self) -> None:
        """
        Called after the final replay batch to promote remaining STM facts,
        clean up, and run HLD housekeeping (what-if, fine-tune).

        Most promoted STM facts were already deleted per-batch in
        _replay_batch.  This method handles the tail: STM facts whose
        timestamps fell after the last log entry.
        """
        stm = getattr(self, 'stm', None)
        processed_stm_ids: set = getattr(self, '_replay_processed_stm', set())
        discovered_facts: List[Dict[str, Any]] = getattr(self, '_replay_discovered', [])
        sorted_facts: List[Dict[str, Any]] = getattr(self, '_replay_stm_sorted', [])
        cursor: int = getattr(self, '_replay_stm_cursor', 0)

        # Promote any STM facts past the cursor (timestamps after last entry).
        tail_ids: List[str] = []
        for fact in sorted_facts[cursor:]:
            if fact['id'] not in processed_stm_ids:
                fact_id = self._promote_stm_to_ltm(fact, entry=None)
                if fact_id:
                    processed_stm_ids.add(fact['id'])
                    tail_ids.append(fact['id'])
                    discovered_facts.append({'id': fact_id, 'text': fact['text']})

        if stm is not None and tail_ids:
            stm.remove_by_ids(tail_ids)

        log.info(
            "Sleep replay done: promoted %d fact(s) to LTM total (%d in tail)",
            len(processed_stm_ids), len(tail_ids),
        )

        self._fine_tune_dnns(discovered_facts)
        self._run_what_if_simulations(self.last_conscious_log)
        self.last_conscious_log = []

        # Clean up per-replay scratch attributes.
        for attr in ('_replay_stm_sorted', '_replay_stm_cursor',
                     '_replay_processed_stm', '_replay_discovered'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    def _promote_stm_to_ltm(
        self,
        stm_fact: Dict[str, Any],
        entry: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Write a single STM fact into long-term memory, preserving all HLD tags.
        Returns the LTM fact id on success, None on failure.
        Ephemeral and session facts are never promoted (ephemeral = transient
        metrics; session = relevant only to the current wakeup cycle).
        Content matching noise patterns (LLM placeholders, system status,
        vague cognitive state) is also rejected.
        Facts originating from LLM-generated streams are also rejected —
        their internal bookkeeping is not durable knowledge about the world.
        """
        if stm_fact.get('time_frame') in ('ephemeral', 'session'):
            return None
        text = stm_fact.get('text', '')
        if _EPHEMERAL_METRIC_RE.search(text) or _LTM_NOISE_RE.search(text):
            return None
        # Reject facts from LLM-generated / planned streams.  Their output
        # is internal recommendations or operational noise, never durable
        # knowledge about the world — regardless of whether the attention
        # stream promoted one to consciousness.
        prov = stm_fact.get('provenance', '')
        if prov:
            prov_lower = prov.lower()
            if (any(prov_lower.startswith(p) for p in _REPLAY_SKIP_PREFIXES)
                    or any(kw in prov_lower for kw in _REPLAY_SKIP_KEYWORDS)):
                return None
        try:
            provenance = "Promoted from STM during sleep replay"
            if entry:
                provenance += (
                    f" (stream '{entry.get('stream')}'"
                    f" at {entry.get('timestamp')})"
                )
            return self.memory.store_fact(
                text=stm_fact['text'],
                confidence=float(stm_fact.get('confidence', 0.7)),
                source=stm_fact.get('provenance', 'agent'),
                provenance=provenance,
                time_frame=stm_fact.get('time_frame', 'permanent'),
                media_path=stm_fact.get('media_path'),
            )
        except Exception as exc:
            log.warning("Failed to promote STM fact to LTM: %s", exc)
            return None

    def _extract_key_facts(self, text: str, stream_name: str = "unknown") -> List[str]:
        """
        Extract key facts from conscious stream text using LLM.
        Falls back to heuristic extraction if the LLM is unreachable.
        """
        try:
            from llm_client import LLMClient
            client = LLMClient(no_think=True)
            response = client.complete_from_file(
                "extract_facts",
                stream_name=stream_name,
                stream_output=text,
            )
            facts = []
            for line in response.splitlines():
                line = line.strip()
                if not line:
                    continue
                # Skip lines that are clearly LLM reasoning artefacts
                if _LTM_NOISE_RE.search(line):
                    continue
                if _EPHEMERAL_METRIC_RE.search(line):
                    continue
                # Skip HTML / thinking tags
                if line.startswith('<') and '>' in line:
                    continue
                # Skip markdown headings and horizontal rules
                if line.startswith('#') or line == '---':
                    continue
                # Skip very short lines (likely fragments)
                if len(line) < 10:
                    continue
                facts.append(line)
            log.debug("LLM extracted %d facts from %s", len(facts), stream_name)
            return facts[:10]
        except Exception as exc:
            log.warning("LLM fact extraction failed, using heuristic fallback: %s", exc)

        # Heuristic fallback
        facts = []
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            s = sentence.strip()
            if 20 < len(s) < 200:
                if any(kw in s.lower() for kw in ['learned', 'discovered', 'found',
                                                   'determined', 'concluded', 'noted']):
                    facts.append(s)
        return facts[:5]

    def _run_what_if_simulations(self, conscious_log: List[Dict[str, Any]]) -> None:
        """
        Run counterfactual analysis on close-call decisions.
        HLD: "For decisions where there were close call of conflicting
        motivations, runs what-if simulations."
        """
        if not conscious_log:
            return

        # Find decisions with close alignment scores (conflicting motivations)
        close_decisions = []
        for entry in conscious_log:
            if 'alignment_scores' in entry and entry['alignment_scores']:
                scores = list(entry['alignment_scores'].values())
                if len(scores) > 1:
                    max_diff = max(scores) - min(scores)
                    if max_diff < 0.3:  # Close call threshold - conflicting goals
                        close_decisions.append(entry)
        
        if close_decisions:
            log.info("Running what-if simulations on %d close decisions", len(close_decisions))
            
            for decision in close_decisions[:5]:  # Limit to prevent overload
                self._simulate_alternative(decision)
                
    
    def _simulate_alternative(self, decision: Dict[str, Any]) -> None:
        """Simulate what would have happened with a different choice."""
        alignment_scores = decision.get('alignment_scores', {})
        if not alignment_scores:
            return
            
        # Find the highest and second highest alignment to simulate alternative
        sorted_goals = sorted(alignment_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_goals) < 2:
            return
            
        chosen_goal, chosen_score = sorted_goals[0]
        alternative_goal, alternative_score = sorted_goals[1]
        
        # TODO: Use LLM to generate detailed alternative outcome
        simulation_result = (
            f"Instead of '{chosen_goal}' (score={chosen_score:.2f}), "
            f"what if chose '{alternative_goal}' (score={alternative_score:.2f})? "
            f"Context: {decision.get('result', 'unknown')[:100]}"
        )

        log.debug("What-if simulation (not stored): %s", simulation_result[:120])
 
__all__ = ['BaseSensorQueue', 'BaseActuator', 'ProcessingStream', 
           'MindState', 'IyyeBrain', 'MemoryClient']
# --------------------------------------------------------------------------- #
# Start the optional Flask UI (non‑blocking)
# --------------------------------------------------------------------------- #
try:
    from web_chat_2 import start_web_chat   # local helper
except Exception as exc:                     # Defensive – if the module is missing.
    log.warning("Web chat UI not available: %s", exc)

if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--debug", action="store_true", help="Set log level to DEBUG")
    _args, _ = _parser.parse_known_args()
    if _args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Start the web-chat Flask UI *before* the brain so it is reachable
    # while the heavy IyyeBrain() constructor runs (LanceDB, embedder, etc.).
    try:
        start_web_chat()                     # type: ignore[name-defined]
        log.info("Web‑chat UI listening on http://127.0.0.1:5000/")
    except Exception as exc:  # pragma: no‑cover – UI optional
        log.debug("Web chat not started (%s)", exc)

    brain = IyyeBrain()

    try:
        # Run indefinitely by default.  Pass a plain integer as the first
        # CLI argument to cap the run duration (useful for tests/dev).
        _first_arg = sys.argv[1] if len(sys.argv) > 1 else ""
        run_seconds = int(_first_arg) if _first_arg.lstrip("-").isdigit() else None

        start = time.time()
        while run_seconds is None or (time.time() - start) < run_seconds:
            brain.run_once()
            time.sleep(1.0)
    except KeyboardInterrupt:   # pragma: no‑cover
        pass
    finally:
        # Explicitly run cleanup before the hard exit — atexit handlers do NOT
        # fire with _os._exit, so shutdown() (which saves state, stops MCP
        # subprocesses, and closes memory) must be called here.
        try:
            brain.shutdown()
        except Exception:
            pass
        # Daemon threads (e.g. alignment LLM thread) can hold Python-internal
        # locks during Py_FinalizeEx, causing a hang.  Exit immediately instead
        # of waiting for a clean teardown.
        import os as _os
        _os._exit(0)
