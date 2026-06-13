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
import threading
import multiprocessing as mp
from datetime import datetime, timezone
from enum import Enum, auto
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Callable, Optional, Tuple

from iyye_base import PROJECT_ROOT, BaseSensorQueue, BaseActuator, ProcessingStream
from memory_filters import (
    EPHEMERAL_METRIC_RE as _EPHEMERAL_METRIC_RE,
    SKIP_STREAM_NAMES as _REPLAY_SKIP_LLM,
    SKIP_STREAM_PREFIXES as _REPLAY_SKIP_PREFIXES,
    SKIP_STREAM_KEYWORDS as _REPLAY_SKIP_KEYWORDS,
)

# LTM-specific noise filter — kept here because it is only used during
# sleep replay, not by the STM pipeline.
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

# --------------------------------------------------------------------------- #
# Wind-down pause / settle constants
# --------------------------------------------------------------------------- #
# Per-stream join budget during settle.  Set low enough that a stuck thread
# (e.g. blocked LLM start) doesn't push sleep latency past acceptable.
_PAUSE_SETTLE_TIMEOUT_S = 5.0
# Total wall-clock cap across all streams' settle calls.  Wakeup health checks
# reconcile any state that didn't make it under the wire.
_PAUSE_SETTLE_TOTAL_S   = 15.0

# --------------------------------------------------------------------------- #
# In-sleep wakeup policy
# --------------------------------------------------------------------------- #
# HLD: Telegram users are untrusted by default.  By default only a *trusted*
# Telegram sender can force an urgent wakeup; messages from unknown senders
# are queued for the next natural wakeup instead of interrupting sleep.  Set
# IYYE_TELEGRAM_URGENT_WAKE=1 to additionally allow urgent-keyword messages
# from untrusted senders to wake the system (opt-in, looser policy).
_TELEGRAM_URGENT_WAKE = bool(os.getenv("IYYE_TELEGRAM_URGENT_WAKE", ""))
# Per-sensor cap on input buffered during sleep awaiting natural wakeup, so a
# flood of untrusted traffic cannot grow memory without bound.  Newest kept.
_DEFERRED_INPUT_CAP = 500

# --------------------------------------------------------------------------- #
# Inter-stream mailbox: pause-time delivery policy
# --------------------------------------------------------------------------- #
# Message actions that a *paused* recipient (winding-down) may still drain and
# act on immediately.  Empty by default: during wind-down all streams are
# paused and their handlers spawn background work (LLM starts, codegen) that
# pause explicitly forbids, so acting on such a message mid-pause would be a
# no-op that silently consumes it — better to defer to the next awake tick.
# A message can opt in regardless by setting ``"urgent": True``, but only do so
# when its handler is genuinely pause-safe (in-memory state, no new threads).
_URGENT_MAILBOX_ACTIONS: frozenset = frozenset()

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
# STM wrapper for LLM-generated streams
# --------------------------------------------------------------------------- #
class _SessionOnlySTM:
    """Thin proxy around ShortTermMemory that forces generated-stream facts
    to ``time_frame='session'`` with a fixed provenance prefix.

    LLM-generated codegen streams receive this wrapper instead of the raw
    STM, so they cannot write durable facts that later seed goal-suggestion
    loops via ``_find_goal_evidence``.  Read operations (search, get_recent)
    are delegated unchanged.
    """
    __slots__ = ('_stm', '_provenance')

    def __init__(self, stm, provenance: str) -> None:
        self._stm = stm
        self._provenance = provenance

    # -- constrained write --
    def add_fact(self, text, confidence=0.7, provenance=None,
                 time_frame=None, media_path=None):
        return self._stm.add_fact(
            text=text,
            confidence=confidence,
            provenance=self._provenance,
            time_frame='session',
            media_path=media_path,
        )

    # -- read-only delegates --
    def search(self, *a, **kw):        return self._stm.search(*a, **kw)
    def get_recent(self, *a, **kw):    return self._stm.get_recent(*a, **kw)
    def save_media(self, *a, **kw):    return self._stm.save_media(*a, **kw)


class _GraduatedSTM:
    """STM proxy for *graduated* generated streams.

    Unlike :class:`_SessionOnlySTM`, a graduated stream has proven useful over
    several awake cycles, so it is allowed to contribute durable knowledge:
    its facts keep a promotable ``time_frame`` (session/ephemeral are upgraded
    to ``recent`` so sleep replay doesn't discard them) and a
    ``gen_graduated:<name>`` provenance that ``_promote_stm_to_ltm`` accepts
    (it matches none of the LTM skip prefixes/keywords).  This is what lets a
    stable handler's learning actually accumulate in long-term memory.
    """
    __slots__ = ('_stm', '_provenance')

    # time_frames that would be filtered out before LTM — coerced upward so a
    # graduated stream's facts are eligible for promotion.
    _NON_DURABLE = frozenset({None, '', 'session', 'ephemeral'})

    def __init__(self, stm, provenance: str) -> None:
        self._stm = stm
        self._provenance = provenance

    def add_fact(self, text, confidence=0.7, provenance=None,
                 time_frame=None, media_path=None):
        tf = 'recent' if time_frame in self._NON_DURABLE else time_frame
        return self._stm.add_fact(
            text=text,
            confidence=confidence,
            provenance=self._provenance,
            time_frame=tf,
            media_path=media_path,
        )

    # -- read-only delegates --
    def search(self, *a, **kw):        return self._stm.search(*a, **kw)
    def get_recent(self, *a, **kw):    return self._stm.get_recent(*a, **kw)
    def save_media(self, *a, **kw):    return self._stm.save_media(*a, **kw)


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
        # Event journal — single ordered source of truth for the memory
        # pipeline.  Phase 1: written in shadow alongside the existing stores
        # (STM JSONL, streams_history, io_history, last_conscious_log); later
        # phases derive STM/replay from it.  Created before STM so STM can
        # emit stm_fact/stm_merge events into it.
        # ------------------------------------------------------------------- #
        from event_journal import EventJournal
        self.journal = EventJournal()
        self._journal_cycle: int = int(self._load_iyye_state().get("journal_cycle", 0))
        self.journal.start_cycle(self._journal_cycle)
        # Actuators are discovered before the journal exists; backfill the
        # handle so BaseActuator.actuate can shadow-record output (Phase 0).
        for _act in self.actuators.values():
            _act.journal = self.journal

        # ------------------------------------------------------------------- #
        # Short-term memory (structured fact store, in-memory + daily JSONL)
        # ------------------------------------------------------------------- #
        self.stm = ShortTermMemory()
        # Let STM mirror fact adds/merges into the journal (shadow), then make
        # the journal authoritative: recover any facts the JSONL cache lost
        # (Phase 3 — STM is a projection of the journal; JSONL is a cache).
        self.stm.journal = self.journal
        try:
            # STM is a fold over only the stm_* events — skip sensor_input etc.
            # so reconciliation doesn't materialize the whole partition (#9).
            self.stm.reconcile_with_journal(
                self.journal.read_cycle(
                    self._journal_cycle,
                    types=frozenset({'stm_fact', 'stm_merge', 'stm_remove'}),
                )
            )
        except Exception as exc:
            log.warning("STM journal reconciliation failed: %s", exc)

        # ------------------------------------------------------------------- #
        # Long term plans — durable across sleep cycles and restarts (HLD:
        # "Long term plans").  One shared store: PlannerStream drives it, chat
        # plan actions read it, and the in-sleep deadline check polls it.
        # ------------------------------------------------------------------- #
        from plans import PlanStore
        self.plan_store = PlanStore()
        # The plan deadline this brain last woke up for.  A given overdue
        # deadline interrupts sleep ONCE; until the store's earliest deadline
        # changes (the due step completed, or a different deadline became
        # due), later sleeps proceed normally.  Without this, an overdue
        # deadline that can't be cleared quickly (LLM down, abstract step)
        # re-fires the wakeup gate every asleep tick — before replay (order
        # 70) ever runs — starving dreaming, day advancement and the full
        # adenosine refill forever.
        self._plan_deadline_wake_fired: Optional[datetime] = None

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
        # Monotonic wake-cycle counter ("epoch").  Stamped onto every async LLM
        # job by the scheduler; results from a cycle that has since ended are
        # discarded on poll.  Bumped in _enter_waking_up.
        self._wake_epoch: int = 0
        # Streams that held consciousness during the current awake cycle.
        # Reset per cycle in _enter_waking_up (the start of the cycle) — NOT in
        # _enter_awake, which runs *after* the interrupt path's conscious
        # selection and would otherwise wipe that stream's credit.  Initialised
        # here so the very first interrupted wakeup can't hit an AttributeError
        # (conscious selection happens before _enter_awake on that path).
        self._was_conscious_streams: set = set()
        self.winding_down_started: bool = False
        self._wakeup_reason: Optional[str] = None
        # Lightweight inter-stream mailbox.  Any stream can post a message
        # addressed to another stream by name; the recipient drains its
        # mailbox at the start of its execute() tick.  Guarded by a lock
        # because background threads (alignment LLM scoring, LLM start/stop)
        # can post concurrently with the main loop — e.g. a scoring thread
        # calling router.get_client() → _request_ensure_role() → post_message.
        self._mailboxes: Dict[str, List[Dict[str, Any]]] = {}
        self._mailbox_lock = threading.Lock()
        # User input that arrived during sleep but did not warrant an urgent
        # wakeup (e.g. untrusted Telegram).  Merged into the first awake tick
        # by _merge_deferred_sensor_data so it is processed, not dropped.
        self._deferred_sleep_sensors: Dict[str, List[Any]] = {}
        # Names of generated streams that have graduated (proven useful over
        # several cycles).  Maintained by StreamFactory; consulted in
        # _run_regular_streams to grant durable-fact STM permissions.
        self._graduated_stream_names: set = set()
        # Per-cycle tally of facts a graduated stream contributed that survived
        # sleep replay into LTM.  Written in _promote_stm_to_ltm, consumed by
        # StreamFactory's end-of-cycle evaluation, then reset.
        self._graduated_fact_credit: Dict[str, int] = {}
        # Names of sleep-housekeeping phases already completed this cycle.
        # Reset in _enter_asleep; consumed by the sleep-phase scheduler.
        self._sleep_phases_done: set = set()
        # HLD: "dreaming" replay is skipped on the very first sleep of this process run.
        self._is_first_sleep: bool = True

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
            data: Dict[str, Any] = {
                "iyye_day": self.iyye_day,
                "journal_cycle": getattr(self, "_journal_cycle", 0),
            }
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
                            # Attach the journal so BaseActuator.actuate can
                            # shadow-record real output (Phase 0 causal record).
                            instance.journal = getattr(self, 'journal', None)
                            self.actuators[key] = instance
                            log.info("Loaded actuator %s from %s", key, fname)

            except Exception as exc:  # pragma: no‑cover
                log.error("Failed to load IO sensor %s – %s", fname, exc)

    def _load_streams(self) -> None:
        """Search ./streams/ for .py files and instantiate subclasses of ProcessingStream.

        LLM-generated stream files (``llm_*``) are short-lived artefacts
        created by StreamFactory codegen.  They are **not** reloaded on
        restart — StreamFactory recreates them on demand when fresh sensor
        data or goal gaps appear.  Stale files left over from a previous
        session are deleted here so they don't accumulate.

        The _factory_created guard only applies to shipped streams like
        UserChatStream / PlannedContinuationStream that require constructor
        arguments.
        """
        streams_dir = PROJECT_ROOT / "streams"
        if not streams_dir.is_dir():
            log.warning("Directory 'streams' not found – no streams loaded.")
            return

        for fname in sorted(os.listdir(streams_dir)):
            if not (fname.endswith(".py") and fname != "__init__.py"):
                continue
            # LLM-generated stream files are recreated on demand by
            # StreamFactory.  Delete stale leftovers from previous sessions
            # so they don't accumulate on disk.
            if fname.startswith("llm_") and fname != "llm_management_stream.py":
                stale_path = streams_dir / fname
                try:
                    os.remove(stale_path)
                    log.info("Deleted stale LLM-generated stream file: %s", fname)
                except OSError as exc:
                    log.warning("Failed to delete stale stream file %s: %s", fname, exc)
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
        from streams.planner_stream import PlannerStream

        # Create special streams if not already present
        special_names = {'attention_stream', 'alignment_stream',
                        'stream_factory', 'self_reflection', 'adenosine_stream',
                        'stm_update', 'llm_management', 'theory_of_mind',
                        'planner'}
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
            elif name == 'planner':
                stream = PlannerStream(self)
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
        # Shadow-journal the tick boundary (Phase 0): the replay clock and the
        # frame every other causal event is positioned within.
        from event_journal import emit
        conscious = getattr(self, '_current_conscious', None)
        emit(getattr(self, 'journal', None), 'tick',
             tick=self._tick_counter, state=self.state.name,
             conscious=(conscious.name if conscious is not None else None))

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
                stamped = [_stamp(p) for p in payloads]
                sensors_data[name] = stamped
                # Shadow-journal raw sensor inputs (HLD: collected in all
                # states).  Best-effort; never let journaling break the tick.
                journal = getattr(self, 'journal', None)
                if journal is not None:
                    for item in stamped:
                        journal.append('sensor_input', sensor=name, payload=item)
                # NOTE: a cursor-based sensor (Telegram) is NOT acknowledged
                # here.  Journaling alone is not recoverable — nothing replays
                # sensor_input on restart — so acking after journaling could
                # still lose a message that crashes before it is processed.
                # The acknowledgment happens only once the consumer (the chat
                # stream) has actually processed the message
                # (UserChatStream -> sensor.mark_processed), with Telegram
                # itself acting as the durable inbox (it re-delivers anything
                # not yet acked).

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
            # Kick LLM readiness off the waking path: even the health-probe +
            # port-seeding does HTTP probes that can take seconds, and the cold
            # start is now async — run the whole thing on a daemon thread so
            # waking stays "very short" (P2-b).  The loop tolerates the LLM not
            # being ready on the first awake tick (scheduler model_unavailable +
            # chat retry + the health-check loop).
            llm_mgmt = next(
                (s for s in self.streams if s.name == 'llm_management'), None
            )
            if llm_mgmt is not None:
                threading.Thread(
                    target=llm_mgmt.ensure_running,
                    name="llm_ensure_running", daemon=True,
                ).start()
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

    # Names of streams that run in a fixed order after the conscious stream.
    _SPECIAL_STREAM_NAMES = frozenset({
        'attention_stream', 'alignment_stream',
        'stream_factory', 'adenosine_stream', 'stm_update',
    })
    # Execution order for special streams: factory creates streams BEFORE
    # attention_stream decides which to promote.
    _SPECIAL_STREAM_ORDER = [
        'alignment_stream', 'stm_update', 'stream_factory',
        'adenosine_stream', 'attention_stream',
    ]

    def _awake_actions(self, sensors_data: Dict[str, List[Any]]) -> Optional[List[Any]]:
        """Main awake state processing.

        Phase order:
        1. merge_deferred_sensor_data — inject data from waking/interrupt/factory
        2. build_tick_context — assemble the dict every stream receives
        3. run_regular_streams — subconscious (non-special, non-conscious)
        4. ensure_conscious_stream — pick one if none assigned
        5. run_conscious_stream — chat / active task
        6. run_special_streams — alignment → stm → factory → adenosine → attention
        7. apply_attention_result — promote/demote if attention said so
        8. maybe_wind_down — periodic save + adenosine depletion check
        """
        if not self.streams:
            log.warning("No active streams!")
            return None

        sensors_data = self._merge_deferred_sensor_data(sensors_data)
        context = self._build_tick_context(sensors_data)
        subconscious_results = self._run_regular_streams(context)
        self._ensure_conscious_stream()
        conscious_result = self._run_conscious_stream(context)
        subconscious_results += self._run_special_streams(context)
        self._apply_attention_result(subconscious_results)
        self._maybe_wind_down()

        # Reconcile declared liveness from this tick's end state and journal
        # any transitions (Phase 2): one place computes busy/idle, and the
        # per-stream state timeline lands in the replay tape.
        self.reconcile_lifecycles()

        return [conscious_result] if conscious_result is not None else subconscious_results

    # ---- awake phase helpers ----------------------------------------- #

    def _merge_deferred_sensor_data(
        self, sensors_data: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        """Inject sensor payloads that were stashed during prior transitions."""
        _dict_attrs = ('_waking_up_sensors_data', '_deferred_sleep_sensors')
        for attr in ('_waking_up_sensors_data',
                     '_deferred_sleep_sensors',
                     '_pending_interrupt_data',
                     '_pending_factory_replay'):
            stashed = getattr(self, attr, None)
            if stashed:
                for name, items in stashed.items():
                    if items:
                        sensors_data.setdefault(name, [])
                        sensors_data[name] = items + sensors_data[name]
                setattr(self, attr, {} if attr in _dict_attrs else None)
        return sensors_data

    def _build_tick_context(self, sensors_data: Dict[str, List[Any]]) -> dict:
        return {
            'sensors_data': sensors_data,
            'streams': self.streams,
            'memory': self.memory,
            'stm': self.stm,
            'current_conscious': getattr(self, '_current_conscious', None),
            'adenosine': self.adenosine.level,
            'tick_counter': getattr(self, '_tick_counter', 0),
            'actuators': self.actuators,
            'self_reflection_state': self.self_reflection_snapshot(),
        }

    def stream_views(self):
        """Immutable read-only snapshots of all active streams.

        The cross-stream *query* contract: attention/alignment/factory/
        self-reflection inspect peers through these views instead of holding
        raw mutable stream objects (so internals can change freely and an
        observer cannot mutate a peer)."""
        return [s.to_view() for s in self.streams]

    def stream_view(self, name: str):
        """View of a single stream by name, or None."""
        for s in self.streams:
            if s.name == name:
                return s.to_view()
        return None

    # ------------------------------------------------------------------ #
    # Stream liveness contract (Phase 2): one declared state, one predicate,
    # read by every actor that might reap or stop a stream — instead of each
    # re-inferring "busy?" from side effects.  See StreamLifecycle, replay.py.
    # ------------------------------------------------------------------ #

    # Special/infrastructure streams that must NEVER be reaped, regardless of
    # apparent idleness.  The single canonical set (the factory and brain
    # previously kept two that had drifted — 'planner' was in neither, so the
    # pruner could reap it after ~50 ticks).
    _NEVER_REAP = frozenset({
        'attention_stream', 'alignment_stream', 'stream_factory',
        'self_reflection', 'adenosine_stream', 'stm_update',
        'llm_management', 'theory_of_mind', 'planner',
    })

    def stream_busy_state(self, stream) -> "StreamLifecycle":
        """Compute a stream's effective liveness from authoritative signals.

        AWAITING wins: an in-flight scheduler job or an open multi-stage turn
        means an async result is outstanding and reaping would orphan it — the
        prune-while-rephrase race.  Then RETIRING (self-requested), then ACTIVE
        (unprocessed messages or inputs awaiting an output), else IDLE."""
        from iyye_base import StreamLifecycle
        sched = getattr(self, 'llm_scheduler', None)
        if getattr(stream, '_turn', None):
            return StreamLifecycle.AWAITING
        if sched is not None:
            try:
                if sched.has_inflight(stream.name):
                    return StreamLifecycle.AWAITING
            except Exception:
                pass
        if getattr(stream, '_retiring', False):
            return StreamLifecycle.RETIRING
        if len(getattr(stream, '_pending_messages', []) or []):
            return StreamLifecycle.ACTIVE
        inputs = len(getattr(stream, 'input_history', []) or [])
        outputs = len(getattr(stream, 'output_history', []) or [])
        if inputs > outputs:
            return StreamLifecycle.ACTIVE
        return StreamLifecycle.IDLE

    def is_reapable(self, stream) -> bool:
        """Single safety predicate: may this stream be removed/stopped right
        now without losing work?  The contract that replaces the four
        per-actor inferences.  Does NOT decide *whether* to reap (that is the
        factory's policy) — only whether it is *safe* to."""
        from iyye_base import StreamLifecycle
        if stream is getattr(self, '_current_conscious', None):
            return False
        if stream.name in self._NEVER_REAP:
            return False
        if getattr(stream, '_in_critical_section', False):
            return False
        return self.stream_busy_state(stream) in (
            StreamLifecycle.IDLE, StreamLifecycle.RETIRING,
        )

    def reconcile_lifecycles(self) -> None:
        """Recompute every stream's declared lifecycle and journal changes.

        Run once per awake tick so the busy/idle decision is made in one place
        and the per-stream state timeline lands in the replay tape.  Reaping
        safety reads live signals via is_reapable(); this maintains the
        observable record."""
        for stream in list(self.streams):
            try:
                stream.transition(self.stream_busy_state(stream), reason="reconcile")
            except Exception as exc:
                log.debug("reconcile_lifecycles: %s failed: %s", stream.name, exc)

    def theory_of_mind(self):
        """Stable accessor for the Theory-of-Mind stream (or None if not yet
        started).  Consumers use this instead of reaching for the private
        ``_tom_stream`` attribute."""
        return getattr(self, '_tom_stream', None)

    def self_reflection_snapshot(self):
        """Stable accessor for the latest self-reflection system snapshot."""
        return getattr(self, '_self_reflection_snapshot', None)

    def _stream_by_name(self, name: str):
        """Resolve a stream name to the live object (brain-internal use)."""
        for s in self.streams:
            if s.name == name:
                return s
        return None

    def record_alignment(self, scores_by_name: Dict[str, Dict[str, float]]) -> None:
        """Apply alignment scores to streams (the owner applies them).

        The alignment stream computes scores from read-only views and hands
        them here instead of writing into peer stream objects directly."""
        for name, scores in (scores_by_name or {}).items():
            stream = self._stream_by_name(name)
            if stream is not None:
                stream.alignment_scores = scores

    def _run_regular_streams(self, context: dict) -> List[Any]:
        """Execute non-special, non-conscious subconscious streams."""
        current_conscious = getattr(self, '_current_conscious', None)
        results: List[Any] = []
        for stream in self.streams:
            if stream is current_conscious:
                continue
            if stream.name in self._SPECIAL_STREAM_NAMES:
                continue
            # LLM-generated streams run with a capability-scoped context (no
            # raw brain / actuators / cross-stream / LTM-write).  Shipped,
            # reviewed streams keep the full context.
            if getattr(stream, '_source_file', None):
                ctx = self._scoped_context_for(stream, context)
            else:
                ctx = context
            try:
                result = stream.execute(ctx)
                if result:
                    results.append(result)
                # Stream activity is captured by the event journal via
                # ProcessingStream.add_to_log (stream_activity events); no
                # separate last_conscious_log capture is needed.
            except StopIteration:
                log.info("Stream %s stopped at checkpoint", stream.name)
            except Exception as exc:
                log.error("Subconscious stream %s error: %s", stream.name, exc)
        return results

    def _scoped_context_for(self, stream: ProcessingStream, base: dict) -> dict:
        """Build a least-privilege context for an LLM-generated stream.

        Drops the broad raw handles — all actuators, every other stream object,
        and the raw LTM client — and swaps in scoped façades: a stage-scoped
        STM wrapper (session for candidates, durable for graduated), read-only
        LTM, and a ``cap`` handle.  Sensor inputs and adenosine/tick pass
        through (reads).  Graduated streams additionally receive a mediated,
        rate-limited ``emit`` (Phase 2)."""
        from capabilities import ReadOnlyMemory, Capabilities
        graduated = stream.name in self._graduated_stream_names
        if self.stm is None:
            stm = None
        elif graduated:
            stm = _GraduatedSTM(self.stm, f"gen_graduated:{stream.name}")
        else:
            stm = _SessionOnlySTM(self.stm, f"llm_gen:{stream.name}")
        ro_mem = ReadOnlyMemory(self.memory)
        emit_fn = self._emitter_for(stream) if graduated else None
        cap = Capabilities(
            stm=stm, memory=ro_mem, stream=stream,
            tier=('graduated' if graduated else 'candidate'), emit_fn=emit_fn,
        )
        ctx = dict(base)
        ctx['stm'] = stm
        ctx['memory'] = ro_mem        # read-only façade (writes raise)
        ctx['streams'] = []           # no cross-stream access
        ctx['actuators'] = {}         # no direct actuator access
        ctx['current_conscious'] = None
        ctx['cap'] = cap
        # Read scoping: a generated stream sees only the sensor(s) it was
        # registered for.  _cap_sensors is stamped at creation/reload from the
        # coverage key — a sensor handler gets {its_sensor}, a goal stream gets
        # the empty set (it works from STM/memory, not raw sensor data).
        allowed = getattr(stream, '_cap_sensors', None)
        if allowed is not None:
            sd = base.get('sensors_data', {}) or {}
            ctx['sensors_data'] = {k: v for k, v in sd.items() if k in allowed}
        return ctx

    def _web_chat_actuator(self):
        """Return the local web-chat actuator (the only channel generated
        streams may reach), or None."""
        for name, act in self.actuators.items():
            if 'web' in name.lower() or 'chat' in name.lower():
                return act
        return None

    def _emitter_for(self, stream: ProcessingStream):
        """Mediated, rate-limited output grant for a *graduated* stream.

        Routes only through the local web chat (never Telegram/TTS) and caps
        messages per awake cycle.  The emitter is cached on the stream so its
        per-cycle counter persists across ticks; it is reset in _enter_awake."""
        from capabilities import MediatedEmitter
        em = getattr(stream, '_cap_emitter', None)
        if em is None:
            em = MediatedEmitter(self._web_chat_actuator(), stream.name)
            try:
                stream._cap_emitter = em
            except Exception:
                return None
        return em

    def _ensure_conscious_stream(self) -> None:
        """Select a conscious stream if none is currently assigned."""
        if getattr(self, '_current_conscious', None) is not None:
            return
        if not self.streams:
            return
        candidates = [s for s in self.streams
                      if getattr(s, '_can_be_conscious', True)]
        if candidates:
            candidates.sort(key=lambda s: -getattr(s, 'priority', 1))
            self._current_conscious = candidates[0]
            self._was_conscious_streams.add(self._current_conscious.name)
            log.info("Selected %s as conscious stream", self._current_conscious.name)

    def _run_conscious_stream(self, context: dict) -> Any:
        """Execute the conscious stream. Returns its result or None."""
        current_conscious = self._current_conscious
        if current_conscious is None:
            return None
        current_conscious._last_conscious_tick = getattr(self, '_tick_counter', 0)
        try:
            result = current_conscious.execute(context)
            # Activity captured by the journal (stream_activity events).
            # The stream may have retired itself (removed from self.streams)
            # during execute().  Clear the pointer so the next tick picks fresh.
            if current_conscious not in self.streams:
                log.info("Conscious stream %s retired itself", current_conscious.name)
                current_conscious.is_conscious = False
                self._current_conscious = None
            return result
        except StopIteration:
            log.info("Conscious stream %s stopped at checkpoint",
                     current_conscious.name)
            self._current_conscious = None
        except Exception as exc:
            log.error("Conscious stream %s error: %s", current_conscious.name, exc)
        return None

    def _run_special_streams(self, context: dict) -> List[Any]:
        """Run special subconscious streams in fixed order."""
        by_name = {s.name: s for s in self.streams
                   if s.name in self._SPECIAL_STREAM_NAMES}
        results: List[Any] = []
        for sp_name in self._SPECIAL_STREAM_ORDER:
            stream = by_name.get(sp_name)
            if stream is None:
                continue
            try:
                result = stream.execute(context)
                if result:
                    results.append(result)
            except Exception as exc:
                log.error("Special stream %s error: %s", sp_name, exc)
        return results

    def _apply_attention_result(self, subconscious_results: List[Any]) -> None:
        """Promote/demote streams if the attention stream requested a swap."""
        attention_result = None
        for r in subconscious_results:
            if isinstance(r, dict) and 'promote' in r:
                attention_result = r
                break

        if not (attention_result and attention_result.get('promote')):
            return

        # Attention returns the NAME to promote (read contract); resolve it to
        # the live stream here.
        promote_name = attention_result['promote']
        target = self._stream_by_name(promote_name)
        if target is None:
            log.warning("Attention: promote target %r not found", promote_name)
            return

        old_conscious = self._current_conscious
        if old_conscious:
            old_conscious.is_conscious = False
            old_conscious._last_conscious_tick = getattr(self, '_tick_counter', 0)
        self._current_conscious = target
        self._current_conscious.is_conscious = True
        self._current_conscious._last_conscious_tick = getattr(self, '_tick_counter', 0)
        self._was_conscious_streams.add(self._current_conscious.name)
        log.info("Attention: promoting %s (demoting %s)",
                 self._current_conscious.name,
                 old_conscious.name if old_conscious else "none")
        if hasattr(self, 'adenosine'):
            self.adenosine.drain_activity("consciousness_switch")

    def _maybe_wind_down(self) -> None:
        """Periodic state save and adenosine depletion check."""
        _awake_ticks = getattr(self, '_awake_tick_count', 0) + 1
        self._awake_tick_count = _awake_ticks
        if _awake_ticks % 100 == 0:
            self._save_iyye_state()

        _MIN_AWAKE_TICKS = 10
        if self.adenosine.is_depleted() and _awake_ticks >= _MIN_AWAKE_TICKS:
            log.info("Adenosine depleted - initiating wind-down (awake %d ticks)",
                     _awake_ticks)
            self.state = MindState.WINDING_DOWN
            self.winding_down_started = False

    def _run_system_check(self) -> Dict[str, Any]:
        """Drive the asleep system-check.

        HLD assigns the system-description to SelfReflectionStream, so the
        brain (scheduler) delegates to it when it is running.  On the very
        first sleep — before subconscious streams are started — self-reflection
        does not exist yet, so the brain bootstraps the check directly via the
        same shared producer in ``system_description``."""
        from system_description import run_system_check
        sr = next((s for s in self.streams if s.name == 'self_reflection'), None)
        if sr is not None and callable(getattr(sr, 'perform_system_check', None)):
            return sr.perform_system_check()
        return run_system_check(self)


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
            if self._current_conscious is not None:
                self._current_conscious.request_stop()
                log.info("Requested conscious stream %s to stop",
                         self._current_conscious.name)

            # HLD: "all subconscious streams pause" — set the pause flag now
            # so background work (alignment LLM scoring, LLM start/stop
            # threads) stops spawning even before the conscious stream has
            # reached its checkpoint.  Threads already in flight are joined
            # by _settle_subconscious_streams() below.
            for stream in self.streams:
                if stream is self._current_conscious:
                    continue
                try:
                    stream.pause()
                except Exception as exc:
                    log.warning("pause() failed for %s: %s", stream.name, exc)

            # Stop the async LLM scheduler accepting new jobs.  In-flight jobs
            # keep running; they are bounded-joined in _settle_subconscious_streams
            # and any that don't finish have their results discarded on next wake.
            sch = getattr(self, 'llm_scheduler', None)
            if sch is not None:
                sch.begin_pause()

        # Wait for conscious stream to finish any in-progress work.
        current_conscious = self._current_conscious
        if current_conscious is not None:
            if not current_conscious.can_stop_safely():
                log.debug("Waiting for conscious stream to reach safe checkpoint...")
                return
            current_conscious.is_conscious = False
            self._current_conscious = None

        # Drain in-flight background threads with a bounded budget, then flush
        # per-stream persistent state via on_pause().  This is the second phase
        # of the HLD pause protocol — first phase (the flag) was set above.
        self._settle_subconscious_streams()

        self.winding_down_started = False
        # HLD odd req: push "starting sleep" while actuators are still alive.
        self._push_to_web_chat("starting sleep")
        self._stop_actuators()
        self._enter_asleep()

    def _settle_subconscious_streams(self) -> None:
        """Wait briefly for stream background threads to drain, then flush.

        HLD: "all subconscious streams pause" — after pause() blocks new
        background work, settle() joins in-flight threads with a per-stream
        timeout and a total wall-clock cap.  on_pause() then runs for each
        stream so persistent in-memory state (e.g. ToM dirty contacts) gets
        flushed before _enter_asleep snapshots the brain.
        """
        deadline = time.monotonic() + _PAUSE_SETTLE_TOTAL_S
        for stream in self.streams:
            if stream is self._current_conscious:
                continue
            remaining = max(0.1, deadline - time.monotonic())
            budget = min(_PAUSE_SETTLE_TIMEOUT_S, remaining)
            try:
                if not stream.settle(timeout_s=budget):
                    log.warning(
                        "Stream %s did not settle within %.1fs — daemon "
                        "threads may still mutate state past sleep boundary",
                        stream.name, budget,
                    )
            except Exception as exc:
                log.warning("settle() failed for %s: %s", stream.name, exc)
            try:
                stream.on_pause()
            except Exception as exc:
                log.warning("on_pause() failed for %s: %s", stream.name, exc)

        # Bounded-drain in-flight async LLM jobs within the remaining budget.
        sch = getattr(self, 'llm_scheduler', None)
        if sch is not None:
            remaining = max(0.1, deadline - time.monotonic())
            if not sch.settle(min(_PAUSE_SETTLE_TIMEOUT_S, remaining)):
                log.warning(
                    "LLM scheduler did not drain in-flight jobs before sleep — "
                    "their results will be discarded on next wake",
                )

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
        # _was_conscious_streams is reset in _enter_waking_up (cycle start), not
        # here — the interrupt path selects its conscious stream before
        # _enter_awake runs, so resetting here would erase that stream's credit.
        # Lift the wind-down pause so streams can spawn background work again.
        # Missing this call would silently leave alignment scoring and LLM
        # management disabled after the first sleep cycle.
        for stream in self.streams:
            try:
                stream.resume()
            except Exception as exc:
                log.warning("resume() failed for %s: %s", stream.name, exc)
            # Reset per-cycle mediated-emit budgets for graduated streams.
            em = getattr(stream, '_cap_emitter', None)
            if em is not None:
                em.reset()
        log.info("Transition → AWAKE (Iyye day %d, interrupted=%s)",
                 self.iyye_day, self._waking_interrupted)
        self.state = MindState.AWAKE

    def _rotate_journal_cycle(self) -> None:
        """Open a fresh journal partition once a cycle's events are replayed.

        Mirrors the clearing of last_cycle.json: after replay consumes a
        cycle's events, subsequent events belong to the next cycle.  The
        new cycle id is persisted so a restart resumes the right partition."""
        journal = getattr(self, 'journal', None)
        if journal is None:
            return
        self._journal_cycle = getattr(self, '_journal_cycle', 0) + 1
        journal.start_cycle(self._journal_cycle)
        self._save_iyye_state()
        log.debug("Journal rotated to cycle %d", self._journal_cycle)

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
        # Reset sleep-phase completion so each new sleep cycle runs all phases.
        self._sleep_phases_done = set()

        # Journal replay: drop the cached event list + pairing so the next
        # sleep re-reads the (now longer) append-only partition.  Keep the
        # cursor / promoted-set / discovered list — the partition's processed
        # prefix is stable, so we resume rather than re-promote.  This also
        # covers the interrupted-wakeup case (the awake cycle appended more
        # events to the same partition; we resume past the processed prefix).
        for attr in ('_jreplay_events', '_jreplay_fact_activity',
                     '_jreplay_activity_facts'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

        # ToM dirty contacts were already flushed by on_pause() during
        # _settle_subconscious_streams().  The crash-path flush in
        # _save_iyye_state remains as a safety net for KeyboardInterrupt /
        # OS exit that skips the normal winding-down sequence.

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
        """Post *message* to *target*'s mailbox (thread-safe).

        Safe to call from background threads (the lock serialises concurrent
        posts/drains).  *message* is a typed ``messaging.Message`` (preferred,
        built via ``Messages.*``) or a legacy dict — both are normalized to a
        validated Message here.  A message may set ``urgent`` (or use an action
        in ``_URGENT_MAILBOX_ACTIONS``) to request delivery even while the
        recipient is paused during wind-down — see ``drain_messages``.
        """
        from messaging import normalize_message
        msg = normalize_message(target, message)
        with self._mailbox_lock:
            self._mailboxes.setdefault(target, []).append(msg)

    def drain_messages(
        self, target: str, urgent_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return pending messages for *target*, removing them (thread-safe).

        With ``urgent_only=False`` (awake default) all messages are returned
        and the mailbox cleared.  With ``urgent_only=True`` — used by a paused
        stream during wind-down — only urgent control messages are returned;
        the rest stay queued, preserving order, for the next awake tick.  This
        is the explicit "which messages may be drained while winding down"
        policy: by default nothing is urgent, so a paused consumer defers
        everything (sleep quieting) unless a message opted in as pause-safe.
        """
        with self._mailbox_lock:
            if not urgent_only:
                return self._mailboxes.pop(target, [])
            queue = self._mailboxes.get(target)
            if not queue:
                return []
            urgent, deferred = [], []
            for m in queue:
                (urgent if self._is_urgent_message(m) else deferred).append(m)
            if deferred:
                self._mailboxes[target] = deferred
            else:
                self._mailboxes.pop(target, None)
            return urgent

    def peek_messages(self, target: str) -> List[Dict[str, Any]]:
        """Return a copy of *target*'s pending messages without removing them.

        Thread-safe read for callers that need to test for pending mail (e.g.
        ToM blocking wind-down until its mailbox is drained) without consuming
        it or touching the shared dict directly."""
        with self._mailbox_lock:
            return list(self._mailboxes.get(target, ()))

    @staticmethod
    def _is_urgent_message(message: Any) -> bool:
        """True if *message* may be delivered to a paused recipient.

        Works for both ``messaging.Message`` and legacy dicts (both expose
        ``.get`` and an ``action``/``urgent``)."""
        get = getattr(message, "get", None)
        if not callable(get):
            return False
        if get("urgent"):
            return True
        return get("action") in _URGENT_MAILBOX_ACTIONS

    # --------------------------------------------------------------- #
    # Graceful shutdown
    # --------------------------------------------------------------- #
    def shutdown(self) -> None:
        log.info("Shutting down IyyeBrain")
        # Flush Theory-of-Mind contacts before anything else — catches
        # interactions posted after the last ToM execute() or during a
        # crash/exit that skipped the normal WINDING_DOWN path.
        tom = self.theory_of_mind()
        if tom is not None and callable(getattr(tom, 'flush', None)):
            tom.flush()
        # Stop async LLM scheduler workers.
        sch = getattr(self, 'llm_scheduler', None)
        if sch is not None:
            try:
                sch.close()
            except Exception:
                pass
        # The awake cycle's activity is already durably in the event journal
        # (stream_activity / stm_fact events), so there's no separate cycle log
        # to persist here — replay folds the journal on the next sleep.
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
        journal = getattr(self, 'journal', None)
        if journal is not None:
            try:
                journal.close()
            except Exception:
                pass

    # Keywords that escalate a message to an "urgent" wake.  For web chat the
    # local owner is trusted so any message wakes; these only refine the logged
    # reason.  For Telegram they matter only under the opt-in urgent policy.
    _URGENT_KEYWORDS = ('urgent', 'emergency', 'help', 'important', 'wake')

    def _check_wakeup_triggers(self, sensors_data: Dict[str, List[Any]]) -> bool:
        """
        HLD: "Each important input source has a simple in-sleep processing routine
        that checks if latest input item should force urgent wakeup."

        "Input arrived" is deliberately distinct from "urgent wake":
        - web_chat is the local, fully-trusted owner → any message wakes.
        - microphone wake words and critical hardware readings → wake.
        - Telegram is untrusted by default (HLD): only a *trusted* sender
          forces a wake.  Messages from senders Iyye has not been told to
          trust are left queued and handled at the next natural wakeup (see
          _defer_sleep_input), so untrusted traffic can no longer repeatedly
          interrupt sleep, sap partial adenosine, or stall day-advancement
          and replay.  IYYE_TELEGRAM_URGENT_WAKE opts in to also waking on
          urgent-keyword messages from untrusted senders.
        """
        # Web chat — trusted local owner; any message is an explicit request.
        if "web_chat" in sensors_data and sensors_data["web_chat"]:
            for msg in sensors_data["web_chat"]:
                if isinstance(msg, str) and any(
                    kw in msg.lower() for kw in self._URGENT_KEYWORDS
                ):
                    self._wakeup_reason = f"urgent web_chat message: {msg[:50]}"
                    log.info("Wakeup triggered by urgent web_chat input")
                    return True
            self._wakeup_reason = "web_chat message"
            log.info("Wakeup triggered by web_chat input")
            return True

        # Microphone wake words.
        if "microphone_sensor" in sensors_data and sensors_data["microphone_sensor"]:
            for transcription in sensors_data["microphone_sensor"]:
                if isinstance(transcription, dict):
                    text = transcription.get("text", "").lower()
                    if any(ww in text for ww in ('iyye', 'hey iyye', 'wake up')):
                        self._wakeup_reason = f"wake word detected: {text[:50]}"
                        log.info("Wakeup triggered by wake word")
                        return True

        # Critical hardware conditions.
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

        # Telegram — untrusted by default.  Only a trusted sender (or, under
        # the opt-in urgent policy, an urgent-keyword message) forces a wake.
        # Everything else is deferred to natural wakeup by _defer_sleep_input.
        for key in sensors_data:
            if 'telegram' not in key.lower():
                continue
            saw_untrusted = False
            for item in sensors_data[key]:
                msgs = self._iter_telegram_messages(item)
                if not msgs:
                    saw_untrusted = saw_untrusted or bool(item)
                    continue
                for m in msgs:
                    if self._telegram_message_trusted(m, key):
                        sender = (m.get('first_name') or m.get('username')
                                  or 'trusted contact')
                        self._wakeup_reason = f"trusted Telegram message from {sender}"
                        log.info("Wakeup triggered by trusted Telegram sender (%s)", key)
                        return True
                    if _TELEGRAM_URGENT_WAKE:
                        text = str(m.get('text') or '').lower()
                        if any(kw in text for kw in self._URGENT_KEYWORDS):
                            self._wakeup_reason = f"urgent Telegram message on {key}"
                            log.info("Wakeup triggered by urgent Telegram input "
                                     "under opt-in policy (%s)", key)
                            return True
                    saw_untrusted = True
            if saw_untrusted:
                log.info("Telegram input on %s from untrusted sender(s) — deferring "
                         "to natural wakeup", key)

        # Long term plan deadlines — the scheduler input (HLD: a deadline on
        # an active plan step is a valid urgent-wakeup source, same mechanism
        # as a high priority sensor input).  Cheap in-memory timestamp check.
        # Each distinct deadline fires at most once (_plan_deadline_wake_fired)
        # so a deadline that stays overdue across an awake cycle cannot put
        # the brain in a sleep→instant-wake loop that starves dreaming.
        try:
            due = self.plan_store.next_due_deadline()
            if due is not None and due <= datetime.now(timezone.utc):
                if due != self._plan_deadline_wake_fired:
                    self._plan_deadline_wake_fired = due
                    self._wakeup_reason = f"plan step due at {due.isoformat()}"
                    log.info("Wakeup triggered by long term plan deadline")
                    return True
                log.debug(
                    "Plan deadline %s already woke this brain once — "
                    "letting sleep housekeeping proceed", due.isoformat(),
                )
        except Exception as exc:
            log.warning("Plan deadline wakeup check failed: %s", exc)

        # Don't check git sensor for code changes, only self can write to git.
        return False

    @staticmethod
    def _iter_telegram_messages(item: Any) -> List[Dict[str, Any]]:
        """Normalize a raw Telegram sensor payload into a list of message dicts.

        Handles the direct per-message dict ({'text','chat_id','user_id',...}),
        the legacy MCP batch format ({'count': N, 'messages': [...]}), and
        returns [] for anything without a recognizable message shape (e.g. a
        bare string or a count-only batch) — those carry no sender identity so
        they cannot establish trust."""
        if isinstance(item, dict):
            msgs = item.get('messages')
            if isinstance(msgs, list):
                return [m for m in msgs if isinstance(m, dict)]
            if 'text' in item or 'chat_id' in item or 'update_id' in item:
                return [item]
        return []

    def _telegram_message_trusted(self, msg: Dict[str, Any], source: str) -> bool:
        """Return True if *msg*'s sender is a trusted Theory-of-Mind contact.

        HLD: a Telegram sender is trusted only after the owner explicitly
        trusts them via the local web chat (the sole channel that can change
        trust).  When the ToM stream isn't running yet (the very first sleep
        of a process, before any wake cycle) nobody is trusted, so untrusted
        Telegram cannot wake the system."""
        tom = self.theory_of_mind()
        if tom is None:
            return False
        chat_id = msg.get('chat_id')
        user_id = msg.get('user_id')
        if not (chat_id or user_id):
            return False  # no stable identity → cannot be a trusted contact
        first = msg.get('first_name') or ''
        username = msg.get('username') or ''
        sender = first or (f'@{username}' if username else None)
        try:
            cid = tom.make_contact_id(
                sender, source,
                int(chat_id) if chat_id else None,
                int(user_id) if user_id else None,
            )
            return bool(tom.is_contact_trusted(cid))
        except Exception as exc:
            log.debug("Telegram trust check failed: %s", exc)
            return False

    def _defer_sleep_input(self, sensors_data: Dict[str, List[Any]]) -> None:
        """Buffer user input that arrived during sleep but did not warrant an
        urgent wakeup, so the next natural wakeup processes it instead of
        dropping it (the payloads were already popped from their queues).

        Only message-bearing sensors (Telegram) are deferred — ephemeral
        hardware/metric readings are not re-injected.  Buffer is capped per
        sensor so untrusted floods cannot grow memory without bound."""
        buf = getattr(self, '_deferred_sleep_sensors', None) or {}
        for name, items in sensors_data.items():
            if 'telegram' in name.lower() and items:
                merged = buf.setdefault(name, [])
                merged.extend(items)
                if len(merged) > _DEFERRED_INPUT_CAP:
                    del merged[:-_DEFERRED_INPUT_CAP]
        if buf:
            self._deferred_sleep_sensors = buf

    # Sleep-phase scheduling --------------------------------------------- #
    # Phases with order < the gate run before the wakeup-trigger check (so the
    # system description is current even on an interrupt wake); phases ≥ the
    # gate run after.  The brain owns sequencing + the wakeup/transition
    # decisions; the *work* of each phase is owned by a stream/producer.
    _SLEEP_WAKEUP_GATE_ORDER = 50
    _REPLAY_BATCH = 3  # LLM extraction calls per replay tick
    # Event types sleep replay reads (everything else — sensor_input, stm_merge
    # — is high-volume noise replay ignores; filtering them keeps the replay
    # working set bounded, issue #9):
    #   stream_activity — key-fact extraction source + alignment for what-if
    #   stm_fact        — STM facts to promote to LTM
    #   stm_remove / ltm_promotion / extracted — idempotency markers re-seeded
    #                     on a cold restart so work isn't repeated
    _REPLAY_EVENT_TYPES = frozenset({
        'stream_activity', 'stm_fact', 'stm_remove', 'ltm_promotion', 'extracted',
    })

    def _sleep_core_phases(self) -> List["SleepPhase"]:
        from iyye_base import SleepPhase
        return [
            # system check → SelfReflectionStream (owns system_description),
            # with first-sleep bootstrap inside _run_system_check.
            SleepPhase("system_check", lambda b: (b._run_system_check(), True)[1], 10),
            # STM flush → StmUpdateStream (owns fact extraction).
            SleepPhase("stm_flush", lambda b: b._sleep_phase_stm_flush(), 20),
            SleepPhase("prewarm",   lambda b: b._sleep_phase_prewarm(), 60),
            SleepPhase("replay",    lambda b: b._sleep_phase_replay(), 70),
            SleepPhase("cleanup",   lambda b: b._sleep_phase_cleanup(), 80),
        ]

    def _sleep_phases(self) -> List["SleepPhase"]:
        """Assemble the ordered sleep pipeline: brain core phases plus any a
        stream registers via ``sleep_phases()`` (HLD: housekeeping is stream
        work; the brain only schedules it)."""
        phases = list(self._sleep_core_phases())
        for s in self.streams:
            try:
                phases.extend(s.sleep_phases() or [])
            except Exception as exc:
                log.warning("sleep_phases() failed for %s: %s", s.name, exc)
        phases.sort(key=lambda p: p.order)
        return phases

    def _run_sleep_phases(self, phases, done: set) -> bool:
        """Run not-yet-done *phases* in order; stop at the first that reports
        it needs more ticks (e.g. batched replay).  Returns True when all are
        done."""
        for ph in phases:
            if ph.name in done:
                continue
            if ph.run(self):
                done.add(ph.name)
            else:
                return False
        return True

    def _sleep_phase_stm_flush(self) -> bool:
        stm_stream = next((s for s in self.streams if s.name == 'stm_update'), None)
        if stm_stream is not None and callable(getattr(stm_stream, 'flush', None)):
            stm_stream.flush()
        return True

    def _sleep_phase_prewarm(self) -> bool:
        router = getattr(self, 'llm_router', None)
        if router is not None:
            hp = router._healthy_ports
            stm_model = router._find_model("stm")
            if stm_model and hp is not None and stm_model["port"] not in hp:
                from messaging import Messages
                self.post_message("llm_management", Messages.ensure_role(
                    role="stm", model_name=stm_model["name"],
                    task={"prompt_tokens": 800, "expected_output_tokens": 200,
                          "quality_need": 0.3, "latency_budget_s": 15,
                          "urgency": 0.6},
                    reason="prewarm for sleep replay fact extraction",
                ))
        return True

    def _sleep_phase_replay(self) -> bool:
        """Dreaming: fold this cycle's event journal, one batch per tick.

        Returns True when replay is complete (or skipped on the first sleep).

        Replay's LLM fact-extraction (``_extract_key_facts`` →
        ``_get_replay_extraction_client``) runs **synchronously**, by design —
        it is *not* routed through the async LLM scheduler (issue #3).  The
        scheduler exists to keep blocking LLM calls off the AWAKE main loop;
        during sleep there is no conscious stream to starve and the loop is
        otherwise idle, replay is already cooperatively batched
        (``_REPLAY_BATCH`` extractions per tick), and the scheduler is paused
        (``begin_pause``) for the duration of sleep.  Making replay async would
        require running the scheduler during sleep and turning the sleep-phase
        pipeline into a job-polling state machine — added complexity for no
        latency benefit.  This is the deliberate resolution of migration
        step 7; see llm_scheduler_plan.md."""
        if self._is_first_sleep:
            self._is_first_sleep = False   # HLD: skip dreaming on first sleep
            return True
        if not hasattr(self, '_jreplay_events'):
            self._replay_journal_init()
        if self._replay_journal_step(self._REPLAY_BATCH):
            self._replay_journal_finish()
            self._rotate_journal_cycle()
            return True
        return False  # more replay ticks needed

    def _sleep_phase_cleanup(self) -> bool:
        # HLD: "keeps deleting the processed part of STM."  Trim in-memory
        # queues and old on-disk history.
        for name, q in self.sensors.items():
            if len(q) > 1000:
                while len(q) > 500:
                    q.popleft()
                log.debug("Trimmed sensor queue: %s", name)
        self._cleanup_stm_files(keep_days=3)
        return True

    def _asleep_actions(self, sensors_data: Dict[str, List[Any]]) -> None:
        """
        HLD: "Asleep state runs only 'housekeeping' subconscious tasks and
        checks some sensor inputs."

        The brain acts as the *scheduler*: it replenishes adenosine, re-injects
        deferred input, drives the ordered sleep-housekeeping phase pipeline
        (whose work is owned by streams/producers), evaluates the wakeup
        trigger, and performs state transitions.
        """
        # HLD: adenosine "partially/proportionally replenished" — owned math on
        # the AdenosineStream; the scheduler decides when to tick it.
        self.adenosine.replenish(0.05)

        # Re-inject sensor data accumulated during WINDING_DOWN (already popped
        # from queues) for the first awake tick — without treating it as a new
        # interrupt (else the day counter would never advance).
        saved = getattr(self, '_winding_down_sensors', None)
        if saved:
            self._pending_interrupt_data = saved
            self._winding_down_sensors = {}

        phases = self._sleep_phases()
        done = self._sleep_phases_done

        # Pre-gate housekeeping (system check, STM flush) must complete before
        # the wakeup check so system_description.md is current even on an
        # interrupt wake.
        self._run_sleep_phases(
            [p for p in phases if p.order < self._SLEEP_WAKEUP_GATE_ORDER], done,
        )

        # Wakeup gate (lifecycle decision — stays with the scheduler).
        if self._check_wakeup_triggers(sensors_data):
            existing = getattr(self, '_pending_interrupt_data', None) or {}
            for _n, _items in sensors_data.items():
                existing.setdefault(_n, [])
                existing[_n].extend(_items)
            self._pending_interrupt_data = existing
            self._enter_waking_up(interrupted=True)
            return

        # Preserve deferrable input (e.g. untrusted Telegram) for natural wake.
        self._defer_sleep_input(sensors_data)

        # Post-gate housekeeping (prewarm, dreaming/replay, cleanup).  Replay
        # yields across ticks; when every post-gate phase is done, wake.
        if self._run_sleep_phases(
            [p for p in phases if p.order >= self._SLEEP_WAKEUP_GATE_ORDER], done,
        ):
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
        # Start of a new awake cycle: clear per-cycle conscious tracking before
        # any stream is selected.  (Selection for the interrupt path happens in
        # the next tick's _waking_up_actions, after this reset.)
        self._was_conscious_streams = set()
        # New wake epoch: bump and resume the async LLM scheduler (re-enable
        # accepting, drop any results stranded from the previous cycle).  The
        # scheduler may not exist yet on the very first wakeup — it is created
        # by LlmManagementStream during _waking_up_actions and adopts the epoch
        # itself.
        self._wake_epoch += 1
        sch = getattr(self, 'llm_scheduler', None)
        if sch is not None:
            sch.on_wake(self._wake_epoch)

        # _start_subconscious_streams() is called in _waking_up_actions()

    def _cleanup_stm_files(self, keep_days: int = 3) -> None:
        """
        HLD: "keeps deleting the processed part of short term memory from last
        awake cycle (can also compact)."

        Deletes day-log files older than `keep_days` from:
        - io_history/<subdir>/YYYY-MM-DD.txt
        - streams_history/<subdir>/YYYY-MM-DD.txt
        - stm_history/YYYY-MM-DD.jsonl  (and any media .tgz they reference)

        STM JSONL files persist all facts including those that sleep replay
        rejected as unpromotable; without this cleanup those files would
        grow without bound on fact-heavy days.
        """
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=keep_days)).isoformat()
        deleted = 0

        # io_history/ and streams_history/ — txt files under per-source subdirs
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

        # stm_history/ — jsonl files at the root; media/ subdir is preserved
        # (live media references are managed by ShortTermMemory.remove_by_ids).
        stm_root = PROJECT_ROOT / "stm_history"
        if stm_root.exists():
            for fpath in list(stm_root.iterdir()):
                if not fpath.is_file() or fpath.suffix != ".jsonl":
                    continue
                if fpath.stem >= cutoff:
                    continue
                # Collect media paths referenced by this day-file so the
                # .tgz archives don't outlive their JSONL index.  Handles
                # both the legacy ``media_path`` scalar and the new
                # ``media_paths`` list produced by dedup merges.
                media_paths: List[str] = []
                try:
                    with fpath.open(encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                fact = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(fact, dict):
                                continue
                            mp_list = fact.get("media_paths")
                            if isinstance(mp_list, list):
                                media_paths.extend(p for p in mp_list if p)
                            mp = fact.get("media_path")
                            if mp:
                                media_paths.append(mp)
                except Exception as exc:
                    log.warning("Could not scan STM file for media %s: %s", fpath, exc)
                try:
                    fpath.unlink()
                    deleted += 1
                    log.debug("Deleted old STM file: %s", fpath)
                except Exception as exc:
                    log.warning("Could not delete STM file %s: %s", fpath, exc)
                    continue
                for mp in media_paths:
                    try:
                        Path(mp).unlink()
                    except OSError:
                        pass  # already gone or never existed

        if deleted:
            log.info("STM cleanup: deleted %d file(s) older than %s", deleted, cutoff)

        # Prune old event-journal partitions too — keep roughly the last
        # `keep_days` cycles (a cycle ≈ a day).  The current cycle is never
        # pruned by EventJournal.prune.
        journal = getattr(self, 'journal', None)
        if journal is not None:
            journal.prune(keep_after_cycle=getattr(self, '_journal_cycle', 0) - keep_days)

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

    # ------------------------------------------------------------------ #
    # Phase 2: replay by folding the event journal
    # ------------------------------------------------------------------ #
    def _replay_journal_init(self) -> None:
        """Load this cycle's journal events and precompute fact↔activity pairing.

        Pairing is plain adjacency: each ``stm_fact`` event is associated with
        the nearest preceding ``stream_activity`` event (they were appended in
        true temporal order), so no timestamp parsing or windowed scoring is
        needed.

        Replay is made **idempotent** so a mid-replay process restart cannot
        re-promote facts or repeat extraction: progress is reconstructed from
        the journal itself.  Every STM fact replay consumes is removed via a
        ``stm_remove`` event, and every stream_activity whose key-fact
        extraction ran emits an ``extracted`` event — so on a cold start we
        re-seed the processed/extracted sets from those events and skip
        anything already done (rather than relying on an in-memory cursor that
        a crash would lose).  Within a process, the sets simply persist across
        ticks and across an interrupted wakeup."""
        cid = getattr(self, '_journal_cycle', 0)
        journal = getattr(self, 'journal', None)
        # Stream the partition but materialize ONLY the event types replay
        # actually uses — sensor_input (camera/hardware/telegram volume) and
        # stm_merge are skipped, so the working set is bounded by cognitive
        # activity, not sensor noise (issue #9).
        self._jreplay_events: List[Dict[str, Any]] = (
            journal.read_cycle(cid, types=self._REPLAY_EVENT_TYPES)
            if journal is not None else []
        )
        # fact_id -> activity event; activity index -> [paired fact texts]
        self._jreplay_fact_activity: Dict[str, Dict[str, Any]] = {}
        self._jreplay_activity_facts: Dict[int, List[str]] = {}
        last_act: Optional[int] = None
        for i, e in enumerate(self._jreplay_events):
            etype = e.get('type')
            if etype == 'stream_activity':
                last_act = i
            elif etype == 'stm_fact' and last_act is not None:
                fid = e.get('fact_id')
                if fid:
                    self._jreplay_fact_activity[fid] = self._jreplay_events[last_act]
                    self._jreplay_activity_facts.setdefault(last_act, []).append(
                        e.get('text', '')
                    )

        # Reconstruct durable progress on a cold start.  These sets persist in
        # memory across ticks/interrupts within a process; after a restart they
        # are rebuilt from the journal so already-done work is not repeated.
        restored = False
        if not hasattr(self, '_jreplay_promoted'):
            processed: set = set()
            for e in self._jreplay_events:
                t = e.get('type')
                if t == 'stm_remove':
                    # Every STM fact replay consumed (promoted or discarded).
                    processed.update(e.get('ids') or [])
                elif t == 'ltm_promotion':
                    # A promotion is journaled per-fact *before* the batched
                    # stm_remove at the end of the step.  Seeding from these
                    # too closes the crash window where a fact was promoted to
                    # LTM but the process died before stm_remove was appended —
                    # otherwise that fact would be promoted a second time.
                    fid = e.get('fact_id')
                    if fid:
                        processed.add(fid)
            self._jreplay_promoted = processed
            restored = bool(processed)
        if not hasattr(self, '_jreplay_extracted'):
            self._jreplay_extracted = {
                e.get('activity_seq') for e in self._jreplay_events
                if e.get('type') == 'extracted' and e.get('activity_seq') is not None
            }
            restored = restored or bool(self._jreplay_extracted)
        if not hasattr(self, '_jreplay_cursor'):
            self._jreplay_cursor = 0
        if not hasattr(self, '_jreplay_discovered'):
            self._jreplay_discovered: List[Dict[str, Any]] = []
        if self._jreplay_events:
            log.info(
                "Sleep replay (journal) %s: cycle %d, %d events, %d facts paired, "
                "%d already processed, %d already extracted",
                "resuming" if (restored or self._jreplay_promoted) else "starting",
                cid, len(self._jreplay_events),
                len(self._jreplay_fact_activity), len(self._jreplay_promoted),
                len(self._jreplay_extracted),
            )

    def _replay_activity_extractable(
        self, stream_name: str, text: str,
    ) -> Optional[str]:
        """Return the meaningful (non-metric) text to extract from a
        stream_activity event, or None when it should be skipped — same policy
        as the legacy replay (chat/telegram and housekeeping streams skipped,
        pure metric snapshots dropped)."""
        if not text:
            return None
        sn = stream_name.lower()
        if (any(kw in sn for kw in ('chat', 'telegram'))
                or stream_name in _REPLAY_SKIP_LLM
                or any(sn.startswith(p) for p in _REPLAY_SKIP_PREFIXES)
                or any(kw in sn for kw in _REPLAY_SKIP_KEYWORDS)):
            return None
        lines = [
            ln for ln in text.splitlines()
            if ln.strip() and not _EPHEMERAL_METRIC_RE.search(ln)
        ]
        if not lines:
            return None
        return '\n'.join(lines)

    def _replay_journal_step(self, batch_limit: int) -> bool:
        """Process journal events from the cursor; return True when finished.

        Walks events in order: ``stm_fact`` → promote the (live) STM fact to
        LTM with its paired activity as provenance; ``stream_activity`` → LLM
        key-fact extraction (capped at *batch_limit* LLM calls per tick).
        Processed STM facts are deleted from STM in step (HLD: "keeps deleting
        the processed part of STM"), so an interrupt can't re-promote them."""
        events = self._jreplay_events
        cursor = self._jreplay_cursor
        stm = getattr(self, 'stm', None)
        live_by_id = (
            {f['id']: f for f in stm.get_all_today()} if stm is not None else {}
        )
        extractions = 0
        step_removed: List[str] = []

        while cursor < len(events):
            e = events[cursor]
            etype = e.get('type')

            if etype == 'stm_fact':
                fid = e.get('fact_id')
                if fid and fid not in self._jreplay_promoted:
                    fact = live_by_id.get(fid) or {
                        'id': fid,
                        'text': e.get('text', ''),
                        'confidence': e.get('confidence', 0.7),
                        'provenance': e.get('provenance', ''),
                        'time_frame': e.get('time_frame', 'permanent'),
                    }
                    paired = self._jreplay_fact_activity.get(fid)
                    entry = (
                        {'stream': paired.get('stream'), 'timestamp': paired.get('ts')}
                        if paired else None
                    )
                    result = self._promote_stm_to_ltm(fact, entry)
                    if result is None:
                        # Transient LTM failure — keep the STM fact (do NOT add
                        # to step_removed) and leave it unmarked so a restart or
                        # later pass retries it.  Never delete the only copy of a
                        # fact whose promotion failed (HLD: promote then delete).
                        log.warning("Replay: promotion failed for %s — keeping "
                                    "STM fact for retry", fid)
                    else:
                        # Promoted (id) or intentionally filtered — either way the
                        # STM copy is now safe to delete.
                        self._jreplay_promoted.add(fid)
                        if result != self._PROMOTE_FILTERED:
                            self._jreplay_discovered.append(
                                {'id': result, 'text': fact.get('text', '')}
                            )
                            if getattr(self, 'journal', None) is not None:
                                self.journal.append(
                                    'ltm_promotion', ltm_id=result,
                                    fact_id=fid, src_seq=e.get('seq'),
                                    text=fact.get('text', '')[:200],
                                )
                        step_removed.append(fid)

            elif etype == 'stream_activity':
                seq = e.get('seq')
                # Skip activities whose extraction already ran (durably marked
                # by an `extracted` event) — prevents repeat extraction after a
                # mid-replay restart.
                if seq in self._jreplay_extracted:
                    cursor += 1
                    continue
                clean = self._replay_activity_extractable(
                    e.get('stream', 'unknown'), e.get('text', ''),
                )
                if clean is not None:
                    if extractions >= batch_limit:
                        break  # resume next tick (cursor not advanced past this)
                    extractions += 1
                    stream_name = e.get('stream', 'unknown')
                    paired_facts = self._jreplay_activity_facts.get(cursor, [])
                    key_facts = self._extract_key_facts(
                        clean, stream_name=stream_name, paired_facts=paired_facts,
                    )
                    for kf in key_facts:
                        kf_text = kf.get('text', '')
                        if (_EPHEMERAL_METRIC_RE.search(kf_text)
                                or _LTM_NOISE_RE.search(kf_text)):
                            continue
                        # Per-fact tags from extraction (HLD: LTM fact format
                        # is the same tagged format as STM).  Ephemeral facts
                        # never belong in LTM.
                        if kf.get('time_frame') == 'ephemeral':
                            continue
                        sid = self.memory.store_fact(
                            text=kf_text,
                            confidence=kf.get('confidence', 0.7),
                            source=stream_name,
                            provenance=(
                                f"Extracted during sleep replay from "
                                f"'{stream_name}' at {e.get('ts')}"
                            ),
                            time_frame=kf.get('time_frame', 'permanent'),
                        )
                        self._jreplay_discovered.append(
                            {'id': sid, 'text': kf_text})
                    # Mark extraction done AFTER the facts are durable
                    # (at-least-once).  A crash mid-store leaves the activity
                    # unmarked, so restart re-extracts and re-stores — and LTM's
                    # semantic dedup in store_fact makes the re-store idempotent
                    # (no duplicates).  This is safer than marking-first
                    # (at-most-once), which permanently dropped facts whose store
                    # was interrupted (P1-d).
                    self._jreplay_extracted.add(seq)
                    if getattr(self, 'journal', None) is not None:
                        self.journal.append('extracted', activity_seq=seq)

            cursor += 1

        self._jreplay_cursor = cursor
        if stm is not None and step_removed:
            stm.remove_by_ids(step_removed)
            log.debug("Replay (journal): removed %d processed STM fact(s)",
                      len(step_removed))
        return cursor >= len(events)

    def _replay_journal_finish(self) -> None:
        """Run end-of-replay housekeeping and clear journal-replay scratch."""
        discovered = getattr(self, '_jreplay_discovered', [])
        self._fine_tune_dnns(discovered)
        # What-if reads conscious decisions (alignment_scores) directly from the
        # journal's stream_activity events — no separate last_conscious_log.
        decisions = [
            {'alignment_scores': e.get('alignment_scores') or {},
             'result': e.get('text', '')}
            for e in getattr(self, '_jreplay_events', [])
            if e.get('type') == 'stream_activity' and e.get('alignment_scores')
        ]
        self._run_what_if_simulations(decisions)
        log.info("Sleep replay (journal) done: %d item(s) to LTM", len(discovered))
        for attr in ('_jreplay_events', '_jreplay_cursor', '_jreplay_promoted',
                     '_jreplay_extracted', '_jreplay_discovered',
                     '_jreplay_fact_activity', '_jreplay_activity_facts'):
            try:
                delattr(self, attr)
            except AttributeError:
                pass

    # Sentinel distinguishing "intentionally not promoted, safe to delete from
    # STM" from a transient store failure (None) which must NOT delete the only
    # surviving copy (P1-b: failed promotion was deleting the STM fact).
    _PROMOTE_FILTERED = "__filtered__"

    def _promote_stm_to_ltm(
        self,
        stm_fact: Dict[str, Any],
        entry: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Write a single STM fact into long-term memory, preserving all HLD tags.
        Returns the LTM fact id on success, ``_PROMOTE_FILTERED`` when the fact
        is intentionally not promoted (safe to delete from STM), or None on a
        transient store failure (caller must KEEP the STM fact and retry).
        Ephemeral and session facts are never promoted (ephemeral = transient
        metrics; session = relevant only to the current wakeup cycle).
        Content matching noise patterns (LLM placeholders, system status,
        vague cognitive state) is also rejected.
        Facts originating from LLM-generated streams are also rejected —
        their internal bookkeeping is not durable knowledge about the world.
        """
        if stm_fact.get('time_frame') in ('ephemeral', 'session'):
            return self._PROMOTE_FILTERED
        text = stm_fact.get('text', '')
        if _EPHEMERAL_METRIC_RE.search(text) or _LTM_NOISE_RE.search(text):
            return self._PROMOTE_FILTERED
        # Reject facts from LLM-generated / planned streams.  Their output
        # is internal recommendations or operational noise, never durable
        # knowledge about the world — regardless of whether the attention
        # stream promoted one to consciousness.
        prov = stm_fact.get('provenance', '')
        if prov:
            prov_lower = prov.lower()
            if (any(prov_lower.startswith(p) for p in _REPLAY_SKIP_PREFIXES)
                    or any(kw in prov_lower for kw in _REPLAY_SKIP_KEYWORDS)):
                return self._PROMOTE_FILTERED
        try:
            provenance = "Promoted from STM during sleep replay"
            if entry:
                provenance += (
                    f" (stream '{entry.get('stream')}'"
                    f" at {entry.get('timestamp')})"
                )
            # LTM's pyarrow schema stores a single media_path; if the STM
            # fact accumulated multiple archives via dedup merges, pass the
            # first and log so the loss is visible.  Follow-up: extend LTM
            # schema to a list.
            from iyye_io.short_term_memory import media_paths_of
            paths = media_paths_of(stm_fact)
            if len(paths) > 1:
                log.info(
                    "STM→LTM: fact has %d media archives; promoting first only "
                    "(LTM is single-media): %s",
                    len(paths), stm_fact.get('text', '')[:80],
                )
            ltm_id = self.memory.store_fact(
                text=stm_fact['text'],
                confidence=float(stm_fact.get('confidence', 0.7)),
                source=stm_fact.get('provenance', 'agent'),
                provenance=provenance,
                time_frame=stm_fact.get('time_frame', 'permanent'),
                media_path=paths[0] if paths else None,
            )
            # Credit a graduated generated stream when its fact reaches LTM —
            # this is a concrete impact signal the factory uses to keep the
            # stream graduated (or demote it if it stops contributing).
            if ltm_id and prov.startswith('gen_graduated:'):
                name = prov.split(':', 1)[1].split(',', 1)[0].strip()
                if name:
                    self._graduated_fact_credit[name] = (
                        self._graduated_fact_credit.get(name, 0) + 1
                    )
            return ltm_id
        except Exception as exc:
            log.warning("Failed to promote STM fact to LTM: %s", exc)
            return None

    def _extract_key_facts(
        self,
        text: str,
        stream_name: str = "unknown",
        paired_facts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract key facts from conscious stream text using LLM.
        Falls back to heuristic extraction if the LLM is unreachable.

        Returns tagged fact dicts (``text``/``confidence``/``time_frame``) —
        HLD: LTM fact format is the same tagged format as STM, so dreaming
        must carry per-fact tags rather than storing every extraction with
        one hardcoded confidence/time_frame.

        ``paired_facts`` are STM facts that the replay pairing step
        associated with this log entry — used as additional context per
        HLD ("processes [temporally associated facts] using LLM inference").
        """
        # Format paired-fact context block for the prompt template.  Empty
        # list collapses to a "(none)" sentinel so the prompt stays valid.
        if paired_facts:
            paired_block = "\n".join(f"- {pf}" for pf in paired_facts[:10])
        else:
            paired_block = "(none)"
        client = self._get_replay_extraction_client()
        if client is not None:
            try:
                response = client.complete_from_file(
                    "extract_facts",
                    stream_name=stream_name,
                    stream_output=text,
                    paired_facts=paired_block,
                )
                return self._parse_extracted_facts(response, stream_name)
            except Exception as exc:
                log.warning("LLM fact extraction failed for %s, using heuristic "
                            "fallback: %s", stream_name, exc)

        # Heuristic fallback — reached when the router/LLM is unavailable or the
        # extraction call raised.  Lower quality; the warnings above make the
        # degradation visible rather than silent.  Conservative default tags:
        # keyword-matched sentences are uncertain (0.5) and not provably
        # durable, so 'recent' rather than 'permanent'.
        facts = []
        for sentence in text.replace('!', '.').replace('?', '.').split('.'):
            s = sentence.strip()
            if 20 < len(s) < 200:
                if any(kw in s.lower() for kw in ['learned', 'discovered', 'found',
                                                   'determined', 'concluded', 'noted']):
                    facts.append({'text': s, 'confidence': 0.5,
                                  'time_frame': 'recent'})
        return facts[:5]

    def _get_replay_extraction_client(self):
        """Return an LLM client for sleep-replay fact extraction.

        Routes through ``brain.llm_router`` using the same ``stm`` role the
        sleep prewarm requested (see _asleep_actions), so extraction uses the
        prewarmed/fast model and inherits router health checks, role selection,
        and model lifecycle.  Only when the router is unavailable (LLM
        management stream never started) does it fall back to a bare
        LLMClient — logged clearly so degraded routing is visible.

        Returns None if even the fallback client cannot be constructed; the
        caller then uses heuristic extraction.
        """
        router = getattr(self, 'llm_router', None)
        if router is not None:
            try:
                return router.get_client(role="stm", no_think=True)
            except Exception as exc:
                log.warning("Replay extraction: router.get_client(role='stm') "
                            "failed (%s) — falling back to default LLMClient", exc)
        else:
            log.warning("Replay extraction: llm_router unavailable — falling back "
                        "to default LLMClient (no role routing or health checks)")
        try:
            from llm_client import LLMClient
            return LLMClient(no_think=True)
        except Exception as exc:
            log.warning("Replay extraction: could not construct fallback "
                        "LLMClient (%s) — using heuristic extraction", exc)
            return None

    def _parse_extracted_facts(
        self, response: str, stream_name: str,
    ) -> List[Dict[str, Any]]:
        """Parse LLM extraction output into tagged fact dicts.

        Primary format is one JSON object per line (text/confidence/
        time_frame — see prompts/extract_facts.md); a non-JSON line that
        passes the noise filters is kept as a plain-text fact with
        conservative default tags, so a model that ignores the JSON
        instruction degrades gracefully instead of losing the fact.
        """
        from iyye_io.short_term_memory import TIME_FRAMES
        facts: List[Dict[str, Any]] = []
        for line in response.splitlines():
            line = line.strip().strip('`')
            if not line:
                continue
            text, confidence, time_frame = line, 0.7, 'permanent'
            if line.startswith('{'):
                try:
                    obj = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue  # malformed JSON line — not salvageable as text
                if not isinstance(obj, dict) or not obj.get('text'):
                    continue
                text = str(obj['text']).strip()
                try:
                    confidence = max(0.2, min(1.0, float(obj.get('confidence', 0.7))))
                except (TypeError, ValueError):
                    confidence = 0.7
                tf = str(obj.get('time_frame', '')).strip().lower()
                if tf in TIME_FRAMES:
                    time_frame = tf
            # Noise filters apply to the fact text in either format.
            if _LTM_NOISE_RE.search(text):
                continue
            if _EPHEMERAL_METRIC_RE.search(text):
                continue
            # Skip HTML / thinking tags
            if text.startswith('<') and '>' in text:
                continue
            # Skip markdown headings and horizontal rules
            if text.startswith('#') or text == '---':
                continue
            # Skip very short lines (likely fragments)
            if len(text) < 10:
                continue
            facts.append({'text': text, 'confidence': confidence,
                          'time_frame': time_frame})
        log.debug("LLM extracted %d facts from %s", len(facts), stream_name)
        return facts[:10]

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


