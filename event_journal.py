# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Append-only typed event journal — the single ordered source of truth for
the memory pipeline.

HLD: activity logs, STM facts, IO history and replay are all parts of one
memory pipeline.  Historically the implementation kept those as parallel
stores (``streams_history``, ``io_history``, STM JSONL, ``last_conscious_log``)
that sleep replay had to *re-correlate* by timestamp.  This journal is the
canonical interleaved record: every memory-relevant event is appended in true
temporal order with a monotonic per-cycle sequence number, so STM and the
dreaming/replay phase can be derived as a simple fold over it instead of
re-joining separate logs.

Layout: ``journal/cycle_<id>.jsonl`` — one JSON event per line.  Each event is::

    {"seq": int, "ts": ISO-8601, "type": str, ...payload}

Event types currently emitted (Phase 1 — shadow mode, written alongside the
existing stores; readers are introduced in later phases):

* ``sensor_input``    {"sensor", "payload"}                 — a raw sensor item
* ``stream_activity`` {"stream", "text"}                    — one activity-log line
* ``stm_fact``        {"fact_id","text","confidence",
                       "provenance","time_frame","media_paths"} — a new STM fact
* ``stm_merge``       {"into_id","text","provenance","time_frame"} — dedup merge
* ``ltm_promotion``   {"ltm_id","fact_id","src_seq","text"} — a fact promoted to LTM
* ``extracted``       {"activity_seq"}                       — key-fact extraction
                                                              ran for a stream_activity

Causal-loop events (Phase 0 — shadow recording of the cognitive loop's
non-deterministic inputs/outputs, the basis for the deterministic replay
harness and the forensic flight recorder).  These are NOT yet consumed; they
record so a recorded cycle can later be replayed/asserted:

* ``tick``        {"tick","state","conscious"}              — one logical tick of the loop
* ``llm_submit``  {"job_id","stream","kind","role",
                   "conscious","prompt","prompt_name",
                   "prompt_version"}                        — an async LLM job was enqueued
                                                              (``prompt_version`` is the serving prompt
                                                              version, "base" or a learned vid — the
                                                              per-version reward key for #6 prompt tuning)
* ``llm_result``  {"job_id","stream","ok","error","model",
                   "latency_s","discarded","text"}          — that job resolved (raw response)
* ``tool_exec``   {"stream","kind","code","output"}         — a tool ran (e.g. python subprocess)
* ``actuate``     {"actuator","payload"}                    — what actually reached an output device
                                                              (post dedup/suppression guardrails)
* ``lifecycle``   {"stream","old","new","reason"}           — a stream's declared liveness changed
* ``recall``         {"query_id","query","n","sources",
                      "refs"}                                — a unified memory recall ran; ``refs`` are the
                                                              retrieved fact ids (the usefulness denominator)
* ``recall_used``    {"query_id","refs","sources"}          — recalled facts that informed a response
* ``recall_feedback``{"query_id","signal"}                  — next-turn implicit feedback (satisfied /
                                                              dissatisfied) on a fact-using turn
                                                              (recall/recall_used/recall_feedback join on
                                                              query_id — the retrieval-quality signal #5 B/C
                                                              folds into per-fact usefulness)
* ``plan_review``  {"plan_id","lifecycle","candidate",
                    "reasons","days_since_progress",...}     — a dreaming replan assessment of one plan
* ``plan_revised``  {"plan_id","old_pending","new_pending",
                    "escalated","lifecycle"}                 — dreaming replanned a plan's pending steps
* ``plan_resolved`` {"plan_id","verdict","reason"}          — dreaming judged the goal achieved/moot
                                                              (suspended for owner confirmation)
* ``attention_decision`` {"stream","score","features",
                          "weights"}                        — attention promoted a stream (the feature
                                                              vector + weights behind the decision)
* ``attention_tuning``   {"old","proposed","n",
                          "mean_reward","applied"}          — sleep feedback loop's per-cycle weight
                                                              adjustment (#4; shadow until applied)
* ``prompt_tuning``      {"name","version","n",
                          "mean_reward","decision"}         — sleep self-improvement of a prompt: per-
                                                              version success rate this cycle and any
                                                              promote/rollback of a running prompt trial
                                                              (#6; ``prompt_version`` on llm_submit is the
                                                              attribution key)
* ``prompt_proposed``    {"name","success_rate",
                          "accepted","reason"}              — the gated act phase proposed an LLM rewrite
                                                              of an underperforming prompt; ``accepted``
                                                              records whether it passed validation and
                                                              became a trial (#6)
* ``memory_decay``       {"fact_id","old","new",
                          "time_frame","retrieved","used",
                          "applied"}                        — sleep LTM hygiene down-weighted a fact from
                                                              age-vs-durability + recall usefulness
* ``memory_prune_candidate`` {"fact_id","time_frame",
                          "age_days","decayed","retrieved",
                          "used","applied"}                 — a non-durable, aged-out, unused fact eligible
                                                              for deletion (``applied`` gated)
* ``memory_superseded``  {"winner","loser","applied"}      — two facts judged to contradict; the loser
                                                              (lower durability/recency/confidence) retired
* ``memory_maintenance`` {"scanned","decayed",
                          "prune_candidates","pruned",
                          "superseded", ...applied flags}   — per-sleep summary of the maintenance pass
                                                              (decay/prune/supersede; shadow until gated on)

Sleep replay is made restart-safe by these events rather than a saved cursor:
on a cold start it rebuilds progress from ``stm_remove`` (every STM fact it
already consumed) and ``extracted`` (every activity whose extraction already
ran), so a mid-replay restart neither re-promotes facts nor repeats LLM
extraction — and because each fact/activity is marked as it completes, this
survives a crash *mid-batch*, which a single saved cursor would not.

The journal is append-only and thread-safe: background threads (alignment LLM
scoring, LLM start/stop) may emit concurrently with the main loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.Journal")

# Recorded text fields (LLM responses, tool output) are bounded so a
# pathological response can't bloat a partition.  Generous enough that normal
# chat/codegen output is captured whole — replay fidelity wants the full text.
_MAX_RECORDED_CHARS = 64_000


def clip(text: Any, limit: int = _MAX_RECORDED_CHARS) -> str:
    """Stringify and bound *text* for journaling, marking truncation."""
    s = text if isinstance(text, str) else str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"…[+{len(s) - limit} chars]"


def fingerprint(text: Any) -> str:
    """Short stable digest — identifies a prompt/script without storing it twice."""
    s = text if isinstance(text, str) else str(text)
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()[:12]


def emit(journal: Optional["EventJournal"], event_type: str, **fields: Any) -> None:
    """Best-effort shadow emit: no-op when *journal* is None; never raises.

    The single guarded entry point every causal-event call site uses, so
    instrumentation can never break the cognitive loop over a journal hiccup."""
    if journal is None:
        return
    try:
        journal.append(event_type, **fields)
    except Exception:
        pass


class EventJournal:
    """Per-cycle append-only event log with monotonic sequence numbers."""

    def __init__(self, base_dir: str = "journal") -> None:
        self._base = PROJECT_ROOT / base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._cycle_id: Optional[int] = None
        self._seq: int = 0
        self._fh = None

    # ------------------------------------------------------------------
    # Partition management
    # ------------------------------------------------------------------

    def _partition_path(self, cycle_id: int) -> Path:
        return self._base / f"cycle_{cycle_id}.jsonl"

    def _close_fh(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def start_cycle(self, cycle_id: int) -> None:
        """Begin (or resume) the partition for *cycle_id*.

        If the partition already exists — e.g. a restart mid-cycle before
        replay cleared it — the sequence counter resumes past the existing
        events so seq numbers stay monotonic and nothing is overwritten."""
        with self._lock:
            self._close_fh()
            self._cycle_id = cycle_id
            path = self._partition_path(cycle_id)
            resumed = 0
            if path.exists():
                try:
                    with path.open(encoding="utf-8") as fh:
                        resumed = sum(1 for _ in fh)
                except OSError:
                    resumed = 0
            self._seq = resumed
            try:
                self._fh = path.open("a", encoding="utf-8")
            except OSError as exc:
                log.warning("Journal: could not open %s: %s", path, exc)
                self._fh = None
            log.debug("Journal: cycle %s started (resumed at seq=%d)", cycle_id, resumed)

    @property
    def cycle_id(self) -> Optional[int]:
        return self._cycle_id

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, event_type: str, **fields: Any) -> int:
        """Append a typed event; return its per-cycle sequence number.

        Best-effort: a journal write failure is logged but never raised, so a
        disk hiccup cannot crash the main loop.  Returns -1 if not written."""
        with self._lock:
            if self._fh is None:
                # Lazily open a default partition if start_cycle wasn't called.
                self.start_cycle(self._cycle_id if self._cycle_id is not None else 0)
            if self._fh is None:
                return -1
            seq = self._seq
            record = {
                "seq": seq,
                "ts": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
            }
            record.update(fields)
            try:
                self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._fh.flush()
            except Exception as exc:
                log.warning("Journal: append failed (%s): %s", event_type, exc)
                return -1
            self._seq += 1
            return seq

    # ------------------------------------------------------------------
    # Read (used by later phases; safe to call any time)
    # ------------------------------------------------------------------

    def iter_cycle(self, cycle_id: int,
                   types: Optional["frozenset"] = None) -> "Iterator[Dict[str, Any]]":
        """Yield events for *cycle_id* in append order, one at a time.

        Streams the partition file line by line rather than building the whole
        list in memory.  When *types* is given, only events whose ``type`` is in
        the set are yielded — so a consumer that only needs a few event kinds
        (e.g. sleep replay, which ignores the high-volume ``sensor_input`` and
        ``stm_merge`` events) never materializes the rest (issue #9)."""
        path = self._partition_path(cycle_id)
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if types is None or ev.get("type") in types:
                        yield ev
        except OSError as exc:
            log.warning("Journal: could not read %s: %s", path, exc)

    def read_cycle(self, cycle_id: int,
                   types: Optional["frozenset"] = None) -> List[Dict[str, Any]]:
        """Return events for *cycle_id* in append order (optionally filtered to
        *types*).  Convenience wrapper over :meth:`iter_cycle`."""
        return list(self.iter_cycle(cycle_id, types=types))

    def cycle_ids(self) -> List[int]:
        """All cycle ids currently on disk, ascending."""
        ids: List[int] = []
        for p in self._base.glob("cycle_*.jsonl"):
            try:
                ids.append(int(p.stem.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return sorted(ids)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune(self, keep_after_cycle: int) -> int:
        """Delete partitions for cycles strictly older than *keep_after_cycle*.

        The current (open) cycle is never deleted.  Returns the number of
        partitions removed."""
        removed = 0
        with self._lock:
            for cid in self.cycle_ids():
                if cid < keep_after_cycle and cid != self._cycle_id:
                    try:
                        self._partition_path(cid).unlink()
                        removed += 1
                    except OSError as exc:
                        log.warning("Journal: could not prune cycle %d: %s", cid, exc)
        if removed:
            log.info("Journal: pruned %d old partition(s)", removed)
        return removed

    def close(self) -> None:
        with self._lock:
            self._close_fh()


__all__ = ["EventJournal", "emit", "clip", "fingerprint"]
