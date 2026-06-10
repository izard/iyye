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

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.Journal")


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

    def read_cycle(self, cycle_id: int) -> List[Dict[str, Any]]:
        """Return all events for *cycle_id* in append order."""
        path = self._partition_path(cycle_id)
        events: List[Dict[str, Any]] = []
        if not path.exists():
            return events
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as exc:
            log.warning("Journal: could not read %s: %s", path, exc)
        return events

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


__all__ = ["EventJournal"]
