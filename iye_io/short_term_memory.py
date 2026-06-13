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
Short-Term Memory — structured fact store, resident in memory and on disk.

HLD: "a text file, also resident in memory, which contains list of facts.
Each fact has a tag, which includes date and time, confidence level (0.2-1.0),
provenance (person, agent, subsystem), time frame when fact is true
(e.g. always, today, 2 weeks ago, etc), and text describing the inputs
that contributed to fact inference."

File layout: stm_history/YYYY-MM-DD.jsonl — one JSON object per line per day.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import tarfile
import uuid
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.STM")

_DEDUP_SIMILARITY = 0.70   # Jaccard word-overlap threshold for near-duplicates
_MAX_PROVENANCE   = 500    # hard cap on provenance string length
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TS_TAIL_RE = re.compile(r"\s+at\s+\d{4}-\d{2}-\d{2}T[\d:.+Z-]+$")

# Durability rank: higher = survives longer.  Used by dedup to upgrade
# time_frame when a near-duplicate arrives with a more durable classification.
_TF_RANK = {tf: i for i, tf in enumerate(
    ("ephemeral", "session", "today", "recent", "dated", "permanent")
)}


def _tokens(text: str) -> frozenset:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union)


def media_paths_of(fact: Dict[str, Any]) -> List[str]:
    """Return all media archive paths attached to *fact*.

    Reads the new list-shaped ``media_paths`` field if present, falls back
    to the legacy scalar ``media_path``.  Use this helper anywhere downstream
    code needs to enumerate media (cleanup, LTM promotion) so that
    multi-media facts (produced by add_fact dedup merges) aren't truncated.
    """
    paths = fact.get('media_paths')
    if paths:
        return [p for p in paths if p]
    legacy = fact.get('media_path')
    return [legacy] if legacy else []


def _set_media_paths(fact: Dict[str, Any], paths: List[str]) -> None:
    """Write *paths* back into *fact*, picking the most compact encoding.

    Single-element lists are stored as the legacy ``media_path`` scalar so
    on-disk JSONL stays backward-compatible with anything that still reads
    that field.  Multi-element lists are stored as ``media_paths`` and the
    scalar field is removed to avoid a stale duplicate.
    """
    paths = [p for p in paths if p]
    fact.pop('media_path',  None)
    fact.pop('media_paths', None)
    if len(paths) == 1:
        fact['media_path']  = paths[0]
    elif len(paths) > 1:
        fact['media_paths'] = paths


def _merge_provenance(old: str, new: str) -> str:
    """Combine two provenance strings within the _MAX_PROVENANCE cap.

    Preserves all distinct source identities (stripping ISO timestamps so
    repeats from the same source don't accumulate).  When the merged string
    would exceed the cap, **oldest** segments are dropped from the left and
    an ellipsis marker is prepended — the new identity is always retained
    rather than silently discarded.
    """
    if not new:
        return old
    if not old:
        return new[:_MAX_PROVENANCE]
    old_segs = [s.strip() for s in old.split(',') if s.strip()]
    new_seg  = new.strip()
    new_key  = _TS_TAIL_RE.sub('', new_seg)
    existing_keys = {_TS_TAIL_RE.sub('', s) for s in old_segs}
    if new_key in existing_keys:
        return old  # already represented
    segments = old_segs + [new_seg]
    merged = ', '.join(segments)
    if len(merged) <= _MAX_PROVENANCE:
        return merged
    # Drop oldest segments until it fits, keep the new one no matter what.
    while len(segments) > 1 and len(merged) > _MAX_PROVENANCE:
        segments.pop(0)
        merged = '…, ' + ', '.join(segments)
    if len(merged) > _MAX_PROVENANCE:
        # Single segment still too long — truncate the new identity itself.
        merged = merged[:_MAX_PROVENANCE - 1] + '…'
    return merged


# Valid time_frame values (open-ended; these are the canonical ones).
# "ephemeral" = valid for only a few seconds (system snapshots: CPU%, mem%, etc.)
# — these are never promoted to LTM.
TIME_FRAMES = ("permanent", "session", "today", "recent", "dated", "ephemeral")


class ShortTermMemory:
    """
    In-memory list of tagged facts backed by a daily JSONL file.

    Each fact dict:
        id         — uuid string
        timestamp  — ISO UTC string
        confidence — float 0.2-1.0
        provenance — free-form string (person name, stream name, subsystem)
        time_frame — one of TIME_FRAMES
        text       — the fact statement
        media_path — optional path to a tgz archive of supporting media
    """

    def __init__(self, base_dir: str = "stm_history") -> None:
        self._base = PROJECT_ROOT / base_dir
        self._base.mkdir(parents=True, exist_ok=True)
        # Optional event-journal sink (set by the brain).  When present,
        # fact adds and dedup merges are mirrored as stm_fact/stm_merge events
        # so STM can later be derived as a fold over the journal.
        self.journal = None
        self._media_dir = self._base / "media"
        self._media_dir.mkdir(parents=True, exist_ok=True)
        self._facts: List[Dict[str, Any]] = []
        # Previous-day facts kept after day rollover so sleep replay can still
        # access pre-midnight facts from the same awake cycle.  Never written
        # to the new day's file.
        self._prev_day_facts: List[Dict[str, Any]] = []
        self._today: date = date.today()
        self._load_today()
        # On a cold restart the next calendar day, _prev_day_facts is empty
        # because _maybe_roll_day never ran.  Load yesterday's file so sleep
        # replay can still temporally associate STM facts from the previous
        # awake cycle (which may span the midnight boundary).
        if not self._prev_day_facts:
            self._load_previous_day()
        log.info("ShortTermMemory ready at %s (%d facts, %d prev-day)",
                 base_dir, len(self._facts), len(self._prev_day_facts))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_media(
        self,
        data: Union[bytes, List[tuple]],
        filename: str = "media.bin",
    ) -> str:
        """Create a tgz archive of supporting media and return its path.

        HLD: "Optional field is a tgz file that may contain supporting media."

        Args:
            data: Either raw bytes for a single file, or a list of
                  (filename, bytes) tuples for multiple files.
            filename: Name of the single file inside the archive (ignored
                      when *data* is a list of tuples).

        Returns:
            Absolute path to the created .tgz file.
        """
        archive_id = str(uuid.uuid4())[:12]
        tgz_path = self._media_dir / f"{archive_id}.tgz"

        with tarfile.open(str(tgz_path), "w:gz") as tar:
            if isinstance(data, bytes):
                items = [(filename, data)]
            else:
                items = data
            for fname, fdata in items:
                info = tarfile.TarInfo(name=fname)
                info.size = len(fdata)
                tar.addfile(info, io.BytesIO(fdata))

        log.debug("STM media saved: %s (%d bytes)", tgz_path, tgz_path.stat().st_size)
        return str(tgz_path)

    def add_fact(
        self,
        text: str,
        confidence: float = 0.7,
        provenance: str = "unknown",
        time_frame: str = "session",
        media_path: Optional[str] = None,
    ) -> str:
        """Append a fact to in-memory store and persist to today's file.

        HLD: "When a newly inferred fact is near-duplicate of an existing STM
        fact, STM may merge them instead of adding a second fact. Merging
        must preserve the more durable ``time_frame``, raise confidence
        conservatively, and retain provenance/metadata needed to understand
        why the fact was inferred."

        Dedup scans **all** of today's facts (no recent-N window) so a weak
        early-session version of a fact cannot block a later more-durable
        version from being promoted to LTM.  When media archives are
        attached to both sides of a merge, both paths are preserved via the
        ``media_paths`` list — earlier media is no longer dropped.

        Returns the id of the affected fact (existing or new).
        """
        self._maybe_roll_day()
        text = text.strip()
        new_tokens = _tokens(text)

        # Near-duplicate check.  Scan today's full fact list (newest-first so
        # the most recent matching ancestor wins when several exist) — HLD's
        # anti-blocking guarantee requires we look beyond the last N facts.
        for existing in reversed(self._facts):
            if _jaccard(new_tokens, _tokens(existing["text"])) < _DEDUP_SIMILARITY:
                continue

            old_conf = existing["confidence"]
            existing["confidence"] = round(min(1.0, old_conf + 0.05), 3)

            # Upgrade time_frame if the new fact is more durable (e.g.
            # session → permanent) so sleep replay doesn't skip it.
            old_tf = existing.get("time_frame", "session")
            if _TF_RANK.get(time_frame, 0) > _TF_RANK.get(old_tf, 0):
                existing["time_frame"] = time_frame
                log.debug(
                    "STM dedup: upgraded time_frame %s→%s for: %s",
                    old_tf, time_frame, existing["text"][:80],
                )

            # Preserve every independent source identity, dropping oldest
            # segments first if the cap is hit (rather than silently
            # discarding the incoming identity).
            existing["provenance"] = _merge_provenance(
                existing.get("provenance", ""), provenance,
            )

            # Preserve media from every contributing fact — single-media
            # facts use the legacy scalar field; multi-media facts use the
            # new ``media_paths`` list.  Both readable via media_paths_of().
            if media_path:
                paths = media_paths_of(existing)
                if media_path not in paths:
                    paths.append(media_path)
                _set_media_paths(existing, paths)

            self._rewrite_file()
            self._journal_event(
                "stm_merge",
                into_id=existing["id"],
                text=existing["text"],
                confidence=existing["confidence"],
                provenance=existing.get("provenance", ""),
                time_frame=existing.get("time_frame", ""),
                media_paths=media_paths_of(existing),
            )
            log.debug(
                "STM dedup: bumped confidence %.2f→%.2f for: %s",
                old_conf, existing["confidence"], existing["text"][:80],
            )
            return existing["id"]

        fact: Dict[str, Any] = {
            "id":         str(uuid.uuid4()),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "confidence": float(max(0.2, min(1.0, confidence))),
            "provenance": provenance,
            "time_frame": time_frame if time_frame in TIME_FRAMES else "session",
            "text":       text,
        }
        if media_path:
            fact["media_path"] = media_path
        self._facts.append(fact)
        self._append_to_file(fact)
        self._journal_event(
            "stm_fact",
            fact_id=fact["id"],
            text=fact["text"],
            confidence=fact["confidence"],
            provenance=fact["provenance"],
            time_frame=fact["time_frame"],
            media_paths=media_paths_of(fact),
        )
        log.debug("STM fact added [%s/%.2f]: %s", time_frame, confidence, text[:80])
        return fact["id"]

    def _journal_event(self, event_type: str, **fields: Any) -> None:
        """Mirror an STM mutation into the event journal, if one is attached."""
        journal = self.journal
        if journal is None:
            return
        try:
            journal.append(event_type, **fields)
        except Exception as exc:  # never let journaling break STM writes
            log.debug("STM: journal append failed (%s): %s", event_type, exc)

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent *limit* facts (newest last)."""
        all_facts = self._prev_day_facts + self._facts
        return list(all_facts[-limit:])

    def get_all_today(self) -> List[Dict[str, Any]]:
        """Return all facts from the current awake cycle (including any
        previous-day carry-over kept for sleep replay)."""
        return list(self._prev_day_facts) + list(self._facts)

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Case-insensitive substring search over all in-memory facts.

        Returns up to *limit* matching facts (newest first) so the API
        mirrors ``MemoryClient.search()`` and LLM-generated streams can
        use STM and LTM interchangeably.
        """
        q = query.lower()
        all_facts = self._prev_day_facts + self._facts
        hits = [f for f in all_facts if q in f.get("text", "").lower()]
        return list(reversed(hits[-limit:]))

    # Alias so generated streams that call search_text also work.
    search_text = search

    def remove_by_ids(self, ids: List[str]) -> int:
        """Remove facts by id. Rewrites today's file only for today's facts;
        previous-day carry-over facts are removed from memory only (their
        on-disk file is not touched).  Media archives referenced by removed
        facts are deleted from disk — including the full media_paths list
        when a fact accumulated multiple archives via dedup merges."""
        id_set = set(ids)
        # Collect the ids actually present and their media paths up front.
        all_facts = self._prev_day_facts + self._facts
        present_ids: List[str] = []
        media_paths: List[str] = []
        for f in all_facts:
            if f["id"] in id_set:
                present_ids.append(f["id"])
                media_paths.extend(media_paths_of(f))
        if not present_ids:
            return 0

        # Journal the removal BEFORE any destructive change (in-memory drop,
        # JSONL rewrite, media delete).  Ordering matters for crash safety:
        #   - crash *after* journaling, *before* destruction → benign: the fact
        #     simply lingers in the cache (reconciliation is additive, never
        #     resurrects with dangling media), and is dropped on next rewrite.
        #   - the reverse order risked destroying cache/media while the journal
        #     still implied the fact existed, so a rebuild would resurrect a
        #     fact whose media was already gone.
        self._journal_event("stm_remove", ids=present_ids)

        before_prev = len(self._prev_day_facts)
        before_today = len(self._facts)
        self._prev_day_facts = [f for f in self._prev_day_facts if f["id"] not in id_set]
        self._facts = [f for f in self._facts if f["id"] not in id_set]
        removed_today = before_today - len(self._facts)
        removed_prev = before_prev - len(self._prev_day_facts)
        removed = removed_today + removed_prev
        if removed_today:
            self._rewrite_file()
        # Clean up media archives (last — losing these on a crash is harmless;
        # the stm_remove event already records the intent).
        for mp in media_paths:
            try:
                os.remove(mp)
            except OSError:
                pass
        log.info("STM: removed %d fact(s) (%d today, %d prev-day, %d media)",
                 removed, removed_today, removed_prev, len(media_paths))
        return removed

    def count(self) -> int:
        return len(self._prev_day_facts) + len(self._facts)

    # ------------------------------------------------------------------
    # STM as a projection of the event journal
    # ------------------------------------------------------------------

    @staticmethod
    def project_from_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstruct the STM fact list by folding journal events.

        This is the canonical definition of STM as a projection of the event
        journal: ``stm_fact`` creates a fact, ``stm_merge`` updates the target
        fact's mutable fields, ``stm_remove`` deletes.  The on-disk JSONL is a
        derived cache of exactly this fold — see :meth:`reconcile_with_journal`.
        """
        facts: Dict[str, Dict[str, Any]] = {}
        for e in events:
            t = e.get("type")
            if t == "stm_fact":
                fid = e.get("fact_id")
                if not fid:
                    continue
                f = {
                    "id": fid,
                    "timestamp": e.get("ts", ""),
                    "text": e.get("text", ""),
                    "confidence": e.get("confidence", 0.7),
                    "provenance": e.get("provenance", ""),
                    "time_frame": e.get("time_frame", "session"),
                }
                mp = [p for p in (e.get("media_paths") or []) if p]
                if mp:
                    _set_media_paths(f, mp)
                facts[fid] = f
            elif t == "stm_merge":
                f = facts.get(e.get("into_id"))
                if f is not None:
                    f["text"] = e.get("text", f["text"])
                    f["provenance"] = e.get("provenance", f["provenance"])
                    f["time_frame"] = e.get("time_frame", f["time_frame"])
                    if e.get("confidence") is not None:
                        f["confidence"] = e.get("confidence")
                    mp = [p for p in (e.get("media_paths") or []) if p]
                    if mp:
                        _set_media_paths(f, mp)
            elif t == "stm_remove":
                for rid in (e.get("ids") or []):
                    facts.pop(rid, None)
        return list(facts.values())

    def reconcile_with_journal(self, events: List[Dict[str, Any]]) -> int:
        """Recover facts the JSONL cache lost, using the journal as truth.

        Additive only: facts present in the journal projection but missing
        from the loaded in-memory store are restored (the common cache-loss or
        truncation failure).  Facts are never removed here — explicit removals
        are captured as ``stm_remove`` events and already folded out by
        :meth:`project_from_events`.  Returns the number recovered."""
        proj = self.project_from_events(events)
        if not proj:
            return 0
        have = {f["id"] for f in self._facts} | {f["id"] for f in self._prev_day_facts}
        recovered = 0
        for pf in proj:
            if pf["id"] not in have:
                self._facts.append(pf)
                recovered += 1
        if recovered:
            self._rewrite_file()
            log.warning("STM: recovered %d fact(s) from journal absent in JSONL cache",
                        recovered)
        return recovered

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _today_path(self) -> Path:
        return self._base / f"{date.today().isoformat()}.jsonl"

    def _load_today(self) -> None:
        path = self._today_path()
        if not path.exists():
            return
        loaded = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact = json.loads(line)
                    if isinstance(fact, dict) and "text" in fact:
                        self._facts.append(fact)
                        loaded += 1
                except json.JSONDecodeError:
                    pass
        log.debug("STM loaded %d facts from %s", loaded, path)

    def _load_previous_day(self) -> None:
        """Load yesterday's JSONL into _prev_day_facts.

        On a cold restart the day after the last awake cycle, _maybe_roll_day
        never ran so _prev_day_facts is empty.  Loading the previous day's file
        ensures sleep replay can still temporally associate STM facts that were
        written before midnight.
        """
        from datetime import timedelta
        yesterday = (self._today - timedelta(days=1)).isoformat()
        path = self._base / f"{yesterday}.jsonl"
        if not path.exists():
            return
        loaded = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    fact = json.loads(line)
                    if isinstance(fact, dict) and "text" in fact:
                        self._prev_day_facts.append(fact)
                        loaded += 1
                except json.JSONDecodeError:
                    pass
        if loaded:
            log.info("STM loaded %d prev-day facts from %s", loaded, path)

    def _append_to_file(self, fact: Dict[str, Any]) -> None:
        try:
            with self._today_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(fact, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.warning("STM: failed to persist fact: %s", exc)

    def _rewrite_file(self) -> None:
        try:
            with self._today_path().open("w", encoding="utf-8") as fh:
                for fact in self._facts:
                    fh.write(json.dumps(fact, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.warning("STM: failed to rewrite file: %s", exc)

    def _maybe_roll_day(self) -> None:
        """If the calendar day has changed, move current facts to the
        previous-day partition and start a fresh list for the new day.
        Previous-day facts are kept in memory so that sleep replay can still
        promote pre-midnight facts from the same awake cycle."""
        today = date.today()
        if today != self._today:
            self._prev_day_facts = self._prev_day_facts + self._facts
            self._facts = []
            self._today = today
            log.info("STM: rolled to new day %s (carried over %d prev-day facts)", today, len(self._prev_day_facts))
