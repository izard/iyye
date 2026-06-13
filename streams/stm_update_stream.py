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
# streams/stm_update_stream.py
#!/usr/bin/env python3
"""
Runs every tick, but only calls the LLM when new activity-log entries
have appeared in other streams since the last scan (throttled further to
at most once every LLM_INTERVAL ticks so it never stalls the main loop).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from iyye_base import ProcessingStream
from llm_scheduler import LLMCall, LLMConsumerMixin
from memory_filters import (
    EPHEMERAL_METRIC_RE as _METRIC_RE,
    should_skip_stream as _should_skip_stream,
)

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


def _classify_time_frame(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Override the LLM's time_frame to 'ephemeral' for system metric snapshots."""
    if _METRIC_RE.search(fact.get("text", "")):
        return {**fact, "time_frame": "ephemeral"}
    return fact


class StmUpdateStream(LLMConsumerMixin, ProcessingStream):
    """
    Scans all other streams for new activity-log entries, extracts facts
    via the async LLM scheduler, and writes them to brain.stm.

    Async, one extraction in flight at a time (the scheduler's one-job-per-
    stream rule): each tick collects fresh activity into a buffer, applies any
    completed extraction, and — when idle — submits the next source stream's
    batch.  The blocking LLM call runs on a scheduler worker, never on the main
    loop.  flush() (sleep) still uses a direct synchronous call.

    Can be promoted to conscious: when conscious it logs what was found.
    """

    LLM_INTERVAL = 10  # minimum ticks between extraction submissions

    def __init__(self, brain: "IyyeBrain") -> None:
        super().__init__(name="stm_update")
        self.brain = brain
        self.priority = 2
        self._can_be_conscious = True

        # Per-stream cursor: how many activity_log entries we have already seen.
        self._cursors: Dict[str, int] = {}
        self._ticks_since_llm: int = self.LLM_INTERVAL  # allow first call immediately
        self._llm = None  # lazy
        # Entries collected from live streams but not yet sent to the LLM.
        # Persists across ticks so retiring streams don't lose their log entries.
        self._buffered_work: Dict[str, List[str]] = {}
        # References to every stream we've ever seen, keyed by name.
        # A stream that retires (removes itself from brain.streams) between
        # collection ticks would otherwise lose its final activity-log entries.
        # Keeping a reference here lets us drain those entries on the next tick.
        self._known_streams: Dict[str, Any] = {}
        # The extraction batch currently submitted to the scheduler (its source
        # stream name, the full entry list, and how many entries the batch
        # consumed), or None when idle.  Stashed so a failed/discarded result
        # can be requeued instead of silently lost.
        self._inflight_extraction: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Collect activity, apply finished extractions, submit the next batch.

        Async state machine (one extraction in flight at a time):
          1. every tick — collect fresh activity-log entries into the buffer;
          2. apply a completed extraction result (store facts) if ready;
          3. if an extraction is still running, return;
          4. if idle with buffered work (throttled), submit the next batch.
        The blocking LLM call runs on a scheduler worker, not the main loop.
        """
        stm = getattr(self.brain, "stm", None)
        if stm is None:
            return None

        self._ticks_since_llm += 1
        self._collect(context.get("streams", self.brain.streams))

        # Apply a finished extraction.
        result = self._llm_poll()
        if result is not None:
            return self._on_extraction_result(result, stm)

        # An extraction is still running — collect only this tick.
        if self._llm_busy():
            return None

        # Not busy and no result, but we still hold a stashed batch → its result
        # was dropped (e.g. the wake epoch rotated across a sleep boundary).
        # Requeue it so the entries aren't stranded.
        if self._inflight_extraction is not None:
            ctx = self._inflight_extraction
            self._inflight_extraction = None
            sn = ctx["stream_name"]
            self._buffered_work[sn] = ctx["entries"] + self._buffered_work.get(sn, [])

        # Idle: submit the next batch if there is work and the throttle allows.
        if not self._buffered_work or self._ticks_since_llm < self.LLM_INTERVAL:
            return None
        self._submit_next_batch()
        return None

    def _collect(self, streams) -> None:
        """Drain new activity-log entries from all known streams into the
        buffer.  Keeps a reference to streams that retire between ticks until
        their final entries are drained; uses the absolute cursor so a busy
        stream's entries survive activity_log compaction."""
        live_names = set()
        for s in streams:
            if s is not self and not _should_skip_stream(s.name):
                self._known_streams[s.name] = s
                live_names.add(s.name)
        for name, s in list(self._known_streams.items()):
            seen = self._cursors.get(name, 0)
            getter = getattr(s, "get_log_since", None)
            if callable(getter):
                fresh, lost, new_cursor = getter(seen)
            else:
                log_entries = getattr(s, "activity_log", [])
                fresh, lost, new_cursor = log_entries[seen:], 0, len(log_entries)
            if lost:
                log.warning("StmUpdateStream: %d log entries from %s trimmed "
                            "before extraction (busy stream)", lost, name)
            if fresh:
                self._buffered_work.setdefault(name, []).extend(fresh)
            self._cursors[name] = new_cursor
            if name not in live_names and not fresh:
                del self._known_streams[name]

    def _submit_next_batch(self) -> None:
        """Pick one source stream's buffered entries and submit an extraction
        job.  Entries are stashed (not dropped) so a failed/discarded result is
        requeued rather than lost (see HLD issue #9)."""
        stream_name = next(iter(self._buffered_work))
        entries = self._buffered_work.pop(stream_name)
        entries_text, consumed = self._prepare_batch(entries)
        if entries_text is None:
            # All noise — nothing to extract; the entries are dropped (consumed).
            return
        submitted = self._llm_submit(
            role="stm", kind="stm",
            call=LLMCall.from_file(
                "stm_extract_facts",
                stream_name=stream_name, stream_entries=entries_text,
            ),
            client_kwargs={"no_think": True, "max_tokens": 512},
            # Background fact extraction: not user-facing, latency-insensitive.
            task={"prompt_tokens": 700, "expected_output_tokens": 300,
                  "quality_need": 0.5, "latency_budget_s": 90, "urgency": 0.15},
        )
        if not submitted:
            # Scheduler paused or busy — put the entries back and retry later.
            self._buffered_work[stream_name] = (
                entries + self._buffered_work.get(stream_name, [])
            )
            return
        self._ticks_since_llm = 0
        self._inflight_extraction = {
            "stream_name": stream_name, "entries": entries, "consumed": consumed,
        }
        self.add_to_log(
            f"Submitted STM extraction for {stream_name} "
            f"({consumed}/{len(entries)} entries)"
        )

    def _on_extraction_result(self, result, stm) -> Optional[Dict[str, Any]]:
        """Apply a completed extraction on the main thread: store facts, and
        requeue unconsumed (or failed) entries so nothing is silently lost."""
        ctx = self._inflight_extraction or {}
        self._inflight_extraction = None
        stream_name = ctx.get("stream_name", "?")
        entries = ctx.get("entries", [])
        consumed = ctx.get("consumed", len(entries))

        def _requeue(items):
            if items:
                self._buffered_work[stream_name] = (
                    list(items) + self._buffered_work.get(stream_name, [])
                )

        if result.discarded or not result.ok:
            # Don't lose work: requeue the whole batch (fixes the silent-drop
            # on LLM failure — HLD issue #9).
            _requeue(entries)
            if not result.discarded:
                self.add_to_log(
                    f"STM extraction failed for {stream_name} "
                    f"({result.error}) — requeued {len(entries)} entries"
                )
            return None

        _requeue(entries[consumed:])  # leftover beyond the 20-entry batch
        raw = result.text
        facts = _parse_fact_lines(raw) if raw and raw.strip() else []
        if not facts:
            self.add_to_log(f"No facts extracted from {stream_name}")
            return None

        all_facts: List[Dict[str, Any]] = []
        for f in facts:
            f = _classify_time_frame(f)  # override metric facts to ephemeral
            try:
                fact_id = stm.add_fact(
                    text=f["text"],
                    confidence=float(f.get("confidence", 0.7)),
                    provenance=f.get("provenance", stream_name),
                    time_frame=f.get("time_frame", "session"),
                )
                all_facts.append({**f, "id": fact_id})
                # HLD: adenosine depletes on storing facts to STM.
                adenosine = getattr(self.brain, 'adenosine', None)
                if adenosine is not None:
                    adenosine.drain_activity("stm_write")
            except Exception as exc:
                log.warning("STM add_fact failed: %s", exc)

        if all_facts:
            self.add_to_log(f"Stored {len(all_facts)} STM fact(s) from {stream_name}")
            self.add_output({"facts": all_facts, "streams": [stream_name]}, target="stm")
            for f in all_facts:
                self.add_to_log(
                    f"  [{f.get('time_frame','?')}/{f.get('confidence',0):.2f}]"
                    f" {f['text'][:120]}"
                )
            return {"action": "stm_updated", "facts_added": len(all_facts)}
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            router = getattr(getattr(self, 'brain', None), 'llm_router', None)
            if router is not None:
                self._llm = router.get_client(role="stm", no_think=True, max_tokens=512)
            else:
                from llm_client import LLMClient
                self._llm = LLMClient(no_think=True, max_tokens=512)
        return self._llm

    def _prepare_batch(self, entries: List[str]):
        """Select the oldest <=20 meaningful (non-noise) entries and build the
        prompt text.  Returns ``(entries_text, consumed)``, or
        ``(None, len(entries))`` when every entry is noise (advance past all)."""
        meaningful_indexed = [
            (i, e) for i, e in enumerate(entries) if not _is_noise(e)
        ]
        if not meaningful_indexed:
            return None, len(entries)
        batch = meaningful_indexed[:20]
        consumed = batch[-1][0] + 1   # raw index of last batched entry + 1
        entries_text = "\n".join(e for _, e in batch)
        return entries_text, consumed

    def _extract_facts(
        self, stream_name: str, entries: List[str]
    ) -> tuple:
        """Synchronous extraction — used by flush() during sleep only.

        During sleep there is no conscious stream to starve and the loop is
        otherwise idle, so a direct blocking call is fine (and the scheduler is
        paused).  The awake path goes through the async scheduler instead.

        Returns (facts, consumed); consumed is how many raw entries were sent.
        """
        entries_text, consumed = self._prepare_batch(entries)
        if entries_text is None:
            return [], consumed
        try:
            llm = self._get_llm()
            raw = llm.complete_from_file(
                "stm_extract_facts",
                stream_name=stream_name,
                stream_entries=entries_text,
            )
            if not raw or not raw.strip():
                log.warning("StmUpdateStream: LLM returned empty response for %s", stream_name)
                return [], consumed
            facts = _parse_fact_lines(raw)
            if not facts:
                log.warning("StmUpdateStream: no JSON facts parsed from LLM output for %s: %r",
                            stream_name, raw[:200])
            return facts, consumed
        except Exception as exc:
            log.warning("StmUpdateStream LLM call failed for %s: %s", stream_name, exc)
            return [], consumed

    def flush(self) -> None:
        """Force-process all buffered entries to STM synchronously.

        Called at sleep start so chat entries reach STM before replay promotes
        facts to LTM.  Uses direct synchronous LLM calls rather than the async
        scheduler: the scheduler is paused during wind-down, and during sleep
        there is no conscious stream to starve.
        """
        stm = getattr(self.brain, "stm", None)
        if stm is None:
            return
        # Fold any batch that was in flight at wind-down back into the buffer.
        if self._inflight_extraction:
            ctx = self._inflight_extraction
            self._inflight_extraction = None
            sn = ctx["stream_name"]
            self._buffered_work[sn] = ctx["entries"] + self._buffered_work.get(sn, [])
        # Final collection pass to catch entries written since the last tick.
        self._collect(getattr(self.brain, "streams", []))
        for _ in range(50):  # safety cap
            if not self._buffered_work:
                break
            stream_name = next(iter(self._buffered_work))
            entries = self._buffered_work.pop(stream_name)
            facts, consumed = self._extract_facts(stream_name, entries)
            remainder = entries[consumed:]
            if remainder:
                self._buffered_work.setdefault(stream_name, []).extend(remainder)
            for f in facts:
                f = _classify_time_frame(f)
                try:
                    stm.add_fact(
                        text=f["text"],
                        confidence=float(f.get("confidence", 0.7)),
                        provenance=f.get("provenance", stream_name),
                        time_frame=f.get("time_frame", "session"),
                    )
                except Exception as exc:
                    log.warning("STM add_fact failed during flush: %s", exc)

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

_NOISE_PATTERNS = re.compile(
    r"checkpoint|tick \d|DEBUG|cpu_percent|memory_percent"
    r"|adenosine_stream|attention_stream|alignment_stream"
    r"|stream_factory|stm_update"
    # Operational chatter from LLM-generated streams:
    r"|(?:hardware concern|proactive.*(?:assertions?|greetings?|follow-up)|"
    r"concern message)\s+(?:was|were)\s+sent"
    r"|interactions were analyzed"
    r"|capability was discovered"
    r"|auto-generated by StreamFactory"
    r"|Sent proactive",
    re.IGNORECASE,
)


def _is_noise(line: str) -> bool:
    """Return True for lines that carry no factual content."""
    stripped = re.sub(r"^\[.*?\]\s*", "", line)  # strip timestamp prefix
    if len(stripped) < 10:
        return True
    return bool(_NOISE_PATTERNS.search(stripped))


def _parse_fact_lines(raw: str) -> List[Dict[str, Any]]:
    """Parse JSONL fact output from the LLM, tolerating extra text and markdown fences."""
    facts = []
    for line in raw.splitlines():
        line = line.strip()
        # Skip blank lines, markdown fence delimiters, and non-JSON lines
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        # Find the first '{' in the line (handles e.g. "1. {..." prefixes)
        brace = line.find("{")
        if brace == -1:
            continue
        candidate = line[brace:]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "text" in obj:
                facts.append(obj)
        except json.JSONDecodeError:
            pass
    return facts
