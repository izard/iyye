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


class StmUpdateStream(ProcessingStream):
    """
    Scans all other streams for new activity-log entries, extracts facts
    via LLM, and writes them to brain.stm (ShortTermMemory).

    Can be promoted to conscious: when conscious it logs what was found.
    """

    LLM_INTERVAL = 10  # minimum ticks between LLM calls

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

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Scan all streams for new activity-log entries.
        When enough new content has accumulated, call LLM to extract facts
        and store them in brain.stm.
        """
        stm = getattr(self.brain, "stm", None)
        if stm is None:
            return None

        streams = context.get("streams", self.brain.streams)
        live_names = set()
        self._ticks_since_llm += 1

        # Register any new streams we haven't seen before.
        for s in streams:
            if s is not self and not _should_skip_stream(s.name):
                self._known_streams[s.name] = s
                live_names.add(s.name)

        # Collect fresh entries from all known streams (live + recently retired).
        # This ensures a stream that retires between collection ticks still gets
        # its final activity-log entries drained.
        for name, s in list(self._known_streams.items()):
            log_entries = getattr(s, "activity_log", [])
            seen = self._cursors.get(name, 0)
            fresh = log_entries[seen:]
            if fresh:
                self._buffered_work.setdefault(name, []).extend(fresh)
                self._cursors[name] = seen + len(fresh)
            # Drop retired streams once fully drained so we don't hold
            # references (and memory) indefinitely.
            if name not in live_names and not fresh:
                del self._known_streams[name]

        if not self._buffered_work:
            return None

        # Throttle LLM calls.
        if self._ticks_since_llm < self.LLM_INTERVAL:
            return None

        # Throttle passed — snapshot the buffer and clear it for the next cycle.
        self._ticks_since_llm = 0
        new_work = self._buffered_work
        self._buffered_work = {}

        stream_list = ", ".join(new_work.keys())
        entry_count = sum(len(v) for v in new_work.values())
        self.add_to_log(f"Scanning {len(new_work)} stream(s) [{stream_list}] — {entry_count} new entries")
        self.checkpoint()

        all_facts: List[Dict[str, Any]] = []
        for stream_name, entries in new_work.items():
            facts, consumed = self._extract_facts(stream_name, entries)
            # If _extract_facts only processed a batch of 20, put the rest back
            # into the buffer so they're included in the next LLM call.
            remainder = entries[consumed:]
            if remainder:
                self._buffered_work.setdefault(stream_name, []).extend(remainder)
            if not facts:
                self.add_to_log(
                    f"No facts extracted from {stream_name} "
                    f"({consumed}/{len(entries)} entries consumed)"
                )
            if facts:
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
            self.checkpoint()

        if all_facts:
            summary = f"Stored {len(all_facts)} STM fact(s) from {list(new_work)}"
            self.add_to_log(summary)
            self.add_output(
                {"facts": all_facts, "streams": list(new_work.keys())},
                target="stm",
            )
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

    def _extract_facts(
        self, stream_name: str, entries: List[str]
    ) -> tuple:
        """Call LLM to extract tagged facts from activity-log entries.

        Returns (facts, consumed) where consumed is the number of raw entries
        whose content was sent to the LLM.  The caller advances the cursor by
        exactly this amount, leaving any remaining entries for the next call.
        """
        # Pair each raw entry with its index so we can track the cursor correctly
        # after noise filtering.  Process oldest entries first so nothing is lost.
        meaningful_indexed = [
            (i, e) for i, e in enumerate(entries)
            if not _is_noise(e)
        ]
        if not meaningful_indexed:
            # All noise — safe to advance past the whole batch.
            return [], len(entries)

        # Take the oldest 20 meaningful entries; leave the rest for the next call.
        batch = meaningful_indexed[:20]
        # consumed = raw index of the last entry in the batch + 1
        consumed = batch[-1][0] + 1

        entries_text = "\n".join(e for _, e in batch)
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
        """Force-process all remaining buffered entries to STM.

        Bypasses the LLM_INTERVAL throttle and loops until the buffer is
        fully drained (entries are batched in groups of 20 by _extract_facts,
        so multiple passes may be needed).  Called at sleep start so chat
        entries are in STM before replay promotes facts to LTM.
        """
        for _ in range(50):  # safety cap
            self._ticks_since_llm = self.LLM_INTERVAL
            self.execute({})
            if not self._buffered_work:
                break

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
