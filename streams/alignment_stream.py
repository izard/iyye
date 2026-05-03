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
# streams/alignment_stream.py
#!/usr/bin/env python3
"""
Alignment Stream - Scores how well each stream aligns to high-level goals.

HLD Goals:
1. Self-preservation
2. Curiosity (getting new facts)
3. Agency (making impact to outer world)
4. Social (strive to being liked)
"""

import json
import re
import threading
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging

from iyye_base import ProcessingStream

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class AlignmentStream(ProcessingStream):
    """
    Iterates over processing streams and produces alignment scores via LLM,
    with keyword-matching fallback when the LLM is unreachable.
    """

    GOALS = [
        "self_preservation",
        "curiosity",
        "agency",
        "social",
    ]

    GOAL_DESCRIPTIONS = {
        "self_preservation": "Self-preservation - protecting system integrity",
        "curiosity": "Curiosity - seeking and acquiring new facts",
        "agency": "Agency - making impact on the outer world",
        "social": "Social - striving to be liked by others",
    }

    def __init__(self, brain: "IyyeBrain"):
        super().__init__(name="alignment_stream")
        self.brain = brain
        self.priority = 0
        self._can_be_conscious = False

        self._llm = None  # lazy-initialised
        self._tick_count = 0
        self._LLM_INTERVAL = 50  # only run LLM scoring every N ticks
        self._cached_results: Dict[str, Dict[str, float]] = {}
        # Non-blocking LLM scoring: a background thread runs the batch call and
        # writes results here when done; the main loop never waits for it.
        self._pending_results: Optional[Dict[str, Dict[str, float]]] = None
        self._score_thread: Optional[threading.Thread] = None

    # Never becomes conscious
    @property
    def is_conscious(self) -> bool:
        return False

    @is_conscious.setter
    def is_conscious(self, value: bool) -> None:
        pass

    # ------------------------------------------------------------------
    # LLM helper
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            router = getattr(getattr(self, 'brain', None), 'llm_router', None)
            if router is not None:
                self._llm = router.get_client(role="alignment", no_think=True, max_tokens=512)
            else:
                from llm_client import LLMClient
                self._llm = LLMClient(no_think=True, max_tokens=512)
        return self._llm

    def _parse_scores(self, text: str) -> Optional[Dict[str, float]]:
        """Extract a JSON scores dict from LLM output, tolerating extra text."""
        # Strip markdown code fences if present
        text = re.sub(r"```[a-z]*\n?", "", text).strip()
        # Find the first {...} block
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not match:
            return None
        try:
            raw = json.loads(match.group())
            scores = {}
            for goal in self.GOALS:
                val = raw.get(goal)
                if isinstance(val, (int, float)):
                    scores[goal] = float(max(0.0, min(1.0, val)))
                else:
                    scores[goal] = 0.0
            return scores
        except (json.JSONDecodeError, KeyError):
            return None

    # ------------------------------------------------------------------
    # execute — try batch LLM → per-stream LLM → keyword fallback
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Checkpoint-aware snapshotting
    # ------------------------------------------------------------------

    def _snapshot_candidates(self, candidates) -> list:
        """Snapshot activity logs on the main thread while streams are paused.

        HLD: alignment stream "stops each [stream] at checkpoint" before
        checking activity logs.  In cooperative multitasking, streams are
        between execute() calls here, so the pause is largely ceremonial
        but ensures correctness if execution ever moves to concurrent threads.
        """
        snapshots = []
        for s in candidates:
            s.request_checkpoint_pause()
            snapshots.append({
                "name": s.name,
                "activity_log": [str(e) for e in list(getattr(s, 'activity_log', []))[-20:]],
                "output_history": [str(o) for o in list(getattr(s, 'output_history', []))[-5:]],
            })
            s.resume_from_checkpoint_pause()
        return snapshots

    # ------------------------------------------------------------------
    # execute — try batch LLM → per-stream LLM → keyword fallback
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Score all streams. Tries one batch LLM call first for efficiency.

        LLM scoring is throttled to every _LLM_INTERVAL ticks to avoid blocking
        the main loop for ~20s on every tick.  Between LLM ticks the cached
        scores are used for known streams and keyword fallback for new ones.
        """
        self._tick_count += 1
        streams = context.get('streams', self.brain.streams)
        candidates = [
            s for s in streams
            if not self._is_self(s) and not getattr(s, '_in_critical_section', False)
        ]

        # Cooperative checkpoint (never blocks — just checks stop flag).
        try:
            self.checkpoint()
        except StopIteration:
            return {}

        # Collect completed background results if the thread just finished.
        if self._pending_results is not None and (
            self._score_thread is None or not self._score_thread.is_alive()
        ):
            self._cached_results.update(self._pending_results)
            self._pending_results = None
            self._score_thread = None
            log.debug("AlignmentStream: background scoring results applied")

        # Throttle: only launch a new LLM thread every _LLM_INTERVAL ticks,
        # and only if no thread is already in flight.
        if (
            self._tick_count % self._LLM_INTERVAL == 1
            and (self._score_thread is None or not self._score_thread.is_alive())
        ):
            # Guard: skip background LLM scoring when the alignment role would
            # be routed to the same port as the top (conscious) model.  llama.cpp
            # typically has a single inference slot; a long-running background
            # alignment call would block that slot and cause 503 "Service
            # Unavailable" for every other stream's request.
            skip_llm = False
            router = getattr(self.brain, 'llm_router', None)
            if router is not None:
                align_model = router.get_model_info("alignment")
                top_model = router._find_top_model()
                if (align_model is not None and top_model is not None
                        and align_model["port"] == top_model["port"]):
                    skip_llm = True
                    log.debug(
                        "AlignmentStream: skipping LLM scoring — alignment "
                        "shares port %d with top model (slot contention)",
                        align_model["port"],
                    )
                # Clear cached LLM client so routing changes are picked up
                # when a dedicated alignment model becomes available later.
                self._llm = None

            if not skip_llm:
                # HLD: "stopping each at checkpoint, and checking their activity
                # log" — snapshot logs on the main thread while streams are paused
                # between execute() calls.  The background LLM thread receives
                # these immutable snapshots instead of live stream objects, which
                # avoids reading mutable state from a concurrent thread.
                log_snapshots = self._snapshot_candidates(candidates)

                def _score_worker(snapshots=log_snapshots) -> None:
                    result = self._batch_llm_score(snapshots)
                    if result is not None:
                        self._pending_results = result

                self._score_thread = threading.Thread(
                    target=_score_worker, name="alignment_llm", daemon=True
                )
                self._score_thread.start()
                log.debug("AlignmentStream: launched background scoring thread (%d stream snapshots)", len(log_snapshots))

        # Always return from cached scores — never wait for the thread.
        results = dict(self._cached_results)

        # Score any stream not yet covered by the batch result — always via
        # keyword matching.  Per-stream LLM fallback was removed: if the batch
        # call failed or was skipped, N × LLM calls would block the main loop
        # for N × ~34s, which is worse than the batch call itself.
        for stream in candidates:
            if stream.name not in results:
                results[stream.name] = self._keyword_scores(
                    list(getattr(stream, 'activity_log', []))[-20:],
                    list(getattr(stream, 'output_history', []))[-5:],
                )

        # Persist results for non-LLM ticks.
        self._cached_results = dict(results)

        for stream in candidates:
            if stream.name in results:
                stream.alignment_scores = results[stream.name]

        return results

    def _batch_llm_score(self, snapshots: list) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Score all streams in a single LLM call using alignment_batch_streams prompt.

        *snapshots* is a list of dicts ``{"name", "activity_log", "output_history"}``
        built on the main thread by ``_snapshot_candidates`` so that this method
        (which runs in a background thread) never touches live stream objects.

        Returns None if LLM is unavailable or the response can't be parsed.
        """
        if not snapshots:
            return {}
        try:
            llm = self._get_llm()
            response = llm.complete_from_file(
                "alignment_batch_streams",
                streams_snapshot=json.dumps(snapshots, indent=2),
            )

            # Parse outer dict {stream_name: {goal: score}}
            text = re.sub(r"```[a-z]*\n?", "", response).strip()
            raw = json.loads(text)

            results = {}
            for stream_name, score_map in raw.items():
                scores = {}
                for goal in self.GOALS:
                    val = score_map.get(goal, 0.0)
                    scores[goal] = float(max(0.0, min(1.0, val)))
                results[stream_name] = scores

            log.debug("Batch LLM scored %d streams", len(results))
            return results

        except Exception as exc:
            log.warning("Batch LLM alignment failed, falling back per-stream: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Keyword fallback (original logic, extracted)
    # ------------------------------------------------------------------

    def _keyword_scores(self, log_entries: list, outputs: list) -> Dict[str, float]:
        scores = {goal: 0.0 for goal in self.GOALS}
        log_text = " ".join(str(e) for e in log_entries).lower()

        preservation_kw = ['error', 'crash', 'fail', 'danger', 'protect',
                           'save', 'backup', 'recover', 'resource', 'memory',
                           'checkpoint', 'safe', 'stable']
        n = sum(1 for kw in preservation_kw if kw in log_text)
        if n:
            scores['self_preservation'] = min(0.9, 0.6 + n * 0.03)

        curiosity_kw = ['learn', 'discover', 'new fact', 'explore', 'query',
                        'search', 'read', 'analyze', 'investigate', 'question',
                        'why', 'how', 'what if']
        n = sum(1 for kw in curiosity_kw if kw in log_text)
        if n:
            scores['curiosity'] = min(0.9, 0.6 + n * 0.05)

        agency_kw = ['send', 'create', 'modify', 'actuate', 'output',
                     'write', 'execute', 'perform', 'action', 'change',
                     'impact', 'effect']
        n = sum(1 for kw in agency_kw if kw in log_text)
        if n:
            scores['agency'] = min(0.9, 0.6 + n * 0.05)

        social_kw = ['user', 'chat', 'help', 'friendly', 'polite',
                     'respond', 'answer', 'greet', 'conversation',
                     'dialogue', 'interact', 'social']
        n = sum(1 for kw in social_kw if kw in log_text)
        if n:
            scores['social'] = min(0.9, 0.6 + n * 0.05)

        if outputs:
            output_text = " ".join(str(o.get('data', '')) for o in outputs).lower()
            if any(kw in output_text for kw in ['actuate', 'send', 'create', 'write', 'execute']):
                scores['agency'] = min(scores['agency'] + 0.15, 0.9)
            if any(kw in output_text for kw in ['response', 'answer', 'hello', 'help', 'user']):
                scores['social'] = min(scores['social'] + 0.1, 0.9)
            if any(kw in output_text for kw in ['query', 'search', 'find', 'discover']):
                scores['curiosity'] = min(scores['curiosity'] + 0.1, 0.9)

        return scores

    # ------------------------------------------------------------------

    def _is_self(self, stream) -> bool:
        return stream is self or (hasattr(stream, 'name') and stream.name == self.name)

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
