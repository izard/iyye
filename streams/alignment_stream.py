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
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import logging

from iyye_base import ProcessingStream
from llm_scheduler import LLMCall, LLMConsumerMixin

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class AlignmentStream(LLMConsumerMixin, ProcessingStream):
    """
    Iterates over processing streams and produces alignment scores via the
    async LLM scheduler (kind="alignment"), with keyword-matching fallback when
    the LLM is unreachable.  The batch scoring call runs on a scheduler worker —
    no bespoke background thread — and the scheduler's per-port priority keeps
    it from contending with conscious chat (which made the old port-sharing
    guard redundant).
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

        self._tick_count = 0
        self._LLM_INTERVAL = 50  # only run LLM scoring every N ticks
        self._cached_results: Dict[str, Dict[str, float]] = {}

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

    # ------------------------------------------------------------------
    # execute — async batch LLM (scheduler) → keyword fallback
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Score all streams.  A single batch LLM call (throttled to every
        _LLM_INTERVAL ticks) is submitted to the async scheduler; between/while
        it runs, cached scores are used for known streams and a keyword
        fallback for new ones.  The call never blocks the main loop.
        """
        self._tick_count += 1
        # Inspect peers through the read-only view contract (immutable snapshots).
        candidates = [
            v for v in self.brain.stream_views()
            if v.name != self.name and not v.in_critical_section
        ]

        # Cooperative checkpoint (never blocks — just checks stop flag).
        try:
            self.checkpoint()
        except StopIteration:
            return {}

        # Apply a finished batch scoring result.
        result = self._llm_poll()
        if result is not None and not result.discarded and result.ok and result.text:
            parsed = self._parse_batch_scores(result.text)
            if parsed:
                self._cached_results.update(parsed)
                log.debug("AlignmentStream: applied batch scores for %d streams",
                          len(parsed))

        # Submit a new batch every _LLM_INTERVAL ticks when idle and not paused.
        if (
            not self._paused
            and not self._llm_busy()
            and self._tick_count % self._LLM_INTERVAL == 1
        ):
            snapshots = [
                {"name": v.name,
                 "activity_log": list(v.recent_activity),
                 "output_history": list(v.recent_outputs)}
                for v in candidates
            ]
            if snapshots:
                self._llm_submit(
                    role="alignment", kind="alignment",
                    call=LLMCall.from_file(
                        "alignment_batch_streams",
                        streams_snapshot=json.dumps(snapshots, indent=2),
                    ),
                    client_kwargs={"no_think": True, "max_tokens": 512},
                )

        # Build the returned scores: cached batch scores + keyword fallback for
        # any stream not yet covered by a batch result.
        results = dict(self._cached_results)
        for v in candidates:
            if v.name not in results:
                results[v.name] = self._keyword_scores(
                    list(v.recent_activity), list(v.recent_outputs),
                )
        self._cached_results = dict(results)

        # Apply scores through the brain (owner), not by mutating peers.
        self.brain.record_alignment(results)
        return results

    def _parse_batch_scores(self, raw: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Parse the batch LLM response ``{stream_name: {goal: score}}``."""
        try:
            text = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(text)
            results: Dict[str, Dict[str, float]] = {}
            for stream_name, score_map in data.items():
                if not isinstance(score_map, dict):
                    continue
                scores = {}
                for goal in self.GOALS:
                    val = score_map.get(goal, 0.0)
                    try:
                        scores[goal] = float(max(0.0, min(1.0, val)))
                    except (TypeError, ValueError):
                        scores[goal] = 0.0
                results[stream_name] = scores
            log.debug("Batch LLM scored %d streams", len(results))
            return results
        except Exception as exc:
            log.warning("AlignmentStream: batch score parse failed: %s", exc)
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
            # StreamView.recent_outputs are strings; older callers may pass
            # {'data': ...} dicts — handle both.
            output_text = " ".join(
                str(o.get('data', '')) if isinstance(o, dict) else str(o)
                for o in outputs
            ).lower()
            if any(kw in output_text for kw in ['actuate', 'send', 'create', 'write', 'execute']):
                scores['agency'] = min(scores['agency'] + 0.15, 0.9)
            if any(kw in output_text for kw in ['response', 'answer', 'hello', 'help', 'user']):
                scores['social'] = min(scores['social'] + 0.1, 0.9)
            if any(kw in output_text for kw in ['query', 'search', 'find', 'discover']):
                scores['curiosity'] = min(scores['curiosity'] + 0.1, 0.9)

        return scores

    # ------------------------------------------------------------------

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
