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
# streams/attention_stream.py
#!/usr/bin/env python3
"""
Attention Stream - Selects which subconscious stream becomes conscious.
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
import logging

from iyye_base import ProcessingStream

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class AttentionStream(ProcessingStream):
    """
    Monitors all active streams and decides which deserves consciousness.
    Never becomes conscious itself
    """

    def __init__(self, brain: "IyyeBrain"):
        super().__init__(name="attention_stream")
        self.brain = brain
        self.priority = 0
        self._can_be_conscious = False
        self._swap_cooldown = 0

    # Never becomes conscious
    @property
    def is_conscious(self) -> bool:
        return False

    @is_conscious.setter
    def is_conscious(self, value: bool) -> None:
        pass

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate all streams and select the most important/urgent.

        HLD: "looks at what are all processing streams doing at the moment
        and decides which one is the most important and/or urgent."
        """
        # Inspect peers through the read-only view contract, never raw objects.
        views = self.brain.stream_views()
        if not views:
            return None

        # Cooldown to prevent excessive swapping
        if self._swap_cooldown > 0:
            self._swap_cooldown -= 1
            return None

        # Decay _last_top_score so the promotion bar doesn't stay permanently
        # high after one strong promotion.  Each non-cooldown tick, the bar
        # drops a little, making it progressively easier for a new contender.
        self._last_top_score = max(
            getattr(self, '_last_top_score', 0.5) - 0.05, 0.0
        )

        scored = []
        for view in views:
            if view.name == self.name:
                continue
            if not view.can_be_conscious:
                continue
            if view.in_critical_section:
                log.debug("Skipping %s (in critical section)", view.name)
                continue
            scored.append((view, self._calculate_importance(view)))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        top_view, top_score = scored[0]

        # Identify the current conscious stream by view, not object identity.
        current_view = next((v for v in views if v.is_conscious), None)

        should_promote = False
        if current_view is None:
            should_promote = True
        elif current_view.name != top_view.name:
            if current_view.in_critical_section:
                self.add_to_log(f"Waiting for {current_view.name} to exit critical section")
                return None
            current_score = self._calculate_importance(current_view)
            # Promote if the contender meaningfully outscores the incumbent,
            # or if it has pending user messages (always urgent).
            if top_score > current_score + 0.1 or top_view.pending > 0:
                should_promote = True

        if should_promote:
            self.add_to_log(f"Promoting {top_view.name} (score={top_score:.2f})")
            self._swap_cooldown = 2
            self._last_top_score = top_score
            # Return the NAME to promote; the brain resolves it to the live
            # stream.  Attention no longer hands out raw stream objects.
            return {
                'promote': top_view.name,
                'demote': current_view.name if current_view else None,
                'reason': f"score={top_score:.2f}",
            }

        return None

    def _calculate_importance(self, view) -> float:
        """
        Calculate importance score for a stream view.

        Factors: base priority (30%), pending work (30%), alignment (25%),
        urgency (15%), recency bonus, and a curiosity bonus.
        """
        score = 0.0

        # Base priority weight (30%)
        score += (min(view.priority, 10) / 10.0) * 0.3

        # Pending user messages — direct user input always gets high priority.
        if view.pending > 0:
            score += min(view.pending * 0.5, 0.5)
        elif view.plan_remaining is not None:
            # Planned streams use unexecuted steps as the pending-work proxy.
            # Continuous (subconscious/generated) streams have plan_remaining
            # None, so they get no pending-work bonus.
            score += min(max(view.plan_remaining, 0) * 0.1, 0.3)

        # Alignment to goals (25%)
        if view.alignment_scores:
            avg = sum(view.alignment_scores.values()) / len(view.alignment_scores)
            score += avg * 0.25

        # Urgency (15%)
        score += min(view.urgency, 1.0) * 0.15

        # Bonus for streams that haven't had conscious time recently
        current_tick = getattr(self.brain, '_tick_counter', 0)
        if view.last_conscious_tick == 0 or current_tick - view.last_conscious_tick > 50:
            score += 0.1

        # Curiosity bonus for recent fact-bearing outputs
        if any('fact' in o.lower() for o in view.recent_outputs):
            score += 0.05

        return min(score, 1.0)

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self._swap_cooldown = state.get('swap_cooldown', 0)
