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
        streams = context.get('streams', self.brain.streams)

        if not streams:
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
        for stream in streams:
            if self._is_self(stream):
                continue
            if not getattr(stream, '_can_be_conscious', True):
                continue
            # Skip streams in critical section
            if getattr(stream, '_in_critical_section', False):
                log.debug("Skipping %s (in critical section)", stream.name)
                continue

            score = self._calculate_importance(stream, context)
            scored.append((stream, score))

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)

        current_conscious = context.get('current_conscious')
        top_stream, top_score = scored[0]

        # Only promote if significantly more important or current is None
        should_promote = False
        if current_conscious is None:
            should_promote = True
        elif current_conscious != top_stream:
            # Check if current stream is in critical section
            if getattr(current_conscious, '_in_critical_section', False):
                self.add_to_log(f"Waiting for {current_conscious.name} to exit critical section")
                return None

            # Score the current conscious stream to compare directly.
            current_score = self._calculate_importance(current_conscious, context)

            # Promote if the contender meaningfully outscores the incumbent,
            # or if the contender has pending user messages (always urgent).
            has_pending = len(getattr(top_stream, '_pending_messages', [])) > 0
            if top_score > current_score + 0.1 or has_pending:
                should_promote = True

        if should_promote:
            self.add_to_log(f"Promoting {top_stream.name} (score={top_score:.2f})")
            self._swap_cooldown = 2
            self._last_top_score = top_score

            return {
                'promote': top_stream,
                'demote': current_conscious,
                'reason': f"score={top_score:.2f}"
            }

        return None

    def _is_self(self, stream) -> bool:
        """Check if stream is self."""
        return stream is self or (hasattr(stream, 'name') and stream.name == self.name)

    def _calculate_importance(self, stream, context: Dict[str, Any]) -> float:
        """
        Calculate importance score for a stream.

        Factors:
        - Base priority (30%)
        - Pending inputs (30%)
        - Alignment to goals (25%)
        - Urgency (15%)
        - Time since last conscious (bonus)
        """
        score = 0.0

        # Base priority weight (30%)
        priority = getattr(stream, 'priority', 1)
        score += (min(priority, 10) / 10.0) * 0.3

        # Pending user messages — direct user input always gets high priority.
        # _pending_messages is set by UserChatStream for unprocessed messages.
        pending_msgs = len(getattr(stream, '_pending_messages', []))
        if pending_msgs > 0:
            score += min(pending_msgs * 0.5, 0.5)
        else:
            # For planned streams use unexecuted steps as the pending-work proxy.
            # Continuous streams (subconscious, LLM-generated) don't use
            # input/output pairing, so they get no pending-work bonus —
            # otherwise the input-output gap grows forever and inflates priority.
            plan_steps = getattr(stream, '_plan_steps', None)
            if plan_steps is not None:
                remaining = len(plan_steps) - getattr(stream, '_current_step', 0)
                score += min(max(remaining, 0) * 0.1, 0.3)

        # Alignment to goals (25%) - HLD: alignment scores from alignment_stream
        alignment = getattr(stream, 'alignment_scores', {})
        if alignment:
            avg_alignment = sum(alignment.values()) / len(alignment)
            score += avg_alignment * 0.25

        # Urgency (15%) - NEW: from stream.urgency attribute
        urgency = getattr(stream, 'urgency', 0.0)
        score += min(urgency, 1.0) * 0.15

        # Bonus for streams that haven't had conscious time recently
        last_conscious = getattr(stream, '_last_conscious_tick', 0)
        current_tick = getattr(self.brain, '_tick_counter', 0)
        if last_conscious == 0 or current_tick - last_conscious > 50:
            score += 0.1

        # NEW: Bonus for streams with recent high-value outputs
        outputs = getattr(stream, 'output_history', [])
        if outputs:
            recent = outputs[-3:]
            if any('fact' in str(o.get('data', '')).lower() for o in recent):
                score += 0.05  # Curiosity bonus

        return min(score, 1.0)

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self._swap_cooldown = state.get('swap_cooldown', 0)
