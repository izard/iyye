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
# streams/adenosine_stream.py
#!/usr/bin/env python3
"""
Adenosine Stream - Manages tiredness metric as a subconscious stream.

HLD: "subconscious stream that owns 'tiredness metric' called 'adenosine',
which depletes during awake and triggers state transition to winding down"
"""

from typing import Dict, Any, TYPE_CHECKING
import logging

from iyye_base import ProcessingStream

if TYPE_CHECKING:
    from main_loop import MindState, IyyeBrain

log = logging.getLogger("Iyye")


class AdenosineStream(ProcessingStream):
    """
    Subconscious stream that manages the tiredness/energy metric.
    Can never become conscious (HLD requirement).
    """

    MAX = 1.0
    # HLD: "It also depletes with time too, but extremely slowly."
    DRAIN_PER_TICK = 0.0002
    THRESHOLD = 0.15
    REFILL_PER_SLEEP_TICK = 0.05

    # HLD: "depletes on storing facts to STM, changing consciousness focus,
    # heavy actions like starting/stopping LLMs and streams making important
    # choices."
    ACTIVITY_COSTS: Dict[str, float] = {
        "stm_write":            0.005,   # storing a fact to STM
        "consciousness_switch": 0.015,   # changing conscious stream focus
        "llm_start":            0.025,   # starting an LLM (heavy)
        "llm_stop":             0.020,   # stopping an LLM (heavy)
        "stream_create":        0.010,   # creating a new processing stream
        "important_choice":     0.010,   # stream making a significant decision
    }

    def __init__(self, brain: "IyyeBrain"):
        super().__init__(name="adenosine_stream")
        self.brain = brain
        self.priority = 0
        self._can_be_conscious = False
        self._level: float = 0.0

    @property
    def level(self) -> float:
        return self._level

    @level.setter
    def level(self, v: float) -> None:
        new_val = max(0.0, min(v, self.MAX))
        if abs(new_val - self._level) > 1e-9:
            self.add_to_log(f"Adenosine updated to {new_val:.3f}")
        self._level = new_val

    # Never becomes conscious
    @property
    def is_conscious(self) -> bool:
        return False

    @is_conscious.setter
    def is_conscious(self, value: bool) -> None:
        pass

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage adenosine based on brain state.

        HLD: "depletes during awake and triggers state transition to
        winding down state when approaches 0."
        """
        from main_loop import MindState

        brain_state = getattr(self.brain, 'state', None)

        if brain_state == MindState.AWAKE:
            self._drain()
        elif brain_state == MindState.ASLEEP:
            self.replenish()
        elif brain_state == MindState.WAKING_UP:
            pass
        elif brain_state == MindState.WINDING_DOWN:
            pass

        return {
            'level': self._level,
            'depleted': self.is_depleted(),
            'percentage': self._level / self.MAX * 100,
            'state': brain_state.name if brain_state else 'UNKNOWN',
        }

    def drain_activity(self, activity: str) -> None:
        """Drain adenosine for a specific meaningful activity.

        Called from outside the stream (via brain.adenosine) when an HLD-
        specified event occurs.  Unknown activity types are silently ignored.
        """
        cost = self.ACTIVITY_COSTS.get(activity, 0.0)
        if cost > 0:
            self.level = self._level - cost
            log.debug("Adenosine drained %.4f for %s (now %.3f)",
                      cost, activity, self._level)

    def _drain(self, amount: float = None) -> None:
        """Drain adenosine during awake activity (slow passive tick)."""
        amount = amount or self.DRAIN_PER_TICK
        self.level = self._level - amount
        self.checkpoint()

    def replenish(self, amount: float = None) -> None:
        """Replenish adenosine during sleep."""
        amount = amount or self.REFILL_PER_SLEEP_TICK
        self.level = min(self.MAX, self._level + amount)
        self.checkpoint()

    def is_depleted(self) -> bool:
        """Check if adenosine is below threshold."""
        return self._level <= self.THRESHOLD

    def request_stop(self) -> None:
        """Adenosine stream ignores stop requests — it must keep running through
        winding-down to track state, and replenish() during sleep must not raise
        StopIteration.  can_stop_safely() already returns True so the winding-down
        wait check passes without needing to actually stop the inner stream."""
        pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state['level'] = self._level
        state['urgency'] = self.urgency
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self._level = state.get('level', self.MAX)
        self.urgency = state.get('urgency', 0.0)
