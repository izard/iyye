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

import json
from typing import Dict, Any, Optional, List, TYPE_CHECKING
import logging

from iyye_base import PROJECT_ROOT, ProcessingStream, SleepPhase

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class AttentionStream(ProcessingStream):
    """
    Monitors all active streams and decides which deserves consciousness.
    Never becomes conscious itself
    """

    # Importance is a linear model over per-stream features.  These default
    # weights reproduce the old hand-tuned formula; the sleep feedback loop
    # (#4) nudges them from outcomes so attention learns what consciousness
    # actually pays off (gap: the magic-number control law).
    _DEFAULT_WEIGHTS: Dict[str, float] = {
        "priority":  0.30,
        "pending":   0.50,   # waiting user messages
        "plan":      0.30,   # unexecuted plan steps
        "alignment": 0.25,
        "urgency":   0.15,
        "recency":   0.10,   # hasn't been conscious recently
        "curiosity": 0.05,   # recent fact-bearing output
    }
    _WEIGHTS_PATH = PROJECT_ROOT / "attention_weights.json"

    def __init__(self, brain: "IyyeBrain"):
        super().__init__(name="attention_stream")
        self.brain = brain
        self.priority = 0
        self._can_be_conscious = False
        self._swap_cooldown = 0
        self._weights: Dict[str, float] = self._load_weights()

    # ------------------------------------------------------------------
    # Tunable weights (persisted; learned by the sleep feedback loop)
    # ------------------------------------------------------------------

    def _load_weights(self) -> Dict[str, float]:
        w = dict(self._DEFAULT_WEIGHTS)
        try:
            if self._WEIGHTS_PATH.exists():
                saved = json.loads(self._WEIGHTS_PATH.read_text())
                if isinstance(saved, dict):
                    for k in w:
                        if isinstance(saved.get(k), (int, float)):
                            w[k] = float(saved[k])
        except Exception as exc:
            log.warning("AttentionStream: could not load weights: %s", exc)
        return w

    def _save_weights(self, weights: Dict[str, float]) -> bool:
        try:
            tmp = self._WEIGHTS_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(weights, indent=2))
            tmp.replace(self._WEIGHTS_PATH)
            return True
        except Exception as exc:
            log.warning("AttentionStream: could not save weights: %s", exc)
            return False

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
            # Capture the decision's feature vector + weights so the sleep
            # feedback loop can do credit assignment: which features drove a
            # promotion, and did the resulting consciousness pay off (#4).
            from event_journal import emit
            emit(getattr(self.brain, 'journal', None), 'attention_decision',
                 stream=top_view.name, score=round(top_score, 4),
                 features={k: round(v, 4) for k, v in self._features(top_view).items()},
                 weights={k: round(v, 4) for k, v in self._weights.items()})
            # Return the NAME to promote; the brain resolves it to the live
            # stream.  Attention no longer hands out raw stream objects.
            return {
                'promote': top_view.name,
                'demote': current_view.name if current_view else None,
                'reason': f"score={top_score:.2f}",
            }

        return None

    def _features(self, view) -> Dict[str, float]:
        """Per-stream feature vector, each in [0, 1].  The importance score is
        the dot product of this with the tunable weights — separating the
        signal (features) from the control law (weights) is what lets the sleep
        loop learn the weights from outcomes."""
        current_tick = getattr(self.brain, '_tick_counter', 0)
        align = view.alignment_scores or {}
        return {
            "priority":  min(view.priority, 10) / 10.0,
            "pending":   min(view.pending, 1),                  # any waiting msg
            "plan":      (min(max(view.plan_remaining, 0), 3) / 3.0
                          if view.plan_remaining is not None else 0.0),
            "alignment": (sum(align.values()) / len(align)) if align else 0.0,
            "urgency":   min(view.urgency, 1.0),
            "recency":   1.0 if (view.last_conscious_tick == 0
                                 or current_tick - view.last_conscious_tick > 50) else 0.0,
            "curiosity": 1.0 if any('fact' in o.lower() for o in view.recent_outputs) else 0.0,
        }

    def _calculate_importance(self, view) -> float:
        """Importance = weights · features, clamped to [0, 1]."""
        feats = self._features(view)
        score = sum(self._weights.get(k, 0.0) * v for k, v in feats.items())
        return min(max(score, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Sleep feedback loop (#4): learn the importance weights from outcomes
    # ------------------------------------------------------------------

    # Reward credited to the conscious stream per outcome event during its
    # tenure — what consciousness actually produced.
    _OUTCOME_REWARD = {"stm_fact": 1.0, "ltm_promotion": 1.0, "actuate": 2.0}
    # Reward is a RATE (outcomes per tick of consciousness consumed), not a
    # total: the real data showed a low-output stream that hogs consciousness
    # for ~1200 ticks would otherwise score the same as an efficient one
    # producing the same total in 60 ticks.  This rate gives reward 1.0.
    _TARGET_REWARD_RATE = 0.1
    _TUNE_LR = 0.02            # gradient step toward predicting reward
    _TUNE_REG = 0.01           # pull-back toward defaults (anti-drift)
    # Phase B: enabled after validation — 9 recorded cycles produced tiny,
    # regularized, bounded weight deltas with net drift ≈ 0 (no divergence).
    # Reversible: delete attention_weights.json to restore _DEFAULT_WEIGHTS.
    _TUNE_APPLY = True

    def sleep_phases(self) -> List[SleepPhase]:
        # Order 76: after plan review (75), before cleanup (80).
        return [SleepPhase(
            "attention_tuning", lambda brain: self._sleep_tune_weights(), 76,
        )]

    def _build_tuning_samples(
        self, events: List[Dict[str, Any]],
    ) -> List["tuple"]:
        """From a cycle's journal, pair each promoted stream's decision feature
        vector with the (normalized) reward its consciousness produced.

        Credit assignment uses the ``tick`` events' ``conscious`` field: an
        outcome event is credited to whichever stream was conscious when it
        occurred — i.e. we score the *attention decision*, not the producer."""
        conscious = None
        outcomes: Dict[str, float] = {}    # outcome reward earned while conscious
        ticks: Dict[str, int] = {}         # ticks each stream held consciousness
        features_by_stream: Dict[str, Dict[str, float]] = {}
        for e in events:
            t = e.get("type")
            if t == "tick":
                conscious = e.get("conscious")
                if conscious:
                    ticks[conscious] = ticks.get(conscious, 0) + 1
            elif t == "attention_decision":
                s = e.get("stream")
                if s and isinstance(e.get("features"), dict):
                    features_by_stream[s] = e["features"]
            elif t in self._OUTCOME_REWARD and conscious:
                outcomes[conscious] = (
                    outcomes.get(conscious, 0.0) + self._OUTCOME_REWARD[t])
        samples = []
        for s, feats in features_by_stream.items():
            n = ticks.get(s, 0)
            rate = (outcomes.get(s, 0.0) / n) if n > 0 else 0.0
            reward = min(rate / self._TARGET_REWARD_RATE, 1.0)
            samples.append((feats, reward))
        return samples

    def _tune(self, samples: List["tuple"]) -> "tuple":
        """One bounded, regularized pass of the linear-reward update.

        ``score = w·f``; nudge ``w`` so it predicts the realized reward
        (features that co-occur with payoff gain weight, others lose it), then
        pull back toward the hand-tuned defaults and clamp — so a single odd
        day can't wreck attention.  Returns (proposed_weights, mean_reward)."""
        w = dict(self._weights)
        total = 0.0
        for feats, reward in samples:
            pred = sum(w.get(k, 0.0) * feats.get(k, 0.0) for k in w)
            err = reward - pred
            for k in w:
                w[k] += self._TUNE_LR * err * feats.get(k, 0.0)
            total += reward
        for k in w:
            w[k] = (1 - self._TUNE_REG) * w[k] + self._TUNE_REG * self._DEFAULT_WEIGHTS.get(k, 0.0)
            w[k] = min(max(w[k], 0.0), 1.0)
        return w, (total / len(samples) if samples else 0.0)

    def _sleep_tune_weights(self) -> bool:
        journal = getattr(self.brain, "journal", None)
        # The cycle being consolidated this sleep, captured before replay
        # rotated the journal — not the live _journal_cycle (now the empty
        # next partition).  Falls back to _journal_cycle for safety.
        cid = getattr(self.brain, "_consolidating_cycle",
                      getattr(self.brain, "_journal_cycle", None))
        if journal is None or cid is None:
            return True
        try:
            events = journal.read_cycle(cid)
        except Exception as exc:
            log.debug("Attention tuning: read failed: %s", exc)
            return True
        samples = self._build_tuning_samples(events)
        if not samples:
            return True
        proposed, mean_reward = self._tune(samples)
        from event_journal import emit
        emit(journal, "attention_tuning",
             old={k: round(v, 4) for k, v in self._weights.items()},
             proposed={k: round(v, 4) for k, v in proposed.items()},
             n=len(samples), mean_reward=round(mean_reward, 3),
             applied=self._TUNE_APPLY)
        if self._TUNE_APPLY:
            self._weights = proposed
            self._save_weights(proposed)
            self.add_to_log(
                f"Attention weights tuned from {len(samples)} episode(s), "
                f"mean reward {mean_reward:.2f}")
        return True

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self._swap_cooldown = state.get('swap_cooldown', 0)
