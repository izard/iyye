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
# streams/self_reflection_stream.py
#!/usr/bin/env python3
"""
Self-Reflection Stream - Monitors self position and system resources.

"""

import os
import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, TYPE_CHECKING

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from iyye_base import ProcessingStream, SleepPhase
from llm_scheduler import LLMCall, LLMConsumerMixin
from memory_filters import should_skip_stream as _should_skip_provenance

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class SelfReflectionStream(LLMConsumerMixin, ProcessingStream):
    """
    Monitors system state and self-awareness.
    Can be promoted to conscious (unlike other special streams).

    The conscious introspective report is generated through the async
    scheduler so producing it never blocks the main loop.
    """

    def __init__(self, brain: "IyyeBrain"):
        self._is_conscious = False
        self._conscious_since_tick: Optional[int] = None
        self._urgency: float = 0.0  # must exist before super().__init__
        super().__init__(name="self_reflection")
        self.brain = brain
        self.priority = 3
        self._can_be_conscious = True

        self._last_check: Optional[Dict[str, Any]] = None
        self._gather_tick: int = 0
        self._GATHER_INTERVAL: int = 10

        # Stream-creation suggestion cooldown — tracks (sensor/goal) keys
        # recently posted to stream_factory so we don't spam every gather tick.
        self._suggest_cooldown: Dict[str, int] = {}  # key → tick when posted
        self._SUGGEST_COOLDOWN_TICKS: int = 150

        # HLD: "It updates the system description markdown file."
        # Write to disk at most once per iyye_day; keep in-memory copy current.
        self._last_description_day: int = -1

    @property
    def is_conscious(self) -> bool:
        return self._is_conscious

    @is_conscious.setter
    def is_conscious(self, value: bool) -> None:
        if value and not self._is_conscious:
            self._conscious_since_tick = getattr(self, 'brain', None) and getattr(self.brain, '_tick_counter', 0)
        elif not value:
            self._conscious_since_tick = None
        self._is_conscious = value

    @property
    def urgency(self) -> float:
        return self._urgency

    @urgency.setter
    def urgency(self, value: float) -> None:
        self._urgency = value
        if value > 0:
            self.priority = min(10, self.priority + 1)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check system resources and self state.

        HLD: "monitors Iyye position in space, system resources used,
        introspection about all running streams, IO and LLMs."
        When conscious, produces a first-person introspective report via LLM
        and stores key findings in long-term memory.
        """
        self._gather_tick += 1
        if self._gather_tick % self._GATHER_INTERVAL == 1 or self._last_check is None:
            state = self._gather_system_state()
            self._last_check = state
        else:
            state = self._last_check

        stream_summary = ", ".join(
            f"{s['name']}(p={s['priority']},u={s['urgency']:.2f}"
            + (",conscious" if s['is_conscious'] else "")
            + (f",pending={s['pending']}" if s['pending'] else "")
            + ")"
            for s in state.get('streams_meta', [])
        )
        llm_info = state['llm']
        active_llms = [m['name'] for m in llm_info.get('models', []) if m.get('healthy')]
        llm_summary = (
            f"{len(active_llms)} active ({', '.join(active_llms)})"
            if active_llms else llm_info['status']
        )
        self.add_to_log(
            f"[Day {state['iyye_day']}] "
            f"CPU={state['cpu_percent']:.1f}%, "
            f"Mem={state['memory_percent']:.1f}%, "
            f"Adenosine={state['adenosine_level']:.2f} | "
            f"LLMs: {llm_summary} | "
            f"Streams: {stream_summary or 'none'}"
        )

        # Alert if resources critical — boost priority and urgency
        if state['memory_percent'] > 90:
            self.priority = 10
            self.urgency = 0.9
            self.add_to_log("CRITICAL: Memory usage above 90%")
        elif state['cpu_percent'] > 95:
            self.priority = 10
            self.urgency = 0.9
            self.add_to_log("CRITICAL: CPU usage above 95%")
        elif state['llm']['status'] == 'unreachable':
            # Consult the authoritative health owner (llm_management) rather
            # than acting on this bare snapshot: if a restart is already in
            # flight, observe and wait — re-requesting one per tick is the
            # restart flapping.  Only escalate when truly down-and-unattended.
            if self._chat_llm_state() == "restoring":
                self.priority = 5
                self.urgency = 0.3
                self.add_to_log("LLM restarting (handled by llm_management)")
            else:
                self.priority = 7
                self.urgency = 0.6
                self.add_to_log("WARNING: LLM server unreachable")
                from messaging import Messages
                self.brain.post_message("llm_management", Messages.restart_llm(
                    role="chat", reason="self_reflection: LLM unreachable",
                ))
        elif state['adenosine_level'] < 0.2:
            self.priority = 8
            self.urgency = 0.7
            self.add_to_log("WARNING: Low adenosine level")
        else:
            self.priority = 3
            self.urgency = 0.0

        self._last_check = state
        # Cache on brain for UserChatStream and other consumers.
        try:
            self.brain._self_reflection_snapshot = state
        except Exception:
            pass

        # HLD: "It updates the system description markdown file."
        # Refresh in memory every gather tick; flush to disk once per iyye_day.
        if self._gather_tick % self._GATHER_INTERVAL == 1:
            self._update_system_description(state)

        # HLD: "flag new streams for creation" — check for coverage gaps
        # on every gather tick (when state is fresh).
        # Pass sensors_data from context: the live queues have already been
        # drained by pop_all() in run_once(), so we must look at the popped
        # payloads rather than the (now-empty) queue lengths.
        if self._gather_tick % self._GATHER_INTERVAL == 1:
            self._flag_stream_creation_opportunities(
                state, context.get('sensors_data', {}),
            )

        # Async conscious introspective report: apply a finished one, and —
        # when conscious, due, and idle — submit a new one (never blocks).
        self._pump_conscious_report(state)

        return state

    # ------------------------------------------------------------------
    # System description file (HLD: "It updates the system description
    # markdown file.")
    # ------------------------------------------------------------------

    def perform_system_check(self) -> Dict[str, Any]:
        """Run the asleep system-check and (re)publish system_description.md.

        HLD assigns the system description to this stream; the brain's sleep
        scheduler calls this hook so the *content* is owned here rather than in
        the orchestrator.  Delegates to the single producer in
        ``system_description`` (shared with the brain's first-sleep bootstrap)."""
        from system_description import run_system_check
        return run_system_check(self.brain)

    def _update_system_description(self, state: Dict[str, Any]) -> None:
        """Translate gathered state into canonical shape and publish.

        Rendering, cache update, and on-disk write are all handled by
        :func:`system_description.publish_system_description` so the brain's
        sleep-time writer and this awake-time writer cannot diverge.  Disk
        write is content-deduplicated by the publisher — the old once-per-day
        throttle is no longer needed.
        """
        from system_description import publish_system_description

        streams_meta = state.get('streams_meta', []) or []
        canonical_streams = [
            {'name': s.get('name'), 'priority': s.get('priority', 0),
             'is_conscious': bool(s.get('is_conscious')),
             'pending': s.get('pending')}
            for s in streams_meta
        ]
        conscious_name = next(
            (s['name'] for s in streams_meta if s.get('is_conscious')),
            None,
        )

        io_health = state.get('io_health') or {}
        canonical_sensors = [
            {'name': s['name'], 'queue_size': s.get('queue_size', 0),
             'healthy': s.get('healthy')}
            for s in io_health.get('sensors', [])
        ]
        canonical_actuators = [
            {'name': a['name'], 'reachable': a.get('reachable')}
            for a in io_health.get('actuators', [])
        ]

        llm_info = state.get('llm') or {}
        canonical_llms: Optional[list] = [
            {'name':    m.get('name', '?'),
             'healthy': bool(m.get('healthy', False)),
             'size_gb': m.get('size_gb', 0),
             'roles':   m.get('roles', []) or []}
            for m in llm_info.get('models', [])
        ]
        # Empty list = "unreachable"; None signal stays reserved for "no
        # information at all" (e.g. brain reading a missing llm-active.json).

        pos = state.get('position', {})
        adenosine_max = 1.0
        if hasattr(self.brain, 'adenosine') and hasattr(self.brain.adenosine, 'MAX'):
            adenosine_max = self.brain.adenosine.MAX

        canonical = {
            'timestamp':        state.get('timestamp'),
            'iyye_day':         state.get('iyye_day'),
            'hardware': {
                'cpu_percent':    state.get('cpu_percent', 0),
                'memory_percent': state.get('memory_percent', 0),
                'disk_percent':   state.get('disk_percent', 0),
            },
            'sensors':          canonical_sensors,
            'actuators':        canonical_actuators,
            'llms':             canonical_llms,
            'streams':          canonical_streams,
            'conscious_stream': conscious_name,
            'memory_facts':     pos.get('facts_in_memory', 0),
            'adenosine':        state.get('adenosine_level', 0.0),
            'adenosine_max':    adenosine_max,
        }
        publish_system_description(self.brain, canonical)
        # Track day for the existing _last_description_day consumers (logging
        # cadence elsewhere).  Disk-write dedup is handled by the publisher,
        # so this no longer gates writes.
        self._last_description_day = state.get('iyye_day', self._last_description_day)

    # Substrings identifying chat/user-facing sensors (same heuristic as StreamFactory).
    _CHAT_KEYWORDS = ('chat', 'telegram', 'microphone', 'whisper', 'message')
    _SYSTEM_KEYWORDS = ('cpu', 'gpu', 'memory', 'disk', 'hardware', '_hw_')

    # Alignment goals as defined in AlignmentStream.
    _GOALS = ('self_preservation', 'curiosity', 'agency', 'social')

    # STM search queries per goal — used to find concrete evidence that
    # justifies creating a new goal stream.
    _GOAL_EVIDENCE_QUERIES: Dict[str, List[str]] = {
        'agency': ['pending action', 'user request', 'task', 'should do', 'queued'],
        'social': ['user said', 'conversation', 'message from', 'chat with'],
        'curiosity': ['question', 'unknown', 'new data', 'discovered'],
        'self_preservation': ['error', 'resource low', 'critical', 'warning', 'unhealthy'],
    }

    def _find_goal_evidence(self, goal: str) -> List[Dict[str, Any]]:
        """Search STM for concrete facts that justify creating a goal stream.

        Returns up to 3 recent, non-ephemeral facts whose provenance is a
        first-party stream (not an LLM-generated or planned stream).
        Returns [] when nothing concrete is found, which suppresses stream
        creation and prevents a self-reinforcing loop where generated-stream
        output becomes evidence for creating more generated streams.
        """
        stm = getattr(self.brain, 'stm', None)
        if stm is None:
            return []
        queries = self._GOAL_EVIDENCE_QUERIES.get(goal, [goal])
        results: List[Dict[str, Any]] = []
        for q in queries:
            try:
                results.extend(stm.search(q, limit=3))
            except Exception:
                pass
        # Keep only recent, non-ephemeral/non-session facts from
        # first-party streams.  Facts from generated/planned streams are
        # excluded to prevent a feedback loop where generated output
        # seeds new goal suggestions.
        recent_ids = {f.get('id') for f in stm.get_recent(50)}
        # Collect names of codegen streams (have _source_file) whose
        # LLM-chosen name might not match standard skip patterns.
        codegen_names = {
            s.name for s in getattr(self.brain, 'streams', [])
            if getattr(s, '_source_file', None)
        }
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for r in results:
            rid = r.get('id', id(r))
            tf = r.get('time_frame', '')
            if rid in seen or tf in ('ephemeral', 'session'):
                continue
            if rid not in recent_ids:
                continue
            prov = r.get('provenance', '')
            if prov and (_should_skip_provenance(prov) or prov in codegen_names):
                continue
            seen.add(rid)
            unique.append(r)
        return unique[:3]

    def _flag_stream_creation_opportunities(
        self,
        state: Dict[str, Any],
        sensors_data: Dict[str, List],
    ) -> None:
        """
        HLD: self-reflection "can flag new streams for creation."

        Check for coverage gaps and post suggestions to stream_factory
        via the mailbox system.  Called every GATHER_INTERVAL ticks.

        *sensors_data* is the dict of payloads already popped from queues
        by run_once() this tick.  The live queues are empty at this point,
        so we must use this snapshot to know which sensors had data.
        """
        streams = getattr(self.brain, 'streams', [])
        stream_names_lower = {s.name.lower() for s in streams}
        tick = self._gather_tick

        # Expire old cooldown entries.
        expired = [k for k, t in self._suggest_cooldown.items()
                   if tick - t > self._SUGGEST_COOLDOWN_TICKS]
        for k in expired:
            del self._suggest_cooldown[k]

        # 1. Sensor coverage gap: non-chat sensors with data this tick but no
        #    matching stream.  Uses sensors_data (already-popped payloads)
        #    because the live queues have been drained by pop_all().
        for sensor_name, payloads in sensors_data.items():
            if not payloads:
                continue
            sensor_lower = sensor_name.lower()
            # Skip chat sensors (handled by StreamFactory._create_streams_from_inputs)
            if any(kw in sensor_lower for kw in self._CHAT_KEYWORDS):
                continue
            # Check if stream_factory already created a stream for this sensor.
            factory = next(
                (s for s in streams if s.name == 'stream_factory'), None
            )
            if factory is not None and hasattr(factory, '_is_sensor_covered'):
                if factory._is_sensor_covered(sensor_lower):
                    continue
            key = f"sensor:{sensor_name}"
            if key in self._suggest_cooldown:
                continue
            self._suggest_cooldown[key] = tick
            from messaging import Messages
            self.brain.post_message("stream_factory", Messages.suggest_stream(
                reason=(f"Sensor '{sensor_name}' has {len(payloads)} unprocessed "
                        f"item(s) with no handling stream"),
                sensor=sensor_name,
                goal="curiosity",
            ))
            self.add_to_log(
                f"Flagged stream creation: sensor '{sensor_name}' "
                f"({len(payloads)} items, no handler)"
            )

        # 2. Alignment goal gap: no stream scores above threshold on a goal.
        # Read peer alignment through the view contract, not raw objects.
        goal_max: Dict[str, float] = {g: 0.0 for g in self._GOALS}
        for v in self.brain.stream_views():
            for g in self._GOALS:
                val = v.alignment_scores.get(g, 0.0)
                if val > goal_max[g]:
                    goal_max[g] = val

        for goal, best in goal_max.items():
            if best >= 0.3:
                continue
            key = f"goal:{goal}"
            if key in self._suggest_cooldown:
                continue
            # Require grounded evidence — "agency is low" alone is not
            # enough to justify creating another planned stream.
            evidence = self._find_goal_evidence(goal)
            if not evidence:
                continue
            self._suggest_cooldown[key] = tick
            from messaging import Messages
            self.brain.post_message("stream_factory", Messages.suggest_stream(
                reason=(f"Goal '{goal}' underserved (best={best:.2f}), "
                        f"evidence: {evidence[0].get('text', '')[:80]}"),
                goal=goal,
                sensor=None,
                evidence_ids=[e.get('id', '') for e in evidence if e.get('id')],
                evidence_texts=[e['text'][:200] for e in evidence],
            ))
            self.add_to_log(
                f"Flagged stream creation: goal '{goal}' underserved "
                f"(best={best:.2f}, {len(evidence)} evidence facts)"
            )
            # A goal that stays underserved across several suggestion rounds
            # is a recurring desire, not a one-tick gap — escalate it to a
            # long term plan (HLD: planner creates plans from self-reflection
            # coverage gaps).  The planner dedups repeat proposals by
            # fingerprint, so over-firing here is harmless.
            counts = getattr(self, '_goal_gap_counts', None)
            if counts is None:
                counts = self._goal_gap_counts = {}
            counts[goal] = counts.get(goal, 0) + 1
            if counts[goal] >= 3:
                counts[goal] = 0
                self.brain.post_message("planner", Messages.plan_propose(
                    goal=(f"Persistently address underserved goal "
                          f"'{goal}': {evidence[0].get('text', '')[:120]}"),
                    source="self_reflection",
                    alignment_weights={goal: 1.0},
                ))
                self.add_to_log(
                    f"Escalated recurring goal gap '{goal}' to a long term "
                    f"plan proposal"
                )

        # 3. Promise gap (HLD: coverage gaps include "promises with no
        #    matching delivery").  Promise/delivery facts use the canonical
        #    phrasing convention from stm_extract_facts.md.
        self._flag_unfulfilled_promises(tick)

    # Give an in-flight turn time to deliver on its own before nagging —
    # a promise is only "unkept" once it has aged past normal turn latency.
    _PROMISE_MIN_AGE_S = 600

    def _flag_unfulfilled_promises(self, tick: int) -> None:
        """Match today's "Iyye promised X: ..." STM facts against
        "Iyye delivered to X: ..." facts; an aged unmatched promise becomes a
        chat follow-up routed via the factory back into the conversation the
        promise was made in (its stream name travels as fact provenance).
        """
        stm = getattr(self.brain, 'stm', None)
        if stm is None:
            return
        try:
            facts = stm.get_all_today()
        except Exception as exc:
            log.debug("Promise sweep: STM read failed: %s", exc)
            return
        promises: List[Dict[str, Any]] = []
        deliveries: List[Dict[str, Any]] = []
        for f in facts:
            t = (f.get('text') or '').strip().lower()
            if t.startswith('iyye promised'):
                promises.append(f)
            elif t.startswith('iyye delivered to'):
                deliveries.append(f)
        if not promises:
            return
        now = datetime.now(timezone.utc)
        for p in promises:
            key = f"promise:{p.get('id', '')}"
            if key in self._suggest_cooldown:
                continue
            try:
                ts = datetime.fromisoformat(p.get('timestamp', ''))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if (now - ts).total_seconds() < self._PROMISE_MIN_AGE_S:
                    continue
            except ValueError:
                pass
            contact = self._promise_contact(p.get('text', ''))
            if contact and any(
                contact.lower() in (d.get('text') or '').lower()
                and (d.get('timestamp') or '') > (p.get('timestamp') or '')
                for d in deliveries
            ):
                continue  # delivered after the promise — kept
            # Route back to the originating conversation: promise facts carry
            # the chat stream name as provenance (merges may concatenate).
            provenance = p.get('provenance') or ''
            chat_seg = next(
                (seg.strip() for seg in provenance.split('|')
                 if seg.strip().startswith('chat_')),
                None,
            )
            if chat_seg is None:
                log.debug("Promise sweep: no chat provenance on %r — skipped",
                          p.get('text', '')[:80])
                continue
            self._suggest_cooldown[key] = tick
            from messaging import Messages
            follow_up = (
                f"[internal follow-up — not a user message] You promised "
                f"{contact or 'the user'}: \"{p.get('text', '')}\" at "
                f"{p.get('timestamp', '')} and no delivery has been recorded. "
                f"Deliver the promised result NOW using your tools, or "
                f"apologize briefly and explain why you cannot. "
                f"Do not promise again."
            )
            self.brain.post_message("stream_factory", Messages.chat_follow_up(
                stream_name=chat_seg, text=follow_up, contact=contact,
            ))
            self.add_to_log(
                f"Flagged unkept promise to {contact or '?'} — "
                f"follow-up via '{chat_seg}'"
            )

    @staticmethod
    def _promise_contact(text: str) -> str:
        """Extract <contact> from the canonical 'Iyye promised <contact>: …'."""
        m = re.match(r"(?i)iyye promised\s+([^:]{1,60}):", text.strip())
        return m.group(1).strip() if m else ""

    # ------------------------------------------------------------------
    # Promise backstop (dreaming): catch commitments the STM extractor's
    # phrasing missed, so the awake promise sweep can still repair them.
    # ------------------------------------------------------------------

    # First-person future-action commitments ("I'll send you …", "let me check …").
    _PROMISE_INTENT_RE = re.compile(
        r"\b(i['’]?ll|i\s+will|i['’]?m\s+going\s+to|let\s+me)\b[^.!?]{0,60}?\b"
        r"(get|send|fetch|grab|look(?:\s+up|\s+into)?|check|find|research|remind|"
        r"update|tell|let\s+you\s+know|get\s+back|reach\s+out|message|email|"
        r"notify|figure\s+out|work\s+on|put\s+together|draft|prepare|schedule|"
        r"set\s+up|follow\s+up|dig\s+into)\b",
        re.IGNORECASE)

    # Prompt self-improvement (#6): fold per-version outcomes from the journal
    # and let the registry promote/roll back a trialled prompt rewrite.  The
    # fold + select are side-effect-light (no-ops until a trial exists);
    # proposing a new candidate (which changes what serves traffic) is gated
    # behind _PROMPT_TRIAL_APPLY.  Enabled after the shadow fold established
    # baselines: alignment_batch_streams succeeds only ~0.556 over 72 samples —
    # a well-sampled rewrite target — and trials are bounded by validate_candidate
    # plus select()'s margin-or-rollback, so a worse rewrite is auto-reverted.
    _PROMPT_TRIAL_APPLY = True

    def sleep_phases(self) -> List[SleepPhase]:
        # Order 65: before replay (70) consolidates/clears STM, so the day's
        # promise facts are still present for the dedup check.
        return [
            SleepPhase("promise_backstop",
                       lambda brain: self._sleep_promise_backstop(), 65),
            # Order 77: after attention tuning (76), before cleanup (80).
            SleepPhase("prompt_tuning",
                       lambda brain: self._sleep_prompt_tuning(), 77),
        ]

    def _sleep_prompt_tuning(self) -> bool:
        """Attribute this cycle's LLM outcomes to the prompt version that served
        each job, accumulate per-version reward in the registry, and evaluate
        any running prompt trial (promote on a margin win, roll back if it
        fails to beat the baseline).

        Reward is a robust, prompt-agnostic success signal: a job whose result
        landed ``ok`` and was not discarded scored the prompt's job as good;
        a failed/discarded result (the "Sorry, I couldn't generate a response"
        class of incident) scores it bad.  Exact attribution uses the
        ``prompt_version`` journaled on the matching ``llm_submit``."""
        journal = getattr(self.brain, "journal", None)
        # The cycle consolidated this sleep, captured before replay rotated the
        # journal — see IyyeBrain._enter_asleep.  Reading the live cycle_id here
        # (post-rotation) would fold the empty next partition.
        cycle = getattr(self.brain, "_consolidating_cycle", None)
        if cycle is None and journal is not None:
            cycle = getattr(journal, "cycle_id", None)
        if journal is None or cycle is None:
            return True
        try:
            from prompt_registry import get_registry, fold_outcomes
            events = journal.read_cycle(
                cycle, types=frozenset({"llm_submit", "llm_result"}))
        except Exception as exc:
            log.debug("Prompt tuning: read failed: %s", exc)
            return True

        reg = get_registry()
        counts = fold_outcomes(reg, events)

        from event_journal import emit
        for name, c in counts.items():
            decision = reg.select(name) if reg.status(name).get("trial") else None
            mean = c["sum"] / c["n"] if c["n"] else 0.0
            emit(journal, "prompt_tuning", name=name,
                 version=reg.active_version_id(name), n=int(c["n"]),
                 mean_reward=round(mean, 3), decision=decision)
            if decision:
                self.add_to_log(f"Prompt '{name}': {decision}")

        # Act phase (gated): when no trial is running, propose one rewrite of
        # the worst eligible prompt and trial it.  Shadow-first — disabled until
        # the per-version reward signal above is validated over real cycles.
        if self._PROMPT_TRIAL_APPLY:
            self._propose_prompt_candidate(reg, journal)
        return True

    def _propose_prompt_candidate(self, reg, journal) -> None:
        """Pick the worst-performing eligible prompt, ask a quality LLM to
        rewrite it, validate the rewrite, and start a trial.  Best-effort: any
        failure (no eligible prompt, no client, invalid rewrite) is a no-op."""
        from prompt_registry import (select_prompt_to_improve, validate_candidate)
        from event_journal import emit
        name = select_prompt_to_improve(reg, reg.tracked_names())
        if not name:
            return
        base = reg.status(name).get("versions", {}).get("base", {})
        n = base.get("n", 0)
        success_rate = (base.get("sum_reward", 0.0) / n) if n else 0.0
        try:
            from llm_client import _load_prompt
            current = _load_prompt(name)              # the serving base content
            template = _load_prompt("prompt_rewrite")  # the meta-prompt
        except Exception as exc:
            log.debug("Prompt rewrite: load failed: %s", exc)
            return
        import re as _re
        placeholders = ", ".join(sorted("{%s}" % p for p in
                                        _re.findall(r"{(\w+)}", current))) or "(none)"
        # Literal-token substitution (NOT str.format): the embedded prompt keeps
        # its own {placeholders} verbatim for the validator to match against.
        filled = (template
                  .replace("<<prompt_name>>", name)
                  .replace("<<success_rate>>", f"{success_rate:.0%}")
                  .replace("<<placeholders>>", placeholders)
                  .replace("<<current_prompt>>", current))
        client = self._improve_client()
        if client is None:
            return
        try:
            raw = client.complete(filled)
        except Exception as exc:
            log.debug("Prompt rewrite: LLM call failed: %s", exc)
            return
        candidate = self._strip_fences(raw)
        ok, reason = validate_candidate(current, candidate)
        emit(journal, "prompt_proposed", name=name,
             success_rate=round(success_rate, 3), accepted=ok, reason=reason)
        if not ok:
            self.add_to_log(f"Prompt rewrite for '{name}' rejected: {reason}")
            return
        vid = reg.start_trial(name, candidate)
        if vid:
            self.add_to_log(
                f"Trialling rewritten prompt '{name}' as {vid} "
                f"(baseline success {success_rate:.0%})")

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Drop a leading/trailing markdown code fence the model may have added."""
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("\n", 1)[1] if "\n" in t else ""
            if t.rstrip().endswith("```"):
                t = t.rstrip()[:-3]
        return t.strip()

    def _improve_client(self):
        """A quality-biased synchronous client for prompt rewriting, routed
        through the LLM manager when available, else a bare client."""
        router = getattr(self.brain, "llm_router", None)
        if router is not None:
            try:
                return router.get_client(role="reasoning", task={
                    "quality_need": 0.8, "latency_budget_s": 60, "urgency": 0.1})
            except Exception as exc:
                log.debug("Prompt rewrite: router client failed (%s)", exc)
        try:
            from llm_client import LLMClient
            return LLMClient()
        except Exception as exc:
            log.debug("Prompt rewrite: no client (%s)", exc)
            return None

    def _sleep_promise_backstop(self) -> bool:
        """Scan this cycle's outward chat replies for commitment language and,
        for any commitment not already captured as a promise fact (the
        extractor phrased it differently), record one — so the awake promise
        sweep repairs it.  Deduped against today's promise facts to avoid
        double follow-ups."""
        journal = getattr(self.brain, "journal", None)
        stm = getattr(self.brain, "stm", None)
        if journal is None or stm is None:
            return True
        cycle = getattr(journal, "cycle_id", None)
        if cycle is None:
            return True
        try:
            today = stm.get_all_today()
            events = journal.read_cycle(cycle, types=frozenset({"stream_activity"}))
        except Exception as exc:
            log.debug("Promise backstop: read failed: %s", exc)
            return True
        promise_toks = [self._toks(f.get("text", "")) for f in today
                        if f.get("text", "").lower().startswith("iyye promised")]
        last_sender: Dict[str, str] = {}
        added = 0
        for e in events:
            stream = e.get("stream", "") or ""
            text = e.get("text", "") or ""
            m = re.match(r"USER \(([^)]+)\):", text)
            if m:
                last_sender[stream] = m.group(1).strip()
                continue
            if not (stream.startswith("chat_") and text.startswith("IYYE:")):
                continue
            reply = text[len("IYYE:"):].strip()
            if not self._PROMISE_INTENT_RE.search(reply):
                continue
            rt = self._toks(reply)
            if any(self._tok_overlap(rt, pt) >= 0.5 for pt in promise_toks):
                continue  # already captured by extraction
            contact = last_sender.get(stream, "the user")
            try:
                # provenance = the chat stream name so the sweep routes the
                # follow-up back to that conversation.
                stm.add_fact(f"Iyye promised {contact}: {reply[:160]}",
                             confidence=0.6, provenance=stream, time_frame="today")
                promise_toks.append(rt)
                added += 1
            except Exception as exc:
                log.debug("Promise backstop: add_fact failed: %s", exc)
        if added:
            self.add_to_log(
                f"Promise backstop: recovered {added} missed commitment(s)")
        return True

    @staticmethod
    def _toks(text: str) -> frozenset:
        return frozenset(w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                         if len(w) > 2)

    @staticmethod
    def _tok_overlap(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _pump_conscious_report(self, state: Dict[str, Any]) -> None:
        """Apply a finished introspective report, and — when conscious, due,
        and idle — submit a new one through the async scheduler."""
        # Apply a completed report (may land after a demotion — that's fine).
        result = self._llm_poll()
        if result is not None:
            if not result.discarded and result.ok and result.text:
                self.add_to_log(f"Introspective report: {result.text[:120]}")
                self.add_output(result.text, target="introspection")
            return
        # A report is still being generated — wait.
        if self._llm_busy():
            return
        if not self.is_conscious:
            return

        tick = getattr(self.brain, '_tick_counter', 0)
        if self._conscious_since_tick is None:
            self._conscious_since_tick = tick
        # Rate-limit: only produce a report once every 20 ticks while conscious.
        if (tick - self._conscious_since_tick) % 20 != 0:
            return
        self._submit_conscious_report(state)

    def _submit_conscious_report(self, state: Dict[str, Any]) -> None:
        """Build the introspection prompt and submit it (reasoning, conscious)."""
        self.add_to_log("Conscious: generating introspective report")
        try:
            self.checkpoint()
        except StopIteration:
            return

        try:
            llm_info = state['llm']
            models = llm_info.get('models', [])
            if models:
                lines = []
                for m in models:
                    tag = "UP" if m.get('healthy') else "DOWN"
                    roles = ", ".join(m.get('roles', [])) or "—"
                    prefill = m.get('prefill_tps', 0)
                    decode = m.get('decode_tps', 0)
                    perf = (f" prefill={prefill}t/s decode={decode}t/s"
                            if prefill or decode else "")
                    lines.append(
                        f"  {m['name']}: [{tag}] {m.get('size_gb',0)}GB"
                        f"{perf}"
                        f" roles=[{roles}] reqs={m.get('requests_total',0)}"
                        f" avg_lat={m.get('avg_latency_s',0):.1f}s"
                    )
                llm_text = "\n".join(lines)
            else:
                llm_text = "unreachable"

            sensor_lines = "\n".join(
                f"  {s['name']}: queue={s['queue_size']}, "
                f"healthy={s['healthy']}, mcp_ok={s['mcp_initialized']}"
                for s in state.get('io_health', {}).get('sensors', [])
            ) or "  (none)"

            actuator_lines = "\n".join(
                f"  {a['name']}: reachable={a['reachable']}"
                for a in state.get('io_health', {}).get('actuators', [])
            ) or "  (none)"

            stream_lines = "\n".join(
                f"  {s['name']}: priority={s['priority']}, pending={s['pending']}"
                + (" [CONSCIOUS]" if s['is_conscious'] else "")
                for s in state.get('streams_meta', [])
            ) or "  (none)"

            recent_facts = self._get_recent_facts_text()

            # Cognitive position (HLD: self-reflection "monitors Iyye
            # position") — gathered in state['position'] but previously never
            # passed to the prompt, leaving "what are you focused on?"
            # unanswerable.
            pos = state.get('position', {}) or {}
            position_text = (
                f"focused on: {pos.get('conscious_stream') or '(none)'}, "
                f"active streams: {pos.get('streams_active', 0)}, "
                f"facts known: {pos.get('facts_in_memory', 0)}"
            )

            plan_store = getattr(self.brain, 'plan_store', None)
            plan_lines = "(none)"
            if plan_store is not None:
                try:
                    summaries = [p.summary_line()
                                 for p in plan_store.all_plans()][:5]
                    if summaries:
                        plan_lines = "\n".join(f"  {s}" for s in summaries)
                except Exception:
                    pass

            self._llm_submit(
                role="reasoning", kind="reflection", conscious=True,
                call=LLMCall.from_file(
                    "self_reflection_conscious",
                    timestamp=state['timestamp'],
                    iyye_day=str(state['iyye_day']),
                    hardware=(
                        f"CPU={state['cpu_percent']:.1f}%, "
                        f"mem={state['memory_percent']:.1f}%, "
                        f"disk={state['disk_percent']:.1f}%"
                    ),
                    adenosine=f"{state['adenosine_level']:.3f}",
                    brain_state=state['current_state'],
                    llm_status=llm_text,
                    sensors=sensor_lines,
                    actuators=actuator_lines,
                    streams=stream_lines,
                    position=position_text,
                    plans=plan_lines,
                    recent_facts=recent_facts,
                ),
            )
        except Exception as exc:
            log.warning("Self-reflection report build failed: %s", exc)

    def _gather_system_state(self) -> Dict[str, Any]:
        """Gather comprehensive system state."""
        state = {
            'timestamp': self._get_timestamp(),
            'iyye_day': getattr(self.brain, 'iyye_day', 0),
            'adenosine_level': self._get_adenosine_level(),
            'current_state': self._get_brain_state_name(),
            'active_streams': len(getattr(self.brain, 'streams', [])),
            'conscious_stream': None,
            'sensor_queues': {},
        }

        # Safe sensor queue access
        if hasattr(self.brain, 'sensors'):
            state['sensor_queues'] = {k: len(v) for k, v in self.brain.sensors.items()}

        # Hardware metrics
        if HAS_PSUTIL:
            try:
                state.update({
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'available_memory_gb': psutil.virtual_memory().available / (1024**3),
                })
            except Exception as e:
                log.warning("Failed to gather hardware metrics: %s", e)
                state.update(self._get_default_hardware_metrics())
        else:
            state.update(self._get_default_hardware_metrics())

        # Self position in "space" - HLD requirement
        state['position'] = self._get_position_info()

        # Meta-level view of all currently running streams
        state['streams_meta'] = self._get_streams_meta()

        # LLM substrate health (HLD: "introspection about LLMs")
        state['llm'] = self._get_llm_health()

        # IO health — sensor and actuator status (HLD: "introspection about IO")
        state['io_health'] = self._get_io_health()

        return state

    def _get_adenosine_level(self) -> float:
        """Safely get adenosine level."""
        if hasattr(self.brain, '_adenosine_stream') and self.brain._adenosine_stream:
            return self.brain._adenosine_stream.level
        elif hasattr(self.brain, 'adenosine'):
            return self.brain.adenosine.level
        return 1.0

    def _get_brain_state_name(self) -> str:
        """Safely get brain state name."""
        if hasattr(self.brain, 'state'):
            return self.brain.state.name
        return 'UNKNOWN'

    def _get_default_hardware_metrics(self) -> Dict[str, float]:
        """Return default hardware metrics when psutil unavailable."""
        return {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'disk_percent': 0.0,
            'available_memory_gb': 0.0,
        }

    def _get_position_info(self) -> Dict[str, Any]:
        """
        Get position info for self-awareness.
        HLD: "monitors self position in space"
        """
        conscious_name = None
        if hasattr(self.brain, '_current_conscious') and self.brain._current_conscious:
            conscious_name = getattr(self.brain._current_conscious, 'name', None)

        facts_count = 0
        if hasattr(self.brain, 'memory') and hasattr(self.brain.memory, 'count'):
            try:
                facts_count = self.brain.memory.count()
            except Exception:
                pass

        return {
            'streams_active': len(getattr(self.brain, 'streams', [])),
            'conscious_stream': conscious_name,
            'facts_in_memory': facts_count,
            'iyye_day': getattr(self.brain, 'iyye_day', 0),
            'is_conscious': self.is_conscious,
        }

    def _get_streams_meta(self) -> list:
        """
        Collect a meta-level snapshot of every currently registered stream.

        For each stream, captures:
          - name, priority, urgency
          - is_conscious   — whether it is the current focused stream
          - can_be_conscious — whether it is eligible to be promoted
          - pending        — unprocessed work count:
                             * UserChatStream: len(_pending_messages)
                             * PlannedContinuationStream: remaining plan steps
                             * others: max(inputs - outputs, 0)
          - alignment      — dict of goal→score from AlignmentStream (if available)
          - ticks_since_conscious — how long since last conscious promotion
        """
        streams = getattr(self.brain, 'streams', [])
        conscious = getattr(self.brain, '_current_conscious', None)
        tick = getattr(self.brain, '_tick_counter', 0)
        meta = []
        for s in streams:
            # Pending work
            pending_msgs = getattr(s, '_pending_messages', None)
            if pending_msgs is not None:
                pending = len(pending_msgs)
            else:
                plan_steps = getattr(s, '_plan_steps', None)
                if plan_steps is not None:
                    pending = max(len(plan_steps) - getattr(s, '_current_step', 0), 0)
                else:
                    # Continuous streams (subconscious, LLM-generated) don't use
                    # input/output pairing — report 0 to avoid zombie inflation.
                    pending = 0

            last_conscious = getattr(s, '_last_conscious_tick', 0)
            ticks_since = tick - last_conscious if last_conscious else None

            meta.append({
                'name':               s.name,
                'priority':           getattr(s, 'priority', 0),
                'urgency':            round(float(getattr(s, 'urgency', 0.0)), 3),
                'is_conscious':       s is conscious,
                'can_be_conscious':   bool(getattr(s, '_can_be_conscious', False)),
                'pending':            pending,
                'alignment':          dict(getattr(s, 'alignment_scores', {})),
                'ticks_since_conscious': ticks_since,
            })
        return meta

    def _chat_llm_state(self) -> str:
        """Authoritative chat-model state from the health owner (llm_management),
        or "down" if it isn't reachable — so self-reflection defers the restart
        decision to the owner instead of racing the health check."""
        mgmt = next(
            (s for s in getattr(self.brain, 'streams', [])
             if hasattr(s, 'chat_llm_state')),
            None,
        )
        try:
            return mgmt.chat_llm_state() if mgmt is not None else "down"
        except Exception:
            return "down"

    def _get_llm_health(self) -> Dict[str, Any]:
        """Return LLM health for all registered models.

        Pulls the full status list from LlmManagementStream.get_all_status()
        so self-reflection can report which models are available and active.
        """
        mgmt = next(
            (s for s in getattr(self.brain, 'streams', [])
             if hasattr(s, 'get_all_status')),
            None,
        )
        all_status = mgmt.get_all_status() if mgmt else []

        if not all_status:
            # Fall back to the single-model cache for backward compat
            cached = getattr(self.brain, '_llm_status', None)
            if cached:
                healthy = cached.get('healthy', False)
                return {
                    'status': 'ok' if healthy else 'unreachable',
                    'models': [{
                        'name': cached.get('model', '?'),
                        'healthy': healthy,
                        'roles': [],
                        'size_gb': 0,
                    }],
                }
            return {'status': 'unreachable', 'models': []}

        active = [m for m in all_status if m.get('healthy')]
        return {
            'status': 'ok' if active else 'unreachable',
            'models': [
                {
                    'name': m['name'],
                    'healthy': m.get('healthy', False),
                    'roles': m.get('roles', []),
                    'size_gb': m.get('size_gb', 0),
                    'parameters_b': m.get('parameters_b', 0),
                    'prefill_tps': m.get('prefill_tps', 0),
                    'decode_tps': m.get('decode_tps', 0),
                    'port': m.get('port'),
                    'requests_total': m.get('requests_total', 0),
                    'avg_latency_s': m.get('avg_latency_s', 0),
                }
                for m in all_status
            ],
        }

    def _get_io_health(self) -> Dict[str, Any]:
        """Sensor and actuator health. HLD: 'introspection about IO'."""
        sensors = []
        for name, q in getattr(self.brain, 'sensors', {}).items():
            mcp = getattr(q, 'mcp_client', None)
            sensors.append({
                'name': name,
                'queue_size': len(q) if hasattr(q, '__len__') else 0,
                'healthy': not getattr(q, '_startup_failed', False),
                'mcp_initialized': getattr(mcp, '_initialized', None),
            })
        actuators = []
        for name, a in getattr(self.brain, 'actuators', {}).items():
            # Lightweight reachability: check for a known error flag or just True
            reachable = not getattr(a, '_startup_failed', False)
            actuators.append({'name': name, 'reachable': reachable})
        return {'sensors': sensors, 'actuators': actuators}

    def _get_recent_facts_text(self) -> str:
        """Fetch last 5 facts from long-term memory as readable text."""
        if not hasattr(self.brain, 'memory'):
            return "(none)"
        try:
            facts = self.brain.memory.get_recent_facts(limit=5)
            if not facts:
                return "(none)"
            return "\n".join(
                f"  - [{f.get('confidence', 0):.2f}] {str(f.get('text', ''))[:100]}"
                for f in facts
            )
        except Exception:
            return "(unavailable)"

    def _get_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
        self.urgency = state.get('urgency', 0.0)
