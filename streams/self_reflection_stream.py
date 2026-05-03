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
import logging
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

from iyye_base import ProcessingStream

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class SelfReflectionStream(ProcessingStream):
    """
    Monitors system state and self-awareness.
    Can be promoted to conscious (unlike other special streams).
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
            self.priority = 7
            self.urgency = 0.6
            self.add_to_log("WARNING: LLM server unreachable")
            self.brain.post_message("llm_management", {
                "action": "restart", "role": "chat",
                "reason": "self_reflection: LLM unreachable",
            })
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

        # When conscious: produce and store an introspective report.
        if self.is_conscious:
            self._produce_conscious_report(state)

        return state

    # ------------------------------------------------------------------
    # System description file (HLD: "It updates the system description
    # markdown file.")
    # ------------------------------------------------------------------

    def _update_system_description(self, state: Dict[str, Any]) -> None:
        """Build system_description.md in memory; write to disk once per day."""
        from iyye_base import PROJECT_ROOT

        iyye_day = state.get('iyye_day', 0)

        # --- Build markdown from gathered state ---
        ts = state.get('timestamp', '')

        hw_lines = (
            f"| CPU      | {state.get('cpu_percent', 0):.1f}% |\n"
            f"| Memory   | {state.get('memory_percent', 0):.1f}% |\n"
            f"| Disk     | {state.get('disk_percent', 0):.1f}% |"
        )

        sensors_md = "\n".join(
            f"- **{s['name']}**: queue={s['queue_size']}, "
            f"healthy={s['healthy']}"
            for s in state.get('io_health', {}).get('sensors', [])
        ) or "_(none)_"

        actuators_md = "\n".join(
            f"- **{a['name']}**: reachable={a['reachable']}"
            for a in state.get('io_health', {}).get('actuators', [])
        ) or "_(none)_"

        llm_info = state.get('llm', {})
        llm_lines = []
        for m in llm_info.get('models', []):
            tag = "UP" if m.get('healthy') else "DOWN"
            roles = ", ".join(m.get('roles', [])) or "-"
            llm_lines.append(
                f"- **{m['name']}** [{tag}] {m.get('size_gb', 0)}GB "
                f"roles=[{roles}]"
            )
        llm_md = "\n".join(llm_lines) or "_(unreachable)_"

        streams_meta = state.get('streams_meta', [])
        streams_md = "\n".join(
            f"- **{s['name']}** (priority={s['priority']}, "
            f"pending={s['pending']}"
            + (", conscious" if s['is_conscious'] else "")
            + ")"
            for s in streams_meta
        ) or "_(none)_"

        conscious_name = next(
            (s['name'] for s in streams_meta if s['is_conscious']),
            "_(none)_",
        )

        pos = state.get('position', {})
        facts_count = pos.get('facts_in_memory', 0)

        md = (
            f"# Iyye System Description\n"
            f"_Generated: {ts} UTC — Iyye day {iyye_day}_\n\n"
            f"## Hardware\n"
            f"| Resource | Usage |\n"
            f"|----------|-------|\n"
            f"{hw_lines}\n\n"
            f"## Sensors\n{sensors_md}\n\n"
            f"## Actuators\n{actuators_md}\n\n"
            f"## LLMs\n{llm_md}\n\n"
            f"## Execution Streams\n"
            f"Active: {len(streams_meta)}  |  Conscious: {conscious_name}\n\n"
            f"{streams_md}\n\n"
            f"## Long-term Memory\n"
            f"Stored facts: {facts_count}\n\n"
            f"## Adenosine\n"
            f"Level: {state.get('adenosine_level', 0):.3f}\n"
        )

        # Always update in-memory cache for other streams.
        try:
            self.brain._system_description_md = md
        except Exception:
            pass

        # Write to disk at most once per iyye_day.
        if iyye_day != self._last_description_day:
            md_path = PROJECT_ROOT / "system_description.md"
            try:
                md_path.write_text(md, encoding="utf-8")
                self._last_description_day = iyye_day
                log.info("Updated system_description.md (day %d)", iyye_day)
            except Exception as exc:
                log.warning("Failed to write system_description.md: %s", exc)

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

        Returns up to 3 recent, non-ephemeral facts.  Returns [] when
        nothing concrete is found, which suppresses stream creation.
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
        # Keep only recent, non-ephemeral/non-session facts.
        recent_ids = {f.get('id') for f in stm.get_recent(50)}
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for r in results:
            rid = r.get('id', id(r))
            tf = r.get('time_frame', '')
            if rid in seen or tf in ('ephemeral', 'session'):
                continue
            if rid not in recent_ids:
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
            self.brain.post_message("stream_factory", {
                "action": "suggest_stream",
                "reason": (f"Sensor '{sensor_name}' has {len(payloads)} unprocessed "
                           f"item(s) with no handling stream"),
                "sensor": sensor_name,
                "goal": "curiosity",
            })
            self.add_to_log(
                f"Flagged stream creation: sensor '{sensor_name}' "
                f"({len(payloads)} items, no handler)"
            )

        # 2. Alignment goal gap: no stream scores above threshold on a goal.
        goal_max: Dict[str, float] = {g: 0.0 for g in self._GOALS}
        for s in streams:
            scores = getattr(s, 'alignment_scores', {})
            for g in self._GOALS:
                val = scores.get(g, 0.0)
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
            self.brain.post_message("stream_factory", {
                "action": "suggest_stream",
                "reason": (f"Goal '{goal}' underserved (best={best:.2f}), "
                           f"evidence: {evidence[0].get('text', '')[:80]}"),
                "goal": goal,
                "sensor": None,
                "evidence_ids": [e.get('id', '') for e in evidence if e.get('id')],
                "evidence_texts": [e['text'][:200] for e in evidence],
            })
            self.add_to_log(
                f"Flagged stream creation: goal '{goal}' underserved "
                f"(best={best:.2f}, {len(evidence)} evidence facts)"
            )

    def _produce_conscious_report(self, state: Dict[str, Any]) -> None:
        """
        Generate a first-person introspective summary via LLM and store
        key findings in long-term memory.  Called only when this stream
        is the conscious stream.
        """
        tick = getattr(self.brain, '_tick_counter', 0)
        if self._conscious_since_tick is None:
            self._conscious_since_tick = tick

        # Rate-limit: only produce a report once every 20 ticks while conscious.
        if (tick - self._conscious_since_tick) % 20 != 0:
            return

        self.add_to_log("Conscious: generating introspective report")
        self.checkpoint()

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

            router = getattr(getattr(self, 'brain', None), 'llm_router', None)
            if router is not None:
                llm = router.get_client(role="reasoning", conscious=True)
            else:
                from llm_client import LLMClient
                llm = LLMClient()
            report = llm.complete_from_file(
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
                recent_facts=recent_facts,
            )
            self.add_to_log(f"Introspective report: {report[:120]}")
            self.add_output(report, target="introspection")

        except Exception as exc:
            log.warning("Self-reflection conscious report failed: %s", exc)
        finally:
            self.checkpoint()

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
