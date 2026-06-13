#!/usr/bin/env python3
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
"""Read-only replay harness over the event journal (Phase 1 of record/replay).

Phase 0 turned the journal into a tape of the cognitive loop's
non-deterministic inputs/outputs (`tick`, `llm_submit`/`llm_result`,
`tool_exec`, `actuate`) plus the memory events. This module reads that tape and:

1. **Reconstructs a turn** end to end (sensor input → stream activity → LLM
   submit/result → tool exec → actuation) for human inspection — the flight
   recorder query that turns "grep four stores" into one command.

2. **Reports anomalies** in a recorded cycle without re-running the brain:
   undelivered results (the prune-while-rephrase signature — an answer
   computed but never sent), orphaned submits (a wedged stream), silent
   failures (an error with no fallback reaching the user), tool errors.

3. **Provides the deterministic-injection primitives** (`ReplayClock`,
   `ReplayScheduler`) that a future full-loop replay (Phase 3) drives a real
   brain with: the scheduler returns recorded results by per-stream submit
   order, gated to the tick the result was recorded as resolving on, so
   multi-tick LLM latency — and the interleavings that produced races — are
   reproduced from the tape rather than from wall-clock timing.

This is read-only and assertion-free by design (Phase 1): it reports what the
recording already contains. Phase 2 declares stream liveness; Phase 3 turns
these reports into replay assertions seeded from real cycles.

CLI:
    python replay.py                # analyze the latest cycle
    python replay.py <cycle_id>     # analyze a specific cycle
    python replay.py <cycle_id> --turn <stream_or_chat_substring>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from event_journal import EventJournal

# Causal event types Phase 0 records (absent in pre-Phase-0 cycles — the
# harness degrades to what exists rather than failing).
_CAUSAL = frozenset(
    {"tick", "llm_submit", "llm_result", "tool_exec", "actuate", "lifecycle"}
)


# ----------------------------------------------------------------------
# Tape: a recorded cycle, indexed for replay and analysis
# ----------------------------------------------------------------------

class CycleTape:
    """A recorded cycle's events with the indexes replay and analysis need."""

    def __init__(self, events: List[Dict[str, Any]]):
        # Stored in append (seq) order — the journal already guarantees this.
        self.events = sorted(events, key=lambda e: e.get("seq", 0))
        self.by_type: Dict[str, List[Dict[str, Any]]] = {}
        # Per-stream submit events in order (drives order-based result match).
        self.submits_by_stream: Dict[str, List[Dict[str, Any]]] = {}
        self.result_by_job: Dict[str, Dict[str, Any]] = {}
        self.ticks: List[Dict[str, Any]] = []
        for e in self.events:
            t = e.get("type")
            self.by_type.setdefault(t, []).append(e)
            if t == "llm_submit":
                self.submits_by_stream.setdefault(e.get("stream", ""), []).append(e)
            elif t == "llm_result":
                self.result_by_job[e.get("job_id", "")] = e
            elif t == "tick":
                self.ticks.append(e)

    # --- construction ---

    @classmethod
    def load(cls, cycle_id: int, journal_dir: str = "journal") -> "CycleTape":
        return cls(EventJournal(base_dir=journal_dir).read_cycle(cycle_id))

    @classmethod
    def from_events(cls, events: List[Dict[str, Any]]) -> "CycleTape":
        return cls(list(events))

    # --- queries ---

    def has_causal_events(self) -> bool:
        return any(t in self.by_type for t in _CAUSAL)

    def tick_at_seq(self, seq: int) -> Optional[int]:
        """The tick number in effect at *seq* (greatest tick event seq <= seq)."""
        cur = None
        for tk in self.ticks:
            if tk.get("seq", 0) <= seq:
                cur = tk.get("tick")
            else:
                break
        return cur

    def result_for_submit(self, submit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return self.result_by_job.get(submit.get("job_id", ""))

    def actuate_stream(self, actuate_event: Dict[str, Any]) -> Optional[str]:
        """Attribute an `actuate` to the stream that caused it.

        `actuate` events carry the device, not the caller. Heuristic (exact
        once Phase 2 emits per-stream lifecycle): the send is logged then the
        device fires, same tick, so the nearest preceding `stream_activity`
        names the caller."""
        seq = actuate_event.get("seq", 0)
        best = None
        for e in self.events:
            if e.get("seq", 0) >= seq:
                break
            if e.get("type") == "stream_activity":
                best = e.get("stream")
        return best


# ----------------------------------------------------------------------
# Replay primitives (deterministic injection — Phase 3 drives a brain with these)
# ----------------------------------------------------------------------

class ReplayClock:
    """Tick clock advanced over the recorded `tick` events."""

    def __init__(self, tape: CycleTape):
        self._ticks = [t.get("tick", 0) for t in tape.ticks]
        self._i = -1

    @property
    def current_tick(self) -> int:
        return self._ticks[self._i] if 0 <= self._i < len(self._ticks) else 0

    def advance(self) -> Optional[int]:
        """Move to the next recorded tick; None when the tape is exhausted."""
        if self._i + 1 >= len(self._ticks):
            return None
        self._i += 1
        return self._ticks[self._i]


class ReplayScheduler:
    """Mock LLMScheduler returning recorded results — the deterministic LLM.

    Drop-in for the real scheduler's main-thread surface
    (submit_call/poll/has_inflight/cancel_stream). Results are matched to
    submits by **per-stream submit order** (robust to prompt-template drift)
    and released only once the replay clock reaches the tick the result was
    recorded as resolving on — so the recorded multi-tick latency, and the
    interleavings it produced, are reproduced from the tape.
    """

    def __init__(self, tape: CycleTape, clock: ReplayClock):
        self._tape = tape
        self._clock = clock
        # stream -> count of submits seen so far in the replayed run.
        self._submit_count: Dict[str, int] = {}
        # stream -> (resolve_tick, result_event) pending release.
        self._pending: Dict[str, tuple] = {}

    def submit_call(self, *, stream_name: str, **_: Any) -> bool:
        if stream_name in self._pending:
            return False  # 1-in-flight rule, same as the real scheduler
        n = self._submit_count.get(stream_name, 0)
        recorded = self._tape.submits_by_stream.get(stream_name, [])
        if n >= len(recorded):
            # The replayed run issued more submits than were recorded — a
            # divergence. Report by rejecting; Phase 3 asserts on this.
            return False
        submit = recorded[n]
        self._submit_count[stream_name] = n + 1
        result = self._tape.result_for_submit(submit)
        if result is None:
            # Recorded as never resolving (orphaned submit) — stays in flight.
            self._pending[stream_name] = (None, None)
            return True
        resolve_tick = self._tape.tick_at_seq(result.get("seq", 0))
        self._pending[stream_name] = (resolve_tick, result)
        return True

    def has_inflight(self, stream_name: str) -> bool:
        return stream_name in self._pending

    def poll(self, stream_name: str) -> Optional[Dict[str, Any]]:
        pending = self._pending.get(stream_name)
        if pending is None:
            return None
        resolve_tick, result = pending
        if result is None:
            return None  # orphaned submit never resolves
        if resolve_tick is not None and self._clock.current_tick < resolve_tick:
            return None  # not yet — preserve recorded latency
        del self._pending[stream_name]
        return result

    def cancel_stream(self, stream_name: str) -> None:
        self._pending.pop(stream_name, None)


# ----------------------------------------------------------------------
# Anomaly analysis (the Phase 1 "report divergence" deliverable)
# ----------------------------------------------------------------------

@dataclass
class Anomaly:
    kind: str
    stream: str
    seq: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] seq={self.seq} stream={self.stream}: {self.detail}"


def analyze(tape: CycleTape) -> List[Anomaly]:
    """Flag the bug-class signatures the recorded tape already contains.

    Heuristic and conservative (Phase 1 reports, it does not assert): each
    finding is a candidate to inspect, not a proven fault. The detectors map
    directly to incidents we have actually hit.
    """
    out: List[Anomaly] = []
    out.extend(_undelivered_results(tape))
    out.extend(_orphan_submits(tape))
    out.extend(_silent_failures(tape))
    out.extend(_tool_errors(tape))
    return out


def _stream_actuated_after(tape: CycleTape, stream: str, seq: int) -> bool:
    for a in tape.by_type.get("actuate", []):
        if a.get("seq", 0) > seq and tape.actuate_stream(a) == stream:
            return True
    return False


def _undelivered_results(tape: CycleTape) -> List[Anomaly]:
    """ok, non-empty, non-discarded result whose stream never actuates after
    it — an answer computed but never sent (the prune-while-rephrase signature)."""
    out = []
    for r in tape.by_type.get("llm_result", []):
        if not r.get("ok") or r.get("discarded"):
            continue
        if not (r.get("text") or "").strip():
            continue
        stream = r.get("stream", "")
        if not _stream_actuated_after(tape, stream, r.get("seq", 0)):
            out.append(Anomaly(
                "undelivered_result", stream, r.get("seq", 0),
                f"result computed (\"{(r.get('text') or '')[:60]}…\") but the "
                f"stream sent nothing afterward",
            ))
    return out


def _orphan_submits(tape: CycleTape) -> List[Anomaly]:
    """submit with no matching result — a stream wedged in flight."""
    out = []
    for s in tape.by_type.get("llm_submit", []):
        if s.get("job_id") not in tape.result_by_job:
            out.append(Anomaly(
                "orphan_submit", s.get("stream", ""), s.get("seq", 0),
                f"submit kind={s.get('kind')} never resolved",
            ))
    return out


def _silent_failures(tape: CycleTape) -> List[Anomaly]:
    """error result (not discarded) with no actuation after — the user got
    nothing, not even an apology."""
    out = []
    for r in tape.by_type.get("llm_result", []):
        if r.get("ok") or r.get("discarded"):
            continue
        stream = r.get("stream", "")
        if not _stream_actuated_after(tape, stream, r.get("seq", 0)):
            out.append(Anomaly(
                "silent_failure", stream, r.get("seq", 0),
                f"error={r.get('error')} with no fallback sent to the user",
            ))
    return out


_TOOL_ERROR_MARKERS = ("[python]", "traceback", "could not", "[exit code")


def _tool_errors(tape: CycleTape) -> List[Anomaly]:
    out = []
    for t in tape.by_type.get("tool_exec", []):
        output = (t.get("output") or "").lower()
        if not output.strip() or any(m in output for m in _TOOL_ERROR_MARKERS):
            out.append(Anomaly(
                "tool_error", t.get("stream", ""), t.get("seq", 0),
                f"{t.get('kind')} output: {(t.get('output') or '(empty)')[:80]}",
            ))
    return out


# ----------------------------------------------------------------------
# Invariant assertions (Phase 3): the bug classes are now properties of the
# tape, so a recorded cycle can be ASSERTED, not just reported. Phase 2's
# declared liveness is what makes these exact rather than heuristic.
# ----------------------------------------------------------------------

@dataclass
class Violation:
    invariant: str
    stream: str
    seq: int
    detail: str

    def __str__(self) -> str:
        return f"[{self.invariant}] seq={self.seq} stream={self.stream}: {self.detail}"


def _stream_appears_after(tape: CycleTape, stream: str, seq: int) -> bool:
    """Any event involving *stream* after *seq* (activity, submit, lifecycle,
    or an actuate attributed to it) — i.e. the stream was still alive."""
    for e in tape.events:
        if e.get("seq", 0) <= seq:
            continue
        if e.get("stream") == stream:
            return True
        if e.get("type") == "actuate" and tape.actuate_stream(e) == stream:
            return True
    return False


def _tick_after(tape: CycleTape, seq: int) -> bool:
    """Did the loop run at least one more tick after *seq*?  Guards the
    cycle-boundary case where a result lands just as the cycle ends."""
    return any(t.get("seq", 0) > seq for t in tape.ticks)


def check_invariants(tape: CycleTape) -> List[Violation]:
    """Exact invariants the cognitive loop must satisfy.  Empty == healthy.

    - ``result_orphaned``: an ok, non-empty LLM result whose stream then
      vanishes without sending or continuing — an answer computed but lost
      (the prune-while-rephrase bug). Requires a later tick so a result at the
      very end of a cycle isn't falsely flagged.
    - ``submit_orphaned``: a submit with no matching result — a wedged stream.
    """
    out: List[Violation] = []
    for r in tape.by_type.get("llm_result", []):
        if not r.get("ok") or r.get("discarded"):
            continue
        if not (r.get("text") or "").strip():
            continue
        seq, stream = r.get("seq", 0), r.get("stream", "")
        if not _tick_after(tape, seq):
            continue  # landed at cycle end — no chance to act, not a fault
        if not _stream_appears_after(tape, stream, seq):
            out.append(Violation(
                "result_orphaned", stream, seq,
                f"ok result (\"{(r.get('text') or '')[:50]}…\") then the stream "
                f"vanished — answer never delivered",
            ))
    for s in tape.by_type.get("llm_submit", []):
        if s.get("job_id") not in tape.result_by_job:
            out.append(Violation(
                "submit_orphaned", s.get("stream", ""), s.get("seq", 0),
                f"submit kind={s.get('kind')} never resolved",
            ))
    return out


def assert_cycle(tape: CycleTape, allow: "frozenset" = frozenset()) -> None:
    """Raise AssertionError if the cycle violates any invariant (except those
    named in *allow*).  This is the regression gate: a recorded cycle that
    trips it contains a real loop bug."""
    violations = [v for v in check_invariants(tape) if v.invariant not in allow]
    if violations:
        raise AssertionError(
            f"{len(violations)} invariant violation(s):\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ----------------------------------------------------------------------
# Turn reconstruction (works on pre-Phase-0 cycles too)
# ----------------------------------------------------------------------

def reconstruct_turn(tape: CycleTape, match: str) -> str:
    """Render the end-to-end trace for events whose stream/sensor/actuator
    contains *match* (case-insensitive), in seq order."""
    m = match.lower()
    lines: List[str] = []
    for e in tape.events:
        t = e.get("type")
        who = (e.get("stream") or e.get("sensor") or e.get("actuator") or "")
        # An actuate names the device, not the caller — attribute it to the
        # stream that triggered it so a stream-filtered trace shows its sends.
        attributed = tape.actuate_stream(e) if t == "actuate" else None
        if m not in who.lower() and (attributed is None or m not in attributed.lower()):
            # Keep events that mention the match in payload/text too.
            blob = str(e.get("payload") or e.get("text") or "")
            if m not in blob.lower():
                continue
        seq, ts = e.get("seq", 0), (e.get("ts") or "")[11:23]
        if t == "sensor_input":
            payload = e.get("payload", {})
            txt = payload.get("text") if isinstance(payload, dict) else str(payload)
            lines.append(f"  {seq:>5} {ts} INPUT  [{who}] {str(txt)[:100]}")
        elif t == "stream_activity":
            lines.append(f"  {seq:>5} {ts} ACT    [{who}] {str(e.get('text',''))[:100]}")
        elif t == "llm_submit":
            lines.append(f"  {seq:>5} {ts} LLM>   [{who}] {e.get('kind')} prompt={e.get('prompt')}")
        elif t == "llm_result":
            tag = "ok" if e.get("ok") else f"ERR:{e.get('error')}"
            lines.append(f"  {seq:>5} {ts} LLM<   [{who}] {tag} {str(e.get('text',''))[:80]}")
        elif t == "tool_exec":
            lines.append(f"  {seq:>5} {ts} TOOL   [{who}] {e.get('kind')} -> {str(e.get('output',''))[:80]}")
        elif t == "actuate":
            lines.append(f"  {seq:>5} {ts} SEND   [{who}] {str(e.get('payload',''))[:100]}")
        elif t == "lifecycle":
            lines.append(f"  {seq:>5} {ts} STATE  [{who}] {e.get('old')} -> {e.get('new')} ({e.get('reason')})")
        elif t == "stm_fact":
            lines.append(f"  {seq:>5} {ts} FACT   [{e.get('time_frame')}] {str(e.get('text',''))[:90]}")
    return "\n".join(lines) if lines else f"(no events matching {match!r})"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _summary(tape: CycleTape) -> str:
    counts = {t: len(v) for t, v in sorted(tape.by_type.items())}
    return f"events={len(tape.events)} types={counts}"


def main(argv: List[str]) -> int:
    journal = EventJournal(base_dir="journal")
    ids = journal.cycle_ids()
    if not ids:
        print("no journal cycles found under ./journal")
        return 1

    args = [a for a in argv if not a.startswith("--")]
    cycle_id = int(args[0]) if args else ids[-1]
    turn_match = None
    if "--turn" in argv:
        i = argv.index("--turn")
        if i + 1 < len(argv):
            turn_match = argv[i + 1]

    tape = CycleTape.load(cycle_id)
    print(f"=== cycle {cycle_id}: {_summary(tape)}")
    if not tape.has_causal_events():
        print("  (pre-Phase-0 cycle — no causal events; anomaly detectors "
              "activate once the app runs with Phase 0 instrumentation)")

    if turn_match is not None:
        print(f"\n--- turn trace matching {turn_match!r} ---")
        print(reconstruct_turn(tape, turn_match))
        return 0

    if "--assert" in argv:
        try:
            assert_cycle(tape)
            print("\n--- invariants: PASS ---")
            return 0
        except AssertionError as exc:
            print(f"\n--- invariants: FAIL ---\n{exc}")
            return 2

    anomalies = analyze(tape)
    print(f"\n--- anomalies: {len(anomalies)} ---")
    for a in anomalies:
        print(f"  {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
