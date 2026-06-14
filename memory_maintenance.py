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
"""Memory maintenance — the *forgetting* half of the memory pipeline.

Everything else in the pipeline accretes: dedup-on-insert, confidence bumps,
STM→LTM promotion.  Nothing down-weights, prunes, or reconciles a fact once it
is in LTM, so a long run accumulates dead weight — stale ``today``/``recent``
facts at full confidence, process bookkeeping, redundant profiles, and pairs of
facts that quietly contradict each other.

This module is the sleep-time counter-pressure, in three layers built on signals
already captured but never consumed:

* **Decay** (deterministic, reversible) — ``decayed_confidence`` combines an
  age-vs-durability factor (the ``time_frame`` ladder + the fact's timestamp;
  ``permanent`` never decays on age) with a usefulness factor folded from the
  recall journal (``recall`` / ``recall_used`` / ``recall_feedback``): a fact
  retrieved but never used sinks, a used/positively-fed-back one is reinforced.
* **Prune** (the destructive actuator, gated) — delete a fact only when it is
  non-durable *and* aged well past its window *and* never used.  ``dated`` and
  ``permanent`` facts are never pruned by decay alone.
* **Supersession / truth maintenance** (LLM judgment, gated) — when two
  near-duplicate facts materially disagree, an LLM decides whether they
  contradict and which holds; the loser is retired.  The *which-wins* rule is
  deterministic (durability, then recency, then confidence).

The pure functions here carry the logic and are unit-tested; ``MemoryMaintenance``
is the thin orchestrator that walks LTM at sleep.  Both destructive layers are
shadow-first — they journal what they *would* do and act only when their gate is
flipped, mirroring attention/prompt tuning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("Iyye.MemMaint")

# Durability ladder — MUST match iyye_io.memory_mcp_client._TF_RANK and STM's.
_TF_RANK = {tf: i for i, tf in enumerate(
    ("ephemeral", "session", "today", "recent", "dated", "permanent"))}

# Age half-life (days) per durability tier: a fact loses half its age factor
# after this long.  ``permanent`` (and anything unknown, treated as durable)
# does not decay on age at all.
_HALF_LIFE_DAYS = {
    "ephemeral": 0.25,   # ~6 hours
    "session":   1.0,
    "today":     2.0,
    "recent":    10.0,
    "dated":     45.0,
    "permanent": None,   # no age decay
}

# Below this decayed confidence a non-durable, unused, aged-out fact is a prune
# candidate.  Above the store's 0.2 confidence floor would never prune; the
# decayed *score* is allowed below 0.2 precisely to express "dead".
_PRUNE_FLOOR = 0.15
# Only ranks at or below this are ever prune-eligible (ephemeral/session/today).
_PRUNE_MAX_RANK = _TF_RANK["today"]
# A fact must be older than this multiple of its half-life to be pruned.
_PRUNE_AGE_MULT = 2.0


@dataclass
class Usage:
    """Per-fact retrieval/use signal folded from the recall journal."""
    retrieved: int = 0
    used: int = 0
    satisfied: int = 0
    dissatisfied: int = 0


# ----------------------------------------------------------------------
# Pure decay logic
# ----------------------------------------------------------------------

def age_days_of(fact: Dict[str, Any], now: datetime) -> float:
    """Days since the fact was stored; 0 if the timestamp is missing/unparseable
    (treat as fresh — never decay a fact we can't date)."""
    ts = fact.get("timestamp")
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (now - dt).total_seconds() / 86400.0)


def age_factor(time_frame: str, age_days: float) -> float:
    """Exponential age decay scaled to the tier's half-life; 1.0 (no decay) for
    permanent/durable tiers."""
    hl = _HALF_LIFE_DAYS.get(time_frame, None)
    if hl is None or age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / hl)


def usefulness_factor(usage: Usage) -> float:
    """Multiplier in roughly [0.3, 1.2] from how a fact fared in recall.

    No retrieval at all → neutral 1.0 (no evidence; let age decide).  Otherwise
    the used/retrieved ratio drives a 0.5→1.0 band (retrieved-but-never-used is
    the dead-weight signal), nudged by explicit satisfaction feedback."""
    if usage.retrieved <= 0:
        return 1.0
    use_rate = min(usage.used / usage.retrieved, 1.0)
    f = 0.5 + 0.5 * use_rate
    f += 0.10 * usage.satisfied - 0.15 * usage.dissatisfied
    return max(0.3, min(1.2, f))


def decayed_confidence(fact: Dict[str, Any], now: datetime,
                       usage: Usage) -> float:
    """``base · age_factor · usefulness_factor``, clamped to [0, 1].

    May fall below the store's 0.2 floor — that sub-floor region is the prune
    signal, not a stored confidence."""
    base = float(fact.get("confidence", 0.5) or 0.5)
    tf = fact.get("time_frame", "permanent") or "permanent"
    af = age_factor(tf, age_days_of(fact, now))
    uf = usefulness_factor(usage)
    return max(0.0, min(1.0, base * af * uf))


def is_prune_candidate(fact: Dict[str, Any], decayed: float,
                       usage: Usage, now: datetime) -> bool:
    """A fact is prunable only if it is non-durable, aged well past its window,
    never actually used, and has decayed below the floor.  Durable
    (``dated``/``permanent``) facts are never prunable by decay."""
    tf = fact.get("time_frame", "permanent") or "permanent"
    rank = _TF_RANK.get(tf, _TF_RANK["permanent"])
    if rank > _PRUNE_MAX_RANK:
        return False
    if usage.used > 0:
        return False
    if decayed >= _PRUNE_FLOOR:
        return False
    hl = _HALF_LIFE_DAYS.get(tf)
    if hl is None:
        return False
    return age_days_of(fact, now) > _PRUNE_AGE_MULT * hl


def fold_usage(events: List[Dict[str, Any]]) -> Dict[str, Usage]:
    """Build per-fact :class:`Usage` from a cycle's recall journal.

    Joins ``recall`` (retrieved ``refs`` per ``query_id``), ``recall_used``
    (the ``refs`` that informed a response), and ``recall_feedback`` (a
    ``signal`` per ``query_id``) — feedback is attributed to the facts that were
    *used* for that query (or, lacking a used set, to the retrieved ones)."""
    usage: Dict[str, Usage] = {}

    def u(fid: str) -> Usage:
        return usage.setdefault(fid, Usage())

    used_by_query: Dict[str, List[str]] = {}
    retrieved_by_query: Dict[str, List[str]] = {}
    for e in events:
        t = e.get("type")
        if t == "recall":
            refs = [r for r in (e.get("refs") or []) if r]
            retrieved_by_query[e.get("query_id")] = refs
            for fid in refs:
                u(fid).retrieved += 1
        elif t == "recall_used":
            refs = [r for r in (e.get("refs") or []) if r]
            used_by_query[e.get("query_id")] = refs
            for fid in refs:
                u(fid).used += 1
    for e in events:
        if e.get("type") != "recall_feedback":
            continue
        qid = e.get("query_id")
        signal = e.get("signal")
        targets = used_by_query.get(qid) or retrieved_by_query.get(qid) or []
        for fid in targets:
            if signal == "satisfied":
                u(fid).satisfied += 1
            elif signal == "dissatisfied":
                u(fid).dissatisfied += 1
    return usage


# ----------------------------------------------------------------------
# Pure truth-maintenance logic
# ----------------------------------------------------------------------

def supersession_winner(a: Dict[str, Any],
                        b: Dict[str, Any]) -> Tuple[str, str]:
    """Given two facts judged to contradict, return ``(winner_id, loser_id)``.

    Deterministic precedence: more durable ``time_frame`` wins; tie-broken by
    more recent timestamp; then by higher confidence.  Stable and explainable —
    the LLM only judges *whether* they conflict, never which survives."""
    def key(f: Dict[str, Any]):
        rank = _TF_RANK.get(f.get("time_frame", "permanent") or "permanent", 0)
        ts = str(f.get("timestamp") or "")
        conf = float(f.get("confidence", 0.0) or 0.0)
        return (rank, ts, conf)
    return (a["id"], b["id"]) if key(a) >= key(b) else (b["id"], a["id"])


_VERDICT_RE = re.compile(r"\b(contradict|independent|same)\b", re.IGNORECASE)


def parse_contradiction_verdict(text: str) -> str:
    """Reduce an LLM contradiction judgment to one of
    ``contradict`` | ``independent`` | ``same``.  Defaults to ``independent``
    (the safe no-op) when the response is unclear."""
    if not text:
        return "independent"
    m = _VERDICT_RE.search(text)
    return m.group(1).lower() if m else "independent"


def resolve_pair(a: Dict[str, Any], b: Dict[str, Any],
                 verdict: str) -> Optional[Tuple[str, str, str]]:
    """Decide what to do with a judged fact pair.

    Both ``contradict`` (the facts disagree — keep the more durable/recent one)
    and ``same`` (redundant restatements — the cluster of "Alex is trusted" /
    "trust was granted for Alex" facts) retire the deterministic loser; only
    ``same`` is *consolidation*, ``contradict`` is *correction*.  Returns
    ``(winner_id, loser_id, relation)`` or None for ``independent`` (no-op)."""
    if verdict not in ("contradict", "same"):
        return None
    winner, loser = supersession_winner(a, b)
    return winner, loser, verdict


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ----------------------------------------------------------------------
# Orchestrator (thin; shadow-first, both destructive layers gated)
# ----------------------------------------------------------------------

class MemoryMaintenance:
    # How many facts to inspect per sleep (bounded work; oldest-first via the
    # store's recency tail is a reasonable scan window for a hygiene pass).
    _SCAN_BUDGET = 200
    # Truth-maintenance is expensive (an LLM call per candidate pair); cap it.
    _SUPERSEDE_BUDGET = 5
    # Near-duplicate threshold for *considering* two facts a contradiction pair
    # (high overlap but not identical — identical is dedup's job, not ours).
    _PAIR_OVERLAP = 0.5

    # Gates — shadow until validated on recorded cycles, like _TUNE_APPLY.
    _DECAY_APPLY = False       # write decayed confidence back to the store
    _PRUNE_APPLY = False       # delete prune candidates
    _SUPERSEDE_APPLY = False   # retire the loser of a contradiction

    def __init__(self, brain: Any):
        self.brain = brain
        self.memory = getattr(brain, "memory", None)
        self.journal = getattr(brain, "journal", None)

    def _emit(self, event_type: str, **fields: Any) -> None:
        from event_journal import emit
        emit(self.journal, event_type, **fields)

    def run_sleep_pass(self) -> bool:
        """Decay → prune → supersede over a bounded LTM window.  Best-effort:
        any failure is logged and the phase still completes (returns True)."""
        if self.memory is None:
            return True
        now = datetime.now(timezone.utc)
        usage = self._load_usage()
        try:
            facts = self.memory.get_recent_facts(limit=self._SCAN_BUDGET)
        except Exception as exc:
            log.debug("Memory maintenance: scan failed: %s", exc)
            return True

        decayed_n = pruned_n = 0
        prune_candidates: List[str] = []
        for fact in facts:
            fid = fact.get("id")
            if not fid:
                continue
            u = usage.get(fid, Usage())
            new_conf = decayed_confidence(fact, now, u)
            old_conf = float(fact.get("confidence", 0.5) or 0.5)

            if is_prune_candidate(fact, new_conf, u, now):
                prune_candidates.append(fid)
                self._emit("memory_prune_candidate", fact_id=fid,
                           time_frame=fact.get("time_frame"),
                           age_days=round(age_days_of(fact, now), 2),
                           decayed=round(new_conf, 3),
                           retrieved=u.retrieved, used=u.used,
                           applied=self._PRUNE_APPLY)
                if self._PRUNE_APPLY:
                    try:
                        self.memory.delete_fact(fid)
                        pruned_n += 1
                    except Exception as exc:
                        log.debug("Prune of %s failed: %s", fid[:8], exc)
                continue  # don't bother decaying a fact we just pruned

            # Decay: only act on a meaningful drop, and never below the store's
            # 0.2 floor (a still-live fact stays retrievable).
            if old_conf - new_conf > 0.02:
                stored = max(0.2, new_conf)
                self._emit("memory_decay", fact_id=fid,
                           old=round(old_conf, 3), new=round(stored, 3),
                           time_frame=fact.get("time_frame"),
                           retrieved=u.retrieved, used=u.used,
                           applied=self._DECAY_APPLY)
                decayed_n += 1
                if self._DECAY_APPLY:
                    self._apply_confidence(fid, stored)

        superseded_n = self._maybe_supersede(facts, now)

        self._emit("memory_maintenance", scanned=len(facts),
                   decayed=decayed_n, prune_candidates=len(prune_candidates),
                   pruned=pruned_n, superseded=superseded_n,
                   decay_applied=self._DECAY_APPLY,
                   prune_applied=self._PRUNE_APPLY,
                   supersede_applied=self._SUPERSEDE_APPLY)
        if decayed_n or prune_candidates or superseded_n:
            log.info("Memory maintenance: scanned %d, decayed %d, "
                     "prune-candidates %d (pruned %d), superseded %d",
                     len(facts), decayed_n, len(prune_candidates),
                     pruned_n, superseded_n)
        return True

    def _load_usage(self) -> Dict[str, Usage]:
        cid = getattr(self.brain, "_consolidating_cycle",
                      getattr(self.brain, "_journal_cycle", None))
        if self.journal is None or cid is None:
            return {}
        try:
            events = self.journal.read_cycle(cid, types=frozenset(
                {"recall", "recall_used", "recall_feedback"}))
        except Exception as exc:
            log.debug("Memory maintenance: usage read failed: %s", exc)
            return {}
        return fold_usage(events)

    def _apply_confidence(self, fid: str, conf: float) -> None:
        try:
            bump = getattr(self.memory, "_bump_confidence", None)
            if callable(bump):
                bump(fid, conf)
            else:
                self.memory.update_fact(fid, confidence=conf)
        except Exception as exc:
            log.debug("Decay write for %s failed: %s", fid[:8], exc)

    # ------------------------------------------------------------------
    # Truth maintenance (#3) — LLM contradiction judgment, gated
    # ------------------------------------------------------------------

    def _maybe_supersede(self, facts: List[Dict[str, Any]],
                         now: datetime) -> int:
        """Find near-duplicate-but-not-identical fact pairs, ask an LLM whether
        each contradicts, and retire the deterministic loser.  Always journals
        ``memory_superseded``; retires only when the gate is set."""
        pairs = self._contradiction_pairs(facts)
        if not pairs:
            return 0
        client = self._judge_client()
        if client is None:
            return 0
        retired = 0
        for a, b in pairs[:self._SUPERSEDE_BUDGET]:
            verdict = self._judge_pair(client, a, b)
            decision = resolve_pair(a, b, verdict)
            if decision is None:
                continue  # independent — leave both
            winner, loser, relation = decision
            self._emit("memory_superseded", winner=winner, loser=loser,
                       relation=relation, applied=self._SUPERSEDE_APPLY)
            if self._SUPERSEDE_APPLY:
                try:
                    self.memory.delete_fact(loser)
                    retired += 1
                except Exception as exc:
                    log.debug("Supersede delete %s failed: %s", loser[:8], exc)
        return retired

    def _contradiction_pairs(
        self, facts: List[Dict[str, Any]],
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Candidate pairs: high token overlap but not identical text.  Uses the
        store's semantic neighbour lookup when available, else an O(n²) overlap
        scan over the (bounded) scan window."""
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        seen: set = set()
        by_id = {f.get("id"): f for f in facts}
        search = getattr(self.memory, "search_semantic", None)
        if callable(search):
            for f in facts:
                try:
                    neighbours = search(f.get("text", ""), limit=3)
                except Exception:
                    neighbours = []
                for nb in neighbours:
                    nid = nb.get("id")
                    if not nid or nid == f.get("id"):
                        continue
                    key = tuple(sorted((f.get("id"), nid)))
                    if key in seen:
                        continue
                    other = by_id.get(nid, nb)
                    ov = _overlap(f.get("text", ""), other.get("text", ""))
                    if self._PAIR_OVERLAP <= ov < 0.98:
                        seen.add(key)
                        pairs.append((f, other))
            return pairs
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                ov = _overlap(facts[i].get("text", ""), facts[j].get("text", ""))
                if self._PAIR_OVERLAP <= ov < 0.98:
                    pairs.append((facts[i], facts[j]))
        return pairs

    def _judge_pair(self, client, a: Dict[str, Any],
                    b: Dict[str, Any]) -> str:
        try:
            from llm_client import _load_prompt
            template = _load_prompt("memory_contradiction")
            filled = (template
                      .replace("<<fact_a>>", a.get("text", ""))
                      .replace("<<fact_b>>", b.get("text", "")))
            raw = client.complete(filled)
            return parse_contradiction_verdict(raw)
        except Exception as exc:
            log.debug("Contradiction judge failed: %s", exc)
            return "independent"

    def _judge_client(self):
        router = getattr(self.brain, "llm_router", None)
        if router is not None:
            try:
                return router.get_client(role="reasoning", task={
                    "quality_need": 0.7, "latency_budget_s": 30, "urgency": 0.1})
            except Exception:
                pass
        try:
            from llm_client import LLMClient
            return LLMClient()
        except Exception:
            return None


__all__ = [
    "Usage", "age_days_of", "age_factor", "usefulness_factor",
    "decayed_confidence", "is_prune_candidate", "fold_usage",
    "supersession_winner", "parse_contradiction_verdict", "resolve_pair",
    "MemoryMaintenance",
]
