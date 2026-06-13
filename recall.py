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
"""Unified memory recall — one query path across STM, LTM, and Theory of Mind.

The three memory stores each have their own retrieval (STM substring, LTM
vector, ToM per-contact history), and callers stitched them together by hand —
so a question whose answer lived in the *wrong* store was missed.  Concretely:
"When did Jacob contact you?" was answered from LTM facts while Jacob's
interaction history lives only in ToM, so the lookup came back empty.

``Recall.query()`` fans one query out to all three stores, normalizes the
results into ranked :class:`RecallResult` records with provenance, and — the
key fix — detects *people named in the query* (not just the sender) and pulls
their ToM interaction history, timestamps included.  It journals a ``recall``
event per query (query, result count, per-source breakdown): the shadow record
that lets a later pass measure which retrievals actually prove useful and feed
that back into confidence/pruning (the missing quality signal).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("Iyye.Recall")

_STOPWORDS = frozenset(
    "the a an of to in is are was were and or for on at by with what when "
    "where who how why did do does you your i me my it that this".split()
)

# Per-source weight applied to a result's in-store relevance.
_SOURCE_WEIGHT = {"ltm": 1.0, "stm": 0.85, "tom": 1.0}

# Fraction of a recalled fact's distinctive tokens that must appear in the
# reply for it to count as "used" — the cheap attribution backstop. (The
# LLM-citation path is the planned more-precise refinement.)
_ATTRIBUTION_THRESHOLD = 0.4


@dataclass
class RecallResult:
    text: str
    source: str                         # "ltm" | "stm" | "tom"
    score: float
    provenance: str = ""
    time_frame: Optional[str] = None
    confidence: Optional[float] = None
    ref: Optional[str] = None           # store id/handle (for usefulness feedback)
    query_id: Optional[str] = None      # joins the `recall` and `recall_used` events
    metadata: Dict[str, Any] = field(default_factory=dict)


def _tokens(text: str) -> frozenset:
    return frozenset(
        w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()
        if w not in _STOPWORDS and len(w) > 2
    )


def _overlap(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(sub: frozenset, sup: frozenset) -> float:
    """Fraction of *sub*'s tokens present in *sup* — the right measure for
    "did the reply use this fact" (Jaccard underrates a short fact against a
    longer reply)."""
    if not sub:
        return 0.0
    return len(sub & sup) / len(sub)


class Recall:
    """Read-only facade over STM, LTM, and ToM for relevance retrieval."""

    def __init__(self, brain: Any):
        self.brain = brain

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        *,
        limit: int = 8,
        sender: Optional[str] = None,
        per_store: int = 8,
    ) -> List[RecallResult]:
        """Ranked cross-store results relevant to *text*.

        *sender* (the current speaker, if any) is always consulted in ToM;
        people *named in the query* are detected and consulted too — that is
        what surfaces a third party's interaction history."""
        import uuid
        query_id = uuid.uuid4().hex[:12]
        results: List[RecallResult] = []
        results += self._from_ltm(text, per_store)
        results += self._from_stm(text, per_store)
        results += self._from_tom(text, sender)
        ranked = self._merge_rank(results, limit)
        for r in ranked:
            r.query_id = query_id   # join key for the usefulness signal
        self._journal(text, ranked, query_id)
        return ranked

    # ------------------------------------------------------------------
    # Per-store retrieval → RecallResult
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_score(index: int, n: int) -> float:
        """In-store relevance from rank (stores return best-first)."""
        return 1.0 - index / (n + 1)

    def _from_ltm(self, text: str, limit: int) -> List[RecallResult]:
        memory = getattr(self.brain, "memory", None)
        if memory is None:
            return []
        try:
            facts = memory.search_semantic(text, limit=limit)
        except Exception as exc:
            log.debug("Recall: LTM search failed: %s", exc)
            return []
        out = []
        for i, f in enumerate(facts):
            conf = float(f.get("confidence", 0.5))
            out.append(RecallResult(
                text=f.get("text", ""), source="ltm",
                score=self._rank_score(i, len(facts)) * _SOURCE_WEIGHT["ltm"]
                      * (0.6 + 0.4 * conf),
                provenance=f.get("provenance") or f.get("source") or "",
                time_frame=f.get("time_frame"), confidence=conf,
                ref=str(f.get("id") or ""), metadata=f.get("metadata") or {},
            ))
        return out

    def _from_stm(self, text: str, limit: int) -> List[RecallResult]:
        stm = getattr(self.brain, "stm", None)
        if stm is None:
            return []
        try:
            facts = stm.search(text, limit=limit)
        except Exception as exc:
            log.debug("Recall: STM search failed: %s", exc)
            return []
        out = []
        for i, f in enumerate(facts):
            if f.get("time_frame") == "ephemeral":
                continue  # metric snapshots — never relevant context
            conf = float(f.get("confidence", 0.7))
            out.append(RecallResult(
                text=f.get("text", ""), source="stm",
                score=self._rank_score(i, len(facts)) * _SOURCE_WEIGHT["stm"]
                      * (0.6 + 0.4 * conf),
                provenance=f.get("provenance") or "",
                time_frame=f.get("time_frame"), confidence=conf,
                ref=str(f.get("id") or ""),
            ))
        return out

    def _from_tom(self, text: str, sender: Optional[str]) -> List[RecallResult]:
        tom = self.brain.theory_of_mind() if hasattr(self.brain, "theory_of_mind") else None
        if tom is None:
            return []
        # Contacts named in the query (the third-party case) plus the sender.
        contacts: Dict[str, str] = {}
        try:
            if hasattr(tom, "detect_contacts_in_text"):
                for cid, disp in tom.detect_contacts_in_text(text):
                    contacts[cid] = disp
            if sender and hasattr(tom, "find_contacts"):
                for cid, disp in tom.find_contacts(sender.lower()):
                    contacts.setdefault(cid, disp)
        except Exception as exc:
            log.debug("Recall: ToM contact detection failed: %s", exc)
            return []
        out = []
        for cid, disp in contacts.items():
            try:
                interactions = tom.get_recent_interactions(cid, limit=3)
            except Exception:
                interactions = []
            for it in interactions:
                ts = (it.get("timestamp") or "")[:19]
                said = (it.get("user_said") or "")[:160]
                if not said:
                    continue
                out.append(RecallResult(
                    text=f"{disp} said (at {ts}): {said}",
                    source="tom", score=_SOURCE_WEIGHT["tom"] * 0.9,
                    provenance=f"theory_of_mind:{cid}",
                    confidence=0.9, ref=cid,
                    metadata={"contact_id": cid, "timestamp": it.get("timestamp")},
                ))
        return out

    # ------------------------------------------------------------------
    # Merge / rank / dedup
    # ------------------------------------------------------------------

    def _merge_rank(self, results: List[RecallResult], limit: int) -> List[RecallResult]:
        results.sort(key=lambda r: r.score, reverse=True)
        kept: List[RecallResult] = []
        kept_tokens: List[frozenset] = []
        for r in results:
            if not (r.text or "").strip():
                continue
            toks = _tokens(r.text)
            # Drop a near-duplicate of something already kept (e.g. an STM fact
            # also promoted to LTM) — the higher-scored copy is already in.
            if any(_overlap(toks, kt) >= 0.6 for kt in kept_tokens):
                continue
            kept.append(r)
            kept_tokens.append(toks)
            if len(kept) >= limit:
                break
        return kept

    # ------------------------------------------------------------------
    # Rendering + usefulness feedback
    # ------------------------------------------------------------------

    @staticmethod
    def render(results: List[RecallResult]) -> str:
        if not results:
            return "(none)"
        lines = []
        for r in results:
            tag = r.source.upper()
            extra = ""
            if r.time_frame and r.confidence is not None:
                extra = f"/{r.time_frame}/{r.confidence:.2f}"
            prov = f" from {r.provenance}" if r.provenance else ""
            lines.append(f"[{tag}{extra}{prov}] {r.text}")
        return "\n".join(lines)

    @staticmethod
    def attribute(
        results: List[RecallResult],
        response_text: str,
        threshold: float = _ATTRIBUTION_THRESHOLD,
    ) -> List[RecallResult]:
        """Which recalled results the reply actually leaned on — token
        containment of each fact in *response_text*.  Cheap and honest-enough
        backstop; the LLM-citation path is the planned refinement."""
        if not results or not (response_text or "").strip():
            return []
        resp = _tokens(response_text)
        return [r for r in results
                if _containment(_tokens(r.text), resp) >= threshold]

    def mark_used(self, results: List[RecallResult]) -> None:
        """Record that *results* informed a response — the usefulness signal a
        later pass folds back into confidence/pruning (Phase B/C).  Shadow
        only: journals which recalled facts were used, joined to the originating
        recall by query_id."""
        from event_journal import emit
        qid = next((r.query_id for r in results if r.query_id), None)
        emit(getattr(self.brain, "journal", None), "recall_used",
             query_id=qid,
             refs=[r.ref for r in results if r.ref],
             sources=[r.source for r in results])

    def _journal(self, query: str, results: List[RecallResult],
                 query_id: str) -> None:
        from event_journal import emit, clip
        by_source: Dict[str, int] = {}
        for r in results:
            by_source[r.source] = by_source.get(r.source, 0) + 1
        # Log the RETRIEVED fact ids, not just the count: the per-fact
        # retrieval denominator the usefulness signal needs (a fact retrieved
        # often but never used — recall_used — is decay-worthy; one never in
        # any `recall.refs` is dead weight).  Joinable to recall_used by
        # query_id.
        emit(getattr(self.brain, "journal", None), "recall",
             query_id=query_id, query=clip(query, 200),
             n=len(results), sources=by_source,
             refs=[r.ref for r in results if r.ref])


__all__ = ["Recall", "RecallResult"]
