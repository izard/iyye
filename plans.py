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
"""Long term plans — the third durable artifact (HLD: "Long term plans").

Streams are ephemeral (they pause during sleep and reset across cycles) and
facts are passive (they record, they don't drive).  A :class:`LongTermPlan` is
durable like a fact and active like a stream: it survives sleep cycles on disk
under ``plans/<plan_id>/plan.json`` and drives work by feeding its next due
step to the PlannerStream, which dispatches execution to StreamFactory.

Lifecycle: ``proposed -> approved -> active -> suspended -> completed/abandoned``.

SECURITY (HLD): plans whose steps reach actuators beyond the local web chat
require owner approval *from the local web chat* to enter ``active``.  Nothing
arriving over telegram may create-approved, approve or abandon a plan — the
``source`` argument of :meth:`PlanStore.set_lifecycle` is stamped by shipped
code (never by the LLM), and the gate here is defence-in-depth below the
capability-profile check in UserChatStream, mirroring how trust is enforced
twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.Plans")

PLANS_ROOT = PROJECT_ROOT / "plans"

LIFECYCLE_STATES = (
    "proposed", "approved", "active", "suspended", "completed", "abandoned",
)

# state -> states it may move to.  "approved" exists as a distinct resting
# state so the owner can approve now and let the planner activate on its own
# schedule; approve-and-activate in one command is also legal.
_TRANSITIONS: Dict[str, set] = {
    "proposed":  {"approved", "active", "abandoned"},
    "approved":  {"active", "suspended", "abandoned"},
    "active":    {"suspended", "completed", "abandoned"},
    "suspended": {"active", "abandoned"},
    "completed": set(),
    "abandoned": set(),
}

# Sources allowed to approve/activate an external-reaching plan.  Only the
# local web chat (the machine owner) qualifies — same posture as trust.
_OWNER_SOURCES = ("web_chat",)

# Step types that reach beyond Iyye's own memory: they produce actions or
# social output that may travel through non-local actuators.
_EXTERNAL_STEP_TYPES = {"action", "social"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] or "plan"


class LongTermPlan:
    """One durable plan: goal, alignment linkage, lazily decomposed steps.

    Steps share the dict shape PlannedContinuationStream already executes
    (``description``/``type``/``input``) plus plan-level bookkeeping:
    ``status`` (pending|dispatched|done|stale), optional ``due`` (ISO 8601,
    drives the in-sleep deadline wakeup), ``abstract`` (needs LLM
    decomposition before it can be dispatched) and optional ``contact``
    (Theory-of-Mind context is pulled before dispatch).
    """

    def __init__(
        self,
        plan_id: str,
        goal: str,
        alignment_weights: Optional[Dict[str, float]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        provenance: str = "unknown",
        deadline: Optional[str] = None,
        lifecycle: str = "proposed",
    ):
        self.plan_id = plan_id
        self.goal = goal
        # HLD alignment goals; lets AlignmentStream machinery score plan work.
        self.alignment_weights = alignment_weights or {"agency": 0.5}
        self.steps = steps or []
        self.provenance = provenance
        self.deadline = deadline          # plan-level due date (ISO), optional
        self.lifecycle = lifecycle
        self.created_at = _utcnow()
        self.lifecycle_changed_at = self.created_at
        self.last_progress_at: Optional[str] = None
        self.progress: List[Dict[str, str]] = []   # [{ts, text}]
        for i, s in enumerate(self.steps):
            s.setdefault("step", i)
            s.setdefault("status", "pending")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def requires_owner_approval(self) -> bool:
        """True when any step reaches beyond Iyye's own memory/web chat.

        HLD: plans whose steps reach actuators beyond the local web chat
        (telegram sends, self-modification, spawning capabilities) need owner
        approval.  Steps are conservative-by-default: an *abstract* step's
        eventual shape is unknown, so it counts as external too.
        """
        for s in self.steps:
            if s.get("abstract"):
                return True
            if s.get("type") in _EXTERNAL_STEP_TYPES:
                return True
            if s.get("actuator") or s.get("contact"):
                return True
        return False

    def next_pending_step(self) -> Optional[Dict[str, Any]]:
        for s in self.steps:
            if s.get("status") == "pending":
                return s
        return None

    def dispatched_step(self) -> Optional[Dict[str, Any]]:
        for s in self.steps:
            if s.get("status") == "dispatched":
                return s
        return None

    def all_steps_done(self) -> bool:
        return bool(self.steps) and all(
            s.get("status") == "done" for s in self.steps
        )

    def next_due(self) -> Optional[str]:
        """Earliest unmet deadline (ISO) across pending/dispatched steps and
        the plan itself, or None."""
        candidates = [
            s.get("due") for s in self.steps
            if s.get("status") in ("pending", "dispatched") and s.get("due")
        ]
        if self.deadline and not self.all_steps_done():
            candidates.append(self.deadline)
        return min(candidates) if candidates else None

    def fingerprint(self) -> str:
        """Near-duplicate detection key: goal + step descriptions."""
        basis = self.goal.lower() + "|" + "|".join(
            str(s.get("description", "")).lower() for s in self.steps
        )
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]

    def summary_line(self) -> str:
        done = sum(1 for s in self.steps if s.get("status") == "done")
        due = self.next_due()
        return (
            f"[{self.lifecycle}] {self.plan_id}: {self.goal[:80]} "
            f"({done}/{len(self.steps)} steps"
            + (f", due {due}" if due else "") + ")"
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "alignment_weights": self.alignment_weights,
            "steps": self.steps,
            "provenance": self.provenance,
            "deadline": self.deadline,
            "lifecycle": self.lifecycle,
            "created_at": self.created_at,
            "lifecycle_changed_at": self.lifecycle_changed_at,
            "last_progress_at": self.last_progress_at,
            "progress": self.progress[-200:],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LongTermPlan":
        plan = cls(
            plan_id=d["plan_id"],
            goal=d.get("goal", ""),
            alignment_weights=d.get("alignment_weights"),
            steps=d.get("steps"),
            provenance=d.get("provenance", "unknown"),
            deadline=d.get("deadline"),
            lifecycle=d.get("lifecycle", "proposed"),
        )
        plan.created_at = d.get("created_at", plan.created_at)
        plan.lifecycle_changed_at = d.get(
            "lifecycle_changed_at", plan.created_at)
        plan.last_progress_at = d.get("last_progress_at")
        plan.progress = d.get("progress", [])
        return plan


class PlanStore:
    """Disk-backed registry of long term plans.

    One instance lives on the brain (``brain.plan_store``) so the planner
    stream, the chat plan actions and the in-sleep deadline check all see the
    same state.  Every mutation persists immediately — plans must survive a
    process restart mid-cycle (that durability is their whole point).
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else PLANS_ROOT
        self.archive_dir = self.base_dir / "archive"
        self._plans: Dict[str, LongTermPlan] = {}
        self._load()

    def _load(self) -> None:
        if not self.base_dir.is_dir():
            return
        for plan_file in sorted(self.base_dir.glob("*/plan.json")):
            if plan_file.parent.parent == self.archive_dir:
                continue
            try:
                data = json.loads(plan_file.read_text(encoding="utf-8"))
                plan = LongTermPlan.from_dict(data)
                self._plans[plan.plan_id] = plan
            except Exception as exc:
                log.warning("Could not load plan %s: %s", plan_file, exc)
        if self._plans:
            log.info("PlanStore: loaded %d plan(s)", len(self._plans))

    def save(self, plan: LongTermPlan) -> None:
        plan_dir = self.base_dir / plan.plan_id
        try:
            plan_dir.mkdir(parents=True, exist_ok=True)
            tmp = plan_dir / "plan.json.tmp"
            tmp.write_text(
                json.dumps(plan.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(plan_dir / "plan.json")
        except Exception as exc:
            log.error("Could not persist plan %s: %s", plan.plan_id, exc)

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def create(
        self,
        goal: str,
        steps: Optional[List[Dict[str, Any]]] = None,
        provenance: str = "unknown",
        alignment_weights: Optional[Dict[str, float]] = None,
        deadline: Optional[str] = None,
    ) -> Optional[LongTermPlan]:
        """Create a plan in ``proposed`` state.  Returns None on duplicate.

        A plan with no steps gets a single abstract step so the planner has
        something to decompose lazily (HLD: only the next step is concrete).
        """
        goal = (goal or "").strip()
        if not goal:
            return None
        if steps is None:
            steps = [{
                "step": 0,
                "description": f"Decompose goal into first concrete steps: {goal[:120]}",
                "abstract": True,
            }]
        candidate = LongTermPlan(
            plan_id="", goal=goal, alignment_weights=alignment_weights,
            steps=steps, provenance=provenance, deadline=deadline,
        )
        fp = candidate.fingerprint()
        for existing in self._plans.values():
            if existing.lifecycle in ("completed", "abandoned"):
                continue
            if existing.fingerprint() == fp:
                log.info("PlanStore: duplicate of %s — not created",
                         existing.plan_id)
                return None
        base = _slugify(goal)
        plan_id, n = base, 1
        while plan_id in self._plans or (self.base_dir / plan_id).exists():
            n += 1
            plan_id = f"{base}_{n}"
        candidate.plan_id = plan_id
        self._plans[plan_id] = candidate
        self.save(candidate)
        log.info("PlanStore: created plan %s (%s)", plan_id, candidate.lifecycle)
        return candidate

    # ------------------------------------------------------------------
    # Lifecycle (security boundary)
    # ------------------------------------------------------------------

    def set_lifecycle(
        self, plan_id: str, new_state: str, source: str = "unknown",
    ) -> Optional[str]:
        """Transition a plan; returns an error string or None on success.

        *source* identifies the requesting channel and is stamped by shipped
        code, never taken from LLM output.  ``approved``/``active`` on an
        external-reaching plan requires a local-owner source; abandoning is
        owner-only too (a remote contact must not kill the owner's plans).
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return f"unknown plan '{plan_id}'"
        if new_state not in LIFECYCLE_STATES:
            return f"unknown lifecycle state '{new_state}'"
        if new_state == plan.lifecycle:
            return None
        if new_state not in _TRANSITIONS[plan.lifecycle]:
            return (f"illegal transition {plan.lifecycle} -> {new_state} "
                    f"for '{plan_id}'")
        local = any(s in source.lower() for s in _OWNER_SOURCES)
        if new_state in ("approved", "active"):
            if plan.requires_owner_approval() and not local:
                log.warning(
                    "SECURITY: blocked %s of external-reaching plan %s from "
                    "source '%s' — owner approval is local-web-chat only",
                    new_state, plan_id, source,
                )
                return ("owner approval from the local web chat is required "
                        f"to {new_state[:7]} '{plan_id}'")
        if new_state == "abandoned" and not local and "planner" not in source:
            # The planner itself may abandon (stale plans during sleep
            # review); remote chat contacts may not.
            return f"only the owner may abandon '{plan_id}'"
        plan.lifecycle = new_state
        plan.lifecycle_changed_at = _utcnow()
        self.save(plan)
        log.info("PlanStore: %s -> %s (source=%s)", plan_id, new_state, source)
        return None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, plan_id: str) -> Optional[LongTermPlan]:
        return self._plans.get(plan_id)

    def all_plans(self) -> List[LongTermPlan]:
        return list(self._plans.values())

    def active_plans(self) -> List[LongTermPlan]:
        return [p for p in self._plans.values() if p.lifecycle == "active"]

    def by_lifecycle(self, state: str) -> List[LongTermPlan]:
        return [p for p in self._plans.values() if p.lifecycle == state]

    def next_due_deadline(self) -> Optional[datetime]:
        """Earliest deadline across active plans — the in-sleep scheduler
        input (HLD: a due plan step is a valid urgent-wakeup source).

        Cheap (in-memory timestamp comparison), safe to call every asleep
        tick while all streams are paused.
        """
        earliest: Optional[datetime] = None
        for plan in self.active_plans():
            due = plan.next_due()
            if not due:
                continue
            try:
                dt = datetime.fromisoformat(due)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if earliest is None or dt < earliest:
                earliest = dt
        return earliest

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def record_progress(self, plan_id: str, text: str) -> None:
        plan = self._plans.get(plan_id)
        if plan is None:
            return
        ts = _utcnow()
        plan.progress.append({"ts": ts, "text": text[:500]})
        plan.last_progress_at = ts
        self.save(plan)

    def mark_step(self, plan_id: str, step_index: int, status: str) -> bool:
        plan = self._plans.get(plan_id)
        if plan is None:
            return False
        for s in plan.steps:
            if s.get("step") == step_index:
                s["status"] = status
                self.save(plan)
                return True
        return False

    def replace_abstract_step(
        self, plan_id: str, step_index: int, concrete_steps: List[Dict[str, Any]],
    ) -> bool:
        """Swap one abstract step for its LLM-decomposed concrete steps."""
        plan = self._plans.get(plan_id)
        if plan is None or not concrete_steps:
            return False
        idx = next(
            (i for i, s in enumerate(plan.steps) if s.get("step") == step_index),
            None,
        )
        if idx is None:
            return False
        old = plan.steps[idx]
        for s in concrete_steps:
            s.setdefault("status", "pending")
            s.pop("abstract", None)
            if old.get("due") and not s.get("due"):
                s["due"] = old["due"]
        plan.steps[idx:idx + 1] = concrete_steps
        for i, s in enumerate(plan.steps):
            s["step"] = i
        self.save(plan)
        return True

    # ------------------------------------------------------------------
    # Archival (sleep review)
    # ------------------------------------------------------------------

    def archive(self, plan_id: str) -> Optional[LongTermPlan]:
        """Move a completed/abandoned plan's directory under plans/archive/
        and drop it from the live registry.  Returns the plan for LTM
        summarization, or None."""
        plan = self._plans.get(plan_id)
        if plan is None or plan.lifecycle not in ("completed", "abandoned"):
            return None
        try:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            src = self.base_dir / plan_id
            dst = self.archive_dir / plan_id
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(str(dst))
                shutil.move(str(src), str(dst))
        except Exception as exc:
            log.warning("Could not archive plan %s: %s", plan_id, exc)
        del self._plans[plan_id]
        return plan


__all__ = ["LongTermPlan", "PlanStore", "PLANS_ROOT", "LIFECYCLE_STATES"]
