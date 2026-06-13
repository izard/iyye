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
"""Planner stream — owns the long term plan store (HLD: special stream #10).

Creates plans from owner requests, self-reflection proposals and chat
commands; lazily decomposes the next step of each active plan via LLM; posts
execution requests to StreamFactory through the mailbox.  The planner never
executes plan work itself — execution is ordinary PlannedContinuationStreams
tagged with their (plan_id, step), so attention, alignment scoring and
checkpoint-stopping apply unchanged.

Sleep: owns the ``plan_review`` sleep phase (after replay, before cleanup) —
assess progress, reset orphaned dispatches, suspend stale plans, archive
finished ones into LTM facts.
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from iyye_base import ProcessingStream, SleepPhase
from llm_scheduler import LLMConsumerMixin, LLMCall
from plans import PlanStore

log = logging.getLogger("Iyye.Planner")


class PlannerStream(LLMConsumerMixin, ProcessingStream):
    """Owns ``brain.plan_store``; drives active plans one step at a time."""

    # Dispatch a step early enough to matter, late enough to stay lazy.
    _DUE_LOOKAHEAD = timedelta(days=1)
    # Active plan with no progress for this long is suspended at sleep review.
    _STALE_DAYS = 14
    # Active plan drifting at least this long (but not yet stale) is a replan
    # candidate — dreaming should revisit it (HLD: replanning happens while
    # dreaming).  Phase A only *detects and journals* candidates; Phase B will
    # make the synchronous LLM revision.
    _REPLAN_STALL_DAYS = 3
    # Unapproved proposal older than this is expired (abandoned + archived)
    # at sleep review, unblocking its fingerprint for future re-proposals.
    _PROPOSAL_EXPIRY_DAYS = 7
    # Re-check the plan set every N ticks (no need to scan every tick).
    _DRIVE_INTERVAL = 20
    # Ticks after a dispatch before a missing executor is treated as a dropped
    # dispatch (the factory creates the executor within a tick or two; this
    # grace avoids racing it).  Liveness itself is matched by the executor's
    # structured _plan_ref, not its name, so there is no prefix collision.
    _DISPATCH_GRACE_TICKS = 3

    def __init__(self, brain: "IyyeBrain"):
        super().__init__(name="planner")
        self.brain = brain
        self.priority = 2
        self._can_be_conscious = True
        store = getattr(brain, "plan_store", None)
        self.store: PlanStore = store if store is not None else PlanStore()
        # plan_id -> tick at which we dispatched its current step.  Liveness is
        # checked via the executor's _plan_ref (exact), not this; the tick is
        # only used to detect a dropped dispatch after a grace period.
        self._dispatched: Dict[str, int] = {}
        # (plan_id, step_index) currently being decomposed by the LLM.
        self._decomposing: Optional[Tuple[str, int]] = None
        self._tick = 0
        # Wake epoch last seen — reset the drive cadence on a fresh awake cycle
        # so the day's due steps are dispatched promptly on wakeup rather than
        # up to _DRIVE_INTERVAL ticks later.
        self._last_wake_epoch: Optional[int] = None
        # Sleep-review work list; None outside a review pass.
        self._review_pending: Optional[List[str]] = None
        # Executors do not survive a process restart or sleep cycle: any step
        # left 'dispatched' has no live stream behind it — reset to pending.
        self._reset_orphaned_dispatches()

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # On a new awake cycle, reset the tick counter so the first drive pass
        # (gated to _tick % _DRIVE_INTERVAL == 1) runs within a couple of ticks
        # of wakeup instead of lagging the whole interval.
        epoch = getattr(self.brain, "_wake_epoch", None)
        if epoch is not None and epoch != self._last_wake_epoch:
            self._last_wake_epoch = epoch
            self._tick = 0
        self._tick += 1
        for msg in self.brain.drain_messages(self.name):
            try:
                self._handle_message(msg)
            except Exception as exc:
                log.error("Planner: message handling failed: %s", exc)
        try:
            self.checkpoint()
        except StopIteration:
            return None

        result = self._llm_poll()
        if result is not None:
            return self._on_decompose_result(result)

        self.urgency = self._deadline_urgency()

        if self._llm_busy() or self._paused:
            return None
        if self._tick % self._DRIVE_INTERVAL != 1:
            return None
        return self._drive_plans(context)

    # ------------------------------------------------------------------
    # Mailbox
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "plan_propose":
            self._handle_propose(msg)
        elif action == "plan_approve":
            err = self.store.set_lifecycle(
                msg.get("plan_id", ""), "active",
                source=msg.get("source", "unknown"),
            )
            self.add_to_log(
                f"Approve '{msg.get('plan_id')}': {err or 'now active'}")
            if err is None:
                plan = self.store.get(msg.get("plan_id", ""))
                if plan is not None:
                    self._index_plan_in_ltm(plan)
        elif action == "plan_abandon":
            err = self.store.set_lifecycle(
                msg.get("plan_id", ""), "abandoned",
                source=msg.get("source", "unknown"),
            )
            self.add_to_log(
                f"Abandon '{msg.get('plan_id')}': {err or 'abandoned'}")
            if err is None:
                plan = self.store.get(msg.get("plan_id", ""))
                if plan is not None:
                    self._index_plan_in_ltm(plan)
        elif action == "plan_step_done":
            self._handle_step_done(msg)
        else:
            log.debug("Planner: unknown message action %r", action)

    def _handle_propose(self, msg: Dict[str, Any]) -> None:
        source = msg.get("source", "unknown")
        plan = self.store.create(
            goal=msg.get("goal", ""),
            steps=msg.get("steps"),
            provenance=source,
            alignment_weights=msg.get("alignment_weights"),
            deadline=msg.get("deadline"),
        )
        if plan is None:
            self.add_to_log(
                f"Proposal from {source} rejected (empty or duplicate)")
            return
        self._drain_adenosine("plan_create")
        self.add_to_log(f"Created plan: {plan.summary_line()}")
        self._stm_fact(
            f"New long term plan '{plan.plan_id}' proposed by {source}: "
            f"{plan.goal}",
            time_frame="dated",
        )
        # Activation policy: the owner's own request is its approval; an
        # internal-only plan (nothing reaches beyond memory/web chat) may
        # self-activate.  Everything else waits for the owner.  Owner is
        # matched exactly (is_owner_source) so the gate here and in PlanStore
        # agree — a substring test would let a forged source self-activate.
        from plans import is_owner_source
        if is_owner_source(source):
            self.store.set_lifecycle(plan.plan_id, "active", source=source)
        elif not plan.requires_owner_approval():
            self.store.set_lifecycle(plan.plan_id, "active", source="planner")
        else:
            self._notify_owner(
                f"Plan proposed by {source}: \"{plan.goal[:120]}\" — "
                f"approve with: plan approve {plan.plan_id}"
            )
        # Index after the activation decision so the fact carries the
        # lifecycle the plan actually starts its life in.
        self._index_plan_in_ltm(plan)

    def _handle_step_done(self, msg: Dict[str, Any]) -> None:
        plan_id = msg.get("plan_id", "")
        step_index = int(msg.get("step_index", -1))
        summary = str(msg.get("summary", ""))[:300]
        plan = self.store.get(plan_id)
        if plan is None:
            return
        self.store.mark_step(plan_id, step_index, "done")
        self.store.record_progress(
            plan_id,
            f"step {step_index} done "
            f"(usefulness={msg.get('usefulness', 0.0):.2f}): {summary}",
        )
        self._dispatched.pop(plan_id, None)
        self._drain_adenosine("plan_step_complete")
        self._stm_fact(
            f"Plan '{plan_id}' completed step {step_index}: "
            f"{summary or 'no summary'}",
            time_frame="dated",
        )
        if plan.all_steps_done():
            self.store.set_lifecycle(plan_id, "completed", source="planner")
            self.add_to_log(f"Plan '{plan_id}' completed")
            # Refresh the index fact (dated, 'completed') so no stale
            # 'active' version lingers, then store the permanent outcome —
            # HLD: plan outcomes are promoted to permanent on completion.
            self._index_plan_in_ltm(plan)
            self._ltm_fact(
                f"Iyye completed long term plan '{plan_id}': {plan.goal}. "
                f"Steps: {len(plan.steps)}.",
            )
            self._notify_owner(f"Completed plan '{plan_id}': {plan.goal[:120]}")

    # ------------------------------------------------------------------
    # Driving active plans
    # ------------------------------------------------------------------

    def _drive_plans(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)

        def sort_key(p):
            # Alignment-recomputed priority first (HLD: plans subordinate to
            # the motivation system), deadline pressure as tie-breaker.
            due = p.next_due()
            return (-p.priority, due or "9999", p.created_at)

        for plan in sorted(self.store.active_plans(), key=sort_key):
            dispatched_tick = self._dispatched.get(plan.plan_id)
            if dispatched_tick is not None:
                if self._executor_alive(plan.plan_id):
                    continue  # an executor is still working this plan's step
                # No live executor.  Within the grace window the factory may
                # simply not have created it yet — wait.  Past it, the dispatch
                # was dropped or the executor vanished: reset and retry.
                if self._tick - dispatched_tick < self._DISPATCH_GRACE_TICKS:
                    continue
                self._dispatched.pop(plan.plan_id, None)
                step = plan.dispatched_step()
                if step is not None:
                    self.store.mark_step(
                        plan.plan_id, step["step"], "pending")
                    self.add_to_log(
                        f"Executor for '{plan.plan_id}' step "
                        f"{step['step']} missing — step reset for retry")

            step = plan.next_pending_step()
            if step is None:
                if plan.all_steps_done():
                    self.store.set_lifecycle(
                        plan.plan_id, "completed", source="planner")
                continue
            if not self._step_is_due(step, plan, now):
                continue
            if step.get("abstract"):
                if self._submit_decompose(plan, step):
                    return {"action": "decomposing", "plan": plan.plan_id,
                            "step": step["step"]}
                return None  # scheduler busy/paused — retry next drive tick
            return self._dispatch_step(plan, step)
        return None

    def _step_is_due(self, step: Dict[str, Any], plan, now: datetime) -> bool:
        due = step.get("due") or plan.deadline
        if not due:
            return True
        try:
            dt = datetime.fromisoformat(due)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        return dt <= now + self._DUE_LOOKAHEAD

    def _step_due_urgency(self, step: Dict[str, Any], plan) -> float:
        """Urgency of one step from its due date (falling back to the plan
        deadline), on the same scale as _deadline_urgency."""
        due = step.get("due") or plan.deadline
        if not due:
            return 0.0
        try:
            dt = datetime.fromisoformat(due)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return 0.0
        remaining = (dt - datetime.now(timezone.utc)).total_seconds()
        return self._urgency_from_seconds(remaining)

    def _dispatch_step(self, plan, step: Dict[str, Any]) -> Dict[str, Any]:
        """Hand one concrete step to StreamFactory for execution."""
        exec_step = {
            "step": 0,
            "description": step.get("description", plan.goal),
            "type": step.get("type", "learning"),
        }
        if step.get("input"):
            exec_step["input"] = step["input"]
        # HLD: any plan step involving a contact must pull context from the
        # Theory of Mind stream before execution.
        contact_ctx = self._contact_context(step)
        if contact_ctx:
            exec_step["input"] = (
                f"{exec_step.get('input', '')}\n\n"
                f"Contact context (Theory of Mind):\n{contact_ctx}"
            ).strip()

        from messaging import Messages
        exec_plan = {
            "source": f"ltp_{plan.plan_id}",
            "primary_goal": max(
                plan.alignment_weights, key=plan.alignment_weights.get,
            ) if plan.alignment_weights else "agency",
            # Executor stream priority (int 1..5 scale used by attention)
            # follows the plan's alignment-recomputed priority, so plan work
            # competes for consciousness with exactly the weight the
            # motivation system gave the plan.
            "priority": max(1, min(5, round(1 + plan.priority * 4))),
            # Deadline pressure travels WITH the work: the executor carries
            # the step's urgency so attention favors the stream that can
            # actually meet the deadline (the planner's own urgency drops to
            # pending-only once this step is dispatched).
            "urgency": self._step_due_urgency(step, plan),
            "steps": [exec_step],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        plan_ref = {"plan_id": plan.plan_id, "step_index": step["step"]}
        self.brain.post_message("stream_factory", Messages.create_for_plan(
            plan=exec_plan, plan_ref=plan_ref,
        ))
        self.store.mark_step(plan.plan_id, step["step"], "dispatched")
        # Record the dispatch tick; liveness is tracked by the executor's
        # _plan_ref (see _executor_alive), not its name.
        self._dispatched[plan.plan_id] = self._tick
        self.add_to_log(
            f"Dispatched '{plan.plan_id}' step {step['step']}: "
            f"{exec_step['description'][:80]}")
        return {"action": "step_dispatched", "plan": plan.plan_id,
                "step": step["step"]}

    def _executor_alive(self, plan_id: str) -> bool:
        """True if a live stream is executing a step for *plan_id*.

        Matched by the executor's structured ``_plan_ref`` (set by the factory
        in _handle_create_for_plan), not by name — so plan 'foo' is never
        confused with plan 'foo_2's executor (the old prefix-match collision)."""
        for s in getattr(self.brain, "streams", []):
            ref = getattr(s, "_plan_ref", None)
            if isinstance(ref, dict) and ref.get("plan_id") == plan_id:
                return True
        return False

    def _contact_context(self, step: Dict[str, Any]) -> str:
        """Theory-of-Mind context for the contact this step involves, or "".

        HLD: "any plan step involving a contact must pull context from the
        Theory of Mind stream before execution."  An explicit ``contact``
        field (owner-authored or emitted by decomposition) is used first;
        when it is absent the step text is scanned against known contact
        names, so the guarantee holds even when decomposition forgot to tag
        the step.
        """
        tom = self.brain.theory_of_mind()
        if tom is None:
            return ""
        cid = None
        contact = (step.get("contact") or "").strip()
        try:
            if contact:
                matches = tom.find_contacts(contact.lower())
                if matches:
                    cid = matches[0][0]
            if cid is None:
                cid = self._detect_contact_in_text(
                    tom, f"{step.get('description', '')} {step.get('input', '')}",
                )
            if cid is not None and hasattr(tom, "get_contact_context"):
                return str(tom.get_contact_context(cid))[:600]
        except Exception as exc:
            log.debug("Planner: ToM context failed for %r: %s",
                      contact or step.get("description", ""), exc)
        return ""

    @staticmethod
    def _detect_contact_in_text(tom, text: str) -> Optional[str]:
        """First known contact whose display name appears as a whole word in
        *text*, or None.  Names shorter than 3 chars and placeholder names
        are skipped to avoid false positives."""
        if not hasattr(tom, "known_contacts"):
            return None
        low = (text or "").lower()
        if not low:
            return None
        for cid, display in tom.known_contacts():
            name = (display or "").strip().lower()
            if len(name) < 3 or name == "unknown":
                continue
            if re.search(rf"\b{re.escape(name)}\b", low):
                return cid
        return None

    # ------------------------------------------------------------------
    # LLM decomposition (lazy: only the next step becomes concrete)
    # ------------------------------------------------------------------

    def _submit_decompose(self, plan, step: Dict[str, Any]) -> bool:
        progress = "\n".join(
            f"[{p['ts']}] {p['text']}" for p in plan.progress[-5:]
        ) or "(none)"
        submitted = self._llm_submit(
            role="fast", kind="plan_decompose",
            call=LLMCall.from_file(
                "plan_decompose",
                goal=plan.goal,
                step_description=step.get("description", ""),
                progress=progress,
                system_context=self._system_context(),
            ),
            client_kwargs={"no_think": True, "max_tokens": 512},
            # Background work: no user is waiting.  A loose budget + low urgency
            # routes this off the interactive models and concentrates the
            # router's optimistic-init exploration on latency-insensitive roles.
            task={"prompt_tokens": 600, "expected_output_tokens": 400,
                  "quality_need": 0.5, "latency_budget_s": 90, "urgency": 0.2},
        )
        if submitted:
            self._decomposing = (plan.plan_id, step["step"])
            self.add_to_log(
                f"Decomposing '{plan.plan_id}' step {step['step']}")
        return submitted

    def _on_decompose_result(self, result) -> Optional[Dict[str, Any]]:
        ref = self._decomposing
        self._decomposing = None
        if ref is None or result.discarded:
            return None
        plan_id, step_index = ref
        steps = self._parse_steps(result.text if result.ok else "")
        if not steps:
            # Codegen-style fallback: keep the plan moving with a single
            # concrete learning step instead of stalling on a parse failure.
            plan = self.store.get(plan_id)
            if plan is None:
                return None
            steps = [{
                "description": f"Gather information towards: {plan.goal[:120]}",
                "type": "learning",
                "input": plan.goal,
            }]
            self.add_to_log(
                f"Decomposition unparseable for '{plan_id}' — using fallback step")
        if self.store.replace_abstract_step(plan_id, step_index, steps):
            self._drain_adenosine("plan_replan")
            self.store.record_progress(
                plan_id,
                f"decomposed step {step_index} into {len(steps)} concrete step(s)",
            )
            return {"action": "decomposed", "plan": plan_id,
                    "steps": len(steps)}
        return None

    @staticmethod
    def _parse_steps(text: str) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []
        for line in (text or "").splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and obj.get("description"):
                step = {
                    "description": str(obj["description"])[:300],
                    "type": obj.get("type", "learning"),
                    "input": str(obj.get("input", ""))[:1000],
                }
                contact = str(obj.get("contact", "")).strip()
                if contact:
                    step["contact"] = contact[:80]
                steps.append(step)
            if len(steps) >= 3:
                break
        return steps

    def _system_context(self) -> str:
        try:
            sr = self.brain.self_reflection_snapshot()
            if sr:
                return str(sr)[:400]
        except Exception:
            pass
        return "(no system description available)"

    # ------------------------------------------------------------------
    # Urgency (attention integration: deadline pressure counts as urgency)
    # ------------------------------------------------------------------

    @staticmethod
    def _urgency_from_seconds(remaining: Optional[float]) -> float:
        """Map seconds-until-deadline to the 0..1 urgency scale attention
        weighs: overdue 1.0, <1h 0.8, <1d 0.4, further 0.1, none 0.0."""
        if remaining is None:
            return 0.0
        if remaining <= 0:
            return 1.0
        if remaining < 3600:
            return 0.8
        if remaining < 86400:
            return 0.4
        return 0.1

    def _deadline_urgency(self) -> float:
        """Urgency from deadlines the planner still has to ACT on.

        Pending steps only: once a step is dispatched, its deadline pressure
        rides on the executor stream (stamped at dispatch) — keeping the
        planner urgent for work already in flight would have attention
        promote the planner, which has nothing left to do, over the executor
        actually doing the work.
        """
        earliest = self.store.next_due_deadline(statuses=("pending",))
        if earliest is None:
            return 0.0
        remaining = (earliest - datetime.now(timezone.utc)).total_seconds()
        return self._urgency_from_seconds(remaining)

    # ------------------------------------------------------------------
    # Sleep review (HLD: replanning happens while dreaming, when it's cheap)
    # ------------------------------------------------------------------

    def sleep_phases(self) -> List[SleepPhase]:
        # Order 75: after replay (70) so the day's facts are folded first,
        # before cleanup (80).
        return [SleepPhase(
            "plan_review", lambda brain: self._sleep_plan_review(), 75,
        )]

    def _sleep_plan_review(self) -> bool:
        """Review one plan per tick; True when the pass is finished."""
        if self._review_pending is None:
            self._review_pending = [p.plan_id for p in self.store.all_plans()]
        if not self._review_pending:
            self._review_pending = None
            self._dispatched.clear()
            return True
        plan_id = self._review_pending.pop(0)
        plan = self.store.get(plan_id)
        if plan is None:
            return False
        try:
            self._review_one(plan)
        except Exception as exc:
            log.warning("Plan review failed for %s: %s", plan_id, exc)
        return False

    def _review_one(self, plan) -> None:
        # Proposal expiry: an unapproved proposal the owner has ignored for
        # _PROPOSAL_EXPIRY_DAYS is abandoned (falling through to the archive
        # branch below).  Without this, proposals pile up forever AND — since
        # create() dedups against every non-terminal plan — a stale proposal
        # permanently blocks re-proposing the same goal.  Abandoning frees
        # the fingerprint for a future, possibly better-timed proposal.
        if plan.lifecycle == "proposed":
            try:
                created = datetime.fromisoformat(plan.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - created
            except ValueError:
                age = timedelta(0)
            if age > timedelta(days=self._PROPOSAL_EXPIRY_DAYS):
                self.store.set_lifecycle(
                    plan.plan_id, "abandoned", source="planner_sleep_review")
                self.store.record_progress(
                    plan.plan_id,
                    f"expired at sleep review: unapproved for {age.days} days",
                )
                self._index_plan_in_ltm(plan)
                self.add_to_log(
                    f"Expired unapproved proposal '{plan.plan_id}' "
                    f"({age.days} days old)")
                # fall through: now abandoned → archived below this pass

        # Finished plans: summarize into LTM, then archive the directory.
        if plan.lifecycle in ("completed", "abandoned"):
            outcome = ("achieved its goal" if plan.lifecycle == "completed"
                       else "was abandoned")
            self._ltm_fact(
                f"Long term plan '{plan.plan_id}' ({plan.goal}) {outcome} "
                f"after {len(plan.progress)} recorded progress events.",
            )
            self.store.archive(plan.plan_id)
            self.add_to_log(f"Archived plan '{plan.plan_id}'")
            return
        if plan.lifecycle != "active":
            return
        now = datetime.now(timezone.utc)
        # Orphaned dispatches: the cycle's executors are gone now.
        for step in plan.steps:
            if step.get("status") == "dispatched":
                self.store.mark_step(plan.plan_id, step["step"], "pending")
        # Staleness: no progress for _STALE_DAYS → suspend, owner can resume.
        idle_days = self._days_since(
            plan.last_progress_at or plan.lifecycle_changed_at, now)
        if idle_days > self._STALE_DAYS:
            self.store.set_lifecycle(
                plan.plan_id, "suspended", source="planner_sleep_review")
            self.store.record_progress(
                plan.plan_id,
                f"suspended at sleep review: no progress for {idle_days:.0f} days",
            )
            self._index_plan_in_ltm(plan)
            self.add_to_log(f"Suspended stale plan '{plan.plan_id}'")
            return  # suspended — no longer an active replan candidate
        # Replanning (HLD: dreaming runs a replanning pass) — but only at the
        # plan's review cadence, so a slow plan isn't churned every dream.
        if self._review_due(plan, now):
            self.store.mark_reviewed(plan.plan_id, now.isoformat())
            self._journal_plan_review(plan, now)

    def _review_due(self, plan, now: datetime) -> bool:
        """Whether the plan's review cadence has elapsed since its last review."""
        if not plan.last_reviewed_at:
            return True
        return self._days_since(plan.last_reviewed_at, now) >= plan.review_cadence_days

    # ------------------------------------------------------------------
    # Replanning assessment (Phase A: detect + journal, no mutations)
    # ------------------------------------------------------------------

    @staticmethod
    def _days_since(iso: Optional[str], now: datetime) -> float:
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return max(0.0, (now - dt).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 0.0

    # STM time_frames that signal recently-learned context (vs stable
    # biographical/permanent facts the plan already accounts for).
    _RECENT_TIME_FRAMES = frozenset({"session", "today", "recent"})

    def _recall_for_plan(self, plan) -> List[Any]:
        """Memory relevant to the plan's goal, via unified recall (LTM/STM/ToM).
        The day's facts are already consolidated into LTM by the replay phase
        (order 70) that runs before this one.  Queried once per review and
        reused by both the assessment and the revision."""
        try:
            from recall import Recall
            return Recall(self.brain).query(plan.goal, limit=6)
        except Exception as exc:
            log.debug("Replan: recall failed for %s: %s", plan.plan_id, exc)
            return []

    def _assess_replan(self, plan, now: datetime, recalled) -> Dict[str, Any]:
        """Decide whether an active plan is a replan candidate, and why.

        Cheap, grounded signals (no LLM): drifting-but-not-yet-stale, fresh
        relevant context, or deadline pressure with pending work."""
        reasons: List[str] = []
        days_since = self._days_since(
            plan.last_progress_at or plan.lifecycle_changed_at, now)
        if self._REPLAN_STALL_DAYS <= days_since < self._STALE_DAYS:
            reasons.append("stalled")
        fresh = sum(1 for r in recalled
                    if (getattr(r, "time_frame", "") or "") in self._RECENT_TIME_FRAMES)
        if fresh:
            reasons.append("new_context")
        if plan.next_pending_step() is not None and plan.next_due() is not None:
            reasons.append("deadline")
        return {
            "candidate": bool(reasons),
            "reasons": reasons,
            "days_since_progress": round(days_since, 1),
            "fresh_facts": fresh,
            "pending_steps": sum(
                1 for s in plan.steps if s.get("status") == "pending"),
        }

    def _journal_plan_review(self, plan, now: datetime) -> Dict[str, Any]:
        """Assess one active plan, journal the assessment, and — for a
        candidate — run the Phase B revision."""
        recalled = self._recall_for_plan(plan)
        a = self._assess_replan(plan, now, recalled)
        from event_journal import emit
        emit(getattr(self.brain, "journal", None), "plan_review",
             plan_id=plan.plan_id, lifecycle=plan.lifecycle,
             candidate=a["candidate"], reasons=a["reasons"],
             days_since_progress=a["days_since_progress"],
             fresh_facts=a["fresh_facts"], pending_steps=a["pending_steps"])
        if a["candidate"]:
            self.add_to_log(
                f"Replan candidate '{plan.plan_id}': {', '.join(a['reasons'])}")
            self._revise_plan(plan, now, recalled)
        return a

    # ------------------------------------------------------------------
    # Replanning revision (Phase B: synchronous-LLM revise, guarded)
    # ------------------------------------------------------------------

    def _sync_llm_client(self):
        """A synchronous LLM client usable during sleep (the scheduler is
        paused).  Prefers a reasoning model, falls back to the same accessor
        replay uses; None if no client can be built (caller keeps the plan)."""
        router = getattr(self.brain, "llm_router", None)
        if router is not None:
            for role in ("reasoning", "stm"):
                try:
                    c = router.get_client(role=role, no_think=True)
                    if c is not None:
                        return c
                except Exception:
                    continue
        getter = getattr(self.brain, "_get_replay_extraction_client", None)
        return getter() if callable(getter) else None

    def _revise_plan(self, plan, now: datetime, recalled) -> None:
        """Ask the model whether the plan's pending steps still fit and revise
        them if not.  Conservative: malformed/keep output changes nothing.
        Idempotent per dream via last_replanned_cycle (restart-safe)."""
        cycle = getattr(getattr(self.brain, "journal", None), "cycle_id", None)
        if cycle is not None and plan.last_replanned_cycle == cycle:
            return  # already revised this dream
        client = self._sync_llm_client()
        if client is None:
            return
        try:
            raw = client.complete_from_file(
                "plan_revise", **self._revise_inputs(plan, recalled))
        except Exception as exc:
            log.debug("Replan revise LLM failed for %s: %s", plan.plan_id, exc)
            return
        revision = self._parse_revision(raw)
        if revision is None:
            return  # 'keep' / malformed — the conservative default
        action = revision["action"]
        if action == "revise":
            self._apply_revision(plan, revision["steps"], cycle)
        elif action in ("complete", "abandon"):
            self._resolve_plan(plan, action, revision.get("reason", ""), cycle)

    def _revise_inputs(self, plan, recalled) -> Dict[str, str]:
        from recall import Recall
        steps_text = "\n".join(
            f"- [{s.get('status')}] {s.get('description', '')}"
            for s in plan.steps) or "(none)"
        progress = "\n".join(
            f"[{p['ts']}] {p['text']}" for p in plan.progress[-5:]) or "(none)"
        try:
            facts = Recall.render(recalled)
        except Exception:
            facts = "(none)"
        return {"goal": plan.goal, "current_steps": steps_text,
                "progress": progress, "recent_facts": facts}

    @staticmethod
    def _parse_revision(raw: str) -> Optional[Dict[str, Any]]:
        """Parse the revise model output: {"action":"keep"} or
        {"action":"revise","steps":[...]}.  Returns None for anything malformed
        or an empty revision — the caller treats that as 'keep'."""
        text = (raw or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            return None
        if obj.get("action") == "keep":
            return {"action": "keep"}
        if obj.get("action") == "revise":
            steps = [s for s in obj.get("steps", [])
                     if isinstance(s, dict) and str(s.get("description", "")).strip()]
            if not steps:
                return None
            return {"action": "revise", "steps": steps[:5]}
        if obj.get("action") in ("complete", "abandon"):
            return {"action": obj["action"],
                    "reason": str(obj.get("reason", ""))[:300]}
        return None

    def _apply_revision(self, plan, revised_steps, cycle) -> None:
        """Replace the pending tail with the revised steps, guarded:
        preserve the done prefix and old steps in provenance, and — the
        security guard — if the revision newly requires owner approval
        (introduces external reach) on an active plan, suspend it for the owner
        rather than silently activating the new reach."""
        old_pending = [s.get("description", "")
                       for s in plan.steps if s.get("status") == "pending"]
        req_before = plan.requires_owner_approval()
        new_pending: List[Dict[str, Any]] = []
        for s in revised_steps:
            ns = {"description": str(s.get("description", ""))[:300],
                  "type": s.get("type", "learning"),
                  "input": str(s.get("input", ""))[:1000],
                  "status": "pending"}
            if s.get("contact"):
                ns["contact"] = str(s["contact"])[:80]
            new_pending.append(ns)
        plan = self.store.revise_pending_steps(plan.plan_id, new_pending)
        if plan is None:
            return  # empty revision — kept
        plan.last_replanned_cycle = cycle
        self._drain_adenosine("plan_replan")
        self.store.record_progress(
            plan.plan_id,
            f"dreaming replan: replaced {len(old_pending)} pending step(s) with "
            f"{len(new_pending)} (old: {'; '.join(old_pending)[:200]})")
        req_after = plan.requires_owner_approval()
        escalated = req_after and not req_before and plan.lifecycle == "active"
        if escalated:
            self.store.set_lifecycle(
                plan.plan_id, "suspended", source="planner_sleep_review")
            self.store.record_progress(
                plan.plan_id,
                "suspended: dreaming replan introduced external-reaching steps "
                "needing owner approval")
            self._index_plan_in_ltm(plan)
            self._notify_owner(
                f"Plan '{plan.plan_id}' was replanned with steps that reach "
                f"beyond local chat — approve to resume: plan approve {plan.plan_id}")
            self.add_to_log(
                f"Replan of '{plan.plan_id}' escalated capability — suspended "
                f"for owner approval")
        else:
            self.add_to_log(
                f"Replanned '{plan.plan_id}': {len(new_pending)} new pending step(s)")
        from event_journal import emit
        emit(getattr(self.brain, "journal", None), "plan_revised",
             plan_id=plan.plan_id, old_pending=len(old_pending),
             new_pending=len(new_pending), escalated=bool(escalated),
             lifecycle=plan.lifecycle)

    def _resolve_plan(self, plan, action: str, reason: str, cycle) -> None:
        """Phase C: dreaming judged the goal already achieved or now moot.

        SAFETY: never autonomously complete or abandon the owner's plan — those
        are terminal and hard to reverse, and an LLM judgment can be wrong.
        Instead SUSPEND (stops wasted execution, fully owner-reversible via
        ``plan approve`` / ``plan abandon``) and surface the judgment for the
        owner to confirm.  Recorded durably in the plan log and LTM so it
        survives even if the chat notification doesn't reach the owner asleep."""
        plan.last_replanned_cycle = cycle
        verdict = ("goal already achieved" if action == "complete"
                   else "goal no longer relevant")
        self.store.set_lifecycle(
            plan.plan_id, "suspended", source="planner_sleep_review")
        self.store.record_progress(
            plan.plan_id,
            f"dreaming judged {verdict}: {reason[:200]} — suspended for owner "
            f"confirmation")
        self._index_plan_in_ltm(plan)
        self._ltm_fact(
            f"During dreaming, Iyye judged long term plan '{plan.plan_id}' "
            f"({plan.goal}): {verdict}. Awaiting owner confirmation."
            + (f" Reason: {reason[:160]}" if reason else ""))
        self._notify_owner(
            f"Plan '{plan.plan_id}': I think the {verdict}"
            + (f" — {reason[:120]}" if reason else "")
            + f". Paused it; 'plan approve {plan.plan_id}' to resume or "
            f"'plan abandon {plan.plan_id}' to close it.")
        self._drain_adenosine("plan_replan")
        self.add_to_log(
            f"Resolved '{plan.plan_id}' as {verdict} → suspended for owner")
        from event_journal import emit
        emit(getattr(self.brain, "journal", None), "plan_resolved",
             plan_id=plan.plan_id, verdict=action, reason=reason[:120])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_orphaned_dispatches(self) -> None:
        for plan in self.store.active_plans():
            for step in plan.steps:
                if step.get("status") == "dispatched":
                    self.store.mark_step(plan.plan_id, step["step"], "pending")

    def _drain_adenosine(self, activity: str) -> None:
        try:
            adenosine = self.brain.adenosine
            if adenosine is not None:
                adenosine.drain_activity(activity)
        except Exception:
            pass

    def _stm_fact(self, text: str, time_frame: str = "dated") -> None:
        stm = getattr(self.brain, "stm", None)
        if stm is None:
            return
        try:
            stm.add_fact(text, confidence=0.9, provenance="planner",
                         time_frame=time_frame)
        except Exception as exc:
            log.debug("Planner: STM write failed: %s", exc)

    def _ltm_fact(
        self,
        text: str,
        time_frame: str = "permanent",
        provenance: str = "long term plan outcome",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        memory = getattr(self.brain, "memory", None)
        if memory is None:
            return
        try:
            memory.store_fact(text, confidence=0.9, source="planner",
                              provenance=provenance, time_frame=time_frame,
                              metadata=metadata)
        except Exception as exc:
            log.debug("Planner: LTM write failed: %s", exc)

    def _index_plan_in_ltm(self, plan) -> None:
        """HLD: plans are "indexed in LTM so it is searchable alongside
        facts" — for their whole life, not just at completion.

        One canonical fact per plan, re-stored whenever the lifecycle
        changes; the LTM client's semantic dedup updates the existing row
        (newer text, merged provenance) rather than piling up versions.
        ``dated`` while the plan lives (HLD: plan-derived facts default to
        dated); completion/archive separately store the permanent outcome.
        Priority is deliberately not in the text — it changes every
        alignment cycle and would churn the index.
        """
        self._ltm_fact(
            f"Iyye long term plan '{plan.plan_id}' is {plan.lifecycle}: "
            f"{plan.goal}",
            time_frame="dated",
            provenance=f"plan proposed by {plan.provenance}",
            metadata={"kind": "long_term_plan", "plan_id": plan.plan_id},
        )

    def _notify_owner(self, text: str) -> None:
        push = getattr(self.brain, "_push_to_web_chat", None)
        if callable(push):
            try:
                push(f"[planner] {text}")
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["active_plans"] = [p.plan_id for p in self.store.active_plans()]
        return state
