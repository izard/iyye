# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Typed inter-stream messages — the mailbox communication contract.

HLD: streams communicate through the brain mailbox.  Historically messages
were free-form dicts validated ad hoc inside each recipient's
``_handle_message``, so the set of valid (target, action, fields) lived only in
scattered call sites.  This module makes the mailbox a *contract*: every
message is a typed :class:`Message` built through a validated constructor in
:class:`Messages`, and the full registry of known actions lives in one place
(:data:`KNOWN_ACTIONS`).

For a smooth migration the :class:`Message` object quacks like the old dict
(``.get``, ``[]``, ``in``), so existing ``_handle_message`` bodies keep working
unchanged while senders move to the typed constructors.  ``normalize_message``
wraps any legacy dict and warns on an unregistered action so drift is visible.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

log = logging.getLogger("Iyye.Messaging")

# action -> (target stream, required field names).  The single source of truth
# for what may travel the mailbox; `urgent` is a universal optional flag.
KNOWN_ACTIONS: Dict[str, Dict[str, Any]] = {
    # → llm_management
    "restart":      {"target": "llm_management", "required": ()},
    "start":        {"target": "llm_management", "required": ("script",)},
    "stop":         {"target": "llm_management", "required": ("name",)},
    "ensure_role":  {"target": "llm_management", "required": ("role",)},
    # → stream_factory
    "suggest_stream":   {"target": "stream_factory", "required": ()},
    "stream_completed": {"target": "stream_factory",
                         "required": ("stream_name", "usefulness")},
    "create_for_plan":  {"target": "stream_factory",
                         "required": ("plan", "plan_ref")},
    # → theory_of_mind
    "interaction":  {"target": "theory_of_mind", "required": ("contact_id",)},
    # → planner.  ``source`` on lifecycle messages is stamped by shipped code
    # (never LLM output) — PlanStore gates approval on it (see plans.py).
    "plan_propose":   {"target": "planner", "required": ("goal", "source")},
    "plan_approve":   {"target": "planner", "required": ("plan_id", "source")},
    "plan_abandon":   {"target": "planner", "required": ("plan_id", "source")},
    "plan_step_done": {"target": "planner",
                       "required": ("plan_id", "step_index")},
}


class Message:
    """A typed mailbox message that still behaves like the legacy dict.

    Construct via :class:`Messages` (validated) rather than directly."""

    __slots__ = ("_data",)

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    # -- typed accessors --
    @property
    def action(self) -> Optional[str]:
        return self._data.get("action")

    @property
    def urgent(self) -> bool:
        return bool(self._data.get("urgent"))

    @property
    def data(self) -> Dict[str, Any]:
        return self._data

    # -- dict-compatible surface (so _handle_message bodies are unchanged) --
    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Message(action={self.action!r}, urgent={self.urgent})"


def make_message(action: str, urgent: bool = False, **fields: Any) -> Message:
    """Build a validated :class:`Message` for *action*.

    Unknown actions and missing required fields are logged (not raised) so a
    messaging mistake degrades to a visible warning rather than crashing the
    cognitive loop."""
    spec = KNOWN_ACTIONS.get(action)
    if spec is None:
        log.warning("Unknown mailbox action %r — delivering unvalidated", action)
    else:
        missing = [f for f in spec["required"] if f not in fields or fields[f] is None]
        if missing:
            log.warning("Message %r missing required field(s): %s",
                        action, ", ".join(missing))
    data = {"action": action, **fields}
    if urgent:
        data["urgent"] = True
    return Message(data)


def normalize_message(target: str, message: Any) -> Message:
    """Coerce a Message or legacy dict into a validated Message.

    Used by ``brain.post_message`` so even un-migrated dict senders get the
    contract check (and a warning if the action isn't registered)."""
    if isinstance(message, Message):
        return message
    if isinstance(message, dict):
        action = message.get("action")
        spec = KNOWN_ACTIONS.get(action)
        if spec is None:
            log.warning("post_message(%r): unregistered action %r", target, action)
        elif spec["target"] != target:
            log.warning("post_message(%r): action %r is registered for %r",
                        target, action, spec["target"])
        return Message(dict(message))
    log.warning("post_message(%r): non-dict message %r — wrapping", target, type(message))
    return Message({"action": None, "payload": message})


class Messages:
    """Typed constructors for every known mailbox message.

    Senders use these instead of dict literals so field names and required
    arguments are checked at the call site."""

    @staticmethod
    def restart_llm(role: str = "chat", reason: str = "", urgent: bool = False) -> Message:
        return make_message("restart", urgent=urgent, role=role, reason=reason)

    @staticmethod
    def start_llm(script: str) -> Message:
        return make_message("start", script=script)

    @staticmethod
    def stop_llm(name: str) -> Message:
        return make_message("stop", name=name)

    @staticmethod
    def ensure_role(role: str, model_name: Optional[str] = None,
                    task: Optional[Dict[str, Any]] = None, reason: str = "") -> Message:
        return make_message("ensure_role", role=role, model_name=model_name,
                            task=task, reason=reason)

    @staticmethod
    def suggest_stream(reason: str = "", sensor: Optional[str] = None,
                       goal: str = "curiosity", evidence_ids=None,
                       evidence_texts=None) -> Message:
        return make_message("suggest_stream", reason=reason, sensor=sensor,
                            goal=goal, evidence_ids=evidence_ids,
                            evidence_texts=evidence_texts)

    @staticmethod
    def stream_completed(stream_name: str, usefulness: float,
                         vague_outputs: int = 0, total_outputs: int = 0) -> Message:
        return make_message("stream_completed", stream_name=stream_name,
                            usefulness=usefulness, vague_outputs=vague_outputs,
                            total_outputs=total_outputs)

    @staticmethod
    def interaction(contact_id: str, **fields: Any) -> Message:
        return make_message("interaction", contact_id=contact_id, **fields)

    @staticmethod
    def create_for_plan(plan: Dict[str, Any], plan_ref: Dict[str, Any]) -> Message:
        return make_message("create_for_plan", plan=plan, plan_ref=plan_ref)

    @staticmethod
    def plan_propose(goal: str, source: str, steps=None, deadline=None,
                     alignment_weights=None) -> Message:
        return make_message("plan_propose", goal=goal, source=source,
                            steps=steps, deadline=deadline,
                            alignment_weights=alignment_weights)

    @staticmethod
    def plan_approve(plan_id: str, source: str) -> Message:
        return make_message("plan_approve", plan_id=plan_id, source=source)

    @staticmethod
    def plan_abandon(plan_id: str, source: str) -> Message:
        return make_message("plan_abandon", plan_id=plan_id, source=source)

    @staticmethod
    def plan_step_done(plan_id: str, step_index: int, usefulness: float = 0.0,
                       summary: str = "") -> Message:
        return make_message("plan_step_done", plan_id=plan_id,
                            step_index=step_index, usefulness=usefulness,
                            summary=summary)


__all__ = ["Message", "Messages", "make_message", "normalize_message", "KNOWN_ACTIONS"]
