# Copyright 2026 Alexander Komarov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Capability-scoped handles for LLM-generated streams.

The HLD describes streams using tools, sensors, actuators, privileged users
and Python, but historically a generated stream received the *raw* brain plus
a context dict containing every actuator, every other stream object, and the
raw long-term-memory client.  AST validation (streams/stream_factory.py) blocks
OS/network/`exec` escape but says nothing about that capability surface, so
LLM-authored code could (while passing validation) message Telegram users,
pollute/delete LTM, grant itself trust, start/stop LLMs, or mutate other
streams.

This module gives generated streams *least privilege* by replacing the raw
handles with scoped façades:

* :class:`ScopedBrain`   — exposes only the journal (so base-class logging keeps
  working); every other brain attribute raises ``AttributeError``.
* :class:`ReadOnlyMemory`— long-term memory search/count only; no store/delete.
* :class:`Capabilities`  — the handle passed as ``context['cap']`` that a
  generated stream is expected to use: scoped STM, read-only memory search,
  activity logging, and (graduated tier only) a mediated, rate-limited
  ``emit``.

Grants widen with the candidate→graduated lifecycle: a *candidate* can read
its inputs and write session-scoped STM facts but cannot reach any actuator; a
*graduated* stream (proven useful over several cycles) additionally gets a
single mediated output channel.  Shipped/reviewed streams are unaffected — only
generated code (identified by ``_source_file``) is scoped.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("Iyye.Capabilities")


class ReadOnlyMemory:
    """Long-term-memory façade granting search/count but never writes."""

    __slots__ = ("_mem",)

    def __init__(self, memory: Any) -> None:
        self._mem = memory

    def search(self, *a, **kw):
        return self._mem.search(*a, **kw)

    def count(self):
        try:
            return self._mem.count()
        except Exception:
            return 0

    # Explicitly deny mutation so a generated stream cannot pollute LTM.
    def store_fact(self, *a, **kw):
        raise PermissionError("generated streams may not write long-term memory")

    def delete_fact(self, *a, **kw):
        raise PermissionError("generated streams may not delete long-term memory")


class ScopedBrain:
    """Minimal brain façade for generated streams.

    Only attributes in the allow-list are reachable; everything else raises
    ``AttributeError``.  The base ``ProcessingStream.add_to_log`` reaches
    ``self.brain.journal`` to record activity — that is the one capability a
    generated stream needs from the brain, and it is append-only.  All the
    dangerous surfaces (``post_message``, ``llm_router``, ``_tom_stream``,
    ``streams``, ``actuators``, ``sensors``, raw ``memory``) are absent.
    """

    _ALLOWED = frozenset({"journal"})

    def __init__(self, brain: Any) -> None:
        # Bypass __getattr__ for our own bookkeeping attribute.
        object.__setattr__(self, "_brain", brain)

    def __getattr__(self, name: str) -> Any:
        if name in ScopedBrain._ALLOWED:
            return getattr(object.__getattribute__(self, "_brain"), name)
        raise AttributeError(
            f"generated stream may not access brain.{name} (capability denied)"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("generated stream may not mutate the brain")


class Capabilities:
    """The scoped handle a generated stream uses (``context['cap']``).

    Exposes exactly what the stream's lifecycle tier grants.  Built by
    :func:`build_generated_context`; candidates get read + scoped STM write,
    graduated streams additionally get :meth:`emit`.
    """

    __slots__ = ("_stm", "_memory", "_stream", "_emit_fn", "tier")

    def __init__(
        self,
        stm: Any,
        memory: ReadOnlyMemory,
        stream: Any,
        tier: str,
        emit_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._stm = stm
        self._memory = memory
        self._stream = stream
        self.tier = tier
        self._emit_fn = emit_fn

    # -- read --
    def search_memory(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search long-term memory (read-only)."""
        return self._memory.search(query, limit=limit)

    def recent_facts(self, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            return self._stm.get_recent(limit)
        except Exception:
            return []

    # -- scoped write --
    def add_fact(self, text: str, confidence: float = 0.7,
                 time_frame: Optional[str] = None) -> Optional[str]:
        """Store a short-term-memory fact through the stream's scoped STM
        wrapper (session for candidates, durable for graduated)."""
        try:
            return self._stm.add_fact(text=text, confidence=confidence,
                                      time_frame=time_frame)
        except Exception as exc:
            log.debug("cap.add_fact failed: %s", exc)
            return None

    def log(self, message: str) -> None:
        """Record an activity-log line (journaled)."""
        try:
            self._stream.add_to_log(message)
        except Exception:
            pass

    # -- mediated output (graduated tier only; Phase 2) --
    def emit(self, text: str) -> bool:
        """Send a user-visible message through the mediated output channel.

        Returns False (no-op) when the stream's tier does not grant output —
        candidates cannot reach any actuator."""
        if self._emit_fn is None:
            log.debug("cap.emit denied for tier=%s", self.tier)
            return False
        return self._emit_fn(text)


class MediatedEmitter:
    """A rate-limited, single-channel output grant for graduated streams.

    Routes through one chosen actuator (default: the local web chat — never
    Telegram/external by default) and caps the number of messages per awake
    cycle so a graduated stream cannot spam.  Reset each cycle by the brain.
    """

    def __init__(self, actuator: Any, name: str, max_per_cycle: int = 5) -> None:
        self._actuator = actuator
        self._name = name
        self._max = max_per_cycle
        self._sent = 0

    def reset(self) -> None:
        self._sent = 0

    def __call__(self, text: str) -> bool:
        if self._actuator is None or self._sent >= self._max:
            return False
        try:
            self._actuator.actuate(str(text))
            self._sent += 1
            return True
        except Exception as exc:
            log.debug("MediatedEmitter(%s) failed: %s", self._name, exc)
            return False


# ======================================================================
# Declared capability profiles (uniform "this actor may call Z" model)
# ======================================================================

class CapabilityProfile:
    """A named, declarative grant of which action types an actor may perform.

    Used at the trust boundary that actually matters — the chat streams, where
    untrusted Telegram users must be denied privileged tools (Python, LLM admin)
    while the local web owner gets everything (HLD §9: "privileged users have
    access to all tools including python interpreter").  ``allow_all`` models
    the privileged tier without enumerating every present and future action."""

    __slots__ = ("name", "_actions", "_allow_all")

    def __init__(self, name: str, actions=(), allow_all: bool = False) -> None:
        self.name = name
        self._actions = frozenset(actions)
        self._allow_all = allow_all

    def allows(self, action_type: Optional[str]) -> bool:
        if self._allow_all:
            return True
        return action_type in self._actions

    def __repr__(self) -> str:
        scope = "ALL" if self._allow_all else sorted(self._actions)
        return f"CapabilityProfile({self.name!r}, {scope})"


# Chat action tiers.  Trust management is anchored to the *channel*, not the
# contact: the ONLY way a Telegram user becomes trusted is an explicit command
# from the local web chat (127.0.0.1, the machine owner).  Remote sources can
# never change trust — not even already-trusted Telegram contacts (delegation
# was removed: it allowed escalation by talking the LLM into a trust/persona
# action).  Hence three tiers:
#
# * untrusted remote  — read-only lookups only.
# * trusted remote    — adds the privileged tools (python, llm admin) per HLD
#                       §9, but still no trust/untrust/persona.  Explicit
#                       enumeration, not allow_all, so future actions are
#                       denied-by-default for remote senders.
# * local owner       — everything, including trust/untrust/persona.
CHAT_UNTRUSTED = CapabilityProfile(
    "chat_untrusted", {"wikipedia", "url"},
)
CHAT_TRUSTED_REMOTE = CapabilityProfile(
    # "plan" for a trusted remote contact means propose/list only: the
    # approve/abandon sub-commands are additionally source-gated inside
    # UserChatStream._execute_plan_action and again in PlanStore.
    "chat_trusted_remote", {"wikipedia", "url", "python", "llm", "plan"},
)
CHAT_LOCAL_OWNER = CapabilityProfile("chat_local_owner", allow_all=True)
# Backward-compat alias (old name for the allow-all tier).
CHAT_PRIVILEGED = CHAT_LOCAL_OWNER


def chat_profile(trusted: bool, local: bool = False) -> CapabilityProfile:
    """Return the chat capability profile for the current source.

    *local* is True only for the web chat bound to 127.0.0.1 — the machine
    owner.  Only that profile may execute trust/untrust/persona actions."""
    if local:
        return CHAT_LOCAL_OWNER
    return CHAT_TRUSTED_REMOTE if trusted else CHAT_UNTRUSTED


# Declarative record of what each shipped (reviewed, trusted) stream is
# entitled to.  Documentation + a single place to see the capability map;
# enforcement is applied where untrusted input meets capability — generated
# streams (ScopedBrain/Capabilities, above) and chat (chat_profile).  Shipped
# subconscious streams run reviewed code and keep full brain access.
SHIPPED_STREAM_CAPABILITIES: Dict[str, str] = {
    "user_chat":        "chat I/O; tools (python/llm/research) gated by chat_profile(trusted, local); trust/persona local-web-chat only",
    "stream_factory":   "create / refine / graduate generated streams; codegen",
    "llm_management":   "start / stop / route LLMs",
    "theory_of_mind":   "contact list, trust flags, psychological profiles",
    "self_reflection":  "system introspection; posts stream-creation suggestions",
    "attention_stream": "promote/demote conscious stream (no external I/O)",
    "alignment_stream": "score streams against goals (LLM read)",
    "adenosine_stream": "owns the tiredness metric",
    "stm_update":       "extract facts from stream activity into STM",
    "planner":          "owns the long term plan store; dispatches plan steps to stream_factory; lifecycle changes gated by source (owner approval local-web-chat only)",
}


__all__ = [
    "ReadOnlyMemory",
    "ScopedBrain",
    "Capabilities",
    "MediatedEmitter",
    "CapabilityProfile",
    "CHAT_UNTRUSTED",
    "CHAT_TRUSTED_REMOTE",
    "CHAT_LOCAL_OWNER",
    "CHAT_PRIVILEGED",
    "chat_profile",
    "SHIPPED_STREAM_CAPABILITIES",
]
