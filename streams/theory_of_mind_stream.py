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
# streams/theory_of_mind_stream.py
#!/usr/bin/env python3
"""
Theory of Mind Stream — Dedicated Theory of Mind stream, owns contact list of humans and agents
with linked social interactions history and inferred psychological details.
Chat handling streams extract relevant context from context of this stream."
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from iyye_base import PROJECT_ROOT, ProcessingStream
from llm_scheduler import LLMCall, LLMConsumerMixin

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")

_CONTACTS_PATH = PROJECT_ROOT / "iyye_memory" / "contacts.json"


class TheoryOfMindStream(LLMConsumerMixin, ProcessingStream):
    """
    Maintains a contact list of humans and agents with social interaction
    history and inferred psychological profiles.  Provides a synchronous
    query interface (get_contact_context) for chat streams.
    """

    _PROFILE_INTERVAL = 30   # ticks between LLM profile update attempts
    _SAVE_INTERVAL = 3       # ticks between disk saves (when dirty)
    _MAX_INTERACTIONS = 50   # stored interactions per contact (oldest trimmed)
    _PROFILE_MIN_NEW = 3     # new interactions needed to trigger re-profiling

    def __init__(self, brain: "IyyeBrain") -> None:
        super().__init__(name="theory_of_mind")
        self.brain = brain
        self.priority = 2
        self._can_be_conscious = True

        self._contacts: Dict[str, Dict[str, Any]] = {}
        self._profile_tick: int = 0
        self._save_tick: int = 0
        self._dirty: bool = False
        # Context for the profile inference currently submitted to the async
        # scheduler (or None).  Stashed by contact id so the result is applied
        # against the live contact map even if it changed while the job ran.
        self._pending_profile: Optional[Dict[str, Any]] = None

        self._load_contacts()

        # Expose on brain so chat streams can call get_contact_context directly.
        brain._tom_stream = self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_contacts(self) -> None:
        if not _CONTACTS_PATH.exists():
            return
        try:
            data = json.loads(_CONTACTS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._contacts = data
                log.info("TheoryOfMind: loaded %d contact(s)", len(data))
        except Exception as exc:
            log.warning("TheoryOfMind: could not load contacts: %s", exc)

    def _save_contacts(self) -> None:
        try:
            _CONTACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            _CONTACTS_PATH.write_text(
                json.dumps(self._contacts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._dirty = False
            log.debug("TheoryOfMind: saved %d contact(s)", len(self._contacts))
        except Exception as exc:
            log.warning("TheoryOfMind: could not save contacts: %s", exc)

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 1. Drain mailbox — interaction reports from chat streams.
        for msg in self.brain.drain_messages("theory_of_mind"):
            self._handle_message(msg)

        self._profile_tick += 1
        self._save_tick += 1

        # 2. Apply a finished profile inference (ran on a scheduler worker, not
        #    the main loop).
        result = self._llm_poll()
        if result is not None:
            self._on_profile_result(result)

        # 3. Periodically submit a profile refresh for one stale contact/persona,
        #    unless one is already in flight (one-job-per-stream rule).
        if self._profile_tick >= self._PROFILE_INTERVAL and not self._llm_busy():
            self._profile_tick = 0
            self._submit_stale_profile()

        # 4. Persist to disk when dirty.
        if self._dirty and self._save_tick >= self._SAVE_INTERVAL:
            self._save_tick = 0
            self._save_contacts()

        self.checkpoint()
        return None

    # ------------------------------------------------------------------
    # Mailbox handling
    # ------------------------------------------------------------------

    def _handle_message(self, msg: Dict[str, Any]) -> None:
        action = msg.get("action")
        if action == "interaction":
            self._record_interaction(msg)
        else:
            log.debug("TheoryOfMind: unknown action %r", action)

    def _record_interaction(self, msg: Dict[str, Any]) -> None:
        contact_id = msg.get("contact_id", "unknown")
        now = msg.get("timestamp", datetime.now(timezone.utc).isoformat())

        contact = self._contacts.get(contact_id)
        if contact is None:
            contact = {
                "contact_id": contact_id,
                "display_name": msg.get("display_name") or "unknown",
                "source": msg.get("source", ""),
                "chat_id": msg.get("chat_id"),
                "first_seen": now,
                "last_seen": now,
                "interaction_count": 0,
                "interactions": [],
                "psychological_profile": None,
            }
            self._contacts[contact_id] = contact
            self.add_to_log(f"New contact: {contact['display_name']} ({contact_id})")
            # Persist to STM so the fact reaches LTM via sleep replay.
            stm = getattr(self.brain, 'stm', None)
            if stm:
                stm.add_fact(
                    f"New contact: {contact['display_name']} via {contact['source']}.",
                    confidence=1.0,
                    provenance="theory_of_mind",
                    time_frame="permanent",
                )

        contact["last_seen"] = now
        contact["interaction_count"] += 1
        if msg.get("display_name") and msg["display_name"] != "unknown":
            contact["display_name"] = msg["display_name"]

        contact["interactions"].append({
            "timestamp": now,
            "user_said": (msg.get("user_said") or "")[:500],
            "iyye_said": (msg.get("iyye_said") or "")[:500],
            "source": msg.get("source", ""),
        })

        # Trim oldest interactions.
        if len(contact["interactions"]) > self._MAX_INTERACTIONS:
            contact["interactions"] = contact["interactions"][-self._MAX_INTERACTIONS:]

        self._dirty = True

    # ------------------------------------------------------------------
    # Public query interface (called synchronously by chat streams)
    # ------------------------------------------------------------------

    @staticmethod
    def make_contact_id(
        sender_name: Optional[str],
        source: str,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
    ) -> str:
        """Derive a stable contact_id from available identifiers.

        For Telegram, ``user_id`` identifies the individual sender (unique
        across all chats) while ``chat_id`` identifies the conversation
        (same for every member of a group).  When both are available the
        contact id is ``telegram_{user_id}`` so that group-chat members
        are not conflated into one profile.
        """
        src = source.lower()
        if "telegram" in src:
            if user_id:
                return f"telegram_{user_id}"
            if chat_id:
                return f"telegram_{chat_id}"
        if sender_name and sender_name != "unknown":
            clean = re.sub(r"[^a-z0-9_]", "_", sender_name.lower())
            return f"{src}_{clean}"
        return f"{src}_anonymous"

    # ------------------------------------------------------------------
    # Persona grouping
    # ------------------------------------------------------------------

    def _persona_group(self, contact_id: str) -> List[Dict[str, Any]]:
        """Return all contacts sharing the same persona, or [contact] alone."""
        contact = self._contacts.get(contact_id)
        if contact is None:
            return []
        persona = contact.get("persona")
        if not persona:
            return [contact]
        return [c for c in self._contacts.values()
                if c.get("persona") == persona]

    def set_persona(self, contact_id: str, persona: str) -> bool:
        """Assign a contact to a named persona group."""
        contact = self._contacts.get(contact_id)
        if contact is None:
            return False
        contact["persona"] = persona
        self._dirty = True
        return True

    def link_contacts(self, contact_ids: List[str],
                      persona: Optional[str] = None) -> bool:
        """Link multiple contacts into one persona.

        If *persona* is None, uses the display_name of the first known
        contact.  Propagates trust: if any linked contact is trusted, all
        become trusted.
        """
        known = [self._contacts[cid]
                 for cid in contact_ids if cid in self._contacts]
        if len(known) < 2:
            return False

        if persona is None:
            for c in known:
                n = c.get("display_name", "unknown")
                if n != "unknown":
                    persona = n
                    break
            if persona is None:
                persona = known[0]["contact_id"]

        any_trusted = any(c.get("trusted", False) for c in known)
        for c in known:
            c["persona"] = persona
            if any_trusted:
                c["trusted"] = True

        self._dirty = True
        ids = [c["contact_id"] for c in known]
        self.add_to_log(f"Linked {len(known)} contacts under persona '{persona}'")

        stm = getattr(self.brain, 'stm', None)
        if stm:
            stm.add_fact(
                f"Contacts {', '.join(ids)} are the same person"
                f" (persona: {persona}).",
                confidence=1.0,
                provenance="theory_of_mind",
                time_frame="permanent",
            )
        return True

    def link_by_display_name(self, display_name: str) -> bool:
        """Link all contacts whose display_name matches *display_name*."""
        ids = [cid for cid, c in self._contacts.items()
               if c.get("display_name", "").lower() == display_name.lower()]
        if len(ids) < 2:
            return False
        return self.link_contacts(ids, persona=display_name)

    # ------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------

    def is_contact_trusted(self, contact_id: str) -> bool:
        """Return whether a contact (or any contact in its persona) is trusted.

        HLD: telegram users are assumed not trusted; the ONLY way one becomes
        trusted is an explicit owner command from the local web chat.  The
        flag is set via set_contact_trusted(), whose chat-side caller
        (UserChatStream._execute_trust_action) enforces the local-only rule.
        """
        for c in self._persona_group(contact_id):
            if c.get("trusted", False):
                return True
        return False

    def find_contacts(self, query: str):
        """Return ``[(contact_id, display_name), ...]`` for contacts matching
        *query* by explicit contact id (e.g. ``telegram_123``), exact id, or a
        display-name substring (case-insensitive).

        Public lookup so callers (e.g. chat trust actions) don't reach into the
        private ``_contacts`` map directly."""
        q = (query or "").strip().lower()
        explicit = set(re.findall(r"telegram_\d+", q))
        out = []
        for cid, contact in self._contacts.items():
            display = contact.get("display_name") or ""
            if cid in explicit or cid.lower() == q or (display and q in display.lower()):
                out.append((cid, display or cid))
        return out

    def known_contacts(self):
        """Return ``[(contact_id, display_name), ...]`` for every known contact.

        Public enumeration (companion to :meth:`find_contacts`) so callers —
        e.g. the planner detecting contacts mentioned in plan-step text —
        don't reach into the private ``_contacts`` map."""
        return [
            (cid, c.get("display_name") or cid)
            for cid, c in self._contacts.items()
        ]

    def detect_contacts_in_text(self, text: str):
        """Return ``[(contact_id, display_name), ...]`` for every known contact
        whose display name appears as a whole word in *text*.

        Used by unified recall to surface a third party's interaction history
        when they are *named in a question* ("when did Jacob contact you?").
        Names under 3 chars and the "unknown" placeholder are skipped to avoid
        false positives; deduped by contact id."""
        import re as _re
        low = (text or "").lower()
        if not low:
            return []
        out = []
        seen = set()
        for cid, contact in self._contacts.items():
            name = (contact.get("display_name") or "").strip().lower()
            if len(name) < 3 or name == "unknown" or cid in seen:
                continue
            if _re.search(rf"\b{_re.escape(name)}\b", low):
                out.append((cid, contact.get("display_name") or cid))
                seen.add(cid)
        return out

    def ensure_contact(
        self,
        contact_id: str,
        display_name: str = "unknown",
        source: str = "",
        chat_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return the contact record for *contact_id*, creating it if needed.

        This lets callers (e.g. trust actions) guarantee the contact exists
        even if the ToM stream hasn't processed its mailbox yet.
        """
        contact = self._contacts.get(contact_id)
        if contact is not None:
            return contact
        now = datetime.now(timezone.utc).isoformat()
        contact = {
            "contact_id": contact_id,
            "display_name": display_name,
            "source": source,
            "chat_id": chat_id,
            "first_seen": now,
            "last_seen": now,
            "interaction_count": 0,
            "interactions": [],
            "psychological_profile": None,
        }
        self._contacts[contact_id] = contact
        self._dirty = True
        self.add_to_log(f"New contact: {display_name} ({contact_id})")
        stm = getattr(self.brain, 'stm', None)
        if stm:
            stm.add_fact(
                f"New contact: {display_name} via {source}.",
                confidence=1.0,
                provenance="theory_of_mind",
                time_frame="permanent",
            )
        return contact

    def set_contact_trusted(self, contact_id: str, trusted: bool) -> bool:
        """Grant or revoke trust for a contact.  Returns False if unknown."""
        contact = self._contacts.get(contact_id)
        if contact is None:
            return False
        contact["trusted"] = trusted
        self._dirty = True
        name = contact.get('display_name', contact_id)
        self.add_to_log(
            f"{'Granted' if trusted else 'Revoked'} trust for {name}"
        )
        # Persist to STM so trust decisions reach LTM via sleep replay.
        stm = getattr(self.brain, 'stm', None)
        if stm:
            label = "now a trusted" if trusted else "no longer a trusted"
            stm.add_fact(
                f"{name} is {label} user.",
                confidence=1.0,
                provenance="theory_of_mind",
                time_frame="permanent",
            )
        return True

    def get_contact_context(self, contact_id: str) -> str:
        """Return a formatted context block about a contact for LLM prompts.

        When the contact belongs to a persona group, interactions and profile
        are aggregated across all accounts in the group.

        Returns empty string for unknown contacts so the caller can substitute
        a default like "(new contact)".
        """
        contact = self._contacts.get(contact_id)
        if contact is None:
            return ""

        group = self._persona_group(contact_id)

        lines: List[str] = []
        name = contact.get("display_name", "unknown")
        total_count = sum(c.get("interaction_count", 0) for c in group)
        first = min((c.get("first_seen", "\xff") for c in group))[:10]
        trusted = "yes" if any(c.get("trusted") for c in group) else "no"

        if len(group) > 1:
            lines.append(
                f"{name} — {total_count} interactions across"
                f" {len(group)} accounts, first seen {first}"
            )
        else:
            lines.append(f"{name} — {total_count} interactions, first seen {first}")
        lines.append(f"Trusted: {trusted}")

        # Profile — use whichever group member has one.
        profile = None
        for c in group:
            p = c.get("psychological_profile")
            if p and p.get("summary"):
                profile = p
                break

        if profile:
            n_at_update = profile.get("interaction_count_at_update", "?")
            updated = (profile.get("last_updated") or "?")[:10]
            lines.append(
                f"Inferred profile (based on {n_at_update} interactions,"
                f" last updated {updated}):"
            )
            lines.append(f"  {profile['summary']}")
            traits = profile.get("traits")
            if traits:
                lines.append(f"  Observed traits: {', '.join(traits)}")
            prefs = profile.get("preferences")
            if prefs:
                lines.append(f"  Observed preferences: {', '.join(prefs)}")

        # Recent interaction topics — merge across all accounts.
        all_interactions: List[Dict[str, Any]] = []
        for c in group:
            all_interactions.extend(c.get("interactions", []))
        all_interactions.sort(key=lambda i: i.get("timestamp", ""))
        recent = all_interactions[-5:]
        if recent:
            lines.append("Recent topics:")
            for i in recent:
                snippet = (i.get("user_said") or "")[:100]
                if snippet:
                    lines.append(f"  - {snippet}")

        return "\n".join(lines)

    def get_recent_interactions(
        self, contact_id: str, limit: int = 10,
    ) -> List[Dict[str, str]]:
        """Return the last *limit* interactions for a contact.

        When the contact belongs to a persona group, interactions from all
        accounts are merged and sorted chronologically.

        Each item has keys ``user_said``, ``iyye_said``, ``source``.
        Returns an empty list for unknown contacts.
        """
        group = self._persona_group(contact_id)
        if not group:
            return []
        all_interactions: List[Dict[str, Any]] = []
        for c in group:
            all_interactions.extend(c.get("interactions", []))
        all_interactions.sort(key=lambda i: i.get("timestamp", ""))
        return all_interactions[-limit:]

    # ------------------------------------------------------------------
    # LLM-based psychological profiling
    # ------------------------------------------------------------------

    def _find_stale_profile(self):
        """Return ``(contact, interactions, total_count, group)`` for one
        contact/persona that has accumulated enough new interactions to warrant
        a profile refresh, or None.  One per cycle to avoid LLM contention."""
        seen_personas: set = set()
        for contact in list(self._contacts.values()):
            persona = contact.get("persona")
            if persona:
                if persona in seen_personas:
                    continue
                seen_personas.add(persona)

            group = self._persona_group(contact["contact_id"])

            # Merge interactions across the persona group.
            all_interactions: List[Dict[str, Any]] = []
            for c in group:
                all_interactions.extend(c.get("interactions", []))
            if not all_interactions:
                continue

            total_count = sum(c.get("interaction_count", 0) for c in group)
            profile = contact.get("psychological_profile") or {}
            last_count = profile.get("interaction_count_at_update", 0)
            if total_count - last_count < self._PROFILE_MIN_NEW:
                continue

            all_interactions.sort(key=lambda i: i.get("timestamp", ""))
            return contact, all_interactions, total_count, group
        return None

    def _submit_stale_profile(self) -> None:
        """Find one stale contact/persona and submit its profile inference to
        the async scheduler.  The result is applied in _on_profile_result."""
        found = self._find_stale_profile()
        if found is None:
            return
        contact, interactions, total_count, group = found

        recent = interactions[-15:]
        interactions_text = "\n".join(
            f"  [{i.get('timestamp', '?')[:16]}] User: {i.get('user_said', '')[:200]}\n"
            f"  Iyye: {i.get('iyye_said', '')[:200]}"
            for i in recent
        )
        old_profile = contact.get("psychological_profile")
        existing = ""
        if old_profile and old_profile.get("summary"):
            existing = old_profile["summary"]

        submitted = self._llm_submit(
            role="fast", kind="profile",
            call=LLMCall.from_file(
                "theory_of_mind_profile",
                contact_name=contact.get("display_name", "unknown"),
                interaction_count=str(total_count),
                interactions=interactions_text,
                existing_profile=existing or "(first assessment)",
            ),
            client_kwargs={"no_think": True, "max_tokens": 512},
        )
        if submitted:
            # Stash by contact id (not dict ref) so the result is applied
            # against the live contact map, which may change while the job runs.
            self._pending_profile = {
                "contact_name": contact.get("display_name", "unknown"),
                "group_ids": [c["contact_id"] for c in group],
                "total_count": total_count,
            }

    def _on_profile_result(self, result) -> None:
        """Apply a finished profile inference on the main thread."""
        ctx = self._pending_profile or {}
        self._pending_profile = None
        if result.discarded or not result.ok:
            if not result.discarded:
                log.warning("TheoryOfMind: profile update failed: %s", result.error)
            return
        profile = self._parse_profile(result.text)
        if not profile:
            return
        profile["last_updated"] = datetime.now(timezone.utc).isoformat()
        profile["interaction_count_at_update"] = ctx.get("total_count", 0)

        # Store profile on every (still-present) contact in the group so any
        # lookup returns it without needing a join.
        applied: List[Dict[str, Any]] = []
        for cid in ctx.get("group_ids", []):
            c = self._contacts.get(cid)
            if c is not None:
                c["psychological_profile"] = profile
                applied.append(c)
        if not applied:
            return
        self._dirty = True
        pname = ctx.get("contact_name", "?")
        self.add_to_log(f"Updated profile for {pname}")
        # Persist profile summary to STM so it reaches LTM.
        stm = getattr(self.brain, 'stm', None)
        if stm:
            stm.add_fact(
                f"Psychological profile for {pname}: {profile['summary']}",
                confidence=0.6,
                provenance="theory_of_mind",
                time_frame="permanent",
            )

    @staticmethod
    def _parse_profile(response: str) -> Optional[Dict[str, Any]]:
        """Extract structured profile from LLM response."""
        text = re.sub(r"```[a-z]*\n?", "", response).strip()
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return {
                    "summary": str(data.get("summary", ""))[:300],
                    "traits": list(data.get("traits", []))[:10],
                    "preferences": list(data.get("preferences", []))[:10],
                }
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: use the whole response as summary.
        stripped = response.strip()
        if len(stripped) > 10:
            return {"summary": stripped[:300], "traits": [], "preferences": []}
        return None

    # ------------------------------------------------------------------
    # Stream protocol
    # ------------------------------------------------------------------

    def can_stop_safely(self) -> bool:
        """Block wind-down while we have unsaved contact data or pending mail."""
        if self._dirty:
            return False
        # Thread-safe peek instead of reaching into the shared mailbox dict.
        if self.brain.peek_messages("theory_of_mind"):
            return False
        return super().can_stop_safely()

    def flush(self) -> None:
        """Force-drain mailbox and save contacts.  Called on wind-down / shutdown."""
        for msg in self.brain.drain_messages("theory_of_mind"):
            self._handle_message(msg)
        if self._dirty:
            self._save_contacts()

    def on_pause(self) -> None:
        """Hook called during the brain's settle phase before sleep.

        Flushes pending mailbox messages and dirty contacts so the post-sleep
        snapshot is consistent.  Replaces the explicit ToM-flush special-case
        that used to live in IyyeBrain._enter_asleep.
        """
        self.flush()

    def restore_state(self, state: Dict[str, Any]) -> None:
        super().restore_state(state)
