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
# streams/user_chat_stream.py
#!/usr/bin/env python3
"""
User Chat Stream - Processes incoming user messages via LLM and sends responses.

This is the primary conscious stream created by StreamFactory when sensor
queues (web_chat, telegram, microphone) contain new messages.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from iyye_base import PROJECT_ROOT, ProcessingStream

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")


class UserChatStream(ProcessingStream):
    """
    Processes incoming user messages with LLM and routes responses to actuators.
    Can become conscious (priority 5, higher than self-reflection's 3).
    """

    # Instances are created on-demand by StreamFactory, not by the stream loader.
    _factory_created: bool = True

    def __init__(
        self,
        name: str,
        messages: List[Any],
        brain: "IyyeBrain",
        sensor_name: str = "",
    ):
        super().__init__(name=name)
        self.brain = brain
        self.priority = 5
        self._can_be_conscious = True
        self._sensor_name: str = sensor_name  # e.g. "TelegramSensor", "web_chat"
        self._last_chat_id: Optional[int] = None  # for telegram reply routing
        self._last_user_id: Optional[int] = None   # telegram user id (distinct from chat in groups)
        self._last_sender_name: Optional[str] = None  # display name of the last sender
        self._pending_messages: List[Any] = list(messages)
        # HLD: "local web chat … user is assumed trusted"; "telegram …
        # Users assumed not trusted, unless instructed by trusted user."
        self._trusted: bool = self._is_trusted_source(sensor_name)

        # Populate input_history so _select_conscious_for_interrupt finds this stream
        for msg in messages:
            text, _, _, _ = self._extract_text_and_chat_id(msg)
            if text:
                src = sensor_name or (msg.get('source', 'sensor') if isinstance(msg, dict) else 'sensor')
                self.add_input(text, source=src)

    # ------------------------------------------------------------------
    # Trust
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trusted_source(sensor_name: str) -> bool:
        """HLD: web_chat is local → trusted.  Everything else → untrusted."""
        return 'web_chat' in sensor_name.lower() or 'webchat' in sensor_name.lower()

    def _check_contact_trusted(self) -> bool:
        """Re-evaluate trust for the current contact via Theory of Mind.

        A contact starts untrusted but can be promoted to trusted when a
        trusted user instructs Iyye (e.g. "trust Jacob").  The ToM stream
        stores this flag per contact.
        """
        if self._is_trusted_source(self._sensor_name):
            return True
        tom = getattr(self.brain, '_tom_stream', None)
        if tom is None:
            return False
        contact_id = tom.make_contact_id(
            self._last_sender_name or 'unknown',
            self._sensor_name,
            self._last_chat_id,
            self._last_user_id,
        )
        return tom.is_contact_trusted(contact_id)

    _TRUST_RE = re.compile(
        r'\b(trust|untrust)\s+(?:@?(\w+))'           # "trust Alex", "untrust Jacob"
        r'|\b(?:make|promote|add)\s+(?:@?(\w+))\s+.*?\b(trusted)\b'  # "make Alex trusted", "promote Alex to trusted"
        r'|\b(trusted)\b.*?\b(?:for|to)\s+(?:@?(\w+))',              # "grant trusted to Alex"
        re.IGNORECASE,
    )

    def _detect_trust_change(self, user_text: str) -> Optional[str]:
        """If a trusted user says 'trust <name>' or 'untrust <name>', update trust.

        Also matches natural phrasings: "make Alex trusted", "promote Alex to
        trusted status", "grant trusted to Alex".

        Searches Theory of Mind contacts by contact_id first, then by display
        name.  When multiple contacts share a display name, **all** of them
        are updated.  Only callable from an already-trusted stream.
        Returns a feedback string for the user, or None if no command matched.
        """
        m = self._TRUST_RE.search(user_text)
        if not m:
            return None

        # The regex has three alternatives with different group layouts:
        #   Alt 1 (groups 1,2): "trust Alex" / "untrust Alex"
        #   Alt 2 (groups 3,4): "make Alex trusted" / "promote Alex to trusted"
        #   Alt 3 (groups 5,6): "grant trusted to Alex"
        if m.group(1):
            verb = m.group(1).lower()
            target_name = m.group(2).lower()
            granting = verb != "untrust"
        elif m.group(3):
            target_name = m.group(3).lower()
            granting = True  # "make X trusted" is always granting
        else:
            target_name = m.group(6).lower()
            granting = True  # "grant trusted to X" is always granting

        tom = getattr(self.brain, '_tom_stream', None)
        if tom is None:
            return None

        # Also scan the full message for contact_id patterns (telegram_123...)
        # so "grant trust to telegram_6473529683" works even though the regex
        # only captured a partial target_name.
        _CID_RE = re.compile(r'telegram_\d+')
        explicit_cids = set(_CID_RE.findall(user_text))

        updated = []
        for cid, contact in tom._contacts.items():
            display = (contact.get("display_name") or "").lower()
            matched = (
                cid in explicit_cids
                or (display and target_name in display)
            )
            if matched and tom.set_contact_trusted(cid, granting):
                actual_name = contact.get('display_name', cid)
                updated.append(f"{actual_name} ({cid})")

        if updated:
            action_word = "Granted" if granting else "Revoked"
            summary = ", ".join(updated)
            self.add_to_log(f"Trusted user {action_word.lower()} trust for {summary}")
            return f"{action_word} trust for {summary}."

        # Contact may not exist yet if they just sent their first message
        # in this same tick.  Don't fail — report clearly so the user can retry.
        self.add_to_log(f"Trust change failed: no contact matching '{target_name}'")
        return f"No known contact matching '{target_name}'. They need to send a message first so I can identify them."

    def _execute_trust_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Handle ACTION: {"type": "trust"/"untrust", "contact": "<name>"}.

        Allows the LLM to programmatically grant/revoke trust when it decides
        a user has authenticated (e.g. via PIN verification).

        Matches contacts by contact_id first, then by display name.  When
        multiple contacts share a display name all of them are updated.
        If no contact is found, creates one for the current sender so that
        self-trust (PIN verification) succeeds even on the first interaction.
        """
        target_name = (action.get('contact') or '').strip().lower()
        if not target_name:
            self.add_to_log("Trust action missing 'contact' field — ignored")
            return None
        granting = action['type'] == 'trust'
        tom = getattr(self.brain, '_tom_stream', None)
        if tom is None:
            self.add_to_log("Trust action failed: Theory of Mind stream not available")
            return None

        # Scan for explicit contact_id in the action value (e.g. "telegram_123").
        _CID_RE = re.compile(r'telegram_\d+')
        explicit_cids = set(_CID_RE.findall(target_name))

        # Match by contact_id or display name — update ALL matches.
        updated = []
        for cid, contact in tom._contacts.items():
            display = (contact.get("display_name") or "").lower()
            matched = (
                cid in explicit_cids
                or cid == target_name
                or (display and target_name in display)
            )
            if matched and tom.set_contact_trusted(cid, granting):
                actual_name = contact.get('display_name', cid)
                updated.append(f"{actual_name} ({cid})")

        if updated:
            verb = "Granted" if granting else "Revoked"
            summary = ", ".join(updated)
            self.add_to_log(f"ACTION {verb.lower()} trust for {summary}")
            return f"{verb} trust for {summary}."

        # Contact not found — likely a timing issue: the interaction was
        # posted to the ToM mailbox but hasn't been processed yet.  Create
        # the contact for the current sender so self-trust doesn't fail.
        contact_id = tom.make_contact_id(
            self._last_sender_name, self._sensor_name,
            self._last_chat_id, self._last_user_id,
        )
        display = self._last_sender_name or target_name
        tom.ensure_contact(
            contact_id,
            display_name=display,
            source=self._sensor_name,
            chat_id=self._last_chat_id,
        )
        if tom.set_contact_trusted(contact_id, granting):
            verb = "Granted" if granting else "Revoked"
            self.add_to_log(f"ACTION {verb.lower()} trust for {display} (created contact {contact_id})")
            return f"{verb} trust for {display} ({contact_id})."

        self.add_to_log(f"Trust action failed for '{target_name}' (contact_id={contact_id})")
        return f"Failed to {'grant' if granting else 'revoke'} trust for '{target_name}'."

    # ------------------------------------------------------------------
    # LLM management actions
    # ------------------------------------------------------------------

    def _execute_llm_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Handle ACTION: {"type": "llm", "command": "start|stop|use_for_chat", "model": "..."}.

        Dispatches to LlmManagementStream via inter-stream messages and/or
        sets a role override on the LLM router.
        """
        command = (action.get('command') or '').strip().lower()
        model_name = (action.get('model') or '').strip()
        if not command or not model_name:
            return None

        if command == 'start':
            # Find registry entry to get the script name
            router = getattr(self.brain, 'llm_router', None)
            if router is None:
                return "[LLM] Router not available."
            entry = next((m for m in router._registry if m['name'] == model_name), None)
            if entry is None:
                return f"[LLM] Unknown model '{model_name}'."
            self.brain.post_message("llm_management", {
                "action": "start",
                "script": entry["script"],
            })
            self.add_to_log(f"Requested LLM start: {model_name}")
            return f"[LLM] Starting {model_name} — this may take a minute."

        elif command == 'stop':
            self.brain.post_message("llm_management", {
                "action": "stop",
                "name": model_name,
            })
            self.add_to_log(f"Requested LLM stop: {model_name}")
            return f"[LLM] Stopping {model_name}."

        elif command == 'use_for_chat':
            router = getattr(self.brain, 'llm_router', None)
            if router is None:
                return "[LLM] Router not available."
            if not router.set_role_override("chat", model_name):
                return f"[LLM] Unknown model '{model_name}'."
            self.add_to_log(f"Switched chat model to {model_name}")
            return f"[LLM] Chat responses will now use {model_name}."

        else:
            return f"[LLM] Unknown command '{command}'."

    def _execute_persona_action(self, action: Dict[str, Any]) -> Optional[str]:
        """Handle ACTION: {"type": "persona", "name": "..."}."""
        tom = getattr(self.brain, '_tom_stream', None)
        if tom is None:
            return "[Persona] Theory of Mind stream not available."
        name = (action.get('name') or '').strip()
        if not name:
            return "[Persona] No display name provided."
        if tom.link_by_display_name(name):
            return f"[Persona] Linked all contacts named '{name}' into one persona."
        return f"[Persona] Found fewer than 2 contacts named '{name}' — nothing to link."

    # ------------------------------------------------------------------
    # Message extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_and_chat_id(
        message: Any,
    ) -> Tuple[str, Optional[int], Optional[str], Optional[int]]:
        """
        Extract text, chat_id, sender display name, and user_id from any
        message format.

        Returns: (text, chat_id, sender_name, user_id)

        sender_name is the human-readable name of the person who sent the
        message: first_name if available, otherwise @username, otherwise None.
        user_id is the Telegram user id (distinct from chat_id in groups).

        Handles:
        - Plain string (web_chat)
        - Direct Telegram message dict: {"text": "...", "chat_id": 123, ...}
          (produced after call_tool() unwraps content[0]["text"] and
          _expand_mcp_batches() splits the batch)
        - Generic dict with 'data' key
        - Legacy structuredContent wrapper (kept for safety)
        """
        if isinstance(message, str):
            return message.strip(), None, None, None

        if not isinstance(message, dict):
            return str(message).strip(), None, None, None

        # Direct Telegram message dict — top-level "text" and "chat_id"
        if 'chat_id' in message:
            text = str(message.get('text') or '').strip()
            chat_id = message.get('chat_id')
            first_name = message.get('first_name') or ''
            username = message.get('username') or ''
            sender_name = first_name or (f'@{username}' if username else None)
            user_id = message.get('user_id')
            return text, int(chat_id) if chat_id else None, sender_name, \
                int(user_id) if user_id else None

        # Legacy structuredContent wrapper
        sc = message.get('structuredContent', {})
        if isinstance(sc, dict):
            msgs = sc.get('messages', [])
            if msgs and isinstance(msgs, list):
                first = msgs[0]
                text = first.get('text', '').strip()
                chat_id = first.get('chat_id')
                first_name = first.get('first_name') or ''
                username = first.get('username') or ''
                sender_name = first_name or (f'@{username}' if username else None)
                user_id = first.get('user_id')
                return text, chat_id, sender_name, \
                    int(user_id) if user_id else None
            if 'count' in sc:
                return '', None, None, None

        # Generic dict with 'data' (web_chat format)
        data = message.get('data', '')
        return str(data).strip() if data else '', None, None, None

    # ------------------------------------------------------------------
    # LLM lazy init
    # ------------------------------------------------------------------

    def _get_llm(self):
        # Re-acquire each call: conscious status can change between ticks,
        # and the router directs conscious streams to the most powerful model.
        # Web chat is local/admin-only — use the fast model to keep the heavy
        # model free for Telegram and subconscious streams.
        try:
            router = getattr(getattr(self, 'brain', None), 'llm_router', None)
            if router is not None:
                is_web = self._is_trusted_source(self._sensor_name)
                role = "fast" if is_web else "chat"
                return router.get_client(
                    role=role, conscious=self.is_conscious, no_think=True,
                )
            from llm_client import LLMClient
            return LLMClient(no_think=True)
        except Exception as exc:
            log.warning("UserChatStream: LLM unavailable: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process the next pending message, call LLM, send response."""
        # Drain any empty/unparseable messages silently — peek before popping
        # so a message is never lost if a stop is requested mid-execution.
        user_text = ''
        chat_id = None
        sender_name = None
        user_id = None
        while self._pending_messages and not user_text:
            message = self._pending_messages[0]  # peek
            user_text, chat_id, sender_name, user_id = self._extract_text_and_chat_id(message)
            if not user_text:
                self._pending_messages.pop(0)  # discard unparseable, safe to lose

        if not user_text:
            # Queue is empty and all previous inputs have a matching output —
            # retire this stream so it doesn't accumulate as dead weight.
            if len(self.input_history) > 0 and len(self.output_history) >= len(self.input_history):
                brain = getattr(self, 'brain', None)
                if brain is not None:
                    try:
                        brain.streams.remove(self)
                        log.debug("UserChatStream '%s' retired (all messages processed)", self.name)
                    except ValueError:
                        pass
            return None  # nothing real to process this tick

        if chat_id:
            self._last_chat_id = chat_id
        if user_id:
            self._last_user_id = user_id
        if sender_name:
            self._last_sender_name = sender_name

        source = self._sensor_name or self.name
        sender_label = self._last_sender_name or source
        self.add_to_log(f"USER ({sender_label}): {user_text}")

        # Re-evaluate trust early so the LLM prompt can reflect it.
        # Constructor sets _trusted based on sensor name alone (web_chat=True),
        # but ToM may have granted trust to this telegram contact since then.
        self._trusted = self._check_contact_trusted()

        # Cooperative-multitasking checkpoint BEFORE the blocking LLM call.
        # If a stop has been requested (e.g. winding-down) this raises
        # StopIteration here rather than committing to a 60s LLM request.
        # The message is still at index 0 of _pending_messages — not yet consumed.
        self.checkpoint()

        # Commit the pop only after passing the checkpoint: if StopIteration was
        # raised above, the message stays in the queue for the next wake cycle.
        self._pending_messages.pop(0)

        adenosine = context.get('adenosine', 1.0)
        active_streams = len(context.get('streams', []))
        conversation_history = self._build_history()
        # Prefer the rich self-reflection snapshot over bare counts.
        # Falls back to brain attribute (set by SelfReflectionStream each tick).
        sr_snapshot = context.get('self_reflection_state') or getattr(
            self.brain, '_self_reflection_snapshot', None
        )

        system_description = self._read_system_description()
        stm_facts = self._build_stm_context()
        ltm_facts = self._build_ltm_context(user_text)
        contact_context = self._get_contact_context()

        response = self._generate_response(
            user_text=user_text,
            source=source,
            sender_name=self._last_sender_name,
            conversation_history=conversation_history,
            stm_facts=stm_facts,
            ltm_facts=ltm_facts,
            adenosine=adenosine,
            active_streams=active_streams,
            sr_snapshot=sr_snapshot,
            system_description=system_description,
            contact_context=contact_context,
            context=context,
        )

        # Parse and strip any ACTION: line before sending to the user.
        response, action = self._extract_action(response)

        self.add_to_log(f"IYYE: {response}")
        self.add_output(response, target=source)
        if response:
            self._send_to_actuator(response, context)
        self._report_interaction(user_text, response)

        # When a trusted user mentions trusting/untrusting someone, update via ToM.
        if self._trusted:
            trust_feedback = self._detect_trust_change(user_text)
            if trust_feedback:
                self._send_to_actuator(trust_feedback, context)

        # Read-only research actions (wikipedia, url) are safe from any
        # source.  Trust/untrust are self-authenticating (the LLM verifies
        # the PIN before emitting the ACTION), so they must be allowed from
        # untrusted sources — otherwise users can never become trusted.
        # Only privileged actions (python, llm) require prior trust.
        _SAFE_ACTION_TYPES = {'wikipedia', 'url', 'trust', 'untrust'}
        if action and not self._trusted:
            if action.get('type') not in _SAFE_ACTION_TYPES:
                self.add_to_log(f"Suppressed ACTION from untrusted source {source}")
                action = None

        if action:
            action_type = action.get('type')
            if action_type == 'python':
                # Execute inline — short subprocess, no need for a separate stream.
                py_result = self._execute_python(action.get('code', ''), context)
                self._send_to_actuator(py_result, context)
                self.add_to_log(f"Python execution result: {py_result[:200]}")
            elif action_type in ('trust', 'untrust'):
                feedback = self._execute_trust_action(action)
                if feedback:
                    self._send_to_actuator(feedback, context)
                # Re-evaluate own trust: the action may have granted trust
                # to the current sender (self-trust via PIN verification).
                self._trusted = self._check_contact_trusted()
            elif action_type == 'llm':
                feedback = self._execute_llm_action(action)
                if feedback:
                    self._send_to_actuator(feedback, context)
            elif action_type == 'persona':
                feedback = self._execute_persona_action(action)
                if feedback:
                    self._send_to_actuator(feedback, context)
            else:
                action['chat_id'] = self._last_chat_id
                action['sensor_name'] = self._sensor_name
                action['sender_name'] = self._last_sender_name
                if not hasattr(self.brain, '_pending_research_tasks'):
                    self.brain._pending_research_tasks = []
                self.brain._pending_research_tasks.append(action)
                self.add_to_log(f"Queued research task: {action_type} — {action.get('query') or action.get('url', '')[:60]}")

        self.checkpoint()

        return {'type': 'chat_reply', 'text': response, 'source': source}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_action(response: str):
        """Strip the ACTION: line from LLM output and return (clean_text, action_dict).

        The ACTION line may appear at the end of the response. Returns
        (response, None) if no valid ACTION line is present.
        """
        lines = response.splitlines()
        action = None
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("ACTION:"):
                payload = stripped[len("ACTION:"):].strip()
                try:
                    data = json.loads(payload)
                    if isinstance(data, dict) and data.get('type') in ('wikipedia', 'url', 'python', 'trust', 'untrust', 'llm', 'persona'):
                        action = data
                except (json.JSONDecodeError, ValueError):
                    pass  # malformed — ignore, keep line as text
                # Either way, don't include the ACTION line in visible response
                continue
            clean_lines.append(line)
        return "\n".join(clean_lines).strip(), action

    def _read_system_description(self) -> str:
        """Return current system description, preferring the in-memory copy
        updated by SelfReflectionStream every gather tick over the on-disk
        snapshot which is only written once per iyye_day."""
        cached = getattr(self.brain, '_system_description_md', None)
        if cached:
            return cached
        path = PROJECT_ROOT / "system_description.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return "(unavailable)"

    def _build_ltm_context(self, query: str, limit: int = 8) -> str:
        """Semantic search in LTM for facts relevant to the current message."""
        memory = getattr(self.brain, 'memory', None)
        if memory is None:
            return "(none)"
        try:
            facts = memory.search_semantic(query, limit=limit)
        except Exception as exc:
            log.warning("UserChatStream: LTM search failed: %s", exc)
            return "(none)"
        if not facts:
            return "(none)"
        lines = []
        for f in facts:
            tf = f.get('time_frame', '?')
            conf = float(f.get('confidence', 0.5))
            prov = f.get('provenance') or f.get('source') or '?'
            lines.append(f"[{tf}/{conf:.2f} from {prov}] {f['text']}")
        return "\n".join(lines)

    def _build_stm_context(self) -> str:
        """Format non-ephemeral STM facts for injection into the LLM prompt."""
        stm = getattr(self.brain, 'stm', None)
        if stm is None:
            return "(none)"
        facts = stm.get_recent(limit=30)
        # Skip ephemeral metric snapshots — they're stale by the time the LLM sees them
        relevant = [f for f in facts if f.get('time_frame') != 'ephemeral']
        if not relevant:
            return "(none)"
        lines = []
        for f in relevant:
            tf = f.get('time_frame', '?')
            conf = float(f.get('confidence', 0.7))
            prov = f.get('provenance', '?')
            lines.append(f"[{tf}/{conf:.2f} from {prov}] {f['text']}")
        return "\n".join(lines)

    def _get_contact_context(self) -> str:
        """Query Theory of Mind stream for context about the current sender."""
        tom = getattr(self.brain, '_tom_stream', None)
        if tom is None:
            return "(unavailable)"
        contact_id = tom.make_contact_id(
            self._last_sender_name, self._sensor_name,
            self._last_chat_id, self._last_user_id,
        )
        ctx = tom.get_contact_context(contact_id)
        return ctx or "(new contact)"

    def _report_interaction(self, user_text: str, response: str) -> None:
        """Post this interaction to Theory of Mind via mailbox."""
        tom = getattr(self.brain, '_tom_stream', None)
        if tom is None:
            return
        contact_id = tom.make_contact_id(
            self._last_sender_name, self._sensor_name,
            self._last_chat_id, self._last_user_id,
        )
        self.brain.post_message("theory_of_mind", {
            "action": "interaction",
            "contact_id": contact_id,
            "display_name": self._last_sender_name,
            "source": self._sensor_name,
            "chat_id": self._last_chat_id,
            "user_id": self._last_user_id,
            "user_said": user_text,
            "iyye_said": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def _generate_response(
        self,
        user_text: str,
        source: str,
        sender_name: Optional[str],
        conversation_history: str,
        stm_facts: str = "(none)",
        ltm_facts: str = "(none)",
        adenosine: float = 1.0,
        active_streams: int = 0,
        sr_snapshot: Optional[Dict[str, Any]] = None,
        system_description: str = "(unavailable)",
        contact_context: str = "(unavailable)",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        llm = self._get_llm()
        if llm is None:
            return f"[Iyye offline] Echo: {user_text}"

        # Send a typing indicator so the user knows a response is being
        # generated — large models can take minutes.
        if context is not None:
            self._send_typing_indicator(context)

        system_state = self._build_system_state(adenosine, active_streams, sr_snapshot)
        sender_label = sender_name or "unknown"
        try:
            return llm.complete_from_file(
                "chat_response",
                user_message=user_text,
                source=source,
                sender_name=sender_label,
                conversation_history=conversation_history or "(none)",
                stm_facts=stm_facts,
                ltm_facts=ltm_facts,
                system_state=system_state,
                system_description=system_description,
                contact_context=contact_context,
            )
        except Exception as exc:
            exc_name = type(exc).__name__
            log.error("UserChatStream LLM error (%s): %s", exc_name, exc)
            # Surface a short user-visible message rather than a raw traceback.
            if "timeout" in exc_name.lower() or "timeout" in str(exc).lower():
                return (
                    "Sorry, the response is taking longer than expected. "
                    "The model may be overloaded — please try again in a moment."
                )
            return f"[Iyye error] Could not generate response ({exc_name})"

    @staticmethod
    def _build_system_state(
        adenosine: float,
        active_streams: int,
        sr: Optional[Dict[str, Any]],
    ) -> str:
        """Build a human-readable system state string for the LLM prompt.

        Uses the self-reflection snapshot when available for rich introspective
        detail; falls back to bare adenosine + stream count otherwise.
        """
        parts = [f"adenosine={adenosine:.2f}"]
        if sr:
            cpu = sr.get('cpu_percent')
            mem = sr.get('memory_percent')
            if cpu is not None:
                parts.append(f"cpu={cpu:.1f}%")
            if mem is not None:
                parts.append(f"memory={mem:.1f}%")
            pos = sr.get('position', {})
            facts = pos.get('facts_in_memory')
            if facts is not None:
                parts.append(f"facts_in_memory={facts}")
            conscious = pos.get('conscious_stream')
            if conscious:
                parts.append(f"conscious_stream={conscious}")
            iyye_day = sr.get('iyye_day') or pos.get('iyye_day')
            if iyye_day is not None:
                parts.append(f"iyye_day={iyye_day}")
        parts.append(f"active_streams={active_streams}")
        return ", ".join(parts)

    def _build_history(self) -> str:
        """Format last 10 input/output pairs as a readable conversation.

        When this stream's own history is short, supplements with recent
        cross-channel interactions from Theory of Mind so that a channel
        switch (e.g. web → Telegram) doesn't lose conversational context.
        """
        _HISTORY_CAP = 10
        lines: List[str] = []
        inputs = self.input_history[-_HISTORY_CAP:]
        outputs = self.output_history[-_HISTORY_CAP:]

        # Supplement with cross-channel history from ToM when own is thin.
        own_pairs = min(len(inputs), len(outputs))
        if own_pairs < _HISTORY_CAP:
            tom = getattr(self.brain, '_tom_stream', None)
            if tom is not None:
                contact_id = tom.make_contact_id(
                    self._last_sender_name, self._sensor_name,
                    self._last_chat_id, self._last_user_id,
                )
                need = _HISTORY_CAP - own_pairs
                prior = tom.get_recent_interactions(contact_id, limit=need)
                for i in prior:
                    src_tag = i.get("source", "")
                    user_said = (i.get("user_said") or "")[:200]
                    iyye_said = (i.get("iyye_said") or "")[:200]
                    if user_said:
                        prefix = f"User (via {src_tag})" if src_tag else "User"
                        lines.append(f"{prefix}: {user_said}")
                        lines.append(f"Iyye: {iyye_said}")

        for inp, out in zip(inputs, outputs):
            user = inp.get('data', '') if isinstance(inp, dict) else inp
            reply = out.get('data', '') if isinstance(out, dict) else out
            lines.append(f"User: {str(user)[:200]}")
            lines.append(f"Iyye: {str(reply)[:200]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Python execution (HLD: privileged users have access to python interpreter)
    # ------------------------------------------------------------------

    _PYTHON_TIMEOUT = 30     # seconds
    _PYTHON_MAX_LINES = 100  # max lines in a submitted script
    _PYTHON_MAX_OUTPUT = 4000  # max chars of captured output

    def _execute_python(self, code: str, context: Dict[str, Any]) -> str:
        """Run a Python script in a subprocess and return the captured output.

        HLD §9: "privileged users have access to all tools including python
        interpreter."  Only called when self._trusted is True (enforced by
        the action trust gate in execute()).
        """
        import subprocess
        import tempfile

        if not code or not code.strip():
            return "[Python] No code provided."

        line_count = len(code.strip().splitlines())
        if line_count > self._PYTHON_MAX_LINES:
            return f"[Python] Script too long ({line_count} lines, max {self._PYTHON_MAX_LINES})."

        # Find the venv python
        venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", dir=str(PROJECT_ROOT),
                delete=False, encoding="utf-8",
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                [venv_python, tmp_path],
                capture_output=True,
                text=True,
                timeout=self._PYTHON_TIMEOUT,
                cwd=str(PROJECT_ROOT),
            )

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            parts = []
            if stdout:
                parts.append(stdout[:self._PYTHON_MAX_OUTPUT])
            if result.returncode != 0:
                parts.append(f"[exit code {result.returncode}]")
                if stderr:
                    parts.append(stderr[:self._PYTHON_MAX_OUTPUT])

            output = "\n".join(parts) if parts else "(no output)"
            return f"[Python result]\n{output}"

        except subprocess.TimeoutExpired:
            return f"[Python] Script timed out after {self._PYTHON_TIMEOUT}s."
        except Exception as exc:
            return f"[Python] Execution error: {exc}"
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _send_typing_indicator(self, context: Dict[str, Any]) -> None:
        """Notify the user that a response is being generated.

        For Telegram: sends the "typing..." chat action so the user sees
        the animated dots.  For web chat: sends a brief status message.
        This is best-effort — failures are silently ignored.
        """
        try:
            if 'telegram' in self._sensor_name.lower() and self._last_chat_id:
                import os, requests as _req
                token = os.getenv("TELEGRAM_BOT_TOKEN", "")
                if token:
                    _req.post(
                        f"https://api.telegram.org/bot{token}/sendChatAction",
                        json={"chat_id": self._last_chat_id, "action": "typing"},
                        timeout=5,
                    )
            else:
                # Web chat uses the fast model — responses are quick enough
                # that a typing indicator would just add noise to the chat.
                pass
        except Exception:
            pass  # best-effort

    def _send_to_actuator(self, response: str, context: Dict[str, Any]) -> None:
        """Route response to the actuator matching this stream's sensor source."""
        actuators: Dict[str, Any] = context.get('actuators') or getattr(self.brain, 'actuators', {})
        if not actuators:
            return

        sensor_lower = self._sensor_name.lower()  # e.g. "telegramsensor", "web_chat"

        # Find actuator whose name shares a keyword with the sensor name
        candidates = [
            name for name in actuators
            if self._names_match(sensor_lower, name.lower())
        ]
        if not candidates:
            # Explicit fallback priority: web_chat > console (avoid console if possible)
            for preferred in ('webchat', 'web_chat', 'console'):
                for name in actuators:
                    if preferred in name.lower():
                        candidates = [name]
                        break
                if candidates:
                    break
        if not candidates:
            candidates = list(actuators.keys())[:1]

        for actuator_name in candidates:
            actuator = actuators[actuator_name]
            try:
                # Telegram actuator accepts JSON payload with chat_id
                if 'telegram' in actuator_name.lower() and self._last_chat_id:
                    payload = json.dumps({'text': response, 'chat_id': self._last_chat_id})
                else:
                    payload = response
                ok = actuator.actuate(payload)
                if ok is False:
                    # Explicit False means a known failure (e.g. missing chat_id,
                    # bot send error).  Try the next candidate rather than silently
                    # claiming success.
                    log.warning("Actuator %s declined to send (returned False)", actuator_name)
                    continue
                self.add_to_log(f"Sent response via {actuator_name}")
                # Speak web-chat responses aloud via TTS if available.
                if 'web' in actuator_name.lower() or 'chat' in actuator_name.lower():
                    tts = next(
                        (a for n, a in actuators.items() if 'tts' in n.lower()),
                        None,
                    )
                    if tts is not None:
                        try:
                            tts.actuate(response)
                        except Exception as tts_exc:
                            log.warning("TTS actuator failed: %s", tts_exc)
                return
            except Exception as exc:
                log.warning("Actuator %s failed: %s", actuator_name, exc)

        log.error("All actuators failed to send response for sensor=%s", self._sensor_name)
        self.add_to_log("FAILED: all actuators declined or errored")

    @staticmethod
    def _names_match(sensor_lower: str, actuator_lower: str) -> bool:
        """True if sensor and actuator share a meaningful keyword."""
        keywords = ('telegram', 'web_chat', 'webchat', 'chat', 'tts', 'microphone')
        return any(kw in sensor_lower and kw in actuator_lower for kw in keywords)
