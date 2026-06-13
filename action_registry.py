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
"""Action cards — the chat tool catalog, tracked separately from the persona
prompt and selected per conversation turn.

Each file in ``prompts/actions/*.md`` documents ONE chat ACTION: frontmatter
metadata (action type, capability, relevance triggers) plus the doc text shown
to the chat LLM.  ``select_actions()`` renders, for a given turn:

1. full card bodies for cards whose triggers match the current message /
   recent history ("the recipe", e.g. the stock-quote User-Agent requirement),
2. a one-line index of every other available card — so the model always knows
   a tool exists even when relevance matching misses a phrasing.

SECURITY: cards are *presentation only*.  ``capabilities.py`` profiles remain
the enforcement layer — the registry merely filters which docs are shown via
``profile.allows()``, so an untrusted telegram sender's prompt contains no
trust/persona/python documentation at all (smaller injection surface), while
execution-side gating in UserChatStream still runs regardless of what was
displayed.

Card bodies are inserted verbatim (dynamic ``{key}`` values via str.replace,
never ``str.format``), so JSON examples with braces are safe — the rendered
block enters chat_response.md as a substituted variable, which format_map
does not recurse into.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.Actions")

ACTIONS_DIR = PROJECT_ROOT / "prompts" / "actions"

# Hardwired floor for _extract_action when the cards directory is missing or
# unreadable: stripping/validation must never break chat over a packaging
# problem.
_FALLBACK_TYPES = frozenset(
    {"wikipedia", "url", "python", "trust", "untrust", "llm", "persona", "plan"}
)


class ActionCard:
    """One parsed action card."""

    def __init__(self, name: str, meta: Dict[str, Any], body: str):
        self.name = name
        self.action_type: str = meta.get("action_type", name)
        self.extra_types: List[str] = meta.get("extra_types", [])
        # Capability checked against the chat profile; defaults to the action
        # type itself (matches how CapabilityProfile sets are keyed).
        self.capability: str = meta.get("capability", self.action_type)
        self.summary: str = meta.get("summary", "")
        self.triggers: List[str] = meta.get("triggers", [])
        self.trigger_patterns: List[re.Pattern] = []
        for pat in meta.get("trigger_patterns", []):
            try:
                self.trigger_patterns.append(re.compile(pat))
            except re.error as exc:
                log.warning("Card %s: bad trigger pattern %r: %s", name, pat, exc)
        # A recipe card can extend a base card: when the recipe matches, the
        # base card's full body is included too.
        self.extends: Optional[str] = meta.get("extends") or None
        # index=False keeps a card out of the one-line index (it is then only
        # ever visible through its base card when triggered).
        self.index: bool = str(meta.get("index", "true")).lower() != "false"
        self.body = body.strip()
        self._trigger_res = [
            re.compile(rf"\b{re.escape(t.lower())}\b")
            for t in self.triggers if t
        ]

    def matches(self, text_lower: str, raw_text: str) -> bool:
        if any(r.search(text_lower) for r in self._trigger_res):
            return True
        return any(p.search(raw_text) for p in self.trigger_patterns)


def _parse_card(path: Path) -> Optional[ActionCard]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Cannot read action card %s: %s", path, exc)
        return None
    meta: Dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].strip().splitlines():
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key in ("triggers", "extra_types"):
                    meta[key] = [v.strip() for v in value.split(",") if v.strip()]
                elif key == "trigger_patterns":
                    # Regexes legitimately contain commas ({3,5}) — split on
                    # ';;' so a single pattern needs no escaping.
                    meta[key] = [v.strip() for v in value.split(";;") if v.strip()]
                else:
                    meta[key] = value
            body = text[end + 4:]
    if not meta.get("action_type"):
        log.warning("Action card %s missing action_type — skipped", path.name)
        return None
    return ActionCard(path.stem, meta, body)


# path -> (mtime, card); reloaded per call when files change (cheap stat).
_cache: Dict[Path, Tuple[float, ActionCard]] = {}


def _load_cards() -> List[ActionCard]:
    cards: List[ActionCard] = []
    if not ACTIONS_DIR.is_dir():
        return cards
    for path in sorted(ACTIONS_DIR.glob("*.md")):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        cached = _cache.get(path)
        if cached is None or cached[0] != mtime:
            card = _parse_card(path)
            if card is None:
                continue
            _cache[path] = (mtime, card)
        cards.append(_cache[path][1])
    return cards


def action_types() -> frozenset:
    """Every action type any card documents — feeds ACTION-line validation.

    Falls back to the hardwired set when no cards load, so chat keeps
    stripping/validating ACTION lines even if prompts/actions/ is missing.
    """
    types = set()
    for card in _load_cards():
        types.add(card.action_type)
        types.update(card.extra_types)
    return frozenset(types) if types else _FALLBACK_TYPES


def select_actions(
    message: str,
    history: str = "",
    profile: Any = None,
    dynamic: Optional[Dict[str, str]] = None,
) -> str:
    """Render the per-turn action documentation block.

    *profile* is a CapabilityProfile (or anything with ``allows(str)``);
    None shows every card.  *dynamic* values replace ``{key}`` placeholders
    in card bodies (plain replace — bodies are never format()-ed).

    Returns "" when the sender's profile allows no actions at all.
    """
    cards = _load_cards()
    if profile is not None:
        cards = [c for c in cards if profile.allows(c.capability)]
    if not cards:
        return ""

    by_type: Dict[str, ActionCard] = {}
    for c in cards:
        # Base cards keyed by action_type for extends-resolution; first wins.
        by_type.setdefault(c.name, c)
        by_type.setdefault(c.action_type, c)

    text_lower = f"{message}\n{history}".lower()
    raw_text = f"{message}\n{history}"

    matched: List[ActionCard] = []
    matched_names = set()

    def _include(card: ActionCard) -> None:
        if card.name in matched_names:
            return
        matched_names.add(card.name)
        matched.append(card)

    for card in cards:
        if not card.matches(text_lower, raw_text):
            continue
        # A matched recipe pulls in its base card first so the recipe text
        # reads as an extension of the base usage doc.
        if card.extends:
            base = by_type.get(card.extends)
            if base is not None:
                _include(base)
        _include(card)

    dynamic = dynamic or {}

    def _render_body(card: ActionCard) -> str:
        body = card.body
        for key, value in dynamic.items():
            body = body.replace("{" + key + "}", str(value))
        return body

    parts: List[str] = []
    if matched:
        parts.append("Action details relevant to this conversation:")
        parts.extend(_render_body(c) for c in matched)

    index_lines = [
        f"- {c.action_type}: {c.summary}"
        for c in cards
        if c.index and c.name not in matched_names and c.summary
    ]
    if index_lines:
        parts.append(
            "Other available actions (brief — you may use any of them; "
            "emit the ACTION line with the fields the action needs):"
        )
        parts.append("\n".join(index_lines))

    block = "\n\n".join(parts)
    if matched:
        log.debug("Action cards matched: %s", ", ".join(c.name for c in matched))
    return block


__all__ = ["ActionCard", "action_types", "select_actions", "ACTIONS_DIR"]
