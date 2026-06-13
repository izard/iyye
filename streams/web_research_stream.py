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
# streams/web_research_stream.py
#!/usr/bin/env python3
"""
Web Research Stream — fetches Wikipedia summaries or arbitrary URLs on behalf
of a chat stream and sends the result back to the originating chat.

Created by StreamFactory when UserChatStream queues a research task in
brain._pending_research_tasks.
"""

import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, TYPE_CHECKING

from iyye_base import ProcessingStream
from llm_scheduler import LLMCall, LLMConsumerMixin

if TYPE_CHECKING:
    from main_loop import IyyeBrain

log = logging.getLogger("Iyye")

_WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKIPEDIA_SEARCH_URL  = (
    "https://en.wikipedia.org/w/api.php"
    "?action=opensearch&search={query}&limit=1&format=json"
)
_USER_AGENT = "IyyeBot/1.0 (research assistant; contact via Telegram)"


class WebResearchStream(LLMConsumerMixin, ProcessingStream):
    """
    Executes a single web research task (Wikipedia lookup or URL fetch),
    then sends the result as a follow-up message to the originating chat
    and retires itself.

    The URL-rephrasing LLM call goes through the async scheduler: the stream
    fetches (synchronously), submits the rephrase, then stays alive across ticks
    until the result lands, delivers it, and retires.
    """

    # Created on-demand by StreamFactory, not loaded by the stream loader.
    _factory_created: bool = True

    def __init__(self, task: Dict[str, Any], brain: "IyyeBrain") -> None:
        query_or_url = task.get('query') or task.get('url', 'unknown')
        safe = query_or_url[:30].replace(' ', '_')
        super().__init__(name=f"research_{safe}")
        self.brain = brain
        self.task = task
        self.priority = 4
        self._can_be_conscious = False
        self._done = False
        # 'fetch' until the page is fetched; 'rephrasing' while the async
        # rephrase job is in flight.
        self._stage = 'fetch'
        self._raw: Optional[str] = None

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Waiting on an async rephrase?
        if self._stage == 'rephrasing':
            return self._poll_rephrase(context)

        if self._done:
            self._retire()
            return None

        self._done = True  # fetch exactly once
        task_type = self.task.get('type')

        try:
            if task_type == 'wikipedia':
                # Wikipedia summaries are already clean prose — send as-is.
                result = self._fetch_wikipedia(self.task.get('query', ''))
            elif task_type == 'url':
                raw = self._fetch_url(self.task.get('url', ''))
                # A fetched web page (esp. a search engine) is JS/CSS/anti-bot
                # junk after crude tag-stripping.  NEVER send it raw: if it's
                # unusable, answer gracefully; otherwise have the LLM phrase an
                # answer to the user's question from it (async).
                if self._looks_unusable(raw):
                    log.info("WebResearchStream: url content unusable, declining raw dump")
                    result = ("I tried to look that up, but that page didn't return "
                              "readable content I could use. Want me to check "
                              "Wikipedia, or do you have a specific source?")
                elif self._submit_rephrase(raw):
                    # Stay alive until the rephrase result lands.
                    self._raw = raw
                    self._stage = 'rephrasing'
                    return None
                else:
                    # Scheduler unavailable — fall back to the raw text.
                    result = raw
            else:
                result = f"Unknown research task type: {task_type!r}"
        except Exception as exc:
            result = f"Research failed: {exc}"
            log.warning("WebResearchStream: %s", exc)

        self._deliver(result, context)
        return {'type': 'research_result', 'task': self.task, 'result': result}

    def _submit_rephrase(self, raw: str) -> bool:
        """Submit the URL-content rephrase to the async scheduler."""
        question = (self.task.get("user_text")
                    or self.task.get("query")
                    or self.task.get("url", "the user's question"))
        return self._llm_submit(
            role="fast", kind="research",
            call=LLMCall.from_file(
                "rephrase_tool_result",
                user_message=question,
                tool_output=raw[:3000],
                conversation_history="(none)",
            ),
            client_kwargs={"no_think": True},
        )

    def _poll_rephrase(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply the finished rephrase, or fall back to the raw text."""
        result = self._llm_poll()
        if result is None:
            if self._llm_busy():
                return None
            # Result was dropped (cycle rotation) — deliver the raw text.
            self._deliver(self._raw or "I couldn't read that page.", context)
            return None
        if result.discarded or not result.ok or not result.text:
            text = self._raw or "I couldn't read that page."
        else:
            text = result.text
        self._deliver(text, context)
        return {'type': 'research_result', 'task': self.task, 'result': text}

    def _deliver(self, result: str, context: Dict[str, Any]) -> None:
        """Log, send to the originating chat, and retire."""
        self.add_to_log(f"Research result ({self.task.get('type')}): {result[:500]}")
        self.checkpoint()
        self._send_result(result, context)
        self._retire()

    # ------------------------------------------------------------------
    # Fetch helpers
    # ------------------------------------------------------------------

    def _fetch_wikipedia(self, query: str) -> str:
        """Search Wikipedia and return a summary paragraph."""
        if not query:
            return "No search query provided."

        # Step 1: find the canonical page title via opensearch
        search_url = _WIKIPEDIA_SEARCH_URL.format(
            query=urllib.parse.quote(query)
        )
        raw = self._get(search_url)
        if raw is None:
            return "Could not reach Wikipedia."
        try:
            results = json.loads(raw)
            titles = results[1] if len(results) > 1 else []
            if not titles:
                return f"No Wikipedia article found for: {query!r}"
            title = titles[0]
        except (json.JSONDecodeError, IndexError):
            return "Wikipedia search returned unexpected data."

        # Step 2: fetch the summary of the canonical article
        summary_url = _WIKIPEDIA_SUMMARY_URL.format(
            title=urllib.parse.quote(title, safe='')
        )
        raw = self._get(summary_url)
        if raw is None:
            return f"Found article '{title}' but could not fetch its summary."
        try:
            data = json.loads(raw)
            # If this is a disambiguation page, retry with the second opensearch result
            # or the first listed topic (e.g. "Stinging nettle" → "Urtica dioica").
            if data.get('type') == 'disambiguation':
                fallback_titles = results[1] if len(results) > 1 else []
                retry_title = None
                if len(fallback_titles) > 1:
                    retry_title = fallback_titles[1]  # second opensearch hit
                else:
                    # Try extracting the first link from the disambiguation extract
                    dis_extract = data.get('extract', '')
                    import re as _re
                    m = _re.search(r':\s*([A-Z][^\n,;]+)', dis_extract)
                    if m:
                        retry_title = m.group(1).strip()
                if retry_title:
                    retry_url = _WIKIPEDIA_SUMMARY_URL.format(
                        title=urllib.parse.quote(retry_title, safe='')
                    )
                    retry_raw = self._get(retry_url)
                    if retry_raw:
                        try:
                            data = json.loads(retry_raw)
                            title = data.get('title', retry_title)
                        except json.JSONDecodeError:
                            pass  # fall through with disambiguation data

            extract = data.get('extract', '').strip()
            page_url = data.get('content_urls', {}).get('desktop', {}).get('page', '')
            if not extract:
                return f"Wikipedia article '{title}' has no summary text."
            # Return up to 3000 chars (Telegram limit is 4096; leave room for title/URL)
            summary = extract[:3000]
            if len(extract) > 3000:
                summary += "…"
            return f"**{title}**\n{summary}\n{page_url}"
        except json.JSONDecodeError:
            return "Could not parse Wikipedia summary."

    def _fetch_url(self, url: str) -> str:
        """Fetch a URL and return up to 1500 chars of plain text."""
        if not url:
            return "No URL provided."
        raw = self._get(url, content_type_check=False)
        if raw is None:
            return f"Could not fetch URL: {url}"
        # Strip obvious HTML tags for readability
        import re
        text = re.sub(r'<[^>]+>', ' ', raw)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:1500] + ("…" if len(text) > 1500 else "")

    def _get(self, url: str, content_type_check: bool = True) -> Optional[str]:
        """HTTP GET with a short timeout; returns response body as str or None."""
        try:
            req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as resp:
                charset = 'utf-8'
                ct = resp.headers.get_content_type() or ''
                if content_type_check and 'html' in ct and 'wikipedia' not in url:
                    return None  # skip non-text pages for URL fetches
                return resp.read().decode(charset, errors='replace')
        except Exception as exc:
            log.debug("WebResearchStream._get(%s): %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Result cleaning / phrasing
    # ------------------------------------------------------------------

    # Markers that indicate the "text" is really web boilerplate / JS / a
    # base64 blob rather than readable content.
    _JUNK_MARKERS = (
        "sourcemappingurl", "data:application/json", "window.google",
        "function(", "var _", "{background", "googleusercontent",
        "enable javascript", "<!doctype", "noscript", ".call(this)",
    )

    def _looks_unusable(self, text: str) -> bool:
        """True if *text* is web/JS junk rather than readable prose.

        Catches the search-engine / anti-bot page case (the source of the raw
        JSON/JS that was leaking to users)."""
        if not text or len(text.strip()) < 20:
            return True
        low = text.lower()
        if any(m in low for m in self._JUNK_MARKERS):
            return True
        # Prose ratio: fraction of alphabetic/space chars.  JS/CSS/base64 is
        # heavy on symbols and digits.
        sample = text[:1000]
        prose = sum(c.isalpha() or c.isspace() for c in sample)
        return (prose / max(1, len(sample))) < 0.6

    # ------------------------------------------------------------------
    # Result delivery
    # ------------------------------------------------------------------

    def _send_result(self, result: str, context: Dict[str, Any]) -> None:
        """Send the research result to the originating chat actuator."""
        actuators: Dict[str, Any] = context.get('actuators') or getattr(self.brain, 'actuators', {})
        sensor_name = self.task.get('sensor_name', '')
        chat_id = self.task.get('chat_id')

        # Find the matching actuator the same way UserChatStream does
        sensor_lower = sensor_name.lower()
        keywords = ('telegram', 'web_chat', 'webchat', 'chat')
        candidates = [
            name for name in actuators
            if any(kw in sensor_lower and kw in name.lower() for kw in keywords)
        ]
        if not candidates:
            candidates = list(actuators.keys())[:1]

        from iyye_base import ACTUATE_SUPPRESSED
        for actuator_name in candidates:
            actuator = actuators[actuator_name]
            try:
                if 'telegram' in actuator_name.lower() and chat_id:
                    payload = json.dumps({'text': result, 'chat_id': chat_id})
                else:
                    payload = result
                # A research answer is user-visible — don't let it be dropped as
                # a duplicate (the raw-data net still applies).
                ok = actuator.actuate(payload, allow_duplicate=True)
                if ok is ACTUATE_SUPPRESSED:
                    log.warning("WebResearchStream: %s suppressed result (guardrail)",
                                actuator_name)
                    continue
                if ok is not False:
                    self.add_to_log(f"Sent research result via {actuator_name}")
                    return
            except Exception as exc:
                log.warning("WebResearchStream: actuator %s failed: %s", actuator_name, exc)

        log.error("WebResearchStream: all actuators failed for task %s", self.task)

    def _retire(self) -> None:
        try:
            self.request_retire("research task complete")
            self.brain.streams.remove(self)
        except ValueError:
            pass
