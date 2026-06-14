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
"""
Generic local LLM client using smolagents OpenAIServerModel.

Connects to any OpenAI-compatible HTTP endpoint (llama.cpp, LM Studio, vLLM, etc.)
and loads prompt templates from the prompts/ directory.

Configuration (environment variables):
    LLM_API_BASE   - base URL of the local server  (default: http://127.0.0.1:8080/v1)
    LLM_MODEL_ID   - model identifier               (default: local)
    LLM_API_KEY    - API key if required             (default: local)
    LLM_TIMEOUT    - request timeout in seconds      (default: 60)
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from smolagents import OpenAIServerModel

from iyye_base import PROJECT_ROOT

log = logging.getLogger("Iyye.LLM")

_PROMPTS_DIR = PROJECT_ROOT / "prompts"


def _strip_frontmatter(raw: str) -> str:
    """Drop a leading YAML frontmatter block (--- ... ---)."""
    return re.sub(r"^---\n.*?\n---\n", "", raw, flags=re.DOTALL).strip()


def _load_prompt(name: str) -> str:
    """Load a prompt template, resolving through the version registry first.

    Default behaviour is unchanged: with no registered non-base version, the
    registry returns None and we load the shipped ``prompts/<name>.md`` file.
    When a learned version is active, its content is used instead — letting the
    sleep self-improvement pass trial prompt rewrites (gap #6) without any call
    site change."""
    try:
        from prompt_registry import get_registry
        active = get_registry().active_content(name)
        if active is not None:
            return _strip_frontmatter(active)
    except Exception:
        pass  # registry must never break prompt loading
    raw = (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return _strip_frontmatter(raw)


def active_prompt_version(name: str) -> str:
    """Version id serving *name* right now ("base" or a learned vid) — for
    journaling so outcomes can be attributed to the version that produced them."""
    try:
        from prompt_registry import get_registry
        return get_registry().active_version_id(name)
    except Exception:
        return "base"


class LLMClient:
    """
    Thin wrapper around smolagents OpenAIServerModel for single-turn completions.

    Usage:
        client = LLMClient()
        result = client.complete("What is 2+2?")
        result = client.complete_from_file("extract_facts", stream_name="...", stream_output="...")
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_tokens: Optional[int] = None,
        no_think: bool = False,
    ) -> None:
        self.api_base   = api_base   or os.environ.get("LLM_API_BASE", "http://127.0.0.1:8080/v1")
        self.model_id   = model_id   or os.environ.get("LLM_MODEL_ID", "local")
        self.api_key    = api_key    or os.environ.get("LLM_API_KEY", "local")
        self.timeout    = timeout    or int(os.environ.get("LLM_TIMEOUT", "300"))
        # Hard cap on generated tokens — prevents multi-hour hangs when the LLM
        # generates tokens slowly but steadily (each token within the per-chunk
        # read timeout, so the 60 s timeout never fires).
        self.max_tokens = max_tokens or int(os.environ.get("LLM_MAX_TOKENS", "2048"))
        # When True, prepend a system instruction to suppress internal reasoning
        # (thinking tokens).  Speeds up STM/alignment/chat calls on thinking models
        # like Gemma 4, where reasoning tokens eat into max_tokens budget.
        self.no_think   = no_think

        # smolagents does not forward the timeout kwarg to the openai client,
        # so we pass it via client_kwargs to ensure httpx actually uses it.
        import httpx
        self._model = OpenAIServerModel(
            model_id=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            client_kwargs={"timeout": httpx.Timeout(
                connect=10.0, read=float(self.timeout),
                write=30.0, pool=30.0,
            )},
        )
        log.info("LLMClient ready: %s @ %s (max_tokens=%d, no_think=%s)",
                 self.model_id, self.api_base, self.max_tokens, self.no_think)

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------

    def complete(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a single-turn prompt and return the text response."""
        messages = []
        if self.no_think:
            prefix = "Do not think or reason internally. Respond directly and concisely."
            combined = f"{prefix}\n\n{system_prompt}" if system_prompt else prefix
            messages.append({"role": "system", "content": combined})
        elif system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt_chars = sum(len(m["content"]) for m in messages)
        log.info("LLM request: %d chars", prompt_chars)
        log.debug("LLM prompt:\n%s", user_prompt)

        t0 = time.monotonic()
        response = self._model(messages, max_tokens=self.max_tokens)
        elapsed = time.monotonic() - t0

        content = response.content.strip()
        log.info("LLM response: %d chars in %.1fs", len(content), elapsed)
        log.debug("LLM response:\n%s", content)

        return content

    # ------------------------------------------------------------------
    # Prompt-file helpers
    # ------------------------------------------------------------------

    def complete_from_file(self, prompt_name: str, **variables: str) -> str:
        """
        Load prompts/<prompt_name>.md, fill in {variables}, and call complete().

        Example:
            client.complete_from_file(
                "extract_facts",
                stream_name="planning_stream",
                stream_output="...",
            )
        """
        template = _load_prompt(prompt_name)
        prompt = template.format_map(variables)
        return self.complete(prompt)

    def load_prompt(self, prompt_name: str) -> str:
        """Return a raw prompt template string (frontmatter stripped)."""
        return _load_prompt(prompt_name)
