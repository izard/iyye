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
Vision LLM client — sends an image + text prompt to a multimodal llama.cpp
server (gemma-4 + mmproj) and returns a text description.

The server must be started with `--mmproj-auto` (or an explicit `--mmproj`
path) so that it accepts image_url content in chat messages.

Configuration (environment variables):
    LLM_VISION_API_BASE  - base URL  (default: http://127.0.0.1:8081/v1)
    LLM_VISION_MODEL_ID  - model id  (default: local)
    LLM_VISION_API_KEY   - API key   (default: local)
    LLM_VISION_TIMEOUT   - timeout s (default: 60)
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("Iyye.VisionLLM")

_DEFAULT_SYSTEM = (
    "You are a vision assistant for an AI called Iyye. "
    "Describe the image concisely in 1-3 sentences: what you see, "
    "any text visible, notable objects, people, or actions. "
    "Be factual and brief."
)


class VisionClient:
    """
    Sends a JPEG/PNG image to a multimodal llama.cpp server and returns
    a text description.

    Uses the OpenAI vision API format:
        content: [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
                  {"type": "text", "text": "...prompt..."}]
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_tokens: int = 512,
    ) -> None:
        self.api_base   = (api_base or os.environ.get("LLM_VISION_API_BASE",
                           "http://127.0.0.1:8081/v1")).rstrip("/")
        self.model_id   = model_id or os.environ.get("LLM_VISION_MODEL_ID", "local")
        self.api_key    = api_key  or os.environ.get("LLM_VISION_API_KEY",  "local")
        self.timeout    = timeout
        self.max_tokens = max_tokens
        self._session   = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        log.info("VisionClient ready: %s @ %s", self.model_id, self.api_base)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_image_file(
        self,
        image_path: str | Path,
        prompt: str = "Describe what you see in this image.",
        system: str = _DEFAULT_SYSTEM,
    ) -> str:
        """Read an image file and return an LLM description."""
        data = Path(image_path).read_bytes()
        mime = "image/jpeg" if str(image_path).lower().endswith((".jpg", ".jpeg")) else "image/png"
        return self.describe_image_bytes(data, mime=mime, prompt=prompt, system=system)

    def describe_image_bytes(
        self,
        image_bytes: bytes,
        mime: str = "image/jpeg",
        prompt: str = "Describe what you see in this image.",
        system: str = _DEFAULT_SYSTEM,
    ) -> str:
        """Encode image bytes as base64 and return an LLM description."""
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        return self._call(data_url, prompt, system)

    def is_healthy(self) -> bool:
        """Return True if the vision server responds to /health."""
        try:
            r = self._session.get(
                self.api_base.replace("/v1", "") + "/health",
                timeout=5,
            )
            return r.ok
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, image_data_url: str, prompt: str, system: str) -> str:
        payload = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        }
        t0 = time.monotonic()
        try:
            resp = self._session.post(
                f"{self.api_base}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            log.info("VisionLLM: %.1fs, %d chars", time.monotonic() - t0, len(content))
            return content
        except Exception as exc:
            log.warning("VisionLLM call failed: %s", exc)
            raise
