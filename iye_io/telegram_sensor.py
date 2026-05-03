#!/usr/bin/env python3
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
Telegram sensor — polls the Telegram Bot API directly via HTTP (no MCP subprocess).

Uses a background thread (same pattern as HardwareSensor) so new messages are
fetched independently of main-loop ticks.  The update cursor is persisted in a
small JSON sidecar file so restarts don't re-deliver already-seen messages.

Required env:
  TELEGRAM_BOT_TOKEN=123456:ABC...

Optional env:
  TELEGRAM_DEFAULT_CHAT_ID=<int>  — stored on first message if absent
  TELEGRAM_ALLOWED_CHAT_IDS=123,456  — comma-separated allowlist (empty = allow all)
  TELEGRAM_STATE_PATH=./telegram_sensor_state.json
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from iyye_base import PROJECT_ROOT, BaseSensorQueue

log = logging.getLogger("Iyye.Sensors.Telegram")

_API_BASE = "https://api.telegram.org/bot{token}/{method}"


class TelegramSensor(BaseSensorQueue):
    """
    Polls Telegram getUpdates in a background thread and pushes individual
    message dicts onto the sensor queue.

    Each queued item has the shape::

        {
            "update_id": int,
            "message_id": int,
            "chat_id": int,
            "user_id": int | None,
            "username": str | None,
            "first_name": str | None,
            "date_unix": int | None,
            "text": str,
            "timestamp": "<iso8601>",
        }
    """

    def __init__(
        self,
        poll_interval: float = 5.0,
        maxlen: int = 10_000,
        # kept for compatibility with discovery/config that passes these kwargs
        mcp_script_path: str = "./iyye_io/mcp_telegram.py",
    ):
        super().__init__(name="telegram_sensor", maxlen=maxlen)

        self.poll_interval = poll_interval
        self._token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._running = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tg_sensor"
        )

        # Allowed-chat filter (empty set = accept all)
        raw_allowed = os.getenv("TELEGRAM_ALLOWED_CHAT_IDS", "")
        self._allowed: Set[int] = set()
        for part in raw_allowed.split(","):
            s = part.strip()
            if s:
                try:
                    self._allowed.add(int(s))
                except ValueError:
                    pass

        # Persistent state: last seen update_id and default_chat_id
        raw_state_path = Path(os.getenv("TELEGRAM_STATE_PATH", "telegram_sensor_state.json"))
        self._state_path = raw_state_path if raw_state_path.is_absolute() else PROJECT_ROOT / raw_state_path
        self._last_update_id: int = 0
        self._default_chat_id: Optional[int] = None
        self._load_state()

        # Allow env override of default chat id (takes precedence over saved state)
        env_chat = os.getenv("TELEGRAM_DEFAULT_CHAT_ID")
        if env_chat:
            try:
                self._default_chat_id = int(env_chat)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Persistent state helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        try:
            if self._state_path.exists():
                data = json.loads(self._state_path.read_text())
                self._last_update_id = int(data.get("last_update_id", 0))
                chat = data.get("default_chat_id")
                if chat is not None:
                    self._default_chat_id = int(chat)
        except Exception as exc:
            log.warning("Could not load Telegram sensor state: %s", exc)

    def _save_state(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(
                    {
                        "last_update_id": self._last_update_id,
                        "default_chat_id": self._default_chat_id,
                    }
                )
            )
        except Exception as exc:
            log.warning("Could not save Telegram sensor state: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_collection(self) -> None:
        """Start background polling thread."""
        if not self._token:
            log.error("TELEGRAM_BOT_TOKEN not set — Telegram sensor disabled")
            return
        if not self._running:
            self._running = True
            self._executor.submit(self._poll_loop)
            log.info(
                "Telegram sensor started (interval=%.1fs, offset=%d)",
                self.poll_interval,
                self._last_update_id + 1,
            )

    def stop_collection(self) -> None:
        """Stop background polling."""
        self._running = False
        self._executor.shutdown(wait=False)
        log.info("Telegram sensor stopped")

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread: calls getUpdates and pushes messages."""
        while self._running:
            try:
                messages = self._fetch_updates()
                for msg in messages:
                    self.push(msg)
                    log.debug(
                        "Telegram: queued message from chat_id=%s text=%r",
                        msg.get("chat_id"),
                        (msg.get("text") or "")[:80],
                    )
            except Exception as exc:
                log.error("Telegram poll loop error: %s", exc)
            time.sleep(self.poll_interval)

    def _fetch_updates(self) -> List[Dict[str, Any]]:
        """
        Call getUpdates with offset = last_update_id + 1.
        Returns a list of normalised message dicts.
        """
        import requests

        url = _API_BASE.format(token=self._token, method="getUpdates")
        try:
            resp = requests.get(
                url,
                params={
                    "offset": self._last_update_id + 1,
                    "limit": 10,
                    "timeout": 0,
                    "allowed_updates": json.dumps(["message"]),
                },
                timeout=10,
            )
        except Exception as exc:
            log.warning("Telegram getUpdates network error: %s", exc)
            return []

        if not resp.ok:
            log.warning(
                "Telegram getUpdates HTTP %s: %s", resp.status_code, resp.text[:200]
            )
            return []

        try:
            data = resp.json()
        except Exception as exc:
            log.warning("Telegram getUpdates JSON parse error: %s", exc)
            return []

        if not data.get("ok"):
            log.warning("Telegram getUpdates not ok: %s", data)
            return []

        out: List[Dict[str, Any]] = []
        max_seen = self._last_update_id

        for upd in data.get("result", []):
            uid = upd.get("update_id", 0)
            max_seen = max(max_seen, uid)

            msg = upd.get("message")
            if not msg:
                continue

            chat_id = msg.get("chat", {}).get("id")
            if self._allowed and chat_id not in self._allowed:
                continue

            user = msg.get("from") or {}
            text = msg.get("text") or msg.get("caption") or ""
            date_unix = msg.get("date")

            item: Dict[str, Any] = {
                "update_id": uid,
                "message_id": msg.get("message_id"),
                "chat_id": chat_id,
                "user_id": user.get("id"),
                "username": user.get("username"),
                "first_name": user.get("first_name"),
                "date_unix": date_unix,
                "text": text,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            out.append(item)

            # Auto-learn default chat id from first seen message
            if self._default_chat_id is None and chat_id is not None:
                self._default_chat_id = chat_id
                log.info("Telegram sensor: learned default chat_id=%s", chat_id)

        if max_seen > self._last_update_id:
            self._last_update_id = max_seen
            self._save_state()

        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "running": self._running,
            "queue_size": len(self),
            "last_update_id": self._last_update_id,
            "default_chat_id": self._default_chat_id,
        }
