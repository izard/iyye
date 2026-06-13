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

Local MCP server that lets an agentic LLM setup communicate with you via Telegram.

Features:
- send_message(text, ...)
- get_new_messages(...)
- wait_for_message(...)
- get_dialog_state()
- set_default_chat(...)

Transport:
- stdio by default (ideal for local MCP clients)

Required env:
- TELEGRAM_BOT_TOKEN=123456:ABC...
Optional env:
- TELEGRAM_DEFAULT_CHAT_ID=<chat id to send to by default>
- TELEGRAM_ALLOWED_CHAT_IDS=123,456
- TELEGRAM_STATE_PATH=./telegram_mcp_state.sqlite3

How to get your chat_id:
1. Create a Telegram bot with @BotFather
2. Start a chat with your bot and send it any message
3. Run this server once and call get_new_messages()
4. The returned message object will include chat_id

Install:
    pip install mcp python-telegram-bot

Run directly:
    python telegram_mcp_server.py

Example MCP client config:
{
  "mcpServers": {
    "telegram": {
      "command": "python",
      "args": ["/absolute/path/to/telegram_mcp_server.py"],
      "env": {
        "TELEGRAM_BOT_TOKEN": "123456:ABC...",
        "TELEGRAM_DEFAULT_CHAT_ID": "123456789"
      }
    }
  }
}
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time

log = logging.getLogger(__name__)
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import Context, FastMCP
from telegram import Bot


# ----------------------------
# Persistent state
# ----------------------------

class StateDB:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                update_id INTEGER PRIMARY KEY,
                message_id INTEGER,
                chat_id INTEGER NOT NULL,
                user_id INTEGER,
                username TEXT,
                first_name TEXT,
                text TEXT,
                date_unix INTEGER,
                raw_json TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('in'))
            )
            """
        )
        self.conn.commit()

    def get_kv(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
        return row["value"] if row else default

    def set_kv(self, key: str, value: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO kv(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    def get_last_update_id(self) -> int:
        value = self.get_kv("last_update_id", "0")
        return int(value or "0")

    def set_last_update_id(self, update_id: int) -> None:
        self.set_kv("last_update_id", str(update_id))

    def get_default_chat_id(self) -> Optional[int]:
        value = self.get_kv("default_chat_id")
        return int(value) if value else None

    def set_default_chat_id(self, chat_id: int) -> None:
        self.set_kv("default_chat_id", str(chat_id))

    def store_incoming_message(
        self,
        update_id: int,
        message_id: Optional[int],
        chat_id: int,
        user_id: Optional[int],
        username: Optional[str],
        first_name: Optional[str],
        text: str,
        date_unix: Optional[int],
        raw_json: str,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO messages(
                update_id, message_id, chat_id, user_id, username, first_name,
                text, date_unix, raw_json, direction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'in')
            """,
            (
                update_id,
                message_id,
                chat_id,
                user_id,
                username,
                first_name,
                text,
                date_unix,
                raw_json,
            ),
        )
        self.conn.commit()

    def recent_messages(self, limit: int = 20, chat_id: Optional[int] = None) -> list[dict[str, Any]]:
        cur = self.conn.cursor()
        if chat_id is None:
            cur.execute(
                """
                SELECT update_id, message_id, chat_id, user_id, username, first_name,
                       text, date_unix, raw_json, direction
                FROM messages
                ORDER BY update_id DESC
                LIMIT ?
                """,
                (limit,),
            )
        else:
            cur.execute(
                """
                SELECT update_id, message_id, chat_id, user_id, username, first_name,
                       text, date_unix, raw_json, direction
                FROM messages
                WHERE chat_id = ?
                ORDER BY update_id DESC
                LIMIT ?
                """,
                (chat_id, limit),
            )
        rows = cur.fetchall()
        return [dict(row) for row in rows]


# ----------------------------
# Telegram bridge
# ----------------------------

class TelegramBridge:
    def __init__(
        self,
        token: str,
        state_db: StateDB,
        default_chat_id: Optional[int] = None,
        allowed_chat_ids: Optional[set[int]] = None,
    ) -> None:
        self.bot = Bot(token=token)
        self.db = state_db
        self.allowed_chat_ids = allowed_chat_ids or set()

        if default_chat_id is not None:
            self.db.set_default_chat_id(default_chat_id)

        self._lock = asyncio.Lock()

    def _resolve_chat_id(self, chat_id: Optional[int]) -> int:
        resolved = chat_id if chat_id is not None else self.db.get_default_chat_id()
        if resolved is None:
            raise ValueError(
                "No chat_id provided and no default chat configured. "
                "Send a message to the bot first, call get_new_messages(), then set_default_chat(chat_id)."
            )
        if self.allowed_chat_ids and resolved not in self.allowed_chat_ids:
            raise ValueError(f"chat_id {resolved} is not in TELEGRAM_ALLOWED_CHAT_IDS.")
        return resolved

    def _chat_allowed(self, chat_id: int) -> bool:
        return not self.allowed_chat_ids or chat_id in self.allowed_chat_ids

    async def send_message(
        self,
        text: str,
        chat_id: Optional[int] = None,
        disable_notification: bool = False,
    ) -> dict[str, Any]:
        target_chat = self._resolve_chat_id(chat_id)
        msg = await self.bot.send_message(
            chat_id=target_chat,
            text=text,
            disable_notification=disable_notification,
        )
        return {
            "ok": True,
            "chat_id": msg.chat_id,
            "message_id": msg.message_id,
            "date_unix": int(msg.date.timestamp()) if msg.date else None,
            "text": msg.text,
        }

    async def _pull_updates_once(self, timeout: int = 0) -> list[dict[str, Any]]:
        async with self._lock:
            last_update_id = self.db.get_last_update_id()
            try:
                updates = await self.bot.get_updates(
                    offset=last_update_id + 1,
                    timeout=timeout,
                    allowed_updates=["message"],
                )
            except Exception as exc:
                # NetworkError, TimedOut, Conflict, RetryAfter, etc.
                # Log and return empty — never crash the server over a transient API error.
                log.warning("Telegram getUpdates failed (offset=%d): %s",
                            last_update_id + 1, exc)
                return []

            out: list[dict[str, Any]] = []
            max_seen = last_update_id

            for upd in updates:
                max_seen = max(max_seen, upd.update_id)

                msg = upd.message
                if msg is None:
                    continue

                chat_id = msg.chat_id
                if not self._chat_allowed(chat_id):
                    continue

                text = msg.text or msg.caption or ""
                user = msg.from_user

                item = {
                    "update_id": upd.update_id,
                    "message_id": msg.message_id,
                    "chat_id": chat_id,
                    "user_id": user.id if user else None,
                    "username": user.username if user else None,
                    "first_name": user.first_name if user else None,
                    "date_unix": int(msg.date.timestamp()) if msg.date else None,
                    "text": text,
                    "raw": upd.to_dict(),
                }
                out.append(item)

                self.db.store_incoming_message(
                    update_id=upd.update_id,
                    message_id=msg.message_id,
                    chat_id=chat_id,
                    user_id=user.id if user else None,
                    username=user.username if user else None,
                    first_name=user.first_name if user else None,
                    text=text,
                    date_unix=int(msg.date.timestamp()) if msg.date else None,
                    raw_json=str(upd.to_dict()),
                )

            if max_seen > last_update_id:
                self.db.set_last_update_id(max_seen)

            return out

    async def get_new_messages(self, limit: int = 20) -> dict[str, Any]:
        updates = await self._pull_updates_once(timeout=0)
        updates = updates[-limit:]
        if updates:
            last_chat = updates[-1]["chat_id"]
            if self.db.get_default_chat_id() is None:
                self.db.set_default_chat_id(last_chat)

        return {
            "count": len(updates),
            "messages": updates,
            "last_update_id": self.db.get_last_update_id(),
            "default_chat_id": self.db.get_default_chat_id(),
        }

    async def wait_for_message(
        self,
        timeout_seconds: int = 60,
        from_chat_id: Optional[int] = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds

        while time.monotonic() < deadline:
            remaining = max(0, int(deadline - time.monotonic()))
            poll_timeout = max(1, min(remaining, 30))

            updates = await self._pull_updates_once(timeout=poll_timeout)
            if from_chat_id is not None:
                updates = [u for u in updates if u["chat_id"] == from_chat_id]

            if updates:
                msg = updates[-1]
                if self.db.get_default_chat_id() is None:
                    self.db.set_default_chat_id(msg["chat_id"])
                return {"ok": True, "message": msg}

        return {"ok": False, "message": None, "reason": "timeout"}

    async def get_me(self) -> dict[str, Any]:
        me = await self.bot.get_me()
        return {
            "id": me.id,
            "username": me.username,
            "first_name": me.first_name,
            "can_join_groups": me.can_join_groups,
            "can_read_all_group_messages": me.can_read_all_group_messages,
            "supports_inline_queries": me.supports_inline_queries,
        }

    def recent_messages(self, limit: int = 20, chat_id: Optional[int] = None) -> list[dict[str, Any]]:
        return self.db.recent_messages(limit=limit, chat_id=chat_id)

    def set_default_chat(self, chat_id: int) -> dict[str, Any]:
        if not self._chat_allowed(chat_id):
            raise ValueError(f"chat_id {chat_id} is not in TELEGRAM_ALLOWED_CHAT_IDS.")
        self.db.set_default_chat_id(chat_id)
        return {"ok": True, "default_chat_id": chat_id}

    def dialog_state(self) -> dict[str, Any]:
        return {
            "default_chat_id": self.db.get_default_chat_id(),
            "last_update_id": self.db.get_last_update_id(),
            "allowed_chat_ids": sorted(self.allowed_chat_ids),
            "state_path": self.db.path,
        }


# ----------------------------
# MCP app context
# ----------------------------

@dataclass
class AppContext:
    tg: TelegramBridge


def _parse_allowed_chat_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    out: set[int] = set()
    for part in raw.split(","):
        s = part.strip()
        if s:
            out.add(int(s))
    return out


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    state_path = os.environ.get("TELEGRAM_STATE_PATH", "./telegram_mcp_state.sqlite3")
    default_chat_id_raw = os.environ.get("TELEGRAM_DEFAULT_CHAT_ID")
    allowed_chat_ids_raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS")

    db = StateDB(state_path)
    allowed_chat_ids = _parse_allowed_chat_ids(allowed_chat_ids_raw)
    default_chat_id = int(default_chat_id_raw) if default_chat_id_raw else None

    tg = TelegramBridge(
        token=token,
        state_db=db,
        default_chat_id=default_chat_id,
        allowed_chat_ids=allowed_chat_ids,
    )
    yield AppContext(tg=tg)


mcp = FastMCP("telegram-mcp", lifespan=app_lifespan)


# ----------------------------
# MCP tools
# ----------------------------

def _app(ctx: Context) -> AppContext:
    return ctx.request_context.lifespan_context


@mcp.tool()
async def send_message(
    text: str,
    ctx: Context,
    chat_id: int | None = None,
    disable_notification: bool = False,
) -> dict[str, Any]:
    """
    Send a Telegram message to the configured/default chat or to an explicit chat_id.
    """
    return await _app(ctx).tg.send_message(
        text=text,
        chat_id=chat_id,
        disable_notification=disable_notification,
    )


@mcp.tool()
async def get_new_messages(
    ctx: Context,
    limit: int = 20,
) -> dict[str, Any]:
    """
    Fetch new incoming Telegram messages since the last poll and advance the update cursor.
    """
    limit = max(1, min(limit, 100))
    return await _app(ctx).tg.get_new_messages(limit=limit)


@mcp.tool()
async def wait_for_message(
    ctx: Context,
    timeout_seconds: int = 60,
    from_chat_id: int | None = None,
) -> dict[str, Any]:
    """
    Long-poll Telegram until a new message arrives or timeout expires.
    """
    timeout_seconds = max(1, min(timeout_seconds, 600))
    return await _app(ctx).tg.wait_for_message(
        timeout_seconds=timeout_seconds,
        from_chat_id=from_chat_id,
    )


@mcp.tool()
async def get_bot_info(ctx: Context) -> dict[str, Any]:
    """
    Return basic Telegram bot metadata.
    """
    return await _app(ctx).tg.get_me()


@mcp.tool()
def set_default_chat(
    chat_id: int,
    ctx: Context,
) -> dict[str, Any]:
    """
    Persist the default chat_id used by send_message().
    """
    return _app(ctx).tg.set_default_chat(chat_id)


@mcp.tool()
def get_dialog_state(ctx: Context) -> dict[str, Any]:
    """
    Show local persisted state: default chat, last update cursor, and allowlist.
    """
    return _app(ctx).tg.dialog_state()


@mcp.tool()
def recent_messages(
    ctx: Context,
    limit: int = 20,
    chat_id: int | None = None,
) -> list[dict[str, Any]]:
    """
    Return recently stored incoming messages from the local SQLite state.
    """
    limit = max(1, min(limit, 100))
    return _app(ctx).tg.recent_messages(limit=limit, chat_id=chat_id)


if __name__ == "__main__":
    # Default FastMCP transport is stdio, which is what local MCP clients want.
    mcp.run()

