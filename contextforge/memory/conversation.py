"""会话记忆：把对话历史持久化，让长任务可以跨请求继续。

与「长期记忆」的区别：会话记忆保存**原始对话**（含工具调用），
是上下文引擎的输入；长期记忆保存**提炼后的结论**，是跨任务的先验。
"""

from __future__ import annotations

import time
from typing import Any, Sequence

from ..storage import Database
from ..types import Message, new_id


class ConversationStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ #
    def ensure(self, conversation_id: str | None = None, title: str = "") -> str:
        conversation_id = conversation_id or new_id("conv")
        now = time.time()
        existing = self.db.query_one(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        )
        if existing is None:
            self.db.execute(
                "INSERT INTO conversations (conversation_id, title, created_at, updated_at, metadata)"
                " VALUES (?, ?, ?, ?, '{}')",
                (conversation_id, title or "新会话", now, now),
            )
        return conversation_id

    def append(self, conversation_id: str, message: Message) -> None:
        self.ensure(conversation_id)
        row = self.db.query_one(
            "SELECT COALESCE(MAX(ord), -1) AS last FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        next_ord = (row["last"] if row else -1) + 1
        tool_calls = message.get("tool_calls")
        self.db.execute(
            "INSERT INTO messages"
            " (conversation_id, ord, role, content, tool_calls, tool_call_id, name, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                next_ord,
                message.get("role", "user"),
                str(message.get("content") or ""),
                self.db.dump_json(tool_calls) if tool_calls else None,
                message.get("tool_call_id"),
                message.get("name"),
                time.time(),
            ),
        )
        self.db.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (time.time(), conversation_id),
        )

    def extend(self, conversation_id: str, messages: Sequence[Message]) -> None:
        for message in messages:
            self.append(conversation_id, message)

    def history(self, conversation_id: str, limit: int = 200) -> list[Message]:
        """取最近 limit 条消息。注意：不在这里裁剪，裁剪是上下文引擎的职责。"""
        rows = self.db.query(
            "SELECT role, content, tool_calls, tool_call_id, name FROM messages"
            " WHERE conversation_id = ? ORDER BY ord DESC LIMIT ?",
            (conversation_id, limit),
        )
        messages: list[Message] = []
        for row in reversed(rows):
            message: Message = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                message["tool_calls"] = self.db.load_json(row["tool_calls"], [])
            if row["tool_call_id"]:
                message["tool_call_id"] = row["tool_call_id"]
            if row["name"]:
                message["name"] = row["name"]
            messages.append(message)
        return messages

    def replace(self, conversation_id: str, messages: Sequence[Message]) -> None:
        """用压缩后的历史替换原始历史（上下文压缩落盘）。"""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        self.extend(conversation_id, messages)

    def clear(self, conversation_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

    def delete(self, conversation_id: str) -> bool:
        cursor = self.db.execute(
            "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
        )
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT c.conversation_id, c.title, c.created_at, c.updated_at,"
            " (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.conversation_id) AS n"
            " FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "messages": row["n"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def count(self, conversation_id: str) -> int:
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        return row["n"] if row else 0
