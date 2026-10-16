"""长期记忆。

必须把三件容易混淆的事分开（这也是项目文档里反复强调的一点）：

| 机制 | 回答的问题 | 本项目的实现 |
| --- | --- | --- |
| RAG | 从大量资料里找出**此刻**需要的信息 | `contextforge.rag` |
| 会话记忆 | 这一轮对话之前说过什么 | `conversation.py` + SQLite |
| 长期记忆 | 哪些信息应该**跨任务**保留 | 本文件 |
| 上下文工程 | 这一轮究竟把哪些信息交给模型 | `contextforge.context` |

向量库不是记忆的全部，RAG 也不是记忆的全部。长期记忆存的是**结论性、稳定、
量少**的信息（用户偏好、项目事实、任务进度、验证过的流程），因此刻意不做
向量化，而用「词面相关性 + 重要度 + 时间衰减 + 命中反馈」的综合排序。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..storage import Database
from ..types import new_id
from ..rag.tokenize import tokenize

KINDS = ("preference", "fact", "goal", "progress", "workflow")
#: 各类型的基础权重 —— 偏好与目标比零散事实更值得长期保留
KIND_WEIGHT = {
    "preference": 1.25,
    "goal": 1.20,
    "workflow": 1.10,
    "fact": 1.0,
    "progress": 0.95,
}
DEFAULT_TTL = {
    "progress": 30 * 24 * 3600,  # 任务进度 30 天过期
}
HALF_LIFE_DAYS = 45.0


@dataclass
class Memory:
    memory_id: str
    kind: str
    content: str
    importance: float = 0.5
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    hit_count: int = 0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "kind": self.kind,
            "content": self.content,
            "importance": round(self.importance, 3),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "hit_count": self.hit_count,
            "source": self.source,
            "score": round(self.score, 4),
        }


class MemoryStore:
    """长期记忆仓储。"""

    def __init__(self, db: Database, dedup_threshold: float = 0.8) -> None:
        self.db = db
        self.dedup_threshold = dedup_threshold

    # ------------------------------------------------------------------ #
    def remember(
        self,
        content: str,
        kind: str = "fact",
        importance: float = 0.6,
        ttl_seconds: float | None = None,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """写入一条记忆。内容高度相似时合并，避免重复记忆淹没检索。"""
        content = " ".join((content or "").split())
        if not content:
            raise ValueError("记忆内容不能为空")

        kind = kind if kind in KINDS else "fact"
        existing = self._find_similar(content, kind)
        now = time.time()

        if existing is not None:
            merged = Memory(
                memory_id=existing["memory_id"],
                kind=kind,
                content=content,
                importance=min(1.0, max(existing["importance"], importance) + 0.05),
                created_at=existing["created_at"],
                updated_at=now,
                expires_at=existing["expires_at"],
                hit_count=existing["hit_count"],
                source=source or existing["source"],
                metadata={**self.db.load_json(existing["metadata"], {}), **(metadata or {})},
            )
            self._upsert(merged)
            return merged

        ttl = ttl_seconds
        if ttl is None and kind in DEFAULT_TTL:
            ttl = DEFAULT_TTL[kind]

        memory = Memory(
            memory_id=new_id("mem"),
            kind=kind,
            content=content,
            importance=max(0.0, min(1.0, importance)),
            created_at=now,
            updated_at=now,
            expires_at=(now + ttl) if ttl else None,
            source=source,
            metadata=metadata or {},
        )
        self._upsert(memory)
        return memory

    def recall(
        self,
        query: str,
        top_k: int = 5,
        kinds: Sequence[str] | None = None,
        min_score: float = 0.05,
    ) -> list[Memory]:
        """按综合得分召回。"""
        self.expire()
        memories = self._all(kinds)
        if not memories:
            return []

        query_tokens = set(tokenize(query))
        now = time.time()
        scored: list[Memory] = []

        for memory in memories:
            tokens = set(tokenize(memory.content))
            if query_tokens:
                overlap = len(query_tokens & tokens) / len(query_tokens)
            else:
                overlap = 0.0

            # 时间衰减：45 天半衰期
            age_days = max((now - memory.updated_at) / 86400, 0.0)
            decay = math.pow(0.5, age_days / HALF_LIFE_DAYS)
            # 命中反馈：经常被用到的记忆更可能有价值，但收敛很快
            usage = 1.0 + min(memory.hit_count, 10) * 0.03

            memory.score = round(
                (0.55 * overlap + 0.30 * memory.importance + 0.15 * decay)
                * KIND_WEIGHT.get(memory.kind, 1.0)
                * usage,
                6,
            )
            if memory.score >= min_score:
                scored.append(memory)

        scored.sort(key=lambda item: (item.score, item.updated_at), reverse=True)
        results = scored[: max(top_k, 0)]
        self.touch([memory.memory_id for memory in results])
        return results

    def touch(self, memory_ids: Sequence[str]) -> None:
        if not memory_ids:
            return
        with self.db.transaction() as conn:
            conn.executemany(
                "UPDATE memories SET hit_count = hit_count + 1 WHERE memory_id = ?",
                [(memory_id,) for memory_id in memory_ids],
            )

    def all(self, kinds: Sequence[str] | None = None, limit: int = 200) -> list[Memory]:
        return self._all(kinds, limit)

    def get(self, memory_id: str) -> Memory | None:
        row = self.db.query_one("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
        return self._row_to_memory(row) if row else None

    def update(self, memory_id: str, **fields: Any) -> bool:
        allowed = {"content", "kind", "importance", "expires_at", "source"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return False
        updates["updated_at"] = time.time()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        params = list(updates.values()) + [memory_id]
        cursor = self.db.execute(f"UPDATE memories SET {assignments} WHERE memory_id = ?", params)
        return cursor.rowcount > 0

    def forget(self, memory_id: str) -> bool:
        cursor = self.db.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        return cursor.rowcount > 0

    def expire(self, now: float | None = None) -> int:
        now = now or time.time()
        cursor = self.db.execute(
            "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,)
        )
        return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT kind, COUNT(*) AS n, AVG(importance) AS avg_importance FROM memories GROUP BY kind"
        )
        total = self.db.query_one("SELECT COUNT(*) AS n FROM memories")
        return {
            "total": total["n"] if total else 0,
            "by_kind": {
                row["kind"]: {
                    "count": row["n"],
                    "avg_importance": round(row["avg_importance"] or 0.0, 3),
                }
                for row in rows
            },
        }

    # ------------------------------------------------------------------ #
    def _all(self, kinds: Sequence[str] | None = None, limit: int = 500) -> list[Memory]:
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            rows = self.db.query(
                f"SELECT * FROM memories WHERE kind IN ({placeholders})"
                " ORDER BY updated_at DESC LIMIT ?",
                (*kinds, limit),
            )
        else:
            rows = self.db.query(
                "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
        return [self._row_to_memory(row) for row in rows]

    def _find_similar(self, content: str, kind: str):
        """用 token 集合的**包含度**做去重判断。

        不直接用 Jaccard：中文长句里多几个字就会把 Jaccard 拉得很低，
        导致「用户偏好中文回答」和「用户偏好使用中文回答」被当成两条记忆。
        包含度 = 交集 / 较短集合，更贴合「新记忆是否是已有记忆的精炼」这个判断。
        """
        target = set(tokenize(content))
        if not target:
            return None
        for row in self.db.query("SELECT * FROM memories WHERE kind = ? LIMIT 200", (kind,)):
            candidate = set(tokenize(row["content"]))
            if not candidate:
                continue
            intersection = len(target & candidate)
            if not intersection:
                continue
            containment = intersection / min(len(target), len(candidate))
            if containment >= self.dedup_threshold:
                return row
        return None

    def _upsert(self, memory: Memory) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO memories"
            " (memory_id, kind, content, importance, created_at, updated_at,"
            "  expires_at, hit_count, source, metadata)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                memory.memory_id,
                memory.kind,
                memory.content,
                memory.importance,
                memory.created_at,
                memory.updated_at,
                memory.expires_at,
                memory.hit_count,
                memory.source,
                self.db.dump_json(memory.metadata),
            ),
        )

    def _row_to_memory(self, row) -> Memory:
        return Memory(
            memory_id=row["memory_id"],
            kind=row["kind"],
            content=row["content"],
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            hit_count=row["hit_count"],
            source=row["source"],
            metadata=self.db.load_json(row["metadata"], {}),
        )


def extract_memories_from_messages(llm, messages: list[dict]) -> list[dict]:
    """让模型从对话中抽取值得长期保留的信息。

    离线模型会返回确定性结果，因此这条链路在无 Key 环境同样可测。
    任何异常都返回空列表 —— 记忆抽取失败不应该影响主任务。
    """
    import json

    if llm is None or not messages:
        return []

    prompt = [
        {
            "role": "system",
            "content": (
                "你是长期记忆提取器。从对话中提取值得跨任务保留的信息，"
                "类型只能是 preference / fact / goal / progress / workflow。"
                "不要把用户的提问当成记忆。"
                "输出 JSON：{\"memories\": [{\"type\": ..., \"content\": ..., \"importance\": 0.0-1.0}]}"
            ),
        },
        # 保留原始轮次结构，而不是拍平成一段文本：
        # 拍平之后「哪句是用户说的、哪句是模型说的」这个信息就丢了。
        *[
            {"role": str(message.get("role", "user")), "content": str(message.get("content") or "")}
            for message in messages
            if message.get("content") and message.get("role") in ("user", "assistant")
        ],
    ]

    try:
        response = llm.chat(prompt, tools=None)
        payload = json.loads(response.content or "{}")
    except Exception:  # noqa: BLE001
        return []

    items = payload.get("memories") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []

    cleaned: list[dict] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        try:
            importance = float(item.get("importance", 0.6))
        except (TypeError, ValueError):
            importance = 0.6
        cleaned.append(
            {
                "kind": str(item.get("type", "fact")),
                "content": str(item["content"])[:300],
                "importance": max(0.0, min(1.0, importance)),
            }
        )
    return cleaned
