"""SQLite 持久化层（被 RAG 与记忆层共用）。

为什么用 SQLite 而不是直接上 PostgreSQL：

* 项目定位是个人助手，单机单用户，SQLite 零运维、零网络、事务完整；
* 「任务状态、会话消息、文档向量、长期记忆、执行轨迹」都是**结构化小数据**，
  关系库比向量库更适合承载元数据与状态；
* 需要服务化时，把 `Database` 换掉即可，上层接口不变。

向量本身以 float32 二进制 BLOB 存储，避免 JSON 序列化带来的体积与精度损失。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    source       TEXT NOT NULL DEFAULT '',
    media_type   TEXT NOT NULL DEFAULT 'text/plain',
    text         TEXT NOT NULL DEFAULT '',
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL,
    ord        INTEGER NOT NULL DEFAULT 0,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT '',
    title      TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    dim        INTEGER NOT NULL DEFAULT 0,
    vector     BLOB,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    ord             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    tool_calls      TEXT,
    tool_call_id    TEXT,
    name            TEXT,
    created_at      REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, ord);

CREATE TABLE IF NOT EXISTS memories (
    memory_id   TEXT PRIMARY KEY,
    kind        TEXT NOT NULL DEFAULT 'fact',
    content     TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 0.5,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    expires_at  REAL,
    hit_count   INTEGER NOT NULL DEFAULT 0,
    source      TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);

CREATE TABLE IF NOT EXISTS skills (
    skill_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    steps       TEXT NOT NULL DEFAULT '[]',
    tools       TEXT NOT NULL DEFAULT '[]',
    use_count   INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    step        TEXT NOT NULL DEFAULT '',
    state       TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_thread ON tasks(thread_id);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    thread_id     TEXT NOT NULL,
    task_id       TEXT NOT NULL DEFAULT '',
    node          TEXT NOT NULL DEFAULT '',
    iteration     INTEGER NOT NULL DEFAULT 0,
    state         TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ckpt_thread ON checkpoints(thread_id, created_at);

CREATE TABLE IF NOT EXISTS traces (
    trace_id    TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL DEFAULT '',
    thread_id   TEXT NOT NULL DEFAULT '',
    question    TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'ok',
    summary     TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
    span_id     TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'internal',
    started_at  REAL NOT NULL,
    duration_ms REAL NOT NULL DEFAULT 0,
    attributes  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
"""


class Database:
    """极简 SQLite 封装：线程安全、自动建表、上下文事务。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.init_schema()

    # ------------------------------------------------------------------ #
    def init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def executemany(self, sql: str, params: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._conn.executemany(sql, params)
            self._conn.commit()

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def dump_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def load_json(value: str | None, default: Any) -> Any:
        if not value:
            return default
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default


def pack_vector(vector: Sequence[float]) -> bytes:
    """float32 打包，比 JSON 省约 60% 空间。"""
    import array

    return array.array("f", [float(x) for x in vector]).tobytes()


def unpack_vector(blob: bytes | None, dim: int = 0) -> list[float]:
    import array

    if not blob:
        return []
    values = array.array("f")
    values.frombytes(blob)
    if dim and len(values) != dim:
        return list(values[:dim])
    return list(values)
