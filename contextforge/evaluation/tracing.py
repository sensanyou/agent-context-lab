"""执行轨迹（可观测性）。

要回答「Agent 为什么这么做」，光有最终回答是不够的。轨迹记录三层信息：

* **结构**：执行了哪些节点、按什么顺序、每次循环的迭代号（`events`）
* **耗时与成本**：每个 LLM 调用与工具调用的延迟、token 用量
* **输入输出**：检索命中了什么、工具拿到了什么、上下文压缩前后的 token 数

这些数据同时服务于两件事：**调试**（定位效果差的环节）和**评测**
（对比不同策略的成本与质量）。
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

from ..storage import Database
from ..types import Span, Usage, new_id


class TraceSession:
    """一次任务的轨迹会话。"""

    def __init__(self, recorder: "TraceRecorder", question: str, task_id: str, thread_id: str) -> None:
        self.recorder = recorder
        self.trace_id = new_id("trace")
        self.question = question
        self.task_id = task_id
        self.thread_id = thread_id
        self.started_at = time.time()
        self.spans: list[Span] = []
        self.status = "running"
        self.summary: dict[str, Any] = {}
        self._stack: list[Span] = []

    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def span(self, name: str, kind: str = "internal") -> Iterator[Span]:
        """记录一个时间片。支持嵌套（用栈维护父子关系）。"""
        parent = self._stack[-1] if self._stack else None
        span = Span(
            span_id=new_id("span"),
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
        )
        self._stack.append(span)
        try:
            yield span
        finally:
            self._stack.pop()
            span.end()
            self.spans.append(span)

    def record(self, name: str, kind: str, **attributes: Any) -> Span:
        """记录一个瞬时事件（没有持续时间）。"""
        span = Span(span_id=new_id("span"), name=name, kind=kind)
        span.attributes.update(attributes)
        span.ended_at = span.started_at
        self.spans.append(span)
        return span

    # ------------------------------------------------------------------ #
    @property
    def duration_ms(self) -> float:
        return (time.time() - self.started_at) * 1000

    def usage(self) -> Usage:
        prompt = sum(int(s.attributes.get("prompt_tokens", 0)) for s in self.spans)
        completion = sum(int(s.attributes.get("completion_tokens", 0)) for s in self.spans)
        return Usage(prompt, completion, prompt + completion)

    def tool_latency_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.kind == "tool")

    def llm_latency_ms(self) -> float:
        return sum(s.duration_ms for s in self.spans if s.kind == "llm")

    def node_path(self) -> list[str]:
        """图节点的执行路径（按发生顺序）。

        不能简单按 span 的 kind 去筛：`reason` 是 llm span、`build_context` 是
        chain span、`retrieve` 是 retriever span —— 它们的 kind 各有语义，
        不能同时用来表示「这是个图节点」。所以由节点自己用
        `span.attributes["node"]` 显式标记，这里只负责按标记收集。
        """
        return [str(s.attributes["node"]) for s in self.spans if "node" in s.attributes]

    def to_dict(self) -> dict[str, Any]:
        usage = self.usage()
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "thread_id": self.thread_id,
            "question": self.question,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "usage": usage.as_dict,
            "llm_latency_ms": round(self.llm_latency_ms(), 2),
            "tool_latency_ms": round(self.tool_latency_ms(), 2),
            "nodes": self.node_path(),
            "summary": self.summary,
            "spans": [span.to_dict() for span in self.spans],
        }

    def finish(self, status: str = "ok", **summary: Any) -> dict[str, Any]:
        self.status = status
        self.summary.update(summary)
        payload = self.to_dict()
        self.recorder.persist(payload)
        return payload


class TraceRecorder:
    """轨迹记录的入口，同时负责持久化与统计聚合。"""

    def __init__(self, db: Database | None = None, trace_dir: str | Path | None = None) -> None:
        self.db = db
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def start(self, question: str, task_id: str = "", thread_id: str = "") -> TraceSession:
        return TraceSession(self, question, task_id or new_id("task"), thread_id)

    # ------------------------------------------------------------------ #
    def persist(self, payload: dict[str, Any]) -> None:
        if self.db is None:
            return
        self.db.execute(
            "INSERT OR REPLACE INTO traces"
            " (trace_id, task_id, thread_id, question, status, summary, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                payload["trace_id"],
                payload.get("task_id", ""),
                payload.get("thread_id", ""),
                payload.get("question", ""),
                payload.get("status", "ok"),
                self.db.dump_json(
                    {
                        "duration_ms": payload.get("duration_ms"),
                        "usage": payload.get("usage"),
                        "nodes": payload.get("nodes"),
                        "llm_latency_ms": payload.get("llm_latency_ms"),
                        "tool_latency_ms": payload.get("tool_latency_ms"),
                        "summary": payload.get("summary"),
                    }
                ),
                time.time(),
            ),
        )
        self.db.executemany(
            "INSERT OR REPLACE INTO spans"
            " (span_id, trace_id, parent_id, name, kind, started_at, duration_ms, attributes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    span["span_id"],
                    payload["trace_id"],
                    span.get("parent_id"),
                    span["name"],
                    span["kind"],
                    span["started_at"],
                    span["duration_ms"],
                    self.db.dump_json(span.get("attributes", {})),
                )
                for span in payload.get("spans", [])
            ],
        )

    # ------------------------------------------------------------------ #
    def get(self, trace_id: str) -> dict[str, Any] | None:
        if self.db is None:
            return None
        row = self.db.query_one("SELECT * FROM traces WHERE trace_id = ?", (trace_id,))
        if row is None:
            return None
        spans = self.db.query(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at", (trace_id,)
        )
        return {
            "trace_id": row["trace_id"],
            "task_id": row["task_id"],
            "thread_id": row["thread_id"],
            "question": row["question"],
            "status": row["status"],
            "created_at": row["created_at"],
            "summary": self.db.load_json(row["summary"], {}),
            "spans": [
                {
                    "span_id": span["span_id"],
                    "parent_id": span["parent_id"],
                    "name": span["name"],
                    "kind": span["kind"],
                    "started_at": span["started_at"],
                    "duration_ms": span["duration_ms"],
                    "attributes": self.db.load_json(span["attributes"], {}),
                }
                for span in spans
            ],
        }

    def list_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        rows = self.db.query(
            "SELECT trace_id, task_id, thread_id, question, status, summary, created_at"
            " FROM traces ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "trace_id": row["trace_id"],
                "task_id": row["task_id"],
                "thread_id": row["thread_id"],
                "question": row["question"],
                "status": row["status"],
                "created_at": row["created_at"],
                **self.db.load_json(row["summary"], {}),
            }
            for row in rows
        ]

    def aggregate(self, limit: int = 200) -> dict[str, Any]:
        """把最近若干条轨迹聚合成可展示的统计。"""
        traces = self.list_traces(limit=limit)
        if not traces:
            return {
                "traces": 0,
                "avg_duration_ms": 0.0,
                "avg_tokens": 0.0,
                "total_tokens": 0,
                "status_counts": {},
                "node_visits": {},
            }

        durations = [t.get("duration_ms") or 0.0 for t in traces]
        usage = [t.get("usage") or {} for t in traces]
        status_counts: dict[str, int] = {}
        node_visits: dict[str, int] = {}

        for trace in traces:
            status_counts[trace["status"]] = status_counts.get(trace["status"], 0) + 1
            for node in trace.get("nodes") or []:
                node_visits[node] = node_visits.get(node, 0) + 1

        total_tokens = sum(int(u.get("total_tokens", 0)) for u in usage)
        return {
            "traces": len(traces),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "avg_tokens": round(total_tokens / len(traces), 2),
            "total_tokens": total_tokens,
            "status_counts": status_counts,
            "node_visits": node_visits,
        }

    def export(self, trace_id: str, path: str | Path | None = None) -> str:
        payload = self.get(trace_id)
        if payload is None:
            raise ValueError(f"轨迹不存在：{trace_id}")

        if path is None:
            if self.trace_dir is None:
                raise ValueError("没有指定导出目录")
            path = self.trace_dir / f"{trace_id}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
