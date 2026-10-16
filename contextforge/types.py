"""核心数据结构。

消息沿用 OpenAI Chat Completions 的 dict 结构（role/content/tool_calls/tool_call_id），
这样可以零成本地在「自研链路」和「真实模型 API」之间切换。
其余结构化对象使用 dataclass，便于序列化与断言。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

Message = dict[str, Any]


# --------------------------------------------------------------------------- #
# 消息工具函数
# --------------------------------------------------------------------------- #
def system(content: str) -> Message:
    return {"role": "system", "content": content}


def user(content: str) -> Message:
    return {"role": "user", "content": content}


def assistant(content: str | None = None, tool_calls: list[dict] | None = None) -> Message:
    message: Message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def tool_result(tool_call_id: str, content: str, name: str | None = None) -> Message:
    message: Message = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
    if name:
        message["name"] = name
    return message


def message_text(message: Message) -> str:
    """把一条消息拍平成纯文本，用于 token 估算与摘要。"""
    parts: list[str] = [str(message.get("role", ""))]

    content = message.get("content")
    if content:
        parts.append(str(content))

    for call in message.get("tool_calls") or []:
        parts.append(str(call.get("id", "")))
        parts.append(str(call.get("type", "")))
        function = call.get("function") or {}
        parts.append(str(function.get("name", "")))
        parts.append(str(function.get("arguments", "")))

    if message.get("role") == "tool":
        parts.append(str(message.get("tool_call_id", "")))

    return "\n".join(parts)


def is_tool_call_message(message: Message) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# 模型层对象
# --------------------------------------------------------------------------- #
@dataclass
class ToolCallRequest:
    """模型请求调用一个工具。"""

    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，保持与 API 一致

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    """一次模型调用的结果。"""

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> Message:
        return assistant(self.content or None, [c.to_openai() for c in self.tool_calls] or None)


# --------------------------------------------------------------------------- #
# 检索层对象
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """一个文档切片。"""

    chunk_id: str
    doc_id: str
    text: str
    index: int = 0
    source: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        label = self.title or self.source or self.doc_id
        return f"{label}#{self.index}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    source_stage: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
            "source_stage": self.source_stage,
        }


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    text: str
    chunk_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self, with_text: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not with_text:
            data.pop("text", None)
        return data


# --------------------------------------------------------------------------- #
# 工具层对象
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Any = None
    dangerous: bool = False
    tags: tuple[str, ...] = ()

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolExecution:
    tool_name: str
    tool_call_id: str
    arguments: dict[str, Any]
    result: str
    ok: bool
    error: str = ""
    latency_ms: float = 0.0
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 观测层对象
# --------------------------------------------------------------------------- #
@dataclass
class Span:
    span_id: str
    name: str
    kind: str = "internal"  # chain | llm | tool | retriever | node
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return (end - self.started_at) * 1000.0

    def end(self, **attributes: Any) -> "Span":
        self.ended_at = time.time()
        self.attributes.update(attributes)
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_ms"] = round(self.duration_ms, 3)
        return data


@dataclass
class ContextReport:
    """上下文构建报告 —— 让「这一轮到底给了模型什么」变成可观测、可对比的数据。"""

    strategy: str = "none"
    budget_total: int = 0
    reserved_output: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    allocation: dict[str, int] = field(default_factory=dict)
    sections: dict[str, int] = field(default_factory=dict)  # 段落 -> token
    kept_turns: int = 0
    dropped_turns: int = 0
    dropped_chunks: list[str] = field(default_factory=list)
    summarized_messages: int = 0
    tool_integrity_ok: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before <= 0:
            return 1.0
        return round(self.tokens_after / self.tokens_before, 4)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["compression_ratio"] = self.compression_ratio
        return data


@dataclass
class Citation:
    chunk_id: str
    label: str
    score: float
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentResult:
    """一次 Agent 任务的对外结果。"""

    answer: str
    task_id: str = ""
    trace_id: str = ""
    citations: list[Citation] = field(default_factory=list)
    tool_executions: list[ToolExecution] = field(default_factory=list)
    context_report: ContextReport | None = None
    usage: Usage = field(default_factory=Usage)
    iterations: int = 0
    latency_ms: float = 0.0
    status: str = "completed"  # completed | interrupted | failed | max_iterations
    error: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "citations": [c.to_dict() for c in self.citations],
            "tool_executions": [t.to_dict() for t in self.tool_executions],
            "context_report": self.context_report.to_dict() if self.context_report else None,
            "usage": self.usage.as_dict,
            "iterations": self.iterations,
            "latency_ms": round(self.latency_ms, 3),
            "status": self.status,
            "error": self.error,
            "events": self.events,
        }


def flatten_turns(messages: Iterable[list[Message]]) -> list[Message]:
    return [message for turn in messages for message in turn]
