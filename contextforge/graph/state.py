"""Agent 状态。

状态对象是图编排的核心：它决定了「跨节点传递什么」以及「断点能恢复到什么程度」。

设计原则 —— **可序列化优先**。任务状态一旦要跨请求恢复，就不能包含不可序列化的
对象（模型客户端、数据库连接、向量矩阵）。所有服务都放在 `RuntimeContext` 里，
状态里只留纯数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..types import AgentResult, ContextReport, Message, ScoredChunk, ToolExecution, Usage


@dataclass
class AgentState:
    """一次 Agent 任务的完整状态。"""

    # 输入
    question: str = ""
    conversation_id: str = ""
    task_id: str = ""
    thread_id: str = ""
    trace_id: str = ""

    # 规划
    plan: list[str] = field(default_factory=list)
    goal: str = ""
    step_index: int = 0

    # 检索
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    need_retrieval: bool = False

    # 记忆先验
    memories: list[dict[str, Any]] = field(default_factory=list)

    # 工具
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    pending_tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # 上下文
    messages: list[Message] = field(default_factory=list)
    #: build_context 节点产出的、本轮真正发给模型的消息（含系统提示/记忆/检索块）
    built_messages: list[Message] = field(default_factory=list)
    context_report: dict[str, Any] | None = None
    memory_block: str = ""

    # 结果
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)

    # 控制流
    iteration: int = 0
    tool_call_count: int = 0
    reflected: bool = False
    retry_requested: bool = False
    observations: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    # 元信息
    usage: dict[str, int] = field(default_factory=dict)
    status: str = "running"  # running | completed | interrupted | failed | max_iterations
    error: str = ""
    next_node: str = ""
    notes: list[str] = field(default_factory=list)
    #: 本次运行使用的策略（用于对比实验，留空表示用全局配置）
    strategy: str = ""
    retrieval_mode: str = "hybrid"

    # ------------------------------------------------------------------ #
    def merge(self, update: dict[str, Any] | None) -> "AgentState":
        """把节点返回的局部更新合并进来。

        列表字段采用**追加**语义（`observations` / `tool_executions` / `events`），
        这与 LangGraph 的 reducer 思路一致：节点只负责产出增量，不负责读取已有内容，
        从而避免并发或重放时丢失信息。

        例外：`retrieved` 与 `citations` 是**覆盖**语义 —— 它们描述「当前这一轮
        检索到了什么」，如果追加，多次组装上下文会把同一批引用重复计入结果。
        """
        if not update:
            return self

        append_keys = {"observations", "tool_executions", "events", "notes"}
        for key, value in update.items():
            if not hasattr(self, key):
                continue
            if key in append_keys and isinstance(value, list):
                current = getattr(self, key)
                if isinstance(current, list):
                    setattr(self, key, current + [item for item in value])
                    continue
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "conversation_id": self.conversation_id,
            "task_id": self.task_id,
            "thread_id": self.thread_id,
            "trace_id": self.trace_id,
            "plan": self.plan,
            "goal": self.goal,
            "step_index": self.step_index,
            "retrieved": self.retrieved,
            "need_retrieval": self.need_retrieval,
            "memories": self.memories,
            "tool_executions": self.tool_executions,
            "pending_tool_calls": self.pending_tool_calls,
            "messages": self.messages,
            "built_messages": self.built_messages,
            "context_report": self.context_report,
            "memory_block": self.memory_block,
            "answer": self.answer,
            "citations": self.citations,
            "iteration": self.iteration,
            "tool_call_count": self.tool_call_count,
            "reflected": self.reflected,
            "retry_requested": self.retry_requested,
            "observations": self.observations,
            "events": self.events,
            "usage": self.usage,
            "status": self.status,
            "error": self.error,
            "next_node": self.next_node,
            "notes": self.notes,
            "strategy": self.strategy,
            "retrieval_mode": self.retrieval_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentState":
        state = cls()
        for key, value in (payload or {}).items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state

    # ------------------------------------------------------------------ #
    def add_event(self, node: str, event_type: str, **detail: Any) -> None:
        self.events.append(
            {
                "node": node,
                "type": event_type,
                "iteration": self.iteration,
                "detail": detail,
            }
        )

    def add_observation(self, text: str) -> None:
        if text:
            self.observations.append(text)

    def to_result(self, latency_ms: float = 0.0) -> AgentResult:
        from dataclasses import fields

        from ..types import Citation

        usage = Usage(
            prompt_tokens=int(self.usage.get("prompt_tokens", 0)),
            completion_tokens=int(self.usage.get("completion_tokens", 0)),
            total_tokens=int(self.usage.get("total_tokens", 0)),
        )

        report = None
        if self.context_report:
            allowed = {field.name for field in fields(ContextReport)}
            payload = {k: v for k, v in self.context_report.items() if k in allowed}
            report = ContextReport(**payload)

        return AgentResult(
            answer=self.answer,
            task_id=self.task_id,
            trace_id=self.trace_id,
            citations=[Citation(**item) for item in self.citations],
            tool_executions=[ToolExecution(**item) for item in self.tool_executions],
            context_report=report,
            usage=usage,
            iterations=self.iteration,
            latency_ms=latency_ms,
            status=self.status,
            error=self.error,
            events=self.events,
        )


def state_from_retrieved(items: list[ScoredChunk]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]


def retrieved_from_state(items: list[dict[str, Any]]) -> list[ScoredChunk]:
    from ..types import Chunk

    restored: list[ScoredChunk] = []
    for item in items or []:
        chunk_payload = item.get("chunk") or {}
        chunk = Chunk(
            chunk_id=chunk_payload.get("chunk_id", ""),
            doc_id=chunk_payload.get("doc_id", ""),
            text=chunk_payload.get("text", ""),
            index=chunk_payload.get("index", 0),
            source=chunk_payload.get("source", ""),
            title=chunk_payload.get("title", ""),
            metadata=chunk_payload.get("metadata", {}),
        )
        restored.append(
            ScoredChunk(
                chunk=chunk,
                score=item.get("score", 0.0),
                dense_score=item.get("dense_score"),
                sparse_score=item.get("sparse_score"),
                rerank_score=item.get("rerank_score"),
                source_stage=item.get("source_stage", ""),
            )
        )
    return restored
