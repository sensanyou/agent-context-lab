"""Agent 运行时：把各层组装成一个可用的应用。

它同时充当图节点的 `RuntimeContext`（提供 llm / engine / tools / memory / tracer），
以及对外应用的**唯一门面** —— CLI、HTTP 接口、评测脚本都只依赖这一个类，
所以「换掉任何一层实现」都不会波及上层。

服务装配是可降级的：没有 API Key、没有知识库、没有代码仓库时，
运行时会自动少注册对应的能力，而不是启动失败。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import Settings, get_settings
from .context import ContextEngine, get_counter
from .evaluation.tracing import TraceRecorder, TraceSession
from .graph import AgentState, CompiledGraph, SqliteCheckpointer, compile_agent_graph
from .llm import BaseLLM, build_llm
from .memory import MemoryManager
from .rag import KnowledgeBase
from .storage import Database
from .tools import RepoAnalyzer, ToolRegistry, build_registry
from .types import AgentResult, new_id


class AgentRuntime:
    """ContextForge 的运行时与应用门面。"""

    def __init__(
        self,
        settings: Settings | None = None,
        db: Database | None = None,
        llm: BaseLLM | None = None,
        knowledge: KnowledgeBase | None = None,
        repo_root: str | Path | None = None,
        enable_tests: bool = True,
        install_skills: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

        self.db = db or Database(self.settings.db_path)
        self.llm = llm or build_llm(self.settings)

        self.engine = ContextEngine(self.settings.context, counter=get_counter(), llm=self.llm)
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase(
            self.db, self.settings, llm=self.llm
        )
        self.memory = MemoryManager(self.db)
        if install_skills:
            self.memory.install_defaults()

        self.repo = RepoAnalyzer(repo_root) if repo_root else None
        self.tools: ToolRegistry = build_registry(
            knowledge=self.knowledge,
            memory=self.memory,
            repo=self.repo,
            allow_tests=enable_tests,
            max_result_chars=self.settings.agent.tool_result_max_chars,
        )

        self.trace_recorder = TraceRecorder(self.db, self.settings.trace_dir)
        self.graph: CompiledGraph = compile_agent_graph(
            self.settings, checkpointer=SqliteCheckpointer(self.db)
        )

        self.extra_instructions = ""
        self._local = threading.local()

    # ------------------------------------------------------------------ #
    @property
    def tracer(self) -> TraceSession:
        session = getattr(self._local, "tracer", None)
        if session is None:
            raise RuntimeError("当前没有活动的轨迹会话，请通过 AgentRuntime.ask 调用")
        return session

    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        conversation_id: str | None = None,
        strategy: str = "",
        retrieval_mode: str = "hybrid",
        thread_id: str | None = None,
        require_approval: bool | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentResult:
        """执行一次完整任务：规划 → 检索 → 组装上下文 → 推理 → 工具 → 反思 → 收尾。"""
        started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return AgentResult(answer="", status="failed", error="问题不能为空")

        conversation_id = self.memory.conversations.ensure(conversation_id, title=question[:40])
        task_id = new_id("task")
        thread_id = thread_id or conversation_id

        session = self.trace_recorder.start(question, task_id=task_id, thread_id=thread_id)
        self._local.tracer = session

        graph = self.graph
        if require_approval is not None and require_approval != self.settings.agent.require_approval:
            # 用副本构造临时图，绝不原地修改共享配置 ——
            # 否则一次带人工确认的调用会污染进程内其他请求的行为。
            import dataclasses

            overridden = dataclasses.replace(self.settings)
            overridden.agent = dataclasses.replace(
                self.settings.agent, require_approval=require_approval
            )
            graph = compile_agent_graph(overridden, checkpointer=SqliteCheckpointer(self.db))

        state = AgentState(
            question=question,
            conversation_id=conversation_id,
            task_id=task_id,
            thread_id=thread_id,
            trace_id=session.trace_id,
            strategy=strategy,
            retrieval_mode=retrieval_mode,
        )

        try:
            state = graph.invoke(state, self, thread_id=thread_id, on_event=on_event)
        except Exception as exc:  # noqa: BLE001 - 兜底，保证调用方拿到结构化结果
            state.status = "failed"
            state.error = f"{type(exc).__name__}: {exc}"

        latency_ms = (time.perf_counter() - started) * 1000
        result = state.to_result(latency_ms)
        result.task_id = task_id
        result.trace_id = session.trace_id

        usage = session.usage()
        if usage.total_tokens:
            result.usage = usage

        session.finish(
            status="ok" if state.status in ("completed", "interrupted") else "error",
            answer_length=len(state.answer or ""),
            iterations=state.iteration,
            tokens=result.usage.as_dict,
            context_report=state.context_report,
        )
        self._local.tracer = None
        return result

    # ------------------------------------------------------------------ #
    def resume(self, thread_id: str, on_event=None) -> AgentResult:
        """从检查点恢复一个被中断的任务（human-in-the-loop 的第二步）。"""
        started = time.perf_counter()
        checkpoint = self.graph.checkpointer.latest(thread_id)  # type: ignore[union-attr]
        if checkpoint is None:
            return AgentResult(answer="", status="failed", error=f"找不到线程 {thread_id} 的检查点")

        session = self.trace_recorder.start(
            (checkpoint.state or {}).get("question", ""),
            task_id=(checkpoint.state or {}).get("task_id", ""),
            thread_id=thread_id,
        )
        self._local.tracer = session

        try:
            state = self.graph.resume(self, thread_id, on_event=on_event)
        except Exception as exc:  # noqa: BLE001
            session.finish(status="error", error=str(exc))
            self._local.tracer = None
            return AgentResult(answer="", status="failed", error=str(exc))

        result = state.to_result((time.perf_counter() - started) * 1000)
        result.trace_id = session.trace_id
        session.finish(status="ok", resumed=True)
        self._local.tracer = None
        return result

    # ------------------------------------------------------------------ #
    # 知识库
    # ------------------------------------------------------------------ #
    def ingest_text(self, text: str, title: str = "", source: str = "") -> dict[str, Any]:
        document = self.knowledge.add_text(text, title=title, source=source)
        return document.to_dict()

    def ingest_file(self, path: str | Path) -> dict[str, Any] | None:
        document = self.knowledge.add_file(path)
        return document.to_dict() if document else None

    def ingest_directory(self, path: str | Path) -> dict[str, Any]:
        documents, skipped = self.knowledge.add_directory(path)
        return {
            "added": len(documents),
            "documents": [document.to_dict() for document in documents],
            "skipped": skipped,
        }

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict[str, Any]]:
        return [
            item.to_dict() for item in self.knowledge.search(query, top_k=top_k, mode=mode)
        ]

    def delete_document(self, doc_id: str) -> bool:
        return self.knowledge.delete_document(doc_id)

    # ------------------------------------------------------------------ #
    # 记忆
    # ------------------------------------------------------------------ #
    def remember(self, content: str, kind: str = "fact", importance: float = 0.6) -> dict[str, Any]:
        return self.memory.long_term.remember(content, kind=kind, importance=importance).to_dict()

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return [memory.to_dict() for memory in self.memory.long_term.recall(query, top_k=top_k)]

    # ------------------------------------------------------------------ #
    # 观测
    # ------------------------------------------------------------------ #
    def observability(self) -> dict[str, Any]:
        return {
            "traces": self.trace_recorder.aggregate(),
            "recent": self.trace_recorder.list_traces(limit=10),
        }

    # ------------------------------------------------------------------ #
    def status(self) -> dict[str, Any]:
        """对外暴露的系统自检信息。"""
        return {
            "version": __import__("contextforge").__version__,
            "llm": self.llm.describe(),
            "embeddings": self.knowledge.embedder.describe(),
            "knowledge": {
                "documents": self.knowledge.stats()["documents"],
                "chunks": self.knowledge.stats()["chunks"],
            },
            "tools": self.tools.names(),
            "memory": self.memory.stats(),
            "repo": str(self.repo.root) if self.repo else None,
            "workflow": self.graph.describe(),
            "context": {
                "strategy": self.settings.context.strategy,
                "token_budget": self.settings.context.token_budget,
                "allocation": self.settings.context.allocation(),
            },
        }

    def close(self) -> None:
        self.db.close()
