"""接口处理器。

业务逻辑只写一遍，FastAPI 与标准库开发服务器共用这一层 ——
避免「两套 HTTP 封装各写一遍接口」导致的接口行为漂移。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..evaluation import (
    default_cases,
    render_agent_report,
    render_context_report,
    run_agent_suite,
    run_context_experiment,
)
from ..runtime import AgentRuntime


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _require(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value in (None, ""):
        raise ApiError(f"缺少参数：{key}")
    return value


class Handlers:
    """把所有接口实现收敛到一个类里，便于测试（可以直接调用方法）。"""

    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime

    # ------------------------------------------------------------------ #
    def health(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "version": __import__("contextforge").__version__}

    def status(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.status()

    def workflow(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "graph": self.runtime.graph.describe(),
            "mermaid": self.runtime.graph.to_mermaid(),
        }

    def tools(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"tools": self.runtime.tools.describe()}

    # ------------------------------------------------------------------ #
    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = _require(payload, "question")
        result = self.runtime.ask(
            str(question),
            conversation_id=payload.get("conversation_id"),
            strategy=str(payload.get("strategy") or ""),
            retrieval_mode=str(payload.get("retrieval_mode") or "hybrid"),
        )
        return result.to_dict()

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = _require(payload, "query")
        top_k = int(payload.get("top_k") or 5)
        mode = str(payload.get("mode") or "hybrid")
        hits = self.runtime.search(str(query), top_k=top_k, mode=mode)
        trace = self.runtime.knowledge.retriever.last_trace
        return {
            "query": query,
            "mode": mode,
            "hits": hits,
            "trace": trace.to_dict() if trace else None,
        }

    # ------------------------------------------------------------------ #
    def list_documents(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "documents": [document.to_dict() for document in self.runtime.knowledge.list_documents()],
            "stats": self.runtime.knowledge.stats(),
        }

    def add_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("path"):
            path = Path(str(payload["path"]))
            if path.is_dir():
                result = self.runtime.ingest_directory(path)
                return {"type": "directory", **result}
            document = self.runtime.ingest_file(path)
            if document is None:
                raise ApiError(f"无法解析文件：{path}", status=422)
            return {"type": "file", "document": document}

        text = _require(payload, "text")
        document = self.runtime.ingest_text(
            str(text), title=str(payload.get("title") or ""), source=str(payload.get("source") or "")
        )
        return {"type": "text", "document": document}

    def delete_document(self, payload: dict[str, Any]) -> dict[str, Any]:
        doc_id = _require(payload, "doc_id")
        return {"deleted": self.runtime.delete_document(str(doc_id))}

    # ------------------------------------------------------------------ #
    def list_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        kind = payload.get("kind")
        memories = self.runtime.memory.long_term.all(
            kinds=[str(kind)] if kind else None, limit=int(payload.get("limit") or 50)
        )
        return {
            "memories": [memory.to_dict() for memory in memories],
            "stats": self.runtime.memory.stats(),
            "skills": [skill.to_dict() for skill in self.runtime.memory.skills.all()],
        }

    def add_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        content = _require(payload, "content")
        return self.runtime.remember(
            str(content),
            kind=str(payload.get("kind") or "fact"),
            importance=float(payload.get("importance") or 0.6),
        )

    def forget_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_id = _require(payload, "memory_id")
        return {"forgotten": self.runtime.memory.long_term.forget(str(memory_id))}

    # ------------------------------------------------------------------ #
    def list_traces(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = int(payload.get("limit") or 20)
        return {
            "traces": self.runtime.trace_recorder.list_traces(limit=limit),
            "aggregate": self.runtime.trace_recorder.aggregate(),
        }

    def get_trace(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = _require(payload, "trace_id")
        trace = self.runtime.trace_recorder.get(str(trace_id))
        if trace is None:
            raise ApiError(f"轨迹不存在：{trace_id}", status=404)
        return trace

    def metrics(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "traces": self.runtime.trace_recorder.aggregate(),
            "context": self.runtime.settings.context.allocation(),
            "knowledge": self.runtime.knowledge.stats(),
            "memory": self.runtime.memory.stats(),
            "tools": len(self.runtime.tools.names()),
        }

    # ------------------------------------------------------------------ #
    def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        modes = payload.get("modes") or ["dense", "sparse", "hybrid"]
        strategies = payload.get("strategies") or ["none", "sliding", "summary", "hybrid"]
        cases = default_cases()

        context_report = run_context_experiment(
            self.runtime, strategies=[str(s) for s in strategies]
        )
        agent_report = run_agent_suite(
            self.runtime,
            cases=cases,
            retrieval_modes=[str(m) for m in modes],
        )

        return {
            "context_experiment": context_report,
            "agent_experiment": {
                "cases": agent_report["cases"],
                "groups": agent_report["groups"],
                "generated_at": agent_report["generated_at"],
            },
            "markdown": {
                "context": render_context_report(context_report),
                "agent": render_agent_report(agent_report),
            },
        }


def build_routes(handlers: Handlers) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
    """路由表：路径 -> 处理函数。"""
    return {
        "/api/health": handlers.health,
        "/api/status": handlers.status,
        "/api/workflow": handlers.workflow,
        "/api/tools": handlers.tools,
        "/api/chat": handlers.chat,
        "/api/search": handlers.search,
        "/api/documents": handlers.list_documents,
        "/api/documents/add": handlers.add_document,
        "/api/documents/delete": handlers.delete_document,
        "/api/memory": handlers.list_memory,
        "/api/memory/add": handlers.add_memory,
        "/api/memory/forget": handlers.forget_memory,
        "/api/traces": handlers.list_traces,
        "/api/traces/get": handlers.get_trace,
        "/api/metrics": handlers.metrics,
        "/api/evaluate": handlers.evaluate,
    }
