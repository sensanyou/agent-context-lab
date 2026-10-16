"""Agent 工作流定义。

    prepare ──┬─> retrieve ──> build_context ──> reason ──┬─> tools ──> build_context（循环）
              └──────────────────────────────────────────  ├─> reflect ─┬─> build_context（重试）
                                                            │            └─> finalize
                                                            └─> finalize

三条控制流各解决一个真实问题：

* **循环**（tools → build_context → reason）：模型可能需要连续调用多个工具，
  且每次调用后都要**重新组装上下文**（工具输出可能已经超预算）。
* **条件分支**（reason → tools / reflect / finalize）：由模型自己的输出来决定走向，
  而不是由代码写死流程。
* **重试**（reflect → build_context）：证据不足时自动再检索一次，
  而不是把「我不知道」直接抛给用户。
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .engine import END, CompiledGraph, Checkpointer, StateGraph
from .nodes import (
    build_context,
    finalize,
    prepare,
    reason,
    reflect,
    retrieve,
    route_after_prepare,
    route_after_reason,
    route_after_reflect,
    route_after_tools,
    tools,
)

NODE_ORDER = ("prepare", "retrieve", "build_context", "reason", "tools", "reflect", "finalize")


def build_agent_graph(settings: Settings | None = None) -> StateGraph:
    """构建 Agent 工作流（未编译）。"""
    settings = settings or get_settings()

    graph = StateGraph(max_steps=max(settings.agent.max_iterations * 4 + 8, 16))
    graph.add_node("prepare", prepare)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_context", build_context)
    graph.add_node("reason", reason)
    graph.add_node("tools", tools)
    graph.add_node("reflect", reflect)
    graph.add_node("finalize", finalize)

    graph.set_entry("prepare")

    graph.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {"retrieve": "retrieve", "build_context": "build_context"},
    )
    graph.add_edge("retrieve", "build_context")
    graph.add_edge("build_context", "reason")
    graph.add_conditional_edges(
        "reason",
        route_after_reason,
        {"tools": "tools", "reflect": "reflect", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools",
        route_after_tools,
        {"build_context": "build_context", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"build_context": "build_context", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)

    if settings.agent.require_approval:
        # human-in-the-loop：执行工具前暂停，等待人工确认
        graph.interrupt_before(["tools"])

    return graph


def compile_agent_graph(
    settings: Settings | None = None,
    checkpointer: Checkpointer | None = None,
) -> CompiledGraph:
    return build_agent_graph(settings).compile(checkpointer=checkpointer)


def workflow_diagram(settings: Settings | None = None) -> str:
    """导出 Mermaid 图，可直接贴进 README。"""
    return compile_agent_graph(settings).to_mermaid()
