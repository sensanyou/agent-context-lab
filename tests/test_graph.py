"""图编排运行时测试：分支、循环、检查点、中断恢复、状态合并。"""

from __future__ import annotations

import pytest

from contextforge.graph import (
    END,
    AgentState,
    MemoryCheckpointer,
    SqliteCheckpointer,
    StateGraph,
    build_agent_graph,
    compile_agent_graph,
    workflow_diagram,
)
from contextforge.storage import Database


class FakeContext:
    """只提供节点所需最少能力的假运行时，用于隔离测试图本身。"""

    class _Settings:
        class agent:  # noqa: N801
            max_iterations = 3
            max_tool_calls = 5
            enable_reflection = True
            require_approval = False
            tool_result_max_chars = 1000

    settings = _Settings()
    llm = None
    tools = None
    knowledge = None
    memory = None
    engine = None
    tracer = None


# --------------------------------------------------------------------------- #
# 基础图行为
# --------------------------------------------------------------------------- #
def test_linear_graph_runs_in_order():
    order: list[str] = []

    def first(state, ctx):
        order.append("first")
        return {"answer": "a"}

    def second(state, ctx):
        order.append("second")
        return {"notes": ["done"]}

    graph = StateGraph()
    graph.add_node("first", first)
    graph.add_node("second", second)
    graph.add_edge("first", "second")
    graph.set_finish("second")

    compiled = graph.compile()
    state = compiled.invoke(AgentState(question="q"), FakeContext())

    assert order == ["first", "second"]
    assert state.answer == "a"
    assert state.notes == ["done"]


def test_conditional_branch_routes_by_state():
    def start(state, ctx):
        return {"need_retrieval": state.question == "需要检索"}

    def router(state, ctx):
        return "retrieve" if state.need_retrieval else "answer"

    def retrieve(state, ctx):
        return {"retrieved": [{"chunk": {"chunk_id": "c"}, "score": 1.0}], "need_retrieval": False}

    def answer(state, ctx):
        return {"answer": "直接回答", "status": "completed"}

    graph = StateGraph()
    graph.add_node("start", start)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.set_entry("start")
    graph.add_conditional_edges("start", router, {"retrieve": "retrieve", "answer": "answer"})
    graph.add_edge("retrieve", "answer")
    graph.set_finish("answer")

    compiled = graph.compile()

    with_retrieval = compiled.invoke(AgentState(question="需要检索"), FakeContext())
    assert len(with_retrieval.retrieved) == 1
    assert with_retrieval.status == "completed"

    without = compiled.invoke(AgentState(question="不需要"), FakeContext())
    assert without.retrieved == []
    assert without.answer == "直接回答"


def test_loop_terminates_via_router():
    def counter(state, ctx):
        return {"iteration": state.iteration + 1}

    def router(state, ctx):
        return "counter" if state.iteration < 3 else END

    graph = StateGraph()
    graph.add_node("counter", counter)
    graph.add_conditional_edges("counter", router, {"counter": "counter"})
    compiled = graph.compile()

    state = compiled.invoke(AgentState(question="q"), FakeContext())
    assert state.iteration == 3


def test_max_steps_guard_prevents_infinite_loop():
    def spin(state, ctx):
        return {"iteration": state.iteration + 1}

    graph = StateGraph(max_steps=5)
    graph.add_node("spin", spin)
    graph.add_edge("spin", "spin")
    compiled = graph.compile()

    state = compiled.invoke(AgentState(question="q"), FakeContext())
    assert state.status == "max_iterations"
    assert "最大步数" in state.error


def test_node_exception_is_captured_not_raised():
    def broken(state, ctx):
        raise ValueError("节点内部错误")

    graph = StateGraph()
    graph.add_node("broken", broken)
    compiled = graph.compile()

    state = compiled.invoke(AgentState(question="q"), FakeContext())
    assert state.status == "failed"
    assert "节点内部错误" in state.error


def test_router_target_must_exist():
    def router(state, ctx):
        return "x"

    graph = StateGraph()
    graph.add_node("a", lambda s, c: None)
    graph.add_conditional_edges("a", router, {"x": "不存在的节点"})
    with pytest.raises(ValueError):
        graph.compile()


def test_router_unmapped_decision_raises_at_runtime():
    graph = StateGraph()
    graph.add_node("a", lambda s, c: None)
    graph.add_conditional_edges("a", lambda s, c: "意料之外", {"其他": "a"})
    compiled = graph.compile()
    with pytest.raises(ValueError):
        compiled.invoke(AgentState(question="q"), FakeContext())


def test_compile_requires_entry_and_nodes():
    with pytest.raises(ValueError):
        StateGraph().compile()

    graph = StateGraph()
    graph.add_node("a", lambda s, c: None)
    graph.add_edge("a", "不存在")
    with pytest.raises(ValueError):
        graph.compile()


# --------------------------------------------------------------------------- #
# 状态合并
# --------------------------------------------------------------------------- #
def test_state_merge_appends_list_fields():
    state = AgentState()
    state.merge({"observations": ["a"], "tool_executions": [{"tool_name": "t"}]})
    state.merge({"observations": ["b"], "tool_executions": [{"tool_name": "u"}]})
    assert state.observations == ["a", "b"]
    assert len(state.tool_executions) == 2


def test_state_merge_replaces_citations_to_avoid_duplicates():
    state = AgentState()
    citation = {"chunk_id": "c1", "label": "文档#0", "score": 0.9, "snippet": "x"}
    state.merge({"citations": [citation]})
    state.merge({"citations": [citation]})
    assert len(state.citations) == 1, "引用是「本轮检索到了什么」，不应累加"


def test_state_merge_ignores_unknown_keys():
    state = AgentState()
    state.merge({"不存在的字段": 1, "answer": "ok"})
    assert state.answer == "ok"
    assert not hasattr(state, "不存在的字段")


def test_state_serialization_roundtrip():
    state = AgentState(question="问题", plan=["a"], observations=["o1"], iteration=2)
    restored = AgentState.from_dict(state.to_dict())
    assert restored.question == "问题"
    assert restored.plan == ["a"]
    assert restored.iteration == 2


def test_retrieved_serialization_roundtrip():
    from contextforge.graph.state import retrieved_from_state, state_from_retrieved
    from contextforge.types import Chunk, ScoredChunk

    items = [
        ScoredChunk(
            chunk=Chunk(chunk_id="c1", doc_id="d1", text="内容", index=0, title="标题"),
            score=0.5,
            dense_score=0.4,
        )
    ]
    restored = retrieved_from_state(state_from_retrieved(items))
    assert restored[0].chunk.chunk_id == "c1"
    assert restored[0].score == 0.5


# --------------------------------------------------------------------------- #
# 检查点与恢复
# --------------------------------------------------------------------------- #
def test_checkpointer_records_history(db: Database):
    checkpointer = SqliteCheckpointer(db)
    checkpointer.save("t1", "task1", "node_a", 0, {"answer": "a"})
    checkpointer.save("t1", "task1", "node_b", 1, {"answer": "b"})

    latest = checkpointer.latest("t1")
    assert latest is not None
    assert latest.node == "node_b"
    assert latest.state["answer"] == "b"
    assert len(checkpointer.history("t1")) == 2

    checkpointer.clear("t1")
    assert checkpointer.latest("t1") is None


def test_interrupt_before_pauses_and_resume_continues():
    executed: list[str] = []

    def prepare(state, ctx):
        executed.append("prepare")
        return {"answer": "草稿"}

    def dangerous(state, ctx):
        executed.append("dangerous")
        return {"answer": "已完成", "status": "completed"}

    graph = StateGraph()
    graph.add_node("prepare", prepare)
    graph.add_node("dangerous", dangerous)
    graph.add_edge("prepare", "dangerous")
    graph.set_finish("dangerous")
    graph.interrupt_before(["dangerous"])

    checkpointer = MemoryCheckpointer()
    compiled = graph.compile(checkpointer=checkpointer)

    paused = compiled.invoke(AgentState(question="q", thread_id="thread-1"), FakeContext(), thread_id="thread-1")
    assert paused.status == "interrupted"
    assert paused.next_node == "dangerous"
    assert executed == ["prepare"]

    resumed = compiled.resume(FakeContext(), "thread-1")
    assert resumed.status == "completed"
    assert resumed.answer == "已完成"
    assert executed == ["prepare", "dangerous"]


def test_resume_without_checkpoint_raises():
    graph = StateGraph()
    graph.add_node("a", lambda s, c: None)
    compiled = graph.compile(checkpointer=MemoryCheckpointer())
    with pytest.raises(ValueError):
        compiled.resume(FakeContext(), "不存在的线程")


def test_resume_without_checkpointer_raises():
    graph = StateGraph()
    graph.add_node("a", lambda s, c: None)
    compiled = graph.compile()
    with pytest.raises(RuntimeError):
        compiled.resume(FakeContext(), "t")


# --------------------------------------------------------------------------- #
# 工作流定义
# --------------------------------------------------------------------------- #
def test_agent_graph_structure():
    graph = build_agent_graph()
    compiled = graph.compile()
    description = compiled.describe()
    assert set(description["nodes"]) == {
        "prepare", "retrieve", "build_context", "reason", "tools", "reflect", "finalize"
    }
    assert description["entry"] == "prepare"
    assert "reason" in description["conditional"]


def test_agent_graph_has_loop_and_retry_paths():
    compiled = compile_agent_graph()
    edges = compiled.describe()["conditional"]
    assert edges["tools"]["build_context"] == "build_context", "工具执行后必须回到上下文组装（循环）"
    assert edges["reflect"]["build_context"] == "build_context", "证据不足时要能重试检索"
    assert edges["reason"]["tools"] == "tools"


def test_require_approval_adds_interrupt_before_tools(settings):
    import dataclasses

    settings.agent = dataclasses.replace(settings.agent, require_approval=True)
    compiled = compile_agent_graph(settings)
    assert compiled.interrupt_before == {"tools"}


def test_workflow_diagram_is_mermaid():
    diagram = workflow_diagram()
    assert diagram.startswith("flowchart TD")
    assert "build_context" in diagram
    assert "|tools|" in diagram


def test_events_are_recorded_per_node():
    graph = StateGraph()
    graph.add_node("a", lambda s, c: {"notes": ["1"]})
    graph.add_edge("a", "b")
    graph.add_node("b", lambda s, c: {"status": "completed"})
    compiled = graph.compile()

    state = compiled.invoke(AgentState(question="q"), FakeContext())
    starts = [event["node"] for event in state.events if event["type"] == "node_start"]
    ends = [event["node"] for event in state.events if event["type"] == "node_end"]
    assert starts == ["a", "b"]
    assert ends == ["a", "b"]


def test_on_event_callback_receives_stream():
    collected: list[dict] = []

    graph = StateGraph()
    graph.add_node("a", lambda s, c: {"status": "completed"})
    compiled = graph.compile()

    compiled.invoke(AgentState(question="q"), FakeContext(), on_event=collected.append)
    assert any(event["type"] == "node_start" for event in collected)
    assert any(event["type"] == "node_end" for event in collected)
