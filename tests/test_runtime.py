"""运行时端到端测试：完整 Agent 链路在离线状态下必须可跑通。"""

from __future__ import annotations

from contextforge.llm import MockLLM, build_llm
from contextforge.runtime import AgentRuntime


# --------------------------------------------------------------------------- #
# 模型工厂
# --------------------------------------------------------------------------- #
def test_build_llm_falls_back_to_mock_without_api_key(settings):
    llm = build_llm(settings)
    assert isinstance(llm, MockLLM)
    assert llm.online is False


def test_build_llm_force_mock(settings):
    assert isinstance(build_llm(settings, force_mock=True), MockLLM)


# --------------------------------------------------------------------------- #
# 知识问答
# --------------------------------------------------------------------------- #
def test_ask_answers_from_knowledge_base_with_citations(runtime: AgentRuntime):
    result = runtime.ask("上下文压缩为什么要以轮次为单位？")
    assert result.status == "completed"
    assert result.answer
    assert result.citations, "基于知识库回答时应当给出引用"
    assert any("上下文工程指南" in citation.label for citation in result.citations)


def test_ask_records_context_report(runtime: AgentRuntime):
    result = runtime.ask("混合检索为什么要把向量和 BM25 融合？")
    assert result.context_report is not None
    report = result.context_report
    assert report.tokens_after > 0
    assert 0 < report.compression_ratio <= 1.0
    assert report.tool_integrity_ok is True
    assert report.sections


def test_ask_runs_expected_nodes(runtime: AgentRuntime):
    result = runtime.ask("RRF 融合相比加权融合有什么优势？")
    nodes = [event["node"] for event in result.events if event["type"] == "node_end"]
    assert nodes[0] == "prepare"
    assert "retrieve" in nodes
    assert "build_context" in nodes
    assert "reason" in nodes
    assert nodes[-1] == "finalize"


def test_ask_reports_usage_and_iterations(runtime: AgentRuntime):
    result = runtime.ask("长期记忆和会话记忆有什么区别？")
    assert result.usage.total_tokens > 0
    assert result.iterations >= 1
    assert result.latency_ms >= 0


# --------------------------------------------------------------------------- #
# 工具调用
# --------------------------------------------------------------------------- #
def test_ask_invokes_calculator_tool(runtime: AgentRuntime):
    result = runtime.ask("计算 (128 + 72) * 3 等于多少？")
    assert result.status == "completed"
    assert [execution.tool_name for execution in result.tool_executions] == ["calculator"]
    assert result.tool_executions[0].ok is True
    assert "600" in result.tool_executions[0].result
    assert "600" in result.answer


def test_ask_invokes_repo_tools(runtime: AgentRuntime):
    result = runtime.ask("这个示例项目的目录结构和主要模块是什么？")
    assert result.tool_executions
    assert any(execution.tool_name.startswith("repo_") for execution in result.tool_executions)
    assert "demo_pkg" in result.answer


def test_tool_loop_rebuilds_context_each_round(runtime: AgentRuntime):
    """工具调用后必须重新组装上下文（循环回到 build_context）。"""
    result = runtime.ask("现在几点了？")
    nodes = [event["node"] for event in result.events if event["type"] == "node_end"]
    assert nodes.count("build_context") >= 2, "工具执行后应当再次组装上下文"
    assert nodes.count("reason") >= 2


def test_tool_failure_does_not_break_task(runtime: AgentRuntime):
    """工具失败时任务仍应完成，并把失败原因如实带出来。"""
    execution = runtime.tools.execute("calculator", {"expression": "1 / 0"})
    assert execution.ok is False
    assert "无法计算" in execution.result


# --------------------------------------------------------------------------- #
# 降级与边界
# --------------------------------------------------------------------------- #
def test_ask_without_knowledge_still_answers(bare_runtime: AgentRuntime):
    result = bare_runtime.ask("随便问一个没有资料支撑的问题，请解释一下你的工作方式。")
    assert result.status == "completed"
    assert result.answer


def test_ask_with_empty_question_fails_gracefully(runtime: AgentRuntime):
    result = runtime.ask("   ")
    assert result.status == "failed"
    assert result.error


def test_ask_survives_missing_retrieval_hit(runtime: AgentRuntime):
    result = runtime.ask("请解释量子引力与弦理论的统一框架的完整数学推导。")
    assert result.status == "completed"
    assert result.answer is not None


# --------------------------------------------------------------------------- #
# 会话与记忆
# --------------------------------------------------------------------------- #
def test_conversation_is_persisted(runtime: AgentRuntime):
    runtime.ask("第一个问题")
    conversations = runtime.memory.conversations.list()
    assert conversations
    assert conversations[0]["messages"] > 0


def test_multi_turn_conversation_keeps_history(runtime: AgentRuntime):
    first = runtime.ask("上下文压缩有哪几种策略？")
    conversation_id = runtime.memory.conversations.list()[0]["conversation_id"]
    before = runtime.memory.conversations.count(conversation_id)

    second = runtime.ask("那以轮次为单位是什么意思？", conversation_id=conversation_id)
    assert second.status == "completed"
    assert runtime.memory.conversations.count(conversation_id) > before, "多轮对话应当累积历史"
    assert first.status == "completed"


def test_runtime_remember_and_recall(runtime: AgentRuntime):
    runtime.remember("用户偏好中文回答", kind="preference", importance=0.9)
    recalled = runtime.recall("偏好什么语言", top_k=3)
    assert recalled and any("中文" in item["content"] for item in recalled)


def test_runtime_ingest_and_search(runtime: AgentRuntime):
    runtime.ingest_text("# 新增文档\n\n这段内容专门用于验证运行时导入接口是否工作。", title="运行时导入")
    hits = runtime.search("验证运行时导入接口", top_k=3)
    assert hits
    assert any(item["chunk"]["title"] == "运行时导入" for item in hits)


def test_runtime_delete_document(runtime: AgentRuntime):
    document = runtime.ingest_text("这段内容稍后会被删除，用于测试删除接口。", title="待删除")
    assert runtime.delete_document(document["doc_id"]) is True
    assert runtime.knowledge.get_document(document["doc_id"]) is None


# --------------------------------------------------------------------------- #
# 可观测性与状态
# --------------------------------------------------------------------------- #
def test_traces_are_recorded(runtime: AgentRuntime):
    result = runtime.ask("上下文压缩有哪些策略？")
    trace = runtime.trace_recorder.get(result.trace_id)
    assert trace is not None
    assert trace["question"]
    names = [span["name"] for span in trace["spans"]]
    assert "reason" in names
    assert any(name.startswith("tool:") for name in names) or "retrieve" in names


def test_trace_aggregate(runtime: AgentRuntime):
    runtime.ask("第一个问题")
    runtime.ask("第二个问题")
    aggregate = runtime.trace_recorder.aggregate()
    assert aggregate["traces"] >= 2
    assert aggregate["avg_duration_ms"] >= 0
    assert aggregate["node_visits"].get("prepare", 0) >= 2


def test_trace_export(runtime: AgentRuntime):
    result = runtime.ask("导出轨迹测试")
    path = runtime.trace_recorder.export(result.trace_id)
    from pathlib import Path

    assert Path(path).exists()
    assert "trace_id" in Path(path).read_text(encoding="utf-8")


def test_status_reports_all_layers(runtime: AgentRuntime):
    status = runtime.status()
    assert status["llm"]["online"] is False
    assert status["embeddings"]["dim"] > 0
    assert status["knowledge"]["documents"] >= 1
    assert status["tools"]
    assert status["workflow"]["nodes"]
    assert set(status["context"]["allocation"])


# --------------------------------------------------------------------------- #
# 人工确认（human-in-the-loop）
# --------------------------------------------------------------------------- #
def test_require_approval_interrupts_before_tools(runtime: AgentRuntime):
    result = runtime.ask("计算 1 + 1 等于多少？", require_approval=True)
    assert result.status == "interrupted"
    assert result.iterations >= 1


def test_resume_after_approval(runtime: AgentRuntime):
    interrupted = runtime.ask("计算 2 + 2 等于多少？", require_approval=True)
    assert interrupted.status == "interrupted"

    conversation_id = runtime.memory.conversations.list()[0]["conversation_id"]
    resumed = runtime.resume(conversation_id)
    assert resumed.status == "completed"
    assert resumed.answer


# --------------------------------------------------------------------------- #
# 策略覆盖
# --------------------------------------------------------------------------- #
def test_strategy_override_is_applied(runtime: AgentRuntime):
    for strategy in ("none", "sliding", "summary", "hybrid"):
        result = runtime.ask("上下文压缩有哪些策略？", strategy=strategy)
        assert result.status == "completed"
        assert result.context_report.strategy == strategy


def test_retrieval_mode_override_is_applied(runtime: AgentRuntime):
    for mode in ("dense", "sparse", "hybrid"):
        result = runtime.ask("上下文压缩有哪些策略？", retrieval_mode=mode)
        assert result.status == "completed"
