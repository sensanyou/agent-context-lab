"""上下文工程引擎测试。

覆盖：分区预算、当前问题保护、锚点消息优先、工具链修复、策略可插拔。
"""

from __future__ import annotations

import pytest

from contextforge.config import ContextSettings
from contextforge.context import (
    BudgetAllocator,
    ContextEngine,
    ContextRequest,
    HeuristicCounter,
    check_tool_calls,
    compress_history,
    estimate_context_tokens,
    get_strategy,
    repair_tool_calls,
    split_messages,
    trim_history,
    validate_tool_calls,
)
from contextforge.context.tokens import truncate_to_tokens
from contextforge.llm import MockLLM
from contextforge.types import ScoredChunk, Chunk

# --------------------------------------------------------------------------- #
# Token 计数
# --------------------------------------------------------------------------- #
def test_heuristic_counter_handles_cjk_and_latin():
    counter = HeuristicCounter()
    assert counter.count_text("") == 0
    # 中文按字计，英文按词计，混排不应退化到「字符数 / 4」
    chinese = counter.count_text("上下文工程")
    english = counter.count_text("context engineering")
    assert 3 <= chinese <= 8
    assert 2 <= english <= 6


def test_estimate_context_tokens_counts_tool_calls():
    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "你好"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_local_time", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "2026-01-01"},
    ]
    tokens = estimate_context_tokens(messages)
    assert isinstance(tokens, int)
    assert tokens > estimate_context_tokens(messages[:2])


def test_truncate_to_tokens_reports_truncation():
    counter = HeuristicCounter()
    text = "上下文工程" * 100
    trimmed, truncated = truncate_to_tokens(text, 20, counter)
    assert truncated is True
    assert counter.count_text(trimmed) <= 30
    assert trimmed.endswith("（已按上下文预算截断）")

    same, flag = truncate_to_tokens("短文本", 100, counter)
    assert flag is False and same == "短文本"


# --------------------------------------------------------------------------- #
# 轮次切分与裁剪
# --------------------------------------------------------------------------- #
def test_split_messages_keeps_tool_chain_in_one_turn():
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "现在几点"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "10:00"},
        {"role": "assistant", "content": "现在是 10 点"},
        {"role": "user", "content": "谢谢"},
    ]
    system_messages, turns = split_messages(messages)
    assert len(system_messages) == 1
    assert len(turns) == 2
    # 工具调用与结果必须在同一轮里
    kinds = [m["role"] for m in turns[0]]
    assert kinds == ["user", "assistant", "tool", "assistant"]


def test_trim_history_keeps_system_and_recent_turns():
    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
        {"role": "user", "content": "问题2"},
        {"role": "assistant", "content": "回答2"},
    ]
    result = trim_history(messages, max_turns=1)
    assert result[0]["role"] == "system"
    assert any(m.get("content") == "问题2" for m in result)
    assert not any(m.get("content") == "问题1" for m in result)


def test_trim_history_rejects_invalid_max_turns():
    with pytest.raises(ValueError):
        trim_history([{"role": "user", "content": "x"}], max_turns=0)


# --------------------------------------------------------------------------- #
# 工具链校验
# --------------------------------------------------------------------------- #
def test_check_tool_calls_accepts_complete_chain():
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    assert validate_tool_calls(messages) is True
    assert check_tool_calls(messages).call_count == 1


def test_check_tool_calls_detects_orphan_tool_result():
    messages = [{"role": "tool", "tool_call_id": "missing", "content": "ok"}]
    report = check_tool_calls(messages)
    assert report.ok is False
    assert "没有对应的调用" in report.problems[0]


def test_check_tool_calls_detects_unfinished_call():
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
    ]
    assert validate_tool_calls(messages) is False


def test_check_tool_calls_detects_duplicate_id():
    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
        ]},
    ]
    assert validate_tool_calls(messages) is False


def test_repair_tool_calls_drops_broken_turn():
    messages = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "好的一轮"},
        {"role": "assistant", "content": "答复"},
        {"role": "user", "content": "坏的一轮"},
        {"role": "assistant", "tool_calls": [
            {"id": "c9", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
    ]
    repaired, notes = repair_tool_calls(messages)
    assert validate_tool_calls(repaired) is True
    assert notes and "工具调用链不完整" in notes[0]
    assert not any(m.get("role") == "tool" for m in repaired)


# --------------------------------------------------------------------------- #
# 压缩策略
# --------------------------------------------------------------------------- #
def _long_history(turns: int = 10) -> list[dict]:
    history: list[dict] = [
        {"role": "user", "content": "我必须用中文回答，我的长期目标是做 Agent 开发。"},
        {"role": "assistant", "content": "好的。"},
    ]
    for index in range(turns):
        history.append({"role": "user", "content": f"第 {index} 轮：" + "上下文压缩需要保留关键信息。" * 6})
        history.append({"role": "assistant", "content": f"答复 {index}：" + "预算需要分区分配。" * 6})
    return history


def test_strategy_none_keeps_everything():
    counter = HeuristicCounter()
    history = _long_history(6)
    result = get_strategy("none", counter).compress(history, keep_recent_turns=2)
    assert len(result.messages) == len(history)
    assert result.tool_integrity_preserved is True


def test_strategy_sliding_drops_early_constraints():
    counter = HeuristicCounter()
    result = get_strategy("sliding", counter).compress(_long_history(6), keep_recent_turns=2)
    text = "\n".join(str(m.get("content")) for m in result.messages)
    assert "必须用中文回答" not in text  # 滑动窗口的已知代价
    assert result.dropped_turns > 0


def test_strategy_summary_keeps_length_bounded():
    counter = HeuristicCounter()
    result = get_strategy("summary", counter).compress(_long_history(10), keep_recent_turns=2)
    assert result.summary_text
    assert result.summarized_messages > 0
    assert result.tool_integrity_preserved is True


def test_strategy_hybrid_pins_constraints():
    counter = HeuristicCounter()
    result = get_strategy("hybrid", counter).compress(_long_history(10), keep_recent_turns=2)
    assert result.pinned, "混合策略应当钉住关键约束"
    assert any("中文" in pin or "目标" in pin for pin in result.pinned)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_strategy("不存在的策略", HeuristicCounter())


def test_compress_history_backward_compatible_signature():
    result = compress_history(_long_history(8), keep_recent_turns=2)
    assert isinstance(result, list)
    assert result[0]["role"] == "system" or result[0]["role"] in ("user", "assistant")


# --------------------------------------------------------------------------- #
# 分区预算
# --------------------------------------------------------------------------- #
def test_allocation_covers_all_sections_and_reserves_output():
    settings = ContextSettings(token_budget=1000, reserved_output_tokens=200)
    allocation = settings.allocation()
    assert set(allocation) == {"system", "memory", "retrieved", "history", "tools", "current"}
    assert sum(allocation.values()) <= 800
    assert allocation["current"] > 0, "当前问题必须单独留额度"


def test_budget_allocator_truncates_within_section_limit():
    settings = ContextSettings(token_budget=400, reserved_output_tokens=0)
    allocator = BudgetAllocator(settings, HeuristicCounter())
    messages, used, notes = allocator.take_text("很长的内容" * 200, "system")
    assert used <= allocator.limit_of("system") + 20
    assert notes, "发生截断时应当给出说明"


def test_allocator_anchors_are_kept_when_dropping():
    settings = ContextSettings(token_budget=500, reserved_output_tokens=0)
    allocator = BudgetAllocator(settings, HeuristicCounter())
    messages = [{"role": "system", "content": "锚点约束"}] + [
        {"role": "user", "content": "内容" * 40} for _ in range(20)
    ]
    kept, _used, _notes = allocator.take_messages(messages, "history", anchor_count=1)
    assert kept[0]["content"] == "锚点约束", "锚点消息必须优先保留"


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
def test_engine_keeps_current_question_last():
    engine = ContextEngine(ContextSettings(token_budget=900), counter=HeuristicCounter())
    bundle = engine.build(
        ContextRequest(
            system_prompt="你是助手。必须用中文回答。",
            user_input="这是当前问题，请保留我。",
            history=_long_history(10),
            strategy="hybrid",
        )
    )
    assert bundle.messages[-1]["content"] == "这是当前问题，请保留我。"
    assert bundle.report.tool_integrity_ok is True


def test_engine_records_report_metrics():
    engine = ContextEngine(ContextSettings(token_budget=900), counter=HeuristicCounter())
    bundle = engine.build(
        ContextRequest(
            system_prompt="你是助手。",
            user_input="问题",
            history=_long_history(10),
            strategy="hybrid",
        )
    )
    report = bundle.report
    assert report.tokens_before > 0
    assert 0 < report.compression_ratio <= 1.0
    assert report.sections
    assert report.strategy == "hybrid"


def test_engine_injects_retrieved_chunks_with_citations():
    engine = ContextEngine(ContextSettings(token_budget=4000), counter=HeuristicCounter())
    chunks = [
        ScoredChunk(
            chunk=Chunk(chunk_id="c1", doc_id="d1", text="上下文压缩必须以轮次为单位。", title="手册", index=0),
            score=0.9,
            dense_score=0.7,
            sparse_score=3.2,
        )
    ]
    bundle = engine.build(
        ContextRequest(system_prompt="助手", user_input="怎么压缩？", retrieved=chunks)
    )
    assert bundle.citations and bundle.citations[0].label == "手册#0"
    joined = "\n".join(str(m.get("content")) for m in bundle.messages)
    assert "【检索到的资料" in joined
    assert "(score=0.900)" in joined


def test_engine_保护在飞消息不被压缩掉():
    """当前任务的工具链必须原样保留，否则会发出非法请求。"""
    engine = ContextEngine(
        ContextSettings(token_budget=700, reserved_output_tokens=100), counter=HeuristicCounter()
    )
    recent = [
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "结果" * 30},
    ]
    bundle = engine.build(
        ContextRequest(
            system_prompt="助手",
            user_input="",
            history=_long_history(12),
            recent=recent,
            strategy="hybrid",
        )
    )
    assert bundle.messages[-2:] == recent
    assert bundle.report.tool_integrity_ok is True


def test_engine_drops_low_score_chunks_when_out_of_budget():
    engine = ContextEngine(
        ContextSettings(token_budget=420, reserved_output_tokens=0, ratio_retrieved=0.2),
        counter=HeuristicCounter(),
    )
    chunks = [
        ScoredChunk(
            chunk=Chunk(chunk_id=f"c{i}", doc_id="d", text="片段内容" * 60, title=f"文档{i}", index=0),
            score=1.0 - i * 0.1,
        )
        for i in range(5)
    ]
    bundle = engine.build(ContextRequest(system_prompt="助手", user_input="问题", retrieved=chunks))
    assert bundle.report.dropped_chunks, "超预算时应记录被丢弃的片段"
    assert len(bundle.citations) < len(chunks)


def test_engine_lower_keep_turns_when_history_over_budget():
    engine = ContextEngine(
        ContextSettings(token_budget=600, reserved_output_tokens=0, keep_recent_turns=8),
        counter=HeuristicCounter(),
    )
    bundle = engine.build(
        ContextRequest(system_prompt="助手", user_input="问题", history=_long_history(14), strategy="hybrid")
    )
    assert bundle.report.kept_turns < 8
    assert any("保留轮数" in note for note in bundle.report.notes)


def test_engine_memory_section_rendered():
    engine = ContextEngine(ContextSettings(token_budget=2000), counter=HeuristicCounter())
    bundle = engine.build(
        ContextRequest(
            system_prompt="助手",
            user_input="问题",
            memories=[{"kind": "preference", "content": "用户偏好中文回答", "score": 0.8}],
        )
    )
    joined = "\n".join(str(m.get("content")) for m in bundle.messages)
    assert "【长期记忆" in joined
    assert "用户偏好中文回答" in joined


def test_engine_with_llm_summary_strategy_offline():
    """LLM 摘要策略在离线模型下也必须可用（由抽取式摘要兜底）。"""
    engine = ContextEngine(ContextSettings(token_budget=900), counter=HeuristicCounter(), llm=MockLLM())
    bundle = engine.build(
        ContextRequest(
            system_prompt="助手", user_input="总结", history=_long_history(10), strategy="llm-summary"
        )
    )
    assert bundle.report.strategy == "llm-summary"
    assert bundle.messages[-1]["content"] == "总结"
