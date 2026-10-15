
from agent.context import (
    estimate_context_tokens,
    trim_history,
    validate_tool_calls,
)
from agent.compression import compress_history


def test_estimate_context_tokens():
    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "你好"},
    ]

    tokens = estimate_context_tokens(messages)

    assert isinstance(tokens, int)
    assert tokens > 0

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


def test_compress_history_allows_zero_recent_turns():
    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "我的长期目标是 Agent 开发"},
        {"role": "assistant", "content": "需要持续练习"},
    ]

    result = compress_history(messages, keep_recent_turns=0)

    assert result[0]["role"] == "system"
    assert len(result) == 2
    assert "摘要" in result[1]["content"]


def test_validate_tool_calls_accepts_complete_call():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "测试结果",
        },
    ]

    assert validate_tool_calls(messages) is True


from agent.core import Agent


def test_fit_context_budget_keeps_messages_within_budget():
    agent = Agent()
    agent.CONTEXT_TOKEN_BUDGET = 100

    messages = [
        {"role": "system", "content": "系统规则"},
        {"role": "user", "content": "你好"},
    ]

    result = agent.fit_context_budget(messages)

    from agent.context import estimate_context_tokens

    assert estimate_context_tokens(result) <= agent.CONTEXT_TOKEN_BUDGET
    assert result[0]["role"] == "system"
    assert result[-1]["content"] == "你好"

def test_fit_context_budget_compresses_history():
    from agent.context import estimate_context_tokens

    agent = Agent()
    agent.CONTEXT_TOKEN_BUDGET = 300

    messages = [
        {"role": "system", "content": "系统规则"},
    ]

    # 构造多轮较长的历史对话
    for i in range(8):
        messages.append({
            "role": "user",
            "content": f"旧问题{i}：" + "学习 Agent 上下文管理。" * 10,
        })
        messages.append({
            "role": "assistant",
            "content": f"旧回答{i}：" + "我们逐步学习上下文压缩。" * 10,
        })

    messages.append({
        "role": "user",
        "content": "这是当前问题，请保留我这个问题。",
    })

    result = agent.fit_context_budget(messages)

    assert estimate_context_tokens(result) <= agent.CONTEXT_TOKEN_BUDGET
    assert result[-1]["content"] == "这是当前问题，请保留我这个问题。"
    assert result[0]["role"] == "system"