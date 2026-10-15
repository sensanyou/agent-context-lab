"""Agent 对话上下文管理。

第一版先提供安全的历史裁剪：保留 system 提示词，并按完整对话轮次裁剪。
工具调用及其 tool 返回消息会作为同一轮的一部分保留。
"""

from typing import Any
from transformers import AutoTokenizer

# 使用 DeepSeek V4 Flash 的 Tokenizer
# cspell:ignore deepseek
_TOKENIZER = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-V4-Flash"
)



Message = dict[str, Any]


def _is_tool_call_message(message: Message) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def _split_into_turns(messages: list[Message]) -> tuple[list[Message], list[list[Message]]]:
    """把 system 消息与后续对话分开，并将对话划分为轮次。

    一轮从 user 消息开始，直到 assistant 最终回答结束。
    assistant tool_calls 和其对应 tool 消息都留在同一轮内。
    """
    system_messages = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    turns: list[list[Message]] = []
    current_turn: list[Message] = []

    for message in non_system:
        if message.get("role") == "user" and current_turn:
            turns.append(current_turn)
            current_turn = []

        current_turn.append(message)

        # 只有不带 tool_calls 的 assistant 消息才表示本轮正常结束。
        if message.get("role") == "assistant" and not message.get("tool_calls"):
            turns.append(current_turn)
            current_turn = []

    if current_turn:
        turns.append(current_turn)

    return system_messages, turns


def trim_history(messages: list[Message], max_turns: int = 6) -> list[Message]:
    """保留 system 提示词和最近 max_turns 个完整对话轮次。"""
    if max_turns < 1:
        raise ValueError("max_turns 必须至少为 1")

    system_messages, turns = _split_into_turns(messages)
    kept_turns = turns[-max_turns:]
    return system_messages + [message for turn in kept_turns for message in turn]



def estimate_context_tokens(messages: list[Message]) -> int:
    """粗略估算消息内容的 Token 数量，不等于 API 实际用量。"""
    text_parts = []

    for message in messages:
        # 统计角色
        text_parts.append(str(message.get("role", "")))

        # 统计消息正文
        content = message.get("content")
        if content is not None:
            text_parts.append(str(content))

        # 统计工具调用
        for tool_call in message.get("tool_calls") or []:
            text_parts.append(str(tool_call.get("id", "")))
            text_parts.append(str(tool_call.get("type", "")))

            function = tool_call.get("function") or {}
            text_parts.append(str(function.get("name", "")))
            text_parts.append(str(function.get("arguments", "")))

        # 统计工具调用结果关联 ID
        if message.get("role") == "tool":
            text_parts.append(str(message.get("tool_call_id", "")))

    text = "\n".join(text_parts)

    return len(
        _TOKENIZER.encode(text, add_special_tokens=False)
    )

def validate_tool_calls(messages: list[Message]) -> bool:
    """检查工具调用和工具结果是否完整匹配。"""
    pending_ids = set()

    for message in messages:
        role = message.get("role")

        if role == "assistant":
            # assistant 发起新一组工具调用前，
            # 上一组工具调用必须已经全部得到结果。
            if pending_ids:
                return False

            tool_calls = message.get("tool_calls") or []
            call_ids = []

            for call in tool_calls:
                call_id = call.get("id")
                if not call_id or call_id in call_ids:
                    return False
                call_ids.append(call_id)

            pending_ids.update(call_ids)

        elif role == "tool":
            tool_call_id = message.get("tool_call_id")

            # 工具结果必须对应一个尚未完成的调用
            if not tool_call_id or tool_call_id not in pending_ids:
                return False

            pending_ids.remove(tool_call_id)

        elif pending_ids:
            # 工具结果还没收齐，就出现了其他角色的消息
            return False

    # 不允许留下没有结果的工具调用
    return not pending_ids
