"""对话轮次切分。

关键不变量：**一次工具调用及其结果永不跨轮拆分**。
这是最容易出问题的地方 —— 一旦把 assistant(tool_calls) 和对应的
tool 消息拆到不同轮次，裁剪后发给模型的上下文就是非法的，API 会直接报错。
"""

from __future__ import annotations

from ..types import Message

SYSTEM_ROLES = ("system", "developer")


def is_tool_call_message(message: Message) -> bool:
    return message.get("role") == "assistant" and bool(message.get("tool_calls"))


def is_turn_end(message: Message) -> bool:
    """不带 tool_calls 的 assistant 消息代表一轮正常结束。"""
    return message.get("role") == "assistant" and not message.get("tool_calls")


def split_messages(messages: list[Message]) -> tuple[list[Message], list[list[Message]]]:
    """把消息拆成 (system 消息, 轮次列表)。

    轮次规则：
    * 一个轮次从 user 消息开始；
    * `assistant(tool_calls)` + 其后的 `tool` 结果留在同一轮；
    * 直到出现「不带 tool_calls 的 assistant」才收口；
    * 开头若出现没有 user 的消息（例如摘要注入、孤立的 tool 结果），
      会被并入第一个轮次，而不是被单独丢弃。
    """
    system_messages: list[Message] = []
    turns: list[list[Message]] = []
    current: list[Message] = []

    for message in messages:
        role = message.get("role")

        if role in SYSTEM_ROLES:
            # system 只允许出现在最前面；中间出现的 system（如注入的摘要/记忆块）
            # 也当作独立段落处理，避免被重排。
            system_messages.append(message)
            continue

        if role == "user" and current:
            turns.append(current)
            current = []

        current.append(message)

        if is_turn_end(message):
            turns.append(current)
            current = []

    if current:
        turns.append(current)

    return system_messages, turns


def flatten_turns(turns: list[list[Message]]) -> list[Message]:
    return [message for turn in turns for message in turn]


def tail_turns(turns: list[list[Message]], count: int) -> list[list[Message]]:
    if count <= 0:
        return []
    return turns[-count:]


def summarize_turn(turn: list[Message]) -> str:
    """把一轮对话压成一行（用于日志与调试输出）。"""
    parts: list[str] = []
    for message in turn:
        role = message.get("role")
        if role == "tool":
            parts.append("tool→结果")
            continue
        content = " ".join(str(message.get("content") or "").split())
        if is_tool_call_message(message):
            names = ", ".join(
                (call.get("function") or {}).get("name", "?")
                for call in message.get("tool_calls") or []
            )
            parts.append(f"assistant→调用[{names}]")
        elif content:
            parts.append(f"{role}: {content[:40]}")
    return " | ".join(parts) if parts else "(空轮次)"


def count_turns(messages: list[Message]) -> int:
    return len(split_messages(messages)[1])
