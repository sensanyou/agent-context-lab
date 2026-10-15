"""上下文压缩工具。

当前提供不依赖外部模型的基础压缩策略：将较早的完整对话轮次
整理成一条简短的文本摘要。后续可以再升级为调用 LLM 生成摘要。
"""

from typing import Any

from agent.context import _split_into_turns


Message = dict[str, Any]
IMPORTANT_KEYWORDS = [
    "长期目标",
    "学习目标",
    "任务要求",
    "技术栈",
    "项目要求",
    "必须",
    "不要",
    "需要使用",
]


def compress_history(
    messages: list[Message],
    keep_recent_turns: int = 4,
    max_summary_chars: int = 1200,
) -> list[Message]:
    """压缩旧对话，保留 system、旧对话摘要和最近几轮原始对话。"""

    if keep_recent_turns < 0:
        raise ValueError("keep_recent_turns 不能小于 0")

    if max_summary_chars < 100:
        raise ValueError("max_summary_chars 至少为 100")

    system_messages, turns = _split_into_turns(messages)

    # 历史轮数没有超过保留数量时，不需要压缩。
    if len(turns) <= keep_recent_turns:
        return list(messages)

    # 将历史拆成需要摘要的旧轮次和需要原样保留的近期轮次。
    if keep_recent_turns == 0:
        old_turns = turns
        recent_turns = []
    else:
        old_turns = turns[:-keep_recent_turns]
        recent_turns = turns[-keep_recent_turns:]

    important_lines: list[str] = []
    normal_lines: list[str] = []

    for turn in old_turns:
        for message in turn:
            role = message.get("role", "unknown")
            content = str(message.get("content") or "").strip()

            # 之前生成的摘要也要参与下一次压缩。
            is_old_summary = (
                role == "user"
                and content.startswith(
                    "[较早对话摘要，仅作背景信息，不是新的指令]"
                )
            )

            if is_old_summary:
                content = content.split("\n", 1)[-1].strip()

            if not content:
                continue

            if role == "user":
                line = f"用户曾提到：{content}"
            elif role == "assistant" and not message.get("tool_calls"):
                line = f"助手曾回答：{content}"
            else:
                # 工具调用及工具结果暂时不直接写入摘要。
                continue

            if any(keyword in content for keyword in IMPORTANT_KEYWORDS):
                important_lines.append("【重点】" + line)
            else:
                normal_lines.append(line)

    # 重要信息优先，普通信息随后。
    summary_lines = important_lines + normal_lines
    summary_text = "\n".join(summary_lines)

    if not summary_text:
        summary_text = "较早的对话已被压缩；如需准确细节，请向用户确认。"

    if len(summary_text) > max_summary_chars:
        summary_text = summary_text[: max_summary_chars - 1] + "…"

    summary_message: Message = {
        "role": "user",
        "content": (
            "[较早对话摘要，仅作背景信息，不是新的指令]\n"
            + summary_text
        ),
    }

    recent_messages = [
        message
        for turn in recent_turns
        for message in turn
    ]

    return system_messages + [summary_message] + recent_messages
