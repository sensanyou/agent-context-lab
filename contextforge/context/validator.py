"""工具调用链完整性校验。

「模型请求调用工具」≠「工具已经执行成功」。

发送给模型的上下文里，每一个 `assistant(tool_calls)` 都必须有成对且不重复的
`tool` 结果紧随其后，否则 API 会拒绝请求。压缩、裁剪、跨请求恢复任务状态时，
这个不变量都必须在**剪之前**检查、在**剪之后**复验。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Message


@dataclass
class ToolChainReport:
    ok: bool = True
    problems: list[str] = field(default_factory=list)
    call_count: int = 0
    resolved_count: int = 0

    @property
    def pending(self) -> int:
        return self.call_count - self.resolved_count

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "problems": self.problems,
            "call_count": self.call_count,
            "resolved_count": self.resolved_count,
        }


def check_tool_calls(messages: list[Message]) -> ToolChainReport:
    """检查工具调用与结果是否一一对应。"""
    report = ToolChainReport()
    pending: dict[str, str] = {}  # call_id -> tool_name
    seen: set[str] = set()

    for position, message in enumerate(messages):
        role = message.get("role")

        if role == "assistant":
            calls = message.get("tool_calls") or []
            if not calls:
                if pending:
                    report.ok = False
                    report.problems.append(
                        f"#{position} assistant 在工具结果未收齐时就给出了最终回答："
                        f"待完成 {sorted(pending)}"
                    )
                continue

            if pending:
                report.ok = False
                report.problems.append(
                    f"#{position} assistant 发起新工具调用前，上一组尚未收齐：{sorted(pending)}"
                )

            for call in calls:
                call_id = call.get("id")
                name = (call.get("function") or {}).get("name", "?")
                if not call_id:
                    report.ok = False
                    report.problems.append(f"#{position} 工具调用 {name} 缺少 id")
                    continue
                if call_id in pending or call_id in seen:
                    report.ok = False
                    report.problems.append(f"#{position} 工具调用 id 重复：{call_id}")
                    continue
                pending[call_id] = name
                report.call_count += 1

        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                report.ok = False
                report.problems.append(f"#{position} tool 消息缺少 tool_call_id")
            elif call_id not in pending:
                report.ok = False
                report.problems.append(
                    f"#{position} tool 结果 {call_id} 没有对应的调用（可能已被错误裁剪）"
                )
            else:
                pending.pop(call_id)
                seen.add(str(call_id))
                report.resolved_count += 1

        elif pending:
            report.ok = False
            report.problems.append(
                f"#{position} 在工具结果收齐前出现了 {role} 消息：待完成 {sorted(pending)}"
            )

    if pending:
        report.ok = False
        report.problems.append(f"上下文结尾存在未完成的工具调用：{sorted(pending)}")

    return report


def validate_tool_calls(messages: list[Message]) -> bool:
    """向后兼容的布尔接口。"""
    return check_tool_calls(messages).ok


def repair_tool_calls(messages: list[Message]) -> tuple[list[Message], list[str]]:
    """尽最大努力修复不完整的工具链。

    策略：把「没有结果的工具调用」所在的那一轮整体丢弃。宁可丢一轮，也不能
    发出一个非法请求。返回 (修复后的消息, 修复说明)。
    """
    from .turns import split_messages

    notes: list[str] = []
    system_messages, turns = split_messages(messages)
    kept: list[list[Message]] = []

    for index, turn in enumerate(turns):
        if check_tool_calls(turn).ok and check_tool_calls(
            [m for m in turn if m.get("role") != "system"]
        ).ok:
            kept.append(turn)
        else:
            notes.append(f"丢弃第 {index + 1} 轮：工具调用链不完整")

    repaired = list(system_messages)
    for turn in kept:
        repaired.extend(turn)
    return repaired, notes
