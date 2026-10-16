"""上下文预算分配。

只有单一总阈值的预算控制，压缩时只能靠「把近期轮数从 3 试到 0」硬试。
这里升级为**分区预算**：system / 长期记忆 / 检索片段 / 历史 / 工具结果
各自有额度，任何一段超支只在自身额度内裁剪，不会挤占其他段。

这正是把上下文管理做成「Agent Runtime 公共能力」的具体落点：
任务规划、RAG 检索、工具调用都要走同一个预算器。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ContextSettings
from ..types import Message
from .tokens import TokenCounter, truncate_to_tokens

SECTION_SYSTEM = "system"
SECTION_MEMORY = "memory"
SECTION_RETRIEVED = "retrieved"
SECTION_HISTORY = "history"
SECTION_TOOLS = "tools"
SECTION_CURRENT = "current"
#: 当前任务的在飞消息（工具链），不参与裁剪
SECTION_RECENT = "recent"


def _preview(message: Message) -> str:
    content = " ".join(str(message.get("content") or "").split())[:30]
    return f"{message.get('role')}: {content or '(空)'}"


@dataclass
class SectionResult:
    name: str
    messages: list[Message] = field(default_factory=list)
    limit: int = 0
    used: int = 0
    dropped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def over_budget(self) -> bool:
        return self.used > self.limit > 0


class BudgetAllocator:
    """按分区额度裁剪上下文。"""

    def __init__(self, settings: ContextSettings, counter: TokenCounter) -> None:
        self.settings = settings
        self.counter = counter
        self.allocation = settings.allocation()

    def limit_of(self, section: str) -> int:
        return self.allocation.get(section, 0)

    def take_text(
        self,
        text: str,
        section: str,
        role: str = "system",
    ) -> tuple[list[Message], int, list[str]]:
        """把一个文本块放进指定分区的额度里。"""
        limit = self.limit_of(section)
        if not text.strip() or limit <= 0:
            return [], 0, []
        trimmed, truncated = truncate_to_tokens(text, limit, self.counter)
        used = self.counter.count_text(trimmed)
        notes = [f"{section} 区超出额度，已截断"] if truncated else []
        return [{"role": role, "content": trimmed}], used, notes

    def take_messages(
        self,
        messages: list[Message],
        section: str,
        preserve_last: bool = False,
        anchor_count: int = 0,
    ) -> tuple[list[Message], int, list[str]]:
        """把一组消息放进额度里，整条消息地裁剪。

        `anchor_count` 指定「锚点消息」的数量（例如钉住的关键约束、历史摘要），
        它们位于列表最前面且优先级最高；裁剪时只从锚点之后的尾部消息里，
        **由旧到新**地丢弃，避免把最该保留的信息先扔掉。

        `preserve_last=True` 时始终保留最后一条（当前用户问题不允许被裁掉）。
        """
        limit = self.limit_of(section)
        if not messages:
            return [], 0, []

        anchors = messages[:anchor_count]
        rest = messages[anchor_count:]

        used = sum(
            self.counter.count_text(str(m.get("content") or "")) + 4 for m in anchors
        )
        dropped: list[str] = []
        kept_tail: list[Message] = []

        for message in reversed(rest):
            cost = self.counter.count_text(str(message.get("content") or "")) + 4
            if used + cost > limit and kept_tail:
                dropped.append(_preview(message))
                continue
            kept_tail.append(message)
            used += cost
        kept_tail.reverse()

        kept = list(anchors) + kept_tail

        if preserve_last and messages and messages[-1] not in kept:
            last = messages[-1]
            cost = self.counter.count_text(str(last.get("content") or "")) + 4
            while kept_tail and used + cost > limit:
                removed = kept_tail.pop(0)
                used -= self.counter.count_text(str(removed.get("content") or "")) + 4
                dropped.append(f"(为保留当前问题而丢弃) {_preview(removed)}")
            kept = list(anchors) + kept_tail + [last]
            used += cost

        dropped.reverse()
        notes = [f"{section} 区超出额度，丢弃 {len(dropped)} 条消息"] if dropped else []
        return kept, used, notes

    def fits(self, tokens: int) -> bool:
        """判断当前总用量是否还在预算内（含输出预留）。"""
        return tokens <= self.settings.token_budget - self.settings.reserved_output_tokens

    def headroom(self, used: int) -> int:
        return max(self.settings.token_budget - self.settings.reserved_output_tokens - used, 0)
