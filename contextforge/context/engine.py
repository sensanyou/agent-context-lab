"""上下文工程引擎 —— 项目的能力中枢。

它是**独立、可复用、可观测的组件**，被 Agent Runtime、RAG 检索、
工具执行共同调用：

    输入：系统提示 + 长期记忆 + 检索片段 + 历史对话 + 工具定义 + 当前问题
    处理：分区预算 → 按策略压缩 → 工具链完整性校验 → 超预算兜底降级
    输出：合法的消息列表 + 一份 ContextReport（谁被保留、谁被丢弃、为什么）

`ContextReport` 是「上下文效率」这件事变得可量化的关键：
评测层拿它计算压缩率、信息保留率，并对比不同策略的实际效果。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ContextSettings, get_settings
from ..types import Citation, ContextReport, Message, ScoredChunk
from .budget import (
    SECTION_CURRENT,
    SECTION_HISTORY,
    SECTION_MEMORY,
    SECTION_RECENT,
    SECTION_RETRIEVED,
    SECTION_SYSTEM,
    BudgetAllocator,
)
from .strategies import get_strategy
from .tokens import TokenCounter, approximate_tokens, get_counter
from .validator import check_tool_calls, repair_tool_calls

RETRIEVED_HEADER = "【检索到的资料｜请优先据此回答，并在句中标注 [编号] 引用】"
MEMORY_HEADER = "【长期记忆｜跨任务保留的用户偏好与项目事实】"


@dataclass
class ContextRequest:
    system_prompt: str
    user_input: str
    history: list[Message] = field(default_factory=list)
    retrieved: list[ScoredChunk] = field(default_factory=list)
    memories: list[dict] = field(default_factory=list)
    tool_specs: list[dict] = field(default_factory=list)
    #: 当前任务「在飞」的消息（assistant 工具调用、tool 结果等）。
    #: 它们原样保留、不参与压缩、也不会被兜底降级丢弃 —— 否则工具链会被打断。
    recent: list[Message] = field(default_factory=list)
    strategy: str | None = None
    keep_recent_turns: int | None = None


@dataclass
class ContextBundle:
    messages: list[Message]
    report: ContextReport
    citations: list[Citation] = field(default_factory=list)
    tool_specs: list[dict] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        return self.report.tokens_after


class ContextEngine:
    """统一构建每一轮真正发给模型的上下文。"""

    def __init__(
        self,
        settings: ContextSettings | None = None,
        counter: TokenCounter | None = None,
        llm=None,
    ) -> None:
        self.settings = settings or get_settings().context
        self.counter = counter or get_counter()
        self.llm = llm

    # ------------------------------------------------------------------ #
    def build(self, request: ContextRequest) -> ContextBundle:
        allocator = BudgetAllocator(self.settings, self.counter)
        report = ContextReport(
            strategy=request.strategy or self.settings.strategy,
            budget_total=self.settings.token_budget,
            reserved_output=self.settings.reserved_output_tokens,
            allocation=allocator.allocation,
        )

        conversation_history = [m for m in request.history if m.get("role") != "system"]

        report.tokens_before = self._count_input_tokens(request, conversation_history)

        # 1) 压缩历史（以轮次为单位，保证工具链不被拆散）
        keep_recent = (
            self.settings.keep_recent_turns
            if request.keep_recent_turns is None
            else request.keep_recent_turns
        )
        strategy = get_strategy(report.strategy, self.counter)

        # 1) 压缩历史（以轮次为单位，保证工具链不被拆散）
        #
        # 这里刻意用「逐步降低保留轮数」而不是「直接丢消息」：
        # 历史区超预算时，先减少原样保留的近期轮数，再让摘要继续兜底。
        # 旧实现直接砍消息，会把摘要和关键约束先砍掉，方向正好是反的。
        history_limit = allocator.limit_of(SECTION_HISTORY)
        # 二维降级：先减少「原样保留的近期轮数」，再缩短摘要长度。
        # 只降一个维度是不够的 —— 摘要本身太长时，光减轮数永远无法落入额度。
        summary_chars = max(int(history_limit * 0.9), 200)
        attempts = [keep_recent] + [k for k in (4, 2, 1, 0) if k < keep_recent]
        shrinks = (1.0, 0.6, 0.3)

        compression = None
        fitted = False
        for attempt_index, candidate in enumerate(attempts):
            for shrink in shrinks:
                compression = strategy.compress(
                    conversation_history,
                    keep_recent_turns=candidate,
                    max_summary_chars=max(int(summary_chars * shrink), 80),
                    llm=self.llm,
                )
                if self.counter.count_messages(compression.messages) <= history_limit:
                    fitted = True
                    break
            if fitted:
                if attempt_index or shrink < 1.0:
                    report.notes.append(
                        f"历史区超预算：保留轮数 {keep_recent}→{candidate}，"
                        f"摘要长度 ×{shrink:g}"
                    )
                break

        assert compression is not None  # attempts 非空
        if not fitted:
            report.notes.append(
                f"历史区额度 {history_limit} tokens 过紧，已交由兜底降级处理"
            )
        report.kept_turns = compression.kept_turns
        report.dropped_turns = compression.dropped_turns
        report.summarized_messages = compression.summarized_messages
        report.notes.extend(compression.notes)
        if compression.pinned:
            report.notes.append(f"钉住关键约束 {len(compression.pinned)} 条")

        # 2) 分区装配
        sections: list[tuple[str, list[Message]]] = []

        system_messages, used, notes = allocator.take_text(request.system_prompt, SECTION_SYSTEM)
        report.notes.extend(notes)
        report.sections[SECTION_SYSTEM] = used
        sections.append((SECTION_SYSTEM, system_messages))

        if request.memories:
            memory_text = _render_memories(request.memories)
            memory_messages, used, notes = allocator.take_text(memory_text, SECTION_MEMORY)
            report.notes.extend(notes)
            report.sections[SECTION_MEMORY] = used
            sections.append((SECTION_MEMORY, memory_messages))

        citations: list[Citation] = []
        if request.retrieved:
            retrieved_messages, used, citations, dropped = self._take_retrieved(
                request.retrieved, allocator.limit_of(SECTION_RETRIEVED)
            )
            report.sections[SECTION_RETRIEVED] = used
            report.dropped_chunks = dropped
            if dropped:
                report.notes.append(f"检索片段超出额度，丢弃 {len(dropped)} 条低分片段")
            sections.append((SECTION_RETRIEVED, retrieved_messages))

        history_messages, used, notes = allocator.take_messages(
            compression.messages,
            SECTION_HISTORY,
            anchor_count=_count_anchors(compression.messages),
        )
        report.notes.extend(notes)
        report.sections[SECTION_HISTORY] = used
        sections.append((SECTION_HISTORY, history_messages))

        current_messages, used, notes = allocator.take_text(
            request.user_input, SECTION_CURRENT, role="user"
        )
        report.notes.extend(notes)
        report.sections[SECTION_CURRENT] = used
        sections.append((SECTION_CURRENT, current_messages))

        # 「在飞」消息原样追加，不裁剪、不压缩
        if request.recent:
            recent_tokens = self.counter.count_messages(request.recent)
            report.sections[SECTION_RECENT] = recent_tokens
            sections.append((SECTION_RECENT, list(request.recent)))

        # 3) 工具定义也占预算，超支时按与问题的相关度取舍
        tool_specs, tool_tokens = self._fit_tools(request.tool_specs, request.user_input, allocator)
        report.sections["tools"] = tool_tokens
        if len(tool_specs) < len(request.tool_specs):
            report.notes.append(
                f"工具定义超出额度，仅保留 {len(tool_specs)}/{len(request.tool_specs)} 个"
            )

        messages = [m for _name, block in sections for m in block]

        # 4) 工具链完整性：校验 → 修复 → 复验
        if self.settings.enforce_tool_integrity:
            check = check_tool_calls(messages)
            if not check.ok:
                report.notes.append("工具链不完整：" + "；".join(check.problems[:3]))
                messages, repair_notes = repair_tool_calls(messages)
                report.notes.extend(repair_notes)
                check = check_tool_calls(messages)
            report.tool_integrity_ok = check.ok
            if not check.ok:
                report.notes.append("修复后仍不完整，已阻止发送")

        # 5) 总量兜底：仍然超预算时逐级降级（在飞消息受保护，不会被丢弃）
        protected = {id(message) for message in request.recent}
        messages = self._degrade_until_fit(messages, allocator, report, protected)

        report.tokens_after = self.counter.count_messages(messages) + tool_tokens

        return ContextBundle(
            messages=messages,
            report=report,
            citations=citations,
            tool_specs=tool_specs,
        )

    # ------------------------------------------------------------------ #
    def estimate(self, messages: list[Message]) -> int:
        return self.counter.count_messages(messages)

    # ------------------------------------------------------------------ #
    def _count_input_tokens(self, request: ContextRequest, history: list[Message]) -> int:
        """压缩前的输入总量。

        关键点：这里按「这些内容若原样发送会占多少 token」来统计，也就是
        **按消息口径**计数（含每条消息的角色开销），而不是按纯文本口径。
        否则 `tokens_before` 用 `count_text`、`tokens_after` 用 `count_messages`，
        两把尺子不一致 —— 连「不压缩」的策略都会算出压缩率 > 1 的荒谬结果。
        """
        blocks: list[Message] = list(history)
        blocks.append({"role": "system", "content": request.system_prompt})
        if request.memories:
            blocks.append({"role": "system", "content": _render_memories(request.memories)})
        if request.retrieved:
            blocks.append(
                {
                    "role": "system",
                    "content": "\n".join(item.chunk.text for item in request.retrieved),
                }
            )
        if request.recent:
            blocks.extend(request.recent)
        blocks.append({"role": "user", "content": request.user_input})

        total = self.counter.count_messages(blocks)
        if request.tool_specs:
            total += approximate_tokens(request.tool_specs, self.counter)
        return total

    def _take_retrieved(
        self,
        retrieved: list[ScoredChunk],
        limit: int,
    ) -> tuple[list[Message], int, list[Citation], list[str]]:
        """按分数从高到低填装检索片段，装不下的如实记录。"""
        if limit <= 0:
            return [], 0, [], [item.chunk.chunk_id for item in retrieved]

        header_cost = self.counter.count_text(RETRIEVED_HEADER)
        used = header_cost
        lines: list[str] = []
        citations: list[Citation] = []
        dropped: list[str] = []

        for rank, item in enumerate(retrieved, start=1):
            label = item.chunk.citation()
            snippet = " ".join(item.chunk.text.split())
            line = f"- [{rank}] {label} (score={item.score:.3f}): {snippet}"
            cost = self.counter.count_text(line)
            if used + cost > limit:
                dropped.append(f"[{rank}] {label}")
                continue
            lines.append(line)
            used += cost
            citations.append(
                Citation(
                    chunk_id=item.chunk.chunk_id,
                    label=label,
                    score=round(item.score, 4),
                    snippet=snippet[:240],
                )
            )

        if not lines:
            return [], 0, [], [item.chunk.chunk_id for item in retrieved]

        content = RETRIEVED_HEADER + "\n" + "\n".join(lines)
        return [{"role": "system", "content": content}], used, citations, dropped

    def _fit_tools(
        self,
        specs: list[dict],
        user_input: str,
        allocator: BudgetAllocator,
    ) -> tuple[list[dict], int]:
        if not specs:
            return [], 0

        limit = allocator.limit_of("tools")
        # 优先保留与当前问题字面相关的工具，避免关键词命中的工具被裁掉
        lowered = user_input.lower()

        def relevance(spec: dict) -> int:
            function = spec.get("function", {})
            haystack = f"{function.get('name', '')} {function.get('description', '')}".lower()
            hits = sum(1 for token in _tokens(lowered) if token and token in haystack)
            return -hits

        ordered = sorted(specs, key=relevance)
        kept: list[dict] = []
        used = 0
        for spec in ordered:
            cost = approximate_tokens(spec, self.counter)
            if used + cost > limit and kept:
                continue
            kept.append(spec)
            used += cost

        # 恢复原始顺序，保持提示词稳定（有利于缓存命中）
        order = {spec["function"]["name"]: i for i, spec in enumerate(specs)}
        kept.sort(key=lambda spec: order.get(spec["function"]["name"], 0))
        return kept, used

    def _degrade_until_fit(
        self,
        messages: list[Message],
        allocator: BudgetAllocator,
        report: ContextReport,
        protected: set[int] | None = None,
    ) -> list[Message]:
        """最后一道防线：超预算时逐级丢信息，绝不发出超长请求。

        `protected` 里的消息（当前任务的在飞工具链）永不丢弃。
        """
        protected = protected or set()
        working = list(messages)
        rounds = 0

        def droppable(index: int) -> bool:
            return id(working[index]) not in protected

        while rounds < 6:
            total = self.counter.count_messages(working)
            if allocator.fits(total):
                break
            rounds += 1

            # 第一优先丢检索片段（最容易被替代），其次丢摘要，最后丢历史
            removable_index = None
            for index in range(len(working) - 1, -1, -1):
                if not droppable(index):
                    continue
                if str(working[index].get("content", "")).startswith(RETRIEVED_HEADER):
                    removable_index = index
                    break
            if removable_index is None:
                removable_index = _first_index_where(
                    working,
                    lambda m: str(m.get("content", "")).startswith("[较早对话摘要"),
                )
            if removable_index is None:
                removable_index = _first_index_where(
                    working,
                    lambda m: str(m.get("content", "")).startswith("[关键约束"),
                )
            if removable_index is None:
                removable_index = _first_index_where(
                    working,
                    lambda m: m.get("role") in ("user", "assistant", "tool"),
                    protected=protected,
                )
            if removable_index is None:
                report.notes.append("已无可丢弃内容，仍超预算")
                break

            removed = working.pop(removable_index)
            preview = " ".join(str(removed.get("content") or "").split())[:30]
            report.notes.append(f"兜底降级：丢弃一条 {removed.get('role')} 消息（{preview}）")

        return working


# --------------------------------------------------------------------------- #
def _count_anchors(messages: list[Message]) -> int:
    """统计位于列表最前面的「锚点消息」数量（钉住的约束、历史摘要）。"""
    count = 0
    for message in messages:
        content = str(message.get("content") or "")
        if message.get("role") == "system" or content.startswith(
            ("[较早对话摘要", "[关键约束")
        ):
            count += 1
            continue
        break
    return count


def _render_memories(memories: list[dict]) -> str:
    lines: list[str] = [MEMORY_HEADER]
    for item in memories:
        kind = item.get("kind") or item.get("type") or "fact"
        content = " ".join(str(item.get("content", "")).split())
        score = item.get("score")
        suffix = f" (score={score:.2f})" if isinstance(score, (int, float)) else ""
        lines.append(f"- ({kind}){suffix} {content}")
    return "\n".join(lines)


def _tokens(text: str) -> list[str]:
    import re

    return [t for t in re.split(r"[^\w\u4e00-\u9fff]+", text) if len(t) >= 2]


def _last_index_where(items: list[Message], predicate) -> int | None:
    for index in range(len(items) - 1, -1, -1):
        if predicate(items[index]):
            return index
    return None


def _first_index_where(items: list[Message], predicate, protected: set[int] | None = None) -> int | None:
    protected = protected or set()
    for index, item in enumerate(items):
        if id(item) in protected:
            continue
        if predicate(item):
            return index
    return None
