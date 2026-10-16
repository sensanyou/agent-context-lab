"""离线确定性模型。

它不是一个「返回假数据」的桩，而是一个**规则推理器**：
在没有真实 API Key 的环境里，它按照可解释的规则完成三类工作——

1. **工具决策**：根据问题意图从可用工具中选择合适的调用，并生成 JSON 参数。
2. **结果综合**：把工具返回内容、检索片段组织成带引用的回答。
3. **结构化子任务**：当提示词要求「规划 / 反思 / 摘要 / 抽取记忆」时，
   返回对应的确定性结果，使 LLM 参与的节点在离线状态下同样可被验证。

这样做的价值在于：整条工程链路（检索 → 编排 → 工具 → 上下文 → 评测）可以在
零成本、零网络、结果可复现的条件下跑通；接入真实模型时只需替换 `BaseLLM` 实现，
上层代码一行不用改。
"""

from __future__ import annotations

import json
import time
from typing import Any

from ..types import LLMResponse, Message, ToolCallRequest, Usage, message_text
from .base import BaseLLM

# 意图 -> 首选工具（精确名优先，其次按子串匹配）-> 触发词
_TOOL_INTENTS: list[tuple[tuple[str, ...], tuple[str, ...], str]] = [
    # (首选工具名/hints, 触发词, 说明)
    (("get_local_time", "time", "clock"), ("几点", "时间", "日期", "现在几点", "今天"), "询问时间"),
    (("calculator", "calc", "compute"), ("计算", "算一下", "多少", "+", "*", "除以", "乘以"), "计算"),
    (("memory_save", "remember"), ("记住", "请记", "我的偏好", "以后都"), "写入长期记忆"),
    (("memory_search", "recall"), ("我之前", "记得我说过", "我的偏好是什么"), "检索长期记忆"),
    (("repo_overview", "repo", "code", "symbol"), (
        "代码", "仓库", "函数", "模块", "调用链", "项目结构", "目录结构",
        "这个项目", "示例项目", "项目",
    ), "代码仓库分析"),
    (("repo_read", "fs_read", "file"), ("读取文件", "打开文件", "看看文件"), "读文件"),
    (("run_tests", "test", "pytest"), ("跑测试", "运行测试", "测试失败"), "运行测试"),
    (("knowledge_search", "search", "knowledge", "retrieve", "kb"),
     ("查", "搜", "检索", "资料", "文档", "知识库", "根据"), "知识检索"),
]

_PLAN_SYSTEM_HINTS = ("任务规划", "制定计划", "拆解", "plan")
_REFLECT_SYSTEM_HINTS = ("反思", "评估回答", "自我检查", "verify")
# 这些提示词必须足够具体：否则上下文里注入的资料块（含「压缩」「摘要」等词）
# 会被误判成「要求我做摘要」的指令。
_SUMMARY_SYSTEM_HINTS = ("你负责压缩", "请压缩", "压缩下面", "压缩以下", "生成摘要", "summarize the")
_MEMORY_SYSTEM_HINTS = ("记忆提取", "抽取记忆", "长期记忆中值得")

#: 上下文引擎注入的「数据块」前缀 —— 它们是资料，不是指令
_INJECTED_PREFIXES = ("【", "[较早对话摘要", "[关键约束", "[较早对话")

_KEYWORD_WEIGHT = (
    "目标", "必须", "不要", "要求", "偏好", "约定", "结构", "缺陷", "结论", "回答",
)


class MockLLM(BaseLLM):
    """确定性、可解释、零依赖的离线模型。"""

    name = "mock"
    online = False

    def __init__(self, model: str = "contextforge-mock-1", seed: int = 0) -> None:
        self.model = model
        self.seed = seed
        self.supports_embeddings = False
        self.call_count = 0

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        self.call_count += 1

        tools = tools or []
        system_text = self._instruction_text(messages)
        user_text = self._last_user_text(messages)
        tool_messages = self._trailing_tool_messages(messages)

        if self._hits(system_text, _PLAN_SYSTEM_HINTS):
            content = self._make_plan(user_text)
        elif self._hits(system_text, _REFLECT_SYSTEM_HINTS):
            content = self._make_reflection(tool_messages, user_text)
        elif self._hits(system_text, _MEMORY_SYSTEM_HINTS):
            content = self._extract_memories(messages)
        elif self._hits(system_text, _SUMMARY_SYSTEM_HINTS):
            content = self._summarize(messages)
        elif tool_messages:
            # 已经有工具结果 —— 综合成最终回答
            content = self._synthesize(user_text, tool_messages)
        else:
            tool_calls = self._decide_tools(user_text, tools)
            if tool_calls:
                return LLMResponse(
                    content="",
                    tool_calls=tool_calls,
                    usage=self._usage(messages, json.dumps([c.to_openai() for c in tool_calls])),
                    model=self.model,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    finish_reason="tool_calls",
                )
            # 没有要调用的工具时，优先基于上下文里注入的检索资料作答
            content = self._grounded_answer(messages, user_text) or self._answer_directly(user_text)

        return LLMResponse(
            content=content,
            usage=self._usage(messages, content),
            model=self.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason="stop",
            extra={"mock": True},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "online": False,
            "note": "离线确定性模型：无需 API Key 即可端到端运行，用于开发、测试与策略对比实验。",
        }

    # ------------------------------------------------------------------ #
    # 工具决策
    # ------------------------------------------------------------------ #
    def _decide_tools(self, question: str, tools: list[dict]) -> list[ToolCallRequest]:
        if not question or not tools:
            return []

        available = {t["function"]["name"]: t for t in tools}
        calls: list[ToolCallRequest] = []
        lowered = question.lower()

        for name_hints, triggers, _label in _TOOL_INTENTS:
            tool_name = self._match_tool(available, name_hints)
            if not tool_name:
                continue
            if not (any(k in question for k in triggers) or any(k in lowered for k in triggers)):
                continue
            if tool_name in {c.name for c in calls}:
                continue
            arguments = self._build_arguments(tool_name, question, available[tool_name])
            if arguments is None:
                continue
            calls.append(
                ToolCallRequest(
                    id=f"call_{len(calls) + 1}_{abs(hash(tool_name)) % 10_000}",
                    name=tool_name,
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            )
            if len(calls) >= 2:  # 允许一次并行两个工具，体现多工具协作
                break

        return calls

    @staticmethod
    def _match_tool(available: dict[str, dict], hints: tuple[str, ...]) -> str | None:
        """优先精确匹配工具名，再用子串；最后退化到描述匹配。

        精确名优先很关键：`repo_grep` / `repo_overview` / `repo_read` 都含 "repo"，
        只按子串匹配会按字母序选到 `repo_grep`，与用户意图不符。
        """
        for hint in hints:
            if hint in available:
                return hint
        for hint in hints:
            for name in available:
                if hint in name:
                    return name
        for hint in hints:
            for name, spec in available.items():
                if hint in str(spec.get("function", {}).get("description", "")).lower():
                    return name
        return None

    @staticmethod
    def _build_arguments(tool_name: str, question: str, spec: dict) -> dict[str, Any] | None:
        params = (spec.get("function", {}).get("parameters") or {}).get("properties") or {}
        properties = set(params)

        if not properties:
            return {}

        if "expression" in properties:
            expression = _extract_expression(question)
            return {"expression": expression} if expression else None

        for key in ("query", "q"):
            if key in properties:
                return {key: question}

        if "path" in properties:
            return {"path": "."}

        # 通用兜底：把第一个字符串参数填成问题
        for key, schema in params.items():
            if schema.get("type") == "string":
                return {key: question}
        return {}

    # ------------------------------------------------------------------ #
    # 回答生成
    # ------------------------------------------------------------------ #
    def _synthesize(self, question: str, tool_messages: list[Message]) -> str:
        citation_lines: list[str] = []
        plain: list[str] = []
        failures: list[str] = []

        for message in tool_messages:
            name = message.get("name") or "tool"
            content = str(message.get("content") or "").strip()

            if content.startswith(("工具执行失败", "错误：", "参数解析失败")):
                failures.append(f"`{name}` 返回：{content}")
                continue

            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- [") and "](" not in stripped:
                    citation_lines.append(stripped)
                elif stripped:
                    plain.append(stripped)

        blocks: list[str] = []

        if citation_lines:
            blocks.append("**基于知识库检索到的内容：**")
            blocks.append("\n".join(citation_lines[:8]))

        if plain:
            blocks.append("**工具执行结果：**")
            blocks.append("\n".join(f"- {line}" for line in plain[:20]))

        if failures:
            blocks.append("**需要注意的失败项：**")
            blocks.append("\n".join(f"- {item}" for item in failures))

        if not blocks:
            return self._answer_directly(question)

        head = "根据目前的执行结果，回答如下：" if not citation_lines else "根据检索到的资料与工具结果："
        tail = (
            "\n\n> 说明：当前运行在离线确定性模型下，以上内容由工具结果与检索片段直接组织，"
            "用于验证链路正确性；接入真实模型后由模型负责最终措辞。"
        )
        return head + "\n\n" + "\n\n".join(blocks) + tail

    def _grounded_answer(self, messages: list[Message], question: str) -> str | None:
        """基于上下文里注入的检索资料与长期记忆组织回答。

        这是离线模型「有据可依」的关键一步：不编造，只做组织与引用。
        """
        citations = self._injected_citation_lines(messages)
        memories = self._injected_memory_lines(messages)

        if not citations and not memories:
            return None

        blocks: list[str] = []
        if citations:
            blocks.append("根据知识库中检索到的资料：")
            blocks.append("\n".join(citations[:8]))
        if memories:
            blocks.append("结合此前记录的长期记忆：")
            blocks.append("\n".join(memories[:5]))

        blocks.append(
            "以上结论均来自上述片段，编号对应上下文中的引用来源，可逐条核对。"
        )
        return "\n\n".join(blocks)

    @staticmethod
    def _injected_citation_lines(messages: list[Message]) -> list[str]:
        lines: list[str] = []
        for message in messages:
            if message.get("role") != "system":
                continue
            content = str(message.get("content") or "")
            if not content.startswith("【检索到的资料"):
                continue
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- [") and "(score=" in stripped:
                    lines.append(stripped)
        return lines

    @staticmethod
    def _injected_memory_lines(messages: list[Message]) -> list[str]:
        lines: list[str] = []
        for message in messages:
            if message.get("role") != "system":
                continue
            content = str(message.get("content") or "")
            if not content.startswith("【长期记忆"):
                continue
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ("):
                    lines.append(stripped)
        return lines

    def _answer_directly(self, question: str) -> str:
        return (
            f"我是 ContextForge 的离线确定性模型。你问的是：「{question}」。\n\n"
            "当前环境没有配置模型 API Key，所以我没有调用真实大模型来做开放式生成。\n"
            "不过完整的工程链路仍然在运行：\n\n"
            "1. 上下文工程引擎会先按预算裁剪/压缩历史，并校验工具调用链完整性；\n"
            "2. 混合检索会从知识库中取回相关片段（向量 + BM25 + 重排序）；\n"
            "3. 图编排运行时按「规划 → 检索 → 工具 → 综合 → 反思」推进；\n"
            "4. 评测层会记录轨迹、token、成本与指标。\n\n"
            "尝试这样提问可以触发工具与检索：「查一下知识库里关于上下文压缩的说明」"
            "或「计算 (12+8)*3」。"
        )

    # ------------------------------------------------------------------ #
    # 结构化子任务
    # ------------------------------------------------------------------ #
    def _make_plan(self, question: str) -> str:
        needs_tools = any(
            k in question for k in ("时间", "几点", "计算", "算一下", "跑测试", "函数", "项目结构")
        )
        # 纯工具型问题（时间/计算/看代码）不需要检索知识库，
        # 否则会往上下文里塞一堆无关片段，既浪费预算也干扰判断。
        knowledge_signals = ("查", "搜", "检索", "资料", "文档", "知识库", "依据", "根据", "为什么", "区别")
        needs_retrieval = (not needs_tools) or any(k in question for k in knowledge_signals)

        steps = ["理解问题意图并确定所需信息"]
        if needs_retrieval:
            steps.append("从知识库检索相关片段")
        if needs_tools:
            steps.append("调用工具获取精确结果")
        steps.append("组织上下文并生成带引用的回答")
        if question.strip():
            steps.append("检查回答是否覆盖了问题的所有要点")

        plan = {
            "goal": question.strip() or "回答用户问题",
            "steps": steps,
            "need_retrieval": needs_retrieval,
            "need_tools": needs_tools,
        }
        return json.dumps(plan, ensure_ascii=False)

    def _make_reflection(self, tool_messages: list[Message], question: str) -> str:
        has_evidence = bool(tool_messages)
        verdict = {
            "sufficient": has_evidence,
            "missing": [] if has_evidence else ["缺少工具或检索证据"],
            "should_retry": not has_evidence,
            "comment": "已有证据支撑，可以直接作答。" if has_evidence else "没有获得证据，建议重新检索。",
        }
        return json.dumps(verdict, ensure_ascii=False)

    def _extract_memories(self, messages: list[Message]) -> str:
        """抽取值得跨任务保留的信息。

        关键是**不要把用户提问当成记忆** —— 否则几轮对话之后，长期记忆里
        全是问题，真正有用的偏好和事实反而被淹没。
        """
        facts: list[dict[str, str]] = []

        for message in messages:
            if message.get("role") != "user":
                continue
            text = str(message.get("content") or "").strip()
            if len(text) < 8:
                continue
            # 提问不是记忆
            if text.endswith(("？", "?")) or text.startswith(("查", "搜", "帮我", "请", "现在")):
                continue

            kind = "fact"
            importance = 0.55
            if any(k in text for k in ("我喜欢", "我偏好", "我希望", "以后都", "请始终", "不要用")):
                kind, importance = "preference", 0.8
            elif any(k in text for k in ("我的目标", "我打算", "我计划", "我想做", "长期目标")):
                kind, importance = "goal", 0.8
            elif any(k in text for k in ("进行到", "已完成", "下一步要", "当前进度")):
                kind, importance = "progress", 0.6
            elif any(k in text for k in ("技术栈", "我用的", "我的项目", "架构是")):
                kind, importance = "fact", 0.65
            else:
                continue  # 普通闲聊不入库

            facts.append({"type": kind, "content": text[:200], "importance": str(importance)})

        return json.dumps({"memories": facts[-5:]}, ensure_ascii=False)

    def _summarize(self, messages: list[Message]) -> str:
        """抽取式摘要：关键词优先 + 覆盖近期内容。"""
        lines: list[str] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            content = str(message.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                lines.append(("用户", content))
            elif role == "assistant" and not message.get("tool_calls"):
                lines.append(("助手", content))

        if not lines:
            return "（无可摘要内容）"

        scored: list[tuple[float, str]] = []
        total = len(lines)
        for position, (role, content) in enumerate(lines):
            recency = (position + 1) / total
            priority = 1.0 if any(k in content for k in _KEYWORD_WEIGHT) else 0.0
            brevity = 1.0 / (1.0 + len(content) / 400.0)
            score = recency * 0.5 + priority * 0.35 + brevity * 0.15
            scored.append((score, f"{role}：{_squeeze(content, 160)}"))

        scored.sort(key=lambda item: item[0], reverse=True)
        picked = [text for _score, text in scored[:8]]
        return "\n".join(f"- {text}" for text in picked)

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _instruction_text(messages: list[Message]) -> str:
        """只取「指令性」的系统提示，跳过上下文引擎注入的资料块。"""
        parts: list[str] = []
        for message in messages:
            if message.get("role") != "system":
                continue
            content = str(message.get("content") or "")
            if content.startswith(_INJECTED_PREFIXES):
                continue
            parts.append(content)
        return "\n".join(parts)

    @staticmethod
    def _last_user_text(messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def _trailing_tool_messages(messages: list[Message]) -> list[Message]:
        """取最后一轮 assistant tool_calls 之后的 tool 结果。"""
        results: list[Message] = []
        for message in reversed(messages):
            role = message.get("role")
            if role == "tool":
                results.append(message)
            elif role == "assistant" and message.get("tool_calls"):
                break
            elif role == "user":
                break
        return list(reversed(results))

    @staticmethod
    def _hits(text: str, hints: tuple[str, ...]) -> bool:
        lowered = text.lower()
        return any(hint.lower() in lowered for hint in hints)

    @staticmethod
    def _usage(messages: list[Message], completion: str) -> Usage:
        prompt = sum(max(len(message_text(m)) // 3, 1) for m in messages)
        output = max(len(completion) // 3, 1) if completion else 0
        return Usage(prompt, output, prompt + output)


def _extract_expression(question: str) -> str | None:
    allowed = set("0123456789+-*/(). %")
    expr = "".join(ch for ch in question if ch in allowed).strip()
    if not expr or not any(ch.isdigit() for ch in expr):
        return None
    return expr


def _squeeze(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
