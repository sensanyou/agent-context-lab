"""Agent 工作流节点。

每个节点只做一件事，并返回**局部状态更新**；跨节点共享的数据全部写在状态里。
这样做的直接好处是：任意节点都可以单独测试，整张图也可以从任意检查点恢复。

    规划 prepare ──(需要检索?)──> 检索 retrieve ──> 组装上下文 build_context
                                                          │
                                      ┌───────────────────▼──────────────────┐
                                      │            推理 reason               │
                                      └───────┬──────────────────┬───────────┘
                                    有工具调用 │                  │ 无工具调用
                                              ▼                  ▼
                                          工具 tools          反思 reflect
                                              │                  │
                                              └──> 组装上下文 <───┘（需要重试）
                                              │
                                              ▼
                                          收尾 finalize
"""

from __future__ import annotations

import json
from typing import Any

from ..context import ContextRequest
from ..types import Message, assistant, user
from .state import AgentState, retrieved_from_state, state_from_retrieved

# --------------------------------------------------------------------------- #
# 提示词
# --------------------------------------------------------------------------- #
PLAN_SYSTEM = """你是任务规划器。把用户的问题拆成可执行的步骤，并判断需要哪些能力。

只输出 JSON，不要输出别的内容：
{"goal": "一句话目标", "steps": ["步骤1", "步骤2"], "need_retrieval": true, "need_tools": false}

判断标准：
- need_retrieval：问题需要依据资料、文档、代码库或历史知识回答时为 true
- need_tools：需要获取实时信息、精确计算或读取本地文件时为 true"""

REFLECT_SYSTEM = """你是回答质量检查器。检查当前回答是否有足够证据支撑、是否覆盖了问题的全部要点。

只输出 JSON：
{"sufficient": true, "missing": ["缺失的要点"], "should_retry": false, "comment": "简短评价"}"""


def build_system_prompt(ctx) -> str:
    """基础系统提示：约束回答风格、引用要求与工具使用原则。"""
    parts = [
        "你是 ContextForge，一个面向个人学习、代码理解与技术研究的智能助手。",
        "",
        "行为准则：",
        "1. 优先依据检索到的资料与工具返回结果作答，不要凭记忆编造事实。",
        "2. 使用检索资料时，在句中标出 [编号] 引用，方便用户核对来源。",
        "3. 工具返回错误时，说明失败原因并给出替代方案，不要假装成功。",
        "4. 信息不足以回答时，明确说明缺什么，而不是猜测。",
        "5. 回答使用中文，先给结论再给依据，代码块标注语言。",
    ]
    if getattr(ctx, "extra_instructions", ""):
        parts.extend(["", ctx.extra_instructions])
    return "\n".join(part for part in parts if part is not None)


# --------------------------------------------------------------------------- #
# 节点
# --------------------------------------------------------------------------- #
def prepare(state: AgentState, ctx) -> dict[str, Any]:
    """收集长期记忆与技能先验，并让模型做一次任务规划。"""
    with ctx.tracer.span("prepare", "node") as span:
        span.attributes["node"] = "prepare"
        memories: list[dict[str, Any]] = []
        memory_block = ""
        skills: list[dict[str, Any]] = []

        if ctx.memory is not None:
            bundle = ctx.memory.prepare(state.question, top_k=ctx.settings.retrieval.final_top_k)
            memories = [memory.to_dict() for memory in bundle.long_term]
            skills = [skill.to_dict() for skill in bundle.skills]
            memory_block = bundle.as_prompt_block()

        plan: dict[str, Any] = {}
        if ctx.llm is not None:
            with ctx.tracer.span("plan", "llm") as plan_span:
                response = ctx.llm.chat(
                    [
                        {"role": "system", "content": PLAN_SYSTEM},
                        {"role": "user", "content": state.question},
                    ]
                )
                plan_span.attributes["prompt_tokens"] = response.usage.prompt_tokens
                plan_span.attributes["completion_tokens"] = response.usage.completion_tokens
                plan = _parse_json(response.content)

        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        span.attributes["steps"] = len(steps)
        span.attributes["memories"] = len(memories)

        return {
            "plan": [str(step) for step in steps][:8],
            "goal": str(plan.get("goal") or state.question),
            "need_retrieval": bool(plan.get("need_retrieval", True)),
            "memories": memories,
            "memory_block": memory_block,
            "messages": [user(state.question)],
            "observations": [
                f"规划 {len(steps)} 步；长期记忆命中 {len(memories)} 条；技能命中 {len(skills)} 个"
            ],
        }


def retrieve(state: AgentState, ctx) -> dict[str, Any]:
    """从知识库检索相关片段。"""
    if ctx.knowledge is None:
        return {"observations": ["未配置知识库，跳过检索"], "need_retrieval": False}

    with ctx.tracer.span("retrieve", "retriever") as span:
        span.attributes["node"] = "retrieve"
        hits = ctx.knowledge.search(
            state.question,
            top_k=ctx.settings.retrieval.final_top_k,
            mode=state.retrieval_mode or "hybrid",
        )
        span.attributes["hits"] = len(hits)
        if ctx.knowledge.retriever.last_trace is not None:
            span.attributes.update(ctx.knowledge.retriever.last_trace.to_dict())

        if not hits:
            return {"observations": ["知识库没有命中相关内容"]}

        summary = "；".join(f"{item.chunk.citation()}({item.score:.2f})" for item in hits[:3])
        return {
            "retrieved": state_from_retrieved(hits),
            "observations": [f"检索到 {len(hits)} 条片段：{summary}"],
        }


def build_context(state: AgentState, ctx) -> dict[str, Any]:
    """调用上下文引擎组装本轮真正发给模型的消息。

    这是「上下文管理作为公共能力」的落点：无论第几轮、无论前面发生了什么，
    发给模型的消息都必须经过同一个引擎、同一套预算与同一条工具链完整性校验。
    """
    with ctx.tracer.span("build_context", "chain") as span:
        span.attributes["node"] = "build_context"
        persisted: list[Message] = []
        if ctx.memory is not None and state.conversation_id:
            persisted = ctx.memory.load_history(state.conversation_id)

        system_prompt = build_system_prompt(ctx)
        if state.plan:
            system_prompt += "\n\n本任务的执行计划：\n" + "\n".join(
                f"{i + 1}. {step}" for i, step in enumerate(state.plan)
            )

        request = ContextRequest(
            system_prompt=system_prompt,
            user_input="",
            history=[m for m in persisted if m.get("role") != "system"],
            retrieved=retrieved_from_state(state.retrieved),
            memories=state.memories,
            tool_specs=ctx.tools.openai_schemas() if ctx.tools is not None else [],
            recent=[dict(m) for m in state.messages],
            strategy=state.strategy or None,
        )

        bundle = ctx.engine.build(request)
        span.attributes["tokens_after"] = bundle.report.tokens_after
        span.attributes["tokens_before"] = bundle.report.tokens_before
        span.attributes["compression_ratio"] = bundle.report.compression_ratio

        return {
            "built_messages": bundle.messages,
            "context_report": bundle.report.to_dict(),
            "citations": [citation.to_dict() for citation in bundle.citations],
            "observations": [
                f"上下文 {bundle.report.tokens_after} tokens（压缩比 {bundle.report.compression_ratio:.2f}）"
            ],
        }


def reason(state: AgentState, ctx) -> dict[str, Any]:
    """让模型基于当前上下文决定「调用工具」还是「给出回答」。"""
    if ctx.llm is None:
        return {"status": "failed", "error": "没有可用的模型"}

    with ctx.tracer.span("reason", "llm") as span:
        span.attributes["node"] = "reason"
        tool_specs = ctx.tools.openai_schemas() if ctx.tools is not None else []
        outbound = state.built_messages or [user(state.question)]
        response = ctx.llm.chat(outbound, tools=tool_specs or None)

        span.attributes["tool_calls"] = len(response.tool_calls)
        span.attributes["prompt_tokens"] = response.usage.prompt_tokens
        span.attributes["completion_tokens"] = response.usage.completion_tokens
        span.attributes["latency_ms"] = round(response.latency_ms, 2)

    usage = state.usage
    if not isinstance(usage, dict):
        usage = {}
    usage = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)) + response.usage.prompt_tokens,
        "completion_tokens": int(usage.get("completion_tokens", 0)) + response.usage.completion_tokens,
        "total_tokens": int(usage.get("total_tokens", 0)) + response.usage.total_tokens,
    }

    iteration = state.iteration + 1
    messages = list(state.messages)

    if response.tool_calls:
        messages.append(assistant(response.content or None, [call.to_openai() for call in response.tool_calls]))
        return {
            "iteration": iteration,
            "usage": usage,
            "messages": messages,
            "pending_tool_calls": [call.to_openai() for call in response.tool_calls],
            "observations": [
                "模型请求调用工具：" + "、".join(call.name for call in response.tool_calls)
            ],
        }

    messages.append(assistant(response.content or ""))
    return {
        "iteration": iteration,
        "usage": usage,
        "messages": messages,
        "answer": response.content or "",
        "pending_tool_calls": [],
        "observations": ["模型给出最终回答"],
    }


def tools(state: AgentState, ctx) -> dict[str, Any]:
    """执行模型请求的工具，并把结果写回会话。"""
    if ctx.tools is None:
        return {"pending_tool_calls": [], "observations": ["没有配置工具"]}

    with ctx.tracer.span("tools", "node") as node_span:
        node_span.attributes["node"] = "tools"
        messages = list(state.messages)
        executions: list[dict[str, Any]] = []
        observations: list[str] = []

        for call in state.pending_tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            call_id = call.get("id", "")

            with ctx.tracer.span(f"tool:{name}", "tool") as span:
                execution = ctx.tools.execute(
                    name, function.get("arguments", "{}"), tool_call_id=call_id
                )
                span.attributes["ok"] = execution.ok
                span.attributes["latency_ms"] = round(execution.latency_ms, 2)
                span.attributes["truncated"] = execution.truncated
                if not execution.ok:
                    span.attributes["error"] = execution.error

            executions.append(execution.to_dict())
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": execution.result,
                }
            )
            status = "成功" if execution.ok else "失败"
            observations.append(f"工具 {name} 执行{status}")

            if ctx.memory is not None:
                ctx.memory.skills.record_use(name, success=execution.ok)

        node_span.attributes["calls"] = len(executions)
        node_span.attributes["failed"] = sum(1 for item in executions if not item["ok"])

    return {
        "messages": messages,
        "tool_executions": executions,
        "tool_call_count": state.tool_call_count + len(executions),
        "pending_tool_calls": [],
        "observations": observations,
    }


def reflect(state: AgentState, ctx) -> dict[str, Any]:
    """检查回答是否有证据支撑；证据不足时请求重试。"""
    with ctx.tracer.span("reflect", "llm") as span:
        span.attributes["node"] = "reflect"
        evidence = "\n".join(
            f"- {item['tool_name']}: {item['result'][:200]}" for item in state.tool_executions[-4:]
        )
        if state.retrieved:
            evidence += "\n" + "\n".join(
                f"- 检索片段 {item['chunk']['title']}" for item in state.retrieved[:4]
            )

        response = ctx.llm.chat(
            [
                {"role": "system", "content": REFLECT_SYSTEM},
                {
                    "role": "user",
                    "content": f"问题：{state.question}\n\n回答：{state.answer}\n\n已有证据：\n{evidence or '（无）'}",
                },
            ]
        )
        verdict = _parse_json(response.content)

    should_retry = bool(verdict.get("should_retry")) and state.iteration < ctx.settings.agent.max_iterations
    span.attributes["sufficient"] = bool(verdict.get("sufficient"))
    span.attributes["should_retry"] = should_retry

    return {
        "reflected": True,
        "retry_requested": should_retry,
        "observations": [f"反思：{verdict.get('comment') or '已完成检查'}"],
        "notes": [f"反思结论：{json.dumps(verdict, ensure_ascii=False)[:200]}"],
    }


def finalize(state: AgentState, ctx) -> dict[str, Any]:
    """落库会话、沉淀长期记忆、标记任务完成。"""
    with ctx.tracer.span("finalize", "node") as span:
        span.attributes["node"] = "finalize"
        status = state.status if state.status != "running" else "completed"

        if ctx.memory is not None and state.conversation_id:
            ctx.memory.save_turn(state.conversation_id, state.messages)

        consolidated: list[str] = []
        if ctx.memory is not None and state.conversation_id and ctx.llm is not None:
            try:
                saved = ctx.memory.consolidate(state.conversation_id, ctx.llm)
                consolidated = [memory.content for memory in saved]
            except Exception as exc:  # noqa: BLE001 - 记忆沉淀失败不影响任务结果
                consolidated = [f"记忆沉淀失败：{exc}"]

        span.attributes["status"] = status
        span.attributes["messages"] = len(state.messages)

        observations = ["任务完成"] if status == "completed" else [f"任务结束：{status}"]
        if consolidated:
            observations.append(f"沉淀长期记忆 {len(consolidated)} 条")

        return {"status": status, "observations": observations}


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
def route_after_prepare(state: AgentState, ctx) -> str:
    if state.need_retrieval and ctx.knowledge is not None:
        return "retrieve"
    return "build_context"


def route_after_reason(state: AgentState, ctx) -> str:
    if state.pending_tool_calls:
        if state.tool_call_count >= ctx.settings.agent.max_tool_calls:
            return "finalize"
        return "tools"
    if state.answer:
        if ctx.settings.agent.enable_reflection and not state.reflected:
            return "reflect"
        return "finalize"
    return "finalize"


def route_after_tools(state: AgentState, ctx) -> str:
    if state.iteration >= ctx.settings.agent.max_iterations:
        return "finalize"
    return "build_context"


def route_after_reflect(state: AgentState, ctx) -> str:
    if state.retry_requested and state.iteration < ctx.settings.agent.max_iterations:
        return "build_context"
    return "finalize"


# --------------------------------------------------------------------------- #
def _parse_json(text: str) -> dict[str, Any]:
    """从模型输出里尽力解析 JSON（容忍 ```json 包裹与前后废话）。"""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1] if "```" in cleaned[3:] else cleaned[3:]
        cleaned = cleaned.removeprefix("json").strip()
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(cleaned[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}
