"""评测运行器：把「不同策略到底差多少」变成一张可对比的表。

两类实验，分别回答两个不同的问题：

1. **检索策略对比**（`run_agent_suite`）
   同一组问题，分别用 `dense` / `sparse` / `hybrid` 三种检索模式跑完整 Agent，
   比较检索指标、回答质量、token 成本与延迟。

2. **压缩策略对比**（`run_context_experiment`）
   同一段长历史，分别用 `none` / `sliding` / `summary` / `hybrid` 压缩，
   比较压缩率、**硬约束保留率**与**关键事实保留率**。
   这是项目最有说服力的一组数据：它量化了「省 token」的代价是什么。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..context import ContextEngine, ContextRequest
from ..context.strategies import STRATEGY_NAMES
from ..types import Message
from .metrics import (
    aggregate_runs,
    answer_metrics,
    context_metrics,
    markdown_table,
    retrieval_metrics,
)

# --------------------------------------------------------------------------- #
# 用例
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationCase:
    case_id: str
    question: str
    gold_docs: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    category: str = "general"


def default_cases() -> list[EvaluationCase]:
    """默认用例集，针对 `examples/sample_docs` 设计。

    `gold_docs` 使用文档标题（即文件名），与检索结果里的 `chunk.title` 对齐。
    """
    return [
        EvaluationCase(
            case_id="ctx-01",
            question="上下文压缩的四种策略分别是什么？",
            gold_docs=["上下文工程指南.md"],
            expected_keywords=["滑动窗口", "摘要", "混合"],
            category="context",
        ),
        EvaluationCase(
            case_id="ctx-02",
            question="为什么上下文压缩必须以轮次为单位？",
            gold_docs=["上下文工程指南.md"],
            expected_keywords=["工具", "完整性", "轮次"],
            category="context",
        ),
        EvaluationCase(
            case_id="rag-01",
            question="混合检索为什么要把向量检索和 BM25 融合起来？",
            gold_docs=["检索与重排序.md"],
            expected_keywords=["精确词", "召回", "重叠"],
            category="retrieval",
        ),
        EvaluationCase(
            case_id="rag-02",
            question="RRF 融合相比加权融合有什么优势？",
            gold_docs=["检索与重排序.md"],
            expected_keywords=["量纲", "鲁棒", "排名"],
            category="retrieval",
        ),
        EvaluationCase(
            case_id="graph-01",
            question="Agent 工作流里为什么需要循环和条件分支？",
            gold_docs=["Agent编排.md"],
            expected_keywords=["工具", "重试", "状态"],
            category="orchestration",
        ),
        EvaluationCase(
            case_id="mem-01",
            question="长期记忆和会话记忆有什么区别？",
            gold_docs=["记忆技能与评测.md"],
            expected_keywords=["跨任务", "结论", "生命周期"],
            category="memory",
        ),
        EvaluationCase(
            case_id="tool-01",
            question="计算 (128 + 72) * 3 等于多少？",
            gold_docs=[],
            expected_keywords=["600"],
            expected_tools=["calculator"],
            category="tool",
        ),
    ]


# --------------------------------------------------------------------------- #
# 单用例
# --------------------------------------------------------------------------- #
def run_case(
    runtime,
    case: EvaluationCase,
    strategy: str = "",
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
) -> dict[str, Any]:
    """跑一个用例，产出该用例的全部指标。"""
    retrieval = None
    if runtime.knowledge is not None and runtime.knowledge.stats()["documents"]:
        started = time.perf_counter()
        hits = runtime.knowledge.search(case.question, top_k=top_k, mode=retrieval_mode)
        retrieval_latency = (time.perf_counter() - started) * 1000
        retrieved_labels = [item.chunk.title or item.chunk.source for item in hits]
        retrieval = retrieval_metrics(retrieved_labels, case.gold_docs, k=top_k).to_dict()
        retrieval["latency_ms"] = round(retrieval_latency, 2)

    result = runtime.ask(case.question, strategy=strategy, retrieval_mode=retrieval_mode)
    answer = answer_metrics(result.answer, case.expected_keywords)
    used_tools = [execution.tool_name for execution in result.tool_executions]

    record: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "strategy": strategy or runtime.settings.context.strategy,
        "retrieval_mode": retrieval_mode,
        "status": result.status,
        "answer": result.answer,
        "answer_metrics": answer.to_dict(),
        "retrieval_metrics": retrieval,
        "latency_ms": round(result.latency_ms, 2),
        "usage": result.usage.as_dict,
        "iterations": result.iterations,
        "tool_executions": [execution.to_dict() for execution in result.tool_executions],
        "expected_tools": case.expected_tools,
        "tool_match": (
            set(case.expected_tools).issubset(set(used_tools)) if case.expected_tools else True
        ),
        "context_report": result.context_report.to_dict() if result.context_report else None,
        "citations": [citation.to_dict() for citation in result.citations],
        "trace_id": result.trace_id,
    }
    return record


# --------------------------------------------------------------------------- #
# 套件
# --------------------------------------------------------------------------- #
def run_agent_suite(
    runtime,
    cases: Sequence[EvaluationCase] | None = None,
    retrieval_modes: Sequence[str] = ("dense", "sparse", "hybrid"),
    strategies: Sequence[str] = (),
) -> dict[str, Any]:
    """跨检索模式 / 压缩策略跑完整 Agent，并聚合指标。"""
    cases = list(cases or default_cases())
    modes = list(retrieval_modes) or ["hybrid"]
    strategy_variants = list(strategies) or [""]

    records: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}

    for strategy in strategy_variants:
        for mode in modes:
            label = f"{mode}" + (f"+{strategy}" if strategy else "")
            group_records: list[dict[str, Any]] = []

            for case in cases:
                record = run_case(runtime, case, strategy=strategy, retrieval_mode=mode)
                record["group"] = label
                records.append(record)
                group_records.append(record)

            groups[label] = _summarize(group_records)

    return {
        "cases": len(cases),
        "groups": groups,
        "records": records,
        "generated_at": time.time(),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    run_metrics = aggregate_runs(records)
    retrieval_scores = [
        record["retrieval_metrics"]
        for record in records
        if record.get("retrieval_metrics")
    ]
    answer_scores = [record["answer_metrics"]["keyword_coverage"] for record in records]
    tool_matches = [1.0 if record.get("tool_match") else 0.0 for record in records]
    context_ratios = [
        record["context_report"]["compression_ratio"]
        for record in records
        if record.get("context_report")
    ]

    summary: dict[str, Any] = {
        "runs": run_metrics.to_dict(),
        "cases": len(records),
        "answer_keyword_coverage": round(sum(answer_scores) / len(answer_scores), 4)
        if answer_scores
        else 0.0,
        "tool_match_rate": round(sum(tool_matches) / len(tool_matches), 4) if tool_matches else 0.0,
        "avg_compression_ratio": round(sum(context_ratios) / len(context_ratios), 4)
        if context_ratios
        else 1.0,
    }

    if retrieval_scores:
        for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "hit_rate"):
            summary[key] = round(
                sum(item[key] for item in retrieval_scores) / len(retrieval_scores), 4
            )
    return summary


# --------------------------------------------------------------------------- #
# 压缩策略实验
# --------------------------------------------------------------------------- #
def build_long_history(turns: int = 12, filler: str = "") -> list[Message]:
    """构造一段包含硬约束与关键事实的长历史。"""
    history: list[Message] = []
    history.append(
        {
            "role": "user",
            "content": "我的长期目标是把 ContextForge 做成求职作品集项目，必须用中文回答，不要用英文长段落。",
        }
    )
    history.append({"role": "assistant", "content": "好的，我会遵守这些要求。"})

    for index in range(turns):
        history.append(
            {
                "role": "user",
                "content": (
                    f"第 {index} 轮补充说明：我在学习 Agent 上下文管理，"
                    f"希望在预算内保留有效信息。{filler}"
                ),
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"第 {index} 轮回复：上下文预算需要分区分配，"
                    f"历史压缩要以轮次为单位以免打断工具调用链。{filler}"
                ),
            }
        )

    history.append(
        {
            "role": "user",
            "content": "还有一个必须遵守的要求：所有工具调用结果都要写进执行轨迹，不要省略。",
        }
    )
    history.append({"role": "assistant", "content": "明白，工具调用结果会完整记录。"})
    return history


CONSTRAINTS = [
    "必须用中文回答",
    "不要用英文长段落",
    "所有工具调用结果都要写进执行轨迹",
]
FACTS = [
    "把 ContextForge 做成求职作品集项目",
    "上下文预算需要分区分配",
]


def run_context_experiment(
    runtime,
    strategies: Sequence[str] = ("none", "sliding", "summary", "hybrid", "llm-summary"),
    turns: int = 12,
    filler: str = "",
    budget: int | None = None,
) -> dict[str, Any]:
    """同一段长历史 × 不同压缩策略，量化「省 token 的代价」。"""
    history = build_long_history(turns=turns, filler=filler)
    settings = runtime.settings.context
    if budget is not None:
        settings.token_budget = budget

    engine = ContextEngine(settings, llm=runtime.llm)
    rows: list[dict[str, Any]] = []

    for name in strategies:
        if name not in STRATEGY_NAMES:
            continue
        bundle = engine.build(
            ContextRequest(
                # 系统提示刻意**不复述** CONSTRAINTS：那些约束只存在于历史里。
                # 否则无论历史被压成什么样，都能从系统提示里找回约束，
                # 「约束保留率」这个指标就失去意义了 —— 实验要度量的正是
                # 「历史压缩会不会把约束压掉」。
                system_prompt="你是 ContextForge 助手，请在预算内完成上下文装配并给出简洁回答。",
                user_input="请总结上面讨论的要点。",
                history=history,
                strategy=name,
            )
        )
        text = "\n".join(str(message.get("content") or "") for message in bundle.messages)
        metrics = context_metrics(
            text,
            constraints=CONSTRAINTS,
            facts=FACTS,
            tokens_before=bundle.report.tokens_before,
            tokens_after=bundle.report.tokens_after,
        )
        rows.append(
            {
                "strategy": name,
                "tokens_before": bundle.report.tokens_before,
                "tokens_after": bundle.report.tokens_after,
                "messages": len(bundle.messages),
                "kept_turns": bundle.report.kept_turns,
                "dropped_turns": bundle.report.dropped_turns,
                "tool_integrity_ok": bundle.report.tool_integrity_ok,
                **metrics.to_dict(),
            }
        )

    rows.sort(key=lambda row: row["tokens_after"])
    return {"budget": settings.token_budget, "turns": turns, "rows": rows}


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def render_agent_report(report: dict[str, Any]) -> str:
    headers = [
        "配置",
        "检索 P@k",
        "检索 R@k",
        "MRR",
        "要点覆盖",
        "工具命中",
        "压缩比",
        "平均 token",
        "平均延迟(ms)",
        "成本",
    ]
    rows = []
    for label, group in report["groups"].items():
        runs = group["runs"]
        rows.append(
            [
                label,
                group.get("precision_at_k", 0.0),
                group.get("recall_at_k", 0.0),
                group.get("mrr", 0.0),
                group["answer_keyword_coverage"],
                group["tool_match_rate"],
                group["avg_compression_ratio"],
                runs["avg_total_tokens"],
                runs["avg_latency_ms"],
                runs["total_cost"],
            ]
        )
    return "### 检索策略对比（同一组问题）\n\n" + markdown_table(headers, rows)


def render_context_report(report: dict[str, Any]) -> str:
    headers = [
        "压缩策略",
        "压缩前 token",
        "压缩后 token",
        "压缩率",
        "约束保留率",
        "事实保留率",
        "保留轮数",
        "工具链完整",
    ]
    rows = []
    for row in report["rows"]:
        rows.append(
            [
                row["strategy"],
                row["tokens_before"],
                row["tokens_after"],
                row["compression_ratio"],
                row["constraint_retention"],
                row["fact_retention"],
                row["kept_turns"],
                "是" if row["tool_integrity_ok"] else "否",
            ]
        )
    return (
        f"### 压缩策略对比（预算 {report['budget']} tokens，历史 {report['turns']} 轮）\n\n"
        + markdown_table(headers, rows)
    )


def render_full_report(
    agent_report: dict[str, Any],
    context_report: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """渲染完整评测报告。

    CLI 的 `bench` 与 `experiments/09_evaluation.py` 都走这个函数，
    避免两条路径产出格式不一致的报告（这是很容易发生、又很难发现的分叉）。
    """
    info = dict(metadata or {})
    info.setdefault("用例数量", agent_report.get("cases", 0))
    info.setdefault(
        "生成时间",
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(agent_report.get("generated_at") or time.time())),
    )

    header = ["# ContextForge 评测报告", ""]
    header.extend(f"- {key}：{value}" for key, value in info.items())

    return "\n\n".join(
        [
            "\n".join(header),
            render_context_report(context_report),
            render_agent_report(agent_report),
            "\n".join(
                [
                    "> 说明：所有指标由本地确定性计算得到，不依赖外部评测服务。",
                    "> 离线模型下「要点覆盖」反映的是检索与上下文装配的质量，而非文案质量。",
                    "> 离线向量为 hashing 表示，只做字面级匹配，因此三种检索模式的差距会被压低；",
                    "> 接入真实 embedding 后重跑本命令即可得到更有区分度的对比。",
                ]
            ),
        ]
    )


def save_report(text: str, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)
