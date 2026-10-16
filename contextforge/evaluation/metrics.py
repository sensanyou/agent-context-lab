"""评测指标。

分四组，对应四个必须分别回答的问题：

| 分组 | 回答的问题 | 指标 |
| --- | --- | --- |
| 检索 | 该找的资料找到了吗？ | Precision@k、Recall@k、MRR、nDCG@k、命中率 |
| 上下文 | 压缩之后关键信息还在吗？ | 压缩率、约束保留率、事实保留率 |
| 回答 | 回答是否覆盖了要点？ | 要点覆盖率、引用完整率 |
| 运行 | 代价多大？ | 成功率、平均延迟、token、成本 |

所有指标都是**确定性计算**，不依赖外部评测服务；接上 RAGAS / LangSmith 时
只需把这些数值替换为对应的服务端指标，接口不变。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

# --------------------------------------------------------------------------- #
# 检索指标
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalMetrics:
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0
    k: int = 0
    retrieved: int = 0
    relevant_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "hit_rate": round(self.hit_rate, 4),
            "k": self.k,
            "retrieved": self.retrieved,
            "relevant_total": self.relevant_total,
        }


def _dedupe_preserving_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def retrieval_metrics(
    retrieved_ids: Sequence[str],
    gold_ids: Sequence[str],
    k: int | None = None,
) -> RetrievalMetrics:
    """按**文档**粒度计算检索指标。

    注意 `retrieved_ids` 是**切片级**的标签：同一篇文档会有多个切片同时命中。
    而指标定义在文档粒度上，所以必须先按出现顺序去重 —— 否则一篇文档命中 3 个切片，
    Recall@k 会被算成 3.0（越相关越"超标"，显然不成立）。
    本项目早期就踩过这个坑，因此这里显式去重并附上回归测试。
    """
    gold = set(gold_ids)
    cutoff = k or len(retrieved_ids)
    top = _dedupe_preserving_order(list(retrieved_ids)[:cutoff])
    metrics = RetrievalMetrics(k=cutoff, retrieved=len(top), relevant_total=len(gold))

    if not gold:
        return metrics

    hits = [1 if doc_id in gold else 0 for doc_id in top]

    metrics.precision_at_k = sum(hits) / len(top) if top else 0.0
    metrics.recall_at_k = min(sum(hits) / len(gold), 1.0)

    for rank, hit in enumerate(hits, start=1):
        if hit:
            metrics.mrr = 1.0 / rank
            break

    metrics.hit_rate = 1.0 if any(hits) else 0.0

    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(gold), len(top))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    metrics.ndcg_at_k = min(dcg / idcg, 1.0) if idcg else 0.0
    return metrics


# --------------------------------------------------------------------------- #
# 上下文指标
# --------------------------------------------------------------------------- #
@dataclass
class ContextMetrics:
    compression_ratio: float = 1.0
    tokens_before: int = 0
    tokens_after: int = 0
    constraint_retention: float = 1.0
    fact_retention: float = 1.0
    dropped_constraints: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compression_ratio": round(self.compression_ratio, 4),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "constraint_retention": round(self.constraint_retention, 4),
            "fact_retention": round(self.fact_retention, 4),
            "dropped_constraints": self.dropped_constraints,
            "missing_facts": self.missing_facts,
        }


def context_metrics(
    built_text: str,
    constraints: Sequence[str] = (),
    facts: Sequence[str] = (),
    tokens_before: int = 0,
    tokens_after: int = 0,
) -> ContextMetrics:
    """检查压缩之后关键信息是否还在。

    `constraints` 是必须原样保留的硬约束（例如「必须用中文回答」），
    `facts` 是应该被摘要覆盖的事实（例如「长期目标是做 Agent 开发」）。
    """
    metrics = ContextMetrics(
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        compression_ratio=(tokens_after / tokens_before) if tokens_before else 1.0,
    )

    for constraint in constraints:
        if not _contains(built_text, constraint):
            metrics.dropped_constraints.append(constraint)

    for fact in facts:
        if not _contains(built_text, fact):
            metrics.missing_facts.append(fact)

    metrics.constraint_retention = (
        1.0 - len(metrics.dropped_constraints) / len(constraints) if constraints else 1.0
    )
    metrics.fact_retention = 1.0 - len(metrics.missing_facts) / len(facts) if facts else 1.0
    return metrics


# --------------------------------------------------------------------------- #
# 回答指标
# --------------------------------------------------------------------------- #
@dataclass
class AnswerMetrics:
    keyword_coverage: float = 0.0
    citation_count: int = 0
    has_citation: bool = False
    length: int = 0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keyword_coverage": round(self.keyword_coverage, 4),
            "citation_count": self.citation_count,
            "has_citation": self.has_citation,
            "length": self.length,
            "matched": self.matched,
            "missing": self.missing,
        }


def answer_metrics(answer: str, expected_keywords: Sequence[str] = ()) -> AnswerMetrics:
    text = answer or ""
    matched = [keyword for keyword in expected_keywords if _contains(text, keyword)]
    missing = [keyword for keyword in expected_keywords if keyword not in matched]
    citations = re.findall(r"\[(\d+)\]", text)

    return AnswerMetrics(
        keyword_coverage=(len(matched) / len(expected_keywords)) if expected_keywords else 0.0,
        citation_count=len(citations),
        has_citation=bool(citations),
        length=len(text),
        matched=matched,
        missing=missing,
    )


# --------------------------------------------------------------------------- #
# 运行指标
# --------------------------------------------------------------------------- #
def cost_estimate(
    prompt_tokens: int,
    completion_tokens: int,
    price_in_per_1k: float = 0.001,
    price_out_per_1k: float = 0.002,
) -> float:
    return round(
        prompt_tokens / 1000 * price_in_per_1k + completion_tokens / 1000 * price_out_per_1k, 6
    )


@dataclass
class RunMetrics:
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    avg_total_tokens: float = 0.0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    total_cost: float = 0.0
    avg_iterations: float = 0.0
    tool_calls: int = 0
    tool_success_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_total_tokens": round(self.avg_total_tokens, 2),
            "avg_prompt_tokens": round(self.avg_prompt_tokens, 2),
            "avg_completion_tokens": round(self.avg_completion_tokens, 2),
            "total_cost": round(self.total_cost, 6),
            "avg_iterations": round(self.avg_iterations, 2),
            "tool_calls": self.tool_calls,
            "tool_success_rate": round(self.tool_success_rate, 4),
        }


def aggregate_runs(records: Sequence[dict[str, Any]], price_in: float = 0.001, price_out: float = 0.002) -> RunMetrics:
    """把一批用例记录聚合成运行指标。"""
    metrics = RunMetrics(total=len(records))
    if not records:
        return metrics

    latencies: list[float] = []
    tokens: list[float] = []
    prompts: list[float] = []
    completions: list[float] = []
    iterations: list[float] = []
    tool_total = 0
    tool_ok = 0

    for record in records:
        if record.get("status") in ("completed", "ok"):
            metrics.succeeded += 1
        else:
            metrics.failed += 1

        latencies.append(float(record.get("latency_ms") or 0.0))
        usage = record.get("usage") or {}
        prompts.append(float(usage.get("prompt_tokens", 0)))
        completions.append(float(usage.get("completion_tokens", 0)))
        tokens.append(float(usage.get("total_tokens", 0)))
        iterations.append(float(record.get("iterations") or 0))

        for execution in record.get("tool_executions") or []:
            tool_total += 1
            if execution.get("ok"):
                tool_ok += 1

        metrics.total_cost += cost_estimate(
            int(prompts[-1]), int(completions[-1]), price_in, price_out
        )

    count = len(records)
    metrics.success_rate = metrics.succeeded / count
    metrics.avg_latency_ms = sum(latencies) / count
    metrics.avg_total_tokens = sum(tokens) / count
    metrics.avg_prompt_tokens = sum(prompts) / count
    metrics.avg_completion_tokens = sum(completions) / count
    metrics.avg_iterations = sum(iterations) / count
    metrics.tool_calls = tool_total
    metrics.tool_success_rate = (tool_ok / tool_total) if tool_total else 1.0
    return metrics


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not headers:
        return ""
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "、".join(str(item) for item in value)
    return str(value)


_NORMALIZE_PATTERN = re.compile(r"""[\s，。、；：！？,.!?;:"'“”‘’（）()\[\]【】《》〈〉的]+""")

#: 长约束允许的改写容忍度：低于该覆盖比例即视为「已被压缩掉」
_FUZZY_THRESHOLD = 0.7
#: 短词不做模糊匹配 —— 两三个字的词在整个文本里必然能找到同序字符，会大量误判
_FUZZY_MIN_LEN = 6


def _normalize(text: str) -> str:
    """去掉空白、常见标点与虚词「的」。"""
    return _NORMALIZE_PATTERN.sub("", text or "")


def _subsequence_ratio(needle: str, haystack: str) -> float:
    """needle 中有多少比例的字符能**按原顺序**在 haystack 里找到。

    这里刻意用「有序子序列覆盖率」而不是编辑距离：
    摘要改写时通常是**插入**成分（「必须用中文回答」→「结论：必须使用中文来回答」），
    编辑距离会被插入进来的长文本稀释，而覆盖率只关心约束本身还在不在。
    """
    if not needle:
        return 1.0
    cursor = 0
    for char in haystack:
        if char == needle[cursor]:
            cursor += 1
            if cursor == len(needle):
                return 1.0
    return cursor / len(needle)


def _contains(haystack: str, needle: str) -> bool:
    """宽松包含判断：忽略空白与常见标点差异，并容忍摘要的措辞改写。"""
    if not needle:
        return True
    compact_needle = _normalize(needle)
    if not compact_needle:
        return True
    compact_haystack = _normalize(haystack)
    if compact_needle in compact_haystack:
        return True
    if len(compact_needle) < _FUZZY_MIN_LEN:
        return False
    return _subsequence_ratio(compact_needle, compact_haystack) >= _FUZZY_THRESHOLD
