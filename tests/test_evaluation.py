"""评测层测试：指标计算、运行器、报告渲染。"""

from __future__ import annotations

import pytest

from contextforge.evaluation import (
    CONSTRAINTS,
    EvaluationCase,
    aggregate_runs,
    answer_metrics,
    build_long_history,
    context_metrics,
    cost_estimate,
    default_cases,
    markdown_table,
    render_agent_report,
    render_context_report,
    retrieval_metrics,
    run_agent_suite,
    run_case,
    run_context_experiment,
    save_report,
)


# --------------------------------------------------------------------------- #
# 检索指标
# --------------------------------------------------------------------------- #
def test_retrieval_metrics_perfect_ranking():
    metrics = retrieval_metrics(["a", "b"], ["a", "b"], k=2)
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg_at_k == pytest.approx(1.0)
    assert metrics.hit_rate == 1.0


def test_retrieval_metrics_partial_hit_affects_mrr():
    metrics = retrieval_metrics(["x", "a"], ["a"], k=2)
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.precision_at_k == pytest.approx(0.5)
    assert metrics.recall_at_k == 1.0


def test_retrieval_metrics_without_gold_is_zero():
    metrics = retrieval_metrics(["a"], [], k=1)
    assert metrics.recall_at_k == 0.0
    assert metrics.hit_rate == 0.0


def test_retrieval_metrics_does_not_double_count_same_document():
    """同一篇文档命中多个切片时，Recall@k 不能超过 1。"""
    metrics = retrieval_metrics(["a", "a", "a"], ["a"], k=3)
    assert metrics.recall_at_k == 1.0
    assert metrics.ndcg_at_k == pytest.approx(1.0)
    assert metrics.precision_at_k == pytest.approx(1.0)
    assert metrics.retrieved == 1


def test_retrieval_metrics_all_scores_are_bounded():
    for retrieved in (["a", "a", "b"], ["b", "b", "b"], ["x", "y", "z"]):
        metrics = retrieval_metrics(retrieved, ["a", "b"], k=3)
        assert 0.0 <= metrics.precision_at_k <= 1.0
        assert 0.0 <= metrics.recall_at_k <= 1.0
        assert 0.0 <= metrics.ndcg_at_k <= 1.0


# --------------------------------------------------------------------------- #
# 上下文指标
# --------------------------------------------------------------------------- #
def test_context_metrics_detects_dropped_constraint():
    metrics = context_metrics(
        "回答中包含了必须用中文这句话的改写版本",
        constraints=["必须用中文回答", "不要用英文长段落"],
        facts=["把 ContextForge 做成求职作品集项目"],
        tokens_before=1000,
        tokens_after=400,
    )
    assert metrics.compression_ratio == pytest.approx(0.4)
    assert 0 < metrics.constraint_retention < 1.0
    assert metrics.dropped_constraints
    assert metrics.missing_facts


def test_context_metrics_tolerates_rewording():
    """摘要难免改写措辞，不应因为少一个「的」就判为丢失。"""
    metrics = context_metrics(
        "结论：必须使用中文来回答",
        constraints=["必须用中文回答"],
        facts=[],
    )
    assert metrics.constraint_retention == 1.0


def test_context_metrics_all_retained():
    metrics = context_metrics(
        "必须用中文回答；所有工具调用结果都要写进执行轨迹",
        constraints=["必须用中文回答", "所有工具调用结果都要写进执行轨迹"],
        facts=[],
        tokens_before=100,
        tokens_after=100,
    )
    assert metrics.constraint_retention == 1.0
    assert metrics.compression_ratio == 1.0


# --------------------------------------------------------------------------- #
# 回答与运行指标
# --------------------------------------------------------------------------- #
def test_answer_metrics_counts_keywords_and_citations():
    metrics = answer_metrics("根据 [1] 与 [2]，压缩以轮次为单位，避免打断工具链。", ["轮次", "工具"])
    assert metrics.keyword_coverage == 1.0
    assert metrics.citation_count == 2
    assert metrics.has_citation is True
    assert metrics.missing == []


def test_answer_metrics_reports_missing():
    metrics = answer_metrics("只提到了一点", ["轮次", "工具"])
    assert metrics.missing == ["轮次", "工具"]
    assert metrics.keyword_coverage == 0.0


def test_cost_estimate():
    assert cost_estimate(1000, 1000, 0.001, 0.002) == pytest.approx(0.003)


def test_aggregate_runs_computes_averages():
    records = [
        {
            "status": "completed",
            "latency_ms": 100,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "iterations": 2,
            "tool_executions": [{"ok": True}],
        },
        {
            "status": "failed",
            "latency_ms": 300,
            "usage": {"prompt_tokens": 200, "completion_tokens": 0, "total_tokens": 200},
            "iterations": 1,
            "tool_executions": [{"ok": False}],
        },
    ]
    metrics = aggregate_runs(records)
    assert metrics.total == 2
    assert metrics.success_rate == 0.5
    assert metrics.avg_latency_ms == 200
    assert metrics.avg_total_tokens == 175
    assert metrics.tool_calls == 2
    assert metrics.tool_success_rate == 0.5


def test_aggregate_runs_handles_empty():
    metrics = aggregate_runs([])
    assert metrics.total == 0
    assert metrics.success_rate == 0.0


# --------------------------------------------------------------------------- #
# 报告渲染
# --------------------------------------------------------------------------- #
def test_markdown_table_formats_values():
    table = markdown_table(["名称", "分数"], [["a", 0.123456], ["b", {"x": 1}]])
    assert "| 名称 | 分数 |" in table
    assert "0.123" in table
    assert "x=1" in table


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
def test_default_cases_are_well_formed():
    cases = default_cases()
    assert len(cases) >= 5
    for case in cases:
        assert case.case_id and case.question
        assert isinstance(case.gold_docs, list)


def test_run_case_produces_metrics(runtime):
    case = EvaluationCase(
        case_id="x1",
        question="上下文压缩为什么要以轮次为单位？",
        gold_docs=["上下文工程指南.md"],
        expected_keywords=["工具"],
    )
    record = run_case(runtime, case)
    assert record["status"] == "completed"
    assert record["retrieval_metrics"] is not None
    assert record["retrieval_metrics"]["recall_at_k"] >= 0
    assert "usage" in record
    assert record["tool_match"] is True


def test_run_case_checks_expected_tools(runtime):
    case = EvaluationCase(
        case_id="x2", question="计算 3 * 4 等于多少？", expected_tools=["calculator"]
    )
    record = run_case(runtime, case)
    assert record["tool_match"] is True


def test_run_agent_suite_compares_retrieval_modes(runtime):
    cases = [
        EvaluationCase(
            case_id="c1",
            question="混合检索为什么要把向量和 BM25 融合？",
            gold_docs=["检索与重排序.md"],
            expected_keywords=["召回"],
        ),
        EvaluationCase(case_id="c2", question="计算 5 + 5", expected_tools=["calculator"]),
    ]
    report = run_agent_suite(runtime, cases=cases, retrieval_modes=["dense", "hybrid"])
    assert set(report["groups"]) == {"dense", "hybrid"}
    for group in report["groups"].values():
        assert group["cases"] == 2
        assert "runs" in group
        assert 0 <= group["answer_keyword_coverage"] <= 1
        assert 0 <= group["tool_match_rate"] <= 1
    assert len(report["records"]) == 4


def test_run_agent_suite_supports_strategy_variants(runtime):
    cases = [EvaluationCase(case_id="c1", question="上下文压缩有哪些策略？")]
    report = run_agent_suite(runtime, cases=cases, retrieval_modes=["hybrid"], strategies=["none", "hybrid"])
    assert set(report["groups"]) == {"hybrid+none", "hybrid+hybrid"}


def test_context_experiment_quantifies_compression_tradeoff(runtime):
    report = run_context_experiment(runtime, strategies=["none", "sliding", "summary", "hybrid"])
    rows = {row["strategy"]: row for row in report["rows"]}

    assert set(rows) == {"none", "sliding", "summary", "hybrid"}
    # 不压缩的策略保留全部内容
    assert rows["none"]["compression_ratio"] == pytest.approx(1.0)
    # 压缩策略确实更省
    assert rows["sliding"]["tokens_after"] < rows["none"]["tokens_after"]
    # 滑动窗口会丢失早期约束
    assert rows["sliding"]["constraint_retention"] < 1.0
    # 混合策略必须保住硬约束
    assert rows["hybrid"]["constraint_retention"] >= 0.99
    # 所有策略都不能破坏工具链
    assert all(row["tool_integrity_ok"] for row in report["rows"])
    # 行按压缩后 token 升序
    assert [row["tokens_after"] for row in report["rows"]] == sorted(
        row["tokens_after"] for row in report["rows"]
    )


def test_build_long_history_contains_constraints():
    history = build_long_history(turns=3)
    joined = "\n".join(str(message.get("content")) for message in history)
    for constraint in CONSTRAINTS:
        assert constraint in joined or "中文" in constraint
    assert len(history) >= 8


def test_render_reports(runtime):
    context_report = run_context_experiment(runtime, strategies=["none", "hybrid"], turns=4)
    agent_report = run_agent_suite(
        runtime,
        cases=[EvaluationCase(case_id="c1", question="上下文压缩有哪些策略？")],
        retrieval_modes=["hybrid"],
    )
    context_text = render_context_report(context_report)
    agent_text = render_agent_report(agent_report)
    assert "压缩策略" in context_text
    assert "压缩率" in context_text
    assert "检索策略对比" in agent_text


def test_save_report_writes_file(runtime, tmp_path):
    report = run_context_experiment(runtime, strategies=["hybrid"], turns=3)
    path = save_report(render_context_report(report), tmp_path / "nested" / "report.md")
    from pathlib import Path

    assert Path(path).exists()
    assert "压缩策略" in Path(path).read_text(encoding="utf-8")
