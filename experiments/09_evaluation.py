"""实验 09 · 评测：把「效果更好」变成一张可复核的表

## 这个实验要回答什么

求职作品集里最容易失效的一句话是「我做的 XX 提升了效果」。
在没有数字的情况下，这句话等价于「我做了 XX」。

本实验跑完整的评测套件，产出两类对比数据：

1. **检索策略对比**：同一组问题 × dense / sparse / hybrid 三种模式，
   比较 Precision@k、Recall@k、MRR、nDCG、要点覆盖率、工具命中率、token 成本与延迟；
2. **上下文压缩对比**：同一段长历史 × 五种压缩策略，
   比较压缩率、约束保留率、事实保留率、工具链完整性。

所有指标都是**本地确定性计算**，不依赖任何外部评测服务；
接 RAGAS / LangSmith 时只需把这些数值换成服务端指标，接口不变。
"""

from __future__ import annotations

from _lab import SAMPLE_DOCS, SAMPLE_REPO, conclusion, note, offline_runtime, step, title

from contextforge.evaluation.runner import (
    default_cases,
    render_agent_report,
    render_context_report,
    render_full_report,
    run_agent_suite,
    run_context_experiment,
    save_report,
)

OUTPUT = "data/report.md"


def main() -> None:
    title("实验 09 · 评测：把「效果更好」变成一张可复核的表")

    with offline_runtime(sample_docs=SAMPLE_DOCS, repo_root=SAMPLE_REPO) as runtime:
        cases = default_cases()

        step("第 1 步：用例集")
        note(f"共 {len(cases)} 个用例，覆盖 5 类任务：")
        by_category: dict[str, int] = {}
        for case in cases:
            by_category[case.category] = by_category.get(case.category, 0) + 1
        for category, count in sorted(by_category.items()):
            note(f"  {category:<16} {count} 个")
        note("每个用例都标注了「标准答案所在的文档」与「回答应覆盖的要点」，")
        note("因此检索指标与回答指标都能自动计算，不需要人工打分。")

        step("第 2 步：检索策略对比（同一组问题 × 三种检索模式）")
        agent_report = run_agent_suite(runtime, cases=cases, retrieval_modes=["dense", "sparse", "hybrid"])
        print()
        print(render_agent_report(agent_report))

        step("第 3 步：上下文压缩对比（同一段长历史 × 五种策略）")
        context_report = run_context_experiment(
            runtime,
            strategies=["none", "sliding", "summary", "hybrid", "llm-summary"],
            turns=30,
        )
        print()
        print(render_context_report(context_report))

        step("第 4 步：关键数字速览")
        groups = agent_report["groups"]
        best = max(groups.items(), key=lambda item: item[1].get("mrr", 0.0))
        note(
            f"MRR 最高的检索模式：{best[0]}（{best[1].get('mrr', 0):.3f}）"
        )
        for label, group in groups.items():
            note(
                f"  {label:<8} P@k={group.get('precision_at_k', 0):.3f}"
                f"  R@k={group.get('recall_at_k', 0):.3f}"
                f"  MRR={group.get('mrr', 0):.3f}"
                f"  nDCG={group.get('ndcg_at_k', 0):.3f}"
                f"  要点覆盖={group['answer_keyword_coverage']:.2f}"
                f"  工具命中={group['tool_match_rate']:.2f}"
                f"  平均 token={group['runs']['avg_total_tokens']:.0f}"
                f"  平均延迟={group['runs']['avg_latency_ms']:.1f}ms"
            )

        rows = {row["strategy"]: row for row in context_report["rows"]}
        note("")
        note("压缩策略的取舍（按 token 从省到贵）：")
        for row in context_report["rows"]:
            note(
                f"  {row['strategy']:<12} token={row['tokens_after']:>5}"
                f"  约束保留={row['constraint_retention']:.0%}"
                f"  事实保留={row['fact_retention']:.0%}"
                f"  工具链={'完整' if row['tool_integrity_ok'] else '断裂'}"
            )
        if "none" in rows and "hybrid" in rows:
            saved = 1 - rows["hybrid"]["tokens_after"] / rows["none"]["tokens_after"]
            note(
                f"最反直觉的一条：hybrid 比「不压缩」省了 {saved:.0%} 的 token，"
                f"约束保留率还从 {rows['none']['constraint_retention']:.0%}"
                f" 提到 {rows['hybrid']['constraint_retention']:.0%}。"
            )

        step("第 5 步：产出报告文件")
        stats = runtime.knowledge.stats()
        text = render_full_report(
            agent_report,
            context_report,
            metadata={
                "语料": f"示例文档 {stats['documents']} 篇 / {stats['chunks']} 个切片",
                "检索模式": "dense / sparse / hybrid",
                "压缩历史": f"{len(context_report['rows'])} 种策略 × 30 轮",
                "模型": "离线确定性模型（无需 API Key）",
            },
        )
        path = save_report(text, _abs(OUTPUT))
        note(f"完整评测报告已写入：{path}")
        note("报告里的每一个数字都可以通过重跑本实验复现 —— 这是它作为")
        note("「可复核的实验结论」而不是「一句自我评价」的关键。")

    conclusion(
        "评测的价值不在于得到一个好看的数字，而在于它**把取舍显式化**了："
        "哪一种检索模式更准、要付多少 token 和延迟；哪一种压缩策略更省、会丢什么。"
        "有了这张表，「我做了上下文工程」才会变成"
        "「滑动窗口省 80% token 但丢 67% 约束，混合策略在同样预算下一条不丢」。"
    )


def _abs(relative: str) -> str:
    from _lab import ROOT

    return str(ROOT / relative)


if __name__ == "__main__":
    main()
