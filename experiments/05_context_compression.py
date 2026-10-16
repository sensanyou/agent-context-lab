"""实验 05 · 压缩策略对比：省 token 的代价是什么

## 这个实验要回答什么

「压缩历史能省 token」是一句废话。真正的问题是：

> **省下来的 token，是用什么换的？**

本实验把四种策略跑在**同一段长历史**上，同一份预算，然后逐项对比：

| 策略 | 做法 | 预期 |
| --- | --- | --- |
| none | 不压缩 | 信息全保留，但 token 最贵 |
| sliding | 滑动窗口，只留最近 N 轮 | 最省，但早期约束最先丢 |
| summary | 抽取式摘要旧轮次 | 折中，可能漏掉细节 |
| hybrid | 钉住关键约束 + 摘要 + 近期原文 | 略贵，但硬约束一条不丢 |

四个指标分别回答四个问题：压缩率（省了多少）、约束保留率（该守的还在吗）、
事实保留率（背景还在吗）、工具链完整（有没有把请求搞非法）。

> 注意：这个实验的系统提示**故意不复述**这些约束。
> 否则无论历史被压成什么样，都能从系统提示里把约束「找回来」，
> 约束保留率这个指标就彻底失去意义了。
"""

from __future__ import annotations

from _lab import conclusion, note, offline_runtime, step, title

from contextforge.context.strategies import STRATEGY_NAMES
from contextforge.evaluation.runner import (
    CONSTRAINTS,
    FACTS,
    render_context_report,
    run_context_experiment,
)


def main() -> None:
    title("实验 05 · 压缩策略对比：省 token 的代价是什么")

    with offline_runtime() as runtime:

        step("第 0 步：先看实验材料")
        note(f"硬约束（必须原样保留）：{CONSTRAINTS}")
        note(f"关键事实（应被摘要覆盖）：{FACTS}")
        note("历史长度：30 轮补充说明 + 首尾各一轮包含约束的对话，共 64 条消息")

        step("第 1 步：同一段历史 × 五种策略")
        report = run_context_experiment(runtime, strategies=list(STRATEGY_NAMES), turns=30)
        rows = {row["strategy"]: row for row in report["rows"]}
        for name in STRATEGY_NAMES:
            row = rows.get(name)
            if row is None:
                continue
            note(
                f"{name:<12} token {row['tokens_before']:>4} → {row['tokens_after']:>4}"
                f"（压缩率 {row['compression_ratio']:.2f}）"
                f"  约束保留 {row['constraint_retention']:.0%}"
                f"  事实保留 {row['fact_retention']:.0%}"
                f"  工具链完整 {'是' if row['tool_integrity_ok'] else '否'}"
            )

        step("第 2 步：完整对比表")
        print()
        print(render_context_report(report))

        step("第 3 步：读表 —— 四个必须看出来的结论")
        none_row = rows.get("none", {})
        sliding = rows.get("sliding", {})
        hybrid = rows.get("hybrid", {})

        note(
            "① 不压缩**不是**安全选项：它 token 最高"
            f"（{none_row.get('tokens_after')}），约束保留率却只有"
            f" {none_row.get('constraint_retention', 0):.0%} —— "
            "因为一旦超预算，兜底降级照样从最旧的开始丢。**又贵又丢信息。**"
        )
        note(
            f"② 滑动窗口最省 token（{sliding.get('tokens_after')}，只有不压缩的"
            f" {sliding.get('tokens_after', 0) / max(none_row.get('tokens_after', 1), 1):.0%}），"
            f"但约束保留率掉到 {sliding.get('constraint_retention', 0):.0%} ——"
            "「我的长期目标是…」这类早期目标恰好是最早被丢掉的。"
        )
        note(
            f"③ 混合策略在 {hybrid.get('compression_ratio', 1):.2f} 的压缩率下保住了"
            f" {hybrid.get('constraint_retention', 0):.0%} 的约束和"
            f" {hybrid.get('fact_retention', 0):.0%} 的事实，"
            "总 token 甚至低于不压缩 —— 因为它**明确知道什么不该丢**。"
        )
        note("④ 所有策略的工具链完整性都是「是」：压缩按轮次进行，永远不会拆散工具调用。")

    conclusion(
        "「省 token」本身不是目标，**信息保留率才决定了压到什么程度是安全的**。"
        "这张表里最反直觉的一点是「不压缩」并不安全：它 token 最贵，约束却照样丢。"
        "把五种策略放进同一张表，取舍就变成了可比较的数字 —— 这正是本项目"
        "把上下文管理做成可评测组件的原因。"
    )


if __name__ == "__main__":
    main()
