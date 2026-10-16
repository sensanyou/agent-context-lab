"""实验 07 · 图编排：工具调用为什么必须回到「重新组装上下文」

## 这个实验要回答什么

很多人以为 Agent 循环是这样：

    用户提问 → 模型回答 → 如果要用工具就调一下 → 把结果贴回去 → 结束

真实情况是：**每一次工具返回，都必须重新走一遍上下文装配**。
原因很直接 —— 工具结果会占用 token，可能把上下文顶爆；
只有重新装配，预算才能重新分配、历史才能重新压缩。

本实验把这条链路显式地画出来，并验证三件事：

1. 一次带工具调用的任务，`build_context` 节点**至少被执行两次**；
2. 完整执行轨迹（每个节点的耗时、工具调用、token 用量）被记录下来；
3. 危险工具可以配置成「执行前需要人工确认」，并且能被恢复继续跑。
"""

from __future__ import annotations

from _lab import SAMPLE_DOCS, SAMPLE_REPO, conclusion, kv_table, note, offline_runtime, step, title

from contextforge.graph.workflow import workflow_diagram


def main() -> None:
    title("实验 07 · 图编排：工具调用为什么必须回到「重新组装上下文」")

    with offline_runtime(sample_docs=SAMPLE_DOCS, repo_root=SAMPLE_REPO) as runtime:
        step("第 1 步：先把工作流本身画出来")
        print()
        print(workflow_diagram(runtime.settings))
        print()
        note("节点注册在 StateGraph 上，条件边决定下一步去哪 —— 这正是 LangGraph 的抽象。")
        note("项目自研这一层，是为了不引入框架依赖也能跑通同样的语义；")
        note("接真实 LangGraph 时，节点函数和状态定义可以原样搬迁。")

        step("第 2 步：跑一个必须用工具的任务，看它绕了几圈")
        result = runtime.ask("计算 (128 + 72) * 3 等于多少？")
        nodes = [
            event["node"] for event in result.events if event.get("type") == "node_end"
        ]
        note(f"问题：计算 (128 + 72) * 3 等于多少？")
        note(f"回答：{' '.join(result.answer.split())[:60]}")
        note(f"节点执行序列：{' → '.join(nodes)}")
        note(
            f"build_context 执行了 {nodes.count('build_context')} 次，"
            f"reason 执行了 {nodes.count('reason')} 次"
        )
        note("第一次 build_context 是为了「决定要不要调工具」；")
        note("第二次 build_context 把工具结果纳入上下文，再让模型给出最终答案。")

        step("第 3 步：工具执行情况")
        for execution in result.tool_executions:
            note(
                f"  {execution.tool_name}({execution.arguments}) → {execution.result[:40]}"
                f"  ok={execution.ok}  耗时={execution.latency_ms:.2f}ms"
            )
        note(f"总迭代次数：{result.iterations}")

        step("第 4 步：上下文报告 —— 这一轮到底发了什么给模型")
        report = result.context_report
        if report:
            note(f"策略：{report.strategy}")
            note(f"分区额度：{report.allocation}")
            note(f"分区实际占用：{report.sections}")
            note(f"token：{report.tokens_before} → {report.tokens_after}"
                 f"（压缩率 {report.compression_ratio:.2f}）")
            note(f"保留轮数 {report.kept_turns}，丢弃轮数 {report.dropped_turns}")
            note(f"工具链完整性：{'通过' if report.tool_integrity_ok else '不通过'}")
            for item in report.notes[:4]:
                note(f"引擎备注：{item}")

        step("第 5 步：换个知识类问题，看检索是怎么进入上下文的")
        kb_result = runtime.ask("混合检索为什么要把向量检索和 BM25 融合起来？")
        note(f"回答：{' '.join(kb_result.answer.split())[:70]}")
        note(f"引用条数：{len(kb_result.citations)}")
        for citation in kb_result.citations[:3]:
            note(f"  [{citation.label}] score={citation.score}  {citation.snippet[:40]}…")

        step("第 6 步：工具执行前的人工确认（human-in-the-loop）")
        note("配置 require_approval=True 时，图会在 tools 节点前插入一个中断点：")
        note("状态与上下文已经写入 SQLite 检查点，进程可以退出，之后再恢复。")
        captured: list[dict] = []
        paused = runtime.ask(
            "现在几点了？",
            require_approval=True,
            thread_id="lab-approval",
            on_event=captured.append,
        )
        note(f"任务状态：{paused.status}")
        interrupted = [event for event in captured if event.get("type") == "interrupt"]
        for event in interrupted:
            note(f"在节点 {event['node']} 执行前暂停")
        note(f"暂停前已完成的节点：{[e['node'] for e in captured if e.get('type') == 'node_end']}")
        if paused.status == "interrupted":
            resumed = runtime.resume("lab-approval")
            note(f"人工批准 → 恢复执行，最终状态：{resumed.status}")
            note(f"恢复后的回答：{' '.join((resumed.answer or '').split())[:56]}")
            note(f"恢复后执行的工具：{[e.tool_name for e in resumed.tool_executions]}")
        else:
            note("本次未产生中断（require_approval 只在 tools 节点生效，需要模型确实要调工具）。")

        step("第 7 步：可观测性汇总")
        stats = runtime.observability()
        note(f"轨迹聚合指标：{stats.get('traces', {})}")
        for item in stats.get("recent", [])[:4]:
            usage = item.get("usage") or {}
            nodes_path = item.get("nodes") or []
            note(
                f"  {str(item.get('question', ''))[:24]:<26}"
                f" 状态={item.get('status', '—'):<8}"
                f" 节点数={len(nodes_path):<3}"
                f" token={usage.get('total_tokens', 0):<5}"
                f" 耗时={item.get('duration_ms', 0):.1f}ms"
            )
            note(f"       路径：{' → '.join(nodes_path)}")
        note("每条轨迹记录了节点路径、各节点耗时、工具调用与 token 用量，")
        note("这些数据正是评测层计算「成功率 / 平均延迟 / 成本」的来源。")

        print()
        print(
            kv_table(
                ["观察点", "结果"],
                [
                    ["build_context 执行次数", f"{nodes.count('build_context')} 次"],
                    ["工具调用", f"{len(result.tool_executions)} 次"],
                    ["上下文重建", "工具返回后重新装配"],
                    ["人工确认中断", "是" if interrupted else "否"],
                ],
            )
        )

    conclusion(
        "Agent 循环不是「模型 → 工具 → 模型」这么简单："
        "**每一次工具返回都要重新装配上下文**。"
        "把这条流程显式建模成图（节点 + 条件边 + 检查点），换来的是可观测、"
        "可中断、可恢复 —— 这些在真实 Agent 里往往比「跑得通」更重要。"
    )


if __name__ == "__main__":
    main()
