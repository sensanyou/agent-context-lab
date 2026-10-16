"""实验 03 · Token 计数与分区预算

## 这个实验要回答什么

「上下文窗口快满了」这句话，需要变成可计算的数字才谈得上工程。

常见的朴素实现会做两件不靠谱的事：
1. 用**字符数**近似 token 数（中文里 1 个字符接近 1 个 token，英文里 4 个字符才 1 个 token，误差很大）；
2. 只有**一个总阈值**，任何一段超支都只能整体砍，导致「该保的被砍，该砍的留着」。

本实验对应两个升级：

* **可插拔计数器**：tiktoken → transformers → 启发式，任何一级不可用就静默降级，
  并把「我用的是哪一级」如实报出来；
* **分区预算**：system / 长期记忆 / 检索片段 / 历史 / 工具 / 当前问题 各有额度，
  某一段超支只在**它自己的额度内**裁剪，不会挤占别人。
"""

from _lab import conclusion, kv_table, note, step, title

from contextforge.config import ContextSettings
from contextforge.context.budget import (
    SECTION_HISTORY,
    SECTION_SYSTEM,
    BudgetAllocator,
)
from contextforge.context.tokens import (
    HeuristicCounter,
    get_counter,
    truncate_to_tokens,
)
from contextforge.types import assistant, user


def main() -> None:
    title("实验 03 · Token 计数与分区预算")

    step("第 1 步：先看计数器是怎么选出来的")
    counter = get_counter()
    note(f"本次使用的计数器：{counter.name}")
    note("选择顺序：tiktoken → transformers → 启发式（任何一级不可用就降级，不影响主流程）")

    step("第 2 步：启发式计数在中文与英文上的差别")
    heuristic = HeuristicCounter()
    samples = [
        "上下文工程的核心是把有限预算分配给最该保留的信息。",
        "Context engineering allocates a limited budget to the most valuable tokens.",
        "把 ContextForge 做成求职作品集项目，必须有可量化的对比数据。",
    ]
    for text in samples:
        chars = len(text)
        tokens = heuristic.count_text(text)
        ratio = tokens / chars if chars else 0
        note(f"{chars:>3} 字符 → {tokens:>3} token（≈{ratio:.2f} token/字符） {text[:26]}…")
    note("中英混排时字符数与 token 数偏差可达 2 倍以上 —— 这就是不能用字符数近似的原因。")

    step("第 3 步：分区预算把「一个总阈值」变成「六份额度」")
    settings = ContextSettings()
    allocator = BudgetAllocator(settings, counter)
    usable = settings.token_budget - settings.reserved_output_tokens
    note(f"总预算 {settings.token_budget} token，为模型输出预留 {settings.reserved_output_tokens} token")
    note(f"可用于输入的部分：{usable} token")
    rows = []
    for section, limit in allocator.allocation.items():
        share = limit / usable if usable else 0
        rows.append([section, f"{limit}", f"{share:.0%}"])
    print()
    print(kv_table(["分区", "额度(token)", "占比"], rows))
    print()
    note("每一段超支时，只会在它自己的额度内裁剪：")
    note("  · 检索片段太多 → 丢掉低分片段，历史不受影响")
    note("  · 历史太长      → 压缩历史，检索片段不受影响")
    note("  · 当前问题      → 单独留额度，保证永远不会被挤掉")

    step("第 4 步：截断是「按 token」而不是「按字符」")
    long_text = "上下文预算需要分区分配。" * 60
    limit = 120
    trimmed, truncated = truncate_to_tokens(long_text, limit, counter)
    note(f"原文 {counter.count_text(long_text)} token，额度 {limit} token")
    note(f"是否发生截断：{truncated}")
    note(f"截断后 {counter.count_text(trimmed)} token（含后缀开销，仍在额度内）")
    note(f"截断结果尾部：…{trimmed[-24:]}")

    step("第 5 步：整条消息地裁剪，避免把一条消息截成半截")
    history = []
    for index in range(12):
        history.append(user(f"第 {index} 轮：我在学习 Agent 的上下文管理，想弄明白预算怎么分配。"))
        history.append(assistant(f"第 {index} 轮回复：预算要按分区分配，历史压缩以轮次为单位。"))
    kept, used, dropped = allocator.take_messages(history, SECTION_HISTORY, anchor_count=0)
    note(f"历史区额度 {allocator.limit_of(SECTION_HISTORY)} token")
    note(f"原始 {len(history)} 条消息 → 保留 {len(kept)} 条，占用 {used} token")
    note(f"丢弃 {len(dropped)} 条（从最旧的开始丢）")
    note("被保留下来的是最新的若干轮 —— 因为最近的信息与当前问题最相关。")

    step("第 6 步：不同分区互不挤占")
    sys_messages, sys_used, _ = allocator.take_text("A" * 5000, SECTION_SYSTEM)
    note(f"system 区额度 {allocator.limit_of(SECTION_SYSTEM)} token，写入 5000 字符后被截到 {sys_used} token")
    note("即便 system 区被截断，历史区的额度也没有减少一丝一毫。")

    print()
    print(
        kv_table(
            ["能力", "朴素实现", "ContextForge"],
            [
                ["计量单位", "字符数", f"token（{counter.name}）"],
                ["预算粒度", "单一总阈值", "六个分区各自定额"],
                ["缺依赖时", "导入即崩", "静默降级并标注"],
                ["裁剪精度", "可能截半条消息", "整条消息 / 整轮"],
            ],
        )
    )

    conclusion(
        "把「上下文快满了」变成可计算的数字，需要两件事：一把**统一的尺子**"
        "（可降级的 token 计数器）和一份**分区的预算表**（六段额度）。"
        "有了它们，「该保留什么」才从感觉变成了取舍。"
    )


if __name__ == "__main__":
    main()
