"""实验 08 · RAG ≠ 记忆：四种机制解决四个不同的问题

## 这个实验要回答什么

「把资料塞进向量库，需要的时候就检索出来」—— 很多人把这叫「记忆」。
但它只解决了四种机制里的**一种**。把它们混在一起，会出现三种典型故障：

| 机制 | 回答的问题 | 存什么 | 时间尺度 |
| --- | --- | --- | --- |
| **RAG 检索** | 这次要用到什么资料？ | 文档切片 | 无（随时可替换） |
| **会话记忆** | 刚才说了什么？ | 消息序列 | 一次会话 |
| **长期记忆** | 什么该跨任务留存？ | 偏好 / 目标 / 进度 / 事实 | 跨会话，带衰减 |
| **上下文工程** | 这一轮到底发什么给模型？ | 以上三者的**装配结果** | 每一轮 |

典型故障：

* 把用户偏好写进知识库 → 换个说法就检索不到（它是「当前相关」而不是「应当遵守」）；
* 把所有对话都写进长期记忆 → 记忆被闲聊淹没，真正重要的偏好再也召不回来；
* 有 RAG 但没有上下文装配 → 检索结果和历史互相挤占预算，工具链被压断。

本实验逐个验证这四种机制确实在**分别工作**、并且**边界清晰**。
"""

from __future__ import annotations

from _lab import SAMPLE_DOCS, conclusion, kv_table, note, offline_runtime, step, title

from contextforge.rag.tokenize import tokenize


def main() -> None:
    title("实验 08 · RAG ≠ 记忆：四种机制解决四个不同的问题")

    with offline_runtime(sample_docs=SAMPLE_DOCS) as runtime:
        step("第 1 步：机制一 —— RAG 检索「这次要用到什么资料」")
        hits = runtime.search("RRF 融合为什么比加权融合鲁棒？", top_k=3)
        note(f"知识库里存的是**文档切片**（{runtime.knowledge.stats()['chunks']} 个切片，来自示例文档）")
        for item in hits:
            chunk = item["chunk"]
            note(f"  {chunk['title']}#{chunk['index']}  score={item['score']:.3f}")
        note("特点：按「与问题的相关度」排序，因此换个说法结果就变。")
        note("它回答的是「这次要用到什么」，而不是「我应该遵守什么」。")

        step("第 2 步：机制二 —— 会话记忆「刚才说了什么」")
        conversation = runtime.memory.conversations.ensure(None, title="记忆实验")
        runtime.memory.save_turn(
            conversation,
            [
                {"role": "user", "content": "我在做一个叫 ContextForge 的项目。"},
                {"role": "assistant", "content": "好的，我记下了。"},
            ],
        )
        history = runtime.memory.load_history(conversation)
        note(f"会话 {conversation} 里有 {len(history)} 条消息")
        for message in history:
            note(f"  {message['role']}: {str(message['content'])[:36]}")
        note("特点：按时间顺序保存**原话**，作用范围限定在一次会话内。")
        note("它的作用是让「刚才」可以被理解，而不是让「重要的事」被长期记住。")

        step("第 3 步：机制三 —— 长期记忆「什么该跨任务留存」")
        runtime.remember("用户偏好使用中文回答，代码示例要带注释", kind="preference", importance=0.9)
        runtime.remember("长期目标：把 ContextForge 做成求职作品集项目", kind="goal", importance=0.85)
        runtime.remember("进度：已完成混合检索与上下文压缩模块", kind="progress", importance=0.6)

        recall = runtime.recall("用户有什么偏好？", top_k=3)
        note("召回「用户有什么偏好？」：")
        for item in recall:
            note(
                f"  [{item['kind']}] score={item['score']:.3f}"
                f" importance={item['importance']:.2f}  {item['content'][:34]}"
            )

        step("第 4 步：长期记忆的排序不是「相似度」，而是加权公式")
        note("score = (0.55 × 词面重叠 + 0.30 × 重要度 + 0.15 × 时间衰减)")
        note("        × 类型权重 × 使用反馈")
        note("  · 词面重叠：这次问的跟这条记忆有多少交集；")
        note("  · 重要度：写入时人工/模型给的权重（偏好与目标高于闲聊）；")
        note("  · 时间衰减：45 天半衰期 —— 老记忆不会突然消失，但会自然让位；")
        note("  · 类型权重：偏好/目标 > 事实 > 进度；")
        note("  · 使用反馈：被召回并真正用上的记忆，下次更容易再被召回。")
        note("这就是为什么第 3 步里，问「偏好」时 preference 排在 progress 前面 ——")
        note("即便 progress 更晚写入（时间更新），类型权重和重要度也会把它压下去。")

        step("第 5 步：边界实验 —— 同样的内容放在不同机制里，行为完全不同")
        statement = "用户偏好使用中文回答"
        runtime.knowledge.add_text(
            f"# 用户偏好说明\n\n{statement}。\n\n这是关于用户偏好的说明文档。",
            title="用户偏好说明.md",
            source="pref.md",
        )
        by_rag = runtime.search(statement, top_k=1)
        by_memory = runtime.recall(statement, top_k=1)
        note(f"放进知识库（RAG）：检索得到 {len(by_rag)} 条，按相关度排序，随时可被新文档覆盖")
        note(f"放进长期记忆：召回得到 {len(by_memory)} 条，带 kind/importance/衰减，跨任务存活")
        note("两者都能「找回来」，但只有长期记忆会在**每一轮**被当作「需要遵守的约束」注入上下文。")

        step("第 6 步：机制四 —— 上下文工程负责「装配」，不是存储")
        result = runtime.ask("我的偏好是什么？")
        report = result.context_report
        note("问「我的偏好是什么？」后，上下文引擎的分区占用：")
        if report:
            for section, used in report.sections.items():
                note(f"  {section:<10} {used:>5} tokens（额度 {report.allocation.get(section, 0)}）")
            note(f"合计 {report.tokens_after} tokens（压缩前 {report.tokens_before}）")
        note("可以看到：长期记忆、检索片段、历史、工具定义各占一块额度，互不挤占。")
        note("注意 recent（当前任务的在飞工具链）没有分配额度 —— 这是刻意的：")
        note("它不是「可裁剪的历史」，而是「正在进行的工具调用」，删掉它请求就非法了。")
        note("上下文工程**不存储**任何东西 —— 它负责的是「这一轮把谁装进去、装多少」。")

        step("第 7 步：记忆的分词与降重")
        note(f"「{statement}」的分词：{tokenize(statement)}")
        same = runtime.remember("用户偏好使用中文回答。", kind="preference", importance=0.9)
        stats = runtime.memory.long_term.stats()
        note("再写入一条几乎一样的偏好（只差一个句号），也不会产生重复记忆：")
        note(f"  合并到已有记忆：{same['content']}")
        note(f"  当前长期记忆统计：{stats}")

        print()
        print(
            kv_table(
                ["机制", "它回答的问题", "换个说法还找得到吗"],
                [
                    ["RAG 检索", "这次要用到什么资料", "可能找不到"],
                    ["会话记忆", "刚才说了什么", "按时间，不按相似度"],
                    ["长期记忆", "什么该跨任务留存", "能（多因素打分）"],
                    ["上下文工程", "这一轮发什么给模型", "不适用（它是装配层）"],
                ],
            )
        )

    conclusion(
        "RAG、会话记忆、长期记忆、上下文工程是**四种不同的机制**，"
        "差别不在实现细节，而在它们回答的问题不同："
        "「这次用什么资料」/「刚才说了什么」/「什么该长期留存」/「这一轮发什么」。"
        "把它们混成一个「向量库」，就会同时失去时效性、可控性和可解释性。"
    )


if __name__ == "__main__":
    main()
