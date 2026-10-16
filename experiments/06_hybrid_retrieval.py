"""实验 06 · 混合检索：为什么要把向量检索和 BM25 融合起来

## 这个实验要回答什么

只做向量检索（或者说，只做「语义」检索）会漏掉什么？

漏掉的是**精确词**。当用户问的是一串模型名（`HashingEmbedder`、`RRF`、`top_k`）
或一个错误码时，字面命中比语义相似可靠得多；反过来，
当用户换了一种说法（「怎么让搜索结果更准」而不是「重排序」）时，
只有向量检索能救。

本实验用同一组问题，分别只跑一路、再跑融合，把差别摆出来：

1. `dense`  —— 只用向量（离线用 hashing 向量）
2. `sparse` —— 只用 BM25（Okapi，k1=1.5, b=0.75）
3. `hybrid` —— 两路召回 → RRF 融合（k=60）→ 启发式重排序

另外还会解释一个容易被忽略的工程选择：**为什么默认用 RRF 而不是加权融合**。
"""

from __future__ import annotations

from _lab import (
    SAMPLE_DOCS,
    conclusion,
    kv_table,
    note,
    offline_runtime,
    step,
    title,
)

from contextforge.rag.retriever import reciprocal_rank_fusion, weighted_fusion

QUERIES = [
    ("精确词", "RRF 融合的公式里 k 取多少？"),
    ("换种说法", "怎么让搜索结果排序更准？"),
    ("领域问题", "上下文压缩为什么要以轮次为单位？"),
]


def main() -> None:
    title("实验 06 · 混合检索：为什么要把向量检索和 BM25 融合起来")

    with offline_runtime(sample_docs=SAMPLE_DOCS) as runtime:
        knowledge = runtime.knowledge
        stats = knowledge.stats()
        note(f"知识库规模：{stats['documents']} 篇文档，{stats['chunks']} 个切片")
        note(
            f"向量后端：{stats['vector_store']['backend']}（{stats['vector_store']['dim']} 维）"
        )
        note(f"BM25 词项：{stats['bm25']['terms']} 个")

        step("第 1 步：同一组问题，三种检索模式的 Top-3 对比")
        for label, query in QUERIES:
            print(f"\n   【{label}】{query}")
            for mode in ("dense", "sparse", "hybrid"):
                hits = knowledge.search(query, top_k=3, mode=mode)
                titles = " | ".join(
                    f"{item.chunk.title}#{item.chunk.index}" for item in hits
                ) or "（无结果）"
                print(f"     {mode:<7} {titles}")
            trace = knowledge.retriever.last_trace
            if trace:
                note(
                    f"轨迹：向量路召回 {trace.dense_hits} 条，BM25 路召回 {trace.sparse_hits} 条，"
                    f"RRF 融合得到 {trace.fused_hits} 条候选，其中 {trace.reranked_hits} 条进入重排，"
                    f"最终按分数取 Top-3"
                )

        step("第 2 步：RRF 到底做了什么")
        dense = [("chunk_A", 0.91), ("chunk_B", 0.77), ("chunk_C", 0.55)]
        sparse = [("chunk_C", 8.4), ("chunk_A", 6.1), ("chunk_D", 5.0)]
        note(f"向量路排名：{dense}")
        note(f"BM25 路排名：{sparse}")
        rrf = reciprocal_rank_fusion([dense, sparse], k=60)
        ordered = sorted(rrf.items(), key=lambda item: item[1], reverse=True)
        note("RRF 融合结果（score = Σ 1/(k+rank)）：")
        for chunk_id, score in ordered:
            in_dense = [i for i, (cid, _s) in enumerate(dense, 1) if cid == chunk_id]
            in_sparse = [i for i, (cid, _s) in enumerate(sparse, 1) if cid == chunk_id]
        note(
            f"  {chunk_id:<9} score={score:.5f}"
            f"  （向量第 {in_dense[0] if in_dense else '—'} 名，"
            f"BM25 第 {in_sparse[0] if in_sparse else '—'} 名）"
        )
        note("看两个关键的片段：")
        note("  chunk_A = 向量第 1 名 + BM25 第 2 名；chunk_C = 向量第 3 名 + BM25 第 1 名。")
        note("  RRF 下两者几乎持平（0.03252 vs 0.03227）—— 因为 1/(60+1) 与 1/(60+3) 差别极小。")
        note("  RRF 真正奖励的是「**两路都榜上有名**」：chunk_B 只被向量路看上，直接掉到一半分数。")

        step("第 3 步：为什么首选 RRF 而不是加权融合")
        weighted = weighted_fusion([dense, sparse], [0.6, 0.4])
        note(f"加权融合结果：{[f'{k}={v:.3f}' for k, v in sorted(weighted.items(), key=lambda i: -i[1])]}")
        note("同样两个片段，在加权融合里被拉成 chunk_A=0.729 vs chunk_C=0.400 —— 差距远大于 RRF。")
        note("差异来自 min-max 归一化：BM25 那一路分数跨度是 8.4→5.0（跨度 3.4），")
        note("  「BM25 第 2 名」被压成 0.324，于是 chunk_A 在 BM25 一路几乎被清零；")
        note("  而 RRF 只看排名，第 1 名与第 2 名的差距只有 1/61 与 1/62。")
        note("加权融合的代价不止于此：")
        note("  · 向量相似度天然在 0~1，BM25 分数无上界，两者量纲不同；")
        note("  · 归一化依赖「当前这一批结果的最高/最低分」，换一个问题分值就漂移；")
        note("  · 权重 0.6/0.4 还要靠人工调参，而 RRF 只需一个几乎不敏感的 k。")
        note("所以项目默认用 RRF —— 不是因为加权融合更差，而是因为 RRF **不用调参、不受量纲影响**。")

        step("第 4 步：融合之后还有一步重排序")
        note("融合只解决了「哪一路更可信」，没有解决「片段本身与问题的贴合度」。")
        note("启发式重排序按四个维度打分：词面重叠 0.50 / 短语命中 0.25 / 标题命中 0.10 / 长度先验 0.15，")
        note("最终分数 = 0.7 × 重排分 + 0.3 × 融合分。")
        hits = knowledge.search("混合检索为什么需要重排序", top_k=3, mode="hybrid")
        for rank, item in enumerate(hits, start=1):
            note(
                f"  [{rank}] {item.chunk.citation()}"
                f"  score={item.score:.3f}"
                f"  dense={item.dense_score if item.dense_score is None else round(item.dense_score, 3)}"
                f"  sparse={item.sparse_score if item.sparse_score is None else round(item.sparse_score, 3)}"
                f"  阶段={item.source_stage}"
            )

        step("第 5 步：离线向量的诚实说明")
        note("本项目离线使用 hashing 向量（blake2b 分桶 + 符号），它只做**字面级**表示，")
        note("不做语义泛化 —— 「怎么让搜索结果更准」这类问法与「重排序」之间没有字面重叠，")
        note("离线向量是接不住的（这正是上面第 1 步里 dense 表现不佳的原因）。")
        note("它的价值在于：让整条 RAG 链路（分片 → 向量 → 融合 → 重排 → 引用）")
        note("在没有 API Key 的情况下依然可以被完整地跑通和测试；接上真实 embedding 只需换一个实现。")

        print()
        print(
            kv_table(
                ["检索模式", "优势", "短板"],
                [
                    ["dense", "容忍换一种说法", "精确词/专有名词容易漏"],
                    ["sparse", "精确词命中强", "换个说法就找不到"],
                    ["hybrid", "两路互补 + 重排", "多两次计算，实现更复杂"],
                ],
            )
        )

    conclusion(
        "单一检索路线的失败模式是**互补**的：向量路怕精确词，BM25 怕改写。"
        "把它们融合起来，真正的技术决策不是「要不要融合」，而是"
        "**用什么融合** —— 选 RRF 是因为它只依赖排名、不受分数量纲漂移的影响。"
    )


if __name__ == "__main__":
    main()
