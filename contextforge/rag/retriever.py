"""混合检索与重排序。

流程：

    查询
      ├─ 稠密检索（向量）  → top 12
      └─ 稀疏检索（BM25）  → top 12
                ↓
           RRF / 加权融合   → top 16
                ↓
              重排序       → top 5

为什么每一步都值得存在（这是面试里最容易被追问的部分）：

* **为什么两路检索**：向量擅长近义改写，BM25 擅长精确词（函数名、报错码、
  版本号）。两者召回的文档集合往往只有部分重叠，融合能同时提高召回与精度。
* **为什么用 RRF 而不是直接加权分数**：向量余弦分与 BM25 分不在同一量纲，
  权重需要反复调参；RRF 只用「排名」信息，对分数量纲不敏感，鲁棒性更好。
* **为什么还要重排序**：融合阶段只看排名，不看查询与文档的真实匹配程度。
  重排序用更贵的特征（词覆盖率、短语命中、长度先验）对少量候选重新打分，
  在不显著增加延迟的前提下提升精度。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..config import RetrievalSettings
from ..types import Chunk, ScoredChunk
from .embeddings import Embedder
from .tokenize import normalize, tokenize
from .vector_store import InMemoryVectorStore


# --------------------------------------------------------------------------- #
# 融合
# --------------------------------------------------------------------------- #
def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
) -> dict[str, float]:
    """RRF：score(d) = Σ 1 / (k + rank_i(d))。"""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def weighted_fusion(
    ranked_lists: Sequence[Sequence[tuple[str, float]]],
    weights: Sequence[float],
) -> dict[str, float]:
    """加权归一化融合：把每一路的分数压到 0~1 后按权重相加。"""
    fused: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        if not ranked:
            continue
        scores = [score for _id, score in ranked]
        low, high = min(scores), max(scores)
        span = (high - low) or 1.0
        for doc_id, score in ranked:
            normalized = (score - low) / span
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * normalized
    return fused


# --------------------------------------------------------------------------- #
# 重排序
# --------------------------------------------------------------------------- #
@dataclass
class HeuristicReranker:
    """轻量重排序器：无需外部模型的交叉特征打分。

    生产环境可换成 bge-reranker / Cohere Rerank 等交叉编码器，
    只要实现相同的 `score(query, chunk) -> float` 接口即可。
    """

    name: str = "heuristic"

    def score(self, query: str, chunk: Chunk) -> tuple[float, dict[str, float]]:
        query_tokens = set(tokenize(query))
        chunk_tokens = set(tokenize(chunk.text))
        if not query_tokens:
            return 0.0, {}

        overlap = len(query_tokens & chunk_tokens) / len(query_tokens)

        compact_query = normalize(query)
        compact_chunk = normalize(chunk.text)
        phrase = 0.0
        if compact_query:
            if compact_query in compact_chunk:
                phrase = 1.0
            else:
                phrase = _longest_common_run(compact_query, compact_chunk) / max(
                    len(compact_query), 1
                )

        title_hit = 0.0
        if chunk.title:
            title_tokens = set(tokenize(chunk.title))
            if title_tokens & query_tokens:
                title_hit = len(title_tokens & query_tokens) / len(title_tokens)

        # 长度先验：太短信息不足，太长则噪声多、占预算
        length = len(chunk.text)
        if length < 80:
            length_prior = 0.4
        elif length <= 1200:
            length_prior = 1.0
        else:
            length_prior = max(1200 / length, 0.3)

        score = (
            0.50 * overlap
            + 0.25 * phrase
            + 0.10 * title_hit
            + 0.15 * length_prior
        )
        details = {
            "overlap": round(overlap, 4),
            "phrase": round(phrase, 4),
            "title": round(title_hit, 4),
            "length_prior": round(length_prior, 4),
        }
        return round(score, 6), details


def _longest_common_run(a: str, b: str) -> int:
    """最长公共子串长度（滚动数组，O(len(a)*len(b))）。"""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalTrace:
    query: str
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_hits: int = 0
    reranked_hits: int = 0
    dense_weight: float = 0.0
    sparse_weight: float = 0.0
    notes: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "dense_hits": self.dense_hits,
            "sparse_hits": self.sparse_hits,
            "fused_hits": self.fused_hits,
            "reranked_hits": self.reranked_hits,
            "notes": self.notes,
        }


class HybridRetriever:
    """稠密 + 稀疏 + 融合 + 重排序。"""

    def __init__(
        self,
        store: InMemoryVectorStore,
        bm25,
        embedder: Embedder,
        settings: RetrievalSettings,
    ) -> None:
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.settings = settings
        self.reranker = HeuristicReranker()
        self.last_trace: RetrievalTrace | None = None

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        mode: str = "hybrid",
    ) -> list[ScoredChunk]:
        """mode: hybrid | dense | sparse —— 便于做「单路 vs 混合」的对比实验。"""
        settings = self.settings
        final_k = top_k or settings.final_top_k
        trace = RetrievalTrace(query=query)
        trace.dense_weight = settings.dense_weight
        trace.sparse_weight = settings.sparse_weight

        if not query.strip() or self.store.count() == 0:
            self.last_trace = trace
            return []

        dense_hits: list[tuple[str, float]] = []
        sparse_hits: list[tuple[str, float]] = []

        if mode in ("hybrid", "dense"):
            vector = self.embedder.embed_one(query)
            dense_hits = self.store.search(vector, settings.dense_top_k, filter_metadata)
        if mode in ("hybrid", "sparse"):
            sparse_hits = self.bm25.search(query, settings.sparse_top_k)

        trace.dense_hits = len(dense_hits)
        trace.sparse_hits = len(sparse_hits)

        if mode == "dense":
            fused = {chunk_id: score for chunk_id, score in dense_hits}
        elif mode == "sparse":
            fused = {chunk_id: score for chunk_id, score in sparse_hits}
        elif settings.fusion == "weighted":
            fused = weighted_fusion(
                [dense_hits, sparse_hits],
                [settings.dense_weight, settings.sparse_weight],
            )
        else:
            fused = reciprocal_rank_fusion(
                [dense_hits, sparse_hits], k=settings.rrf_k
            )

        trace.fused_hits = len(fused)
        if not fused:
            self.last_trace = trace
            return []

        # 融合分归一化，供最终排序使用
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[
            : settings.fusion_top_k
        ]
        top_score = ranked[0][1] or 1.0

        candidates: list[ScoredChunk] = []
        dense_map = dict(dense_hits)
        sparse_map = dict(sparse_hits)

        for chunk_id, fused_score in ranked:
            chunk = self.store.get(chunk_id)
            if chunk is None:
                continue
            candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    score=round(fused_score / top_score, 6),
                    dense_score=dense_map.get(chunk_id),
                    sparse_score=sparse_map.get(chunk_id),
                    source_stage="fused",
                )
            )

        if settings.rerank and candidates:
            for item in candidates:
                rerank_score, details = self.reranker.score(query, item.chunk)
                item.rerank_score = rerank_score
                item.chunk.metadata = {**item.chunk.metadata, "rerank": details}
                # 重排序为主，融合分作为先验：0.7 / 0.3
                item.score = round(0.7 * rerank_score + 0.3 * item.score, 6)
                item.source_stage = "reranked"
            candidates.sort(key=lambda item: item.score, reverse=True)
            trace.notes.append(
                f"重排序启用（{self.reranker.name}），候选 {len(candidates)} 条"
            )
        else:
            trace.notes.append("重排序关闭")

        trace.reranked_hits = len(candidates)

        if settings.min_score > 0:
            candidates = [c for c in candidates if c.score >= settings.min_score]

        self.last_trace = trace
        return candidates[:final_k]

    # ------------------------------------------------------------------ #
    def search_multi_query(
        self,
        queries: Sequence[str],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """多查询召回后融合，用于「一个问题拆成多个子问题」的检索场景。"""
        if not queries:
            return []

        per_query = [self.search(query, top_k=self.settings.fusion_top_k) for query in queries]
        aggregated: dict[str, ScoredChunk] = {}

        for results in per_query:
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk.chunk_id
                bonus = 1.0 / (self.settings.rrf_k + rank)
                if chunk_id in aggregated:
                    aggregated[chunk_id].score = round(
                        aggregated[chunk_id].score + bonus, 6
                    )
                else:
                    item.score = round(item.score + bonus, 6)
                    aggregated[chunk_id] = item

        ordered = sorted(aggregated.values(), key=lambda item: item.score, reverse=True)
        return ordered[: (top_k or self.settings.final_top_k)]
