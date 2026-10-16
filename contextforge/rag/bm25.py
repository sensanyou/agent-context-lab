"""BM25 稀疏检索（纯 Python，零依赖）。

为什么一定要有它：向量检索擅长「意思相近」，但对**精确词**（函数名、
报错码、专有名词、版本号）经常失灵；BM25 恰好相反。两者用 RRF 融合后，
召回率通常明显优于任一单路 —— 这也是项目里 RRF 融合存在的唯一理由。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .tokenize import tokenize


@dataclass
class BM25Index:
    """BM25-Okapi。

    参数默认值取文献常用配置：k1=1.5（词频饱和）、b=0.75（长度归一化强度）。
    """

    k1: float = 1.5
    b: float = 0.75

    _doc_ids: list[str] = field(default_factory=list)
    _term_freqs: dict[str, Counter] = field(default_factory=dict)
    _lengths: dict[str, int] = field(default_factory=dict)
    _df: Counter = field(default_factory=Counter)
    _postings: dict[str, set[str]] = field(default_factory=dict)
    _avg_len: float = 0.0
    _dirty: bool = False

    # ------------------------------------------------------------------ #
    def add(self, doc_id: str, text: str) -> None:
        tokens = tokenize(text)
        counts = Counter(tokens)

        if doc_id in self._term_freqs:
            self.remove(doc_id)

        self._doc_ids.append(doc_id)
        self._term_freqs[doc_id] = counts
        self._lengths[doc_id] = sum(counts.values())
        for term in counts:
            self._df[term] += 1
            self._postings.setdefault(term, set()).add(doc_id)
        self._dirty = True

    def remove(self, doc_id: str) -> None:
        counts = self._term_freqs.pop(doc_id, None)
        if counts is None:
            return
        for term in counts:
            self._df[term] -= 1
            if self._df[term] <= 0:
                del self._df[term]
            postings = self._postings.get(term)
            if postings is not None:
                postings.discard(doc_id)
                if not postings:
                    del self._postings[term]
        self._lengths.pop(doc_id, None)
        if doc_id in self._doc_ids:
            self._doc_ids.remove(doc_id)
        self._dirty = True

    def clear(self) -> None:
        self._doc_ids.clear()
        self._term_freqs.clear()
        self._lengths.clear()
        self._df.clear()
        self._postings.clear()
        self._avg_len = 0.0
        self._dirty = False

    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def _refresh(self) -> None:
        if not self._dirty:
            return
        total = sum(self._lengths.values())
        self._avg_len = (total / len(self._lengths)) if self._lengths else 0.0
        self._dirty = False

    def _idf(self, term: str) -> float:
        n = max(len(self._doc_ids), 1)
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, doc_id: str) -> float:
        self._refresh()
        counts = self._term_freqs.get(doc_id)
        if not counts:
            return 0.0

        length = self._lengths.get(doc_id, 0) or 1
        avg = self._avg_len or 1.0
        normalization = self.k1 * (1 - self.b + self.b * length / avg)

        total = 0.0
        for term in tokenize(query):
            freq = counts.get(term, 0)
            if not freq:
                continue
            total += self._idf(term) * (freq * (self.k1 + 1)) / (freq + normalization)
        return total

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """返回 [(doc_id, score)]，按分数降序。"""
        if not self._doc_ids or not query.strip():
            return []

        self._refresh()
        query_terms = tokenize(query)
        if not query_terms:
            return []

        # 借助倒排表定位候选文档，避免全量扫描
        candidates: set[str] = set()
        for term in query_terms:
            candidates |= self._postings.get(term, set())

        if not candidates:
            return []

        scored = [(doc_id, self.score(query, doc_id)) for doc_id in candidates]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(doc_id, round(value, 6)) for doc_id, value in scored[:top_k] if value > 0]

    def stats(self) -> dict:
        self._refresh()
        return {
            "documents": len(self._doc_ids),
            "terms": len(self._df),
            "avg_length": round(self._avg_len, 2),
        }
