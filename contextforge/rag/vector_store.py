"""向量存储。

用 numpy 做全量余弦相似度检索。这在个人知识库规模（几千到几万切片）下是完全
可接受的：512 维 × 2 万切片的矩阵乘法在毫秒级，比引入一个外部向量数据库
（Chroma / Qdrant）更简单、更快、更好部署。

`VectorStore` 是协议式的，需要服务化时实现同一套方法即可替换：

    ChromaVectorStore / QdrantVectorStore -> add / search / remove_document
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence

import numpy as np

from ..types import Chunk


class VectorStore(Protocol):
    dim: int

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None: ...
    def search(
        self, vector: np.ndarray, top_k: int, filter_metadata: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def remove_document(self, doc_id: str) -> int: ...
    def count(self) -> int: ...


class InMemoryVectorStore:
    """内存向量库：矩阵存向量，列表存切片，支持元数据过滤。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._index: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"向量维度不匹配：存储为 {self.dim}，收到 {vectors.shape[1]}"
            )

        # 归一化后内积即余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms

        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in self._index:
                self._matrix[self._index[chunk.chunk_id]] = vector
                self._chunks[self._index[chunk.chunk_id]] = chunk
                continue
            self._index[chunk.chunk_id] = len(self._chunks)
            self._chunks.append(chunk)
            self._matrix = np.vstack([self._matrix, vector.reshape(1, -1)])

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._chunks or top_k <= 0:
            return []

        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm == 0:
            return []
        query = query / norm

        scores = self._matrix @ query

        if filter_metadata:
            mask = np.array(
                [
                    all(chunk.metadata.get(key) == value for key, value in filter_metadata.items())
                    for chunk in self._chunks
                ]
            )
            if not mask.any():
                return []
            masked = np.where(mask, scores, -np.inf)
        else:
            masked = scores

        k = min(top_k, len(self._chunks))
        candidate_idx = np.argpartition(-masked, k - 1)[:k]
        ordered = candidate_idx[np.argsort(-masked[candidate_idx])]

        results: list[tuple[str, float]] = []
        for index in ordered:
            score = float(masked[index])
            if not np.isfinite(score) or score <= 0:
                continue
            results.append((self._chunks[index].chunk_id, round(score, 6)))
        return results

    # ------------------------------------------------------------------ #
    def get(self, chunk_id: str) -> Chunk | None:
        position = self._index.get(chunk_id)
        return self._chunks[position] if position is not None else None

    def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return [chunk for chunk_id in chunk_ids if (chunk := self.get(chunk_id)) is not None]

    def remove_document(self, doc_id: str) -> int:
        keep = [chunk for chunk in self._chunks if chunk.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if removed == 0:
            return 0

        keep_ids = {chunk.chunk_id for chunk in keep}
        rows = [i for i, chunk in enumerate(self._chunks) if chunk.chunk_id in keep_ids]
        self._matrix = self._matrix[rows] if rows else np.zeros((0, self.dim), dtype=np.float32)
        self._chunks = keep
        self._index = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        return removed

    def count(self) -> int:
        return len(self._chunks)

    def documents(self) -> list[str]:
        return sorted({chunk.doc_id for chunk in self._chunks})

    def export(self) -> list[tuple[Chunk, list[float]]]:
        return [
            (chunk, self._matrix[index].tolist())
            for index, chunk in enumerate(self._chunks)
        ]

    def load(self, rows: Sequence[tuple[Chunk, Sequence[float]]]) -> None:
        self._chunks = []
        self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._index = {}
        if not rows:
            return
        chunks = [row[0] for row in rows]
        vectors = np.asarray([row[1] for row in rows], dtype=np.float32)
        self.add(chunks, vectors)

    def describe(self) -> dict:
        return {
            "backend": "in-memory-numpy",
            "dim": self.dim,
            "chunks": self.count(),
        }
