"""文本向量化。

两种实现，接口完全一致，切换只改一个配置项：

* `HashingEmbedder`（默认，离线）：hashing trick + 子线性 TF + L2 归一化。
  零依赖、零网络、结果确定可复现，适合开发/测试/离线演示。
  需要注意它的能力边界：它是**字面级**表示，不做语义泛化 ——
  「压缩」和「精简」在它眼里是两个不同的特征。这也是它必须和 BM25
  一起走混合检索、并在文档里明确写清适用场景的原因。
* `RemoteEmbedder`：包装任意 OpenAI 兼容的 embeddings 接口，
  接上真实模型后即具备语义检索能力。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .tokenize import tokenize

DEFAULT_DIM = 512


class Embedder(Protocol):
    name: str
    dim: int
    online: bool

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """离线确定性向量化。"""

    name = "hashing"
    online = False

    def __init__(self, dim: int = DEFAULT_DIM, use_bigrams: bool = True) -> None:
        self.dim = dim
        self.use_bigrams = use_bigrams

    # ------------------------------------------------------------------ #
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_one(text) for text in texts]).astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vector

        counts = Counter(tokens)
        for token, count in counts.items():
            bucket, sign = self._bucket(token)
            # 子线性 TF：抑制高频词的主导作用
            weight = (1.0 + math.log(count)) * sign
            # 二元组权重略高，增强词组信号
            if self.use_bigrams and len(token) > 1:
                weight *= 1.15
            vector[bucket] += weight

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        bucket = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return bucket, sign

    def describe(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "online": False,
            "note": "离线 hashing 向量：字面级表示，不做语义泛化；生产环境建议替换为远程 embedding。",
        }


class RemoteEmbedder:
    """包装 OpenAI 兼容的 embeddings 接口。"""

    name = "remote"
    online = True

    def __init__(self, llm, dim: int | None = None) -> None:
        self.llm = llm
        self.dim = dim or 1536
        self._probe_dim: int | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        raw = self.llm.embed(list(texts))
        matrix = np.asarray(raw, dtype=np.float32)
        self._probe_dim = matrix.shape[1] if matrix.ndim == 2 else self.dim
        self.dim = self._probe_dim
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def describe(self) -> dict:
        return {"name": self.name, "dim": self.dim, "online": True}


def build_embedder(llm, dim: int = DEFAULT_DIM) -> Embedder:
    """有真实模型且支持 embeddings 时用远程，否则用离线实现。"""
    if getattr(llm, "supports_embeddings", False):
        try:
            return RemoteEmbedder(llm, dim=dim)
        except Exception:  # noqa: BLE001 - 探测失败不影响启动
            pass
    return HashingEmbedder(dim=dim)
