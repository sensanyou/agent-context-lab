"""文本向量化。

两种实现，接口完全一致，选择由配置项 `CONTEXTFORGE_EMBEDDING_MODE` 决定：

* `HashingEmbedder`（离线）：hashing trick + 子线性 TF + L2 归一化。
  零依赖、零网络、结果确定可复现，适合开发/测试/离线演示。
  需要注意它的能力边界：它是**字面级**表示，不做语义泛化 ——
  「压缩」和「精简」在它眼里是两个不同的特征。这也是它必须和 BM25
  一起走混合检索、并在文档里明确写清适用场景的原因。
* `RemoteEmbedder`：包装任意 OpenAI 兼容的 embeddings 接口，
  接上真实模型后即具备语义检索能力。

关于「降级」这件事
------------------
向量化模式是**独立于聊天模型**的。早期实现把 `llm.supports_embeddings` 这个
**声明**直接当成了「服务可用」的证据，于是出现了一个很难排查的缺口：

    配了 DeepSeek 的 Key → 判定在线 → 声明支持 embeddings → 选远程向量化
    → 实际请求 https://api.deepseek.com/embeddings → 404（该服务没有这个端点）

而且那层 `try/except` 包住的是**构造**动作，构造时并不发请求，所以它永远成功、
永远兜不住真正的失败。

现在改成**启动时真的探测一次**（`build_embedder(..., probe=True)`）：

* 探测通过 → 用远程；
* 探测失败 → 自动降级到离线向量，并**把失败原因记在 embedder 上**，
  在 `/api/status` 的 `embeddings` 字段里能直接看到，不是静默吞掉。

「能连上就用、连不上就降级，但降级这件事必须可见」——这是本模块的核心约定。
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol, Sequence

import numpy as np

from .tokenize import tokenize

DEFAULT_DIM = 512

#: 探测用文本：足够短（几乎不花成本），但能真正走完一次请求。
PROBE_TEXT = "contextforge 向量化连通性探测"

#: 进程级探测结果缓存：key -> "" 表示可用，否则是失败原因。
#: 避免同一个进程里反复构造 KnowledgeBase 时重复探测。
_probe_cache: dict[str, str] = {}

#: 降级到离线向量时给出的可执行建议。和「失败原因」分开存放，
#: 这样 `/api/status` 里原因是原因、建议是建议，一眼能看完。
LOCAL_FALLBACK_HINT = (
    "离线向量只有字面级匹配能力，检索链路完全可用。要消除这次降级可以二选一："
    "① 若本来就用不着远程向量，在 .env 里设 CONTEXTFORGE_EMBEDDING_MODE=local"
    "（不再发起探测）；"
    "② 若确实要语义检索，用 CONTEXTFORGE_EMBEDDING_BASE_URL / "
    "CONTEXTFORGE_EMBEDDING_API_KEY 单独指向一个支持 /embeddings 的服务。"
)


class EmbeddingError(RuntimeError):
    """向量化失败。消息里带可执行的修复提示。"""


class Embedder(Protocol):
    name: str
    mode: str
    dim: int
    online: bool
    degraded: bool

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_one(self, text: str) -> np.ndarray: ...

    def describe(self) -> dict: ...


class HashingEmbedder:
    """离线确定性向量化。"""

    name = "hashing"
    mode = "local"
    online = False

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        use_bigrams: bool = True,
        degraded_from: str = "",
        degraded_reason: str = "",
    ) -> None:
        self.dim = dim
        self.use_bigrams = use_bigrams
        # 如果它是「远程不可用后退回来的」，把来龙去脉留下来
        self.degraded_from = degraded_from
        self.degraded_reason = degraded_reason

    @property
    def degraded(self) -> bool:
        return bool(self.degraded_reason)

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
        info = {
            "name": self.name,
            "mode": "local",
            "dim": self.dim,
            "online": False,
            "degraded": self.degraded,
            "note": "离线 hashing 向量：字面级表示，不做语义泛化。",
        }
        if self.degraded:
            info["degraded_from"] = self.degraded_from
            info["degraded_reason"] = self.degraded_reason
            info["degraded_hint"] = LOCAL_FALLBACK_HINT
            info["note"] = "已从远程向量化降级到离线向量：检索仍然可用，但只有字面级匹配能力。"
        return info


class RemoteEmbedder:
    """包装 OpenAI 兼容的 embeddings 接口。"""

    name = "remote"
    mode = "remote"
    online = True
    degraded = False

    def __init__(self, llm, dim: int | None = None, model: str = "") -> None:
        self.llm = llm
        self.dim = dim or DEFAULT_DIM
        self.model = model or str(getattr(llm, "embedding_model", "") or "")
        self._probe_dim: int | None = None

    # ------------------------------------------------------------------ #
    def embed(self, texts: Sequence[str], timeout: float | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        raw = self._call(list(texts), timeout)
        matrix = np.asarray(raw, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise EmbeddingError(
                f"向量服务返回了非预期的形状 {matrix.shape}，期望 ({len(texts)}, dim)"
            )
        self.dim = int(matrix.shape[1])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def probe(self, timeout: float | None = None) -> None:
        """真的发一次最小请求，确认端点可用。失败抛异常（由调用方决定是否降级）。"""
        vector = self.embed([PROBE_TEXT], timeout=timeout)[0]
        if vector.size == 0:
            raise EmbeddingError("向量服务返回了空向量")

    # ------------------------------------------------------------------ #
    def _call(self, texts: list[str], timeout: float | None):
        try:
            if timeout is None:
                return self.llm.embed(texts)
            return self.llm.embed(texts, timeout=timeout)
        except TypeError as exc:
            # 第三方向量实现可能没有 timeout 形参 —— 只在真的是这个原因时退化重试
            if "timeout" not in str(exc):
                raise
            return self.llm.embed(texts)
        except Exception as exc:  # noqa: BLE001 - 统一加一条可执行的修复提示
            raise EmbeddingError(
                f"{exc}\n"
                "提示：当前向量化模式为 remote。如果该服务不提供 /embeddings 接口，"
                "可以把 .env 里的 CONTEXTFORGE_EMBEDDING_MODE 设为 local（改用离线向量，"
                "检索链路照常可用），或者用 CONTEXTFORGE_EMBEDDING_BASE_URL 单独指定一个"
                "支持向量接口的服务。"
            ) from exc

    def describe(self) -> dict:
        return {
            "name": self.name,
            "mode": "remote",
            "dim": self.dim,
            "online": True,
            "degraded": False,
            "model": self.model,
        }


# ---------------------------------------------------------------------- #
# 选择逻辑
# ---------------------------------------------------------------------- #
def build_embedder(
    llm=None,
    dim: int = DEFAULT_DIM,
    settings=None,
    probe: bool = True,
    use_cache: bool = True,
) -> Embedder:
    """按配置选择向量化实现，并把「降级」变成一件可见的事。

    参数
    ----
    llm       : 聊天模型客户端。它会作为远程向量化的**默认**载体。
    dim       : 离线向量的维度。
    settings  : `Settings`；读取 `settings.model` 上的向量化配置。
    probe     : 是否真的发一次探测请求。测试里可以关掉。
    use_cache : 是否复用进程内的探测结果。
    """
    model_settings = getattr(settings, "model", None)
    mode = _resolve_mode(model_settings)

    if mode == "local":
        return HashingEmbedder(dim=dim)

    candidate = _build_remote_candidate(llm, dim=dim, model_settings=model_settings)
    if candidate is None:
        # 既没配独立向量端点、聊天模型也不支持 —— 直接离线，这不是「降级」
        return HashingEmbedder(dim=dim)
    if not probe:
        return candidate

    key = _probe_key(candidate)
    if use_cache and key in _probe_cache:
        reason = _probe_cache[key]
    else:
        reason = _probe(candidate, timeout=_probe_timeout(model_settings))
        if use_cache:
            _probe_cache[key] = reason

    if reason:
        return HashingEmbedder(
            dim=dim,
            degraded_from=f"remote（{candidate.model or '未指定模型'}）",
            degraded_reason=reason,
        )
    return candidate


def reset_probe_cache() -> None:
    """清空探测缓存（测试用）。"""
    _probe_cache.clear()


# ---------------------------------------------------------------------- #
# 内部
# ---------------------------------------------------------------------- #
def _resolve_mode(model_settings) -> str:
    if model_settings is None:
        return "auto"
    normalized = getattr(model_settings, "normalized_embedding_mode", None)
    if normalized is not None:
        return normalized
    raw = str(getattr(model_settings, "embedding_mode", "auto") or "auto").strip().lower()
    return raw if raw in ("auto", "local", "remote") else "auto"


def _probe_timeout(model_settings) -> float | None:
    value = getattr(model_settings, "embedding_probe_timeout", None)
    try:
        timeout = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _build_remote_candidate(llm, dim: int, model_settings):
    """构造远程向量化候选；没有远程能力时返回 None。"""
    # ① 单独配置了向量端点 → 单独建一个客户端（聊天与向量可以使用不同服务）
    if model_settings is not None and getattr(
        model_settings, "has_separate_embedding_endpoint", False
    ):
        client = _build_dedicated_client(model_settings)
        if client is not None:
            return RemoteEmbedder(client, dim=dim)
        return None

    # ② 聊天模型自己声称支持 embeddings
    if getattr(llm, "supports_embeddings", False):
        return RemoteEmbedder(llm, dim=dim)
    return None


def _build_dedicated_client(model_settings):
    """按 `embedding_base_url` / `embedding_api_key` 构造一个只用于向量化的客户端。"""
    from ..llm.openai_compat import OpenAICompatibleLLM

    base_url, api_key = model_settings.embedding_endpoint()
    if not api_key:
        return None
    try:
        return OpenAICompatibleLLM(
            api_key=api_key,
            base_url=base_url,
            model=model_settings.model,
            embedding_model=model_settings.embedding_model,
            timeout=max(model_settings.embedding_probe_timeout, 1.0),
        )
    except Exception:  # noqa: BLE001 - 构造失败就当没有远程能力
        return None


def _probe_key(candidate: RemoteEmbedder) -> str:
    llm = candidate.llm
    return "|".join(
        [
            str(getattr(llm, "base_url", "") or ""),
            str(getattr(llm, "embedding_model", "") or ""),
            candidate.model,
        ]
    )


def _probe(candidate: RemoteEmbedder, timeout: float | None) -> str:
    """探测远程端点。返回 "" 表示可用，否则返回失败原因（不抛异常）。

    只保留**原因**本身：修复建议放在 `LOCAL_FALLBACK_HINT` 里单独展示，
    避免 `/api/status` 和命令行输出被一段长提示淹没。
    """
    try:
        candidate.probe(timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - 探测失败只降级，绝不阻断启动
        cause = str(exc).split("提示：", 1)[0]
        cause = " ".join(cause.split())
        return (cause or type(exc).__name__)[:200]
    return ""
