"""模型接口协议。"""

from __future__ import annotations

import abc
from typing import Any

from ..types import LLMResponse, Message


class LLMError(RuntimeError):
    """模型调用失败。上层可据此做重试或降级。"""


class BaseLLM(abc.ABC):
    """所有模型客户端的共同接口。"""

    name: str = "base"
    model: str = ""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """发送一次对话请求。"""

    # -- 可选能力：真实模型才有，Mock 会退化为本地实现 ---------------------- #
    supports_embeddings: bool = False

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - 默认不支持
        raise NotImplementedError(f"{self.name} 不支持向量化")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "online": self.online}

    @property
    def online(self) -> bool:
        return False
