"""OpenAI 兼容模型客户端。

覆盖 DeepSeek / OpenAI / 通义 / 任何 OpenAI 兼容网关。只依赖 `openai` SDK 的
最小表面（chat.completions、embeddings），其余全部收敛在本文件内。
"""

from __future__ import annotations

import time
from typing import Any

from ..types import LLMResponse, Message, ToolCallRequest, Usage
from .base import BaseLLM, LLMError

try:  # pragma: no cover - 取决于环境
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


class OpenAICompatibleLLM(BaseLLM):
    name = "openai-compatible"
    online = True
    supports_embeddings = True

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        embedding_model: str = "text-embedding-3-small",
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise LLMError("未安装 openai SDK，请先执行 pip install openai")
        if not api_key:
            raise LLMError("缺少 API Key")

        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**payload)
                break
            except Exception as exc:  # noqa: BLE001 - 统一收敛为 LLMError
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMError(f"模型调用失败：{exc}") from exc
                time.sleep(0.5 * (attempt + 1))
        else:  # pragma: no cover
            raise LLMError(f"模型调用失败：{last_error}")

        latency_ms = (time.perf_counter() - started) * 1000
        choice = response.choices[0]
        message = choice.message

        raw_calls = getattr(message, "tool_calls", None) or []
        tool_calls = [
            ToolCallRequest(
                id=call.id,
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for call in raw_calls
        ]

        usage = Usage()
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = Usage(
                prompt_tokens=raw_usage.prompt_tokens or 0,
                completion_tokens=raw_usage.completion_tokens or 0,
                total_tokens=raw_usage.total_tokens or 0,
            )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            model=self.model,
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "stop",
        )

    # ------------------------------------------------------------------ #
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"向量化失败：{exc}") from exc
        return [item.embedding for item in response.data]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "online": True,
            "embedding_model": self.embedding_model,
        }
