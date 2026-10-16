"""模型工厂：根据配置决定用真实 API 还是离线模型。"""

from __future__ import annotations

from ..config import ModelSettings, Settings, get_settings
from .base import BaseLLM, LLMError
from .mock import MockLLM
from .openai_compat import OpenAICompatibleLLM


def build_llm(settings: Settings | None = None, force_mock: bool = False) -> BaseLLM:
    """构造模型客户端。

    优先级：显式 force_mock > 配置了 API Key > 离线 Mock。
    无论哪种情况都不抛异常，保证系统始终可启动。
    """
    settings = settings or get_settings()
    model: ModelSettings = settings.model

    if force_mock or not model.online:
        return MockLLM()

    try:
        return OpenAICompatibleLLM(
            api_key=model.api_key,
            base_url=model.base_url,
            model=model.model,
            embedding_model=model.embedding_model,
        )
    except LLMError:
        # 例如未安装 openai SDK：降级而不是崩溃
        return MockLLM()


def describe_llm(llm: BaseLLM) -> dict:
    return llm.describe()
