"""模型适配层。

对外只暴露一个协议 `BaseLLM`，上层（Agent Runtime / 评测）不关心背后是
真实 API 还是离线 Mock。这样带来三个好处：

* **可测试**：单元测试在无网络环境下跑，不产生费用。
* **可对比**：Mock 模型的行为是确定性的，做策略对比实验时不会引入模型噪声。
* **可替换**：换成任意 OpenAI 兼容服务只需改 `base_url` / `model`。
"""

from .base import BaseLLM, LLMError
from .factory import build_llm, describe_llm
from .mock import MockLLM
from .openai_compat import OpenAICompatibleLLM

__all__ = [
    "BaseLLM",
    "LLMError",
    "MockLLM",
    "OpenAICompatibleLLM",
    "build_llm",
    "describe_llm",
]
