"""上下文工程引擎。

对外接口：

    estimate_context_tokens(messages)      # 估算 token
    trim_history(messages, max_turns)      # 按轮次裁剪历史
    compress_history(messages, ...)        # 按策略压缩历史
    validate_tool_calls(messages)          # 校验工具调用链完整性

    ContextEngine(...).build(ContextRequest(...))   # 统一入口
"""

from .budget import BudgetAllocator
from .engine import ContextBundle, ContextEngine, ContextRequest
from .strategies import STRATEGY_NAMES, CompressionResult, CompressionStrategy, get_strategy
from .tokens import HeuristicCounter, TokenCounter, get_counter, truncate_to_tokens
from .turns import flatten_turns, split_messages
from .validator import ToolChainReport, check_tool_calls, repair_tool_calls, validate_tool_calls

__all__ = [
    "BudgetAllocator",
    "CompressionResult",
    "CompressionStrategy",
    "ContextBundle",
    "ContextEngine",
    "ContextRequest",
    "HeuristicCounter",
    "STRATEGY_NAMES",
    "TokenCounter",
    "ToolChainReport",
    "check_tool_calls",
    "compress_history",
    "estimate_context_tokens",
    "flatten_turns",
    "get_counter",
    "get_strategy",
    "repair_tool_calls",
    "split_messages",
    "trim_history",
    "truncate_to_tokens",
    "validate_tool_calls",
]


# --------------------------------------------------------------------------- #
# 向后兼容的旧接口（保持原签名语义，方便迁移已有代码与测试）
# --------------------------------------------------------------------------- #
_default_counter: TokenCounter | None = None


def _counter() -> TokenCounter:
    global _default_counter
    if _default_counter is None:
        _default_counter = get_counter()
    return _default_counter


def estimate_context_tokens(messages) -> int:
    """估算消息列表的 token 数量（不再依赖远程 tokenizer）。"""
    return _counter().count_messages(messages)


def trim_history(messages, max_turns: int = 6):
    """保留 system 提示词与最近 max_turns 个完整轮次。"""
    if max_turns < 1:
        raise ValueError("max_turns 必须至少为 1")
    system_messages, turns = split_messages(messages)
    return list(system_messages) + flatten_turns(turns[-max_turns:])


def compress_history(
    messages,
    keep_recent_turns: int = 4,
    max_summary_chars: int = 1200,
    strategy: str = "summary",
    llm=None,
):
    """压缩历史对话（默认抽取式摘要，可切换策略）。"""
    result = get_strategy(strategy, _counter()).compress(
        messages,
        keep_recent_turns=keep_recent_turns,
        max_summary_chars=max_summary_chars,
        llm=llm,
    )
    return result.messages
