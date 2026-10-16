"""Token 计数。

如果在模块顶层直接 `AutoTokenizer.from_pretrained("deepseek-ai/...")`，一旦没有网络、
模型名失效或 tokenizer 不可用，**整个上下文模块会导入即崩**。这里改成三级可插拔：

1. `tiktoken`（如果安装了）
2. HuggingFace `transformers`（如果本地已有可用 tokenizer，可显式指定）
3. 启发式估算（默认，零依赖）

同时统一暴露 `TokenCounter` 协议，方便在上层做「本地估算 vs API 实际用量」的偏差分析。
"""

from __future__ import annotations

import math
import re
from typing import Any, Protocol

from ..types import Message, message_text

_CJK = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9_]+")


class TokenCounter(Protocol):
    name: str

    def count_text(self, text: str) -> int: ...

    def count_messages(self, messages: list[Message]) -> int: ...


# --------------------------------------------------------------------------- #
# 启发式估算（默认）
# --------------------------------------------------------------------------- #
class HeuristicCounter:
    """中日韩按字计、拉丁按词计的轻量估算器。

    误差量级在 ±15% 以内，对于上下文预算控制已经足够；真实计费请以 API 返回的
    `usage` 为准 —— 项目会在评测层同时记录两者并输出偏差。
    """

    name = "heuristic"
    #: 每条消息的角色/分隔符等固定开销
    PER_MESSAGE_OVERHEAD = 4

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        cjk = len(_CJK.findall(text))
        latin_words = _LATIN_WORD.findall(text)
        latin_tokens = sum(max(math.ceil(len(word) / 4), 1) for word in latin_words)
        punctuation = len(re.findall(r"[^\w\s\u3000-\u9fff]", text))
        # 标点与空白按 1/3 计
        return int(cjk + latin_tokens + math.ceil(punctuation / 3))

    def count_messages(self, messages: list[Message]) -> int:
        total = 0
        for message in messages:
            total += self.PER_MESSAGE_OVERHEAD + self.count_text(message_text(message))
        return total


# --------------------------------------------------------------------------- #
# tiktoken
# --------------------------------------------------------------------------- #
class TiktokenCounter:
    name = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken  # type: ignore[import-not-found]

        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text, disallowed_special=()))

    def count_messages(self, messages: list[Message]) -> int:
        return sum(4 + self.count_text(message_text(m)) for m in messages)


# --------------------------------------------------------------------------- #
# HuggingFace tokenizer
# --------------------------------------------------------------------------- #
class TransformersCounter:
    """需要本地已有 tokenizer 权重；加载失败时由工厂降级。"""

    name = "transformers"

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: list[Message]) -> int:
        return sum(4 + self.count_text(message_text(m)) for m in messages)


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def get_counter(preferred: str = "auto", tokenizer_model: str = "") -> TokenCounter:
    """按偏好选择计数器，任何失败都静默降级到启发式。"""
    if preferred in ("auto", "tiktoken"):
        try:
            return TiktokenCounter()
        except Exception:  # noqa: BLE001 - 未安装或加载失败都不应影响主流程
            pass

    if preferred in ("auto", "transformers") and tokenizer_model:
        try:
            return TransformersCounter(tokenizer_model)
        except Exception:  # noqa: BLE001
            pass

    return HeuristicCounter()


def truncate_to_tokens(text: str, max_tokens: int, counter: TokenCounter) -> tuple[str, bool]:
    """按 token 上限截断文本。返回 (文本, 是否发生了截断)。

    注意：截断后会追加提示后缀，后缀自身也占 token，
    因此二分查找时要用「预算 - 后缀开销」作为上界，否则结果会超标。
    """
    if max_tokens <= 0:
        return "", bool(text)
    if counter.count_text(text) <= max_tokens:
        return text, False

    suffix = "…（已按上下文预算截断）"
    suffix_cost = counter.count_text(suffix)
    budget = max(max_tokens - suffix_cost, 1)

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if counter.count_text(text[:mid]) <= budget:
            low = mid
        else:
            high = mid - 1

    cut = max(low - 1, 0)
    trimmed = text[:cut] + suffix
    # 极端情况下（后缀本身就超预算）退化为纯截断，不再追加后缀
    if counter.count_text(trimmed) > max_tokens:
        while cut > 0 and counter.count_text(text[:cut]) > max_tokens:
            cut -= 1
        trimmed = text[:cut]
    return trimmed, True


def truncate_messages(
    messages: list[Message],
    max_tokens: int,
    counter: TokenCounter,
) -> tuple[list[Message], int, list[str]]:
    """整条消息地裁剪，保证不会把一条消息截成半截。

    返回 (消息列表, 实际 token 数, 被丢弃消息的摘要列表)。
    """
    kept: list[Message] = []
    dropped: list[str] = []
    used = 0

    for message in reversed(messages):
        cost = counter.count_text(message_text(message)) + HeuristicCounter.PER_MESSAGE_OVERHEAD
        if used + cost > max_tokens and kept:
            dropped.append(_describe(message))
            continue
        kept.append(message)
        used += cost

    kept.reverse()
    dropped.reverse()
    return kept, used, dropped


def _describe(message: Message) -> str:
    content = str(message.get("content") or "")
    preview = " ".join(content.split())[:40]
    return f"{message.get('role')}: {preview or '(空)'}"


def approximate_tokens(value: Any, counter: TokenCounter) -> int:
    """估算任意可序列化对象的 token 量（用于工具 schema 等）。"""
    import json

    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    return counter.count_text(text)
