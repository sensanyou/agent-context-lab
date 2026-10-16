"""统一分词。

中英文混排场景下，分词质量直接影响 BM25 与向量检索的效果：
英文按词、CJK 按「单字 + 二元组」切分 —— 二元组能显著缓解中文没有空格带来的
召回问题（例如「上下文压缩」要能被「上下文」和「压缩」同时命中）。
"""

from __future__ import annotations

import re

_CJK_CHAR = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "有", "我", "你", "他", "它",
    "这", "那", "个", "就", "都", "也", "很", "会", "要", "把", "被", "对",
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
    "on", "with", "as", "by", "at", "be", "it", "this", "that",
}


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """返回 token 列表（保留字面形式，仅小写拉丁）。"""
    if not text:
        return []

    tokens: list[str] = []

    for word in _LATIN.findall(text):
        token = word.lower()
        if keep_stopwords or (token not in STOPWORDS and len(token) > 1):
            tokens.append(token)

    for number in _NUMBER.findall(text):
        tokens.append(number)

    cjk_run: list[str] = []
    for char in text:
        if _CJK_CHAR.match(char):
            cjk_run.append(char)
            continue
        _flush_cjk(cjk_run, tokens, keep_stopwords)

    _flush_cjk(cjk_run, tokens, keep_stopwords)
    return tokens


def _flush_cjk(run: list[str], tokens: list[str], keep_stopwords: bool) -> None:
    if not run:
        return
    for char in run:
        if keep_stopwords or char not in STOPWORDS:
            tokens.append(char)
    for i in range(len(run) - 1):
        bigram = run[i] + run[i + 1]
        if keep_stopwords or bigram not in STOPWORDS:
            tokens.append(bigram)
    run.clear()


def tokenize_query(text: str) -> list[str]:
    """查询侧分词：保留更多信息（不过度过滤），提高召回。"""
    return tokenize(text, keep_stopwords=False)


def normalize(text: str) -> str:
    """用于字面匹配的归一化：去空白、去标点、转小写。"""
    return re.sub(r"[\s\W_]+", "", text).lower()


def character_ngrams(text: str, n: int = 2) -> list[str]:
    compact = normalize(text)
    if len(compact) < n:
        return [compact] if compact else []
    return [compact[i : i + n] for i in range(len(compact) - n + 1)]
