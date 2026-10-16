"""文本切片。

两条不同的切分路径：

* **散文/文档**：按「段落 → 句子 → 硬切」的递归顺序切分，保留 overlap，
  避免把一句话拦腰截断。
* **代码**：优先按顶层 `def` / `class` / 顶层语句切分（Python 用 `ast` 精确定位），
  让每个切片尽量是一个语义完整的单元 —— 这对「代码仓库理解」场景至关重要，
  否则检索回来的往往是半个函数。

切分质量直接决定检索上限，因此这里不做「定长硬切」的偷懒实现。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

from .loaders import CODE_FILE_KIND, LoadedFile

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[。！？!?；;\.])\s*")


@dataclass
class TextChunk:
    index: int
    text: str
    kind: str = "text"
    metadata: dict | None = None


def chunk_loaded_file(
    loaded: LoadedFile,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[TextChunk]:
    if not loaded.text.strip():
        return []
    if loaded.kind == CODE_FILE_KIND:
        return chunk_code(loaded.text, chunk_size, chunk_overlap, language=loaded.metadata.get("language", ""))
    return chunk_text(loaded.text, chunk_size, chunk_overlap)


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for block in _split_blocks(text, chunk_size):
        payload = block.strip()
        # 这里刻意不按长度过滤：短标题（"# 持久化测试"）往往承载关键信息，
        # 之前用「少于 20 字丢弃」会让整篇只剩标题的文档检索不到。
        # 碎片问题交给 _merge_small 处理。
        if not payload:
            continue
        chunks.append(TextChunk(index=len(chunks), text=payload, kind="text"))

    return _merge_small(chunks, chunk_size, min_size=max(chunk_size // 4, 80))


def chunk_code(
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    language: str = "py",
) -> list[TextChunk]:
    if language in ("py", "python"):
        blocks = _python_blocks(source)
    else:
        blocks = _brace_blocks(source)

    if not blocks:
        blocks = [source]

    chunks: list[TextChunk] = []
    for block in blocks:
        if len(block) <= chunk_size:
            if block.strip():
                chunks.append(
                    TextChunk(index=len(chunks), text=block.strip(), kind="code")
                )
            continue
        # 超长函数/类：退化为普通文本切分，但保留语言标记
        for piece in _split_with_overlap(block, chunk_size, chunk_overlap):
            if piece.strip():
                chunks.append(
                    TextChunk(index=len(chunks), text=piece.strip(), kind="code")
                )
    return chunks


# --------------------------------------------------------------------------- #
def _python_blocks(source: str) -> list[str]:
    """用 AST 精确切出顶层 def/class，函数体内的内容不会被拆散。"""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    spans: list[tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = getattr(node, "end_lineno", node.lineno)
            spans.append((start, end))

    if not spans:
        return []

    blocks: list[str] = []
    header_end = spans[0][0]
    if header_end > 0:
        header = "\n".join(lines[:header_end]).strip()
        if header:
            blocks.append(header)

    for start, end in spans:
        block = "\n".join(lines[start:end]).strip()
        if block:
            blocks.append(block)
    return blocks


def _brace_blocks(source: str) -> list[str]:
    """通用语言：按空行 + 顶层大括号启发式切分。"""
    blocks = [block.strip() for block in _PARAGRAPH.split(source) if block.strip()]
    merged: list[str] = []
    buffer = ""
    depth = 0
    for block in blocks:
        buffer = f"{buffer}\n\n{block}" if buffer else block
        depth += block.count("{") - block.count("}")
        if depth <= 0:
            merged.append(buffer)
            buffer = ""
            depth = 0
    if buffer:
        merged.append(buffer)
    return merged


def _split_blocks(text: str, chunk_size: int) -> list[str]:
    blocks: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            blocks.append(paragraph)
            continue
        # 段落过长 → 按句子聚合
        buffer = ""
        for sentence in _SENTENCE.split(paragraph):
            if not sentence:
                continue
            if len(buffer) + len(sentence) > chunk_size and buffer:
                blocks.append(buffer)
                buffer = sentence
            else:
                buffer += sentence
        if buffer:
            blocks.append(buffer)
    return blocks


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        pieces.append(text[start : start + chunk_size])
        start += step
    return pieces


def _merge_small(chunks: list[TextChunk], chunk_size: int, min_size: int) -> list[TextChunk]:
    merged: list[TextChunk] = []
    for chunk in chunks:
        if merged and len(merged[-1].text) < min_size:
            combined = merged[-1].text + "\n\n" + chunk.text
            if len(combined) <= chunk_size * 1.3:
                merged[-1] = TextChunk(
                    index=len(merged) - 1,
                    text=combined,
                    kind=merged[-1].kind,
                    metadata=merged[-1].metadata,
                )
                continue
        merged.append(TextChunk(index=len(merged), text=chunk.text, kind=chunk.kind))
    for position, chunk in enumerate(merged):
        chunk.index = position
    return merged
