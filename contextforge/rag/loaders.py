"""文档加载：把磁盘上的文件变成结构化文本。

支持的格式全部走「纯标准库 + 可选依赖」路线：

| 格式 | 实现 |
| --- | --- |
| .md / .txt / .rst / .log / .csv / .json / .yaml / .toml / 源码 | 直接读取 |
| .html / .htm | 去标签 + 去脚本样式 |
| .pdf | 可选 `pypdf`，未安装时明确跳过而不是静默失败 |

跳过 `.git` / `.venv` / `node_modules` / `__pycache__` 等噪声目录，
避免把依赖源码灌进知识库。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

TEXT_SUFFIXES = {
    ".md", ".markdown", ".txt", ".rst", ".log", ".csv", ".tsv",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
}
CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".vue", ".java", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".sh", ".sql",
}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}

SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "dist", "build", ".idea", ".vscode", "data",
}

CODE_FILE_KIND = "code"
TEXT_FILE_KIND = "text"


@dataclass
class LoadedFile:
    path: Path
    title: str
    text: str
    media_type: str = "text/plain"
    kind: str = TEXT_FILE_KIND
    metadata: dict = field(default_factory=dict)
    skipped_reason: str = ""


def load_file(path: str | Path) -> LoadedFile:
    """读取单个文件。无法解析时返回带 `skipped_reason` 的结果，不抛异常。"""
    path = Path(path)
    if not path.exists():
        return _skip(path, "文件不存在")
    if path.is_dir():
        return _skip(path, "这是一个目录")

    suffix = path.suffix.lower()
    size = path.stat().st_size
    if size > 4 * 1024 * 1024:
        return _skip(path, f"文件过大（{size / 1024 / 1024:.1f} MB）")

    try:
        if suffix in PDF_SUFFIXES:
            return _load_pdf(path)
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _skip(path, f"读取失败：{exc}")

    if suffix in HTML_SUFFIXES:
        return LoadedFile(
            path=path,
            title=path.name,
            text=_strip_html(raw),
            media_type="text/html",
            kind=TEXT_FILE_KIND,
            metadata={"suffix": suffix, "bytes": size},
        )

    if suffix in CODE_SUFFIXES:
        return LoadedFile(
            path=path,
            title=path.name,
            text=raw,
            media_type="text/x-source",
            kind=CODE_FILE_KIND,
            metadata={"suffix": suffix, "bytes": size, "language": suffix.lstrip(".")},
        )

    return LoadedFile(
        path=path,
        title=path.name,
        text=raw,
        media_type="text/markdown" if suffix in {".md", ".markdown"} else "text/plain",
        kind=TEXT_FILE_KIND,
        metadata={"suffix": suffix, "bytes": size},
    )


def iter_files(
    root: str | Path,
    suffixes: set[str] | None = None,
    max_files: int = 500,
) -> Iterator[Path]:
    """递归遍历目录，跳过噪声目录。"""
    root = Path(root)
    allowed = suffixes or (TEXT_SUFFIXES | CODE_SUFFIXES | HTML_SUFFIXES | PDF_SUFFIXES)
    count = 0

    for path in sorted(root.rglob("*")):
        if count >= max_files:
            return
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in allowed:
            continue
        count += 1
        yield path


def file_id(path: str | Path) -> str:
    digest = hashlib.blake2b(str(Path(path).resolve()).encode("utf-8"), digest_size=8).hexdigest()
    return f"doc_{digest}"


def _skip(path: Path, reason: str) -> LoadedFile:
    return LoadedFile(
        path=Path(path), title=Path(path).name, text="", skipped_reason=reason
    )


def _load_pdf(path: Path) -> LoadedFile:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return _skip(path, "未安装 pypdf，无法解析 PDF（pip install pypdf 后重试）")

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        return _skip(path, f"PDF 解析失败：{exc}")

    text = "\n\n".join(f"[第 {i + 1} 页]\n{p}" for i, p in enumerate(pages) if p)
    return LoadedFile(
        path=path,
        title=path.name,
        text=text,
        media_type="application/pdf",
        kind=TEXT_FILE_KIND,
        metadata={"pages": len(pages)},
    )


_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_BLANK = re.compile(r"\n{3,}")


def _strip_html(raw: str) -> str:
    text = _SCRIPT_STYLE.sub(" ", raw)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</(p|div|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = _TAG.sub(" ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&(amp|lt|gt|quot);", lambda m: {"amp": "&", "lt": "<", "gt": ">", "quot": '"'}[m.group(1)], text)
    text = re.sub(r"[ \t]+", " ", text)
    return _BLANK.sub("\n\n", text).strip()
