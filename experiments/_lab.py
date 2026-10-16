"""实验脚本共用的小工具：路径引导 + 统一输出格式。

这里的每一个实验都满足同一份契约：

* **离线可跑**：不需要 API Key，不需要联网；
* **自解释**：先讲这个实验要回答什么问题，再跑，最后给结论；
* **可复现**：使用的都是确定性实现，同样的输入必然得到同样的输出。

运行方式（在项目根目录）：::

    python experiments/01_messages.py
    # 或者一次跑完
    python experiments/run_all.py
"""

from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDTH = 78


def title(text: str) -> None:
    """打印实验标题。"""
    print()
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def step(text: str) -> None:
    """打印一个小节标题。"""
    print(f"\n── {text} " + "─" * max(WIDTH - len(text) - 5, 0))


def note(*lines: str) -> None:
    """打印观察到的现象。"""
    for line in lines:
        print(f"   · {line}")


def conclusion(text: str) -> None:
    """打印本实验的结论。"""
    print(f"\n结论：{text}")
    print()


def kv_table(headers: list[str], rows: list[list[object]]) -> str:
    """极简表格渲染（不依赖第三方库，按显示宽度对齐中日韩字符）。"""
    cells = [[str(cell) for cell in row] for row in rows]
    widths = [
        max([_display_width(headers[index]), *(_display_width(row[index]) for row in cells)])
        for index in range(len(headers))
    ]
    lines = [
        "   " + "  ".join(_pad(headers[i], widths[i]) for i in range(len(headers))),
        "   " + "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    for row in cells:
        lines.append("   " + "  ".join(_pad(row[i], widths[i]) for i in range(len(headers))))
    return "\n".join(lines)


def _display_width(text: str) -> int:
    """东亚宽字符占 2 列，其余占 1 列。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def build_offline_registry(knowledge=None, memory=None, repo_root=None, allow_tests=False):
    """构造一个离线工具注册表 —— 实验里反复用到。"""
    from contextforge.tools import RepoAnalyzer, build_registry

    repo = RepoAnalyzer(repo_root) if repo_root else None
    return build_registry(
        knowledge=knowledge,
        memory=memory,
        repo=repo,
        allow_tests=allow_tests,
        default_root=repo_root or ROOT,
    )


SAMPLE_DOCS = ROOT / "examples" / "sample_docs"
SAMPLE_REPO = ROOT / "examples" / "sample_repo"


def offline_settings(tmp_path: Path):
    """一份完全离线的配置：数据写到临时目录，模型固定为离线实现。"""
    import dataclasses

    from contextforge.config import get_settings

    base = get_settings()
    settings = dataclasses.replace(
        base,
        data_dir=tmp_path,
        docs_dir=tmp_path / "documents",
        db_path=tmp_path / "lab.db",
        trace_dir=tmp_path / "traces",
        model=dataclasses.replace(base.model, api_key=""),
        # 浅拷贝会把内层配置共享出去，实验之间会互相污染，所以逐层复制
        context=dataclasses.replace(base.context),
        retrieval=dataclasses.replace(base.retrieval),
        agent=dataclasses.replace(base.agent),
    )
    settings.ensure_dirs()
    return settings


def build_offline_runtime(
    tmp_path: Path,
    sample_docs: Path | None = None,
    repo_root: Path | None = None,
    install_skills: bool = True,
):
    """构造一个完整可用的离线运行时（内存知识库 + 离线模型 + 代码仓库分析器）。"""
    from contextforge.llm.mock import MockLLM
    from contextforge.rag.pipeline import KnowledgeBase
    from contextforge.runtime import AgentRuntime
    from contextforge.storage import Database

    settings = offline_settings(tmp_path)
    db = Database(settings.db_path)
    llm = MockLLM()
    knowledge = KnowledgeBase(db, settings, llm=llm)
    if sample_docs is not None and Path(sample_docs).is_dir():
        knowledge.add_directory(sample_docs)
    return AgentRuntime(
        settings=settings,
        db=db,
        llm=llm,
        knowledge=knowledge,
        repo_root=repo_root,
        enable_tests=False,
        install_skills=install_skills,
    )


@contextmanager
def offline_runtime(
    sample_docs: Path | None = None,
    repo_root: Path | None = None,
    install_skills: bool = True,
):
    """用完即弃的离线运行时。

    做成上下文管理器不是洁癖：SQLite 的连接不关掉，Windows 上连临时目录
    都删不掉（`PermissionError: 另一个程序正在使用此文件`）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        runtime = build_offline_runtime(
            Path(tmp),
            sample_docs=sample_docs,
            repo_root=repo_root,
            install_skills=install_skills,
        )
        try:
            yield runtime
        finally:
            runtime.close()
