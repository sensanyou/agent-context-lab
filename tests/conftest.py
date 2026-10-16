"""pytest 公共夹具。

所有测试都在**离线**状态下运行：不联网、不读真实 API Key、不产生费用。
这正是「核心链路零外部依赖」这条设计原则的价值 —— 测试可以跑得很快，
结果也完全可复现。
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextforge.config import Settings, get_settings  # noqa: E402
from contextforge.llm import MockLLM  # noqa: E402
from contextforge.rag import KnowledgeBase  # noqa: E402
from contextforge.runtime import AgentRuntime  # noqa: E402
from contextforge.storage import Database  # noqa: E402

SAMPLE_DOCS = ROOT / "examples" / "sample_docs"
SAMPLE_REPO = ROOT / "examples" / "sample_repo"


@pytest.fixture(scope="session")
def sample_docs() -> Path:
    return SAMPLE_DOCS


@pytest.fixture(scope="session")
def sample_repo() -> Path:
    return SAMPLE_REPO


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = get_settings()
    scoped = dataclasses.replace(
        base,
        data_dir=tmp_path,
        docs_dir=tmp_path / "documents",
        db_path=tmp_path / "test.db",
        trace_dir=tmp_path / "traces",
        # replace 是浅拷贝：内层配置也要复制，否则测试之间的改动会互相污染
        model=dataclasses.replace(base.model),
        context=dataclasses.replace(base.context),
        retrieval=dataclasses.replace(base.retrieval),
        agent=dataclasses.replace(base.agent),
    )
    # 显式关掉联网模型，保证测试不会意外发出真实请求
    scoped.model.api_key = ""
    scoped.ensure_dirs()
    return scoped


@pytest.fixture
def db() -> Database:
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def kb(db: Database, settings: Settings, llm: MockLLM) -> KnowledgeBase:
    knowledge = KnowledgeBase(db, settings, llm=llm)
    if SAMPLE_DOCS.is_dir():
        knowledge.add_directory(SAMPLE_DOCS)
    return knowledge


@pytest.fixture
def runtime(settings: Settings, db: Database, kb: KnowledgeBase) -> AgentRuntime:
    """完整运行时：离线模型 + 内存数据库 + 已导入示例文档的知识库。"""
    instance = AgentRuntime(
        settings=settings, db=db, llm=MockLLM(), knowledge=kb, repo_root=SAMPLE_REPO,
        enable_tests=False,
    )
    yield instance


@pytest.fixture
def bare_runtime(settings: Settings, db: Database) -> AgentRuntime:
    """空知识库的运行时，用于验证「没有资料时也能正常降级」。"""
    empty = KnowledgeBase(db, settings, llm=MockLLM())
    return AgentRuntime(
        settings=settings, db=db, llm=MockLLM(), knowledge=empty, enable_tests=False
    )
