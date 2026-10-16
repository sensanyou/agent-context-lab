"""向量化模式与降级行为的测试。

这一组测试锁定的是一条具体的缺陷链路：

    有 API Key → 判定在线 → `supports_embeddings = True` 这个**声明**
    → 选远程向量化 → 请求一个并不存在的 /embeddings 端点 → 404

关键点是**不能把声明当证据**，必须真的探测一次；探测失败要降级，
而且降级必须可见（describe 里能读出来）。

所有测试都不联网：假的模型客户端在本地返回或抛错，
唯一涉及「真实网络」的一条用的是 127.0.0.1:1（连接立刻被拒）。
"""

from __future__ import annotations

import dataclasses

import pytest

from contextforge.config import ModelSettings, Settings
from contextforge.llm import MockLLM
from contextforge.rag import KnowledgeBase
from contextforge.rag.embeddings import (
    DEFAULT_DIM,
    EmbeddingError,
    HashingEmbedder,
    RemoteEmbedder,
    build_embedder,
    reset_probe_cache,
)
from contextforge.storage import Database


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class FakeEmbedLLM:
    """一个「声称支持 embeddings」的客户端；可以配置成失败。"""

    name = "fake-remote"
    online = True
    supports_embeddings = True
    embedding_model = "fake-embed-1"
    base_url = "https://fake.example.invalid/v1"
    model = "fake-embed-1"

    def __init__(self, dim: int = 8, fail: bool = False) -> None:
        self.dim = dim
        self.fail = fail
        self.calls = 0
        self.last_timeout: float | None = None

    def embed(self, texts, timeout=None):
        self.calls += 1
        self.last_timeout = timeout
        if self.fail:
            raise RuntimeError("Error code: 404 - Not Found")
        return [[0.25] * self.dim for _ in texts]

    def describe(self):
        return {"name": self.name, "online": True}


class NoTimeoutKeywordLLM(FakeEmbedLLM):
    """embed 没有 timeout 形参的第三方实现 —— 应能退化重试而不是崩掉。"""

    def embed(self, texts):  # type: ignore[override]
        self.calls += 1
        return [[0.5] * self.dim for _ in texts]


@pytest.fixture(autouse=True)
def _clean_probe_cache():
    """探测结果有进程级缓存，测试之间必须隔离。"""
    reset_probe_cache()
    yield
    reset_probe_cache()


def _settings(tmp_path, **model_overrides) -> Settings:
    base = ModelSettings()
    for key, value in model_overrides.items():
        setattr(base, key, value)
    return Settings(
        model=base,
        data_dir=tmp_path,
        docs_dir=tmp_path / "documents",
        db_path=tmp_path / "emb.db",
        trace_dir=tmp_path / "traces",
    )


# --------------------------------------------------------------------------- #
# 配置层
# --------------------------------------------------------------------------- #
def test_embedding_mode_defaults_to_auto():
    assert ModelSettings().normalized_embedding_mode == "auto"


@pytest.mark.parametrize("raw", ["LOCAL", " local ", "remote", "auto", "remote\n"])
def test_embedding_mode_is_normalized(raw):
    assert ModelSettings(embedding_mode=raw).normalized_embedding_mode == raw.strip().lower()


@pytest.mark.parametrize("raw", ["", "nonsense", "online", "None"])
def test_invalid_embedding_mode_falls_back_to_auto(raw):
    """写错值不应该抛异常，也不应该静默变成别的语义。"""
    assert ModelSettings(embedding_mode=raw).normalized_embedding_mode == "auto"


def test_embedding_endpoint_prefers_dedicated_config():
    model = ModelSettings(
        api_key="chat-key",
        base_url="https://chat.example.com",
        embedding_api_key="embed-key",
        embedding_base_url="https://embed.example.com/v1",
    )
    assert model.has_separate_embedding_endpoint
    assert model.embedding_endpoint() == ("https://embed.example.com/v1", "embed-key")


def test_embedding_endpoint_inherits_chat_config_by_default():
    model = ModelSettings(api_key="chat-key", base_url="https://chat.example.com")
    assert not model.has_separate_embedding_endpoint
    assert model.embedding_endpoint() == ("https://chat.example.com", "chat-key")


def test_env_overrides_are_read(monkeypatch):
    monkeypatch.setenv("CONTEXTFORGE_EMBEDDING_MODE", "remote")
    monkeypatch.setenv("CONTEXTFORGE_EMBEDDING_BASE_URL", "https://embed.example.com/v1")
    monkeypatch.setenv("CONTEXTFORGE_EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("CONTEXTFORGE_EMBEDDING_PROBE_TIMEOUT", "3.5")
    settings = Settings.from_env()
    assert settings.model.normalized_embedding_mode == "remote"
    assert settings.model.embedding_base_url == "https://embed.example.com/v1"
    assert settings.model.embedding_api_key == "embed-key"
    assert settings.model.embedding_probe_timeout == 3.5


# --------------------------------------------------------------------------- #
# build_embedder 的选择逻辑
# --------------------------------------------------------------------------- #
def test_local_mode_never_touches_the_network(tmp_path):
    llm = FakeEmbedLLM()
    embedder = build_embedder(llm, settings=_settings(tmp_path, embedding_mode="local"))
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.mode == "local"
    assert llm.calls == 0, "local 模式不应该发出任何请求"
    assert embedder.describe()["degraded"] is False


def test_offline_llm_stays_offline_and_is_not_marked_degraded(tmp_path):
    """MockLLM 不支持 embeddings —— 这是正常配置，不是降级。"""
    embedder = build_embedder(MockLLM(), settings=_settings(tmp_path))
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.degraded is False
    assert "degraded_reason" not in embedder.describe()


def test_auto_mode_uses_remote_when_the_probe_succeeds(tmp_path):
    llm = FakeEmbedLLM(dim=16)
    embedder = build_embedder(llm, dim=16, settings=_settings(tmp_path, embedding_mode="auto"))
    assert isinstance(embedder, RemoteEmbedder)
    assert llm.calls == 1, "应该恰好探测一次"
    assert embedder.describe()["mode"] == "remote"
    assert embedder.describe()["degraded"] is False


def test_auto_mode_degrades_when_endpoint_is_missing(tmp_path):
    """这就是 DeepSeek 那个 404：声明说支持，实际端点不存在。"""
    llm = FakeEmbedLLM(fail=True)
    embedder = build_embedder(llm, settings=_settings(tmp_path, embedding_mode="auto"))

    assert isinstance(embedder, HashingEmbedder)
    assert embedder.degraded is True
    info = embedder.describe()
    assert info["degraded"] is True
    assert "404" in info["degraded_reason"]
    assert "remote" in info["degraded_from"]
    assert info["dim"] == DEFAULT_DIM


def test_remote_mode_also_degrades_instead_of_breaking_startup(tmp_path):
    """显式要求 remote，但端点不可用时也降级 —— 系统始终要能启动。"""
    llm = FakeEmbedLLM(fail=True)
    embedder = build_embedder(llm, settings=_settings(tmp_path, embedding_mode="remote"))
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.degraded is True


def test_probe_result_is_cached_within_a_process(tmp_path):
    llm = FakeEmbedLLM()
    settings = _settings(tmp_path, embedding_mode="auto")
    build_embedder(llm, settings=settings)
    build_embedder(llm, settings=settings)
    assert llm.calls == 1, "同一进程内不应重复探测"

    reset_probe_cache()
    build_embedder(llm, settings=settings)
    assert llm.calls == 2


def test_probe_can_be_disabled_for_tests(tmp_path):
    llm = FakeEmbedLLM(fail=True)
    embedder = build_embedder(
        llm, settings=_settings(tmp_path, embedding_mode="auto"), probe=False
    )
    assert isinstance(embedder, RemoteEmbedder)
    assert llm.calls == 0


def test_probe_timeout_is_passed_through(tmp_path):
    llm = FakeEmbedLLM()
    build_embedder(llm, settings=_settings(tmp_path, embedding_probe_timeout=2.5))
    assert llm.last_timeout == 2.5


def test_third_party_embed_without_timeout_kwarg_still_works(tmp_path):
    llm = NoTimeoutKeywordLLM()
    embedder = build_embedder(llm, settings=_settings(tmp_path))
    assert isinstance(embedder, RemoteEmbedder)
    assert llm.calls == 1


def test_probe_failure_reason_is_trimmed_to_one_line(tmp_path):
    """原因与建议分开：status / 命令行输出才不会变成一大段。"""
    llm = FakeEmbedLLM(fail=True)
    embedder = build_embedder(llm, settings=_settings(tmp_path))
    info = embedder.describe()

    reason = info["degraded_reason"]
    assert "\n" not in reason
    assert len(reason) <= 200
    assert "提示：" not in reason, "长建议不应混进原因字段"
    assert "404" in reason

    hint = info["degraded_hint"]
    assert "CONTEXTFORGE_EMBEDDING_MODE" in hint
    assert hint != reason


# --------------------------------------------------------------------------- #
# 独立向量端点（聊天与向量走不同服务）
# --------------------------------------------------------------------------- #
def test_separate_embedding_endpoint_is_used_and_degrades_on_failure(tmp_path):
    """端口 1 一定有连接被拒 —— 用它验证「单独指向另一个服务」这条分支。"""
    settings = _settings(
        tmp_path,
        api_key="chat-key",
        base_url="https://chat.example.invalid/v1",
        embedding_base_url="http://127.0.0.1:1/v1",
        embedding_api_key="embed-key",
        embedding_probe_timeout=2.0,
    )
    embedder = build_embedder(MockLLM(), settings=settings)
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.degraded is True


# --------------------------------------------------------------------------- #
# RemoteEmbedder 本身
# --------------------------------------------------------------------------- #
def test_remote_embedder_normalizes_and_reports_real_dim():
    embedder = RemoteEmbedder(FakeEmbedLLM(dim=32))
    matrix = embedder.embed(["甲", "乙"])
    assert matrix.shape == (2, 32)
    assert embedder.dim == 32, "真实维度要在第一次调用后回填"
    norms = (matrix**2).sum(axis=1) ** 0.5
    assert all(abs(float(n) - 1.0) < 1e-5 for n in norms)


def test_remote_embedder_empty_input_returns_empty_matrix():
    embedder = RemoteEmbedder(FakeEmbedLLM(dim=4), dim=4)
    assert embedder.embed([]).shape == (0, 4)

    # 没声明维度时只能用默认值 —— 真实维度要等第一次真实调用才知道
    unknown = RemoteEmbedder(FakeEmbedLLM(dim=4))
    assert unknown.embed([]).shape == (0, DEFAULT_DIM)
    unknown.embed(["探测一下"])
    assert unknown.embed([]).shape == (0, 4)


def test_remote_embedder_error_carries_actionable_hint():
    embedder = RemoteEmbedder(FakeEmbedLLM(fail=True))
    with pytest.raises(EmbeddingError) as error:
        embedder.embed(["触发失败"])
    message = str(error.value)
    assert "CONTEXTFORGE_EMBEDDING_MODE" in message
    assert "local" in message


# --------------------------------------------------------------------------- #
# 端到端：知识库在降级后依然完全可用
# --------------------------------------------------------------------------- #
def test_knowledge_base_still_indexes_and_searches_after_degradation(tmp_path):
    db = Database(":memory:")
    settings = _settings(tmp_path, embedding_mode="auto")
    knowledge = KnowledgeBase(db, settings, llm=FakeEmbedLLM(fail=True))

    assert knowledge.stats()["embedder"]["degraded"] is True

    knowledge.add_text(
        "上下文压缩必须以轮次为单位，否则 assistant(tool_calls) 与 tool 结果会被拆开。",
        title="上下文工程指南",
    )
    hits = knowledge.search("上下文压缩 轮次", top_k=3)
    assert hits, "降级之后检索链路必须仍然可用"
    assert "轮次" in hits[0].chunk.text
    db.close()


def test_status_exposes_embedding_degradation(tmp_path):
    """降级不能是静默的：/api/status 的 embeddings 字段必须能读出来。"""
    from contextforge.runtime import AgentRuntime

    db = Database(":memory:")
    settings = _settings(tmp_path, embedding_mode="auto")
    settings.ensure_dirs()
    knowledge = KnowledgeBase(db, settings, llm=FakeEmbedLLM(fail=True))
    runtime = AgentRuntime(
        settings=settings, db=db, llm=MockLLM(), knowledge=knowledge, enable_tests=False
    )

    embeddings = runtime.status()["embeddings"]
    assert embeddings["mode"] == "local"
    assert embeddings["degraded"] is True
    assert embeddings["degraded_reason"]
    runtime.close()


def test_modes_are_preserved_when_settings_are_copied(tmp_path):
    """dataclasses.replace 之后新字段不能丢 —— 否则运行时开关会失效。"""
    settings = _settings(tmp_path, embedding_mode="local", embedding_api_key="k")
    copy = dataclasses.replace(settings)
    assert copy.model.normalized_embedding_mode == "local"
    assert copy.model.embedding_api_key == "k"
