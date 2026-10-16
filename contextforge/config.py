"""集中配置。

设计要点：

1. 缺失 API Key **不再抛异常**。项目默认可以以 offline 模式启动，
   此时使用确定性 Mock 模型，保证整条链路（检索 → 编排 → 工具 → 上下文）
   在没有网络和密钥的环境中依然可以跑通、可以测试、可以演示。
2. 所有可调参数集中在一处，方便做「不同策略的对比实验」。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

try:  # python-dotenv 是可选依赖
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 仅在极简环境触发
    def load_dotenv(*_args, **_kwargs):  # type: ignore[misc]
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().strip('"').strip("'")
    return value or default


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "").lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


#: 向量化的三种模式，含义见 `ModelSettings.embedding_mode`。
EMBEDDING_MODES = ("auto", "local", "remote")


@dataclass
class ModelSettings:
    """模型与计费配置。"""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.2
    request_timeout: float = 60.0

    # -- 向量化：与「聊天模型」相互独立的一条开关 -------------------------- #
    # 为什么要独立？因为「有 API Key」不等于「这个服务提供 /embeddings 接口」。
    # 例如 DeepSeek 只提供 chat，没有 embeddings —— 一旦跟着聊天模式一起切到
    # 远程，导入文档会直接 404。所以这里单独给一个模式：
    #   auto   —— 探测远程是否可用，可用就用；不可用自动降级到离线（默认）
    #   local  —— 强制离线 hashing 向量：零网络、零费用、结果确定
    #   remote —— 强制远程：需要服务确实提供 `/embeddings`
    embedding_mode: str = "auto"
    # 可选的独立向量服务端点：聊天用一个服务、向量用另一个服务时填写。
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    # 探测远程向量端点时的超时（秒）。端点不可达时不应让启动卡住。
    embedding_probe_timeout: float = 10.0

    # 用于成本估算（元 / 千 token），默认取一个量级参考值，可在 .env 覆盖。
    price_in_per_1k: float = 0.001
    price_out_per_1k: float = 0.002

    @property
    def online(self) -> bool:
        return bool(self.api_key)

    @property
    def normalized_embedding_mode(self) -> str:
        """归一化向量化模式；非法值按 auto 处理，不抛异常。"""
        mode = (self.embedding_mode or "").strip().lower()
        return mode if mode in EMBEDDING_MODES else "auto"

    @property
    def has_separate_embedding_endpoint(self) -> bool:
        """是否单独配置了向量服务地址。"""
        return bool(self.embedding_base_url or self.embedding_api_key)

    def embedding_endpoint(self) -> tuple[str, str]:
        """返回向量化实际使用的 (base_url, api_key)。"""
        return (
            self.embedding_base_url or self.base_url,
            self.embedding_api_key or self.api_key,
        )


@dataclass
class ContextSettings:
    """上下文工程配置。"""

    token_budget: int = 6000
    reserved_output_tokens: int = 800
    # 预算在各类信息之间的分配比例（总和应 <= 1）
    # 「current」是当前用户问题，必须单独留额度，否则压缩时会把问题本身裁掉。
    ratio_system: float = 0.08
    ratio_memory: float = 0.10
    ratio_retrieved: float = 0.26
    ratio_history: float = 0.30
    ratio_tools: float = 0.10
    ratio_current: float = 0.16

    keep_recent_turns: int = 6
    summary_max_chars: int = 1200
    strategy: str = "hybrid"  # none | sliding | summary | hybrid | llm-summary
    enforce_tool_integrity: bool = True

    def allocation(self) -> dict[str, int]:
        """把 token 预算拆成各段可用额度（已扣除输出预留）。"""
        usable = max(self.token_budget - self.reserved_output_tokens, 0)
        raw = {
            "system": self.ratio_system,
            "memory": self.ratio_memory,
            "retrieved": self.ratio_retrieved,
            "history": self.ratio_history,
            "tools": self.ratio_tools,
            "current": self.ratio_current,
        }
        total = sum(raw.values()) or 1.0
        return {key: int(usable * value / total) for key, value in raw.items()}


@dataclass
class RetrievalSettings:
    """混合检索配置。"""

    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_dim: int = 512

    dense_top_k: int = 12
    sparse_top_k: int = 12
    fusion_top_k: int = 16
    final_top_k: int = 5
    fusion: str = "rrf"  # rrf | weighted
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    rerank: bool = True
    min_score: float = 0.0


@dataclass
class AgentSettings:
    max_iterations: int = 6
    max_tool_calls: int = 8
    require_approval: bool = False  # True 时进入人工确认断点
    enable_reflection: bool = True
    tool_result_max_chars: int = 4000


@dataclass
class Settings:
    model: ModelSettings = field(default_factory=ModelSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)

    data_dir: Path = PROJECT_ROOT / "data"
    docs_dir: Path = PROJECT_ROOT / "data" / "documents"
    db_path: Path = PROJECT_ROOT / "data" / "contextforge.db"
    trace_dir: Path = PROJECT_ROOT / "data" / "traces"
    web_dir: Path = PROJECT_ROOT / "web"

    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls(
            model=ModelSettings(
                api_key=_env_str("CONTEXTFORGE_API_KEY", _env_str("DEEPSEEK_API_KEY")),
                base_url=_env_str(
                    "CONTEXTFORGE_BASE_URL",
                    _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                ),
                model=_env_str(
                    "CONTEXTFORGE_MODEL",
                    _env_str("DEEPSEEK_MODEL", "deepseek-chat"),
                ),
                embedding_model=_env_str("CONTEXTFORGE_EMBEDDING_MODEL", "text-embedding-3-small"),
                embedding_mode=_env_str("CONTEXTFORGE_EMBEDDING_MODE", "auto"),
                embedding_base_url=_env_str("CONTEXTFORGE_EMBEDDING_BASE_URL", ""),
                embedding_api_key=_env_str("CONTEXTFORGE_EMBEDDING_API_KEY", ""),
                embedding_probe_timeout=_env_float("CONTEXTFORGE_EMBEDDING_PROBE_TIMEOUT", 10.0),
                temperature=_env_float("CONTEXTFORGE_TEMPERATURE", 0.2),
                price_in_per_1k=_env_float("CONTEXTFORGE_PRICE_IN", 0.001),
                price_out_per_1k=_env_float("CONTEXTFORGE_PRICE_OUT", 0.002),
            ),
            context=ContextSettings(
                token_budget=_env_int("CONTEXTFORGE_TOKEN_BUDGET", 6000),
                keep_recent_turns=_env_int("CONTEXTFORGE_KEEP_TURNS", 6),
                strategy=_env_str("CONTEXTFORGE_STRATEGY", "hybrid"),
                enforce_tool_integrity=_env_bool("CONTEXTFORGE_TOOL_INTEGRITY", True),
            ),
            retrieval=RetrievalSettings(
                chunk_size=_env_int("CONTEXTFORGE_CHUNK_SIZE", 800),
                chunk_overlap=_env_int("CONTEXTFORGE_CHUNK_OVERLAP", 120),
                final_top_k=_env_int("CONTEXTFORGE_TOP_K", 5),
                rerank=_env_bool("CONTEXTFORGE_RERANK", True),
            ),
            agent=AgentSettings(
                max_iterations=_env_int("CONTEXTFORGE_MAX_ITERATIONS", 6),
                require_approval=_env_bool("CONTEXTFORGE_REQUIRE_APPROVAL", False),
            ),
            data_dir=Path(_env_str("CONTEXTFORGE_DATA_DIR", str(PROJECT_ROOT / "data"))),
            debug=_env_bool("CONTEXTFORGE_DEBUG", False),
        )
        settings.docs_dir = settings.data_dir / "documents"
        settings.db_path = settings.data_dir / "contextforge.db"
        settings.trace_dir = settings.data_dir / "traces"
        settings.ensure_dirs()
        return settings

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.docs_dir, self.trace_dir):
            path.mkdir(parents=True, exist_ok=True)

    def with_overrides(self, **kwargs) -> "Settings":
        """返回一份参数被覆盖的副本，供对比实验使用。"""
        return replace(self, **kwargs)


_default: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """进程级单例配置。"""
    global _default
    if _default is None or refresh:
        _default = Settings.from_env()
    return _default
