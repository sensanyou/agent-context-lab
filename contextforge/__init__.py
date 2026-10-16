"""ContextForge — Knowledge · Agent · Context Engineering.

一个面向个人学习、代码理解与技术研究的智能助手，覆盖主流 Agent 工程技术的完整实现：

    contextforge.llm         模型适配层（OpenAI 兼容 / 离线 Mock）
    contextforge.context     上下文工程引擎（预算、压缩、工具链完整性）
    contextforge.rag         混合检索（向量 + BM25 + 重排序）
    contextforge.memory      长期记忆与技能库（SQLite 持久化）
    contextforge.tools       工具注册与执行（含代码仓库分析）
    contextforge.graph       图编排运行时（条件分支、循环、断点续跑）
    contextforge.evaluation  执行轨迹、指标与策略对比实验
    contextforge.api         HTTP 接口与业务门面

设计原则：核心链路零外部服务依赖，没有 API Key 也能端到端运行。
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
