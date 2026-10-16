# ContextForge

**Knowledge · Agent · Context Engineering**

> 一个把「上下文工程」当作一等公民来对待的 Agent 项目。
> 它不只做上下文压缩，而是把 **检索、记忆、工具、编排、评测** 串成一条完整链路，
> 并且把每一个技术取舍都用**可复现的数字**说清楚。
>
> **不需要 API Key，不需要联网，克隆下来就能跑通全链路。**

```bash
pip install numpy
python -m contextforge demo        # 端到端跑四种任务
python -m contextforge bench       # 生成评测报告 data/report.md
python -m contextforge serve       # 打开 Vue 3 前端
python experiments/run_all.py      # 跑完 9 个可复现实验
```

---

## 一、先看数字：省 token 的代价是什么

这是整个项目最有说服力的一组数据。同一段 30 轮的长历史（2758 token），
同一份预算（6000 token），五种压缩策略：

| 压缩策略 | 压缩后 token | 压缩率 | 硬约束保留率 | 关键事实保留率 | 工具链完整 |
| --- | --- | --- | --- | --- | --- |
| `sliding` 滑动窗口 | **544** | 0.20 | **33%** | 50% | 是 |
| `llm-summary` 模型摘要 | 681 | 0.25 | 67% | 100% | 是 |
| `summary` 抽取式摘要 | 1271 | 0.46 | 100% | 100% | 是 |
| `hybrid` 钉约束+摘要+近期 | 1296 | 0.47 | **100%** | **100%** | 是 |
| `none` 不压缩 | 1682 | 0.61 | **33%** | 50% | 是 |

三个能直接拿去讲结论的观察：

1. **「不压缩」不是安全选项。** 它 token 花得最多（1682），约束保留率却和
   最激进的滑动窗口一样只有 33% —— 因为一旦超预算，兜底降级照样从最旧的开始丢。
   不压缩 = 又贵又丢信息。
2. **滑动窗口把 token 压到 20%**（544 / 2758，省掉 80%），
   代价是硬约束只剩 33%：「我的长期目标是……」「必须用中文回答」
   这类早期目标，恰好是最早被丢掉的。
3. **混合策略比「不压缩」省 23% 的 token，约束保留率还从 33% 提到 100%。**
   差别不在于「压不压」，而在于它**显式建模了什么不该被压掉**
   （钉住关键约束 → 摘要旧轮次 → 保留近期原文）。

> 复现：`python -m contextforge bench` → `data/report.md`
> 可视化复现：`python experiments/05_context_compression.py`

再做一次诚实声明：这张表里**最难的是「none 会丢信息」这一行**。
它不是一个好看的结论，而恰恰说明为什么需要把取舍量化 ——
凭直觉你会以为「不压缩最安全」。

---

## 二、检索策略对比

7 个用例 × 3 种检索模式（文档粒度指标）：

| 模式 | P@5 | R@5 | MRR | 要点覆盖率 | 工具命中率 | 平均 token | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dense` 纯向量 | 0.345 | 0.857 | 0.786 | 0.809 | 1.000 | 1234 | 12.2 ms |
| `sparse` 纯 BM25 | **0.369** | 0.857 | **0.857** | **1.000** | 1.000 | 1273 | 12.4 ms |
| `hybrid` 融合+重排 | 0.357 | 0.857 | 0.762 | 0.857 | 1.000 | 1248 | 12.5 ms |

**这里必须诚实**：在这个离线环境里 `hybrid` 并没有明显胜出。

原因是离线的向量实现是 **hashing 向量**（blake2b 分桶 + 符号），
它本质上仍是字面表示，不做语义泛化 —— 所以它和 BM25 的能力高度重叠，
融合自然拿不到额外收益。真实的语义 embedding 才能让「换个说法也能召回」
这件事生效。

那为什么还保留这条链路？因为它让**整条 RAG 管线**（分片 → 向量 → BM25 →
RRF 融合 → 重排 → 引用）在没有 API Key 的情况下依然能被完整跑通和测试；
换成真实 embedding 只需要替换 `Embedder` 的一个实现，接口不变。

> 这条「知道自己哪里不行」的说明，比一个虚高的对比表更有价值。

---

## 三、这个项目在解决什么问题

大部分 Agent 项目能跑通 demo，但答不上来这三个问题：

| 问题 | ContextForge 的回答 |
| --- | --- |
| **模型看到的到底是什么？** | 每一轮都产出一份 `ContextReport`：谁被保留、谁被丢弃、为什么 |
| **怎么证明我的方案更好？** | 检索三模式 × 压缩五策略的对比实验，指标全部本地确定性计算 |
| **换个模型/换个策略会不会崩？** | 所有外部依赖（模型、向量、存储）都是可替换实现，缺了会降级而不是崩溃 |

围绕这三点，项目分成六个能力模块：

| 模块 | 做什么 | 关键设计 |
| --- | --- | --- |
| **上下文工程** | 分区预算、策略压缩、工具链校验、上下文报告 | 压缩以**轮次**为最小单位；六分区额度互不挤占 |
| **混合检索** | 向量 + BM25 → RRF 融合 → 启发式重排 | 选 RRF 是因为它只用排名，不受分数量纲漂移影响 |
| **记忆层** | 会话记忆 / 长期记忆 / 技能库 | 长期记忆按「词面+重要度+时间衰减×类型权重×使用反馈」打分 |
| **工具层** | 函数签名自动生成 schema 的安全执行 | 失败用 `ToolFailure` 显式表达，重试路由据此决策 |
| **图编排** | StateGraph + 条件边 + SQLite 检查点 | 工具返回后**必须重新装配上下文**；支持中断/恢复 |
| **评测与可观测** | 指标计算、策略对比、执行轨迹 | 所有指标本地确定性计算，可复核 |

---

## 四、技术选型：每一项都在解决一个具体问题

不是「用了哪些技术」，而是「为什么需要它」。

| 技术 | 解决的问题 | 不用它会怎样 |
| --- | --- | --- |
| **LangGraph 式图运行时**（自研） | Agent 循环需要循环 + 条件分支 + 可中断 + 可恢复 | 用 while 循环写：无法暂停、无法恢复、无法可视化 |
| **RAG（dense + BM25 + RRF）** | 单路检索的失败模式互补 | 只有向量 → 专有名词/错误码漏检；只有 BM25 → 换个说法就找不到 |
| **重排序（Reranker）** | 融合只解决「哪一路更可信」，不解决片段与问题的贴合度 | Top-k 里混入大量相关但不对题的片段 |
| **长期记忆** | 「什么该跨任务留存」是独立问题 | 全塞进知识库 → 偏好被文档淹没；全塞进历史 → 会话一结束就没了 |
| **技能库** | 反复出现的任务应当复用流程而非重新规划 | 每次都从零规划，行为不稳定 |
| **SQLite 检查点** | 进程退出后任务状态不能丢 | 人工确认、长任务、失败重试全都做不了 |
| **可观测（轨迹 + 指标）** | 「为什么慢/贵/答错」需要证据 | 只能靠猜；也无法产出对比数据 |
| **零依赖 HTTP 服务 + Vue 3** | 需要能演示的界面，但不能被环境卡住 | 装不上 FastAPI 就打不开界面 |

**关于「自研 vs 用框架」**：项目刻意**没有**引入 LangChain / LangGraph / Chroma /
FastAPI 的硬依赖，但完全对齐它们的抽象命名（`StateGraph` / `add_node` /
`add_conditional_edges` / `checkpoint` / `Embedder` / `VectorStore`）。
原因很实际：`pip install` 会失败、网络会断、版本会冲突，
而一个「打不开的实验」等于没有实验。接真实框架时，节点函数与状态定义可以原样搬迁。

---

## 五、架构

```
                      ┌──────────────────────────────┐
   CLI / Vue3 Web ───▶│  API 层（零依赖 HTTP + SSE）  │
                      └──────────────┬───────────────┘
                                     │
                      ┌──────────────▼───────────────┐
                      │      AgentRuntime（门面）     │
                      └──────────────┬───────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
┌───────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
│  图编排运行时   │          │  上下文工程引擎  │          │   混合检索       │
│ StateGraph     │─────────▶│ 分区预算/压缩/   │◀─────────│ dense+BM25+RRF  │
│ 条件边/检查点   │          │ 工具链校验/报告  │          │ +重排序          │
└───────┬────────┘          └────────┬────────┘          └────────┬────────┘
        │                            │                            │
        │                   ┌────────▼────────┐          ┌────────▼────────┐
        │                   │    记忆层        │          │   SQLite 存储    │
        │                   │ 会话/长期/技能   │          │  向量/BM25/轨迹  │
        │                   └─────────────────┘          └─────────────────┘
        │                            │
┌───────▼────────┐          ┌────────▼────────┐          ┌─────────────────┐
│    工具层       │          │   模型适配层     │          │  评测与可观测    │
│ 签名→Schema     │          │ 在线/离线可换    │          │ 指标/轨迹/报告   │
└────────────────┘          └─────────────────┘          └─────────────────┘
```

**关键设计**：上下文管理在中间，是**所有能力的公共依赖** ——
检索要往预算里塞片段，工具结果要占预算，记忆要注入，历史要压缩，
它们全部走同一个 `ContextEngine`。这就是「把上下文工程做成 Agent Runtime
公共能力」的具体落点。

> 每个设计决策的来龙去脉（为什么这么选、代价是什么、替代方案为什么不选）
> 记在 [ARCHITECTURE.md](ARCHITECTURE.md) 里。

工作流本身可以随时导出（`python -m contextforge workflow`）：

```
prepare ─┬─▶ retrieve ─┐
         └─▶ build_context ◀─┐
                   │         │
                   ▼         │
                 reason ─┬─▶ tools ──┘   （工具结果必须重新装配上下文）
                         ├─▶ reflect ─┐
                         └─▶ finalize ◀┘
```

```bash
python -m contextforge workflow                  # Mermaid（可直接贴 README）
python -m contextforge workflow --format text    # 节点 / 固定边 / 条件边
python -m contextforge workflow --format json    # 给前端画图
```

---

## 六、快速开始

### 1. 安装

```bash
git clone <repo-url> && cd agent-context-lab

python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS / Linux
```

**只有 `numpy` 是必需的。** 其余依赖（`openai`、`tiktoken`、`pypdf`、
`fastapi`）都是可选增强项，不装也能完整运行。

### 2. 跑起来（三种方式）

```bash
# ① 端到端体验：导入示例文档 → 跑 4 种任务 → 打印节点路径与压缩率
python -m contextforge demo

# ② 生成评测报告（写入 data/report.md）
python -m contextforge bench

# ③ 打开 Web 界面（离线可用的 Vue 3 单页应用）
python -m contextforge serve --port 8000
#  → http://127.0.0.1:8000
```

### 3. 命令行用法

```bash
python -m contextforge status                       # 系统自检
python -m contextforge workflow                     # 导出工作流图（Mermaid）
python -m contextforge ingest examples/sample_docs  # 导入文档
python -m contextforge search "RRF 融合" --mode hybrid
python -m contextforge ask "上下文压缩为什么要以轮次为单位？"
python -m contextforge ask "计算 (128 + 72) * 3" --trace
python -m contextforge chat                          # 交互式对话
```

### 4. 接入真实模型（可选）

不配 Key 时自动使用**离线确定性模型**，全链路照常运行。
要接真实模型，设置环境变量即可，代码无需改动：

```env
CONTEXTFORGE_API_KEY=sk-xxx
CONTEXTFORGE_BASE_URL=https://api.deepseek.com
CONTEXTFORGE_MODEL=deepseek-chat
CONTEXTFORGE_EMBEDDING_MODEL=text-embedding-3-small
```

> 注意：`settings.model.api_key` 为空时**不会抛异常**，而是降级到离线模型，
> 因此整个项目在没有 Key、没有网络的环境下依然可以完整运行。

#### 向量化是**独立**的一条开关

「有 Key」不等于「这个服务提供 `/embeddings` 接口」。例如 DeepSeek 只提供
chat，没有 embeddings —— 如果向量化跟着聊天模式一起切到远程，导入文档会直接
404。所以向量化单独配置：

```env
CONTEXTFORGE_EMBEDDING_MODE=auto     # auto（默认）| local | remote
CONTEXTFORGE_EMBEDDING_BASE_URL=     # 可选：向量单独用另一个服务
CONTEXTFORGE_EMBEDDING_API_KEY=      # 可选
CONTEXTFORGE_EMBEDDING_PROBE_TIMEOUT=10
```

| 模式 | 行为 |
| --- | --- |
| `auto`（默认） | 启动时**真的发一次探测请求**；通了用远程，不通自动降级到离线向量 |
| `local` | 强制离线 hashing 向量：零网络、零费用、结果确定 |
| `remote` | 强制远程；不可用时同样降级，而不是让系统起不来 |

降级**不是静默的**：`python -m contextforge status`、`/api/status` 的
`embeddings` 字段、以及 Web 的「系统状态」页都会显示
`degraded: true` 和具体原因。检索链路在降级后完全可用，只是退化为字面级匹配。

---

## 七、Web 界面

`web/index.html` 是一个 Vue 3 单页应用（Vue 运行时已本地化到
`web/vendor/vue.global.prod.js`，因此**完全离线可用**），六个标签页：

| 标签 | 内容 |
| --- | --- |
| **对话** | 提问 → SSE 流式显示节点执行过程 → 回答 + 引用 |
| **知识库** | 文档列表、导入、检索测试、切片预览 |
| **执行轨迹** | 每个节点的耗时、工具调用、token 用量、上下文报告 |
| **上下文预算** | 六个分区「额度 vs 实际占用」的柱状对比 |
| **评测对比** | 检索三模式 / 压缩五策略的指标表 |
| **系统状态** | 模型、向量、工具、记忆、工作流的自检信息 |

---

## 八、九个可复现实验

`experiments/` 不是示例代码，是**实验**：每个脚本先说清要回答什么问题，
再跑一遍，最后给出带数字的结论。全部离线、确定性、可复现。

```bash
python experiments/run_all.py        # 全部 9 个，约 13 秒
python experiments/05_context_compression.py   # 单独跑
```

| # | 主题 | 一句话结论 |
| --- | --- | --- |
| 01 | 消息、轮次与工具链完整性 | 压缩的最小单位是**轮次**，不是消息 |
| 02 | 模型适配层 | 没有 API Key 时应该**降级**，而不是崩溃 |
| 03 | Token 计数与分区预算 | 六份额度，谁超支只在谁自己的额度里裁 |
| 04 | 提示注入 | 防线在于**区分指令与数据**，不在于过滤坏词 |
| 05 | 压缩策略对比 | 不压缩并不安全：token 最贵，约束照样丢 |
| 06 | 混合检索 | 选 RRF 是因为它只用排名，不受量纲漂移影响 |
| 07 | 图编排 | 每次工具返回都必须**重新装配上下文** |
| 08 | RAG ≠ 记忆 | 四种机制回答四个不同的问题 |
| 09 | 评测 | 把「效果更好」变成一张可复核的表 |

详见 [experiments/README.md](experiments/README.md)。

---

## 九、测试

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
# 242 passed
```

全部测试**离线运行**，不依赖网络与 API Key。覆盖：

| 文件 | 覆盖范围 |
| --- | --- |
| `test_context.py` | 轮次切分、工具链校验、token 计数、预算裁剪、五种压缩策略 |
| `test_rag.py` | 分片、embedding、BM25、向量库、RRF 融合、重排、知识库持久化 |
| `test_memory.py` | 会话记忆、长期记忆打分与降重、TTL 过期、技能库 |
| `test_tools.py` | schema 自动生成、参数类型、安全执行、仓库分析、失败表达 |
| `test_graph.py` | 图构建与校验、条件边、检查点、中断/恢复 |
| `test_runtime.py` | 端到端问答、工具循环重建上下文、策略覆盖、审批中断 |
| `test_evaluation.py` | 指标数学、运行器、报告渲染（含指标边界回归） |
| `test_api.py` | 处理器层 + 真实 HTTP 服务 + SSE 流式事件 |
| `test_cli.py` | 子命令注册、三种输出格式、无 API Key 下的可用性 |
| `test_embedding_mode.py` | 向量化三模式选择、探测、降级可见性、独立向量端点 |
| `test_document_identity.py` | 正文指纹去重、历史脏数据自愈、切片/向量/BM25 同步清理 |

---

## 十、目录结构

```text
README.md                      # 本文件
ARCHITECTURE.md                # 架构与技术决策（为什么这么做）
experiments/README.md          # 9 个实验的导读
data/report.md                 # 评测报告（由 bench 生成）

contextforge/                  # 核心包（零外部服务依赖）
├── config.py                  # 集中配置（缺 Key 不抛异常）
├── types.py                   # 消息与结构化对象
├── storage.py                 # SQLite（WAL / 事务 / 向量 BLOB）
├── runtime.py                 # AgentRuntime 门面：加载、编排、观测
├── cli.py                     # 命令行入口
├── llm/                       # ① 模型适配层
│   ├── base.py                #    BaseLLM 协议
│   ├── mock.py                #    离线确定性模型（工具决策/接地回答/摘要）
│   ├── openai_compat.py       #    OpenAI 兼容适配器（重试/用量）
│   └── factory.py             #    按配置选实现，失败降级
├── context/                   # ② 上下文工程引擎
│   ├── tokens.py              #    三级可插拔 token 计数
│   ├── turns.py               #    轮次切分（压缩的原子单位）
│   ├── validator.py           #    工具链完整性校验与修复
│   ├── budget.py              #    六分区预算分配
│   ├── strategies.py          #    五种压缩策略
│   └── engine.py              #    装配 + 报告
├── rag/                       # ③ 混合检索链路
│   ├── loaders.py             #    md/txt/py/pdf/html 读取
│   ├── chunker.py             #    段落 / 代码（AST）分片
│   ├── tokenize.py            #    中英混排分词（CJK 一元+二元）
│   ├── embeddings.py          #    hashing 向量 / 远程 embedding
│   ├── bm25.py                #    Okapi BM25（倒排索引）
│   ├── vector_store.py        #    内存向量库（numpy 矩阵）
│   ├── retriever.py           #    RRF / 加权融合 / 重排序
│   └── pipeline.py            #    KnowledgeBase 门面
├── memory/                    # ④ 记忆层
│   ├── conversation.py        #    会话记忆
│   ├── long_term.py           #    长期记忆（打分 / 降重 / TTL）
│   └── skills.py              #    技能库
├── tools/                     # ⑤ 工具层
│   ├── registry.py            #    签名 → JSON Schema、安全执行
│   ├── builtin.py             #    内置工具集
│   └── repo.py                #    代码仓库分析（AST / grep / outline）
├── graph/                     # ⑥ 图编排运行时
│   ├── state.py               #    AgentState 与合并语义
│   ├── engine.py              #    StateGraph / 检查点 / 中断恢复
│   ├── nodes.py               #    七个节点 + 条件路由
│   └── workflow.py            #    工作流装配与 Mermaid 导出
├── evaluation/                # ⑦ 评测与可观测性
│   ├── metrics.py             #    四组指标（检索/上下文/回答/运行）
│   ├── tracing.py             #    轨迹、span、聚合
│   └── runner.py              #    用例、套件、报告渲染
└── api/                       # ⑧ 服务层
    ├── handlers.py            #    业务处理器
    ├── devserver.py           #    零依赖 HTTP + SSE
    └── app.py                 #    可选 FastAPI 实现

web/                           # Vue 3 前端（Vue 运行时已本地化）
experiments/                   # 9 个可复现实验
examples/                      # 示例文档与示例代码仓库
tests/                         # 242 个离线测试
```

---

## 十一、已知边界

一个可靠的项目应该写清楚自己**做不到**什么：

| 边界 | 现状 | 怎么补 |
| --- | --- | --- |
| **离线向量无语义能力** | hashing 向量只做字面匹配，hybrid 优势被压低 | 接真实 embedding（换 `Embedder` 实现即可） |
| **离线模型是规则引擎** | 回答质量反映的是检索与装配质量，不是文案质量 | 配 API Key 即切换到真实模型 |
| **重排序是启发式的** | 四维加权，非交叉编码器 | 换成 `Reranker` 接口的模型实现 |
| **向量库在内存中** | 启动时从 SQLite 重建，百万级切片需要换后端 | `VectorStore` 协议已抽出，可接 FAISS/pgvector |
| **无用户体系** | 单用户本地运行 | 加鉴权层与多租户隔离 |
| **抽取式摘要** | 关键词优先 + 时间衰减，非生成式 | 已提供 `llm-summary` 策略走模型摘要 |

---

## 十二、后续路线

- [ ] 接入真实 RAGAS / LangSmith，把本地指标替换为服务端指标
- [ ] 用真实 LangGraph 替换自研图运行时（节点函数原样迁移）
- [ ] 加入交叉编码器重排序
- [ ] 向量库后端插件化（FAISS / pgvector / Milvus）
- [ ] 多模态上下文（图片、表格）
- [ ] 多 Agent 协作与任务分解

---

## License

学习与作品集项目，欢迎交流与改进。
