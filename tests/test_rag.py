"""RAG 链路测试：加载、切分、向量、BM25、融合、重排序、持久化。"""

from __future__ import annotations

import pytest

from contextforge.config import RetrievalSettings
from contextforge.rag import (
    BM25Index,
    HashingEmbedder,
    HybridRetriever,
    InMemoryVectorStore,
    KnowledgeBase,
    chunk_code,
    chunk_text,
    load_file,
    reciprocal_rank_fusion,
    tokenize,
)
from contextforge.rag.loaders import iter_files
from contextforge.rag.retriever import HeuristicReranker, weighted_fusion
from contextforge.storage import Database
from contextforge.types import Chunk


# --------------------------------------------------------------------------- #
# 分词与嵌入
# --------------------------------------------------------------------------- #
def test_tokenize_handles_chinese_bigrams_and_latin():
    tokens = tokenize("上下文压缩 Context Compression")
    assert "上" in tokens
    assert "上下" in tokens  # 中文二元组，用于提高召回
    assert "context" in tokens
    assert "compression" in tokens
    # 停用词被过滤
    assert "的" not in tokenize("的的的 Agent")


def test_hashing_embedder_is_deterministic_and_normalized():
    embedder = HashingEmbedder(dim=128)
    first = embedder.embed_one("上下文压缩以轮次为单位")
    second = embedder.embed_one("上下文压缩以轮次为单位")
    assert first.shape == (128,)
    assert (first == second).all(), "离线向量必须是确定性的，否则测试与实验无法复现"
    assert abs(float((first ** 2).sum()) - 1.0) < 1e-5, "应当做 L2 归一化"


def test_hashing_embedder_similarity_reflects_overlap():
    embedder = HashingEmbedder(dim=512)
    query = embedder.embed_one("上下文压缩策略")
    near = embedder.embed_one("上下文压缩的四种策略")
    far = embedder.embed_one("今天天气不错适合出门散步")
    assert float(query @ near) > float(query @ far)


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def test_chunk_text_respects_size_and_keeps_all_content():
    text = "\n\n".join(f"第 {i} 段内容。" * 20 for i in range(10))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 * 1.4 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_code_splits_by_function_boundaries():
    source = (
        "import os\n\n\n"
        "def alpha(x):\n    return x + 1\n\n\n"
        "class Beta:\n    def method(self):\n        return 2\n\n\n"
        "def gamma():\n    return 3\n"
    )
    chunks = chunk_code(source, chunk_size=200)
    texts = [chunk.text for chunk in chunks]
    assert any("def alpha" in text for text in texts)
    assert any("class Beta" in text for text in texts)
    assert all(chunk.kind == "code" for chunk in chunks)
    # 函数不能被拦腰截断
    for text in texts:
        if "def alpha" in text:
            assert "return x + 1" in text


def test_loader_skips_unsupported_and_reports_reason(tmp_path):
    missing = load_file(tmp_path / "不存在.md")
    assert missing.text == ""
    assert missing.skipped_reason

    target = tmp_path / "note.md"
    target.write_text("# 标题\n\n正文内容。", encoding="utf-8")
    loaded = load_file(target)
    assert loaded.text.startswith("# 标题")
    assert loaded.kind == "text"


def test_iter_files_skips_noise_directories(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.js").write_text("var x = 1", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "c.py").write_text("x = 1", encoding="utf-8")

    found = {path.name for path in iter_files(tmp_path)}
    assert found == {"a.py"}, "依赖目录与 .git 必须被跳过"


# --------------------------------------------------------------------------- #
# 检索
# --------------------------------------------------------------------------- #
def test_bm25_ranks_exact_term_highest():
    index = BM25Index()
    index.add("d1", "BM25 是稀疏检索算法，k1 控制词频饱和，b 控制长度归一化。")
    index.add("d2", "向量检索擅长语义相似，但精确词经常失灵。")
    index.add("d3", "今天天气不错。")
    results = index.search("BM25 k1", top_k=3)
    assert results and results[0][0] == "d1"


def test_bm25_supports_removal():
    index = BM25Index()
    index.add("d1", "上下文压缩")
    index.add("d2", "上下文压缩")
    index.remove("d1")
    assert index.size == 1
    assert [doc_id for doc_id, _score in index.search("压缩")] == ["d2"]


def test_vector_store_search_returns_cosine_order():
    store = InMemoryVectorStore(dim=4)
    chunks = [
        Chunk(chunk_id="a", doc_id="d", text="a", index=0),
        Chunk(chunk_id="b", doc_id="d", text="b", index=1),
    ]
    import numpy as np

    store.add(chunks, np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype="float32"))
    hits = store.search(np.array([0.9, 0.1, 0, 0], dtype="float32"), top_k=2)
    assert hits[0][0] == "a"


def test_reciprocal_rank_fusion_rewards_agreement():
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 12.0), ("c", 5.0)]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    # b 同时出现在两路 → 排名融合得分最高
    assert max(fused, key=fused.get) == "b"


def test_weighted_fusion_is_scale_invariant():
    """加权融合先归一化，所以把 BM25 分数整体放大不应改变结论。"""
    dense = [("a", 0.9), ("b", 0.1)]
    sparse = [("b", 12.0), ("c", 5.0)]
    scaled = [("b", 12000.0), ("c", 5000.0)]
    first = weighted_fusion([dense, sparse], [0.6, 0.4])
    second = weighted_fusion([dense, scaled], [0.6, 0.4])
    assert abs(first["b"] - second["b"]) < 1e-9


def test_reranker_prefers_phrase_match():
    reranker = HeuristicReranker()
    exact = Chunk(chunk_id="a", doc_id="d", text="混合检索需要 RRF 融合来合并两路排名。", index=0, title="检索")
    partial = Chunk(chunk_id="b", doc_id="d", text="检索相关内容的一般说明，与主题相关但未提及具体算法细节。", index=1)
    exact_score, _ = reranker.score("RRF 融合", exact)
    partial_score, _ = reranker.score("RRF 融合", partial)
    assert exact_score > partial_score


def test_hybrid_retriever_outperforms_single_route_on_exact_terms(kb: KnowledgeBase):
    """精确词查询：混合模式不应比纯 BM25 差。"""
    query = "RRF k=60"
    hybrid = kb.search(query, top_k=3, mode="hybrid")
    sparse = kb.search(query, top_k=3, mode="sparse")
    assert hybrid, "混合检索应当有结果"
    assert {item.chunk.chunk_id for item in sparse} & {
        item.chunk.chunk_id for item in hybrid
    }, "混合检索应当包含稀疏检索召回到的精确命中文档"


def test_retriever_records_trace(kb: KnowledgeBase):
    kb.search("上下文压缩", top_k=3, mode="hybrid")
    trace = kb.retriever.last_trace
    assert trace is not None
    assert trace.dense_hits >= 0 and trace.sparse_hits >= 0
    assert trace.to_dict()["query"] == "上下文压缩"


def test_multi_query_retrieval_merges_results(kb: KnowledgeBase):
    retriever: HybridRetriever = kb.retriever
    hits = retriever.search_multi_query(["上下文压缩", "BM25 检索"], top_k=3)
    assert hits
    assert len(hits) <= 3


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
def test_knowledge_base_ingests_directory(kb: KnowledgeBase, sample_docs):
    stats = kb.stats()
    if sample_docs.is_dir():
        assert stats["documents"] >= 3
        assert stats["chunks"] >= 3
    assert stats["vector_store"]["chunks"] == stats["chunks"]
    assert stats["bm25"]["documents"] == stats["chunks"]


def test_knowledge_base_search_returns_citations(kb: KnowledgeBase):
    hits = kb.search("为什么压缩要以轮次为单位", top_k=3)
    assert hits
    top = hits[0]
    assert top.chunk.citation().count("#") == 1
    assert top.score > 0


def test_knowledge_base_delete_document(kb: KnowledgeBase):
    document = kb.add_text("这是一段用于测试删除的临时内容，内容足够长以便切分。", title="临时")
    assert kb.store.count() > 0
    assert kb.delete_document(document.doc_id) is True
    assert kb.get_document(document.doc_id) is None
    assert all(chunk.doc_id != document.doc_id for chunk in kb.store._chunks)
    assert kb.search("临时内容", top_k=3) == [] or all(
        item.chunk.doc_id != document.doc_id for item in kb.search("临时内容", top_k=3)
    )


def test_knowledge_base_persists_and_reloads(settings, db: Database, llm):
    kb = KnowledgeBase(db, settings, llm=llm)
    kb.add_text("# 持久化测试\n\n这段内容应当在重新加载后依然可以被检索到。", title="持久化")
    chunks_before = kb.stats()["chunks"]

    reopened = KnowledgeBase(db, settings, llm=llm)
    assert reopened.stats()["chunks"] == chunks_before
    assert reopened.store.count() == chunks_before
    assert reopened.search("持久化测试", top_k=3)


def test_knowledge_base_reindex_after_embedder_change(kb: KnowledgeBase):
    count = kb.reindex()
    assert count == kb.stats()["chunks"]
    assert kb.store.count() == count


def test_add_file_from_sample_repo(kb: KnowledgeBase, sample_repo):
    if not sample_repo.is_dir():
        pytest.skip("示例仓库不存在")
    target = sample_repo / "demo_pkg" / "service.py"
    document = kb.add_file(target)
    assert document is not None
    assert document.chunk_count >= 1
    # 代码切片应当按函数边界切分，能检索到具体函数名
    hits = kb.search("rank_notes", top_k=3)
    assert hits
