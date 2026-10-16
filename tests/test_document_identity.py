"""文档身份与去重。

`file_id` 由**绝对路径**派生，这带来一个不容易发现的问题：

    D:\\agent-context-lab\\examples\\sample_docs\\X.md   →  doc_1f65...
    D:\\ContextForge\\examples\\sample_docs\\X.md        →  doc_85fc...

项目目录只要改一次名，同一批文件就会拿到全新的 doc_id，旧记录不会被
`INSERT OR REPLACE` 覆盖，于是知识库里留下两份一模一样的内容，
检索时返回近似重复的命中，评测数字也随之失真。

修法不是改 `file_id`（那会让所有历史记录一次性失效），而是**按正文指纹消重**：
内容相同就只保留一份，并且能识别没有指纹字段的历史数据。
"""

from __future__ import annotations

from pathlib import Path

from contextforge.rag import KnowledgeBase
from contextforge.rag.loaders import content_hash, file_id
from contextforge.storage import Database

TEXT = """# 上下文工程指南

上下文压缩必须以轮次为单位，否则 assistant(tool_calls) 与 tool 结果会被拆开，
模型接口会直接拒绝这个请求。这条约束由 validator 校验并修复。
"""


def _kb(db: Database, settings) -> KnowledgeBase:
    from contextforge.llm import MockLLM

    return KnowledgeBase(db, settings, llm=MockLLM())


def test_file_id_depends_on_the_absolute_path(tmp_path):
    """先把「为什么会有重复」这个前提固定下来。"""
    first = tmp_path / "a" / "doc.md"
    second = tmp_path / "b" / "doc.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(TEXT, encoding="utf-8")
    second.write_text(TEXT, encoding="utf-8")

    assert file_id(first) != file_id(second), "路径不同 → doc_id 不同"
    assert content_hash(TEXT) == content_hash(first.read_text(encoding="utf-8"))


def test_same_content_under_a_new_path_replaces_the_old_document(tmp_path, db, settings):
    """模拟「项目改了个名」：内容一样、路径不同，库里只能留一份。"""
    old_dir = tmp_path / "agent-context-lab" / "docs"
    new_dir = tmp_path / "ContextForge" / "docs"
    for folder in (old_dir, new_dir):
        folder.mkdir(parents=True)
        (folder / "指南.md").write_text(TEXT, encoding="utf-8")

    knowledge = _kb(db, settings)
    knowledge.add_file(old_dir / "指南.md")
    assert knowledge.stats()["documents"] == 1

    knowledge.add_file(new_dir / "指南.md")
    assert knowledge.stats()["documents"] == 1, "同一份内容不应留下两条记录"

    # 留下的是「这一次导入的那个」，source 指向新路径
    docs = knowledge.list_documents()
    assert str(new_dir) in docs[0].source


def test_legacy_rows_without_a_content_hash_are_cleaned(tmp_path, db, settings):
    """本次改动之前导入的数据没有 content_hash 字段，也要能被识别出来。"""
    old_path = tmp_path / "old" / "指南.md"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(TEXT, encoding="utf-8")

    knowledge = _kb(db, settings)
    knowledge.add_file(old_path)

    # 手工抹掉指纹字段，模拟历史数据
    db.execute(
        "UPDATE documents SET metadata = ? WHERE doc_id = ?",
        ('{"media_type": "text/markdown"}', file_id(old_path)),
    )

    new_path = tmp_path / "new" / "指南.md"
    new_path.parent.mkdir(parents=True)
    new_path.write_text(TEXT, encoding="utf-8")
    knowledge.add_file(new_path)

    assert knowledge.stats()["documents"] == 1


def test_different_content_is_kept_separately(tmp_path, db, settings):
    """去重只针对「内容完全相同」，不同内容一份都不能少。"""
    knowledge = _kb(db, settings)
    knowledge.add_text("第一条：上下文预算按分区分配。", title="甲", source="a.md")
    knowledge.add_text("第二条：混合检索用 RRF 融合。", title="乙", source="b.md")
    assert knowledge.stats()["documents"] == 2


def test_reimporting_a_directory_is_idempotent(tmp_path, db, settings):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    for index in range(3):
        (docs_dir / f"doc{index}.md").write_text(
            f"# 文档 {index}\n\n这是第 {index} 份内容，用于验证重复导入不会累加。\n",
            encoding="utf-8",
        )

    knowledge = _kb(db, settings)
    for _ in range(3):
        knowledge.add_directory(docs_dir)

    assert knowledge.stats()["documents"] == 3
    assert knowledge.stats()["chunks"] > 0


def test_deduplicate_cleans_up_an_already_polluted_database(tmp_path, db, settings):
    """已经脏掉的知识库（例如用户现在的那个）要能被一次性清理干净。"""
    knowledge = _kb(db, settings)
    for index in range(4):
        knowledge.add_text(f"正文 {index} 的内容，足够长以便生成切片。", title=f"文档 {index}")

    # 绕过去重逻辑，手工塞进 4 条一模一样的副本
    rows = db.query("SELECT doc_id, title, source, text, metadata FROM documents")
    for offset, row in enumerate(rows):
        db.execute(
            "INSERT INTO documents (doc_id, title, source, media_type, text,"
            " chunk_count, metadata, created_at)"
            " VALUES (?, ?, ?, 'text/plain', ?, 0, ?, 0)",
            (
                f"dup_{offset}",
                row["title"],
                f"old/{row['title']}.md",
                row["text"],
                row["metadata"],
            ),
        )
    assert knowledge.stats()["documents"] == 8

    removed = knowledge.deduplicate()
    assert removed == 4
    assert knowledge.stats()["documents"] == 4
    assert knowledge.deduplicate() == 0, "第二次调用应该是空操作"


def test_deduplicate_removes_chunks_and_index_entries(tmp_path, db, settings):
    """去重必须连切片、向量、BM25 一起带走，不能只删 documents 那行。"""
    knowledge = _kb(db, settings)
    knowledge.add_text("键 值 混合检索 RRF 融合的公式与重排序策略。", title="甲", source="a.md")
    before_chunks = knowledge.stats()["chunks"]

    # 换了个 source → doc_id 不同 → 这正是「改名后重复」的成因
    knowledge.add_text("键 值 混合检索 RRF 融合的公式与重排序策略。", title="甲副本", source="b.md")

    assert knowledge.stats()["documents"] == 1
    assert knowledge.stats()["chunks"] == before_chunks, "切片数不应因为一次重复导入而翻倍"
    assert knowledge.store.count() == before_chunks, "向量库要同步清理"
    assert knowledge.bm25.stats()["documents"] == before_chunks, "BM25 倒排表要同步清理"
    assert len(knowledge.store.documents()) == 1


def test_chunks_of_removed_duplicate_are_not_searchable(tmp_path, db, settings):
    knowledge = _kb(db, settings)
    old = tmp_path / "old" / "指南.md"
    new = tmp_path / "new" / "指南.md"
    for path in (old, new):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEXT, encoding="utf-8")

    knowledge.add_file(old)
    knowledge.add_file(new)

    hits = knowledge.search("上下文压缩 轮次", top_k=5)
    doc_ids = {hit.chunk.doc_id for hit in hits}
    assert file_id(old) not in doc_ids, "被去重掉的文档不应再出现在检索结果里"


def test_metadata_still_carries_media_type_after_dedup(tmp_path, db, settings):
    """content_hash 是往 metadata 里加的，不能把原有字段挤掉。"""
    knowledge = _kb(db, settings)
    path = Path(tmp_path) / "指南.md"
    path.write_text(TEXT, encoding="utf-8")
    knowledge.add_file(path)

    document = knowledge.get_document(file_id(path))
    assert document is not None
    assert document.metadata.get("media_type")
    assert document.metadata.get("content_hash") == content_hash(TEXT)
