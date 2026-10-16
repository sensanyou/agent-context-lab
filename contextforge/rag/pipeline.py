"""知识库门面：把加载 → 切分 → 向量化 → 建索引 → 检索 串成一条可用链路。

持久化设计：文档、切片、向量都落在 SQLite，进程重启后自动重建内存索引
（向量库 + BM25 倒排表），不需要额外的索引文件管理。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..config import RetrievalSettings, Settings, get_settings
from ..storage import Database, pack_vector, unpack_vector
from ..types import Chunk, Document, ScoredChunk, new_id
from .bm25 import BM25Index
from .chunker import TextChunk, chunk_loaded_file, chunk_text
from .embeddings import Embedder, build_embedder
from .loaders import LoadedFile, content_hash, file_id, iter_files, load_file
from .retriever import HybridRetriever
from .vector_store import InMemoryVectorStore


class KnowledgeBase:
    """个人知识库：文档导入、混合检索、来源引用。"""

    def __init__(
        self,
        db: Database,
        settings: Settings | None = None,
        llm=None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_settings: RetrievalSettings = self.settings.retrieval
        self.db = db

        # 向量化的选择交给 build_embedder：它会按 embedding_mode 决定用远程还是
        # 离线，并在远程不可用时**降级且保留原因**（见 rag/embeddings.py 的说明）。
        # 传 settings 是关键 —— 向量化模式与聊天模型是两条独立的开关。
        self.embedder = embedder or build_embedder(
            llm,
            dim=self.retrieval_settings.embedding_dim,
            settings=self.settings,
        )

        self.store = InMemoryVectorStore(dim=self.embedder.dim)
        self.bm25 = BM25Index()
        self.retriever = HybridRetriever(self.store, self.bm25, self.embedder, self.retrieval_settings)

        if self.store.count() == 0:
            self.reload()

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add_text(
        self,
        text: str,
        title: str = "",
        source: str = "",
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> Document:
        doc_id = doc_id or file_id(source or title or text[:64] or new_id("doc"))
        chunks = chunk_text(
            text,
            chunk_size=self.retrieval_settings.chunk_size,
            chunk_overlap=self.retrieval_settings.chunk_overlap,
        )
        return self._persist(
            Document(
                doc_id=doc_id,
                title=title or source or doc_id,
                source=source,
                text=text,
                metadata=metadata or {},
            ),
            chunks,
        )

    def add_file(self, path: str | Path) -> Document | None:
        loaded = load_file(path)
        if not loaded.text.strip():
            return None
        return self._add_loaded(loaded)

    def add_directory(
        self,
        path: str | Path,
        suffixes: set[str] | None = None,
        max_files: int = 300,
    ) -> tuple[list[Document], list[str]]:
        """导入整个目录。返回 (成功的文档, 跳过原因)。"""
        documents: list[Document] = []
        skipped: list[str] = []

        for file_path in iter_files(path, suffixes=suffixes, max_files=max_files):
            loaded = load_file(file_path)
            if not loaded.text.strip():
                skipped.append(f"{file_path}: {loaded.skipped_reason or '无文本内容'}")
                continue
            documents.append(self._add_loaded(loaded))

        return documents, skipped

    def delete_document(self, doc_id: str) -> bool:
        row = self.db.query_one("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
        if row is None:
            return False

        chunk_rows = self.db.query("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))
        for chunk_row in chunk_rows:
            self.bm25.remove(chunk_row["chunk_id"])

        self.store.remove_document(doc_id)
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return True

    def clear(self) -> None:
        self.store = InMemoryVectorStore(dim=self.embedder.dim)
        self.bm25.clear()
        self.retriever.store = self.store
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")

    def deduplicate(self) -> int:
        """按正文指纹清掉重复文档，返回删除数量。保留最新的一份。

        典型场景：项目目录被重命名过（`file_id` 依赖绝对路径），
        同一批文件因此被重复导入。
        """
        rows = self.db.query(
            "SELECT doc_id, text FROM documents ORDER BY created_at DESC"
        )
        seen: set[str] = set()
        removed = 0
        for row in rows:
            digest = content_hash(row["text"] or "")
            if digest in seen:
                if self.delete_document(row["doc_id"]):
                    removed += 1
                continue
            seen.add(digest)
        return removed

    # ------------------------------------------------------------------ #
    # 读取
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        doc_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        filter_metadata = None
        results = self.retriever.search(
            query, top_k=top_k, filter_metadata=filter_metadata, mode=mode
        )
        if doc_ids:
            allowed = set(doc_ids)
            results = [item for item in results if item.chunk.doc_id in allowed]
        return results

    def list_documents(self) -> list[Document]:
        rows = self.db.query(
            "SELECT doc_id, title, source, media_type, chunk_count, metadata, created_at"
            " FROM documents ORDER BY created_at DESC"
        )
        return [
            Document(
                doc_id=row["doc_id"],
                title=row["title"],
                source=row["source"],
                text="",
                chunk_count=row["chunk_count"],
                metadata=self.db.load_json(row["metadata"], {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_document(self, doc_id: str) -> Document | None:
        row = self.db.query_one("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        if row is None:
            return None
        return Document(
            doc_id=row["doc_id"],
            title=row["title"],
            source=row["source"],
            text=row["text"],
            chunk_count=row["chunk_count"],
            metadata=self.db.load_json(row["metadata"], {}),
            created_at=row["created_at"],
        )

    def stats(self) -> dict[str, Any]:
        docs = self.db.query_one("SELECT COUNT(*) AS n FROM documents")
        chunks = self.db.query_one("SELECT COUNT(*) AS n FROM chunks")
        total_chars = self.db.query_one("SELECT COALESCE(SUM(LENGTH(text)), 0) AS n FROM chunks")
        return {
            "documents": docs["n"] if docs else 0,
            "chunks": chunks["n"] if chunks else 0,
            "total_chars": total_chars["n"] if total_chars else 0,
            "embedder": self.embedder.describe(),
            "vector_store": self.store.describe(),
            "bm25": self.bm25.stats(),
        }

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def reload(self) -> None:
        """从 SQLite 重建内存索引。"""
        self.store = InMemoryVectorStore(dim=self.embedder.dim)
        self.bm25.clear()

        rows = self.db.query(
            "SELECT chunk_id, doc_id, ord, text, source, title, metadata, dim, vector FROM chunks"
        )

        chunks: list[Chunk] = []
        vectors: list[list[float]] = []

        for row in rows:
            chunk = Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                index=row["ord"],
                source=row["source"],
                title=row["title"],
                metadata=self.db.load_json(row["metadata"], {}),
            )
            chunks.append(chunk)
            self.bm25.add(chunk.chunk_id, chunk.text)
            vectors.append(unpack_vector(row["vector"], row["dim"] or self.embedder.dim))

        if chunks:
            import numpy as np

            matrix = np.asarray(vectors, dtype=np.float32)
            # 维度不一致（例如换过 embedding 模型）时重新向量化，而不是崩溃
            if matrix.shape[1] != self.embedder.dim:
                matrix = self.embedder.embed([chunk.text for chunk in chunks])
            self.store.add(chunks, matrix)

        self.retriever.store = self.store

    def reindex(self) -> int:
        """用当前 embedder 重新向量化全部切片（换 embedding 模型后调用）。"""
        rows = self.db.query("SELECT chunk_id, text FROM chunks")
        if not rows:
            return 0
        matrix = self.embedder.embed([row["text"] for row in rows])
        with self.db.transaction() as conn:
            for row, vector in zip(rows, matrix):
                conn.execute(
                    "UPDATE chunks SET vector = ?, dim = ? WHERE chunk_id = ?",
                    (pack_vector(vector), self.embedder.dim, row["chunk_id"]),
                )
        self.reload()
        return len(rows)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _drop_duplicate_content(self, digest: str, text: str, keep: str) -> list[str]:
        """删除与 `digest` 内容相同的其他文档，返回被删掉的 doc_id。

        老数据（本次改动之前导入的）没有 `content_hash` 字段，
        这时直接用正文现算一次指纹 —— 否则重命名造成的重复记录永远不会被清理。
        """
        rows = self.db.query(
            "SELECT doc_id, text, metadata FROM documents WHERE doc_id != ?", (keep,)
        )
        removed: list[str] = []
        for row in rows:
            stored = self.db.load_json(row["metadata"], {}).get("content_hash")
            same = (
                stored == digest
                if stored
                else content_hash(row["text"] or "") == digest
            )
            if same and self.delete_document(row["doc_id"]):
                removed.append(row["doc_id"])
        return removed

    def _add_loaded(self, loaded: LoadedFile) -> Document:
        chunks = chunk_loaded_file(
            loaded,
            chunk_size=self.retrieval_settings.chunk_size,
            chunk_overlap=self.retrieval_settings.chunk_overlap,
        )
        document = Document(
            doc_id=file_id(loaded.path),
            title=loaded.title,
            source=str(loaded.path),
            text=loaded.text,
            metadata={**loaded.metadata, "media_type": loaded.media_type},
        )
        return self._persist(document, chunks, doc_id=document.doc_id, replace=True)

    def _persist(
        self,
        document: Document,
        chunks: Iterable[TextChunk],
        doc_id: str | None = None,
        replace: bool = False,
    ) -> Document:
        doc_id = doc_id or document.doc_id
        chunk_list = list(chunks)

        if replace:
            self.delete_document(doc_id)

        # 内容去重。`doc_id` 由绝对路径派生，项目目录一改名就会生成新 id，
        # 旧记录不会被替换，于是同一份内容在库里留两份、检索时返回近似重复命中。
        # 这里按正文指纹消除它，并顺手清掉旧版本留下的、没有指纹字段的历史数据。
        digest = content_hash(document.text)
        self._drop_duplicate_content(digest, document.text, keep=doc_id)
        document.metadata["content_hash"] = digest

        now = time.time()
        records: list[tuple[Chunk, list[float]]] = []

        if chunk_list:
            matrix = self.embedder.embed([item.text for item in chunk_list])
        else:
            matrix = []

        for position, item in enumerate(chunk_list):
            chunk = Chunk(
                chunk_id=f"{doc_id}::{item.index}",
                doc_id=doc_id,
                text=item.text,
                index=item.index,
                source=document.source,
                title=document.title,
                metadata={**(item.metadata or {}), "kind": item.kind},
            )
            vector = [float(x) for x in matrix[position]]
            records.append((chunk, vector))
            self.bm25.add(chunk.chunk_id, chunk.text)

        if records:
            import numpy as np

            self.store.add([chunk for chunk, _v in records], np.asarray([v for _c, v in records], dtype=np.float32))

        with self.db.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents"
                " (doc_id, title, source, media_type, text, chunk_count, metadata, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id,
                    document.title,
                    document.source,
                    document.metadata.get("media_type", "text/plain"),
                    document.text,
                    len(records),
                    self.db.dump_json(document.metadata),
                    now,
                ),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO chunks"
                " (chunk_id, doc_id, ord, text, source, title, metadata, dim, vector)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.index,
                        chunk.text,
                        chunk.source,
                        chunk.title,
                        self.db.dump_json(chunk.metadata),
                        self.embedder.dim,
                        pack_vector(vector),
                    )
                    for chunk, vector in records
                ],
            )

        document.doc_id = doc_id
        document.chunk_count = len(records)
        return document
