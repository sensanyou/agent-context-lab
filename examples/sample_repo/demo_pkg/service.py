"""一小段用于演示「代码仓库理解」的示例包。

结构刻意做得有一点嵌套，方便展示 repo_overview / repo_symbols / repo_grep
三个工具分别能看到什么层次的认知。
"""

import math

from .storage import InMemoryStore
from .models import Note

__all__ = ["NoteService", "summarize_scores"]


class NoteService:
    """笔记服务：负责增删查与简单的相关性排序。"""

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()

    def add(self, title: str, body: str, tags: list[str] | None = None) -> Note:
        """新增一条笔记。"""
        note = Note(title=title, body=body, tags=tags or [])
        self.store.put(note)
        return note

    def find(self, keyword: str) -> list[Note]:
        """按关键词查找笔记。"""
        return [note for note in self.store.all() if keyword in note.body]

    def related(self, note: Note, limit: int = 3) -> list[Note]:
        """基于标签重叠度找出相关笔记。"""
        scored = []
        for other in self.store.all():
            if other.note_id == note.note_id:
                continue
            overlap = len(set(note.tags) & set(other.tags))
            if overlap:
                scored.append((overlap, other))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def delete(self, note_id: str) -> bool:
        """删除笔记，返回是否真的删掉了。"""
        return self.store.remove(note_id)


def summarize_scores(values: list[float]) -> dict:
    """给出最小值、最大值和平均值，用于快速了解分数分布。"""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "std": _std(values),
    }


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def rank_notes(notes: list[Note], weights: dict[str, float]) -> list[Note]:
    """按标签权重给笔记排序。"""
    return sorted(notes, key=lambda note: sum(weights.get(tag, 0.0) for tag in note.tags), reverse=True)
