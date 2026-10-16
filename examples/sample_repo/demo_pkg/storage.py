"""示例存储层（内存实现）。"""

from .models import Note


class InMemoryStore:
    """极简内存存储，演示分层结构。"""

    def __init__(self) -> None:
        self._items: dict[str, Note] = {}

    def put(self, note: Note) -> None:
        """写入或覆盖一条笔记。"""
        self._items[note.note_id] = note

    def get(self, note_id: str) -> Note | None:
        """按 id 读取。"""
        return self._items.get(note_id)

    def all(self) -> list[Note]:
        """返回全部笔记。"""
        return list(self._items.values())

    def remove(self, note_id: str) -> bool:
        """删除并返回是否命中。"""
        return self._items.pop(note_id, None) is not None

    def count(self) -> int:
        """当前笔记数量。"""
        return len(self._items)
