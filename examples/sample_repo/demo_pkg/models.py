"""示例数据模型。"""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Note:
    """一条笔记。"""

    title: str
    body: str = ""
    tags: list[str] = field(default_factory=list)
    note_id: str = field(default_factory=lambda: f"note_{uuid.uuid4().hex[:8]}")
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        """返回一行摘要。"""
        head = " ".join(self.body.split())[:40]
        return f"[{','.join(self.tags)}] {self.title} - {head}"

    def to_dict(self) -> dict:
        """转成可序列化结构。"""
        return {
            "note_id": self.note_id,
            "title": self.title,
            "body": self.body,
            "tags": self.tags,
            "created_at": self.created_at,
        }
