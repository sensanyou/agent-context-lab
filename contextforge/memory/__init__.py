"""记忆层：会话记忆、长期记忆、技能库。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..storage import Database
from ..types import Message
from .conversation import ConversationStore
from .long_term import Memory, MemoryStore, extract_memories_from_messages
from .skills import BUILTIN_SKILLS, Skill, SkillStore, install_builtin_skills

__all__ = [
    "BUILTIN_SKILLS",
    "ConversationStore",
    "Memory",
    "MemoryManager",
    "MemoryStore",
    "Skill",
    "SkillStore",
    "extract_memories_from_messages",
    "install_builtin_skills",
]


@dataclass
class MemoryBundle:
    """一次任务开始前，从记忆层取回的全部先验信息。"""

    long_term: list[Memory]
    skills: list[Skill]

    def as_prompt_block(self) -> str:
        """渲染成可注入上下文的文本块。"""
        lines: list[str] = []
        if self.long_term:
            lines.append("【长期记忆｜跨任务保留的用户偏好与项目事实】")
            for memory in self.long_term:
                lines.append(f"- ({memory.kind}) {memory.content}")
        if self.skills:
            if lines:
                lines.append("")
            lines.append("【可复用技能｜如果适用就按这个流程做】")
            for skill in self.skills:
                steps = " → ".join(skill.steps) if skill.steps else skill.description
                lines.append(f"- {skill.name}：{steps}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_term": [memory.to_dict() for memory in self.long_term],
            "skills": [skill.to_dict() for skill in self.skills],
        }


class MemoryManager:
    """把三层记忆组合成一个入口，供 Agent Runtime 调用。"""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.long_term = MemoryStore(db)
        self.conversations = ConversationStore(db)
        self.skills = SkillStore(db)

    def install_defaults(self) -> int:
        """安装内置技能（幂等）。"""
        return install_builtin_skills(self.skills)

    # ------------------------------------------------------------------ #
    def prepare(
        self,
        question: str,
        conversation_id: str | None = None,
        top_k: int = 4,
    ) -> MemoryBundle:
        """任务开始前收集先验。"""
        return MemoryBundle(
            long_term=self.long_term.recall(question, top_k=top_k),
            skills=self.skills.find(question, top_k=2),
        )

    def load_history(self, conversation_id: str, limit: int = 200) -> list[Message]:
        return self.conversations.history(conversation_id, limit)

    def save_turn(self, conversation_id: str, messages: Sequence[Message]) -> None:
        self.conversations.extend(conversation_id, messages)

    def consolidate(self, conversation_id: str, llm) -> list[Memory]:
        """任务结束后把会话中值得保留的信息沉淀为长期记忆。"""
        history = self.conversations.history(conversation_id, limit=20)
        extracted = extract_memories_from_messages(llm, history)
        saved: list[Memory] = []
        for item in extracted:
            saved.append(
                self.long_term.remember(
                    item["content"],
                    kind=item.get("kind", "fact"),
                    importance=item.get("importance", 0.6),
                    source=conversation_id,
                )
            )
        return saved

    def stats(self) -> dict[str, Any]:
        return {
            "long_term": self.long_term.stats(),
            "skills": len(self.skills.all()),
            "conversations": len(self.conversations.list()),
        }
