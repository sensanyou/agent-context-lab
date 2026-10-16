"""技能库：把「被验证过的做法」沉淀成可复用工作流。

和长期记忆的区别：记忆是**陈述性知识**（用户偏好、项目事实），
技能是**程序性知识**（遇到这类任务，按这几步走，用这几个工具）。

技能不是自动生效的 —— 它作为提示注入到上下文里，由 Agent 决定这次要不要用。
这样既避免了「过度自动化」，也保留了可解释性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..storage import Database
from ..rag.tokenize import tokenize


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str = ""
    steps: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    use_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    score: float = 0.0

    @property
    def success_rate(self) -> float:
        return round(self.success_count / self.use_count, 3) if self.use_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "tools": self.tools,
            "use_count": self.use_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "score": round(self.score, 4),
        }


class SkillStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        description: str = "",
        steps: Sequence[str] | None = None,
        tools: Sequence[str] | None = None,
    ) -> Skill:
        name = " ".join((name or "").split())
        if not name:
            raise ValueError("技能名不能为空")

        now = time.time()
        existing = self.db.query_one("SELECT * FROM skills WHERE name = ?", (name,))
        if existing is not None:
            self.db.execute(
                "UPDATE skills SET description = ?, steps = ?, tools = ?, updated_at = ?"
                " WHERE name = ?",
                (
                    description or existing["description"],
                    self.db.dump_json(list(steps) if steps else self.db.load_json(existing["steps"], [])),
                    self.db.dump_json(list(tools) if tools else self.db.load_json(existing["tools"], [])),
                    now,
                    name,
                ),
            )
            return self.get_by_name(name)  # type: ignore[return-value]

        skill = Skill(
            skill_id=f"skill_{abs(hash(name)) % 10**10}",
            name=name,
            description=description,
            steps=list(steps or []),
            tools=list(tools or []),
        )
        self.db.execute(
            "INSERT INTO skills (skill_id, name, description, steps, tools, use_count,"
            " success_count, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)",
            (
                skill.skill_id,
                skill.name,
                skill.description,
                self.db.dump_json(skill.steps),
                self.db.dump_json(skill.tools),
                now,
                now,
            ),
        )
        return skill

    def record_use(self, name: str, success: bool = True) -> None:
        self.db.execute(
            "UPDATE skills SET use_count = use_count + 1,"
            " success_count = success_count + ?, updated_at = ? WHERE name = ?",
            (1 if success else 0, time.time(), name),
        )

    def find(self, query: str, top_k: int = 3) -> list[Skill]:
        query_tokens = set(tokenize(query))
        skills = self.all()
        if not query_tokens:
            return sorted(skills, key=lambda s: s.use_count, reverse=True)[:top_k]

        for skill in skills:
            haystack = set(tokenize(f"{skill.name} {skill.description} {' '.join(skill.steps)}"))
            overlap = len(query_tokens & haystack) / len(query_tokens)
            # 成功率作为先验，但没有使用记录时不惩罚
            quality = 0.5 + 0.5 * skill.success_rate if skill.use_count else 0.6
            skill.score = round(overlap * quality, 6)

        matched = [skill for skill in skills if skill.score > 0.05]
        matched.sort(key=lambda s: s.score, reverse=True)
        return matched[:top_k]

    def all(self) -> list[Skill]:
        rows = self.db.query("SELECT * FROM skills ORDER BY use_count DESC")
        return [self._row_to_skill(row) for row in rows]

    def get_by_name(self, name: str) -> Skill | None:
        row = self.db.query_one("SELECT * FROM skills WHERE name = ?", (name,))
        return self._row_to_skill(row) if row else None

    def remove(self, name: str) -> bool:
        cursor = self.db.execute("DELETE FROM skills WHERE name = ?", (name,))
        return cursor.rowcount > 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_skill(row) -> Skill:
        return Skill(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"],
            steps=Database.load_json(row["steps"], []),
            tools=Database.load_json(row["tools"], []),
            use_count=row["use_count"],
            success_count=row["success_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


#: 内置技能：项目自带、开箱可用，同时也是「技能该怎么写」的示例
BUILTIN_SKILLS: tuple[dict[str, Any], ...] = (
    {
        "name": "知识库问答",
        "description": "用户提出需要依据资料回答的问题时使用：先检索知识库，再基于片段作答并给出引用。",
        "steps": [
            "用 knowledge_search 检索相关片段",
            "检查片段是否覆盖问题的全部要点",
            "覆盖不足时改写查询再检索一次",
            "回答时标注 [编号] 引用来源",
        ],
        "tools": ["knowledge_search"],
    },
    {
        "name": "代码仓库理解",
        "description": "用户想了解一个代码项目的结构、模块职责或调用关系时使用。",
        "steps": [
            "用 repo_overview 获取目录结构与关键文件",
            "用 repo_symbols 列出函数/类定义",
            "用 repo_grep 定位具体符号的引用位置",
            "先讲整体结构，再讲用户关心的那条链路",
        ],
        "tools": ["repo_overview", "repo_symbols", "repo_grep"],
    },
    {
        "name": "失败测试定位",
        "description": "用户报告测试失败或希望验证代码时使用。",
        "steps": [
            "用 run_tests 复现失败",
            "读取报错栈对应的文件与行号",
            "给出最小修改建议并解释原因",
        ],
        "tools": ["run_tests", "fs_read"],
    },
)


def install_builtin_skills(store: SkillStore) -> int:
    count = 0
    for spec in BUILTIN_SKILLS:
        store.register(**spec)
        count += 1
    return count
