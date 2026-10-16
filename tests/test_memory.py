"""记忆层测试：会话记忆、长期记忆（去重/衰减/过期）、技能库。"""

from __future__ import annotations

import time

from contextforge.memory import (
    BUILTIN_SKILLS,
    ConversationStore,
    MemoryManager,
    MemoryStore,
    SkillStore,
    extract_memories_from_messages,
)
from contextforge.llm import MockLLM
from contextforge.storage import Database


# --------------------------------------------------------------------------- #
# 会话记忆
# --------------------------------------------------------------------------- #
def test_conversation_store_roundtrip(db: Database):
    store = ConversationStore(db)
    conversation_id = store.ensure(title="测试会话")
    store.append(conversation_id, {"role": "user", "content": "你好"})
    store.append(
        conversation_id,
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
            ],
        },
    )
    store.append(conversation_id, {"role": "tool", "tool_call_id": "c1", "content": "结果"})

    history = store.history(conversation_id)
    assert len(history) == 3
    assert history[1]["tool_calls"][0]["function"]["name"] == "t"
    assert history[2]["tool_call_id"] == "c1"
    assert store.count(conversation_id) == 3


def test_conversation_replace_supports_compaction(db: Database):
    store = ConversationStore(db)
    conversation_id = store.ensure()
    for index in range(6):
        store.append(conversation_id, {"role": "user", "content": f"消息 {index}"})

    store.replace(conversation_id, [{"role": "user", "content": "压缩后的摘要"}])
    history = store.history(conversation_id)
    assert len(history) == 1
    assert history[0]["content"] == "压缩后的摘要"


def test_conversation_list_and_delete(db: Database):
    store = ConversationStore(db)
    first = store.ensure(title="A")
    store.ensure(title="B")
    assert len(store.list()) == 2
    assert store.delete(first) is True
    assert len(store.list()) == 1


# --------------------------------------------------------------------------- #
# 长期记忆
# --------------------------------------------------------------------------- #
def test_memory_remember_and_recall(db: Database):
    store = MemoryStore(db)
    store.remember("用户偏好中文回答", kind="preference", importance=0.8)
    store.remember("项目使用 FastAPI 与 Vue 3", kind="fact", importance=0.6)

    hits = store.recall("用户偏好什么语言", top_k=2)
    assert hits
    assert "中文" in hits[0].content


def test_memory_deduplicates_similar_content(db: Database):
    store = MemoryStore(db)
    store.remember("用户偏好中文回答", kind="preference", importance=0.6)
    store.remember("用户偏好使用中文回答", kind="preference", importance=0.7)

    all_memories = store.all()
    assert len(all_memories) == 1, "高度相似的记忆应当合并而不是重复写入"
    assert all_memories[0].importance >= 0.7, "合并时重要度应当取较高值"


def test_memory_rejects_empty_content(db: Database):
    store = MemoryStore(db)
    try:
        store.remember("   ")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("空内容应当被拒绝")


def test_memory_expiry_removes_stale_progress(db: Database):
    store = MemoryStore(db)
    store.remember("任务进行到第 3 步", kind="progress", ttl_seconds=-1)
    assert store.expire() == 1
    assert store.all() == []


def test_memory_touch_increases_hit_count(db: Database):
    store = MemoryStore(db)
    memory = store.remember("用户偏好深色主题", kind="preference")
    store.recall("主题", top_k=1)
    refreshed = store.get(memory.memory_id)
    assert refreshed is not None
    assert refreshed.hit_count >= 1


def test_memory_kind_weight_biases_ranking(db: Database):
    store = MemoryStore(db)
    store.remember("记录一条普通事实用于对比排序", kind="fact", importance=0.5)
    store.remember("记录一条用户偏好用于对比排序", kind="preference", importance=0.5)

    hits = store.recall("记录一条用于对比排序", top_k=2)
    assert hits[0].kind == "preference", "同等条件下偏好类记忆应排在前面"


def test_memory_update_and_forget(db: Database):
    store = MemoryStore(db)
    memory = store.remember("初始内容", kind="fact")
    assert store.update(memory.memory_id, content="更新后的内容", importance=0.9) is True
    updated = store.get(memory.memory_id)
    assert updated is not None and updated.content == "更新后的内容"
    assert store.forget(memory.memory_id) is True
    assert store.get(memory.memory_id) is None


def test_memory_stats_groups_by_kind(db: Database):
    store = MemoryStore(db)
    store.remember("偏好一", kind="preference")
    store.remember("事实一", kind="fact")
    stats = store.stats()
    assert stats["total"] == 2
    assert set(stats["by_kind"]) == {"preference", "fact"}


def test_extract_memories_ignores_questions():
    """提问不应被当成记忆，否则长期记忆会被问题淹没。"""
    messages = [
        {"role": "user", "content": "上下文压缩的策略有哪些？"},
        {"role": "assistant", "content": "有四种。"},
        {"role": "user", "content": "我偏好中文回答，以后都用中文。"},
    ]
    extracted = extract_memories_from_messages(MockLLM(), messages)
    contents = [item["content"] for item in extracted]
    assert all("策略有哪些" not in content for content in contents)
    assert any("中文" in content for content in contents)


def test_extract_memories_handles_model_failure():
    class Broken:
        def chat(self, *args, **kwargs):
            raise RuntimeError("模型不可用")

    assert extract_memories_from_messages(Broken(), [{"role": "user", "content": "记住这个"}]) == []


# --------------------------------------------------------------------------- #
# 技能库
# --------------------------------------------------------------------------- #
def test_skill_register_is_idempotent(db: Database):
    store = SkillStore(db)
    store.register("测试技能", description="第一次", steps=["a", "b"])
    store.register("测试技能", description="第二次", steps=["c"])
    assert len(store.all()) == 1
    skill = store.get_by_name("测试技能")
    assert skill is not None
    assert skill.description == "第二次"
    assert skill.steps == ["c"]


def test_skill_find_matches_task(db: Database):
    store = SkillStore(db)
    store.register("知识库问答", description="检索知识库并给出引用", steps=["检索", "回答"])
    store.register("代码仓库理解", description="分析代码结构与符号", steps=["概览", "定位"])

    hits = store.find("帮我理解这个代码仓库的结构", top_k=2)
    assert hits
    assert hits[0].name == "代码仓库理解"


def test_skill_records_usage_and_success_rate(db: Database):
    store = SkillStore(db)
    store.register("可复用流程", description="演示成功率")
    store.record_use("可复用流程", success=True)
    store.record_use("可复用流程", success=False)
    skill = store.get_by_name("可复用流程")
    assert skill is not None
    assert skill.use_count == 2
    assert skill.success_rate == 0.5


def test_memory_manager_installs_builtin_skills(db: Database):
    manager = MemoryManager(db)
    assert manager.install_defaults() == len(BUILTIN_SKILLS)
    assert len(manager.skills.all()) == len(BUILTIN_SKILLS)


def test_memory_manager_prepare_and_save_turn(db: Database):
    manager = MemoryManager(db)
    manager.install_defaults()
    manager.long_term.remember("用户偏好中文回答", kind="preference", importance=0.9)

    bundle = manager.prepare("请用中文回答我的代码问题", top_k=3)
    assert bundle.long_term
    block = bundle.as_prompt_block()
    assert "长期记忆" in block

    conversation_id = manager.conversations.ensure()
    manager.save_turn(conversation_id, [{"role": "user", "content": "你好"}])
    assert manager.conversations.count(conversation_id) == 1
    assert manager.stats()["skills"] >= len(BUILTIN_SKILLS)


def test_memory_manager_consolidate_persists_preferences(db: Database):
    manager = MemoryManager(db)
    conversation_id = manager.conversations.ensure()
    manager.conversations.append(
        conversation_id, {"role": "user", "content": "我偏好中文回答，以后都用中文。"}
    )
    saved = manager.consolidate(conversation_id, MockLLM())
    assert saved
    assert manager.long_term.stats()["total"] >= 1
