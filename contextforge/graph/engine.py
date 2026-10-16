"""图编排运行时。

这是一个 **LangGraph 风格**的最小实现：显式状态、节点、静态边、条件边、
循环、断点、以及基于 SQLite 的检查点恢复。

为什么不用直接装 LangGraph 而是自己实现一套（这是一个必须在文档里讲清楚的取舍）：

* 项目要在**没有网络、没有第三方依赖**的环境里跑通并测试，自研运行时保证这一点；
* 编排是本项目的核心展示点之一，自己实现才能真正解释清楚「条件分支 / 循环 /
  状态恢复」是怎么工作的，而不是把框架当黑盒调用；
* 接口刻意对齐 LangGraph 的概念（StateGraph / add_node / add_conditional_edges /
  checkpoint / interrupt），因此替换为真实 LangGraph 时上层工作流定义几乎不用改。

运行语义：

    state 通过 merge 累积；列表型字段是追加语义
    每个节点执行前后都会写入检查点，因此任意一步失败都能从最近检查点续跑
    `interrupt_before` 中的节点会在执行前暂停，等待人工确认（human-in-the-loop）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..storage import Database
from .state import AgentState

END = "__end__"
START = "__start__"

NodeFn = Callable[[AgentState, "RuntimeContext"], dict[str, Any] | None]
RouterFn = Callable[[AgentState, "RuntimeContext"], str]


class RuntimeContext(Protocol):
    """节点可用的运行时依赖。

    用 Protocol 而不是具体类，是为了让图运行时与业务实现解耦 ——
    测试时可以注入一个只有 `llm` 和 `tools` 的假上下文。
    """

    settings: Any
    llm: Any
    tools: Any


# --------------------------------------------------------------------------- #
# 检查点
# --------------------------------------------------------------------------- #
@dataclass
class Checkpoint:
    checkpoint_id: str
    thread_id: str
    node: str
    iteration: int
    state: dict[str, Any]
    created_at: float


class Checkpointer(Protocol):
    def save(self, thread_id: str, task_id: str, node: str, iteration: int, state: dict) -> None: ...

    def latest(self, thread_id: str) -> Checkpoint | None: ...

    def history(self, thread_id: str, limit: int = 20) -> list[Checkpoint]: ...


class SqliteCheckpointer:
    """把每一步的状态落盘，支持「暂停 → 恢复」。"""

    def __init__(self, db: Database) -> None:
        self.db = db

    def save(self, thread_id: str, task_id: str, node: str, iteration: int, state: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO checkpoints"
            " (checkpoint_id, thread_id, task_id, node, iteration, state, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"{thread_id}:{iteration}:{node}",
                thread_id,
                task_id,
                node,
                iteration,
                self.db.dump_json(state),
                time.time(),
            ),
        )

    def latest(self, thread_id: str) -> Checkpoint | None:
        row = self.db.query_one(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        )
        return self._row(row) if row else None

    def history(self, thread_id: str, limit: int = 20) -> list[Checkpoint]:
        rows = self.db.query(
            "SELECT * FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?",
            (thread_id, limit),
        )
        return [self._row(row) for row in rows]

    def clear(self, thread_id: str) -> None:
        self.db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))

    def _row(self, row) -> Checkpoint:
        return Checkpoint(
            checkpoint_id=row["checkpoint_id"],
            thread_id=row["thread_id"],
            node=row["node"],
            iteration=row["iteration"],
            state=self.db.load_json(row["state"], {}),
            created_at=row["created_at"],
        )


class MemoryCheckpointer:
    """进程内检查点，用于测试。"""

    def __init__(self) -> None:
        self._store: dict[str, list[Checkpoint]] = {}

    def save(self, thread_id: str, task_id: str, node: str, iteration: int, state: dict) -> None:
        import copy

        self._store.setdefault(thread_id, []).append(
            Checkpoint(
                checkpoint_id=f"{thread_id}:{iteration}:{node}",
                thread_id=thread_id,
                node=node,
                iteration=iteration,
                state=copy.deepcopy(state),
                created_at=time.time(),
            )
        )

    def latest(self, thread_id: str) -> Checkpoint | None:
        items = self._store.get(thread_id) or []
        return items[-1] if items else None

    def history(self, thread_id: str, limit: int = 20) -> list[Checkpoint]:
        return list(reversed((self._store.get(thread_id) or [])[-limit:]))


# --------------------------------------------------------------------------- #
# 图定义
# --------------------------------------------------------------------------- #
@dataclass
class CompiledGraph:
    nodes: dict[str, NodeFn]
    edges: dict[str, str]
    routers: dict[str, tuple[RouterFn, dict[str, str] | None]]
    entry: str
    max_steps: int = 40
    interrupt_before: set[str] = field(default_factory=set)
    interrupt_after: set[str] = field(default_factory=set)
    checkpointer: Checkpointer | None = None

    # ------------------------------------------------------------------ #
    def invoke(
        self,
        state: AgentState,
        context: Any,
        thread_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        skip_interrupt_once: bool = False,
    ) -> AgentState:
        thread_id = thread_id or state.thread_id or f"thread_{int(time.time() * 1000)}"
        state.thread_id = thread_id
        current = state.next_node or self.entry
        # 从检查点恢复时，上一次已经暂停过的那个节点不应该再被拦一次，
        # 否则 interrupt_before 会变成「永远无法通过」的死锁。
        pending_interrupt_skip = skip_interrupt_once

        steps = 0
        while current != END and steps < self.max_steps:
            steps += 1
            node_fn = self.nodes.get(current)
            if node_fn is None:
                state.status = "failed"
                state.error = f"未注册的节点：{current}"
                break

            if current in self.interrupt_before and not pending_interrupt_skip:
                state.status = "interrupted"
                state.next_node = current
                self._save(state, current)
                self._emit(on_event, {"type": "interrupt", "node": current, "reason": "before"})
                return state
            pending_interrupt_skip = False

            self._emit(on_event, {"type": "node_start", "node": current, "iteration": state.iteration})
            state.add_event(current, "node_start")

            try:
                update = node_fn(state, context)
            except Exception as exc:  # noqa: BLE001 - 单个节点失败不应吞掉整个任务
                state.status = "failed"
                state.error = f"节点 {current} 执行失败：{type(exc).__name__}: {exc}"
                state.add_event(current, "node_error", error=state.error)
                self._emit(on_event, {"type": "error", "node": current, "error": state.error})
                self._save(state, current)
                return state

            state.merge(update)
            state.add_event(current, "node_end")
            self._save(state, current)
            self._emit(
                on_event,
                {
                    "type": "node_end",
                    "node": current,
                    "iteration": state.iteration,
                    "status": state.status,
                },
            )

            if state.status in ("completed", "failed", "max_iterations"):
                break

            if current in self.interrupt_after:
                state.status = "interrupted"
                state.next_node = self._resolve_next(current, state, context)
                self._save(state, current)
                return state

            current = self._resolve_next(current, state, context)

        if steps >= self.max_steps and state.status == "running":
            state.status = "max_iterations"
            state.error = f"达到最大步数 {self.max_steps}"

        state.next_node = ""
        return state

    def resume(self, context: Any, thread_id: str, on_event=None) -> AgentState:
        """从最近一次检查点继续执行。"""
        if self.checkpointer is None:
            raise RuntimeError("没有配置检查点，无法恢复")

        checkpoint = self.checkpointer.latest(thread_id)
        if checkpoint is None:
            raise ValueError(f"找不到线程 {thread_id} 的检查点")

        state = AgentState.from_dict(checkpoint.state)
        state.status = "running"
        state.next_node = state.next_node or self._resolve_next(checkpoint.node, state, context)
        return self.invoke(
            state, context, thread_id=thread_id, on_event=on_event, skip_interrupt_once=True
        )

    # ------------------------------------------------------------------ #
    def _resolve_next(self, current: str, state: AgentState, context: Any) -> str:
        router_entry = self.routers.get(current)
        if router_entry is not None:
            router, mapping = router_entry
            decision = router(state, context)
            if decision == END:
                return END
            if mapping:
                target = mapping.get(decision)
                if target is None:
                    raise ValueError(
                        f"条件边 {current} 的路由结果 {decision!r} 没有映射到任何节点"
                    )
                return target
            return decision
        return self.edges.get(current, END)

    def _save(self, state: AgentState, node: str) -> None:
        if self.checkpointer is not None:
            self.checkpointer.save(
                state.thread_id, state.task_id, node, state.iteration, state.to_dict()
            )

    @staticmethod
    def _emit(on_event: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    # ------------------------------------------------------------------ #
    def describe(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "entry": self.entry,
            "edges": self.edges,
            "conditional": {name: (mapping or {}) for name, (_r, mapping) in self.routers.items()},
            "max_steps": self.max_steps,
            "interrupt_before": sorted(self.interrupt_before),
        }

    def to_mermaid(self) -> str:
        """导出 Mermaid 流程图，可直接贴进 README。"""
        lines = ["flowchart TD"]

        def key(name: str) -> str:
            return "END" if name == END else "START" if name == START else name.replace("-", "_")

        for name in self.nodes:
            lines.append(f'    {key(name)}["{name}"]')

        for source, target in self.edges.items():
            lines.append(f"    {key(source)} --> {key(target)}")

        for source, (_router, mapping) in self.routers.items():
            for decision, target in (mapping or {}).items():
                lines.append(f'    {key(source)} -->|{decision}| {key(target)}')

        lines.append(f'    START(["入口 {self.entry}"]) --> {key(self.entry)}')
        return "\n".join(lines)


class StateGraph:
    """以 LangGraph 的概念构建图。"""

    def __init__(self, max_steps: int = 40) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, tuple[RouterFn, dict[str, str] | None]] = {}
        self._entry: str | None = None
        self._interrupt_before: set[str] = set()
        self._interrupt_after: set[str] = set()
        self.max_steps = max_steps

    # ------------------------------------------------------------------ #
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        if name in self._nodes:
            raise ValueError(f"节点已存在：{name}")
        self._nodes[name] = fn
        if self._entry is None:
            self._entry = name
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFn,
        mapping: dict[str, str] | None = None,
    ) -> "StateGraph":
        self._routers[source] = (router, mapping)
        return self

    def set_entry(self, name: str) -> "StateGraph":
        if name not in self._nodes:
            raise ValueError(f"入口节点不存在：{name}")
        self._entry = name
        return self

    def set_finish(self, name: str) -> "StateGraph":
        self._edges[name] = END
        return self

    def interrupt_before(self, names: list[str]) -> "StateGraph":
        self._interrupt_before.update(names)
        return self

    def interrupt_after(self, names: list[str]) -> "StateGraph":
        self._interrupt_after.update(names)
        return self

    # ------------------------------------------------------------------ #
    def compile(self, checkpointer: Checkpointer | None = None) -> CompiledGraph:
        if not self._nodes:
            raise ValueError("图中没有任何节点")
        if self._entry is None:
            raise ValueError("没有设置入口节点")

        for source, target in self._edges.items():
            if target != END and target not in self._nodes:
                raise ValueError(f"边 {source} -> {target} 指向不存在的节点")
        for source, (_router, mapping) in self._routers.items():
            if source not in self._nodes:
                raise ValueError(f"条件边的起点不存在：{source}")
            for decision, target in (mapping or {}).items():
                if target != END and target not in self._nodes:
                    raise ValueError(f"条件边 {source} -> {target}（{decision}）指向不存在的节点")

        return CompiledGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            routers=dict(self._routers),
            entry=self._entry,
            max_steps=self.max_steps,
            interrupt_before=set(self._interrupt_before),
            interrupt_after=set(self._interrupt_after),
            checkpointer=checkpointer,
        )
