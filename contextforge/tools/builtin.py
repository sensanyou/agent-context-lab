"""内置工具集。

工具按用途分四类，覆盖「个人助手 + 代码理解」两个主场景：

* 通用：`get_local_time`、`calculator`
* 知识：`knowledge_search`、`knowledge_stats`
* 记忆：`memory_save`、`memory_search`、`skill_lookup`
* 代码：`fs_list`、`repo_overview`、`repo_symbols`、`repo_grep`、`repo_read`、`run_tests`

每个工具都是普通 Python 函数，注册时自动生成 JSON Schema。
新增工具不需要改动任何框架代码。
"""

from __future__ import annotations

import ast
import operator
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .registry import ToolFailure, ToolRegistry, tool
from .repo import RepoAnalyzer

# --------------------------------------------------------------------------- #
# 通用
# --------------------------------------------------------------------------- #
@tool(tags=("utility",))
def get_local_time() -> str:
    """获取当前本地日期和时间。"""
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_POWER = 1000


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("只支持数字常量")
        return node.value
    if isinstance(node, ast.BinOp):
        operation = _BINARY_OPS.get(type(node.op))
        if operation is None:
            raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > _MAX_POWER or abs(left) > 10**6):
            raise ValueError("指数过大，已拒绝计算")
        return operation(left, right)
    if isinstance(node, ast.UnaryOp):
        operation = _UNARY_OPS.get(type(node.op))
        if operation is None:
            raise ValueError(f"不支持的一元运算：{type(node.op).__name__}")
        return operation(_eval_node(node.operand))
    raise ValueError(f"不支持的表达式节点：{type(node).__name__}")


@tool(tags=("utility",))
def calculator(expression: str) -> str:
    """安全地计算一个算术表达式。

    Args:
        expression: 只包含数字与 + - * / // % ** 和括号的表达式
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return ToolFailure(f"表达式语法错误：{exc}")

    try:
        value = _eval_node(tree)
    except (ValueError, ZeroDivisionError) as exc:
        return ToolFailure(f"无法计算：{exc}")

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(round(value, 10))


# --------------------------------------------------------------------------- #
# 注册表工厂
# --------------------------------------------------------------------------- #
def build_registry(
    knowledge=None,
    memory=None,
    repo: RepoAnalyzer | None = None,
    allow_tests: bool = True,
    default_root: str | Path = ".",
    max_result_chars: int = 4000,
) -> ToolRegistry:
    """按可用的服务装配工具集。

    服务可为 None —— 缺失的工具不会被注册，也就不会出现在给模型的工具列表里。
    这比注册一堆「调用即报错」的占位工具要干净得多。
    """
    registry = ToolRegistry()
    registry.max_result_chars = max_result_chars
    registry.register_all(get_local_time, calculator)

    if knowledge is not None:

        @tool(name="knowledge_search", tags=("rag",))
        def knowledge_search(query: str, top_k: int = 5) -> str:
            """在个人知识库中检索相关片段，返回带编号的引用。

            Args:
                query: 检索问题
                top_k: 返回的片段数量
            """
            results = knowledge.search(query, top_k=max(1, min(int(top_k or 5), 20)))
            if not results:
                return "知识库中没有检索到相关内容。"
            lines = [f"检索到 {len(results)} 条相关片段："]
            for rank, item in enumerate(results, start=1):
                snippet = " ".join(item.chunk.text.split())
                lines.append(
                    f"- [{rank}] {item.chunk.citation()} (score={item.score:.3f}): {snippet}"
                )
            return "\n".join(lines)

        @tool(name="knowledge_stats", tags=("rag",))
        def knowledge_stats() -> str:
            """查看知识库的规模与索引状态。"""
            stats = knowledge.stats()
            return (
                f"文档 {stats['documents']} 篇，切片 {stats['chunks']} 个，"
                f"总字数 {stats['total_chars']}；向量后端 {stats['vector_store']['backend']}"
                f"（{stats['vector_store']['dim']} 维）；BM25 词项 {stats['bm25']['terms']} 个。"
            )

        registry.register_all(knowledge_search, knowledge_stats)

    if memory is not None:

        @tool(name="memory_save", tags=("memory",))
        def memory_save(content: str, kind: str = "fact", importance: float = 0.6) -> str:
            """把值得跨任务保留的信息写入长期记忆。

            Args:
                content: 要记住的内容
                kind: preference / fact / goal / progress / workflow
                importance: 重要程度，0 到 1
            """
            try:
                record = memory.long_term.remember(
                    content, kind=kind, importance=float(importance), source="tool"
                )
            except ValueError as exc:
                return ToolFailure(f"写入失败：{exc}")
            return f"已记住（{record.kind}）：{record.content}"

        @tool(name="memory_search", tags=("memory",))
        def memory_search(query: str, top_k: int = 5) -> str:
            """检索长期记忆中的用户偏好、项目事实与任务进度。

            Args:
                query: 检索问题
                top_k: 返回条数
            """
            memories = memory.long_term.recall(query, top_k=max(1, min(int(top_k or 5), 20)))
            if not memories:
                return "长期记忆中没有相关记录。"
            return "\n".join(
                f"- ({item.kind}, score={item.score:.2f}) {item.content}" for item in memories
            )

        @tool(name="skill_lookup", tags=("memory",))
        def skill_lookup(task: str, top_k: int = 2) -> str:
            """查找与任务匹配的可复用技能流程。

            Args:
                task: 任务描述
                top_k: 返回条数
            """
            skills = memory.skills.find(task, top_k=max(1, min(int(top_k or 2), 5)))
            if not skills:
                return "没有匹配的已注册技能。"
            return "\n".join(
                f"- {skill.name}：{' → '.join(skill.steps) or skill.description}" for skill in skills
            )

        registry.register_all(memory_save, memory_search, skill_lookup)

    if repo is not None:
        analyzer = repo

        @tool(name="repo_overview", tags=("code",))
        def repo_overview(max_depth: int = 3) -> str:
            """查看代码仓库的目录结构、文件统计与入口文件。

            Args:
                max_depth: 目录展开深度
            """
            return analyzer.overview(max_depth=max(1, min(int(max_depth or 3), 5)))

        @tool(name="repo_symbols", tags=("code",))
        def repo_symbols(path: str = ".", kind: str = "") -> str:
            """列出代码中的函数与类定义及其行号。

            Args:
                path: 相对仓库根目录的文件或目录
                kind: function / class / method，留空表示全部
            """
            return analyzer.symbols_text(path, kind=kind)

        @tool(name="repo_grep", tags=("code",))
        def repo_grep(pattern: str, glob: str = "") -> str:
            """在仓库源码中按正则搜索，用于定位符号引用与调用点。

            Args:
                pattern: 正则表达式
                glob: 文件后缀过滤，例如 *.py
            """
            return analyzer.grep(pattern, glob=glob)

        @tool(name="repo_read", tags=("code",))
        def repo_read(path: str, max_chars: int = 5000) -> str:
            """读取仓库中的一个文件（带行号）。

            Args:
                path: 相对仓库根目录的文件路径
                max_chars: 最大返回字符数
            """
            return analyzer.read(path, max_chars=max(500, min(int(max_chars or 5000), 20000)))

        @tool(name="repo_outline", tags=("code",))
        def repo_outline(path: str) -> str:
            """查看单个源文件的内部结构（函数、类、方法及其行号）。

            Args:
                path: 相对仓库根目录的文件路径
            """
            return analyzer.outline(path)

        registry.register_all(repo_overview, repo_symbols, repo_grep, repo_read, repo_outline)

        @tool(name="fs_list", tags=("code", "utility"))
        def fs_list(path: str = ".") -> str:
            """列出某个目录下的文件与子目录。

            Args:
                path: 相对仓库根目录的路径
            """
            try:
                target = analyzer._safe(path)
            except ValueError as exc:
                return ToolFailure(str(exc))
            if not target.is_dir():
                return ToolFailure(f"不是目录：{path}")
            entries = sorted(target.iterdir())
            listing = []
            for entry in entries[:60]:
                marker = "/" if entry.is_dir() else ""
                listing.append(f"{entry.name}{marker}")
            return "\n".join(listing) or "（空目录）"

        if allow_tests:

            @tool(name="run_tests", tags=("code",), dangerous=True)
            def run_tests(target: str = "tests", timeout: int = 90) -> str:
                """在仓库中运行 pytest 并返回结果摘要。

                Args:
                    target: 测试路径或文件
                    timeout: 超时秒数
                """
                try:
                    safe_target = analyzer._safe(target)
                except ValueError as exc:
                    return ToolFailure(str(exc))

                command = [
                    sys.executable, "-m", "pytest", str(safe_target),
                    "-q", "--no-header", "-p", "no:cacheprovider", "-x",
                ]
                try:
                    completed = subprocess.run(
                        command,
                        cwd=str(analyzer.root),
                        capture_output=True,
                        text=True,
                        timeout=max(5, min(int(timeout or 90), 300)),
                    )
                except subprocess.TimeoutExpired:
                    return ToolFailure(f"测试超时（>{timeout}s），已终止。")
                except OSError as exc:
                    return ToolFailure(f"无法启动 pytest：{exc}")

                output = (completed.stdout or "") + (completed.stderr or "")
                tail = "\n".join(output.strip().splitlines()[-25:])
                status = "通过" if completed.returncode == 0 else f"失败（退出码 {completed.returncode}）"
                return f"pytest {status}：\n{tail}"

            registry.register(run_tests)

    return registry


def registry_summary(registry: ToolRegistry) -> dict[str, Any]:
    return {
        "count": len(registry.names()),
        "tools": registry.names(),
        "tags": sorted({tag for spec in registry.specs() for tag in spec.tags}),
    }
