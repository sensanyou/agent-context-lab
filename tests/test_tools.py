"""工具层测试：schema 推导、执行与错误处理、代码仓库分析。"""

from __future__ import annotations

from contextforge.rag import KnowledgeBase
from contextforge.memory import MemoryManager
from contextforge.tools import RepoAnalyzer, ToolRegistry, build_registry, calculator
from contextforge.tools.repo import find_repo_root
from contextforge.tools.registry import build_spec, tool


# --------------------------------------------------------------------------- #
# schema 推导
# --------------------------------------------------------------------------- #
def test_decorator_infers_schema_from_annotations():
    @tool
    def sample(query: str, top_k: int = 5, tags: list[str] | None = None) -> str:
        """示例工具。

        Args:
            query: 检索问题
            top_k: 返回条数
        """
        return "ok"

    spec = sample.__tool_spec__
    assert spec.name == "sample"
    assert spec.description == "示例工具。"
    assert spec.parameters["properties"]["query"]["type"] == "string"
    assert spec.parameters["properties"]["top_k"]["type"] == "integer"
    assert spec.parameters["properties"]["top_k"]["default"] == 5
    assert spec.parameters["properties"]["tags"]["type"] == "array"
    assert spec.parameters["required"] == ["query"]
    assert "检索问题" in spec.parameters["properties"]["query"]["description"]


def test_schema_resolves_string_annotations_from_future_import():
    """`from __future__ import annotations` 会把注解变成字符串，
    必须用 get_type_hints 还原，否则所有参数都会退化成 string。"""
    from contextforge.tools.builtin import build_registry as _build

    registry = _build(repo=RepoAnalyzer("."))
    spec = registry.get("repo_overview")
    assert spec is not None
    assert spec.parameters["properties"]["max_depth"]["type"] == "integer"


def test_register_keeps_decorated_metadata():
    @tool(name="危险工具", dangerous=True, tags=("admin",))
    def risky(target: str) -> str:
        """危险操作。"""
        return target

    registry = ToolRegistry()
    registry.register(risky)
    spec = registry.get("危险工具")
    assert spec is not None
    assert spec.dangerous is True
    assert spec.tags == ("admin",)


def test_openai_schemas_shape():
    registry = ToolRegistry()
    registry.register(calculator)
    schemas = registry.openai_schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "calculator"


def test_build_spec_without_docstring_falls_back_to_name():
    def plain(a: str) -> str:
        return a

    spec = build_spec(plain)
    assert spec.description == "plain"


# --------------------------------------------------------------------------- #
# 执行与错误处理
# --------------------------------------------------------------------------- #
def test_execute_unknown_tool_returns_error_result():
    registry = ToolRegistry()
    execution = registry.execute("不存在的工具", {})
    assert execution.ok is False
    assert "未注册的工具" in execution.result


def test_execute_invalid_json_arguments_is_handled():
    registry = ToolRegistry()
    registry.register(calculator)
    execution = registry.execute("calculator", "{不是合法 JSON")
    assert execution.ok is False
    assert "参数解析失败" in execution.result


def test_execute_wrong_argument_type_is_handled():
    registry = ToolRegistry()
    registry.register(calculator)
    execution = registry.execute("calculator", {"unknown": 1})
    assert execution.ok is False
    assert "参数不匹配" in execution.result


def test_execute_truncates_long_output():
    @tool
    def verbose() -> str:
        """输出很长。"""
        return "内容" * 5000

    registry = ToolRegistry()
    registry.max_result_chars = 100
    registry.register(verbose)
    execution = registry.execute("verbose", {})
    assert execution.truncated is True
    assert len(execution.result) < 200


def test_execute_captures_exception_without_raising():
    @tool
    def boom() -> str:
        """总会抛异常。"""
        raise RuntimeError("内部错误")

    registry = ToolRegistry()
    registry.register(boom)
    execution = registry.execute("boom", {})
    assert execution.ok is False
    assert "RuntimeError" in execution.error


# --------------------------------------------------------------------------- #
# 内置工具
# --------------------------------------------------------------------------- #
def test_calculator_evaluates_and_formats():
    assert calculator("(128 + 72) * 3") == "600"
    assert calculator("7 / 2") == "3.5"
    assert calculator("2 ** 10") == "1024"


def test_calculator_rejects_dangerous_expressions():
    assert "无法计算" in calculator("__import__('os').system('echo hi')")
    assert "不支持的表达式节点" in calculator("open('x')")
    assert "指数过大" in calculator("9 ** 999999")


def test_calculator_reports_syntax_error():
    assert "语法错误" in calculator("1 +")


def test_build_registry_only_registers_available_services():
    bare = build_registry()
    assert set(bare.names()) == {"calculator", "get_local_time"}

    with_kb = build_registry(knowledge=object())
    assert "knowledge_search" in with_kb.names()


def test_knowledge_search_tool_returns_citations(db, settings, llm):
    knowledge = KnowledgeBase(db, settings, llm=llm)
    knowledge.add_text("混合检索使用 RRF 融合两路排名，对分数量纲不敏感。", title="检索")
    registry = build_registry(knowledge=knowledge)

    execution = registry.execute("knowledge_search", {"query": "RRF 融合", "top_k": 3})
    assert execution.ok is True
    assert "[1]" in execution.result
    assert "检索#0" in execution.result


def test_knowledge_search_tool_when_empty(db, settings, llm):
    knowledge = KnowledgeBase(db, settings, llm=llm)
    registry = build_registry(knowledge=knowledge)
    execution = registry.execute("knowledge_search", {"query": "任何问题"})
    assert execution.ok is True
    assert "没有检索到" in execution.result


def test_memory_tools_roundtrip(db):
    manager = MemoryManager(db)
    registry = build_registry(memory=manager)

    saved = registry.execute("memory_save", {"content": "用户偏好中文回答", "kind": "preference"})
    assert saved.ok is True
    found = registry.execute("memory_search", {"query": "偏好什么语言"})
    assert found.ok is True
    assert "中文" in found.result


# --------------------------------------------------------------------------- #
# 代码仓库分析
# --------------------------------------------------------------------------- #
def test_repo_overview_lists_structure(sample_repo):
    analyzer = RepoAnalyzer(sample_repo)
    text = analyzer.overview()
    assert "demo_pkg" in text
    assert "文件统计" in text
    assert "README.md" in text


def test_repo_overview_skips_noise_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "site.py").write_text("# 依赖代码", encoding="utf-8")
    text = RepoAnalyzer(tmp_path).overview()
    assert "src/" in text
    assert "site.py" not in text


def test_repo_symbols_extracts_functions_and_methods(sample_repo):
    analyzer = RepoAnalyzer(sample_repo)
    symbols = analyzer.symbols("demo_pkg/service.py")
    names = {symbol.name for symbol in symbols}
    assert {"NoteService", "summarize_scores", "rank_notes"} <= names
    kinds = {symbol.kind for symbol in symbols}
    assert "method" in kinds and "class" in kinds


def test_repo_symbols_text_is_readable(sample_repo):
    text = RepoAnalyzer(sample_repo).symbols_text("demo_pkg")
    assert "function" in text or "class" in text
    assert "service.py" in text


def test_repo_grep_finds_references(sample_repo):
    text = RepoAnalyzer(sample_repo).grep("InMemoryStore")
    assert "storage.py" in text
    assert "service.py" in text


def test_repo_grep_invalid_regex_is_reported(sample_repo):
    assert "正则表达式无效" in RepoAnalyzer(sample_repo).grep("([unclosed")


def test_repo_read_numbers_lines(sample_repo):
    text = RepoAnalyzer(sample_repo).read("demo_pkg/models.py", max_chars=400)
    assert "|" in text
    assert "class Note" in text


def test_repo_outline_for_single_file(sample_repo):
    text = RepoAnalyzer(sample_repo).outline("demo_pkg/service.py")
    assert "NoteService" in text
    assert "summarize_scores" in text


def test_repo_analyzer_blocks_path_escape(sample_repo):
    analyzer = RepoAnalyzer(sample_repo)
    try:
        analyzer.read("../../outside.txt")
    except ValueError as exc:
        assert "路径越界" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("必须阻止越界访问")


def test_repo_imports_graph(sample_repo):
    graph = RepoAnalyzer(sample_repo).imports()
    assert "demo_pkg.service" in graph
    assert "demo_pkg.storage" in graph["demo_pkg.service"]


def test_repo_analyzer_rejects_missing_root(tmp_path):
    try:
        RepoAnalyzer(tmp_path / "不存在")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("不存在的目录应当报错")


def test_find_repo_root_walks_upwards(sample_repo):
    root = find_repo_root(sample_repo / "demo_pkg")
    assert root is not None
    assert (root / "README.md").exists()
