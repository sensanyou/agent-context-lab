"""接口层测试。

直接调用 `Handlers`，不起 HTTP 服务 —— 接口逻辑与传输层解耦正是这么设计的收益。
另外用标准库 dev server 起一个真实进程，验证端到端（含静态文件与 SSE）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from contextforge.api.handlers import ApiError, Handlers, build_routes


@pytest.fixture
def handlers(runtime) -> Handlers:
    return Handlers(runtime)


def _call(handlers: Handlers, path: str, payload: dict | None = None) -> dict:
    routes = build_routes(handlers)
    return routes[path](payload or {})


# --------------------------------------------------------------------------- #
# 基础接口
# --------------------------------------------------------------------------- #
def test_routes_are_registered(handlers: Handlers):
    routes = build_routes(handlers)
    for path in ("/api/health", "/api/chat", "/api/search", "/api/documents", "/api/evaluate"):
        assert path in routes


def test_health(handlers: Handlers):
    assert _call(handlers, "/api/health")["ok"] is True


def test_status(handlers: Handlers):
    status = _call(handlers, "/api/status")
    assert status["llm"]["name"]
    assert status["workflow"]["nodes"]


def test_workflow_returns_mermaid(handlers: Handlers):
    payload = _call(handlers, "/api/workflow")
    assert payload["mermaid"].startswith("flowchart TD")
    assert "build_context" in payload["graph"]["nodes"] or payload["graph"]["nodes"]


def test_tools_listing(handlers: Handlers):
    tools = _call(handlers, "/api/tools")["tools"]
    assert any(tool["name"] == "calculator" for tool in tools)
    assert any(tool["name"] == "knowledge_search" for tool in tools)


# --------------------------------------------------------------------------- #
# 对话
# --------------------------------------------------------------------------- #
def test_chat_endpoint(handlers: Handlers):
    result = _call(handlers, "/api/chat", {"question": "上下文压缩为什么要以轮次为单位？"})
    assert result["status"] == "completed"
    assert result["answer"]
    assert result["context_report"]["tokens_after"] > 0
    assert result["citations"]


def test_chat_requires_question(handlers: Handlers):
    with pytest.raises(ApiError) as error:
        _call(handlers, "/api/chat", {})
    assert "question" in str(error.value.message)


def test_chat_accepts_strategy_override(handlers: Handlers):
    result = _call(handlers, "/api/chat", {"question": "压缩策略有哪些？", "strategy": "sliding"})
    assert result["context_report"]["strategy"] == "sliding"


def test_chat_accepts_retrieval_mode_override(handlers: Handlers):
    result = _call(handlers, "/api/chat", {"question": "压缩策略有哪些？", "retrieval_mode": "dense"})
    assert result["status"] == "completed"


# --------------------------------------------------------------------------- #
# 知识库
# --------------------------------------------------------------------------- #
def test_documents_listing(handlers: Handlers):
    payload = _call(handlers, "/api/documents")
    assert payload["stats"]["chunks"] >= 0
    assert isinstance(payload["documents"], list)


def test_add_and_delete_document(handlers: Handlers, runtime):
    added = _call(
        handlers,
        "/api/documents/add",
        {"text": "这段内容用于验证接口新增与删除文档的完整流程。", "title": "接口测试"},
    )
    doc_id = added["document"]["doc_id"]
    assert any(item["doc_id"] == doc_id for item in _call(handlers, "/api/documents")["documents"])

    deleted = _call(handlers, "/api/documents/delete", {"doc_id": doc_id})
    assert deleted["deleted"] is True


def test_add_document_from_path(handlers: Handlers, sample_docs):
    payload = _call(handlers, "/api/documents/add", {"path": str(sample_docs)})
    assert payload["type"] == "directory"
    assert payload["added"] >= 1


def test_add_document_missing_path_reports_error(handlers: Handlers, tmp_path):
    with pytest.raises(ApiError):
        _call(handlers, "/api/documents/add", {"path": str(tmp_path / "不存在.md")})


def test_search_endpoint(handlers: Handlers):
    payload = _call(handlers, "/api/search", {"query": "RRF 融合", "top_k": 3})
    assert payload["query"] == "RRF 融合"
    assert payload["hits"]
    assert payload["trace"] is not None
    assert "dense_hits" in payload["trace"]


def test_search_requires_query(handlers: Handlers):
    with pytest.raises(ApiError):
        _call(handlers, "/api/search", {})


# --------------------------------------------------------------------------- #
# 记忆
# --------------------------------------------------------------------------- #
def test_memory_add_list_forget(handlers: Handlers):
    created = _call(
        handlers, "/api/memory/add", {"content": "接口测试记忆", "kind": "fact", "importance": 0.7}
    )
    memory_id = created["memory_id"]

    listing = _call(handlers, "/api/memory")
    assert any(item["memory_id"] == memory_id for item in listing["memories"])
    assert listing["skills"]

    assert _call(handlers, "/api/memory/forget", {"memory_id": memory_id})["forgotten"] is True


# --------------------------------------------------------------------------- #
# 观测与评测
# --------------------------------------------------------------------------- #
def test_traces_endpoints(handlers: Handlers):
    result = _call(handlers, "/api/chat", {"question": "轨迹测试问题"})
    trace_id = result["trace_id"]

    listing = _call(handlers, "/api/traces", {"limit": 5})
    assert listing["traces"]
    assert listing["aggregate"]["traces"] >= 1

    trace = _call(handlers, "/api/traces/get", {"trace_id": trace_id})
    assert trace["trace_id"] == trace_id
    assert trace["spans"]


def test_trace_not_found(handlers: Handlers):
    with pytest.raises(ApiError) as error:
        _call(handlers, "/api/traces/get", {"trace_id": "不存在"})
    assert error.value.status == 404


def test_metrics_endpoint(handlers: Handlers):
    payload = _call(handlers, "/api/metrics")
    assert "traces" in payload
    assert "context" in payload


def test_evaluate_endpoint(handlers: Handlers):
    payload = _call(handlers, "/api/evaluate", {"modes": ["hybrid"], "strategies": ["none", "hybrid"]})
    assert "context_experiment" in payload
    assert "agent_experiment" in payload
    rows = payload["context_experiment"]["rows"]
    assert {row["strategy"] for row in rows} == {"none", "hybrid"}
    assert "markdown" in payload
    assert "压缩策略" in payload["markdown"]["context"]


# --------------------------------------------------------------------------- #
# 真实 HTTP（标准库 dev server）
# --------------------------------------------------------------------------- #
@pytest.fixture
def live_server(settings, db, kb):
    """起一个真实的 HTTP 服务，验证传输层。"""
    from contextforge.api.devserver import create_server
    from contextforge.runtime import AgentRuntime
    from contextforge.llm import MockLLM

    runtime = AgentRuntime(settings=settings, db=db, llm=MockLLM(), knowledge=kb, enable_tests=False)
    server, runtime = create_server(host="127.0.0.1", port=0, runtime=runtime, quiet=True)
    thread = __import__("threading").Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_http_health(live_server: str):
    assert _get(f"{live_server}/api/health")["ok"] is True


def test_http_serves_frontend(live_server: str):
    with urllib.request.urlopen(f"{live_server}/", timeout=10) as response:
        html = response.read().decode("utf-8")
    assert response.status == 200
    assert "ContextForge" in html
    assert "vue.global.prod.js" in html


def test_http_vendored_vue_is_present(live_server: str):
    """前端依赖的 Vue 运行时必须随项目一起提供，不能依赖 CDN。"""
    with urllib.request.urlopen(f"{live_server}/vendor/vue.global.prod.js", timeout=10) as response:
        content = response.read().decode("utf-8", errors="replace")
    assert response.status == 200
    assert "Vue" in content[:400]


def test_http_chat_post(live_server: str):
    request = urllib.request.Request(
        f"{live_server}/api/chat",
        data=json.dumps({"question": "上下文压缩有哪些策略？"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    assert payload["status"] == "completed"
    assert payload["answer"]


def test_http_unknown_route_returns_404(live_server: str):
    # 路径必须保持 ASCII：http.client 会用 ascii 编码请求行，
    # 非 ASCII 路径会在客户端就抛 UnicodeEncodeError，测不到服务端行为。
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{live_server}/api/does-not-exist", timeout=10)
    assert error.value.code == 404


def test_http_search_query_params(live_server: str):
    payload = _get(f"{live_server}/api/search?query=RRF&top_k=3")
    assert payload["query"] == "RRF"


def test_http_chat_stream_emits_events(live_server: str):
    request = urllib.request.Request(
        f"{live_server}/api/chat/stream",
        data=json.dumps({"question": "现在几点了？"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")

    assert "event: node_start" in raw
    assert "event: final" in raw
    assert '"result"' in raw
