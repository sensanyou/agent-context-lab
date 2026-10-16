"""零依赖开发服务器。

`python -m contextforge.api.devserver` 即可启动完整后端与前端，
不需要 FastAPI / uvicorn / npm 任何东西 —— 这对「克隆下来就能跑」的求职作品集
至关重要：评审者不会为了看你的项目先装半小时依赖。

生产部署走 `app.py`（FastAPI + uvicorn），两者共用同一套处理器。
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from ..config import get_settings
from ..runtime import AgentRuntime
from .handlers import ApiError, Handlers, build_routes

STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8",
                ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}


class ContextForgeHandler(BaseHTTPRequestHandler):
    server_version = "ContextForge/dev"
    runtime: AgentRuntime
    handlers: Handlers
    routes: dict[str, Any]
    web_dir: Path
    quiet: bool = True

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path.startswith("/api/"):
            payload = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            self._dispatch(path, payload)
            return

        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        payload = self._read_json()

        if path == "/api/chat/stream":
            self._stream_chat(payload)
            return

        self._dispatch(path, payload)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        payload = self._read_json()
        doc_id = unquote(parsed.path.rsplit("/", 1)[-1])
        payload.setdefault("doc_id", doc_id)
        self._dispatch("/api/documents/delete", payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ------------------------------------------------------------------ #
    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {"value": value}

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, path: str, payload: dict[str, Any]) -> None:
        handler = self.routes.get(path)
        if handler is None:
            self._send_json(404, {"error": f"未知接口：{path}"})
            return
        try:
            result = handler(payload)
        except ApiError as exc:
            self._send_json(exc.status, {"error": exc.message})
            return
        except Exception as exc:  # noqa: BLE001 - 保证前端总能拿到结构化错误
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        self._send_json(200, result)

    # ------------------------------------------------------------------ #
    def _stream_chat(self, payload: dict[str, Any]) -> None:
        """SSE 流式返回执行过程事件，让界面能实时显示 Agent 在想什么。"""
        question = str(payload.get("question") or "")
        if not question:
            self._send_json(400, {"error": "缺少参数：question"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._cors()
        self.end_headers()

        def emit(event: dict[str, Any]) -> None:
            try:
                data = json.dumps(event, ensure_ascii=False, default=str)
                self.wfile.write(f"event: {event.get('type', 'message')}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        try:
            result = self.runtime.ask(
                question,
                conversation_id=payload.get("conversation_id"),
                strategy=str(payload.get("strategy") or ""),
                retrieval_mode=str(payload.get("retrieval_mode") or "hybrid"),
                on_event=emit,
            )
            emit({"type": "final", "result": result.to_dict()})
        except Exception as exc:  # noqa: BLE001
            emit({"type": "error", "error": f"{type(exc).__name__}: {exc}"})

    # ------------------------------------------------------------------ #
    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (self.web_dir / relative).resolve()

        if self.web_dir.resolve() not in target.parents and target != self.web_dir.resolve():
            self.send_error(403, "forbidden")
            return
        if not target.is_file():
            # 前端是单页应用：未知路径回落到 index.html
            target = self.web_dir / "index.html"
            if not target.is_file():
                self.send_error(404, "not found")
                return

        content_type = STATIC_TYPES.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ #
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not self.quiet:
            super().log_message(format, *args)


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    runtime: AgentRuntime | None = None,
    quiet: bool = True,
) -> tuple[ThreadingHTTPServer, AgentRuntime]:
    settings = get_settings()
    runtime = runtime or AgentRuntime(settings=settings)
    handlers = Handlers(runtime)

    ContextForgeHandler.runtime = runtime
    ContextForgeHandler.handlers = handlers
    ContextForgeHandler.routes = build_routes(handlers)
    ContextForgeHandler.web_dir = settings.web_dir
    ContextForgeHandler.quiet = quiet

    server = ThreadingHTTPServer((host, port), ContextForgeHandler)
    server.daemon_threads = True
    return server, runtime


def serve(host: str = "127.0.0.1", port: int = 8000, quiet: bool = True, background: bool = False):
    server, runtime = create_server(host=host, port=port, quiet=quiet)
    banner = f"ContextForge 已启动：http://{host}:{port}"
    print(banner)
    print(f"模型：{runtime.llm.describe().get('name')} | 工具 {len(runtime.tools.names())} 个 | "
          f"知识库 {runtime.knowledge.stats()['documents']} 篇文档")

    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, runtime

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        server.shutdown()
    return server, runtime


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ContextForge 开发服务器")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, quiet=not args.verbose)


if __name__ == "__main__":
    main()
