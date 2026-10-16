"""FastAPI 应用（生产路径）。

FastAPI 是可选依赖：未安装时导入本模块不会报错，只是 `create_app()` 会抛异常，
提示改用零依赖的 `contextforge.api.devserver`。

业务逻辑全部委托给 `Handlers`，这里只负责 HTTP 层：参数解析、
状态码、流式响应与静态文件挂载。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..runtime import AgentRuntime
from .handlers import ApiError, Handlers, build_routes

try:  # pragma: no cover - 取决于环境
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment]


def create_app(runtime: AgentRuntime | None = None):
    """构造 FastAPI 应用。"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "未安装 fastapi / uvicorn。请执行 pip install fastapi uvicorn，"
            "或使用零依赖开发服务器：python -m contextforge.api.devserver"
        )

    settings = get_settings()
    runtime = runtime or AgentRuntime(settings=settings)
    handlers = Handlers(runtime)
    routes = build_routes(handlers)

    app = FastAPI(title="ContextForge API", version=__import__("contextforge").__version__)

    def as_endpoint(handler):
        async def endpoint(request: Request):
            payload: dict[str, Any] = {}
            if request.method == "POST":
                try:
                    payload = await request.json()
                except Exception:  # noqa: BLE001
                    payload = {}
            payload.update({key: value for key, value in request.query_params.items()})
            try:
                return JSONResponse(handler(payload))
            except ApiError as exc:
                return JSONResponse({"error": exc.message}, status_code=exc.status)
            except Exception as exc:  # noqa: BLE001
                return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

        return endpoint

    for path, handler in routes.items():
        app.post(path)(as_endpoint(handler))
        app.get(path)(as_endpoint(handler))

    # ------------------------------------------------------------------ #
    @app.post("/api/chat/stream")
    async def chat_stream(request: Request):
        body = await request.json()
        question = str(body.get("question") or "")
        if not question:
            return JSONResponse({"error": "缺少参数：question"}, status_code=400)

        async def event_stream():
            import asyncio

            queue: list[dict[str, Any]] = []
            loop = asyncio.get_running_loop()

            def emit(event: dict[str, Any]) -> None:
                queue.append(event)
                loop.call_soon_threadsafe(lambda: None)

            task = asyncio.create_task(
                asyncio.to_thread(
                    runtime.ask,
                    question,
                    body.get("conversation_id"),
                    str(body.get("strategy") or ""),
                    str(body.get("retrieval_mode") or "hybrid"),
                    None,
                    None,
                    emit,
                )
            )

            while not task.done() or queue:
                while queue:
                    event = queue.pop(0)
                    yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                await asyncio.sleep(0.02)

            result = task.result()
            yield f"event: final\ndata: {json.dumps({'result': result.to_dict()}, ensure_ascii=False, default=str)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ------------------------------------------------------------------ #
    web_dir: Path = settings.web_dir
    index = web_dir / "index.html"

    @app.get("/")
    async def home():
        if index.is_file():
            return FileResponse(index)
        return JSONResponse({"error": "前端文件不存在", "expected": str(index)}, status_code=404)

    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    return app


app = None


def get_asgi_app():  # pragma: no cover - 供 uvicorn 使用
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)
