"""HTTP 接口层。

两条路径，共用同一套处理器：

* `contextforge.api.devserver` —— 标准库实现，零依赖，`python -m` 直接跑
* `contextforge.api.app`       —— FastAPI 实现，用于生产部署

另有直接调用 `Handlers` 的方式，便于写接口级单测（不起 HTTP 服务）。
"""

from .handlers import ApiError, Handlers, build_routes

__all__ = ["ApiError", "Handlers", "build_routes"]
