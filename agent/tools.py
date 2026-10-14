#工具注册与执行
from datetime import datetime
from typing import Any, Callable


# 1. 定义工具的实际执行函数
def get_local_time() -> str:
    """获取运行程序的这台电脑的本地时间。"""
    return datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


# 2. 注册工具：工具名称 -> Python 函数
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_local_time": get_local_time,
}


# 3. 告诉大模型有哪些工具可以使用
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "获取当前本地日期和时间",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


# 4. 根据模型返回的工具名称，执行对应的 Python 函数
def execute_tool(
    name: str,
    arguments: dict[str, Any],
) -> str:
    tool = TOOL_REGISTRY.get(name)

    if tool is None:
        return f"错误：未注册的工具 {name}"

    try:
        result = tool(**arguments)
        return str(result)
    except Exception as exc:
        return f"工具执行失败：{exc}"
