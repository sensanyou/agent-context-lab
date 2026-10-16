"""工具注册表。

设计目标：让「加一个工具」等于「写一个带类型注解和 docstring 的 Python 函数」。
工具 schema 从函数签名自动推导，避免手写 JSON Schema 时出现的
「文档和实现不一致」这类经典问题。

执行侧统一处理四件事，这些常被最简实现忽略：

* 参数 JSON 解析失败 → 返回可读错误而不是抛异常给上层；
* 单个工具异常 → 被捕获并记录，不会中断整个 Agent 循环；
* 超长输出 → 按预算截断，避免一次 `ls` 就把上下文挤爆；
* 执行耗时 → 进入轨迹，用于成本与延迟分析。
"""

from __future__ import annotations

import inspect
import json
import time
import typing
from typing import Any, Callable, get_args, get_origin

from ..types import ToolExecution, ToolSpec

_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolFailure(str):
    """工具「执行了但没成功」的返回值。

    工具函数的返回值要同时满足两个用途：
    1. 作为结果文本回灌给模型（所以必须是 `str`）；
    2. 让注册表知道这次调用失败了（所以要能被识别出来）。

    直接抛异常只能满足第 2 点 —— 但很多失败是「预期内的业务失败」
    （例如表达式非法、路径不存在），直接把异常抛出去既难读也不利于
    直接调用工具的单元测试。因此用一个 `str` 子类承载失败信息：
    它既是可读文本，又能让 `ToolRegistry.execute` 把 `ok` 置为 `False`，
    从而让上层（重试路由、评测指标、执行轨迹）能如实区分成功与失败。
    """

    __slots__ = ()


def tool(
    _func: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    dangerous: bool = False,
    tags: tuple[str, ...] = (),
):
    """把一个函数注册为工具。

    用法::

        @tool
        def calculator(expression: str) -> str:
            \"\"\"计算一个算术表达式。\"\"\"

        @tool(name="kb_search", tags=("rag",))
        def search(query: str, top_k: int = 5) -> str: ...
    """

    def decorate(func: Callable) -> Callable:
        spec = build_spec(func, name=name, description=description, dangerous=dangerous, tags=tags)
        setattr(func, "__tool_spec__", spec)
        return func

    return decorate(_func) if _func is not None else decorate


def build_spec(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
    dangerous: bool = False,
    tags: tuple[str, ...] = (),
) -> ToolSpec:
    signature = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    summary = doc.strip().split("\n", 1)[0] if doc else func.__name__

    # 关键：文件里只要写了 `from __future__ import annotations`，
    # 所有注解都会变成字符串（"int" 而不是 int），类型推导会全部退化成 string。
    # 因此必须用 get_type_hints 把字符串还原成真实类型。
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # noqa: BLE001 - 无法解析时退回原始注解
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter_name, parameter in signature.parameters.items():
        if parameter_name in ("self", "cls"):
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(parameter_name, parameter.annotation)
        schema = _annotation_to_schema(annotation, doc, parameter_name)
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)
        else:
            if parameter.default is not None and not isinstance(parameter.default, (list, dict)):
                schema["default"] = parameter.default
        properties[parameter_name] = schema

    return ToolSpec(
        name=name or func.__name__,
        description=description or summary,
        parameters={"type": "object", "properties": properties, "required": required},
        func=func,
        dangerous=dangerous,
        tags=tags,
    )


def _annotation_to_schema(annotation: Any, doc: str, parameter_name: str) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        return {"type": "string", "description": _param_doc(doc, parameter_name)}

    description = _param_doc(doc, parameter_name)

    # 先剥掉 Optional / 联合类型：`list[str] | None` 的 get_origin 是 UnionType 而不是 list，
    # 不处理就会把数组参数错误地当成 string。
    annotation = _unwrap_optional(annotation)

    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        item_type = _TYPE_MAP.get(args[0], "string") if args else "string"
        return {"type": "array", "items": {"type": item_type}, "description": description}
    if origin is dict:
        return {"type": "object", "description": description}

    if annotation in _TYPE_MAP:
        return {"type": _TYPE_MAP[annotation], "description": description}

    return {"type": "string", "description": description}


def _unwrap_optional(annotation: Any) -> Any:
    """把 `X | None` / `Optional[X]` 还原为 X；多成员联合取第一个非 None 分支。"""
    import types

    origin = get_origin(annotation)
    if origin is types.UnionType or str(origin) == "typing.Union":
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
        if members:
            return members[0]
    return annotation


def _param_doc(doc: str, parameter_name: str) -> str:
    """从 docstring 的 `Args:` 段里提取参数说明。"""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{parameter_name}:") or stripped.startswith(f"{parameter_name} ("):
            return stripped.split(":", 1)[-1].strip()
    return ""


class ToolRegistry:
    """工具集合与执行器。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self.max_result_chars = 4000

    # ------------------------------------------------------------------ #
    def register(self, func: Callable | ToolSpec, **kwargs: Any) -> ToolSpec:
        if isinstance(func, ToolSpec):
            spec = func
        else:
            # 优先复用 @tool 装饰时已经推导好的 schema（含 dangerous / tags 等元信息）
            decorated = getattr(func, "__tool_spec__", None)
            spec = decorated if (decorated is not None and not kwargs) else build_spec(func, **kwargs)
        self._specs[spec.name] = spec
        return spec

    def register_all(self, *funcs: Callable) -> None:
        for func in funcs:
            self.register(func)

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self, only: list[str] | None = None) -> list[ToolSpec]:
        if only is None:
            return [self._specs[name] for name in self.names()]
        return [self._specs[name] for name in only if name in self._specs]

    def openai_schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        return [spec.to_openai() for spec in self.specs(only)]

    # ------------------------------------------------------------------ #
    def execute(
        self,
        name: str,
        arguments: dict[str, Any] | str | None = None,
        tool_call_id: str = "",
    ) -> ToolExecution:
        started = time.perf_counter()
        spec = self._specs.get(name)

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return ToolExecution(
                    tool_name=name,
                    tool_call_id=tool_call_id,
                    arguments={},
                    result=f"参数解析失败：{exc}",
                    ok=False,
                    error=str(exc),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            arguments = parsed if isinstance(parsed, dict) else {}

        arguments = arguments or {}

        if spec is None:
            return ToolExecution(
                tool_name=name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=f"错误：未注册的工具 {name}",
                ok=False,
                error="unknown_tool",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        if spec.func is None:
            return ToolExecution(
                tool_name=name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=f"错误：工具 {name} 没有实现",
                ok=False,
                error="no_implementation",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            result = spec.func(**arguments)
            ok = not isinstance(result, ToolFailure)
            error = str(result) if not ok else ""
        except TypeError as exc:
            result = f"参数不匹配：{exc}"
            ok = False
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - 工具失败不应中断 Agent
            result = f"工具执行失败：{type(exc).__name__}: {exc}"
            ok = False
            error = f"{type(exc).__name__}: {exc}"

        text = str(result)
        truncated = len(text) > self.max_result_chars
        if truncated:
            text = text[: self.max_result_chars] + "\n…（工具输出已截断）"

        return ToolExecution(
            tool_name=name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            result=text,
            ok=ok,
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
        )

    # ------------------------------------------------------------------ #
    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
                "dangerous": spec.dangerous,
                "tags": list(spec.tags),
            }
            for spec in self.specs()
        ]
