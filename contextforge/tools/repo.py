"""代码仓库理解。

这是「Coding Agent」场景里最容易被低估的一环：模型看不到整个仓库，
它只能**用工具去看**。所以工具的输出结构直接决定了模型能否建立正确的结构认知。

三个工具解决三个不同层次的认知：

* `overview`  —— 这个项目由哪些目录/文件构成，入口在哪里（宏观）
* `symbols`   —— 有哪些函数和类，各自在哪一行（中观，Python 用 AST 解析）
* `grep`      —— 某个符号在哪些地方被引用（微观，定位调用链）

全部限制在根目录内，且跳过 `.venv` / `.git` / `node_modules`，
避免把依赖源码当成项目代码返回给模型。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..rag.loaders import SKIP_DIRS, iter_files

ENTRY_HINTS = ("main.py", "app.py", "cli.py", "__main__.py", "manage.py", "server.py")
DOC_HINTS = ("README.md", "README.rst", "ARCHITECTURE.md", "docs")


@dataclass
class Symbol:
    name: str
    kind: str  # function | class | method
    path: str
    lineno: int
    end_lineno: int = 0
    doc: str = ""
    parent: str = ""

    def to_line(self) -> str:
        location = f"{self.path}:{self.lineno}"
        label = f"{self.parent}.{self.name}" if self.parent else self.name
        suffix = f"  # {self.doc}" if self.doc else ""
        return f"{self.kind:<9} {label:<40} {location}{suffix}"


@dataclass
class RepoAnalyzer:
    """针对本地代码仓库的只读分析器。"""

    root: Path
    max_files: int = 400
    _cache: dict[str, list[Symbol]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"目录不存在：{self.root}")

    # ------------------------------------------------------------------ #
    def _safe(self, path: str | Path | None) -> Path:
        if path in (None, "", "."):
            return self.root
        candidate = (self.root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        if self.root not in candidate.parents and candidate != self.root:
            raise ValueError(f"路径越界：{path}")
        return candidate

    # ------------------------------------------------------------------ #
    def overview(self, max_depth: int = 3) -> str:
        """目录结构 + 文件类型统计 + 可能的入口文件。"""
        lines: list[str] = [f"仓库根目录：{self.root}", "", "目录结构："]

        for path in sorted(self.root.iterdir()):
            if path.name.startswith(".") and path.name not in (".env.example",):
                continue
            if path.name in SKIP_DIRS:
                continue
            if path.is_dir():
                lines.append(f"  {path.name}/")
                self._walk_into(path, lines, depth=1, max_depth=max_depth)
            elif path.is_file():
                lines.append(f"  {path.name}")

        suffix_counts: dict[str, int] = {}
        python_files = 0
        total_lines = 0
        for file_path in iter_files(self.root, max_files=self.max_files):
            suffix = file_path.suffix.lower() or "(无后缀)"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
            if suffix == ".py":
                python_files += 1
                try:
                    total_lines += sum(1 for _ in file_path.open("r", encoding="utf-8", errors="replace"))
                except OSError:
                    pass

        top = sorted(suffix_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        lines.append("")
        lines.append("文件统计：" + "，".join(f"{suffix} × {count}" for suffix, count in top))
        lines.append(f"Python 文件 {python_files} 个，约 {total_lines} 行")

        entries = [
            name
            for name in ENTRY_HINTS
            if (self.root / name).exists() or any((self.root / sub / name).exists() for sub in self._subdirs())
        ]
        if entries:
            lines.append("可能的入口文件：" + "，".join(entries))

        docs = [name for name in ("README.md", "ARCHITECTURE.md") if (self.root / name).exists()]
        if docs:
            lines.append("文档：" + "，".join(docs))

        return "\n".join(lines)

    def _subdirs(self, limit: int = 8) -> list[str]:
        return [
            path.name
            for path in sorted(self.root.iterdir())
            if path.is_dir() and not path.name.startswith(".") and path.name not in SKIP_DIRS
        ][:limit]

    @staticmethod
    def _walk_into(directory: Path, lines: list[str], depth: int, max_depth: int) -> None:
        if depth >= max_depth:
            return
        try:
            children = sorted(directory.iterdir())
        except OSError:
            return
        shown = 0
        for child in children:
            if child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            if shown >= 12:
                lines.append("    " * depth + "  …")
                break
            if child.is_dir():
                lines.append("    " * depth + f"  {child.name}/")
                RepoAnalyzer._walk_into(child, lines, depth + 1, max_depth)
            else:
                lines.append("    " * depth + f"  {child.name}")
            shown += 1

    # ------------------------------------------------------------------ #
    def symbols(self, path: str | Path | None = None, kind: str = "") -> list[Symbol]:
        """解析 Python 文件中的函数/类定义。"""
        target = self._safe(path)
        files = [target] if target.is_file() else [
            item for item in iter_files(target, suffixes={".py"}, max_files=self.max_files)
        ]

        symbols: list[Symbol] = []
        for file_path in files:
            symbols.extend(self._parse_file(file_path))
        if kind:
            symbols = [symbol for symbol in symbols if symbol.kind == kind]
        return symbols

    def symbols_text(self, path: str | Path | None = None, kind: str = "", limit: int = 120) -> str:
        symbols = self.symbols(path, kind)
        if not symbols:
            return "未找到函数或类定义。"
        header = f"共 {len(symbols)} 个符号（最多展示 {limit} 个）："
        body = "\n".join(symbol.to_line() for symbol in symbols[:limit])
        return f"{header}\n{body}"

    def _parse_file(self, file_path: Path) -> list[Symbol]:
        key = str(file_path)
        if key in self._cache:
            return self._cache[key]

        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            self._cache[key] = []
            return []

        relative = str(file_path.relative_to(self.root))
        symbols: list[Symbol] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="function",
                        path=relative,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", node.lineno),
                        doc=_first_line(ast.get_docstring(node)),
                    )
                )
            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    Symbol(
                        name=node.name,
                        kind="class",
                        path=relative,
                        lineno=node.lineno,
                        end_lineno=getattr(node, "end_lineno", node.lineno),
                        doc=_first_line(ast.get_docstring(node)),
                    )
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols.append(
                            Symbol(
                                name=child.name,
                                kind="method",
                                path=relative,
                                lineno=child.lineno,
                                end_lineno=getattr(child, "end_lineno", child.lineno),
                                doc=_first_line(ast.get_docstring(child)),
                                parent=node.name,
                            )
                        )

        self._cache[key] = symbols
        return symbols

    # ------------------------------------------------------------------ #
    def imports(self) -> dict[str, list[str]]:
        """本地模块之间的导入关系（用于画依赖图）。"""
        graph: dict[str, list[str]] = {}
        for file_path in iter_files(self.root, suffixes={".py"}, max_files=self.max_files):
            try:
                tree = ast.parse(file_path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue

            relative = str(file_path.relative_to(self.root))
            module = relative[:-3].replace("\\", "/").replace("/", ".")
            targets: list[str] = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    targets.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        # 相对导入（from .storage import X）必须还原成绝对模块名，
                        # 否则包内依赖关系会完全丢失，依赖图就只剩空壳。
                        resolved = _resolve_relative(module, node.level, node.module or "")
                        if resolved:
                            targets.append(resolved)
                    elif node.module:
                        targets.append(node.module)

            local = [t for t in targets if t.split(".")[0] in self._local_roots()]
            if local:
                graph[module] = sorted(set(local))
        return graph

    def _local_roots(self) -> set[str]:
        roots = set()
        for path in self.root.iterdir():
            if path.is_dir() and not path.name.startswith(".") and path.name not in SKIP_DIRS:
                if (path / "__init__.py").exists():
                    roots.add(path.name)
            elif path.suffix == ".py":
                roots.add(path.stem)
        return roots

    # ------------------------------------------------------------------ #
    def grep(
        self,
        pattern: str,
        glob: str = "",
        max_hits: int = 60,
        context_lines: int = 0,
    ) -> str:
        """正则搜索源码，返回 `文件:行号: 内容` 形式的结果。"""
        if not pattern:
            return "请提供要搜索的模式。"

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"正则表达式无效：{exc}"

        suffixes = {f".{glob.lstrip('*.')}"} if glob.startswith("*.") else None
        hits: list[str] = []

        for file_path in iter_files(self.root, suffixes=suffixes, max_files=self.max_files):
            if file_path.suffix.lower() not in {
                ".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cfg", ".ini", ".ts", ".js", ".vue"
            }:
                continue
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue

            relative = str(file_path.relative_to(self.root))
            for number, line in enumerate(lines, start=1):
                if not regex.search(line):
                    continue
                hits.append(f"{relative}:{number}: {line.strip()[:160]}")
                if context_lines and number < len(lines):
                    hits.append(f"{relative}:{number + 1}:   {lines[number].strip()[:160]}")
                if len(hits) >= max_hits:
                    return f"命中过多，仅展示前 {max_hits} 条：\n" + "\n".join(hits)

        if not hits:
            return f"没有匹配 {pattern!r} 的内容。"
        return f"共 {len(hits)} 条命中：\n" + "\n".join(hits)

    # ------------------------------------------------------------------ #
    def read(self, path: str, max_chars: int = 6000, with_line_numbers: bool = True) -> str:
        target = self._safe(path)
        if not target.is_file():
            return f"不是文件或不存在：{path}"
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"读取失败：{exc}"

        if with_line_numbers:
            lines = text.splitlines()
            numbered = "\n".join(f"{i:>4}| {line}" for i, line in enumerate(lines, start=1))
        else:
            numbered = text

        if len(numbered) > max_chars:
            return numbered[:max_chars] + "\n…（内容已截断）"
        return numbered

    def outline(self, path: str) -> str:
        target = self._safe(path)
        if not target.is_file():
            return f"不是文件或不存在：{path}"
        if target.suffix != ".py":
            return self.read(path, max_chars=2000)

        relative = str(target.relative_to(self.root))
        symbols = self._parse_file(target)
        if not symbols:
            return f"{relative} 中没有可解析的定义。"
        return f"{relative} 的结构：\n" + "\n".join(symbol.to_line() for symbol in symbols)

    # ------------------------------------------------------------------ #
    def describe(self) -> dict:
        return {"root": str(self.root), "files": len(list(iter_files(self.root, max_files=self.max_files)))}


def _first_line(text: str | None) -> str:
    if not text:
        return ""
    return text.strip().splitlines()[0][:60]


def _resolve_relative(current_module: str, level: int, module: str) -> str:
    """把相对导入还原为绝对模块名。

    `level=1` 表示当前包，`level=2` 表示上一级包。
    例：模块 `demo_pkg.service`、`from .storage import X` → `demo_pkg.storage`。
    """
    parts = current_module.split(".")
    # 去掉文件名本身，得到所在包
    package = parts[:-1]
    if level > 1:
        package = package[: -(level - 1)] if len(package) >= level - 1 else []
    base = ".".join(package)
    if module:
        return f"{base}.{module}" if base else module
    return base


def find_repo_root(start: str | Path | None = None) -> Path | None:
    """向上寻找包含 `README.md` 或 `.git` 的目录，作为仓库根。"""
    current = Path(start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return current


def iter_python_files(root: str | Path) -> Iterable[Path]:
    return iter_files(root, suffixes={".py"})
