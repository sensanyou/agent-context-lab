"""一次跑完所有实验。

    python experiments/run_all.py            # 跑全部
    python experiments/run_all.py 03 05      # 只跑指定编号

每个实验都是独立进程（`python experiments/XX.py`），这样任何一个实验崩了
都不会影响其他实验，也方便单独调试。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: 编号 → 主题，顺序即阅读顺序
EXPERIMENTS = [
    ("01", "消息、轮次与工具链完整性"),
    ("02", "模型适配层：同一个接口，两种实现"),
    ("03", "Token 计数与分区预算"),
    ("04", "提示注入：把外部内容当数据"),
    ("05", "压缩策略对比：省 token 的代价"),
    ("06", "混合检索：向量 + BM25 + RRF + 重排"),
    ("07", "图编排：工具调用要重新装配上下文"),
    ("08", "RAG ≠ 记忆：四种机制四个问题"),
    ("09", "评测：把「效果更好」变成一张表"),
]


def find_script(number: str) -> Path | None:
    matches = sorted(HERE.glob(f"{number}_*.py"))
    return matches[0] if matches else None


def main(argv: list[str]) -> int:
    wanted = [item for item in argv if item.strip()]
    selected = [
        (number, title)
        for number, title in EXPERIMENTS
        if not wanted or number in wanted
    ]
    if not selected:
        print(f"没有匹配的实验。可选编号：{[n for n, _ in EXPERIMENTS]}")
        return 1

    print(f"即将运行 {len(selected)} 个实验（全部离线，不需要 API Key）")
    print()

    failures: list[tuple[str, str]] = []
    started = time.perf_counter()

    for number, title in selected:
        script = find_script(number)
        if script is None:
            failures.append((number, "找不到脚本文件"))
            continue

        print("=" * 78)
        print(f"[{number}] {title}")
        print("=" * 78)
        begin = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(HERE.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        elapsed = time.perf_counter() - begin

        if completed.returncode == 0:
            # 只回显笔记与结论，完整输出太重
            for line in (completed.stdout or "").splitlines():
                if line.startswith("结论：") or line.startswith("   结论："):
                    print(line)
            print(f"  ✓ 通过（{elapsed:.2f}s）")
        else:
            print(completed.stdout or "")
            print(completed.stderr or "", file=sys.stderr)
            failures.append((number, f"退出码 {completed.returncode}"))
            print(f"  ✗ 失败（{elapsed:.2f}s）")

    total = time.perf_counter() - started
    print()
    print("=" * 78)
    if failures:
        print(f"共 {len(selected)} 个实验，{len(failures)} 个失败：")
        for number, reason in failures:
            print(f"  [{number}] {reason}")
        return 1
    print(f"全部 {len(selected)} 个实验通过，总耗时 {total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
