"""命令行入口。

    python -m contextforge status
    python -m contextforge ingest examples/sample_docs
    python -m contextforge ask "为什么压缩必须以轮次为单位？"
    python -m contextforge chat
    python -m contextforge search "RRF 融合"
    python -m contextforge bench            # 跑策略对比实验并生成报告
    python -m contextforge serve            # 启动 Web 界面
    python -m contextforge demo             # 一键体验：导入示例 + 跑几个问题
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PROJECT_ROOT, get_settings
from .evaluation import (
    default_cases,
    render_full_report,
    run_agent_suite,
    run_context_experiment,
    save_report,
)
from .runtime import AgentRuntime


def _runtime(repo_root: str | None = None) -> AgentRuntime:
    settings = get_settings(refresh=True)
    root = repo_root or str(PROJECT_ROOT)
    return AgentRuntime(settings=settings, repo_root=root)


def _print_header(runtime: AgentRuntime) -> None:
    info = runtime.llm.describe()
    mode = "在线模型" if info.get("online") else "离线确定性模型"
    print(f"模型：{info.get('name')}（{mode}）")
    print(f"工具：{', '.join(runtime.tools.names())}")
    print(f"知识库：{runtime.knowledge.stats()['documents']} 篇文档 / {runtime.knowledge.stats()['chunks']} 个切片")


# --------------------------------------------------------------------------- #
def cmd_status(args: argparse.Namespace) -> int:
    runtime = _runtime(args.repo)
    print(json.dumps(runtime.status(), ensure_ascii=False, indent=2))
    runtime.close()
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    runtime = _runtime(args.repo)
    path = Path(args.path)
    if not path.exists():
        print(f"路径不存在：{path}", file=sys.stderr)
        return 1

    if path.is_dir():
        result = runtime.ingest_directory(path)
        print(f"导入 {result['added']} 个文件")
        for item in result["skipped"][:10]:
            print(f"  跳过：{item}")
    else:
        document = runtime.ingest_file(path)
        if document is None:
            print("无法解析该文件", file=sys.stderr)
            return 1
        print(f"导入：{document['title']}（{document['chunk_count']} 个切片）")

    print(f"知识库现有 {runtime.knowledge.stats()['chunks']} 个切片")
    runtime.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    runtime = _runtime(args.repo)
    hits = runtime.search(args.query, top_k=args.top_k, mode=args.mode)
    if not hits:
        print("没有检索到相关内容。")
    for rank, item in enumerate(hits, start=1):
        chunk = item["chunk"]
        print(f"[{rank}] {chunk['title']}#{chunk['index']}  score={item['score']:.3f}  "
              f"(向量 {item['dense_score']} / BM25 {item['sparse_score']})")
        print(f"    {' '.join(chunk['text'].split())[:160]}…")
    trace = runtime.knowledge.retriever.last_trace
    if trace:
        print(f"\n检索轨迹：向量命中 {trace.dense_hits}，BM25 命中 {trace.sparse_hits}，"
              f"融合 {trace.fused_hits}，最终 {trace.reranked_hits}")
    runtime.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    runtime = _runtime(args.repo)
    if args.verbose:
        _print_header(runtime)
        print()

    result = runtime.ask(args.question, strategy=args.strategy, retrieval_mode=args.mode)

    print(result.answer or "（没有回答）")
    print()
    if result.citations:
        print("引用来源：")
        for citation in result.citations:
            print(f"  [{citation.label}] score={citation.score:.3f}")
    if result.tool_executions:
        print("工具调用：")
        for execution in result.tool_executions:
            flag = "OK " if execution.ok else "ERR"
            print(f"  {flag} {execution.tool_name}（{execution.latency_ms:.0f}ms）")

    if result.context_report:
        report = result.context_report
        print(
            f"\n上下文：{report.strategy} 策略，"
            f"{report.tokens_before} → {report.tokens_after} tokens"
            f"（压缩率 {report.compression_ratio:.2%}），工具链完整：{report.tool_integrity_ok}"
        )
    print(f"用量：{result.usage.as_dict}，迭代 {result.iterations} 轮，"
          f"耗时 {result.latency_ms:.0f}ms，状态 {result.status}")

    if args.trace:
        print("\n执行事件：")
        for event in result.events:
            if event["type"] == "node_end":
                print(f"  · {event['node']}")
    runtime.close()
    return 0 if result.status in ("completed", "interrupted") else 1


def cmd_chat(args: argparse.Namespace) -> int:
    runtime = _runtime(args.repo)
    _print_header(runtime)
    print("\n输入 exit 退出；输入 :trace 查看上一条的执行事件。\n")

    last = None
    while True:
        try:
            user_input = input("你 › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input == ":trace" and last is not None:
            for event in last.events:
                print(f"  {event['type']:<12} {event['node']}")
            continue

        result = runtime.ask(user_input, strategy=args.strategy, retrieval_mode=args.mode)
        last = result
        print(f"\nCF › {result.answer}\n")
        if result.citations:
            print("     引用：" + "，".join(citation.label for citation in result.citations))
        if result.context_report:
            print(
                f"     上下文 {result.context_report.tokens_after} tokens"
                f"（压缩率 {result.context_report.compression_ratio:.0%}）"
                f"｜{result.iterations} 轮｜{result.latency_ms:.0f}ms\n"
            )

    runtime.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    runtime = _runtime(args.repo)

    # 评测用例是针对示例文档设计的，所以先保证语料在场 ——
    # 否则一个空知识库跑出来的检索指标全是 0，报告会误导人。
    sample_dir = PROJECT_ROOT / "examples" / "sample_docs"
    if sample_dir.is_dir() and args.ensure_corpus:
        added = runtime.ingest_directory(sample_dir)["added"]
        if added:
            print(f"已补充导入示例文档 {added} 篇")

    _print_header(runtime)
    print("\n正在运行评测…（离线模型，无需网络）\n")

    context_report = run_context_experiment(
        runtime,
        strategies=["none", "sliding", "summary", "hybrid", "llm-summary"],
        turns=args.turns,
    )
    agent_report = run_agent_suite(
        runtime,
        cases=default_cases(),
        retrieval_modes=["dense", "sparse", "hybrid"],
    )

    stats = runtime.knowledge.stats()
    text = render_full_report(
        agent_report,
        context_report,
        metadata={
            "语料": f"示例文档 {stats['documents']} 篇 / {stats['chunks']} 个切片",
            "检索模式": "dense / sparse / hybrid",
            "压缩历史": f"{args.turns} 轮",
            "模型": "离线确定性模型（无需 API Key）",
        },
    )
    print(text)

    if args.output:
        print(f"\n已保存：{save_report(text, args.output)}")

    runtime.close()
    return 0


def cmd_workflow(args: argparse.Namespace) -> int:
    """打印 Agent 工作流图。

    三种格式各有用途：`mermaid` 贴文档、`json` 给前端画图、`text` 用于终端速览。
    """
    from .graph.workflow import NODE_ORDER, workflow_diagram

    settings = get_settings(refresh=True)
    if args.format == "mermaid":
        print(workflow_diagram(settings))
        return 0

    runtime = _runtime(args.repo)
    describe = runtime.graph.describe()
    if args.format == "json":
        print(json.dumps(describe, ensure_ascii=False, indent=2))
    else:
        print(f"节点（{len(describe['nodes'])} 个）：")
        for name in describe["nodes"]:
            order = NODE_ORDER.index(name) + 1 if name in NODE_ORDER else "—"
            print(f"  {order:>2}. {name}")
        print("\n固定边：")
        for source, target in describe["edges"].items():
            print(f"  {source} → {target}")
        print("\n条件边：")
        for source, branches in describe["conditional"].items():
            for decision, target in branches.items():
                print(f"  {source} --[{decision}]--> {target}")
    runtime.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .api.devserver import serve

    serve(host=args.host, port=args.port, quiet=not args.verbose)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """一键体验：导入示例文档 → 跑几个不同类型的问题。"""
    runtime = _runtime(args.repo)

    sample_dir = PROJECT_ROOT / "examples" / "sample_docs"
    if sample_dir.is_dir():
        result = runtime.ingest_directory(sample_dir)
        print(f"已导入示例文档 {result['added']} 篇")

    repo_dir = PROJECT_ROOT / "examples" / "sample_repo"
    if repo_dir.is_dir():
        runtime.repo = __import__("contextforge.tools", fromlist=["RepoAnalyzer"]).RepoAnalyzer(repo_dir)
        from .tools import build_registry

        runtime.tools = build_registry(
            knowledge=runtime.knowledge,
            memory=runtime.memory,
            repo=runtime.repo,
            allow_tests=False,
        )
        print(f"已挂载示例代码仓库：{repo_dir}")

    _print_header(runtime)

    questions = [
        "上下文压缩为什么要以轮次为单位？",
        "RRF 融合相比加权融合有什么优势？",
        "计算 (128 + 72) * 3 等于多少？",
        "这个示例项目的目录结构是什么样的？",
    ]

    for question in questions:
        print("\n" + "─" * 72)
        print(f"问：{question}")
        result = runtime.ask(question)
        print(f"答：{(result.answer or '')[:400]}")
        print(f"    节点：{' → '.join(e['node'] for e in result.events if e['type'] == 'node_end')}")
        if result.context_report:
            print(f"    上下文 {result.context_report.tokens_after} tokens，"
                  f"压缩率 {result.context_report.compression_ratio:.0%}")

    print("\n" + "─" * 72)
    print("完成。运行 `python -m contextforge serve` 可打开 Web 界面。")
    runtime.close()
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextforge",
        description="ContextForge · Knowledge · Agent · Context Engineering",
    )
    parser.add_argument("--repo", help="作为代码理解目标的仓库根目录", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="查看系统状态").set_defaults(func=cmd_status)

    p_workflow = sub.add_parser("workflow", help="打印 Agent 工作流图")
    p_workflow.add_argument(
        "--format",
        default="mermaid",
        choices=["mermaid", "json", "text"],
        help="mermaid 可直接贴进 README / 支持 Mermaid 的 Markdown",
    )
    p_workflow.set_defaults(func=cmd_workflow)

    p_ingest = sub.add_parser("ingest", help="导入文档或目录")
    p_ingest.add_argument("path")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="检索知识库")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="提一个问题")
    p_ask.add_argument("question")
    p_ask.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    p_ask.add_argument("--strategy", default="", help="压缩策略覆盖")
    p_ask.add_argument("--trace", action="store_true", help="打印执行事件")
    p_ask.add_argument("--verbose", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "sparse"])
    p_chat.add_argument("--strategy", default="")
    p_chat.set_defaults(func=cmd_chat)

    p_bench = sub.add_parser("bench", help="运行策略对比实验")
    p_bench.add_argument(
        "--turns",
        type=int,
        default=30,
        help="压缩实验的历史轮数；轮数太少时历史不超过预算，策略差异看不出来",
    )
    p_bench.add_argument("--output", default="data/report.md")
    p_bench.add_argument(
        "--ensure-corpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="评测前先确保示例文档已导入（默认开启，保证报告可复现）",
    )
    p_bench.set_defaults(func=cmd_bench)

    p_serve = sub.add_parser("serve", help="启动 Web 界面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--verbose", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    sub.add_parser("demo", help="一键体验示例").set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
