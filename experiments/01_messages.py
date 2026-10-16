"""实验 01 · 消息、轮次与工具链完整性

## 这个实验要回答什么

上下文压缩的最小单位应该是「消息」还是「轮次」？

先看一条最容易被忽略的规则：模型 API 要求每一个
`assistant(tool_calls)` 都必须有**成对且不重复**的 `tool` 结果紧随其后。
一旦把这两条消息拆开（比如按条数裁剪、只保留最后 N 条消息），
请求会被直接拒绝 —— 这不是「效果差一点」，而是**报错**。

所以本实验做两件事：

1. 把一个真实的消息序列切成「轮次」，看一条工具链如何构成一个原子单元；
2. 故意把工具消息删掉，用校验器把这种非法序列抓出来并修复。
"""

from _lab import conclusion, kv_table, note, step, title

from contextforge.context.turns import flatten_turns, split_messages
from contextforge.context.validator import check_tool_calls, repair_tool_calls
from contextforge.types import assistant, system, tool_result, user


def build_conversation() -> list[dict]:
    """构造一段包含工具调用的真实对话。"""
    return [
        system("你是 ContextForge 的代码助手。"),
        user("帮我看看 demo_pkg 里有哪些类？"),
        assistant(
            None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "repo_symbols",
                        "arguments": '{"path": "demo_pkg"}',
                    },
                }
            ],
        ),
        tool_result(
            "call_1",
            "demo_pkg/models.py: Document(类), Chunk(类)",
            name="repo_symbols",
        ),
        assistant("demo_pkg 里定义了 Document 和 Chunk 两个类。"),
        user("那 storage.py 呢？"),
        assistant("storage.py 提供向量存储的读写与持久化。"),
    ]


def main() -> None:
    title("实验 01 · 消息、轮次与工具链完整性")

    messages = build_conversation()

    step("第 1 步：把消息序列切成轮次")
    system_messages, turns = split_messages(messages)
    note(f"系统消息 {len(system_messages)} 条（单独拎出来，永远不参与压缩）")
    note(f"对话轮次 {len(turns)} 轮")
    for index, turn in enumerate(turns, start=1):
        roles = " → ".join(str(message.get("role")) for message in turn)
        note(f"轮次 {index}：{roles}（{len(turn)} 条消息）")

    step("第 2 步：为什么「轮次」才是压缩的最小单位")
    tool_turn = next(turn for turn in turns if any(m.get("tool_calls") for m in turn))
    note("第 1 轮里，assistant(tool_calls) 与 tool 结果被绑定在同一个轮次：")
    for message in tool_turn:
        preview = message.get("content") or "(发起工具调用，无正文)"
        note(f"  role={str(message.get('role')):<9} {str(preview)[:44]}")
    note("按『消息条数』裁剪时，最先被丢掉的往往就是 tool 结果 —— 那正是完整性所在。")

    step("第 3 步：故意破坏工具链，用校验器抓出来")
    broken = [dict(message) for message in messages]
    broken.pop(3)  # 删掉 tool 结果，模拟「裁剪时切断了工具链」
    report = check_tool_calls(broken)
    note(f"完整性校验：{'通过' if report.ok else '不通过'}")
    note(f"发现 {report.call_count} 次调用，其中 {report.resolved_count} 次有结果")
    for problem in report.problems:
        note(f"问题：{problem}")

    step("第 4 步：修复 —— 丢弃整条残缺轮次，而不是留下半条")
    repaired, notes = repair_tool_calls(broken)
    after = check_tool_calls(repaired)
    note(f"修复后消息数 {len(broken)} → {len(repaired)}")
    note(f"修复后校验：{'通过' if after.ok else '不通过'}")
    for item in notes:
        note(f"修复动作：{item}")

    step("第 5 步：轮次还能被无损地摊平还原")
    flattened = flatten_turns(turns)
    note(f"原始消息 {len(messages)} 条，去掉系统消息后摊平 {len(flattened)} 条")
    note(f"摊平后顺序与原始一致：{flattened == messages[1:]}")

    print()
    print(
        kv_table(
            ["环节", "结果"],
            [
                ["轮次切分", f"{len(turns)} 轮"],
                ["破坏后校验", "不通过" if not report.ok else "通过"],
                ["修复后校验", "通过" if after.ok else "不通过"],
            ],
        )
    )

    conclusion(
        "压缩必须以**轮次**为最小单位。按消息条数裁剪会把 "
        "assistant(tool_calls) 与 tool 结果拆开，请求会被 API 直接拒绝；"
        "按轮次裁剪只是「少了一段历史」，模型仍然能正常工作。"
    )


if __name__ == "__main__":
    main()
