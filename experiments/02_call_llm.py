"""实验 02 · 模型适配层：同一个接口，两种实现

## 这个实验要回答什么

如果把「调用大模型」这件事散落在业务代码里，会立刻出现两个问题：

1. 没有 API Key 时，整个项目连启动都做不到，更别说跑测试和做实验；
2. 想换模型厂商时，要改的是业务逻辑，而不是配置。

所以把模型收进一个**适配层**（`BaseLLM` 协议）：
上层只说「给我一次 chat 的结果」，至于这句话是发给了远端 API、
还是由本地的确定性规则引擎回答，上层不需要知道。

关键差别在于：如果 `config.py` 在缺少 API Key 时**直接抛异常**，
那么「读代码」这件事都变得需要凭证；这里的降级是**静默且可观测**的。
"""

from _lab import conclusion, kv_table, note, step, title

from contextforge.config import Settings
from contextforge.llm.base import BaseLLM
from contextforge.llm.factory import build_llm
from contextforge.llm.mock import MockLLM
from contextforge.llm.openai_compat import OpenAICompatibleLLM
from contextforge.types import system, user


def main() -> None:
    title("实验 02 · 模型适配层：同一个接口，两种实现")

    step("第 1 步：没有 API Key 时会发生什么")
    settings = Settings().with_overrides()
    settings.model.api_key = ""
    llm = build_llm(settings)
    note(f"配置里 api_key = ''（模拟用户删掉 Key 的情况）")
    note(f"工厂返回的实现：{type(llm).__name__}")
    note(f"online = {llm.online}")
    note("关键：这里没有抛异常 —— 项目依然可以启动、可以跑测试。")

    step("第 2 步：两种实现共享同一个接口")
    adapters = [
        ("离线确定性模型", MockLLM()),
        ("OpenAI 兼容适配器（未实例化）", None),
    ]
    note(f"BaseLLM 规定的接口：chat(messages, tools, temperature)")
    note(f"MockLLM 是 BaseLLM 的子类：{issubclass(MockLLM, BaseLLM)}")
    note(f"OpenAICompatibleLLM 也是：{issubclass(OpenAICompatibleLLM, BaseLLM)}")
    note("所以上层代码只需要依赖 BaseLLM，不需要 import 任何具体实现。")

    step("第 3 步：离线模型是「确定性」的，因此可以被测试")
    question = [system("你是 ContextForge 助手。"), user("现在几点了？")]
    first = llm.chat(list(question), tools=None)
    second = llm.chat(list(question), tools=None)
    preview = lambda response: " ".join((response.content or "").split())[:56]  # noqa: E731
    note(f"第 1 次回答：{preview(first)}")
    note(f"第 2 次回答：{preview(second)}")
    note(f"两次回答完全一致：{first.content == second.content}")
    note("确定性带来一个实际好处：实验结论可复现，回归测试不会因为模型随机性而抖动。")

    step("第 4 步：离线模型仍然会「决定调用工具」")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_local_time",
                "description": "获取当前本地日期和时间。",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    decided = llm.chat(list(question), tools=tools)
    calls = [call.name for call in decided.tool_calls]
    note(f"模型给出的工具调用：{calls}")
    note(f"这是规则匹配的结果，不经网络 —— 但走的是**完全相同的消息协议**。")

    step("第 5 步：在线适配器长什么样")
    note("OpenAICompatibleLLM 负责的是：重试、超时、把 SDK 返回值翻译成内部对象。")
    note("对外暴露的仍是 LLMResponse(content, tool_calls, usage)，上层无感知。")
    note("切到真实模型只需要在配置里填 api_key/base_url/model，不需要改业务代码。")

    print()
    print(
        kv_table(
            ["场景", "实现", "是否联网", "是否抛异常"],
            [
                ["无 API Key", type(llm).__name__, "否", "否"],
                ["有 API Key", "OpenAICompatibleLLM", "是", "否（失败转重试/降级）"],
            ],
        )
    )

    conclusion(
        "把模型收进适配层，换来的不只是「可替换」：更重要的是在缺少凭证时"
        "**降级而不是崩溃**，让整个项目在没有 API Key 的情况下依然能启动、"
        "能测试、能做策略对比实验。"
    )


if __name__ == "__main__":
    main()
