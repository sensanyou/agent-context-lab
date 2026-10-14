#agent核心循环
# cspell:ignore DEEPSEEK deepseek

import json

from openai import OpenAI,APIError
from agent.context import trim_history
from agent.compression import compress_history
from agent.context import (
    trim_history,
    estimate_context_chars,
    validate_tool_calls,
)
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)
from agent.tools import TOOLS, execute_tool



class Agent:
    CONTEXT_CHAR_BUDGET = 8000
    def __init__(self):
        # 创建大模型客户端
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

        # 保存当前对话上下文
        self.messages = [
            {
                "role": "system",
                "content": (
                    "你是一个耐心的 AI Agent 学习助手。"
                    "当用户询问当前时间时，请优先调用获取当前本地时间的工具。"
                    "工具返回结果后，再用中文向用户解释。"
                ),
            }
        ]
    def fit_context_budget(self, messages: list[dict]) -> list[dict]:
        """尽量将上下文控制在字符预算内。"""
        estimated_chars = estimate_context_chars(messages)

        if estimated_chars <= self.CONTEXT_CHAR_BUDGET:
          return messages

        print("【上下文预算】超过预算，尝试压缩历史。")

        # 保留当前用户问题，避免压缩时把它当成旧历史
        current_user_message = messages[-1]
        history = messages[:-1]

        # 逐步减少保留的近期对话轮数
        for keep_turns in (3, 2, 1, 0):
            compressed_history = compress_history(
                history,
                keep_recent_turns=keep_turns,
                max_summary_chars=600,
           )

            candidate = compressed_history + [current_user_message]
         # 检查压缩后的工具调用链是否完整
            if not validate_tool_calls(candidate):
               print(
                f"【工具链检查】保留近期 {keep_turns} 轮时，"
                "工具调用结构不完整，跳过。"
            )
               continue

             #检查字符预算
            estimated_chars = estimate_context_chars(candidate)

            print(
            f"【上下文预算】保留近期 {keep_turns} 轮，"
            f"估算字符数：{estimated_chars}"
        )

            if estimated_chars <= self.CONTEXT_CHAR_BUDGET:
               return candidate

        raise ValueError(
        "无法在预算内找到工具调用结构完整的上下文，"
        "请缩短当前问题或调整预算。"
    )

         
    
    def run(self, user_input: str) -> str:
        # 1. 复制已有历史，在临时列表中执行本轮对话
        working_messages = self.messages.copy()
        # 2. 保存压缩前的历史，作为回退方案
        original_messages = working_messages.copy()
        # 3. 历史较长时，压缩较早的对话
        compressed_messages = compress_history(
              working_messages,
              keep_recent_turns=4,
              max_summary_chars=1200,
              )
        # 4. 检查压缩后的工具调用链
        if validate_tool_calls(compressed_messages):
           working_messages = compressed_messages
           print("【工具链检查】压缩后的工具调用结构完整。")
        else:
           working_messages = original_messages
           print("【工具链检查】压缩结果不完整，保留原始历史。")
        # 5.添加本轮用户消息
        working_messages.append({
            "role": "user",
            "content": user_input,
        })
         # 6.检查上下文字符数
        working_messages = self.fit_context_budget(working_messages)

        estimated_chars = estimate_context_chars(working_messages)
        print(f"【上下文预算】最终字符数：{estimated_chars}")

        print("\n========== 本次发送给模型的消息 ==========")

        for index, message in enumerate(working_messages):
            role = message.get("role")
            content = message.get("content") or ""

            print(f"[{index}] role={role}")
            print(f"内容：{content[:150]}")

            if message.get("tool_calls"):
                print("包含工具调用信息")

        print("========== 消息检查结束 ==========\n")

        # 2. 开始 Agent 循环

        max_iterations = 5

        estimated_chars = estimate_context_chars(working_messages)
        print(f"【上下文检查】消息字符数估算：{estimated_chars}")
        
        for _ in range(max_iterations):
            # 请求大模型
            try:
                response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=working_messages,
                    tools=TOOLS,
                )
            except APIError as exc:
                print(f"【API 错误】请求模型失败：{exc}")
                return "抱歉，当前无法正常请求大模型，请稍后重试。"

            # 获取模型回复
            message = response.choices[0].message

            print(
                "【调试】模型是否请求调用工具：",
                bool(message.tool_calls),
            )

            # 整理模型回复
            assistant_message = {
                "role": "assistant",
                "content": message.content,
            }

            # 如果模型请求调用工具，保存调用信息
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]

            # 把模型回复加入临时上下文
            working_messages.append(assistant_message)

            # 3. 模型不再请求工具，本轮结束
            if not message.tool_calls:
                # 提交本轮完整历史
                self.messages = working_messages
                return message.content or ""

            # 4. 执行模型请求的工具
            for call in message.tool_calls:
                tool_name = call.function.name
                raw_arguments = call.function.arguments

                print(f"【调试】模型请求调用工具：{tool_name}")
                print(f"【调试】工具参数：{raw_arguments}")

                try:
                    # 解析模型生成的 JSON 参数
                    arguments = json.loads(raw_arguments)

                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是 JSON 对象")

                    # 执行工具
                    tool_result = execute_tool(tool_name, arguments)

                    # 输出执行日志
                    if tool_result.startswith(("工具执行失败：", "错误：")):
                        print(f"【工具错误】{tool_name}：{tool_result}")
                    else:
                        print(f"【工具成功】{tool_name}：{tool_result}")

                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    tool_result = f"工具参数解析失败：{exc}"
                    print(f"【工具错误】{tool_name}：{tool_result}")

                # 5. 把工具结果加入临时上下文
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": tool_result,
                })

        # 达到最大循环次数，不提交本轮临时历史
        return "Agent 已达到本轮最大执行次数，请尝试简化问题。"

    

    
