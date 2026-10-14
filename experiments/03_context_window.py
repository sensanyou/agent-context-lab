

messages = [
    {
        "role": "system",
        "content": "你是一个 AI Agent 学习助手。",
    }
]


def add_message(role, content):
    messages.append({
        "role": role,
        "content": content,
    })




def limit_context(max_chars=100):
    """
    按字符数近似控制上下文长度。
    保留 system 消息，并尽量保留最近的对话历史。
    """
    system_messages = [
        message
        for message in messages
        if message["role"] == "system"
    ]

    history_messages = [
        message
        for message in messages
        if message["role"] != "system"
    ]

    # 计算系统提示词占用的字符数
    system_chars = sum(
        len(message["content"])
        for message in system_messages
    )

    # 留给对话历史的字符预算
    remaining = max(0, max_chars - system_chars)

    kept_history = []

    # 从最新的消息开始，向前保留
    for message in reversed(history_messages):
        message_chars = len(message["content"])

        if message_chars <= remaining:
            kept_history.append(message)
            remaining -= message_chars

    # 恢复为原来的时间顺序
    kept_history.reverse()

    messages.clear()
    messages.extend(system_messages)
    messages.extend(kept_history)


def show_messages():
    for index, message in enumerate(messages):
        print(f"第 {index} 条消息")
        print(f"角色：{message['role']}")
        print(f"内容：{message['content']}")
        print("-" * 30)


add_message("user", "什么是 Agent？")
add_message("assistant", "Agent 可以根据目标采取行动。")
add_message("user", "什么是工具调用？")
add_message("assistant", "工具调用让模型请求程序执行指定工具。")
add_message("user", "什么是上下文窗口？")
add_message("assistant", "它决定模型一次能够处理多少上下文。")

print("压缩前的消息数：", len(messages))

limit_context(max_chars=30)

print("压缩后的消息数：", len(messages))


show_messages()
