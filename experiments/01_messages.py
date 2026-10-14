
# 1. 创建对话消息列表
messages = [
    {
        "role": "system",
        "content": "你是一个专业的 AI Agent 学习助手。"
    },
    {
        "role": "user",
        "content": "什么是 Agent？"
    }
]


# 2. 定义一个打印消息的函数
def print_messages(messages):
    for index, message in enumerate(messages):
        print(f"第 {index} 条消息")
        print(f"角色：{message['role']}")
        print(f"内容：{message['content']}")
        print("-" * 30)


# 3. 调用函数
print_messages(messages)
