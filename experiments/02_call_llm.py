
import os

from dotenv import load_dotenv
from openai import OpenAI


# 1. 读取项目根目录中的 .env 配置
load_dotenv()

# cspell:ignore DEEPSEEK_API_KEY
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("没有找到 DEEPSEEK_API_KEY，请检查 .env 文件")


# 2. 创建 API 客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# 3. 准备发给模型的消息
messages = [
    {
        "role": "system",
        "content": "你是一名善于向初学者解释技术的 AI 助手。"
    },
    {
        "role": "user",
        "content": "请用简单的语言解释什么是 AI Agent。"
    }
]


# 4. 发送请求，等待模型返回
response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages
)


# 5. 提取并打印模型回答
answer = response.choices[0].message.content

print("模型回答：")
print(answer)
