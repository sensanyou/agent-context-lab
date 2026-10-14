
import os
from dotenv import load_dotenv

# cspell:ignore DEEPSEEK

# 读取项目根目录中的 .env 文件
load_dotenv()

# 从环境变量中读取 DeepSeek API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com"
)

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-flash"
)

# 检查密钥是否存在
if not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "没有找到 DEEPSEEK_API_KEY，请检查项目根目录中的 .env 文件。"
    )
