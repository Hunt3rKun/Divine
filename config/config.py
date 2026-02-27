import os
from dotenv import load_dotenv

# 从.env文件加载环境变量
load_dotenv()

DEFAULT_PRICING ={
# OpenAI
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    # Zhipu GLM
    "glm-4.7": {"input": 6.94, "output": 6.94},
    "glm-4.6": {"input": 2.78, "output": 2.78},
    "glm-4.5": {"input": 6.94, "output": 6.94},
}


LLM_REQUEST_TIMEOUT = 120.0  # 默认请求超时时间（秒）
