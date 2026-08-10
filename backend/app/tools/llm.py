"""
AgentsLLM — DeepSeek LLM 客户端

基于 DeepSeek API（兼容 OpenAI SDK），默认使用流式响应。
环境变量配置:
- LLM_MODEL_ID: 模型名称，默认 deepseek-chat
- LLM_API_KEY / DEEPSEEK_API_KEY: API 密钥
- LLM_BASE_URL: 服务地址，默认 https://api.deepseek.com
- LLM_TIMEOUT: 超时秒数，默认 60
"""

import os
import re
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# DeepSeek 默认配置
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _sanitize_base_url(url: str) -> str:
    """清理 base_url，移除常见的错误后缀，确保 SDK 能正确拼接路径"""
    if not url:
        return url
    url = re.sub(r'/chat/completions/?$', '', url)
    url = re.sub(r'/completions/?$', '', url)
    url = re.sub(r'/v1/?$', '', url)
    return url.rstrip('/')


class AgentsLLM:
    """
    DeepSeek LLM 客户端，兼容 OpenAI SDK 接口，默认使用流式响应。

    使用示例:
        >>> llm = AgentsLLM()
        >>> messages = [{"role": "user", "content": "你好"}]
        >>> reply = llm.think(messages)
    """

    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，未提供时从环境变量加载。

        Args:
            model: 模型名称，默认 deepseek-chat
            apiKey: API 密钥，默认从 LLM_API_KEY 或 DEEPSEEK_API_KEY 读取
            baseUrl: 服务地址，默认 https://api.deepseek.com
            timeout: 超时秒数，默认 60
        """
        self.model = model or os.getenv("LLM_MODEL_ID") or DEFAULT_MODEL
        apiKey = apiKey or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL") or DEFAULT_BASE_URL
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        # 清理 base_url 中的常见错误
        baseUrl = _sanitize_base_url(baseUrl)

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError(
                "模型ID、API密钥和服务地址必须被提供或在.env文件中定义。\n"
                "DeepSeek 默认配置:\n"
                "  LLM_BASE_URL=https://api.deepseek.com\n"
                "  LLM_MODEL_ID=deepseek-chat"
            )

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用 DeepSeek 模型进行思考，返回流式响应。

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            temperature: 温度参数

        Returns:
            str: 模型完整响应文本
        """
        print(f"[LLM] Calling DeepSeek model: {self.model}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            print("[LLM] Response OK:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()
            return "".join(collected_content)

        except Exception as e:
            print(f"[LLM] DeepSeek API Error: {e}")
            return None
