"""
AgentsLLM — LLM 客户端

基于 DeepSeek API（兼容 OpenAI SDK），默认使用流式响应。
环境变量配置:
- LLM_MODEL_ID: 模型名称，默认 deepseek-chat
- LLM_API_KEY / DEEPSEEK_API_KEY: API 密钥
- LLM_BASE_URL: 服务地址，默认 https://api.deepseek.com
- LLM_TIMEOUT: 超时秒数，默认 60
"""

import os
import re
from typing import Literal, Optional, Iterator
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件（不覆盖已有的系统环境变量）
load_dotenv()

try:
    from .exceptions import AgentsException
except ImportError:
    from backend.app.AgentRuntime.core.exceptions import AgentsException

# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai", "deepseek", "qwen", "modelscope",
    "kimi", "zhipu", "ollama", "vllm", "local", "auto"
]


def _sanitize_base_url(url: str) -> str:
    """清理 base_url，移除常见的错误后缀，确保 SDK 能正确拼接路径"""
    if not url:
        return url
    # 去掉末尾的 /chat/completions 或 /completions
    url = re.sub(r'/chat/completions/?$', '', url)
    url = re.sub(r'/completions/?$', '', url)
    # 去掉末尾多余的 /v1（SDK 会自动添加）
    url = re.sub(r'/v1/?$', '', url)
    # 去掉末尾斜杠
    url = url.rstrip('/')
    return url


class AgentsLLM:
    """
    LLM客户端，兼容 OpenAI SDK 接口，默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认
    - 支持多种LLM提供商自动检测
    - 统一的调用接口
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，未提供时从环境变量加载。

        Args:
            model: 模型名称，默认从 LLM_MODEL_ID 读取
            api_key: API密钥，默认从 LLM_API_KEY 读取
            base_url: 服务地址，默认从 LLM_BASE_URL 读取
            provider: LLM提供商，默认自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时秒数，默认60
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        self.kwargs = kwargs

        # 自动检测provider
        self.provider = provider or self._auto_detect_provider(api_key, base_url)

        # 解析api_key和base_url
        self.api_key, self.base_url = self._resolve_credentials(api_key, base_url)

        # 清理 base_url 中的常见错误（如带了 /chat/completions）
        self.base_url = _sanitize_base_url(self.base_url)

        # 默认模型
        if not self.model:
            self.model = self._get_default_model()
        if not all([self.api_key, self.base_url]):
            raise AgentsException("API密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建OpenAI客户端
        self._client = self._create_client()

        # 打印调试信息
        actual_endpoint = self.base_url.rstrip('/') + '/v1/chat/completions'
        key_preview = self.api_key[:20] + "..." if len(self.api_key) > 20 else self.api_key
        print(f"[LLM] provider={self.provider} model={self.model}")
        print(f"[LLM] base_url={self.base_url}")
        print(f"[LLM] endpoint -> {actual_endpoint}")
        print(f"[LLM] api_key={key_preview}")

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """自动检测LLM提供商

        优先级：用户显式传入的参数 > LLM_BASE_URL > 特定提供商环境变量 > API密钥格式
        """
        # 1. 优先根据 LLM_BASE_URL 判断（这是用户最明确的意图表达）
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url:
            url_lower = actual_base_url.lower()
            if "api.openai.com" in url_lower:
                return "openai"
            elif "api.deepseek.com" in url_lower:
                return "deepseek"
            elif "dashscope.aliyuncs.com" in url_lower:
                return "qwen"
            elif "api-inference.modelscope.cn" in url_lower:
                return "modelscope"
            elif "api.moonshot.cn" in url_lower:
                return "kimi"
            elif "open.bigmodel.cn" in url_lower:
                return "zhipu"
            elif "localhost" in url_lower or "127.0.0.1" in url_lower:
                if ":11434" in url_lower or "ollama" in url_lower:
                    return "ollama"
                elif ":8000" in url_lower and "vllm" in url_lower:
                    return "vllm"
                else:
                    return "local"
            elif any(port in url_lower for port in [":8080", ":7860", ":5000"]):
                return "local"

        # 2. 根据 LLM_API_KEY 格式判断
        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        if actual_api_key:
            key_lower = actual_api_key.lower()
            if actual_api_key.startswith("ms-"):
                return "modelscope"
            elif key_lower == "ollama":
                return "ollama"
            elif key_lower == "vllm":
                return "vllm"
            elif key_lower == "local":
                return "local"

        # 3. 最后检查特定提供商的环境变量（如 OPENAI_API_KEY, DEEPSEEK_API_KEY 等）
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("DEEPSEEK_API_KEY"):
            return "deepseek"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "qwen"
        if os.getenv("MODELSCOPE_API_KEY"):
            return "modelscope"
        if os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"):
            return "kimi"
        if os.getenv("ZHIPU_API_KEY") or os.getenv("GLM_API_KEY"):
            return "zhipu"
        if os.getenv("OLLAMA_API_KEY") or os.getenv("OLLAMA_HOST"):
            return "ollama"
        if os.getenv("VLLM_API_KEY") or os.getenv("VLLM_HOST"):
            return "vllm"

        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple:
        """根据provider解析API密钥和base_url"""
        provider_map = {
            "openai":     ("OPENAI_API_KEY",      "https://api.openai.com/v1"),
            "deepseek":   ("DEEPSEEK_API_KEY",     "https://api.deepseek.com"),
            "qwen":       ("DASHSCOPE_API_KEY",    "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "modelscope": ("MODELSCOPE_API_KEY",   "https://api-inference.modelscope.cn/v1/"),
            "kimi":       ("KIMI_API_KEY",         "https://api.moonshot.cn/v1"),
            "zhipu":      ("ZHIPU_API_KEY",        "https://open.bigmodel.cn/api/paas/v4"),
            "ollama":     ("OLLAMA_API_KEY",       "http://localhost:11434/v1"),
            "vllm":       ("VLLM_API_KEY",         "http://localhost:8000/v1"),
            "local":      ("LLM_API_KEY",          "http://localhost:8000/v1"),
        }

        if self.provider in provider_map:
            env_key, default_url = provider_map[self.provider]

            # kimi 也支持 MOONSHOT_API_KEY
            if self.provider == "kimi":
                resolved_key = (api_key or os.getenv("MOONSHOT_API_KEY")
                                or os.getenv(env_key) or os.getenv("LLM_API_KEY"))
            # zhipu 也支持 GLM_API_KEY
            elif self.provider == "zhipu":
                resolved_key = (api_key or os.getenv("GLM_API_KEY")
                                or os.getenv(env_key) or os.getenv("LLM_API_KEY"))
            # ollama/vllm/local 允许无 key
            elif self.provider in ("ollama", "vllm", "local"):
                resolved_key = (api_key or os.getenv(env_key)
                                or os.getenv("LLM_API_KEY") or self.provider)
            else:
                resolved_key = (api_key or os.getenv(env_key)
                                or os.getenv("LLM_API_KEY"))

            resolved_url = base_url or os.getenv("LLM_BASE_URL") or default_url
            return resolved_key, resolved_url

        # auto：使用通用配置
        return (api_key or os.getenv("LLM_API_KEY"),
                base_url or os.getenv("LLM_BASE_URL"))

    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def _get_default_model(self) -> str:
        """获取默认模型"""
        defaults = {
            "openai":     "gpt-3.5-turbo",
            "deepseek":   "deepseek-chat",
            "qwen":       "qwen-plus",
            "modelscope": "Qwen/Qwen2.5-72B-Instruct",
            "kimi":       "moonshot-v1-8k",
            "zhipu":      "glm-4",
            "ollama":     "llama3.2",
            "vllm":       "meta-llama/Llama-2-7b-chat-hf",
            "local":      "local-model",
        }
        if self.provider in defaults:
            return defaults[self.provider]

        # auto: 根据base_url推断
        url_lower = os.getenv("LLM_BASE_URL", "").lower()
        if "deepseek" in url_lower:
            return "deepseek-chat"
        elif "modelscope" in url_lower:
            return "Qwen/Qwen2.5-72B-Instruct"
        elif "dashscope" in url_lower:
            return "qwen-plus"
        elif "moonshot" in url_lower:
            return "moonshot-v1-8k"
        elif "bigmodel" in url_lower:
            return "glm-4"
        elif "ollama" in url_lower or ":11434" in url_lower:
            return "llama3.2"
        elif "localhost" in url_lower or "127.0.0.1" in url_lower:
            return "local-model"
        return "gpt-3.5-turbo"

    def think(self, messages: list[dict[str, str]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        流式调用LLM，逐步返回响应文本。

        Args:
            messages: 消息列表
            temperature: 温度参数

        Yields:
            str: 流式响应的文本片段
        """
        print(f"[LLM] Calling {self.model} ...")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            print("[LLM] Response:")
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()

        except Exception as e:
            print(f"[LLM] API Error: {e}")
            raise AgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, str]], **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.pop('temperature', self.temperature),
                max_tokens=kwargs.pop('max_tokens', self.max_tokens),
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise AgentsException(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, str]], **kwargs) -> Iterator[str]:
        """流式调用，与think()功能相同。"""
        temperature = kwargs.pop('temperature', None)
        yield from self.think(messages, temperature)


# --- 测试 ---
if __name__ == '__main__':
    try:
        llmClient = AgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "用一句话介绍你自己"}
        ]

        print("--- Calling LLM ---")
        collected = []
        for chunk in llmClient.think(exampleMessages):
            collected.append(chunk)
        responseText = "".join(collected)
        if responseText:
            print("\n--- Full Response ---")
            print(responseText)

    except (ValueError, AgentsException) as e:
        print(f"Error: {e}")
