"""LLM服务模块"""

from typing import Optional
from ..config import get_settings
from ..tools.llm import AgentsLLM

# 全局LLM实例
_llm_instance: Optional[AgentsLLM] = None


def get_llm() -> AgentsLLM:
    """
    获取 LLM 客户端实例 (单例模式)

    配置项从环境变量 / .env 加载：
    - LLM_MODEL_ID: 模型名称
    - LLM_API_KEY: API 密钥
    - LLM_BASE_URL: 服务地址
    - LLM_TIMEOUT: 超时秒数（默认 60）

    Returns:
        AgentsLLM: LLM 客户端实例
    """
    global _llm_instance
    if _llm_instance is None:
        settings = get_settings()

        _llm_instance = AgentsLLM(
            model=settings.llm_model or None,
            api_key=settings.llm_api_key or None,
            base_url=settings.llm_base_url or None,
        )

    return _llm_instance
