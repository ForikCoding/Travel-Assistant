"""配置管理模块"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# 加载环境变量
# 首次尝试加载当前目录的.env
load_dotenv()

class Settings(BaseSettings):
    """应用配置"""

    # 应用基本配置
    app_name: str = "智能旅行助手"
    app_version: str = "1.0.0"
    debug: bool = False

    # 服务器配置
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS配置 - 使用字符串，在代码中分割
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"

    # 高德地图API配置
    amap_api_key: str = ""

    # Unsplash API配置
    unsplash_access_key: str = ""
    unsplash_secret_key: str = ""

    # LLM配置
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    # 日志配置
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore" # 忽略额外的环境变量

    def get_cors_origins_list(self) -> List[str]:
        """将CORS来源字符串转换为列表"""
        return [origin.strip() for origin in self.cors_origins.split(',')]

# 创建全局配置实例
settings = Settings()

def get_settings() -> Settings:
    """获取全局配置实例"""
    return settings

def validate_config():
    """验证配置是否完整"""
    errors = []
    warnings = []

    if not settings.amap_api_key:
        errors.append("高德地图API Key未配置，请在环境变量中设置AMAP_API_KEY。")

    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not llm_api_key:
        errors.append("LLM API Key未配置，请在环境变量中设置LLM_API_KEY或OPENAI_API_KEY。")

    llm_base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not llm_base_url:
        warnings.append("LLM Base URL未配置，默认使用OpenAI API。")

    llm_model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
    if not llm_model:
        warnings.append("LLM模型未配置，默认使用gpt-4。")

    if errors:
       error_msg = "配置错误:\n" + "\n".join(f"  - {e}" for e in errors)
       raise ValueError(error_msg)

    if warnings:
        print("\n⚠️  配置警告:")
        for w in warnings:
            print(f"  - {w}")

    return True

# 打印配置信息(用于调试)
def print_config():
    """打印当前配置(隐藏敏感信息)"""
    print(f"应用名称: {settings.app_name}")
    print(f"版本: {settings.app_version}")
    print(f"服务器: {settings.host}:{settings.port}")
    print(f"高德地图API Key: {'已配置' if settings.amap_api_key else '未配置'}")

    # 检查LLM配置
    llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL") or settings.llm_base_url
    llm_model = os.getenv("LLM_MODEL_ID") or settings.llm_model

    print(f"LLM API Key: {'已配置' if llm_api_key else '未配置'}")
    print(f"LLM Base URL: {llm_base_url}")
    print(f"LLM Model: {llm_model}")
    print(f"日志级别: {settings.log_level}")