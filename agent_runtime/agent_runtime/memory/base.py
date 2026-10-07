"""
记忆熊基础类和配置
- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统设置
- BaseMemory: 记忆基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
import os
from pydantic import BaseModel

# .env 已在包入口 agent_runtime/__init__.py 中加载（保证早于所有子模块生效）

class MemoryItem(BaseModel):
    """记忆项结构"""
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class MemoryConfig(BaseModel):
    """记忆系统配置"""
    
    # 存储路径
    storage_path: str = "./memory_data"

    # 向量/嵌入相关（可选关闭，适用于只用 SQLite 的轻量 episodic）
    # 默认禁用向量存储与嵌入模型：让 agent_runtime 默认就「纯 SQLite 离线」可用，
    # 不依赖 Qdrant / HuggingFace / DashScope 等任何外部服务。
    # 需要启用时再显式置 False（生产环境启用向量检索能力）。
    disable_vector_store: bool = False
    disable_embeddings: bool = False
    
    # 统计显示用的基础配置（仅用于展示）
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """从环境变量构建配置（MEMORY_* 前缀）。

        未设置（或为空字符串）的环境变量回退到字段默认值。
        参考 .env.example 中的「记忆系统配置」一节。
        """
        fields = cls.model_fields

        def _str(env_name: str, field: str) -> str:
            return os.getenv(env_name) or fields[field].default

        def _int(env_name: str, field: str) -> int:
            raw = os.getenv(env_name)
            return int(raw) if raw not in (None, "") else fields[field].default

        def _float(env_name: str, field: str) -> float:
            raw = os.getenv(env_name)
            return float(raw) if raw not in (None, "") else fields[field].default

        def _bool(env_name: str, field: str) -> bool:
            raw = os.getenv(env_name)
            if raw in (None, ""):
                return fields[field].default
            return raw.strip().lower() in ("1", "true", "yes", "on")

        modalities_raw = os.getenv("MEMORY_PERCEPTUAL_MODALITIES")
        if modalities_raw is None:
            modalities = list(fields["perceptual_memory_modalities"].default)
        else:
            modalities = [m.strip() for m in modalities_raw.split(",") if m.strip()]

        return cls(
            storage_path=_str("MEMORY_STORAGE_PATH", "storage_path"),
            disable_vector_store=_bool("MEMORY_DISABLE_VECTOR_STORE", "disable_vector_store"),
            disable_embeddings=_bool("MEMORY_DISABLE_EMBEDDINGS", "disable_embeddings"),
            max_capacity=_int("MEMORY_MAX_CAPACITY", "max_capacity"),
            importance_threshold=_float("MEMORY_IMPORTANCE_THRESHOLD", "importance_threshold"),
            decay_factor=_float("MEMORY_DECAY_FACTOR", "decay_factor"),
            working_memory_capacity=_int("MEMORY_WORKING_CAPACITY", "working_memory_capacity"),
            working_memory_tokens=_int("MEMORY_WORKING_TOKENS", "working_memory_tokens"),
            working_memory_ttl_minutes=_int("MEMORY_WORKING_TTL_MINUTES", "working_memory_ttl_minutes"),
            perceptual_memory_modalities=modalities,
        )


class BaseMemory(ABC):
    """记忆基类

    定义所有记忆类型的通用接口和行为
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项

        Args:
            memory_item: 记忆项对象

        Returns:
            记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索相关记忆

        Args:
            query: 查询内容
            limit: 返回数量限制
            **kwargs: 其他检索参数

        Returns:
            相关记忆列表
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在

        Args:
            memory_id: 记忆ID

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典
        """
        pass

    def _generate_id(self) -> str:
        """生成记忆ID"""
        import uuid
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """计算记忆重要性

        Args:
            content: 记忆内容
            base_importance: 基础重要性

        Returns:
            计算后的重要性分数
        """
        importance = base_importance

        # 基于内容长度
        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()