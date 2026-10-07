"""Skill 抽象基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillParameter:
    """Skill 参数定义"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class SkillMetadata:
    """Skill 元数据"""
    name: str
    description: str
    parameters: List[SkillParameter] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "0.1.0"
    author: Optional[str] = None

    def get_parameter_dict(self) -> Dict[str, Any]:
        """获取参数定义的字典格式（供 LLM 使用）"""
        return {
            p.name: {
                "type": p.type,
                "description": p.description,
                "required": p.required,
                "default": p.default,
            }
            for p in self.parameters
        }


class Skill(ABC):
    """Skill 抽象基类

    所有可复用的任务单元都应该继承此类。
    一个 Skill 可以调用多个 Tools，也可以有自己的内部状态和步骤。
    """

    @property
    @abstractmethod
    def metadata(self) -> SkillMetadata:
        """返回 Skill 的元数据"""
        pass

    @abstractmethod
    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """同步执行入口

        Args:
            context: SkillContext 展开的上下文字典，包含:
                - skill_name: str
                - agent_context: AgentContext | None
                - skill_results: dict[str, Any]  本次任务中其他 Skill 的执行结果
                - shared_state: dict[str, Any]   Skill 间共享状态
            **kwargs: Skill 自身定义的参数（与 metadata.parameters 对应）

        Returns:
            执行结果字典，应包含 "success"（bool）和 "data"/"error"（二选一）
        """
        pass

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """验证参数是否满足 metadata 中的定义"""
        for p in self.metadata.parameters:
            if p.required and p.name not in params:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "parameters": [p.__dict__ for p in self.metadata.parameters],
            "tags": self.metadata.tags,
            "version": self.metadata.version,
        }
