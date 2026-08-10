"""
工具基类

定义 Tool 和 ToolParameter 的基础抽象，参考 hello-agents 的 tools/base.py 实现。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolParameter:
    """工具参数定义

    Attributes:
        name: 参数名称
        type: 参数类型 (string, number, object, array, boolean)
        description: 参数描述
        required: 是否必填
        default: 默认值
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """工具基类

    所有工具的抽象基类，子类必须实现 run() 和 get_parameters() 方法。
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具

        Args:
            parameters: 工具参数字典

        Returns:
            执行结果字符串
        """
        ...

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义

        Returns:
            ToolParameter 列表
        """
        ...

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """验证参数完整性"""
        required_params = [p.name for p in self.get_parameters() if p.required]
        return all(param in parameters for param in required_params)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                }
                for p in self.get_parameters()
            ],
        }

    def __str__(self) -> str:
        return f"Tool(name={self.name})"

    def __repr__(self) -> str:
        return self.__str__()
