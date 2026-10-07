"""SkillLoader - Markdown Skill 文件加载器抽象基类

定义统一的加载接口，所有格式的 Loader 必须实现：
- load(path) -> Optional[Dict]    加载单个文件
- scan_directory(directory) -> List[Dict]  扫描目录
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class SkillLoader(ABC):
    """Markdown Skill 文件加载器抽象基类"""

    @abstractmethod
    def load(self, path: str | Path) -> Dict[str, Any] | None:
        """加载并解析一个 .md Skill 文件

        Args:
            path: .md 文件路径

        Returns:
            解析结果字典：
            {
                "name": str,
                "description": str,
                "version": str,
                "author": str | None,
                "parameters": list[SkillParameter],
                "tags": list[str],
                "instructions": str,
                "file_path": str,
            }
            解析失败返回 None
        """
        pass

    @abstractmethod
    def scan_directory(self, directory: str | Path) -> List[Dict[str, Any]]:
        """扫描目录下所有 .md Skill 文件，返回解析结果列表

        Args:
            directory: 目录路径

        Returns:
            解析结果字典列表（仅包含成功解析的文件）
        """
        pass
