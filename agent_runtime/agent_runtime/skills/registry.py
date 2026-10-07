"""Skill 注册表"""

import os
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import Skill, SkillMetadata


class SkillRegistry:
    """Skill 全局注册表（单例模式）

    管理所有可用 Skills，支持按名字查找、按标签筛选、自动发现。
    """

    _instance: Optional["SkillRegistry"] = None
    _skills: Dict[str, Skill]
    _discovered_paths: Set[str]
    _initialized: bool  # 是否已完成初始化（防止 init_skills 重复注册）

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._skills = {}
            cls._instance._discovered_paths = set()
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # 单例模式下 __init__ 会被调用多次，仅需初始化一次
        pass

    # ------------------------------------------------------------------
    # 注册 / 注销
    # ------------------------------------------------------------------

    def register(self, skill: Skill) -> None:
        """注册一个 Skill 实例

        Args:
            skill: Skill 子类实例

        Raises:
            ValueError: 已有同名 Skill 已注册
        """
        name = skill.metadata.name
        if name in self._skills:
            raise ValueError(f"Skill '{name}' 已被注册，请先注销或使用不同名字")
        self._skills[name] = skill

    def unregister(self, name: str) -> bool:
        """注销指定名字的 Skill

        Returns:
            是否成功注销（不存在时返回 False）
        """
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def clear(self) -> None:
        """清空所有已注册的 Skills（主要用于测试）"""
        self._skills.clear()
        self._discovered_paths.clear()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Skill]:
        """按名字获取 Skill"""
        return self._skills.get(name)

    def list_all(self) -> List[SkillMetadata]:
        """列出所有已注册 Skill 的元数据（不含实现）"""
        return [skill.metadata for skill in self._skills.values()]

    def list_all_with_instances(self) -> List[Skill]:
        """列出所有已注册 Skill 实例"""
        return list(self._skills.values())

    def find_by_tag(self, tag: str) -> List[SkillMetadata]:
        """按标签筛选 Skill"""
        return [
            skill.metadata
            for skill in self._skills.values()
            if tag in skill.metadata.tags
        ]

    def search(self, keyword: str) -> List[SkillMetadata]:
        """在名字和描述中搜索关键词"""
        kw = keyword.lower()
        return [
            skill.metadata
            for skill in self._skills.values()
            if kw in skill.metadata.name.lower() or kw in skill.metadata.description.lower()
        ]

    # ------------------------------------------------------------------
    # 自动发现
    # ------------------------------------------------------------------

    def discover(self, path: str | Path) -> int:
        """扫描指定目录，自动发现并注册 *.skill.py 文件

        符合规范的 Skill 文件必须：
        1. 定义名为 `<PascalCaseName>Skill` 的类（继承 Skill）
        2. 文件名格式：`<name>.skill.py`

        Args:
            path: 搜索路径

        Returns:
            本次新注册的 Skill 数量
        """
        path = Path(path).resolve()
        if not path.is_dir():
            return 0

        if str(path) in self._discovered_paths:
            return 0
        self._discovered_paths.add(str(path))

        count = 0
        for file in path.glob("*.skill.py"):
            try:
                self._load_skill_file(file)
                count += 1
            except Exception as e:
                # 静默失败，仅记录
                import logging
                logging.warning(f"加载 Skill 文件失败 {file}: {e}")
        return count

    def discover_site_packages(self) -> int:
        """扫描 agent_runtime 包内的 skills/builtin 目录"""
        import agent_runtime
        builtin_path = Path(agent_runtime.__file__).parent / "skills" / "builtin"
        return self.discover(builtin_path)

    # ------------------------------------------------------------------
    # Markdown Skill 发现
    # ------------------------------------------------------------------

    def discover_markdown(self, path: str | Path) -> int:
        """扫描指定目录，发现并注册所有 *.md Skill 文件

        支持两种格式：
        1. MarkdownSkillLoader：## 元信息 / ## 指令 区块格式
        2. YamlSkillLoader：YAML frontmatter（--- ... ---）格式

        Args:
            path: 搜索路径（目录）

        Returns:
            本次新注册的 Markdown Skill 数量
        """
        path = Path(path).resolve()
        if not path.is_dir():
            return 0

        from .markdown_skill_loader import MarkdownSkillLoader
        from .yaml_skill_loader import YamlSkillLoader
        from .skill_execute import SkillExecute

        count = 0
        for md_file in path.glob("*.md"):
            try:
                data = None
                # 优先用 BlockMarkdownSkillLoader（## 区块格式）
                data = MarkdownSkillLoader.load(md_file)
                # 其次用 YamlSkillLoader（frontmatter 格式）
                if data is None:
                    data = YamlSkillLoader.load(md_file)
                if data is None:
                    continue
                # 检查是否已注册
                if data["name"] in self._skills:
                    continue
                skill = SkillExecute(
                    name=data["name"],
                    instructions=data["instructions"],
                    description=data["description"],
                    parameters=data["parameters"],
                    tags=data["tags"],
                    version=data["version"],
                    author=data["author"],
                    file_path=data["file_path"],
                )
                self.register(skill)
                count += 1
            except Exception as e:
                import logging
                logging.warning(f"加载 Markdown Skill 失败 {md_file}: {e}")
        return count

    def discover_markdown_all(self, paths: list[str | Path]) -> int:
        """扫描多个路径，批量发现 Markdown Skills

        Args:
            paths: 路径列表

        Returns:
            总共注册的 Skill 数量
        """
        total = 0
        for p in paths:
            total += self.discover_markdown(p)
        return total

    def _load_skill_file(self, file: Path) -> None:
        """从 .skill.py 文件动态加载 Skill 类"""
        # 构造模块名
        module_name = f"agent_runtime.skills.discovered.{file.stem}"

        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 查找以 Skill 结尾的类
        for attr_name in dir(module):
            if attr_name.endswith("Skill"):
                cls = getattr(module, attr_name)
                if isinstance(cls, type) and issubclass(cls, Skill) and cls is not Skill:
                    self.register(cls())

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"<SkillRegistry({len(self._skills)} skills)>"
