"""Skills 系统的初始化入口

在 AgentRuntime 启动时调用此函数，自动：
1. 注册所有内置 Skills（Python 类）
2. 扫描 builtin 目录，注册 Markdown Skills（.md 文件）
3. 扫描 AGENT_RUNTIME_SKILLS_PATHS 环境变量指定的外部路径
"""

import os
from pathlib import Path

from .registry import SkillRegistry
from .builtin import NoteSkill


def init_skills(
    extra_paths: list[str | Path] | None = None,
    markdown_paths: list[str | Path] | None = None,
) -> SkillRegistry:
    """初始化 Skills 系统

    Args:
        extra_paths: 额外的 Python Skill（.skill.py）搜索路径
        markdown_paths: 额外的 Markdown Skill（.md）搜索路径

    Returns:
        已填充的 SkillRegistry 单例
    """
    registry = SkillRegistry()

    # 防止重复初始化（单例模式下同一进程只能初始化一次）
    if registry._initialized:
        return registry

    # 1. 注册内置 Python Skills
    registry.register(NoteSkill())

    # 2. 扫描内置 builtin 目录（.skill.py 和 .md 混合）
    import agent_runtime
    builtin_path = Path(agent_runtime.__file__).parent / "skills" / "builtin"
    if builtin_path.is_dir():
        registry.discover(builtin_path)           # .skill.py
        registry.discover_markdown(builtin_path)  # .md

    # 3. 扫描 extra_paths（Python Skills）
    if extra_paths:
        for p in extra_paths:
            registry.discover(p)

    # 4. 扫描 markdown_paths（Markdown Skills）
    if markdown_paths:
        for p in markdown_paths:
            registry.discover_markdown(p)

    # 5. 扫描 AGENT_RUNTIME_SKILLS_PATHS 环境变量（Markdown Skills）
    env_paths = os.getenv("AGENT_RUNTIME_SKILLS_PATHS", "")
    if env_paths:
        for p in env_paths.split(os.pathsep):  # Windows 上用 ; 分隔
            p = p.strip()
            if p:
                registry.discover_markdown(p)

    registry._initialized = True
    return registry


# 默认初始化（延迟初始化，在首次访问时触发）
_default_registry: SkillRegistry | None = None


def get_default_registry() -> SkillRegistry:
    """获取默认已初始化的 SkillRegistry"""
    global _default_registry
    if _default_registry is None:
        _default_registry = init_skills()
    return _default_registry
