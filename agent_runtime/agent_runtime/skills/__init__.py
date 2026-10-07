"""agent_runtime.skills - 可复用任务单元

提供比 Tool 更高一级的抽象：一个 Skill 可以包含多步逻辑、状态管理和工具组合。

支持两种 Skill 类型：
- Python Skill：继承 Skill 类，实现 execute 方法
- Markdown Skill：格式 C 纯指令模板，通过 .md 文件定义，LLM 自行执行
"""

from .base import Skill, SkillMetadata, SkillParameter
from .context import SkillContext
from .registry import SkillRegistry
from .skill_execute import SkillExecute
from .skill_loader import SkillLoader
from .markdown_skill_loader import MarkdownSkillLoader
from .yaml_skill_loader import YamlSkillLoader
from .init import init_skills, get_default_registry

__all__ = [
    "Skill",
    "SkillMetadata",
    "SkillParameter",
    "SkillContext",
    "SkillRegistry",
    "SkillExecute",
    "SkillLoader",
    "MarkdownSkillLoader",
    "YamlSkillLoader",
    "init_skills",
    "get_default_registry",
]
