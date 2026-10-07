"""SkillExecute - 指令型 Skill 执行器

将 Markdown 文件中定义的指令直接作为 LLM 提示词返回，
由 Agent 在下一轮对话中自行按照指令执行。

MarkdownSkillLoader 和 YamlSkillLoader 加载后的 Skill 均使用此类执行。
"""

from typing import Any, Dict

from .base import Skill, SkillMetadata


class SkillExecute(Skill):
    """指令型 Skill：执行即返回指令文本，由 LLM 自行执行

    与代码型 Skill 不同，此类 Skill 本身不执行具体操作，
    而是将完整指令注入 Agent 的对话上下文，让 LLM 理解后自行决定行动。
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        description: str = "",
        parameters: list = None,
        tags: list = None,
        version: str = "0.1.0",
        author: str = None,
        file_path: str = None,
    ):
        self._metadata = SkillMetadata(
            name=name,
            description=description or f"Markdown Skill: {name}",
            parameters=parameters or [],
            tags=tags or [],
            version=version,
            author=author,
        )
        self._instructions = instructions
        self._file_path = file_path

    @property
    def metadata(self) -> SkillMetadata:
        return self._metadata

    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行 Markdown Skill

        返回指令文本，Agent 应将其注入到下一轮 LLM 对话中，
        让 LLM 在新的上下文中按照 Skill 指令执行任务。

        Returns:
            {
                "success": True,
                "data": {
                    "skill_name": str,           # Skill 名字
                    "instructions": str,         # 要注入的完整指令
                    "parameters": dict,          # 本次调用的参数
                    "directive": str,            # 简短的行动指令，供 LLM 直接执行
                }
            }
        """
        # 构建简短的 directive：告诉 LLM 应该如何执行
        param_lines = []
        for k, v in kwargs.items():
            param_lines.append(f"- {k}: {v}")

        params_str = "\n".join(param_lines) if param_lines else "（无参数）"

        directive = (
            f"你正在执行 Skill 「{self._metadata.name}」。\n"
            f"本次调用参数：\n{params_str}\n\n"
            f"请严格按照该 Skill 的指令执行任务。"
        )

        return {
            "success": True,
            "data": {
                "skill_name": self._metadata.name,
                "instructions": self._instructions,
                "parameters": kwargs,
                "directive": directive,
            },
        }

    def get_instructions(self) -> str:
        """获取原始指令文本"""
        return self._instructions

    def __repr__(self) -> str:
        return f"<MarkdownSkill(name={self._metadata.name})>"
