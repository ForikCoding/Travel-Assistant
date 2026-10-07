"""SkillTool - 将 Skill 注册为 Tool，使 Agent 可通过工具调用协议触发"""

from typing import Any, Dict, List, TYPE_CHECKING

from ..base import Tool, ToolParameter
from ..registry import global_registry as tool_registry

if TYPE_CHECKING:
    from ...skills.registry import SkillRegistry

from ...skills.context import SkillContext


class SkillTool(Tool):
    """Skill 包装为 Tool 的实现

    通过 Tool 接口暴露 Skill 能力，Agent 可以像调用普通 Tool 一样调用 Skill。
    """

    def __init__(self, skill_registry: "SkillRegistry | None" = None):
        # SkillTool 本身没有固定名字和描述，
        # 实际暴露的是各个具体 Skill
        super().__init__(
            name="skill",
            description="调用一个可复用的 Skill 任务单元。参数 skill_name 指定要调用的 Skill，args 传递其所需参数。",
        )
        self._registry = skill_registry

    def _get_registry(self) -> "SkillRegistry":
        """延迟获取 Registry，避免循环导入"""
        if self._registry is None:
            from ...skills.init import get_default_registry
            self._registry = get_default_registry()
        return self._registry

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行指定的 Skill

        Args:
            parameters: 必须包含 skill_name，可选包含 args（dict）和 shared_state（dict）

        Returns:
            Skill 执行结果的 JSON 字符串
        """
        skill_name = parameters.get("skill_name")
        if not skill_name:
            return '{"error": "缺少必填参数 skill_name"}'

        skill = self._get_registry().get(skill_name)
        if skill is None:
            return f'{{"error": "未找到名为 \\"{skill_name}\\" 的 Skill"}}'

        # 解析参数
        skill_args = parameters.get("args", {})
        shared_state = parameters.get("shared_state", {})

        # 构建上下文
        context = SkillContext(
            skill_name=skill_name,
            agent_context=parameters.get("agent_context", {}),
            shared_state=shared_state,
        )

        try:
            # 验证参数
            if not skill.validate_parameters(skill_args):
                missing = [
                    p.name for p in skill.metadata.parameters
                    if p.required and p.name not in skill_args
                ]
                return f'{{"error": "缺少必填参数: {", ".join(missing)}"}}'

            # 执行
            result = skill.execute(context.to_dict(), **skill_args)

            # 序列化结果
            import json
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            import json
            return json.dumps({"error": f"Skill 执行异常: {str(e)}"}, ensure_ascii=False)

    def get_parameters(self) -> List[ToolParameter]:
        """返回工具参数定义"""
        # 获取所有已注册 Skill 的名字和描述，构建动态参数说明
        skills = self._get_registry().list_all()

        # 生成 skill_name 的可选值列表
        skill_names = [s.name for s in skills]
        skill_descriptions = "\n".join(
            f"  - {s.name}: {s.description}" for s in skills
        )

        return [
            ToolParameter(
                name="skill_name",
                type="string",
                description=(
                    f"要调用的 Skill 名字。可选值：\n{skill_descriptions}"
                    if skill_descriptions
                    else "未注册任何 Skill"
                ),
                required=True,
            ),
            ToolParameter(
                name="args",
                type="object",
                description="Skill 所需参数，键值对形式",
                required=False,
                default={},
            ),
            ToolParameter(
                name="shared_state",
                type="object",
                description="Skill 间共享的状态字典",
                required=False,
                default={},
            ),
        ]

    def list_skills(self) -> List[str]:
        """列出所有可用 Skill 名字"""
        return [s.name for s in self._get_registry().list_all()]


# 默认实例，供全局复用
_default_skill_tool: "SkillTool | None" = None


def get_default_skill_tool() -> "SkillTool":
    """获取全局默认 SkillTool 实例"""
    global _default_skill_tool
    if _default_skill_tool is None:
        _default_skill_tool = SkillTool()
    return _default_skill_tool


def register_skill_tool(registry: "SkillRegistry | None" = None) -> None:
    """将 SkillTool 注册到全局 ToolRegistry"""
    tool = get_default_skill_tool()
    if registry is not None:
        tool._registry = registry
    tool_registry.register_tool(tool)
