"""Skill 执行上下文"""

from typing import Any, Dict, Optional


class SkillContext:
    """Skill 执行时的上下文容器

    用于在 Skill 执行过程中传递和共享信息。
    """

    def __init__(
        self,
        skill_name: str,
        agent_context: Optional[Dict[str, Any]] = None,
        skill_results: Optional[Dict[str, Any]] = None,
        shared_state: Optional[Dict[str, Any]] = None,
    ):
        self.skill_name = skill_name
        self.agent_context = agent_context or {}
        self.skill_results = skill_results or {}
        self.shared_state = shared_state or {}

    def to_dict(self) -> Dict[str, Any]:
        """展开为字典，传递给 Skill.execute()"""
        return {
            "skill_name": self.skill_name,
            "agent_context": self.agent_context,
            "skill_results": self.skill_results,
            "shared_state": self.shared_state,
        }

    def set_result(self, skill_name: str, result: Dict[str, Any]) -> None:
        """记录某个 Skill 的执行结果，供其他 Skill 使用"""
        self.skill_results[skill_name] = result

    def get_result(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """获取其他 Skill 的执行结果"""
        return self.skill_results.get(skill_name)

    def set_shared(self, key: str, value: Any) -> None:
        """设置 Skill 间共享的状态"""
        self.shared_state[key] = value

    def get_shared(self, key: str, default: Any = None) -> Any:
        """获取共享状态"""
        return self.shared_state.get(key, default)
