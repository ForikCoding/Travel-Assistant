"""NoteSkill - 笔记管理 Skill 示例

演示如何编写一个 Skill：
1. 继承 Skill 基类
2. 定义 metadata（name、description、parameters）
3. 实现 execute 方法
"""

from pathlib import Path
from typing import Any, Dict

from ..base import Skill, SkillMetadata, SkillParameter
from ..context import SkillContext


class NoteSkill(Skill):
    """笔记管理 Skill：创建、读取、列出笔记"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="note",
            description="管理个人笔记，支持创建、读取和列出操作",
            parameters=[
                SkillParameter(
                    name="action",
                    type="string",
                    description="操作类型：create（创建）/ read（读取）/ list（列出）",
                    required=True,
                ),
                SkillParameter(
                    name="title",
                    type="string",
                    description="笔记标题（create 和 read 时必填）",
                    required=False,
                ),
                SkillParameter(
                    name="content",
                    type="string",
                    description="笔记内容（create 时必填）",
                    required=False,
                ),
            ],
            tags=["utility", "notes"],
            version="0.1.0",
        )

    def execute(self, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """执行笔记操作

        Args:
            context: SkillContext 展开的上下文字典
            **kwargs: action, title, content

        Returns:
            {"success": bool, "data": Any, "error": str}
        """
        action = kwargs.get("action")
        title = kwargs.get("title", "untitled")
        content = kwargs.get("content", "")

        # 从 context 中获取 agent_context，解析笔记存储路径
        agent_context = context.get("agent_context", {})
        notes_dir = agent_context.get("notes_dir")
        if notes_dir:
            notes_path = Path(notes_dir)
        else:
            # 默认存储到当前目录下的 .agent_notes
            notes_path = Path.cwd() / ".agent_notes"

        try:
            if action == "create":
                return self._create(notes_path, title, content)
            elif action == "read":
                return self._read(notes_path, title)
            elif action == "list":
                return self._list(notes_path)
            else:
                return {"success": False, "error": f"未知操作: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create(self, notes_path: Path, title: str, content: str) -> Dict[str, Any]:
        notes_path.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        file_path = notes_path / f"{safe_title}.md"
        file_path.write_text(content, encoding="utf-8")
        return {"success": True, "data": {"path": str(file_path)}}

    def _read(self, notes_path: Path, title: str) -> Dict[str, Any]:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        file_path = notes_path / f"{safe_title}.md"
        if not file_path.exists():
            return {"success": False, "error": f"笔记不存在: {title}"}
        return {"success": True, "data": {"content": file_path.read_text(encoding="utf-8")}}

    def _list(self, notes_path: Path) -> Dict[str, Any]:
        if not notes_path.exists():
            return {"success": True, "data": {"notes": []}}
        notes = [f.name for f in notes_path.glob("*.md")]
        return {"success": True, "data": {"notes": notes, "count": len(notes)}}
