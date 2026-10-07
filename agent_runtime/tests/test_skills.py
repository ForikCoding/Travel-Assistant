"""agent_runtime.skills 模块测试"""

import tempfile
import pytest
from pathlib import Path

from agent_runtime.skills import (
    Skill,
    SkillMetadata,
    SkillParameter,
    SkillContext,
    SkillRegistry,
    SkillExecute,
    MarkdownSkillLoader,
    init_skills,
)


# =============================================================================
# 基类与元数据测试
# =============================================================================

class TestSkillMetadata:
    def test_create_metadata(self):
        meta = SkillMetadata(
            name="test-skill",
            description="测试 Skill",
            parameters=[
                SkillParameter(name="input", type="string", description="输入", required=True),
                SkillParameter(name="count", type="integer", description="次数", required=False, default=1),
            ],
            tags=["test", "demo"],
            version="1.0.0",
            author="tester",
        )
        assert meta.name == "test-skill"
        assert meta.version == "1.0.0"
        assert len(meta.parameters) == 2
        assert meta.parameters[0].required is True
        assert meta.parameters[1].default == 1

    def test_get_parameter_dict(self):
        meta = SkillMetadata(
            name="test",
            description="",
            parameters=[
                SkillParameter(name="x", type="number", description="数字", required=True),
            ],
        )
        param_dict = meta.get_parameter_dict()
        assert param_dict["x"]["type"] == "number"
        assert param_dict["x"]["required"] is True


class DummySkill(Skill):
    """测试用 Skill 实现"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dummy",
            description="一个虚拟 Skill",
            parameters=[
                SkillParameter(name="value", type="string", description="值", required=True),
            ],
        )

    def execute(self, context, **kwargs):
        v = kwargs.get("value", "")
        return {"success": True, "data": {"echo": v * 2}}


class TestSkill:
    def test_validate_parameters(self):
        skill = DummySkill()
        assert skill.validate_parameters({"value": "hello"}) is True
        assert skill.validate_parameters({}) is False  # 缺少必填参数

    def test_to_dict(self):
        skill = DummySkill()
        d = skill.to_dict()
        assert d["name"] == "dummy"
        assert d["description"] == "一个虚拟 Skill"
        assert len(d["parameters"]) == 1

    def test_execute(self):
        skill = DummySkill()
        result = skill.execute({}, value="abc")
        assert result["success"] is True
        assert result["data"]["echo"] == "abcabc"


# =============================================================================
# SkillContext 测试
# =============================================================================

class TestSkillContext:
    def test_create_context(self):
        ctx = SkillContext(skill_name="test", agent_context={"user": "Alice"})
        assert ctx.skill_name == "test"
        assert ctx.agent_context["user"] == "Alice"
        assert ctx.skill_results == {}
        assert ctx.shared_state == {}

    def test_set_get_result(self):
        ctx = SkillContext(skill_name="test")
        ctx.set_result("other", {"success": True, "data": {"x": 1}})
        assert ctx.get_result("other")["success"] is True
        assert ctx.get_result("nonexistent") is None

    def test_shared_state(self):
        ctx = SkillContext(skill_name="test")
        ctx.set_shared("counter", 10)
        assert ctx.get_shared("counter") == 10
        assert ctx.get_shared("missing", default=-1) == -1

    def test_to_dict(self):
        ctx = SkillContext(skill_name="test", shared_state={"k": "v"})
        d = ctx.to_dict()
        assert d["skill_name"] == "test"
        assert d["shared_state"]["k"] == "v"


# =============================================================================
# SkillRegistry 测试
# =============================================================================

class TestSkillRegistry:
    def setup_method(self):
        """每个测试前创建全新 Registry"""
        SkillRegistry._instance = None
        self.registry = SkillRegistry()

    def test_register_and_get(self):
        skill = DummySkill()
        self.registry.register(skill)
        assert self.registry.get("dummy") is skill
        assert self.registry.get("nonexistent") is None

    def test_register_duplicate_raises(self):
        skill = DummySkill()
        self.registry.register(skill)
        with pytest.raises(ValueError, match="已被注册"):
            self.registry.register(skill)

    def test_unregister(self):
        skill = DummySkill()
        self.registry.register(skill)
        assert self.registry.unregister("dummy") is True
        assert self.registry.get("dummy") is None
        assert self.registry.unregister("nonexistent") is False

    def test_list_all(self):
        self.registry.register(DummySkill())
        metas = self.registry.list_all()
        assert len(metas) == 1
        assert metas[0].name == "dummy"

    def test_find_by_tag(self):
        class TaggedSkill(Skill):
            @property
            def metadata(self):
                return SkillMetadata(name="tagged", description="", tags=["color", "blue"])

            def execute(self, context, **kwargs):
                return {"success": True}

        self.registry.register(TaggedSkill())
        results = self.registry.find_by_tag("blue")
        assert len(results) == 1
        assert results[0].name == "tagged"
        assert self.registry.find_by_tag("red") == []

    def test_search(self):
        class AnotherSkill(Skill):
            @property
            def metadata(self):
                return SkillMetadata(name="searchable", description="可搜索的 Skill")

            def execute(self, context, **kwargs):
                return {"success": True}

        self.registry.register(AnotherSkill())
        results = self.registry.search("可搜索")
        assert len(results) == 1
        assert results[0].name == "searchable"

    def test_clear(self):
        self.registry.register(DummySkill())
        self.registry.clear()
        assert len(self.registry) == 0

    def test_contains(self):
        skill = DummySkill()
        self.registry.register(skill)
        assert "dummy" in self.registry
        assert "nonexistent" not in self.registry

    def test_discover_python_skill_file(self):
        """测试发现 .skill.py 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_file = Path(tmpdir) / "hello.skill.py"
            skill_file.write_text(
                "from agent_runtime.skills import Skill, SkillMetadata\n"
                "class HelloSkill(Skill):\n"
                "    @property\n"
                "    def metadata(self):\n"
                "        return SkillMetadata(name='hello', description='greeting')\n"
                "    def execute(self, context, **kwargs):\n"
                "        return {'success': True}\n",
                encoding="utf-8",
            )
            count = self.registry.discover(tmpdir)
            assert count == 1
            assert self.registry.get("hello") is not None
            assert self.registry.get("hello").metadata.description == "greeting"


# =============================================================================
# MarkdownSkillLoader 测试
# =============================================================================

class TestMarkdownSkillLoader:
    SAMPLE_MD = """\
# Skill: office-docx

## 元信息
name: office-docx
description: 创建 Word 文档
version: 0.1.0
author: Travel-Assistant

## 参数规范
- title: string, 必填, 文档标题
- content: string, 必填, 文档正文
- output_path: string, 可选, 保存路径

## 标签
- office
- docx

## 指令
你是一个 Word 文档创建助手。当用户要求创建文档时：
1. 使用 python-docx 库创建 Document 对象
2. 添加标题和段落
3. 保存到用户指定路径

## 使用场景
当用户说"帮我创建一份报告"时调用此 Skill。
"""

    def test_load_valid_md(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write(self.SAMPLE_MD)
            path = f.name

        try:
            data = MarkdownSkillLoader.load(path)
            assert data is not None
            assert data["name"] == "office-docx"
            assert data["description"] == "创建 Word 文档"
            assert data["version"] == "0.1.0"
            assert data["author"] == "Travel-Assistant"
            assert len(data["parameters"]) == 3
            assert data["parameters"][0].name == "title"
            assert data["parameters"][0].required is True
            assert "python-docx" in data["instructions"]
            assert "office" in data["tags"]
            assert "docx" in data["tags"]
        finally:
            Path(path).unlink()

    def test_load_missing_instructions_returns_none(self):
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("# Skill: empty\n\n没有任何指令内容\n")
            path = f.name

        try:
            data = MarkdownSkillLoader.load(path)
            assert data is None  # 缺少指令块
        finally:
            Path(path).unlink()

    def test_load_nonexistent_file(self):
        data = MarkdownSkillLoader.load("/nonexistent/path/xxx.md")
        assert data is None

    def test_load_non_md_file(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
            f.write("not a markdown skill")
            path = f.name

        try:
            data = MarkdownSkillLoader.load(path)
            assert data is None
        finally:
            Path(path).unlink()

    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # 有效 Skill
            valid = Path(tmpdir) / "valid.md"
            valid.write_text(self.SAMPLE_MD, encoding="utf-8")

            # 无效 Skill（缺少指令）
            invalid = Path(tmpdir) / "invalid.md"
            invalid.write_text("# Skill: invalid\n\n没有 ## 指令 块\n", encoding="utf-8")

            results = MarkdownSkillLoader.scan_directory(tmpdir)
            assert len(results) == 1
            assert results[0]["name"] == "office-docx"


# =============================================================================
# SkillExecute 测试
# =============================================================================

class TestSkillExecute:
    def test_execute_returns_instructions(self):
        skill = SkillExecute(
            name="test-md",
            instructions="你是一个文档助手。创建 docx 时使用 python-docx。",
            description="测试指令型 Skill",
            parameters=[
                SkillParameter(name="title", type="string", description="标题", required=True),
            ],
        )

        result = skill.execute({}, title="报告", content="内容")

        assert result["success"] is True
        assert result["data"]["skill_name"] == "test-md"
        assert "python-docx" in result["data"]["instructions"]
        assert result["data"]["parameters"]["title"] == "报告"
        assert "你正在执行 Skill" in result["data"]["directive"]

    def test_get_instructions(self):
        instructions = "原始指令文本"
        skill = SkillExecute(name="test", instructions=instructions)
        assert skill.get_instructions() == instructions


# =============================================================================
# init_skills 集成测试
# =============================================================================

class TestInitSkills:
    def setup_method(self):
        SkillRegistry._instance = None

    def test_init_skills_builtin(self):
        """init_skills 应注册内置 Skills"""
        registry = init_skills()
        # NoteSkill 是内置的
        assert registry.get("note") is not None
        meta = registry.get("note").metadata
        assert meta.name == "note"

    def test_init_skills_with_markdown_paths(self):
        """init_skills 应扫描 markdown_paths"""
        with tempfile.TemporaryDirectory() as tmpdir:
            md_file = Path(tmpdir) / "custom.md"
            md_file.write_text(
                "# Skill: custom-md\n\n"
                "## 元信息\n"
                "name: custom-md\n"
                "description: 自定义 Markdown Skill\n"
                "\n"
                "## 指令\n"
                "执行自定义指令。\n",
                encoding="utf-8",
            )

            registry = init_skills(markdown_paths=[tmpdir])
            assert registry.get("custom-md") is not None

    def test_init_skills_singleton(self):
        """多次调用 init_skills 应返回同一 Registry 实例"""
        r1 = init_skills()
        r2 = init_skills()
        assert r1 is r2


# =============================================================================
# 集成测试：Skills 完整调用链路
# =============================================================================

import json
import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from agent_runtime.skills import init_skills, SkillRegistry, SkillExecute
from agent_runtime.tools.builtin import SkillTool, get_default_skill_tool
from agent_runtime.tools.registry import ToolRegistry, global_registry


class TestSkillsIntegration:
    """Skills 模块完整调用链路集成测试

    测试从 .md 文件加载 → 注册到 Registry → SkillTool 执行 → Agent 调用的全链路。
    """

    def setup_method(self):
        """每个测试前重置 Registry 和 ToolRegistry"""
        SkillRegistry._instance = None
        global_registry.clear()

    # -------------------------------------------------------------------------
    # 工具：创建临时 .md Skill 文件
    # -------------------------------------------------------------------------

    @staticmethod
    def _create_temp_md_skill(
        tmp_path: Path,
        name: str = "test-skill",
        description: str = "测试技能",
        instructions: str = "你是一个测试助手。返回 SUCCESS。",
        parameters: list = None,
    ) -> Path:
        """创建临时 .md Skill 文件并返回路径"""
        params_block = ""
        if parameters:
            params_block = "## 参数规范\n" + "\n".join(
                f"- {p['name']}: {p['type']}, {'必填' if p.get('required') else '可选'}, {p['description']}"
                for p in parameters
            ) + "\n"

        content = f"""\
# Skill: {name}

## 元信息
name: {name}
description: {description}

{params_block}
## 指令
{instructions}

## 使用场景
测试场景下使用此 Skill。
"""
        file_path = tmp_path / f"{name}.md"
        file_path.write_text(content, encoding="utf-8")
        return file_path

    # -------------------------------------------------------------------------
    # 测试1：.md 文件 → Registry → SkillTool 完整链路
    # -------------------------------------------------------------------------

    def test_md_file_to_skilltool_full_pipeline(self):
        """测试 .md 文件加载、注册、SkillTool 执行的完整链路"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Step 1: 创建 .md skill 文件
            skill_file = self._create_temp_md_skill(
                tmp_path,
                name="echo-skill",
                description="回显技能",
                instructions="你是一个回显助手。用户说什么你就原样返回。",
                parameters=[
                    {"name": "message", "type": "string", "required": True, "description": "要回显的消息"},
                ],
            )

            # Step 2: 初始化 Skills（扫描 .md 文件）
            registry = init_skills(markdown_paths=[tmpdir])

            # Step 3: 确认 Registry 中已存在该 Skill
            skill = registry.get("echo-skill")
            assert skill is not None
            assert isinstance(skill, SkillExecute)
            assert skill.metadata.description == "回显技能"
            assert "回显助手" in skill.get_instructions()

            # Step 4: SkillTool 执行该 Skill
            skill_tool = SkillTool(skill_registry=registry)
            result_json = skill_tool.run({
                "skill_name": "echo-skill",
                "args": {"message": "你好，世界"},
            })

            result = json.loads(result_json)
            assert result["success"] is True
            assert result["data"]["skill_name"] == "echo-skill"
            assert "回显助手" in result["data"]["instructions"]
            assert result["data"]["parameters"]["message"] == "你好，世界"

    # -------------------------------------------------------------------------
    # 测试2：SkillTool 注册到 ToolRegistry，Agent 可通过工具名调用
    # -------------------------------------------------------------------------

    def test_skilltool_registered_in_tool_registry(self):
        """SkillTool 注册到全局 ToolRegistry 后可被工具名调用"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并加载 Skill
            self._create_temp_md_skill(
                Path(tmpdir), name="calc-skill",
                instructions="你是一个计算器。返回 2+2=4。",
            )
            registry = init_skills(markdown_paths=[tmpdir])

            # 注册 SkillTool
            skill_tool = SkillTool(skill_registry=registry)
            global_registry.register_tool(skill_tool)

            # 通过 ToolRegistry 执行
            result = global_registry.execute_tool(
                "skill",
                json.dumps({"skill_name": "calc-skill", "args": {}}),
            )
            result_data = json.loads(result)
            assert result_data["success"] is True
            assert "计算器" in result_data["data"]["instructions"]

    # -------------------------------------------------------------------------
    # 测试3：多个 .md 文件批量加载
    # -------------------------------------------------------------------------

    def test_multiple_md_skills_loaded(self):
        """同一目录下多个 .md Skill 文件应全部被加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 创建多个 Skill 文件
            self._create_temp_md_skill(tmp_path, name="skill-a", description="技能 A")
            self._create_temp_md_skill(tmp_path, name="skill-b", description="技能 B")
            self._create_temp_md_skill(tmp_path, name="skill-c", description="技能 C")

            # 加载
            registry = init_skills(markdown_paths=[tmpdir])

            # 验证全部注册
            assert registry.get("skill-a") is not None
            assert registry.get("skill-b") is not None
            assert registry.get("skill-c") is not None
            assert len(registry.list_all()) >= 3  # 至少这3个

    # -------------------------------------------------------------------------
    # 测试4：Agent 模拟调用 Skill（mock LLM）
    # -------------------------------------------------------------------------

    def test_agent_mock_calls_skill_via_skilltool(self):
        """模拟 Agent 通过 ToolRegistry 调用 SkillTool，验证 LLM 工具调用流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # 创建 Skill
            self._create_temp_md_skill(
                tmp_path,
                name="report-skill",
                description="生成报告",
                instructions="你是一个报告生成助手。当用户要求生成报告时，使用 python-docx 生成文档。",
                parameters=[
                    {"name": "title", "type": "string", "required": True, "description": "报告标题"},
                    {"name": "content", "type": "string", "required": True, "description": "报告内容"},
                ],
            )

            # 初始化并注册
            registry = init_skills(markdown_paths=[tmpdir])
            skill_tool = SkillTool(skill_registry=registry)
            global_registry.register_tool(skill_tool)

            # 模拟 LLM 返回工具调用（LLM 决定调用 skill 工具）
            # LLM 会返回：skill[{"skill_name": "report-skill", "args": {"title": "...", "content": "..."}}]
            tool_input = json.dumps({
                "skill_name": "report-skill",
                "args": {"title": "Q1 报告", "content": "第一季度业绩增长 20%"},
            })

            # 通过 ToolRegistry 执行（模拟 Agent._execute_tool_call）
            result = global_registry.execute_tool("skill", tool_input)

            result_data = json.loads(result)
            assert result_data["success"] is True
            assert result_data["data"]["skill_name"] == "report-skill"
            assert result_data["data"]["parameters"]["title"] == "Q1 报告"
            # 指令中应包含 python-docx
            assert "python-docx" in result_data["data"]["instructions"]

    # -------------------------------------------------------------------------
    # 测试5：SkillTool 参数校验——缺少必填参数
    # -------------------------------------------------------------------------

    def test_skilltool_missing_required_parameter(self):
        """SkillTool 执行时应校验必填参数，缺失时返回错误"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_md_skill(
                Path(tmpdir),
                name="param-skill",
                description="参数测试",
                instructions="测试指令",
                parameters=[
                    {"name": "required_arg", "type": "string", "required": True, "description": "必填参数"},
                    {"name": "optional_arg", "type": "string", "required": False, "description": "可选参数"},
                ],
            )

            registry = init_skills(markdown_paths=[tmpdir])
            skill_tool = SkillTool(skill_registry=registry)

            # 缺少必填参数 required_arg
            result = skill_tool.run({
                "skill_name": "param-skill",
                "args": {"optional_arg": "value"},  # 故意漏掉 required_arg
            })

            result_data = json.loads(result)
            assert "error" in result_data, f"缺少必填参数时应返回 error 字段，实际返回: {result_data}"

    # -------------------------------------------------------------------------
    # 测试6：SkillTool 未知 Skill 名
    # -------------------------------------------------------------------------

    def test_skilltool_nonexistent_skill(self):
        """调用不存在的 Skill 应返回明确错误"""
        registry = init_skills()
        skill_tool = SkillTool(skill_registry=registry)

        result = skill_tool.run({
            "skill_name": "this-skill-does-not-exist",
            "args": {},
        })

        result_data = json.loads(result)
        assert "error" in result_data
        assert "未找到" in result_data["error"]

    # -------------------------------------------------------------------------
    # 测试7：init_skills + register_skill_tool 一键初始化
    # -------------------------------------------------------------------------

    def test_init_and_register_workflow(self):
        """测试 init_skills + register_skill_tool 的标准初始化流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_md_skill(
                Path(tmpdir),
                name="workflow-skill",
                description="工作流测试",
                instructions="执行工作流任务。",
            )

            # 标准初始化流程
            registry = init_skills(markdown_paths=[tmpdir])
            skill_tool = SkillTool(skill_registry=registry)
            global_registry.register_tool(skill_tool)

            # 确认全局注册表中有 skill 工具
            assert "skill" in global_registry.list_tools()

            # 执行
            result = global_registry.execute_tool(
                "skill",
                json.dumps({"skill_name": "workflow-skill", "args": {}}),
            )
            result_data = json.loads(result)
            assert result_data["success"] is True

    # -------------------------------------------------------------------------
    # 测试8：内置 Python Skill 与 Markdown Skill 共存
    # -------------------------------------------------------------------------

    def test_builtin_python_skill_coexist_with_markdown(self):
        """内置 Python Skill（NoteSkill）和外部 Markdown Skill 应共存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_md_skill(
                Path(tmpdir),
                name="external-md-skill",
                description="外部 Markdown Skill",
                instructions="外部 Markdown 指令。",
            )

            registry = init_skills(markdown_paths=[tmpdir])

            # 内置 NoteSkill 应存在
            assert registry.get("note") is not None
            # 外部 Markdown Skill 也应存在
            assert registry.get("external-md-skill") is not None

            # 两者类型不同
            assert type(registry.get("note")).__name__ == "NoteSkill"
            assert type(registry.get("external-md-skill")).__name__ == "SkillExecute"

    # -------------------------------------------------------------------------
    # 测试9：AGENT_RUNTIME_SKILLS_PATHS 环境变量扫描
    # -------------------------------------------------------------------------

    def test_env_paths_skills_discovery(self):
        """AGENT_RUNTIME_SKILLS_PATHS 环境变量应被扫描"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_md_skill(
                Path(tmpdir),
                name="env-path-skill",
                description="来自环境变量路径的 Skill",
                instructions="来自环境变量的指令。",
            )

            old_env = os.environ.get("AGENT_RUNTIME_SKILLS_PATHS", "")

            try:
                os.environ["AGENT_RUNTIME_SKILLS_PATHS"] = tmpdir
                # 重置单例以重新初始化
                SkillRegistry._instance = None
                registry = init_skills()

                assert registry.get("env-path-skill") is not None
            finally:
                if old_env:
                    os.environ["AGENT_RUNTIME_SKILLS_PATHS"] = old_env
                else:
                    os.environ.pop("AGENT_RUNTIME_SKILLS_PATHS", None)

    # -------------------------------------------------------------------------
    # 测试10：SkillTool.get_parameters() 动态反映已注册 Skills
    # -------------------------------------------------------------------------

    def test_skilltool_parameters_reflect_registered_skills(self):
        """SkillTool 的参数定义应动态反映 Registry 中已注册的 Skills 列表"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._create_temp_md_skill(
                Path(tmpdir),
                name="dynamic-skill",
                description="动态 Skill",
                instructions="动态指令。",
            )

            registry = init_skills(markdown_paths=[tmpdir])
            skill_tool = SkillTool(skill_registry=registry)

            params = skill_tool.get_parameters()
            skill_name_param = next(p for p in params if p.name == "skill_name")

            # skill_name 参数的 description 应包含已注册 Skill 的名字
            assert "dynamic-skill" in skill_name_param.description
            assert "note" in skill_name_param.description  # 内置 NoteSkill 也应在内


# =============================================================================
# YamlSkillLoader 测试
# =============================================================================

from agent_runtime.skills import YamlSkillLoader


class TestYamlSkillLoader:
    """测试 YAML frontmatter 格式支持"""

    SAMPLE_FRONTMATTER_MD = """\
---
name: browser-act
description: Browser automation CLI for AI agents
version: "2.0.2"
author: BrowserAct
---

# browser-act

Built by BrowserAct — Browser automation CLI for AI agents.

## Start here

This file is a discovery stub. After loading this skill, run:

```bash
browser-act get-skills core --skill-version 2.0.2
```
"""

    def test_load_frontmatter_md(self):
        """测试加载 YAML frontmatter 格式文件"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(self.SAMPLE_FRONTMATTER_MD)
            path = f.name

        try:
            data = YamlSkillLoader.load(path)
            assert data is not None
            assert data["name"] == "browser-act"
            assert data["description"] == "Browser automation CLI for AI agents"
            assert data["version"] == "2.0.2"
            assert data["author"] == "BrowserAct"
            assert "discovery stub" in data["instructions"]
            assert "browser-act get-skills" in data["instructions"]
        finally:
            Path(path).unlink()

    def test_is_frontmatter_format(self):
        """is_frontmatter_format 正确识别 frontmatter 文件"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write(self.SAMPLE_FRONTMATTER_MD)
            path = f.name

        try:
            assert YamlSkillLoader.is_frontmatter_format(path) is True
        finally:
            Path(path).unlink()

    def test_non_frontmatter_returns_none(self):
        """非 frontmatter 文件返回 None"""
        with tempfile.NamedTemporaryFile(
            suffix=".md", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Skill: test\n\n## 指令\n执行任务\n")
            path = f.name

        try:
            assert YamlSkillLoader.load(path) is None
            assert YamlSkillLoader.is_frontmatter_format(path) is False
        finally:
            Path(path).unlink()

    def test_scan_directory(self):
        """scan_directory 只返回 frontmatter 格式文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # frontmatter 格式
            fm_file = Path(tmpdir) / "fm-skill.md"
            fm_file.write_text(self.SAMPLE_FRONTMATTER_MD, encoding="utf-8")

            # 区块格式（非 frontmatter）
            block_file = Path(tmpdir) / "block-skill.md"
            block_file.write_text(
                "# Skill: block-skill\n\n## 指令\n执行任务\n", encoding="utf-8"
            )

            results = YamlSkillLoader.scan_directory(tmpdir)
            assert len(results) == 1
            assert results[0]["name"] == "browser-act"


class TestExternalSkillsIntegration:
    """集成测试：从外部路径加载 Skills"""

    def setup_method(self):
        SkillRegistry._instance = None
        global_registry.clear()

    def test_load_browser_act_from_claude_skills(self):
        """从 C:\\Users\\hspcadmin\\.claude\\skills 加载 browser-act Skill"""
        skills_path = Path(r"C:\Users\hspcadmin\.claude\skills")

        if not skills_path.exists():
            pytest.skip(f"路径不存在: {skills_path}")

        # 检查 browser-act 目录是否存在
        browser_act_dir = skills_path / "browser-act"
        if not browser_act_dir.exists():
            pytest.skip(f"browser-act 目录不存在: {browser_act_dir}")

        # 使用 Registry 直接扫描
        registry = SkillRegistry()
        count = registry.discover_markdown(browser_act_dir)

        assert count >= 1, f"未从 {browser_act_dir} 加载到任何 Skill"

        skill = registry.get("browser-act")
        assert skill is not None, "未找到 browser-act Skill"
        assert skill.metadata.name == "browser-act"
        assert "Browser" in skill.metadata.description or "browser" in skill.metadata.description.lower()

        # 验证可以执行（返回指令）
        result = skill.execute({}, skill_name="browser-act")
        assert result["success"] is True
        assert "browser-act" in result["data"]["instructions"] or "browser" in result["data"]["instructions"].lower()

    def test_browser_act_via_skilltool(self):
        """通过 SkillTool 调用 browser-act Skill"""
        skills_path = Path(r"C:\Users\hspcadmin\.claude\skills")

        if not skills_path.exists():
            pytest.skip(f"路径不存在: {skills_path}")

        browser_act_dir = skills_path / "browser-act"
        if not browser_act_dir.exists():
            pytest.skip(f"browser-act 目录不存在: {browser_act_dir}")

        # 初始化并注册
        registry = SkillRegistry()
        registry.discover_markdown(browser_act_dir)

        skill_tool = SkillTool(skill_registry=registry)
        global_registry.register_tool(skill_tool)

        # 通过 ToolRegistry 调用
        result = global_registry.execute_tool(
            "skill",
            json.dumps({"skill_name": "browser-act", "args": {}}),
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert "browser-act" in result_data["data"]["skill_name"]
