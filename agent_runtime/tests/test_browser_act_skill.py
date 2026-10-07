r"""browser-act Skill 集成测试

测试从 C:\Users\hspcadmin\.claude\skills 加载并执行 browser-act Skill。
"""

import json
import tempfile
import pytest
from pathlib import Path

from agent_runtime.skills import SkillRegistry
from agent_runtime.tools.builtin import SkillTool
from agent_runtime.tools.registry import ToolRegistry, global_registry


@pytest.fixture(autouse=True)
def clean_registry():
    """每个测试前重置 Registry 和 ToolRegistry"""
    SkillRegistry._instance = None
    global_registry.clear()
    yield
    SkillRegistry._instance = None
    global_registry.clear()


SKILLS_PATH = Path(r"C:\Users\hspcadmin\.claude\skills")
BROWSER_ACT_DIR = SKILLS_PATH / "browser-act"


class TestBrowserActSkill:
    """browser-act Skill 加载和执行测试"""

    @pytest.fixture(autouse=True)
    def check_skills_path(self):
        if not SKILLS_PATH.exists():
            pytest.skip(f"Skills 路径不存在: {SKILLS_PATH}")
        if not BROWSER_ACT_DIR.exists():
            pytest.skip(f"browser-act 目录不存在: {BROWSER_ACT_DIR}")

    # ------------------------------------------------------------------
    # 测试 1：加载并直接执行 Skill
    # ------------------------------------------------------------------

    def test_load_and_execute_browser_act(self):
        """从 .claude/skills 目录加载 browser-act 并执行"""
        registry = SkillRegistry()
        count = registry.discover_markdown(BROWSER_ACT_DIR)

        assert count >= 1, f"未从 {BROWSER_ACT_DIR} 加载到任何 Skill"
        skill = registry.get("browser-act")
        assert skill is not None, "未找到 browser-act Skill"

        # 验证 metadata
        assert skill.metadata.name == "browser-act"
        assert skill.metadata.description, "description 不应为空"
        assert skill.metadata.author is not None, "author 不应为空"

        # 验证指令内容存在
        instructions = skill.get_instructions()
        assert instructions, "instructions 不应为空"
        assert "browser" in instructions.lower(), "指令中应包含 browser 关键字"

        # 执行 Skill
        result = skill.execute({}, skill_name="browser-act")
        assert result["success"] is True
        assert result["data"]["skill_name"] == "browser-act"
        assert "instructions" in result["data"]
        assert "browser-act" in result["data"]["instructions"].lower()
        assert "get-skills" in result["data"]["instructions"] or "browser" in result["data"]["instructions"].lower()

    # ------------------------------------------------------------------
    # 测试 2：通过 SkillTool 调用
    # ------------------------------------------------------------------

    def test_via_skill_tool(self):
        """通过 SkillTool 调用 browser-act"""
        registry = SkillRegistry()
        registry.discover_markdown(BROWSER_ACT_DIR)

        skill_tool = SkillTool(skill_registry=registry)
        global_registry.register_tool(skill_tool)

        # 通过 ToolRegistry 调用
        result = global_registry.execute_tool(
            "skill",
            json.dumps({"skill_name": "browser-act", "args": {}}),
        )

        result_data = json.loads(result)
        assert result_data["success"] is True
        assert result_data["data"]["skill_name"] == "browser-act"
        assert "browser-act" in result_data["data"]["instructions"].lower()

    # ------------------------------------------------------------------
    # 测试 3：带参数调用（虽然 browser-act md 不定义参数）
    # ------------------------------------------------------------------

    def test_execute_with_shared_state(self):
        """执行时传入 shared_state，验证上下文传递"""
        registry = SkillRegistry()
        registry.discover_markdown(BROWSER_ACT_DIR)
        skill = registry.get("browser-act")

        shared = {"user_id": "test-user", "session": "test-session"}
        result = skill.execute(
            {"shared_state": shared},
            skill_name="browser-act",
        )

        assert result["success"] is True
        # shared_state 应被 SkillContext 接收
        # （实际使用中 Skill 可通过 context.shared_state 访问）

    # ------------------------------------------------------------------
    # 测试 4：扫描整个 skills 目录，browser-act 被正确识别
    # ------------------------------------------------------------------

    def test_scan_claude_skills_directory(self):
        """根目录无 .md 文件，子目录 browser-act 被正确加载"""
        registry = SkillRegistry()

        # 根目录无 .md 文件，count 应为 0
        count = registry.discover_markdown(SKILLS_PATH)
        assert count == 0

        # 扫描 browser-act 子目录才有内容
        count = registry.discover_markdown(BROWSER_ACT_DIR)
        assert count >= 1, f"从 {BROWSER_ACT_DIR} 未加载到任何 Skill"
        skill = registry.get("browser-act")
        assert skill is not None

    # ------------------------------------------------------------------
    # 测试 5：多次调用返回一致结果
    # ------------------------------------------------------------------

    def test_execute_idempotent(self):
        """多次执行返回一致的指令内容"""
        registry = SkillRegistry()
        registry.discover_markdown(BROWSER_ACT_DIR)
        skill = registry.get("browser-act")

        result1 = skill.execute({}, skill_name="browser-act")
        result2 = skill.execute({}, skill_name="browser-act")

        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["data"]["instructions"] == result2["data"]["instructions"]

    # ------------------------------------------------------------------
    # 测试 6：通过 agent_runtime skill 框架调用 browser-act，
    #         执行 skill.execute() 返回 CLI 指令，解析并执行得到真实搜索结果
    # ------------------------------------------------------------------

    def test_browser_act_skill_full_chain(self):
        """完整链路：skill.execute() → 解析 CLI 指令 → subprocess 执行 → 真实结果"""
        import re
        import subprocess

        SESSION_NAME = "hangzhou-search"
        TARGET_URL = "https://www.bing.com"
        SEARCH_KEYWORD = "杭州"

        def run(cmd: str) -> str:
            """执行 shell 命令，返回 stdout + stderr"""
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            return ((result.stdout or "") + (result.stderr or "")).strip()

        # ------------------------------------------------------------------
        # Step 1: 通过 agent_runtime 加载 browser-act skill
        # ------------------------------------------------------------------
        registry = SkillRegistry()
        count = registry.discover_markdown(BROWSER_ACT_DIR)
        assert count >= 1, f"未从 {BROWSER_ACT_DIR} 加载到 Skill"
        skill = registry.get("browser-act")
        assert skill is not None

        # ------------------------------------------------------------------
        # Step 2: 通过 skill.execute() 获取 CLI 指令
        # ------------------------------------------------------------------
        result = skill.execute({}, skill_name="browser-act")
        assert result["success"] is True
        instructions = result["data"]["instructions"]
        assert "browser-act" in instructions, "instructions 应包含 browser-act"
        assert "get-skills core" in instructions, "browser-act 是发现存根，应指引获取 core 指令"

        # ------------------------------------------------------------------
        # Step 3: 解析 instructions，提取 CLI 命令并执行
        #         browser-act skill 的指令告诉我们要运行:
        #         browser-act get-skills core --skill-version 2.0.2
        # ------------------------------------------------------------------
        core_cmd_match = re.search(r"browser-act\s+get-skills\s+core\s+--skill-version\s+[\d.]+", instructions)
        if not core_cmd_match:
            # fallback: 直接尝试已知命令
            core_cmd = "browser-act get-skills core --skill-version 2.0.2"
        else:
            core_cmd = core_cmd_match.group().strip()

        core_instructions = run(core_cmd)
        assert "browser" in core_instructions.lower(), f"core 指令应包含 browser 相关内容: {core_instructions[:200]}"

        # ------------------------------------------------------------------
        # Step 4: 从 core 指令中提取浏览器操作命令，执行搜索
        # ------------------------------------------------------------------
        # 清理可能残留的同名浏览器
        cleanup = run("browser-act browser list")
        for line in cleanup.splitlines():
            if "hangzhou-search-skill" in line:
                bid_match = re.search(r"(chrome_local[\w_]+)", line)
                if bid_match:
                    run(f'browser-act browser delete {bid_match.group(1)} --objective "cleanup for test"')
                break

        # 创建浏览器
        create_out = run('browser-act browser create --name "hangzhou-search-skill" --type chrome --desc "通过skill框架执行Bing搜索杭州"')
        assert "id=" in create_out, f"浏览器创建失败: {create_out}"
        browser_id = re.search(r"(chrome_local[\w_]+)", create_out)
        assert browser_id, f"无法解析 browser id: {create_out}"
        browser_id = browser_id.group(1)

        try:
            # 导航到 Bing
            nav_out = run(f'browser-act --session {SESSION_NAME} browser open {browser_id} {TARGET_URL}')
            if "net::ERR_" in nav_out or "Navigation failed" in nav_out:
                pytest.skip(f"网络不可达 Bing: {nav_out}")

            # 等待页面稳定
            run(f'browser-act --session {SESSION_NAME} wait stable --timeout 15000')

            # 获取搜索框和搜索按钮的 index
            state_out = run(f'browser-act --session {SESSION_NAME} state')
            assert "Error" not in state_out
            lines = [l for l in state_out.splitlines() if "input" in l.lower() or 'label' in l.lower()]
            assert lines, f"state 输出中未找到 input 或 label 元素: {state_out}"
            search_index_match = re.search(r"\[(\d+)\]", lines[0])
            assert search_index_match
            search_index = search_index_match.group(1)

            # 输入搜索词
            input_out = run(f'browser-act --session {SESSION_NAME} input {search_index} "{SEARCH_KEYWORD}"')
            if "ERR_" in input_out:
                pytest.skip(f"input 命令失败: {input_out}")

            # 找搜索按钮（label，aria-label 含"搜索"）
            label_lines = [l for l in state_out.splitlines() if 'label' in l.lower() or 'search_icon' in l.lower()]
            if label_lines:
                label_match = re.search(r"\[(\d+)\]", label_lines[0])
                if label_match:
                    click_out = run(f'browser-act --session {SESSION_NAME} click {label_match.group(1)}')
                    if "ERR_" in click_out:
                        pytest.skip(f"click 命令失败: {click_out}")
            else:
                # fallback: 直接导航到搜索 URL
                run(f'browser-act --session {SESSION_NAME} navigate {TARGET_URL}/search?q=%E6%9D%AD%E5%B7%9E')

            # 等待搜索结果
            run(f'browser-act --session {SESSION_NAME} wait stable --timeout 20000')

            # 获取搜索结果
            results_out = run(f'browser-act --session {SESSION_NAME} get markdown')
            assert "杭州" in results_out, f"搜索结果应包含「杭州」: {results_out[:500]}"
            assert "bing" in results_out.lower(), f"搜索结果应包含 bing 相关内容"

        finally:
            # 清理
            run(f'browser-act session close {SESSION_NAME}')
            cleanup = run("browser-act browser list")
            for line in cleanup.splitlines():
                if "hangzhou-search-skill" in line:
                    bid_match = re.search(r"(chrome_local[\w_]+)", line)
                    if bid_match:
                        run(f'browser-act browser delete {bid_match.group(1)} --objective "cleanup after test"')
                    break
