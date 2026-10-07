"""Memory 子系统的单元测试。

按被测类分块：
1. MemoryConfig.from_env() — 默认值 / env 覆盖 / 解析
2. MemoryManager — 编排与默认行为
3. MemoryTool — 工具封装
4. 集成：MemoryManager / MemoryTool 接受 config=None 时走 from_env
"""

from datetime import datetime

import pytest

from agent_runtime.memory.base import MemoryConfig, MemoryItem
from agent_runtime.memory.manager import MemoryManager
from agent_runtime.tools.base import ToolParameter
from agent_runtime.tools.builtin.memory_tool import MemoryTool


ALL_ENV_VARS = [
    "MEMORY_STORAGE_PATH",
    "MEMORY_DISABLE_VECTOR_STORE",
    "MEMORY_DISABLE_EMBEDDINGS",
    "MEMORY_MAX_CAPACITY",
    "MEMORY_IMPORTANCE_THRESHOLD",
    "MEMORY_DECAY_FACTOR",
    "MEMORY_WORKING_CAPACITY",
    "MEMORY_WORKING_TOKENS",
    "MEMORY_WORKING_TTL_MINUTES",
    "MEMORY_PERCEPTUAL_MODALITIES",
]


@pytest.fixture
def clean_env(monkeypatch):
    """清空所有 MEMORY_* 环境变量，保证默认值测试不受外部环境影响。"""
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def manager() -> MemoryManager:
    return MemoryManager(
        config=MemoryConfig(),
        user_id="test_user",
        enable_working=True,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=False,
    )


@pytest.fixture
def tool() -> MemoryTool:
    return MemoryTool(user_id="test_user", memory_types=["working"])


# ============================================================
# 1. MemoryConfig.from_env() — 默认值
# ============================================================

def test_from_env_falls_back_to_field_defaults(clean_env):
    cfg = MemoryConfig.from_env()
    assert cfg == MemoryConfig()


def test_from_env_default_values_are_documented_ones(clean_env):
    cfg = MemoryConfig.from_env()
    assert cfg.storage_path == "./memory_data"
    assert cfg.disable_vector_store is False
    assert cfg.disable_embeddings is False
    assert cfg.max_capacity == 100
    assert cfg.importance_threshold == 0.1
    assert cfg.decay_factor == 0.95
    assert cfg.working_memory_capacity == 10
    assert cfg.working_memory_tokens == 2000
    assert cfg.working_memory_ttl_minutes == 120
    assert cfg.perceptual_memory_modalities == ["text", "image", "audio", "video"]


# ============================================================
# 2. MemoryConfig.from_env() — 逐项覆盖
# ============================================================

def test_from_env_overrides_scalars(clean_env):
    clean_env.setenv("MEMORY_STORAGE_PATH", "/tmp/mem")
    clean_env.setenv("MEMORY_MAX_CAPACITY", "7")
    clean_env.setenv("MEMORY_IMPORTANCE_THRESHOLD", "0.42")
    clean_env.setenv("MEMORY_DECAY_FACTOR", "0.8")
    clean_env.setenv("MEMORY_WORKING_CAPACITY", "3")
    clean_env.setenv("MEMORY_WORKING_TOKENS", "500")
    clean_env.setenv("MEMORY_WORKING_TTL_MINUTES", "30")

    cfg = MemoryConfig.from_env()
    assert cfg.storage_path == "/tmp/mem"
    assert cfg.max_capacity == 7
    assert cfg.importance_threshold == 0.42
    assert cfg.decay_factor == 0.8
    assert cfg.working_memory_capacity == 3
    assert cfg.working_memory_tokens == 500
    assert cfg.working_memory_ttl_minutes == 30


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("True", True), ("TRUE", True),
    ("yes", True), ("on", True), (" true ", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("random", False),
])
def test_from_env_bool_parsing(clean_env, raw, expected):
    clean_env.setenv("MEMORY_DISABLE_VECTOR_STORE", raw)
    assert MemoryConfig.from_env().disable_vector_store is expected


def test_from_env_bool_empty_string_falls_back_to_default(clean_env):
    clean_env.setenv("MEMORY_DISABLE_EMBEDDINGS", "")
    assert MemoryConfig.from_env().disable_embeddings is False


def test_from_env_bool_env_enables_disable_switch(clean_env):
    clean_env.setenv("MEMORY_DISABLE_EMBEDDINGS", "true")
    clean_env.setenv("MEMORY_DISABLE_VECTOR_STORE", "yes")
    cfg = MemoryConfig.from_env()
    assert cfg.disable_embeddings is True
    assert cfg.disable_vector_store is True


# ============================================================
# 3. MemoryConfig.from_env() — modalities 列表解析
# ============================================================

def test_from_env_parses_comma_separated_modalities(clean_env):
    clean_env.setenv("MEMORY_PERCEPTUAL_MODALITIES", "text,image")
    assert MemoryConfig.from_env().perceptual_memory_modalities == ["text", "image"]


def test_from_env_strips_whitespace_in_modalities(clean_env):
    clean_env.setenv("MEMORY_PERCEPTUAL_MODALITIES", " text , image , audio ")
    assert MemoryConfig.from_env().perceptual_memory_modalities == ["text", "image", "audio"]


def test_from_env_drops_empty_modality_segments(clean_env):
    clean_env.setenv("MEMORY_PERCEPTUAL_MODALITIES", "text,,image,")
    assert MemoryConfig.from_env().perceptual_memory_modalities == ["text", "image"]


def test_from_env_empty_modalities_string_yields_empty_list(clean_env):
    clean_env.setenv("MEMORY_PERCEPTUAL_MODALITIES", "")
    assert MemoryConfig.from_env().perceptual_memory_modalities == []


def test_from_env_unset_modalities_returns_independent_copy(clean_env):
    """返回的默认列表不能与字段默认共享引用，否则调用方修改会污染全局。"""
    cfg = MemoryConfig.from_env()
    cfg.perceptual_memory_modalities.append("video")
    assert MemoryConfig().perceptual_memory_modalities == ["text", "image", "audio", "video"]


# ============================================================
# 4. MemoryConfig.from_env() — 非法值
# ============================================================

def test_from_env_invalid_int_raises(clean_env):
    clean_env.setenv("MEMORY_MAX_CAPACITY", "not-a-number")
    with pytest.raises(ValueError):
        MemoryConfig.from_env()


def test_from_env_invalid_float_raises(clean_env):
    clean_env.setenv("MEMORY_DECAY_FACTOR", "abc")
    with pytest.raises(ValueError):
        MemoryConfig.from_env()


# ============================================================
# 5. MemoryManager — 编排与默认行为
#
# 注释：这些断言与 test_working_memory.py 中的同名方法语义重叠，但视角不同：
# 此处是「从 Manager 入口触发（manager.add_memory → WorkingMemory.add）」，
# test_working_memory.py 是「WorkingMemory 类直接行为」。两个入口点都值得守住。
# ============================================================

def test_manager_only_enables_working(manager):
    assert set(manager.memory_types.keys()) == {"working"}


def test_add_memory_with_explicit_type_returns_id(manager):
    mid = manager.add_memory(
        content="hello",
        memory_type="working",
        importance=0.6,
        auto_classify=False,
    )
    assert isinstance(mid, str) and len(mid) > 0


def test_add_memory_unsupported_type_raises(manager):
    with pytest.raises(ValueError, match="perceptual"):
        manager.add_memory(content="x", memory_type="perceptual", auto_classify=False)


def test_auto_classify_falls_back_to_working_when_no_keyword(manager):
    manager.add_memory(content="一段普通对话内容", auto_classify=True)
    stats = manager.get_memory_stats()
    assert stats["memories_by_type"]["working"]["count"] >= 1


def test_importance_calculation_boosts_for_important_keywords(manager):
    low = manager.add_memory(content="普通信息", auto_classify=False)
    high = manager.add_memory(content="重要：必须立即处理警告", auto_classify=False)
    items = {m.id: m for m in manager.memory_types["working"].get_all()}
    assert items[high].importance > items[low].importance


def test_retrieve_memories_returns_sorted_by_importance(manager):
    manager.add_memory(content="低重要性内容", memory_type="working", importance=0.2, auto_classify=False)
    manager.add_memory(content="高重要性内容", memory_type="working", importance=0.9, auto_classify=False)

    results = manager.retrieve_memories(query="内容", limit=10, min_importance=0.0)
    assert len(results) >= 2
    assert results[0].importance >= results[-1].importance


def test_remove_memory_returns_true_for_existing(manager):
    mid = manager.add_memory(content="to remove", memory_type="working", auto_classify=False)
    assert manager.remove_memory(mid) is True
    assert manager.remove_memory(mid) is False


def test_clear_all_memories_resets_count(manager):
    manager.add_memory(content="a", memory_type="working", auto_classify=False)
    manager.clear_all_memories()
    stats = manager.get_memory_stats()
    assert stats["total_memories"] == 0


# ============================================================
# 6. MemoryConfig / MemoryItem — 字段有效性
# ============================================================

def test_memory_config_defaults_are_valid():
    cfg = MemoryConfig()
    assert cfg.max_capacity > 0
    assert 0.0 <= cfg.importance_threshold <= 1.0
    assert 0.0 < cfg.decay_factor <= 1.0


def test_memory_item_requires_mandatory_fields():
    with pytest.raises(Exception):
        MemoryItem(
            id="x",
            memory_type="working",
            user_id="u",
            timestamp=datetime.now(),
        )


# ============================================================
# 7. MemoryTool — 工具封装
# ============================================================

def test_tool_inherits_base(tool):
    assert tool.name == "memory"
    assert tool.description


def test_get_parameters_returns_required_action(tool):
    params = tool.get_parameters()
    assert isinstance(params, list) and len(params) > 0
    assert isinstance(params[0], ToolParameter)
    action = next(p for p in params if p.name == "action")
    assert action.required is True


def test_validate_parameters_flags_missing_action(tool):
    assert tool.validate_parameters({"content": "x"}) is False
    assert tool.validate_parameters({"action": "add", "content": "x"}) is True


def test_run_returns_error_on_missing_action(tool):
    out = tool.run({"content": "no action"})
    assert "参数验证失败" in out


def test_add_action_returns_success_message(tool):
    out = tool.run({
        "action": "add", "content": "我去了上海",
        "memory_type": "working", "importance": 0.5,
    })
    assert out.startswith("✅")


def test_search_returns_formatted_results(tool):
    tool.run({
        "action": "add", "content": "北京故宫博物院是著名景点",
        "memory_type": "working", "importance": 0.9,
    })
    out = tool.run({"action": "search", "query": "故宫", "limit": 3})
    # 即便没匹配到，也要返回搜索结果格式而非抛异常
    assert out.startswith("🔍")


def test_stats_action_returns_tool_info(tool):
    out = tool.run({"action": "stats"})
    assert "记忆系统统计" in out
    assert "test_user" not in out  # 不应泄漏内部标识到工具输出


def test_unknown_action_returns_helpful_error(tool):
    out = tool.run({"action": "explode"})
    assert "不支持的操作" in out
    assert "add" in out


# ============================================================
# 8. 集成：MemoryManager / MemoryTool 接受 config=None 时走 from_env
# ============================================================

def test_manager_uses_from_env_when_config_is_none(clean_env):
    clean_env.setenv("MEMORY_WORKING_CAPACITY", "4")
    clean_env.setenv("MEMORY_WORKING_TOKENS", "800")

    manager = MemoryManager(
        config=None,
        enable_working=True,
        enable_episodic=False,
        enable_semantic=False,
        enable_perceptual=False,
    )
    assert manager.config.working_memory_capacity == 4
    # 配置应真正传导到 WorkingMemory
    assert manager.memory_types["working"].max_capacity == 4


def test_manager_explicit_config_overrides_env(clean_env):
    clean_env.setenv("MEMORY_WORKING_CAPACITY", "4")
    manager = MemoryManager(
        config=MemoryConfig(working_memory_capacity=99),
        enable_working=True,
        enable_episodic=False,
        enable_semantic=False,
    )
    assert manager.config.working_memory_capacity == 99


def test_memory_tool_uses_from_env_when_config_is_none(clean_env):
    clean_env.setenv("MEMORY_WORKING_CAPACITY", "6")
    tool = MemoryTool(memory_types=["working"])
    assert tool.memory_config.working_memory_capacity == 6