"""WorkingMemory 的单元测试。

覆盖范围：
- 基础 CRUD（add/retrieve/update/remove/clear/has_memory）
- 容量上限、token 上限、TTL 过期
- 三种遗忘策略（importance / time / capacity）
- 检索打分（关键词匹配、时间衰减、重要性加权）
- 统计与上下文摘要
- 边界场景（空记忆、query 不命中、retrieve 不依赖 TF-IDF）

不依赖任何外部服务（sklearn 仅在 retrieve 路径里被 try/except 吞掉）。
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from agent_runtime.memory.base import MemoryConfig, MemoryItem
from agent_runtime.memory.types.working import WorkingMemory


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def cfg() -> MemoryConfig:
    return MemoryConfig(
        working_memory_capacity=5,
        working_memory_tokens=20,
        working_memory_ttl_minutes=120,
        decay_factor=0.95,
    )


@pytest.fixture
def wm(cfg) -> WorkingMemory:
    return WorkingMemory(config=cfg)


def make_item(content: str, importance: float = 0.5, user_id: str = "u1",
              memory_type: str = "working", timestamp: datetime = None) -> MemoryItem:
    return MemoryItem(
        id=str(uuid4()),
        content=content,
        memory_type=memory_type,
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
    )


# ============================================================
# 1. 基础 add / has_memory / clear
# ============================================================

def test_add_returns_item_id_and_stores_in_list(wm):
    item = make_item("hello world")
    eid = wm.add(item)
    assert eid == item.id
    assert len(wm.memories) == 1
    assert wm.memories[0].id == item.id


def test_has_memory_true_for_existing_false_for_missing(wm):
    item = make_item("hi")
    wm.add(item)
    assert wm.has_memory(item.id) is True
    assert wm.has_memory("nonexistent-id") is False


def test_clear_resets_memories_and_tokens(wm):
    wm.add(make_item("a"))
    wm.add(make_item("b"))
    wm.clear()
    assert wm.memories == []
    assert wm.current_tokens == 0


def test_add_items_with_identical_timestamp_does_not_crash(wm):
    """优先级与时间戳都相同时，堆仍能正常写入（回归：此前会 TypeError）。"""
    ts = datetime.now()
    wm.add(make_item("a", timestamp=ts))
    wm.add(make_item("b", timestamp=ts))   # 曾在此抛 TypeError: < not supported
    wm.add(make_item("c", timestamp=ts))
    assert len(wm.memories) == 3


def test_get_all_returns_copy_not_reference(wm):
    item = make_item("x")
    wm.add(item)
    snap = wm.get_all()
    snap.clear()  # 改快照不该影响内部列表
    assert len(wm.get_all()) == 1


# ============================================================
# 2. update
# ============================================================

def test_update_content_recomputes_tokens(wm):
    item = make_item("one two three")  # 3 tokens
    wm.add(item)
    before = wm.current_tokens

    wm.update(item.id, content="one")  # 1 token

    assert wm.current_tokens == before - 2
    assert wm.memories[0].content == "one"


def test_update_importance_updates_in_place(wm):
    item = make_item("x", importance=0.3)
    wm.add(item)
    wm.update(item.id, importance=0.9)
    assert wm.memories[0].importance == 0.9


def test_update_metadata_merges_not_replaces(wm):
    item = make_item("x", importance=0.0)
    item.metadata = {"a": 1, "b": 2}
    wm.add(item)

    wm.update(item.id, metadata={"b": 99, "c": 3})
    assert wm.memories[0].metadata == {"a": 1, "b": 99, "c": 3}


def test_update_returns_false_for_unknown_id(wm):
    assert wm.update("ghost-id", content="x") is False


# ============================================================
# 3. remove
# ============================================================

def test_remove_returns_true_and_decrements_tokens(wm):
    item = make_item("one two")
    wm.add(item)
    tokens_before = wm.current_tokens
    assert wm.remove(item.id) is True
    assert wm.current_tokens == tokens_before - 2
    assert wm.has_memory(item.id) is False


def test_remove_returns_false_for_unknown_id(wm):
    assert wm.remove("ghost") is False


def test_remove_does_not_corrupt_remaining_items(wm):
    ids = [wm.add(make_item(f"item{i}")) for i in range(3)]
    wm.remove(ids[1])
    remaining_ids = {m.id for m in wm.memories}
    assert remaining_ids == {ids[0], ids[2]}


# ============================================================
# 4. 容量上限 / token 上限 / TTL
# ============================================================

def test_capacity_limit_evicts_lowest_priority(wm):
    """max_capacity=5，加 6 条应驱逐最早/最低优先级那条。"""
    items = [make_item(f"m{i}", importance=0.5) for i in range(5)]
    for it in items:
        wm.add(it)
    last = make_item("m5", importance=0.99)
    wm.add(last)

    assert len(wm.memories) == 5
    assert wm.has_memory(last.id) is True
    # 第一条被驱逐（最早且重要性最低之一）
    assert wm.has_memory(items[0].id) is False


def test_high_importance_survives_eviction(wm):
    """高重要性的早期记忆是否留下来——验证驱逐按优先级而非 FIFO。"""
    survivor = make_item("important early", importance=0.99)
    wm.add(survivor)
    for i in range(5):
        wm.add(make_item(f"filler{i}", importance=0.3))
    assert wm.has_memory(survivor.id) is True


def test_token_limit_evicts_when_total_too_high(wm):
    """max_tokens=20，每个 make_item 默认 1 token，所以加 21 个才触发。"""
    # 先把 max_tokens 调低以便快速测
    wm.max_tokens = 4
    items = [make_item(f"a{i}") for i in range(5)]  # 每个 1 token
    for it in items:
        wm.add(it)
    # 总 token = 5，超过 4；驱逐应发生
    assert wm.current_tokens <= wm.max_tokens


def test_ttl_eviction_drops_expired_items_on_next_add(wm):
    """TTL 清理是惰性的：add 时清理「已有」条目（先于插入新条目）。"""
    item = make_item("old")
    wm.add(item)                                   # 先放入（当时是新鲜的）
    item.timestamp = datetime.now() - timedelta(hours=10)   # 模拟时间流逝
    wm.add(make_item("trigger"))                   # 触发惰性清理
    assert wm.has_memory(item.id) is False


def test_expire_old_memories_updates_token_counter(wm):
    item = make_item("one two three")   # 3 tokens
    wm.add(item)
    assert wm.current_tokens == 3
    item.timestamp = datetime.now() - timedelta(hours=10)
    wm._expire_old_memories()           # 直接触发清理
    assert wm.has_memory(item.id) is False
    assert wm.current_tokens == 0


# ============================================================
# 5. retrieve 检索打分
# ============================================================

def test_retrieve_empty_memories_returns_empty(wm):
    assert wm.retrieve(query="anything") == []


def test_retrieve_exact_substring_match_ranks_first(wm):
    wm.add(make_item("上海迪士尼乐园很好玩"))
    wm.add(make_item("北京故宫博物院"))
    wm.add(make_item("杭州西湖"))

    results = wm.retrieve(query="上海", limit=3)
    assert len(results) >= 1
    assert "上海" in results[0].content


def test_retrieve_keyword_overlap_returns_relevant(wm):
    wm.add(make_item("python 编程 语言 入门 教程"))
    wm.add(make_item("java 编程 语言 入门"))
    wm.add(make_item("今天天气晴朗"))

    results = wm.retrieve(query="python 编程", limit=3)
    assert results[0].content.startswith("python")


def test_retrieve_user_id_filter_excludes_other_users(wm):
    wm.add(make_item("用户A的隐私", user_id="alice"))
    wm.add(make_item("用户B的隐私", user_id="bob"))
    results = wm.retrieve(query="隐私", limit=5, user_id="alice")
    assert all(r.user_id == "alice" for r in results)
    assert any("用户A" in r.content for r in results)


def test_retrieve_higher_importance_ranks_higher_for_same_relevance(wm):
    wm.add(make_item("关键词A 关键词B", importance=0.2))
    wm.add(make_item("关键词A 关键词B", importance=0.9))
    results = wm.retrieve(query="关键词A 关键词B", limit=2)
    assert results[0].importance == 0.9
    assert results[1].importance == 0.2


def test_retrieve_respects_limit(wm):
    for i in range(10):
        wm.add(make_item(f"内容{i}"))
    assert len(wm.retrieve(query="内容", limit=3)) == 3


def test_retrieve_falls_back_to_keyword_when_tfidf_missing(monkeypatch):
    """即使 sklearn 不可用，retrieve 也不能崩，应走降级路径。"""
    # 模拟 sklearn 导入失败
    import sys
    monkeypatch.setitem(sys.modules, "sklearn.feature_extraction.text", None)
    monkeypatch.setitem(sys.modules, "sklearn.metrics.pairwise", None)
    # numpy 可能被 sklearn 间接引入；保留它让 retrieve 的 except 真的吞到 sklearn 异常

    cfg = MemoryConfig(working_memory_capacity=10)
    w = WorkingMemory(config=cfg)
    w.add(make_item("北京 上海 旅游"))

    results = w.retrieve(query="北京", limit=3)
    assert len(results) >= 1
    assert "北京" in results[0].content


# ============================================================
# 6. forget 三策略
# ============================================================

def test_forget_importance_based_drops_low_importance(wm):
    wm.add(make_item("low", importance=0.05))
    wm.add(make_item("mid", importance=0.5))
    wm.add(make_item("high", importance=0.95))

    forgotten = wm.forget(strategy="importance_based", threshold=0.1)
    assert forgotten == 1
    assert not wm.has_memory(wm.memories[0].id) or wm.memories[0].importance >= 0.1
    assert wm.current_tokens >= 0  # 不能减成负数


def test_forget_time_based_drops_old_items(wm):
    """构造一个 TTL 不算过期（2h 内）但 time-based 策略算过期（>1 天）的记忆。

    默认 TTL=120min；通过把 timestamp 设为 3h 前：TTL 不会清（<2h 限制其实是
    >120min 才清——hmm a 误读）——这里重新核对：
    - cutoff_ttl = now - 120min
    - 若 timestamp < cutoff_ttl 即视为过期
    - 3h 前 < 2h 前 → 被 TTL 清掉
    解决方法：把 TTL 调长，或用更短的时间偏移。

    用更简单的办法：max_age_minutes 设很大让 TTL 失效，再用 time_based 测。
    """
    wm.max_age_minutes = 60 * 24 * 30  # 30 天 TTL，基本不触发

    old_item = make_item("old")
    wm.add(old_item)
    old_item.timestamp = datetime.now() - timedelta(days=10)  # 改时间戳，TTL 不会清

    new = make_item("new")
    wm.add(new)

    # 确认两件记忆都在
    assert len(wm.memories) == 2

    forgotten = wm.forget(strategy="time_based", max_age_days=1)
    assert forgotten == 1
    assert wm.has_memory(new.id) is True


def test_forget_capacity_based_trims_to_max(wm):
    for i in range(5):
        wm.add(make_item(f"m{i}", importance=0.5))
    assert len(wm.memories) == 5

    wm.max_capacity = 3  # 模拟容量收紧到 3
    forgotten = wm.forget(strategy="capacity_based")

    assert forgotten == 2
    assert len(wm.memories) == 3


# ============================================================
# 7. 优先级计算 / 时间衰减
# ============================================================

def test_calculate_priority_is_importance_times_time_decay(wm):
    item = make_item("x", importance=0.5)
    item.timestamp = datetime.now()
    expected = 0.5 * wm._calculate_time_decay(item.timestamp)
    assert abs(wm._calculate_priority(item) - expected) < 1e-6


def test_calculate_time_decay_clamps_to_minimum(wm):
    """非常老的记忆时间衰减被 clamp 到 0.1，不会归零。"""
    very_old = datetime.now() - timedelta(days=365 * 10)
    assert wm._calculate_time_decay(very_old) == 0.1


def test_calculate_time_decay_decreases_with_age(wm):
    fresh = datetime.now()
    older = datetime.now() - timedelta(hours=12)
    assert wm._calculate_time_decay(fresh) > wm._calculate_time_decay(older)


# ============================================================
# 8. get_stats / get_context_summary / get_recent / get_important
# ============================================================

def test_get_stats_returns_expected_keys(wm):
    wm.add(make_item("x", importance=0.5))
    stats = wm.get_stats()
    assert stats["count"] == 1
    assert stats["total_count"] == 1
    assert stats["max_capacity"] == 5
    assert stats["max_tokens"] == 20
    assert stats["memory_type"] == "working"
    assert stats["avg_importance"] == 0.5
    assert "session_duration_minutes" in stats


def test_get_stats_on_empty_returns_zero_avg(wm):
    stats = wm.get_stats()
    assert stats["count"] == 0
    assert stats["avg_importance"] == 0.0
    assert stats["capacity_usage"] == 0.0


def test_get_context_summary_truncates_at_max_length(wm):
    for i in range(10):
        wm.add(make_item("a" * 100))  # 每条 100 字符
    summary = wm.get_context_summary(max_length=150)
    body = summary.replace("Working Memory Context:\n", "")
    assert len(body) <= 150 + 3  # 留出 "..." 余地


def test_get_context_summary_returns_placeholder_when_empty(wm):
    assert "No working memories" in wm.get_context_summary()


def test_get_recent_returns_newest_first(wm):
    first = make_item("first")
    wm.add(first)
    second = make_item("second")
    # 手动让 second 的 timestamp 晚一些，保证排序稳定
    second.timestamp = first.timestamp + timedelta(seconds=1)
    wm.add(second)

    recent = wm.get_recent(limit=5)
    assert recent[0].content == "second"


def test_get_important_returns_highest_first(wm):
    wm.add(make_item("low", importance=0.1))
    wm.add(make_item("high", importance=0.9))
    wm.add(make_item("mid", importance=0.5))

    important = wm.get_important(limit=3)
    assert important[0].importance == 0.9
    assert important[-1].importance == 0.1


# ============================================================
# 9. 边界 / 防御
# ============================================================

def test_add_same_id_twice_does_not_dedupe(wm):
    """add 当前不阻止同 id 重复添加，记录下以便将来修复时改成断言。"""
    item = make_item("dup")
    wm.add(item)
    wm.add(item)
    # 当前实现：重复添加后列表里有两条；这是已知行为
    same_id_count = sum(1 for m in wm.memories if m.id == item.id)
    assert same_id_count == 2


def test_remove_after_clear_does_not_corrupt_state(wm):
    item = make_item("x")
    wm.add(item)
    wm.clear()
    assert wm.remove(item.id) is False
    assert wm.current_tokens == 0


def test_retrieve_with_no_match_returns_empty(wm):
    wm.add(make_item("北京"))
    results = wm.retrieve(query="完全不相关的查询xyz123", limit=3)
    assert results == []