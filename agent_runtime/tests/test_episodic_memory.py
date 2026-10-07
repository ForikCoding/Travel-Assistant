"""EpisodicMemory 的单元测试。

覆盖范围：
- 顶部：导入回归（守住 storage/__init__.py 重导出）
- Episode 数据结构
- add / retrieve / update / remove / has_memory / clear
- forget 三种策略
- get_all / get_stats / get_session_episodes / get_timeline / find_patterns
- 私有辅助：_filter_episodes / _calculate_time_span

全部走「离线路径」：disable_embeddings=True + disable_vector_store=True，
只依赖 SQLite（stdlib），不触碰 Qdrant / HuggingFace。

已知缺陷以 xfail(strict=True) 记录，见文件末尾「已知缺陷」一节。
"""

import time
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from agent_runtime.memory.base import MemoryConfig, MemoryItem
from agent_runtime.memory.types.episodic import Episode, EpisodicMemory


# ============================================================
# 0. 导入回归（守住 storage/__init__.py 重导出）
# ============================================================

def test_storage_module_re_exports_required_classes():
    """修复前的 bug 是「漏导 SQLiteDocumentStore / QdrantVectorStore 导致
    EpisodicMemory 无法 import」。此 case 守住 storage/__init__.py 的导出契约。"""
    from agent_runtime.memory.storage import (
        QdrantVectorStore,
        SQLiteDocumentStore,
    )
    from agent_runtime.memory.types.episodic import EpisodicMemory
    from agent_runtime.memory.types.perceptual import PerceptualMemory

    assert EpisodicMemory is not None
    assert PerceptualMemory is not None
    assert SQLiteDocumentStore is not None
    assert QdrantVectorStore is not None


# ============================================================
# Fixtures / helpers
# ============================================================

@pytest.fixture
def em(tmp_path) -> EpisodicMemory:
    # 默认离线：disable_embeddings / disable_vector_store 默认为 True，
    # 所以这里只需指定 SQLite 数据目录即可，不连任何外部服务。
    cfg = MemoryConfig(storage_path=str(tmp_path))
    return EpisodicMemory(config=cfg)


def make_item(
    content: str,
    session_id="s1",
    user_id: str = "u1",
    importance: float = 0.5,
    metadata: dict = None,
    timestamp: datetime = None,
) -> MemoryItem:
    md = dict(metadata or {})
    if session_id is not None:
        md.setdefault("session_id", session_id)
    return MemoryItem(
        id=str(uuid4()),
        content=content,
        memory_type="episodic",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
        metadata=md,
    )


# ============================================================
# 1. Episode 数据结构
# ============================================================

def test_episode_stores_all_attributes():
    ts = datetime.now()
    ep = Episode(
        episode_id="e1",
        user_id="u1",
        session_id="s1",
        timestamp=ts,
        content="去了西湖",
        context={"city": "hangzhou"},
        outcome="很好玩",
        importance=0.7,
    )
    assert ep.episode_id == "e1"
    assert ep.user_id == "u1"
    assert ep.session_id == "s1"
    assert ep.timestamp == ts
    assert ep.content == "去了西湖"
    assert ep.context == {"city": "hangzhou"}
    assert ep.outcome == "很好玩"
    assert ep.importance == 0.7


def test_episode_outcome_and_importance_default():
    ep = Episode(
        episode_id="e1", user_id="u1", session_id="s1",
        timestamp=datetime.now(), content="x", context={},
    )
    assert ep.outcome is None
    assert ep.importance == 0.5


# ============================================================
# 2. 初始化（离线）
# ============================================================

def test_init_disables_embedder_and_vector_store(em):
    assert em.embedder is None
    assert em.vector_store is None
    assert em.episodes == []
    assert em.sessions == {}


def test_init_creates_sqlite_file(tmp_path):
    cfg = MemoryConfig(storage_path=str(tmp_path))
    EpisodicMemory(config=cfg)
    assert (tmp_path / "memory.db").exists()


# ============================================================
# 3. add
# ============================================================

def test_add_returns_id_and_caches_in_memory(em):
    item = make_item("昨天去了西湖")
    eid = em.add(item)
    assert eid == item.id
    assert len(em.episodes) == 1
    assert em.episodes[0].episode_id == item.id


def test_add_persists_to_sqlite(em):
    item = make_item("持久化测试")
    em.add(item)
    doc = em.doc_store.get_memory(item.id)
    assert doc is not None
    assert doc["content"] == "持久化测试"
    assert doc["memory_type"] == "episodic"


def test_add_registers_session(em):
    em.add(make_item("a", session_id="s1"))
    em.add(make_item("b", session_id="s1"))
    em.add(make_item("c", session_id="s2"))
    assert len(em.sessions["s1"]) == 2
    assert len(em.sessions["s2"]) == 1


def test_add_without_session_id_uses_default_session(em):
    item = make_item("没有 session", session_id=None)
    em.add(item)
    assert "default_session" in em.sessions
    assert em.episodes[0].session_id == "default_session"


def test_add_stores_extra_metadata_in_sqlite(em):
    item = make_item(
        "上下文丰富",
        metadata={"context": {"city": "beijing"}, "outcome": "成功",
                  "participants": ["alice", "bob"], "tags": ["travel"]},
    )
    em.add(item)
    props = em.doc_store.get_memory(item.id)["properties"]
    assert props["session_id"] == "s1"
    assert props["context"] == {"city": "beijing"}
    assert props["outcome"] == "成功"
    assert props["participants"] == ["alice", "bob"]
    assert props["tags"] == ["travel"]


# ============================================================
# 4. has_memory / remove
# ============================================================

def test_has_memory(em):
    item = make_item("x")
    em.add(item)
    assert em.has_memory(item.id) is True
    assert em.has_memory("ghost") is False


def test_remove_cleans_memory_and_sqlite(em):
    item = make_item("要删除")
    em.add(item)
    assert em.remove(item.id) is True
    assert em.has_memory(item.id) is False
    assert em.doc_store.get_memory(item.id) is None


def test_remove_cleans_empty_session_bucket(em):
    item = make_item("唯一一条", session_id="solo")
    em.add(item)
    em.remove(item.id)
    assert "solo" not in em.sessions


def test_remove_keeps_other_episodes_in_session(em):
    a = make_item("a", session_id="s1")
    b = make_item("b", session_id="s1")
    em.add(a)
    em.add(b)
    em.remove(a.id)
    assert em.sessions["s1"] == [b.id]


def test_remove_unknown_returns_false(em):
    assert em.remove("ghost") is False


# ============================================================
# 5. update
# ============================================================

def test_update_content_updates_memory_and_sqlite(em):
    item = make_item("旧内容")
    em.add(item)
    assert em.update(item.id, content="新内容") is True
    assert em.episodes[0].content == "新内容"
    assert em.doc_store.get_memory(item.id)["content"] == "新内容"


def test_update_importance(em):
    item = make_item("x", importance=0.2)
    em.add(item)
    em.update(item.id, importance=0.9)
    assert em.episodes[0].importance == 0.9
    assert em.doc_store.get_memory(item.id)["importance"] == 0.9


def test_update_metadata_merges_context_in_memory(em):
    item = make_item("x", metadata={"context": {"a": 1}})
    em.add(item)
    em.update(item.id, metadata={"context": {"b": 2}})
    assert em.episodes[0].context == {"a": 1, "b": 2}


def test_update_metadata_sets_outcome(em):
    item = make_item("x")
    em.add(item)
    em.update(item.id, metadata={"outcome": "完成"})
    assert em.episodes[0].outcome == "完成"


def test_update_unknown_returns_false(em):
    assert em.update("ghost", content="x") is False


# ============================================================
# 6. clear
# ============================================================

def test_clear_removes_all_episodic(em):
    em.add(make_item("a"))
    em.add(make_item("b", session_id="s2"))
    em.clear()

    assert em.episodes == []
    assert em.sessions == {}
    assert em.doc_store.search_memories(memory_type="episodic", limit=100) == []


def test_clear_leaves_other_memory_types_untouched(em):
    em.add(make_item("episodic 一条"))
    em.doc_store.add_memory(
        memory_id="w1", user_id="u1", content="working 一条",
        memory_type="working", timestamp=int(time.time()), importance=0.5,
    )
    em.clear()

    assert em.doc_store.get_memory("w1") is not None


# ============================================================
# 7. retrieve（SQLite 回退路径）
# ============================================================

def test_retrieve_finds_by_text(em):
    em.add(make_item("昨天我去了西湖", importance=0.6))
    em.add(make_item("今天在公司写代码"))

    results = em.retrieve(query="西湖", limit=5)
    assert len(results) == 1
    assert "西湖" in results[0].content


def test_retrieve_no_match_returns_empty(em):
    em.add(make_item("北京故宫"))
    assert em.retrieve(query="完全不存在zzz", limit=5) == []


def test_retrieve_empty_query_returns_all(em):
    em.add(make_item("a"))
    em.add(make_item("b"))
    assert len(em.retrieve(query="", limit=10)) == 2


def test_retrieve_respects_limit(em):
    for i in range(6):
        em.add(make_item(f"内容{i}"))
    assert len(em.retrieve(query="内容", limit=3)) == 3


def test_retrieve_filters_by_user_id(em):
    em.add(make_item("alice 的记录", user_id="alice"))
    em.add(make_item("bob 的记录", user_id="bob"))

    results = em.retrieve(query="的记录", user_id="alice", limit=10)
    assert len(results) == 1
    assert results[0].user_id == "alice"


def test_retrieve_filters_by_session_id(em):
    em.add(make_item("s1 记录", session_id="s1"))
    em.add(make_item("s2 记录", session_id="s2"))

    results = em.retrieve(query="记录", session_id="s1", limit=10)
    assert len(results) == 1
    assert results[0].metadata["session_id"] == "s1"


def test_retrieve_higher_importance_ranks_first(em):
    em.add(make_item("关键词 内容", importance=0.1))
    em.add(make_item("关键词 内容", importance=0.95))

    results = em.retrieve(query="关键词", limit=10)
    assert results[0].importance == 0.95


def test_retrieve_respects_time_range(em):
    now = datetime.now()
    old = make_item("旧记录", timestamp=now - timedelta(days=10))
    recent = make_item("新记录", timestamp=now)
    em.add(old)
    em.add(recent)

    results = em.retrieve(
        query="记录",
        time_range=(now - timedelta(days=1), now + timedelta(days=1)),
        limit=10,
    )
    ids = {r.id for r in results}
    assert recent.id in ids
    assert old.id not in ids


def test_retrieve_respects_importance_threshold(em):
    em.add(make_item("低重要性", importance=0.2))
    em.add(make_item("高重要性", importance=0.9))

    results = em.retrieve(query="重要性", importance_threshold=0.6, limit=10)
    assert len(results) == 1
    assert results[0].importance == 0.9


def test_retrieve_result_carries_scores_in_metadata(em):
    em.add(make_item("打分测试"))
    results = em.retrieve(query="打分", limit=5)
    assert "relevance_score" in results[0].metadata
    assert "recency_score" in results[0].metadata


# ============================================================
# 8. forget 三策略
# ============================================================

def test_forget_importance_based(em):
    em.add(make_item("低", importance=0.05))
    em.add(make_item("高", importance=0.9))

    forgotten = em.forget(strategy="importance_based", threshold=0.1)
    assert forgotten == 1
    assert len(em.episodes) == 1
    assert em.episodes[0].importance == 0.9


def test_forget_time_based(em):
    old = make_item("旧", timestamp=datetime.now() - timedelta(days=10))
    recent = make_item("新", timestamp=datetime.now())
    em.add(old)
    em.add(recent)

    forgotten = em.forget(strategy="time_based", max_age_days=1)
    assert forgotten == 1
    assert em.has_memory(recent.id) is True
    assert em.has_memory(old.id) is False


def test_forget_capacity_based_keeps_most_important(tmp_path):
    cfg = MemoryConfig(
        storage_path=str(tmp_path), max_capacity=2,
        disable_embeddings=True, disable_vector_store=True,
    )
    em = EpisodicMemory(config=cfg)
    low = make_item("低", importance=0.1)
    mid = make_item("中", importance=0.5)
    high = make_item("高", importance=0.9)
    for it in (low, mid, high):
        em.add(it)

    forgotten = em.forget(strategy="capacity_based")
    assert forgotten == 1
    assert em.has_memory(low.id) is False
    assert em.has_memory(high.id) is True


# ============================================================
# 9. stats / session / timeline / patterns
# ============================================================

def test_get_stats_expected_keys(em):
    em.add(make_item("a", session_id="s1", importance=0.6))
    em.add(make_item("b", session_id="s2", importance=0.8))

    stats = em.get_stats()
    assert stats["count"] == 2
    assert stats["total_count"] == 2
    assert stats["sessions_count"] == 2
    assert stats["memory_type"] == "episodic"
    assert abs(stats["avg_importance"] - 0.7) < 1e-6
    assert "vector_store" in stats
    assert "document_store" in stats


def test_get_stats_on_empty(em):
    stats = em.get_stats()
    assert stats["count"] == 0
    assert stats["avg_importance"] == 0.0
    assert stats["time_span_days"] == 0.0


def test_get_session_episodes(em):
    a = make_item("s1-a", session_id="s1")
    b = make_item("s1-b", session_id="s1")
    em.add(a)
    em.add(b)
    em.add(make_item("s2-a", session_id="s2"))

    eps = em.get_session_episodes("s1")
    assert {e.episode_id for e in eps} == {a.id, b.id}


def test_get_session_episodes_unknown_returns_empty(em):
    assert em.get_session_episodes("nope") == []


def test_get_timeline_sorted_newest_first(em):
    now = datetime.now()
    em.add(make_item("旧", timestamp=now - timedelta(hours=2)))
    em.add(make_item("新", timestamp=now))

    timeline = em.get_timeline(limit=10)
    assert timeline[0]["content"] == "新"
    assert timeline[-1]["content"] == "旧"


def test_get_timeline_truncates_long_content(em):
    em.add(make_item("a" * 150))
    timeline = em.get_timeline()
    assert len(timeline[0]["content"]) == 103  # 100 + "..."
    assert timeline[0]["content"].endswith("...")


def test_get_timeline_filters_by_user(em):
    em.add(make_item("alice", user_id="alice"))
    em.add(make_item("bob", user_id="bob"))

    timeline = em.get_timeline(user_id="alice")
    assert len(timeline) == 1
    assert timeline[0]["content"] == "alice"


def test_find_patterns_first_call_returns_frequent_keywords(em):
    em.add(make_item("travel planning session"))
    em.add(make_item("travel itinerary notes"))

    patterns = em.find_patterns(min_frequency=2)
    keyword_patterns = [p for p in patterns if p["type"] == "keyword"]
    assert any(p["pattern"] == "travel" for p in keyword_patterns)


def test_find_patterns_detects_context_patterns(em):
    em.add(make_item("a", metadata={"context": {"city": "beijing"}}))
    em.add(make_item("b", metadata={"context": {"city": "beijing"}}))

    patterns = em.find_patterns(min_frequency=2)
    context_patterns = [p for p in patterns if p["type"] == "context"]
    assert any(p["pattern"] == "city:beijing" for p in context_patterns)


def test_find_patterns_confidence_is_ratio(em):
    # "travel" 在两条记忆里各出现一次 → frequency=2, confidence=2/2=1.0
    em.add(make_item("travel planning"))
    em.add(make_item("travel notes"))

    patterns = em.find_patterns(min_frequency=2)
    travel = next(p for p in patterns if p["pattern"] == "travel")
    assert travel["frequency"] == 2
    assert abs(travel["confidence"] - 1.0) < 1e-6


def test_find_patterns_filters_by_user(em):
    em.add(make_item("alicepattern", user_id="alice"))
    em.add(make_item("bobother", user_id="bob"))

    patterns = em.find_patterns(user_id="alice", min_frequency=1)
    assert all(p["pattern"] != "bobother" for p in patterns)


# ============================================================
# 10. 私有辅助
# ============================================================

def test_filter_episodes_by_user_session_and_time(em):
    now = datetime.now()
    e1 = make_item("a", session_id="s1", user_id="u1", timestamp=now)
    e2 = make_item("b", session_id="s2", user_id="u1", timestamp=now - timedelta(days=5))
    em.add(e1)
    em.add(e2)

    by_session = em._filter_episodes(session_id="s1")
    assert {e.episode_id for e in by_session} == {e1.id}

    by_time = em._filter_episodes(time_range=(now - timedelta(days=1), now + timedelta(days=1)))
    assert {e.episode_id for e in by_time} == {e1.id}

    by_user = em._filter_episodes(user_id="u1")
    assert len(by_user) == 2


def test_calculate_time_span(em):
    now = datetime.now()
    em.add(make_item("a", timestamp=now - timedelta(days=5)))
    em.add(make_item("b", timestamp=now))
    assert em._calculate_time_span() == 5


def test_calculate_time_span_empty(em):
    assert em._calculate_time_span() == 0.0


def test_persist_and_remove_from_storage_noop_without_backend(em):
    """storage 为 None 时 _persist_episode / _remove_from_storage 应为空操作。"""
    ep = Episode("e1", "u1", "s1", datetime.now(), "x", {})
    em._persist_episode(ep)         # 不应抛异常
    em._remove_from_storage("e1")   # 不应抛异常


# ============================================================
# 11. 已知缺陷（xfail，修好后删除对应标记）
# ============================================================

@pytest.mark.xfail(
    strict=True,
    reason="get_all() 读取 episode.metadata，但 Episode 未定义该属性 → AttributeError",
)
def test_get_all_returns_memory_items(em):
    em.add(make_item("a"))
    em.add(make_item("b"))
    items = em.get_all()
    assert len(items) == 2
    assert all(it.memory_type == "episodic" for it in items)


@pytest.mark.xfail(
    strict=True,
    reason="find_patterns() 用 timedelta.hours（不存在），第二次命中缓存分支时抛 AttributeError",
)
def test_find_patterns_second_call_uses_cache(em):
    em.add(make_item("travel planning"))
    em.add(make_item("travel notes"))

    em.find_patterns(min_frequency=2)          # 第一次：写入缓存
    patterns = em.find_patterns(min_frequency=2)  # 第二次：走缓存分支
    assert isinstance(patterns, list)


@pytest.mark.xfail(
    strict=True,
    reason="update(metadata=...) 会把 SQLite properties 整体替换，丢失 session_id，导致按 session 检索不到",
)
def test_update_metadata_preserves_session_id(em):
    item = make_item("原始", session_id="s1")
    em.add(item)

    em.update(item.id, metadata={"context": {"city": "beijing"}})

    results = em.retrieve(query="原始", session_id="s1", limit=10)
    assert len(results) == 1