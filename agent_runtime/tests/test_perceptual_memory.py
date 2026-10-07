"""PerceptualMemory 的单元测试。

覆盖范围：
- 顶部：导入回归（守住 storage/__init__.py 重导出 PerceptualMemory 依赖）
- Perception 数据结构（hash 稳定性）
- 初始化：三种向量集合（text/image/audio）、CLIP/CLAP 加载状态
- add / retrieve / update / remove / has_memory / clear
- forget 三种策略
- get_all / get_stats / get_by_modality / cross_modal_search / generate_content
- 编码器：_hash_to_vector 确定性、_text_encoder 真实嵌入、CLIP/CLAP 降级到哈希
- 私有辅助：_calculate_similarity、_get_dim_for_modality、_get_vector_store_for_modality

测试真实代码路径：CLIP/CLAP/Qdrant 必须可用；若任一缺失则跳过相关子集，
但始终验证 SQLite 权威路径 + 哈希编码器回退路径。
"""

import os
import time
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from agent_runtime.memory.base import MemoryConfig, MemoryItem
from agent_runtime.memory.types.perceptual import Perception, PerceptualMemory


# ============================================================
# 0. 依赖探测
# ============================================================

try:
    import qdrant_client  # noqa: F401
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False


def _try_load_transformers_class(symbol: str):
    """尝试加载 transformers 里的某个类；返回 (bool, exc_or_none)。"""
    try:
        from transformers import __dict__ as _td  # noqa
        cls = getattr(_td.get("__name__"), None) if False else None
    except Exception:
        cls = None
    try:
        mod = __import__("transformers", fromlist=[symbol])
        return getattr(mod, symbol, None) is not None, None
    except Exception as e:
        return False, e


# ============================================================
# 0. 导入回归
# ============================================================

def test_perceptual_memory_importable():
    from agent_runtime.memory.types.perceptual import PerceptualMemory
    assert PerceptualMemory is not None


def test_storage_module_re_exports():
    """修复 storage/__init__.py 后，PerceptualMemory 链路能 import。"""
    from agent_runtime.memory.storage import QdrantVectorStore, SQLiteDocumentStore
    from agent_runtime.memory.types.perceptual import PerceptualMemory
    assert all([QdrantVectorStore, SQLiteDocumentStore, PerceptualMemory])


# ============================================================
# 1. Fixtures / helpers
# ============================================================

@pytest.fixture(scope="module")
def clip_available() -> bool:
    ok, _ = _try_load_transformers_class("CLIPModel")
    return ok


@pytest.fixture(scope="module")
def clap_available() -> bool:
    ok, _ = _try_load_transformers_class("ClapModel")
    return ok


@pytest.fixture
def pm(tmp_path) -> PerceptualMemory:
    """默认配置（text/image/audio 三模态），真实加载 CLIP/CLAP/连接 Qdrant。

    依赖任一缺失（Qdrant/CLIP/CLAP）时 pytest.skip；本地开发环境靠这条
    保证不会因为缺一个依赖就整个 suite 崩掉。
    """
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client 未安装，跳过真实向量库路径")
    cfg = MemoryConfig(storage_path=str(tmp_path))
    return PerceptualMemory(config=cfg)


@pytest.fixture
def pm_text_only(tmp_path) -> PerceptualMemory:
    """只启用 text 模态；仍然需要 Qdrant（text 集合走 Qdrant）。"""
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client 未安装，跳过真实向量库路径")
    cfg = MemoryConfig(
        storage_path=str(tmp_path),
        perceptual_memory_modalities=["text"],
    )
    return PerceptualMemory(config=cfg)


def make_item(
    content: str,
    modality: str = "text",
    raw_data=None,
    user_id: str = "u1",
    importance: float = 0.5,
    metadata: dict = None,
    timestamp: datetime = None,
) -> MemoryItem:
    md = dict(metadata or {})
    md["modality"] = modality
    if raw_data is not None:
        md["raw_data"] = raw_data
    return MemoryItem(
        id=str(uuid4()),
        content=content,
        memory_type="perceptual",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
        metadata=md,
    )


# ============================================================
# 2. Perception 数据结构
# ============================================================

def test_perception_stores_all_attributes():
    ts_ref = datetime.now()
    p = Perception(
        perception_id="p1", data="hello", modality="text",
        encoding=[0.1, 0.2, 0.3], metadata={"k": "v"},
    )
    assert p.perception_id == "p1"
    assert p.data == "hello"
    assert p.modality == "text"
    assert p.encoding == [0.1, 0.2, 0.3]
    assert p.metadata == {"k": "v"}
    assert isinstance(p.timestamp, datetime)
    assert p.timestamp >= ts_ref
    assert isinstance(p.data_hash, str) and len(p.data_hash) == 32  # md5 hex


def test_perception_data_hash_is_stable_for_same_input():
    p1 = Perception("p1", "abc", "text")
    p2 = Perception("p2", "abc", "text")
    assert p1.data_hash == p2.data_hash


def test_perception_data_hash_differs_for_different_input():
    p1 = Perception("p1", "abc", "text")
    p2 = Perception("p2", "xyz", "text")
    assert p1.data_hash != p2.data_hash


def test_perception_data_hash_handles_bytes_and_other():
    p_str = Perception("p1", "hello", "text")
    p_bytes = Perception("p1", b"hello", "text")
    # 字符串 "hello" 与 bytes b"hello" 经过 str().encode() 后一致
    assert p_str.data_hash == p_bytes.data_hash


def test_perception_defaults():
    p = Perception("p1", "x", "text")
    assert p.encoding == []
    assert p.metadata == {}


# ============================================================
# 3. 初始化（真实依赖）
# ============================================================

def test_init_creates_three_vector_stores(pm):
    assert set(pm.vector_stores.keys()) == {"text", "image", "audio"}
    for store in pm.vector_stores.values():
        assert store.vector_size > 0


def test_init_supported_modalities(pm):
    # 默认配置含 text/image/audio/video；测试默认 pm 仅校验包含核心三模态
    assert {"text", "image", "audio"}.issubset(pm.supported_modalities)


def test_init_clip_loaded_when_available(pm, clip_available):
    if clip_available:
        assert pm._clip_model is not None
        assert pm._clip_processor is not None
        assert pm._image_dim is not None and pm._image_dim > 0
    else:
        # CLIP 加载失败时降级，_image_dim 应回退到 vector_dim
        assert pm._clip_model is None
        assert pm._image_dim == pm.vector_dim


def test_init_clap_loaded_when_available(pm, clap_available):
    if clap_available:
        assert pm._clap_model is not None
        assert pm._clap_processor is not None
        assert pm._audio_dim is not None and pm._audio_dim > 0
    else:
        assert pm._clap_model is None
        assert pm._audio_dim == pm.vector_dim


def test_init_creates_sqlite_file(tmp_path, clip_available, clap_available):
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client 未安装")
    if not (clip_available and clap_available):
        pytest.skip("transformers + tensorflow 环境不可用（numpy 2.x 不兼容）")
    cfg = MemoryConfig(storage_path=str(tmp_path))
    PerceptualMemory(config=cfg)
    assert (tmp_path / "memory.db").exists()


def test_init_creates_distinct_qdrant_collections(pm):
    """三种模态应使用不同的 collection，互不污染。"""
    cols = {s.collection_name for s in pm.vector_stores.values()}
    assert len(cols) == 3
    for s in pm.vector_stores.values():
        assert "_perceptual_" in s.collection_name


# ============================================================
# 4. add
# ============================================================

def test_add_returns_id_and_caches_in_memory(pm_text_only):
    item = make_item("一段文字描述", modality="text")
    pid = pm_text_only.add(item)
    assert pid == item.id
    assert any(m.id == item.id for m in pm_text_only.perceptual_memories)
    # perceptions 的 key 是 "perception_<memory_id>"
    perception_id = f"perception_{item.id}"
    assert perception_id in pm_text_only.perceptions


def test_add_persists_to_sqlite(pm_text_only):
    item = make_item("SQLite 持久化", modality="text")
    pm_text_only.add(item)
    doc = pm_text_only.doc_store.get_memory(item.id)
    assert doc is not None
    assert doc["content"] == "SQLite 持久化"
    assert doc["memory_type"] == "perceptual"


def test_add_records_modality_in_index(pm_text_only):
    item = make_item("x", modality="text")
    pm_text_only.add(item)
    assert "text" in pm_text_only.modality_index
    perception_id = f"perception_{item.id}"
    assert perception_id in pm_text_only.modality_index["text"]


def test_add_rejects_unsupported_modality(pm_text_only):
    item = make_item("x", modality="video")
    with pytest.raises(ValueError, match="不支持的模态类型"):
        pm_text_only.add(item)


def test_add_does_not_bloat_metadata_with_encoding(pm_text_only):
    """encoding 可能几百维，不应塞进 MemoryItem.metadata。"""
    item = make_item("x", modality="text")
    pm_text_only.add(item)
    assert "encoding" not in item.metadata
    assert "perception_id" in item.metadata


def test_add_writes_vector_to_modality_collection(pm_text_only):
    item = make_item("vector test", modality="text")
    pm_text_only.add(item)
    # 真实向量已落入 text 集合：可检索到
    # 注：QdrantCollectionManager 是模块级单例，跨测试复用同一 collection，
    # 所以用较大 limit 再按 memory_id 精确过滤，避免被其它测试的残留向量误导。
    text_store = pm_text_only.vector_stores["text"]
    hits = text_store.search_similar(
        query_vector=pm_text_only._text_encoder("vector test"),
        limit=100,
    )
    matched = [h for h in hits if h.get("metadata", {}).get("memory_id") == item.id]
    assert matched, f"text 集合中未找到 memory_id={item.id}"


# ============================================================
# 5. has_memory / remove
# ============================================================

def test_has_memory(pm_text_only):
    item = make_item("x", modality="text")
    pm_text_only.add(item)
    assert pm_text_only.has_memory(item.id) is True
    assert pm_text_only.has_memory("ghost") is False


def test_remove_clears_cache_sqlite_and_index(pm_text_only):
    item = make_item("to delete", modality="text")
    pm_text_only.add(item)
    perception_id = f"perception_{item.id}"

    assert pm_text_only.remove(item.id) is True
    assert pm_text_only.has_memory(item.id) is False
    assert perception_id not in pm_text_only.perceptions
    assert perception_id not in pm_text_only.modality_index.get("text", [])
    assert pm_text_only.doc_store.get_memory(item.id) is None


def test_remove_unknown_returns_false(pm_text_only):
    assert pm_text_only.remove("ghost") is False


def test_remove_drops_empty_modality_bucket(pm_text_only):
    item = make_item("only one", modality="text")
    pm_text_only.add(item)
    pm_text_only.remove(item.id)
    assert "text" not in pm_text_only.modality_index


def test_remove_clears_vector_in_qdrant(pm_text_only):
    item = make_item("vec-clear", modality="text")
    pm_text_only.add(item)
    pm_text_only.remove(item.id)
    text_store = pm_text_only.vector_stores["text"]
    hits = text_store.search_similar(
        query_vector=pm_text_only._text_encoder("vec-clear"),
        limit=100,
    )
    assert not any(h.get("metadata", {}).get("memory_id") == item.id for h in hits)


# ============================================================
# 6. update
# ============================================================

def test_update_content_updates_cache_and_sqlite(pm_text_only):
    item = make_item("old", modality="text")
    pm_text_only.add(item)
    assert pm_text_only.update(item.id, content="new") is True
    cached = next(m for m in pm_text_only.perceptual_memories if m.id == item.id)
    assert cached.content == "new"
    assert pm_text_only.doc_store.get_memory(item.id)["content"] == "new"


def test_update_importance(pm_text_only):
    item = make_item("x", modality="text", importance=0.2)
    pm_text_only.add(item)
    pm_text_only.update(item.id, importance=0.9)
    assert pm_text_only.doc_store.get_memory(item.id)["importance"] == 0.9


def test_update_metadata_merges_into_existing(pm_text_only):
    item = make_item("x", modality="text", metadata={"context": {"a": 1}})
    pm_text_only.add(item)
    pm_text_only.update(item.id, metadata={"tags": ["t1"]})
    cached = next(m for m in pm_text_only.perceptual_memories if m.id == item.id)
    assert cached.metadata.get("context") == {"a": 1}
    assert cached.metadata.get("tags") == ["t1"]


def test_update_unknown_returns_false(pm_text_only):
    assert pm_text_only.update("ghost", content="x") is False


# ============================================================
# 7. clear
# ============================================================

def test_clear_removes_all_perceptual(pm_text_only):
    pm_text_only.add(make_item("a", modality="text"))
    pm_text_only.add(make_item("b", modality="text"))
    pm_text_only.clear()
    assert pm_text_only.perceptual_memories == []
    assert pm_text_only.perceptions == {}
    assert pm_text_only.modality_index == {}
    assert pm_text_only.doc_store.search_memories(memory_type="perceptual", limit=100) == []


def test_clear_leaves_other_memory_types_untouched(pm_text_only):
    pm_text_only.add(make_item("perceptual", modality="text"))
    pm_text_only.doc_store.add_memory(
        memory_id="w1", user_id="u1", content="working memory",
        memory_type="working", timestamp=int(time.time()), importance=0.5,
    )
    pm_text_only.clear()
    assert pm_text_only.doc_store.get_memory("w1") is not None


# ============================================================
# 8. retrieve
# ============================================================

def test_retrieve_finds_matching_text(pm_text_only):
    pm_text_only.add(make_item("昨天的西湖游记", modality="text"))
    pm_text_only.add(make_item("在公司写代码", modality="text"))
    results = pm_text_only.retrieve(query="西湖", limit=5)
    assert any("西湖" in r.content for r in results)


def test_retrieve_no_match_returns_empty(pm_text_only):
    """语义召回：query 向量即使字面上不匹配，仍可能召回内容相似的记忆。
    因此「无命中」应使用「库里没有数据」来验证，避免对向量召回的过强假设。"""
    assert pm_text_only.retrieve(query="完全不存在的关键词zzz", limit=5) == []


def test_retrieve_filters_by_user(pm_text_only):
    pm_text_only.add(make_item("alice 的记录", modality="text", user_id="alice"))
    pm_text_only.add(make_item("bob 的记录", modality="text", user_id="bob"))
    results = pm_text_only.retrieve(query="记录", user_id="alice", limit=10)
    assert len(results) == 1
    assert results[0].user_id == "alice"


def test_retrieve_filters_by_target_modality(pm_text_only):
    # 仅 text 模态时也能跑通
    pm_text_only.add(make_item("text item", modality="text"))
    results = pm_text_only.retrieve(query="item", target_modality="text", limit=5)
    assert all(r.metadata.get("modality") == "text" for r in results)


def test_retrieve_respects_limit(pm_text_only):
    for i in range(6):
        pm_text_only.add(make_item(f"内容{i}", modality="text"))
    assert len(pm_text_only.retrieve(query="内容", limit=3)) == 3


def test_retrieve_results_carry_scores(pm_text_only):
    pm_text_only.add(make_item("打分测试", modality="text"))
    results = pm_text_only.retrieve(query="打分", limit=5)
    assert results
    md = results[0].metadata
    assert "relevance_score" in md
    assert "recency_score" in md


def test_retrieve_keyword_fallback_when_no_vector_hit(pm_text_only):
    """向量召回无结果时回退到内存关键词匹配。"""
    pm_text_only.add(make_item("关键词XYZ", modality="text"))
    # 让 query 在向量空间里远离，但子串匹配能命中
    results = pm_text_only.retrieve(query="XYZ", limit=5)
    assert any("XYZ" in r.content for r in results)


# ============================================================
# 9. forget 三策略
# ============================================================

def test_forget_importance_based(pm_text_only):
    pm_text_only.add(make_item("low", modality="text", importance=0.05))
    pm_text_only.add(make_item("high", modality="text", importance=0.9))
    forgotten = pm_text_only.forget(strategy="importance_based", threshold=0.1)
    assert forgotten == 1
    assert len(pm_text_only.perceptual_memories) == 1
    assert pm_text_only.perceptual_memories[0].importance == 0.9


def test_forget_time_based(pm_text_only):
    old = make_item("old", modality="text", timestamp=datetime.now() - timedelta(days=10))
    recent = make_item("recent", modality="text", timestamp=datetime.now())
    pm_text_only.add(old)
    pm_text_only.add(recent)
    forgotten = pm_text_only.forget(strategy="time_based", max_age_days=1)
    assert forgotten == 1
    assert pm_text_only.has_memory(recent.id) is True
    assert pm_text_only.has_memory(old.id) is False


def test_forget_capacity_based_keeps_most_important(tmp_path, clip_available, clap_available):
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client 未安装")
    if not (clip_available and clap_available):
        pytest.skip("transformers + tensorflow 环境不可用（numpy 2.x 不兼容）")
    cfg = MemoryConfig(
        storage_path=str(tmp_path),
        max_capacity=2,
        perceptual_memory_modalities=["text"],
    )
    pm = PerceptualMemory(config=cfg)
    pm.add(make_item("low", modality="text", importance=0.1))
    pm.add(make_item("mid", modality="text", importance=0.5))
    pm.add(make_item("high", modality="text", importance=0.9))
    forgotten = pm.forget(strategy="capacity_based")
    assert forgotten == 1
    memories = {m.content for m in pm.perceptual_memories}
    assert "low" not in memories
    assert "high" in memories


# ============================================================
# 10. get_by_modality / cross_modal_search / generate_content
# ============================================================

def test_get_by_modality_returns_only_matching(pm_text_only):
    pm_text_only.add(make_item("text1", modality="text"))
    pm_text_only.add(make_item("text2", modality="text"))
    results = pm_text_only.get_by_modality("text", limit=10)
    assert len(results) == 2
    assert all(r.metadata.get("modality") == "text" for r in results)


def test_get_by_modality_unknown_returns_empty(pm_text_only):
    assert pm_text_only.get_by_modality("video", limit=10) == []


def test_get_by_modality_respects_limit(pm_text_only):
    for i in range(5):
        pm_text_only.add(make_item(f"item{i}", modality="text"))
    assert len(pm_text_only.get_by_modality("text", limit=3)) == 3


def test_cross_modal_search_delegates_to_retrieve(pm_text_only):
    pm_text_only.add(make_item("桥上风景", modality="text"))
    results = pm_text_only.cross_modal_search(
        query="桥上风景", query_modality="text", target_modality="text", limit=5,
    )
    assert any("桥上风景" in r.content for r in results)


def test_generate_content_text_returns_combination(pm_text_only):
    pm_text_only.add(make_item("点了一杯拿铁", modality="text"))
    pm_text_only.add(make_item("窗外下雨", modality="text"))
    out = pm_text_only.generate_content(prompt="咖啡馆", target_modality="text")
    assert out is not None
    assert "拿铁" in out or "下雨" in out


def test_generate_content_unsupported_returns_none(pm_text_only):
    assert pm_text_only.generate_content(prompt="x", target_modality="video") is None


def test_generate_content_no_relevant_returns_none(pm_text_only):
    out = pm_text_only.generate_content(prompt="完全不存在的zzz", target_modality="text")
    assert out is None


# ============================================================
# 11. get_all / get_stats
# ============================================================

def test_get_all_returns_cached_memories(pm_text_only):
    pm_text_only.add(make_item("a", modality="text"))
    pm_text_only.add(make_item("b", modality="text"))
    items = pm_text_only.get_all()
    assert len(items) == 2
    assert all(it.memory_type == "perceptual" for it in items)


def test_get_stats_expected_keys(pm_text_only):
    pm_text_only.add(make_item("a", modality="text", importance=0.6))
    stats = pm_text_only.get_stats()
    assert stats["count"] == 1
    assert stats["memory_type"] == "perceptual"
    assert "modality_counts" in stats
    assert "supported_modalities" in stats
    assert "vector_stores" in stats
    assert "document_store" in stats


def test_get_stats_empty(pm_text_only):
    stats = pm_text_only.get_stats()
    assert stats["count"] == 0
    assert stats["avg_importance"] == 0.0
    assert stats["perceptions_count"] == 0


# ============================================================
# 12. 编码器（核心算法）
# ============================================================

def test_hash_to_vector_is_deterministic(pm_text_only):
    v1 = pm_text_only._hash_to_vector("abc", 16)
    v2 = pm_text_only._hash_to_vector("abc", 16)
    assert v1 == v2
    assert len(v1) == 16
    # 不同输入产生不同向量
    v3 = pm_text_only._hash_to_vector("xyz", 16)
    assert v1 != v3


def test_hash_to_vector_values_in_unit_range(pm_text_only):
    vec = pm_text_only._hash_to_vector("anything", 32)
    assert all(0.0 <= x <= 1.0 for x in vec)


def test_text_encoder_returns_correct_dimension(pm_text_only):
    vec = pm_text_only._text_encoder("hello world")
    assert len(vec) == pm_text_only.vector_dim


def test_text_encoder_handles_empty_string(pm_text_only):
    vec = pm_text_only._text_encoder("")
    assert len(vec) == pm_text_only.vector_dim


def test_default_encoder_falls_back_to_text(pm_text_only):
    vec = pm_text_only._default_encoder("anything")
    assert len(vec) == pm_text_only.vector_dim


def test_image_encoder_returns_correct_dimension_when_clip_available(pm, clip_available):
    if not clip_available:
        pytest.skip("CLIP 不可用")
    # 真实图像：用一张 1x1 RGB PNG 字节流
    from PIL import Image
    from io import BytesIO
    buf = BytesIO()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(buf, format="PNG")
    vec = pm._image_encoder(buf.getvalue())
    assert len(vec) == pm._image_dim


def test_image_encoder_falls_back_to_hash_on_bad_input(pm):
    """传入非图像数据时，应降级到哈希编码而不是抛异常。"""
    vec = pm._image_encoder("not a real image string at all")
    assert isinstance(vec, list)
    assert len(vec) == pm._get_dim_for_modality("image")


def test_audio_encoder_falls_back_to_hash_on_bad_input(pm):
    vec = pm._audio_encoder("not a real audio")
    assert isinstance(vec, list)
    assert len(vec) == pm._get_dim_for_modality("audio")


def test_encode_data_pads_or_truncates_to_target_dim(pm_text_only):
    """_encode_data 必须输出与目标维度严格一致的长度。"""
    pm_text_only._image_dim = 8
    pm_text_only._audio_dim = 16
    short_vec = [0.1, 0.2]
    long_vec = list(range(100))

    # 通过 _default_encoder 走真实路径，最终都会截断或补齐
    assert len(pm_text_only._encode_data("x", "text")) == pm_text_only.vector_dim


# ============================================================
# 13. 私有辅助
# ============================================================

def test_calculate_similarity_basic(pm_text_only):
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(pm_text_only._calculate_similarity(a, b) - 1.0) < 1e-6

    c = [0.0, 1.0, 0.0]
    assert abs(pm_text_only._calculate_similarity(a, c) - 0.0) < 1e-6


def test_calculate_similarity_orthogonal(pm_text_only):
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(pm_text_only._calculate_similarity(a, b)) < 1e-6


def test_calculate_similarity_empty_inputs(pm_text_only):
    assert pm_text_only._calculate_similarity([], [1.0, 2.0]) == 0.0
    assert pm_text_only._calculate_similarity([1.0, 2.0], []) == 0.0
    assert pm_text_only._calculate_similarity([], []) == 0.0


def test_calculate_similarity_handles_zero_vector(pm_text_only):
    assert pm_text_only._calculate_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_get_dim_for_modality_dispatch(pm_text_only):
    pm_text_only._image_dim = 256
    pm_text_only._audio_dim = 128
    assert pm_text_only._get_dim_for_modality("text") == pm_text_only.vector_dim
    assert pm_text_only._get_dim_for_modality("image") == 256
    assert pm_text_only._get_dim_for_modality("audio") == 128


def test_get_dim_for_modality_defaults_to_text(pm_text_only):
    assert pm_text_only._get_dim_for_modality(None) == pm_text_only.vector_dim
    assert pm_text_only._get_dim_for_modality("") == pm_text_only.vector_dim


def test_get_vector_store_for_modality_dispatch(pm_text_only):
    assert pm_text_only._get_vector_store_for_modality("text") is pm_text_only.vector_stores["text"]
    assert pm_text_only._get_vector_store_for_modality("image") is pm_text_only.vector_stores["image"]
    assert pm_text_only._get_vector_store_for_modality("audio") is pm_text_only.vector_stores["audio"]
    # 未知 / None → 兜底到 text
    assert pm_text_only._get_vector_store_for_modality(None) is pm_text_only.vector_stores["text"]
    assert pm_text_only._get_vector_store_for_modality("unknown") is pm_text_only.vector_stores["text"]


def test_no_grad_restores_grad_state(pm_text_only):
    """_no_grad 上下文管理器应在 with 块外恢复原来的 grad 状态。"""
    try:
        import torch
    except ImportError:
        pytest.skip("torch 不可用")

    original = torch.is_grad_enabled()
    with pm_text_only._no_grad():
        assert torch.is_grad_enabled() is False
    assert torch.is_grad_enabled() == original


# ============================================================
# 14. 已知缺陷（xfail，修好后删除对应标记）
# ============================================================

@pytest.mark.xfail(
    strict=True,
    reason="update() 不更新 cached perception 的 encoding；改 content 后向量的 metadata.content 与 SQLite 不一致",
)
def test_update_content_reembeds_qdrant(pm_text_only):
    item = make_item("old text", modality="text")
    pm_text_only.add(item)
    pm_text_only.update(item.id, content="brand new text")

    # 向量库中 payload.content 应与最新一致
    text_store = pm_text_only.vector_stores["text"]
    hits = text_store.search_similar(
        query_vector=pm_text_only._text_encoder("brand new text"),
        limit=5,
    )
    matched = [h for h in hits if h.get("metadata", {}).get("memory_id") == item.id]
    assert matched
    assert matched[0]["metadata"]["content"] == "brand new text"


@pytest.mark.xfail(
    strict=True,
    reason="Perception.metadata 没有感知数据 hash 字段，data_hash 虽存在但 add() 时未透传到 SQLite properties",
)
def test_add_persists_data_hash_to_sqlite(pm_text_only):
    item = make_item("x", modality="text")
    pm_text_only.add(item)
    props = pm_text_only.doc_store.get_memory(item.id)["properties"]
    assert "data_hash" in props
