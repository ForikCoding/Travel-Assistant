"""embedding.py 的单元测试。

按章节组织：基类 / TF-IDF / LocalTransformer / DashScope /
工厂与回退 / 单例 / get_dimension / _build_embedder / HF 端点。
所有用例默认走 mock + monkeypatch，零真实网络依赖。
"""

import logging
import os

import numpy as np
import pytest

import agent_runtime.memory.embedding as emb
from agent_runtime.memory.embedding import (
    DashScopeEmbedding,
    EmbeddingModel,
    LocalTransformerEmbedding,
    TFIDFEmbedding,
    _apply_hf_endpoint,
    _build_embedder,
    create_embedding_model,
    create_embedding_model_with_fallback,
    get_dimension,
    get_text_embedder,
    refresh_embedder,
)


# ============================================================
# 1. 基类：未实现方法直接抛 NotImplementedError
# ============================================================

def test_base_encode_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        EmbeddingModel().encode("x")


def test_base_dimension_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        EmbeddingModel().dimension


def test_base_warmup_is_noop():
    # 默认无操作；不抛异常也不必触发任何动作
    EmbeddingModel().warmup()


# ============================================================
# 2. HF 端点注入（huggingface_hub 在导入时读取 HF_ENDPOINT 缓存为常量）
# ============================================================

def test_apply_hf_endpoint_noop_when_unset(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _apply_hf_endpoint()
    assert "HF_ENDPOINT" not in os.environ


def test_apply_hf_endpoint_sets_os_environ_when_not_imported(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", "https://example.com")
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub.constants", None)
    _apply_hf_endpoint()
    # os.environ 应保留该值（供后续 import 时读取）
    assert os.environ.get("HF_ENDPOINT") == "https://example.com"


def test_apply_hf_endpoint_overrides_huggingface_hub_constants(monkeypatch):
    """HF 库已导入时，应覆盖其模块级缓存常量。"""
    import sys as _sys
    fake_mod = type(
        "M",
        (),
        {
            "ENDPOINT": "https://huggingface.co",
            "HUGGINGFACE_CO_URL_TEMPLATE":
                "https://huggingface.co/{repo_id}/resolve/{revision}/{filename}",
        },
    )()
    monkeypatch.setitem(_sys.modules, "huggingface_hub.constants", fake_mod)

    monkeypatch.setenv("HF_ENDPOINT", "https://example.com")
    _apply_hf_endpoint()
    assert fake_mod.ENDPOINT == "https://example.com"
    assert fake_mod.HUGGINGFACE_CO_URL_TEMPLATE == \
        "https://example.com/{repo_id}/resolve/{revision}/{filename}"


# ============================================================
# 3. TFIDFEmbedding（兜底：无需 fit、构造即用）
# ============================================================

def test_tfidf_dimension_equals_max_features():
    assert TFIDFEmbedding(max_features=128).dimension == 128


def test_tfidf_returns_numpy_arrays():
    m = TFIDFEmbedding(max_features=64)
    vec = m.encode("hello")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (64,)


def test_tfidf_batch_returns_list_of_arrays():
    m = TFIDFEmbedding(max_features=32)
    out = m.encode(["a", "b", "c"])
    assert isinstance(out, list) and len(out) == 3
    assert all(isinstance(v, np.ndarray) and v.shape == (32,) for v in out)


def test_tfidf_is_deterministic():
    m = TFIDFEmbedding(max_features=64)
    assert list(m.encode("hello world")) == list(m.encode("hello world"))


def test_tfidf_warmup_is_noop():
    assert TFIDFEmbedding().warmup() is None


# ============================================================
# 4. LocalTransformerEmbedding（延迟加载、encode 形状、idempotent warmup）
# ============================================================

def test_local_transformer_construction_is_lazy(monkeypatch):
    """构造时不触发下载；首次 warmup 才加载。"""
    calls = {"n": 0}

    def fake_load(self):
        calls["n"] += 1
        raise RuntimeError("no net")

    monkeypatch.setattr(LocalTransformerEmbedding, "_load_backend", fake_load)

    m = LocalTransformerEmbedding()
    assert calls["n"] == 0
    with pytest.raises(RuntimeError):
        m.warmup()
    assert calls["n"] == 1


def test_local_transformer_encode_triggers_lazy_load(monkeypatch):
    calls = {"n": 0}

    def fake_load(self):
        calls["n"] += 1
        self._backend = "st"
        self._dimension = 3
        self._st_model = type("M", (), {"encode": lambda _s, xs: [[0.0, 0.0, 0.0] for _ in xs]})()

    monkeypatch.setattr(LocalTransformerEmbedding, "_load_backend", fake_load)

    m = LocalTransformerEmbedding()
    assert calls["n"] == 0
    vec = m.encode("x")
    assert calls["n"] == 1
    assert len(vec) == 3


def test_local_transformer_encode_returns_list_for_batch_input(monkeypatch):
    """backend='st' 时，批量 encode 应返回列表（与 SentenceTransformer 一致）。"""
    class ST:
        def encode(self, inputs):
            return np.ones((len(inputs), 4), dtype=np.float32)

    monkeypatch.setattr(LocalTransformerEmbedding, "_load_backend", lambda self: None)
    m = LocalTransformerEmbedding()
    m._backend = "st"
    m._st_model = ST()

    out = m.encode(["a", "b"])
    assert isinstance(out, list) and len(out) == 2
    assert all(len(v) == 4 for v in out)


def test_local_transformer_encode_hf_branch(monkeypatch):
    """backend='hf' 分支走 transformers 路径，需要 torch + 模型对象。"""
    import torch

    class FakeTokenizer:
        def __call__(self, inputs, **kwargs):
            class T(dict):
                pass
            return T()

    class FakeModel:
        def __call__(self, **kwargs):
            class O:
                last_hidden_state = torch.zeros((2, 3, 5))
            return O()

    monkeypatch.setattr(LocalTransformerEmbedding, "_load_backend", lambda self: None)
    m = LocalTransformerEmbedding()
    m._backend = "hf"
    m._hf_tokenizer = FakeTokenizer()
    m._hf_model = FakeModel()

    out = m.encode(["x", "y"])
    assert isinstance(out, list) and len(out) == 2
    assert all(len(v) == 5 for v in out)


def test_local_transformer_dimension_property_returns_int(monkeypatch):
    monkeypatch.setattr(LocalTransformerEmbedding, "_load_backend", lambda self: None)
    m = LocalTransformerEmbedding()
    m._backend = "st"
    m._dimension = 123
    assert m.dimension == 123
    assert isinstance(m.dimension, int)


def test_local_transformer_ensure_loaded_only_loads_once(monkeypatch):
    """_ensure_loaded 应在已加载时跳过，重复 warmup 不触发新加载。"""
    calls = {"n": 0}

    def fake_load(self):
        calls["n"] += 1
        self._backend = "st"
        self._dimension = 4

    monkeypatch.setattr(LocalTransformerEmbedding, "_load_backend", fake_load)
    m = LocalTransformerEmbedding()
    m.warmup()
    m.warmup()
    m.warmup()
    assert calls["n"] == 1


# ============================================================
# 5. DashScopeEmbedding（懒加载、REST/SDK 分支、env 注入）
# ============================================================

def test_dashscope_construction_is_lazy(monkeypatch):
    """构造时不发网络请求 / 不 import SDK。"""
    calls = {"n": 0}

    def fake_init(self):
        calls["n"] += 1
        raise ImportError("no dashscope")

    monkeypatch.setattr(DashScopeEmbedding, "_init_client", fake_init)

    m = DashScopeEmbedding()
    assert calls["n"] == 0
    with pytest.raises(ImportError):
        m.warmup()
    assert calls["n"] == 1


def test_dashscope_dimension_is_cached_after_warmup(monkeypatch):
    """首次访问 dimension 调一次 encode 探测；之后复用缓存。"""
    calls = {"n": 0}

    def fake_encode(self, texts):
        calls["n"] += 1
        inputs = [texts] if isinstance(texts, str) else list(texts)
        return [np.zeros(8, dtype=np.float32) for _ in inputs]

    monkeypatch.setattr(DashScopeEmbedding, "_ensure_client", lambda self: None)
    monkeypatch.setattr(DashScopeEmbedding, "encode", fake_encode)

    m = DashScopeEmbedding()
    _ = m.dimension
    _ = m.dimension
    _ = m.dimension
    assert calls["n"] == 1


def test_dashscope_init_client_sets_env_var(monkeypatch):
    """有 api_key 时应注入 DASHSCOPE_API_KEY 到 os.environ。"""
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    m = DashScopeEmbedding(api_key="sk-test")
    m._init_client()
    assert os.environ["DASHSCOPE_API_KEY"] == "sk-test"


def test_dashscope_init_client_no_api_key_does_not_set_env(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    m = DashScopeEmbedding(api_key=None)
    m._init_client()
    assert "DASHSCOPE_API_KEY" not in os.environ


def test_dashscope_ensure_client_runs_only_once(monkeypatch):
    calls = {"n": 0}

    def fake_init(self):
        calls["n"] += 1

    monkeypatch.setattr(DashScopeEmbedding, "_init_client", fake_init)
    m = DashScopeEmbedding()
    m._ensure_client()
    m._ensure_client()
    m._ensure_client()
    assert calls["n"] == 1


def test_dashscope_encode_with_base_url_uses_rest(monkeypatch):
    """base_url 非空时，encode 应走 REST 而不是 SDK。"""
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr(DashScopeEmbedding, "_init_client", lambda self: None)

    m = DashScopeEmbedding(base_url="http://localhost:8000/v1", api_key="sk-x")
    out = m.encode("hi")
    assert captured["url"] == "http://localhost:8000/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer sk-x"
    assert captured["json"]["model"]
    assert isinstance(out, np.ndarray) and out.shape == (3,)


def test_dashscope_encode_rest_error_raises(monkeypatch):
    class FakeResp:
        status_code = 401
        text = "unauthorized"

    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp())
    monkeypatch.setattr(DashScopeEmbedding, "_init_client", lambda self: None)

    m = DashScopeEmbedding(base_url="http://localhost:8000/v1", api_key="bad")
    with pytest.raises(RuntimeError, match="401"):
        m.encode("x")


# ============================================================
# 6. 工厂：create_embedding_model（按 model_type 分发到对应实现）
# ============================================================

def test_create_local_aliases_resolve_to_local_implementation():
    assert isinstance(create_embedding_model("local"), LocalTransformerEmbedding)
    assert isinstance(create_embedding_model("sentence_transformer"), LocalTransformerEmbedding)
    assert isinstance(create_embedding_model("huggingface"), LocalTransformerEmbedding)


def test_create_dashscope_factory():
    assert isinstance(create_embedding_model("dashscope"), DashScopeEmbedding)


def test_create_tfidf_factory():
    assert isinstance(create_embedding_model("tfidf"), TFIDFEmbedding)


def test_create_unknown_type_raises():
    with pytest.raises(ValueError, match="不支持的模型类型"):
        create_embedding_model("bogus")


# ============================================================
# 7. 回退工厂：create_embedding_model_with_fallback
# ============================================================

def test_fallback_does_not_leak_preferred_model_name(monkeypatch):
    """修复点：dashscope 的模型名不应串到 local 回退后端。"""
    calls = []

    class FakeModel:
        def __init__(self, name=None):
            self.name = name

        def warmup(self):
            return None

    def fake_create(model_type, **kwargs):
        calls.append((model_type, dict(kwargs)))
        if model_type == "dashscope":
            raise RuntimeError("dashscope down")
        return FakeModel(kwargs.get("model_name"))

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    create_embedding_model_with_fallback(
        preferred_type="dashscope", model_name="text-embedding-v3"
    )

    # 首选 dashscope 拿到用户指定的模型名
    assert calls[0] == ("dashscope", {"model_name": "text-embedding-v3"})
    # 回退到 local 时不再携带该模型名（用自身默认）
    assert calls[1][0] == "local"
    assert "model_name" not in calls[1][1]


def test_fallback_passes_api_key_only_to_dashscope(monkeypatch):
    calls = []

    class FakeModel:
        def warmup(self):
            return None

    def fake_create(model_type, **kwargs):
        calls.append((model_type, dict(kwargs)))
        return FakeModel()

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    create_embedding_model_with_fallback(
        preferred_type="dashscope", api_key="sk-1", base_url="http://x"
    )
    assert calls[0][0] == "dashscope"
    assert calls[0][1] == {"api_key": "sk-1", "base_url": "http://x"}


def test_fallback_switches_backend_when_warmup_fails(monkeypatch):
    """当前后端 warmup 失败 → 切换到下一个。"""
    class BadModel:
        def warmup(self):
            raise RuntimeError("no net")

    class GoodModel:
        def warmup(self):
            return None

    def fake_create(model_type, **kwargs):
        return BadModel() if model_type == "local" else GoodModel()

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    model = create_embedding_model_with_fallback(preferred_type="local")
    assert isinstance(model, GoodModel)


def test_fallback_error_message_lists_all_failed_backends(monkeypatch):
    """全部失败时，错误信息应包含每个后端的失败原因。"""
    def fake_create(model_type, **kwargs):
        raise RuntimeError(f"{model_type}-fail")

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    with pytest.raises(RuntimeError) as exc:
        create_embedding_model_with_fallback(preferred_type="dashscope")
    msg = str(exc.value)
    assert "dashscope-fail" in msg
    assert "local-fail" in msg
    assert "tfidf-fail" in msg


def test_fallback_logs_warning_when_backend_skipped(monkeypatch, caplog):
    """某个后端失败时，应以 warning 记录失败原因（不再静默吞异常）。"""
    class GoodModel:
        def warmup(self):
            return None

    def fake_create(model_type, **kwargs):
        if model_type == "dashscope":
            raise RuntimeError("dashscope boom")
        return GoodModel()

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    with caplog.at_level(logging.WARNING, logger="agent_runtime.memory.embedding"):
        create_embedding_model_with_fallback(preferred_type="dashscope")

    assert "dashscope boom" in caplog.text


def test_fallback_normalizes_aliases(monkeypatch):
    """sentence_transformer / huggingface 应被归一为 local。"""
    seen = []

    class FakeModel:
        def warmup(self):
            return None

    def fake_create(model_type, **kwargs):
        seen.append(model_type)
        return FakeModel()

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    for alias in ("sentence_transformer", "huggingface"):
        seen.clear()
        create_embedding_model_with_fallback(preferred_type=alias)
        assert seen[0] == "local", f"alias={alias} should normalize to local"


def test_fallback_default_preferred_is_dashscope(monkeypatch):
    """不传 preferred 时，dashscope 应排在最前。"""
    seen = []

    class FakeModel:
        def warmup(self):
            return None

    def fake_create(model_type, **kwargs):
        seen.append(model_type)
        return FakeModel()

    monkeypatch.setattr(emb, "create_embedding_model", fake_create)

    create_embedding_model_with_fallback()
    assert seen[0] == "dashscope"


# ============================================================
# 8. 单例 Provider：get_text_embedder / refresh_embedder
# ============================================================

def test_singleton_returns_same_instance(monkeypatch):
    calls = {"n": 0}

    class FakeModel:
        def warmup(self):
            return None

    def fake_build():
        calls["n"] += 1
        return FakeModel()

    monkeypatch.setattr(emb, "_build_embedder", fake_build)

    a = get_text_embedder()
    b = get_text_embedder()
    assert a is b
    assert calls["n"] == 1


def test_singleton_is_thread_safe_double_check(monkeypatch):
    """并发首次调用也只应触发一次 _build_embedder（双重检查）。"""
    import threading

    calls = {"n": 0}
    lock = threading.Lock()

    class FakeModel:
        def warmup(self):
            return None

    def fake_build():
        with lock:
            calls["n"] += 1
        return FakeModel()

    monkeypatch.setattr(emb, "_build_embedder", fake_build)
    monkeypatch.setattr(emb, "_embedder", None)

    threads = [threading.Thread(target=get_text_embedder) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert calls["n"] == 1


def test_refresh_embedder_force_rebuilds(monkeypatch):
    """refresh_embedder 应覆盖已有单例（便于改 env 后重新加载）。"""
    class FakeA: pass
    class FakeB: pass

    monkeypatch.setattr(emb, "_embedder", FakeA())
    monkeypatch.setattr(emb, "_build_embedder", lambda: FakeB())

    assert get_text_embedder() is emb._embedder
    assert isinstance(emb._embedder, FakeA)

    refresh_embedder()
    assert isinstance(emb._embedder, FakeB)


# ============================================================
# 9. get_dimension：embedder 可用时取真实维度；任何异常都兜底 + warning
# ============================================================

def test_get_dimension_returns_embedder_dimension(monkeypatch):
    class Fake:
        dimension = 768

    monkeypatch.setattr(emb, "get_text_embedder", lambda: Fake())
    assert get_dimension(384) == 768


def test_get_dimension_falls_back_to_default_with_warning_on_failure(monkeypatch, caplog):
    """embedder 异常 / 缺属性时：返回 default + 记录 warning。"""
    def boom():
        raise RuntimeError("no service")
    monkeypatch.setattr(emb, "get_text_embedder", boom)

    with caplog.at_level(logging.WARNING, logger="agent_runtime.memory.embedding"):
        dim = get_dimension(384)

    assert dim == 384
    assert "384" in caplog.text and "no service" in caplog.text


# ============================================================
# 10. _build_embedder：环境变量优先级
# ============================================================

def test_build_embedder_model_name_none_when_unset(monkeypatch):
    """EMBED_MODEL_NAME 未设时，model_name 应传 None（避免污染回退后端）。"""
    captured = {}

    def fake_factory(preferred_type=None, model_name=None, api_key=None, base_url=None):
        captured.update(preferred_type=preferred_type, model_name=model_name)
        return object()

    monkeypatch.setattr(emb, "create_embedding_model_with_fallback", fake_factory)
    monkeypatch.setenv("EMBED_MODEL_TYPE", "dashscope")
    monkeypatch.delenv("EMBED_MODEL_NAME", raising=False)

    _build_embedder()
    assert captured["model_name"] is None
    assert captured["preferred_type"] == "dashscope"


def test_build_embedder_model_name_passed_when_set(monkeypatch):
    captured = {}

    def fake_factory(preferred_type=None, model_name=None, api_key=None, base_url=None):
        captured.update(model_name=model_name)
        return object()

    monkeypatch.setattr(emb, "create_embedding_model_with_fallback", fake_factory)
    monkeypatch.setenv("EMBED_MODEL_NAME", "my-model")

    _build_embedder()
    assert captured["model_name"] == "my-model"


def test_build_embedder_respects_all_env_vars(monkeypatch):
    monkeypatch.setenv("EMBED_MODEL_TYPE", "dashscope")
    monkeypatch.setenv("EMBED_MODEL_NAME", "custom-model")
    monkeypatch.setenv("EMBED_API_KEY", "sk-1")
    monkeypatch.setenv("EMBED_BASE_URL", "http://x/v1")

    captured = {}

    def fake_factory(preferred_type=None, model_name=None, api_key=None, base_url=None):
        captured.update(preferred_type=preferred_type, model_name=model_name,
                        api_key=api_key, base_url=base_url)
        return object()

    monkeypatch.setattr(emb, "create_embedding_model_with_fallback", fake_factory)
    emb._build_embedder()

    assert captured == dict(
        preferred_type="dashscope",
        model_name="custom-model",
        api_key="sk-1",
        base_url="http://x/v1",
    )


def test_build_embedder_empty_model_name_becomes_none(monkeypatch):
    """EMBED_MODEL_NAME 留空（含纯空白）应作为 None 传入。"""
    monkeypatch.setenv("EMBED_MODEL_NAME", "   ")
    captured = {}

    def fake_factory(model_name=None, **kw):
        captured["model_name"] = model_name
        return object()

    monkeypatch.setattr(emb, "create_embedding_model_with_fallback", fake_factory)
    emb._build_embedder()
    assert captured["model_name"] is None