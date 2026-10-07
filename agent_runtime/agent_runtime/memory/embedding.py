"""统一嵌入模块（实现 + 提供器）

说明（中文）：
- 提供统一的文本嵌入接口与多实现：本地Transformer、DashScope（通义千问）、散列词频兜底。
- 暴露 get_text_embedder()/get_dimension()/refresh_embedder() 供各记忆类型统一使用。
- 通过环境变量优先级：dashscope > local > tfidf。
- 各实现均为「延迟加载」：构造对象不联网，首次 warmup()/encode()/dimension 时才加载；
  带回退的工厂会在构造后立即 warmup() 以校验可用性，不可用则切换下一个后端。
- 回退链的每一级失败原因都会以 warning 日志输出，不再静默吞掉。

环境变量：
- EMBED_MODEL_TYPE: "dashscope" | "local" | "tfidf"（默认 dashscope）
- EMBED_MODEL_NAME: 模型名称（留空时各后端用自身默认：
  dashscope→text-embedding-v3；local→sentence-transformers/all-MiniLM-L6-v2）
- EMBED_API_KEY: Embedding API Key（统一命名）
- EMBED_BASE_URL: Embedding Base URL（统一命名，可选）
"""

from typing import List, Union, Optional
import threading
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def _apply_hf_endpoint() -> None:
    """在导入 huggingface_hub / transformers / sentence-transformers 之前应用 HF 端点。

    huggingface_hub 在 **导入时** 读取 HF_ENDPOINT 并缓存到常量里，之后再改环境变量无效。
    因此这里在真正 import 之前主动设置；若库已经导入，则一并覆盖其缓存常量，
    保证「换镜像」在任意导入顺序下都生效。

    用法：在 .env 里设置 `HF_ENDPOINT=https://hf-mirror.com` 即可让国内网络顺利下载模型。
    """
    endpoint = os.getenv("HF_ENDPOINT")
    if not endpoint:
        return

    # 尚未导入时，仅需放进 os.environ，import 时自会读取
    os.environ.setdefault("HF_ENDPOINT", endpoint)

    # 已导入时，覆盖其模块级缓存（不同版本常量名略有差异，逐个尝试）
    try:
        import huggingface_hub.constants as hc
        if getattr(hc, "ENDPOINT", None) != endpoint:
            hc.ENDPOINT = endpoint
            hc.HUGGINGFACE_CO_URL_TEMPLATE = endpoint.rstrip("/") + "/{repo_id}/resolve/{revision}/{filename}"
            logger.info("已将 HuggingFace 端点设置为: %s", endpoint)
    except Exception:
        # huggingface_hub 未安装或结构不匹配：交给后续真正 import 时读取 os.environ
        pass


# ==============
# 抽象与实现
# ==============

class EmbeddingModel:
    """嵌入模型基类（最小接口）"""

    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    def warmup(self) -> None:
        """预先加载底层资源以校验可用性，失败时抛异常。

        默认无操作。需要联网/加载文件的实现应重写此方法，
        以便调用方（如带回退的工厂）在构造阶段就发现不可用并切换后端。
        """
        return None


class LocalTransformerEmbedding(EmbeddingModel):
    """本地Transformer嵌入（优先 sentence-transformers，缺失回退 transformers+torch）"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend = None  # "st" 或 "hf"
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._dimension = None
        # 延迟加载：不在构造时联网/下载模型，首次 warmup()/encode()/dimension 时才加载

    def warmup(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self):
        if self._backend is None:
            self._load_backend()

    def _load_backend(self):
        # 应用 HF 镜像端点（若配置），必须在 import HF 库之前
        _apply_hf_endpoint()

        # 优先 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            test_vec = self._st_model.encode("test_text")
            self._dimension = len(test_vec)
            self._backend = "st"
            logger.info("本地嵌入(sentence-transformers)就绪: %s (dim=%s)", self.model_name, self._dimension)
            return
        except Exception as e:
            self._st_model = None
            _st_error = e

        # 回退 transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            with torch.no_grad():
                inputs = self._hf_tokenizer("test_text", return_tensors="pt", padding=True, truncation=True)
                outputs = self._hf_model(**inputs)
                test_embedding = outputs.last_hidden_state.mean(dim=1)
                self._dimension = int(test_embedding.shape[1])
            self._backend = "hf"
            logger.info("本地嵌入(transformers)就绪: %s (dim=%s)", self.model_name, self._dimension)
            return
        except Exception as e:
            self._hf_tokenizer = None
            self._hf_model = None
            raise ImportError(
                f"本地嵌入后端加载失败: model={self.model_name!r}; "
                f"sentence-transformers 错误: {_st_error!r}; transformers 错误: {e!r}"
            ) from e

    def encode(self, texts: Union[str, List[str]]):
        self._ensure_loaded()
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        if self._backend == "st":
            vecs = self._st_model.encode(inputs)
            if hasattr(vecs, "tolist"):
                vecs = [v for v in vecs]
        else:
            import torch
            tokenized = self._hf_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._hf_model(**tokenized)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vecs = [v for v in embeddings]

        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return int(self._dimension or 0)


class TFIDFEmbedding(EmbeddingModel):
    """轻量兜底嵌入（无深度模型时保证离线可用）。

    实现说明：使用 HashingVectorizer 对文本做固定维度的散列词频编码。
    之所以不用 TfidfVectorizer，是因为 TF-IDF 需要先 fit 才能 transform，
    而本模块的工厂只会「构造」兜底后端、不会喂语料 fit，导致 encode 必然报错。
    HashingVectorizer 无需拟合、维度固定、确定性，构造即可用。
    类名与 EMBED_MODEL_TYPE 取值 "tfidf" 予以保留以兼容既有配置。
    """

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self._vectorizer = None
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self):
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
        except ImportError:
            raise ImportError("请安装 scikit-learn: pip install scikit-learn")
        self._vectorizer = HashingVectorizer(
            n_features=self.max_features,
            stop_words="english",
            alternate_sign=False,
            norm="l2",
        )

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False
        matrix = self._vectorizer.transform(inputs)
        embeddings = matrix.toarray()
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class DashScopeEmbedding(EmbeddingModel):
    """阿里云 DashScope（通义千问）Embedding / OpenAI兼容REST 模式

    行为：
    - 如提供 base_url，则优先使用 OpenAI 兼容的 REST 接口（POST {base_url}/embeddings）。
    - 否则使用官方 dashscope SDK 的 TextEmbedding.call。
    """

    def __init__(self, model_name: str = "text-embedding-v3", api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension = None
        self._client_ready = False
        # 延迟初始化：不在构造时 import SDK / 发网络请求

    def warmup(self) -> None:
        self._ensure_client()
        if self._dimension is None:
            self._dimension = len(self.encode("health_check"))
            logger.info("DashScope 嵌入就绪: %s (dim=%s)", self.model_name, self._dimension)

    def _ensure_client(self):
        if self._client_ready:
            return
        # 仅在非REST情况下需要初始化 SDK（import 失败会抛 ImportError）
        if not self.base_url:
            self._init_client()
        self._client_ready = True

    def _init_client(self):
        try:
            if self.api_key:
                # 将统一命名的 API Key 注入到 SDK 期望的位置
                os.environ["DASHSCOPE_API_KEY"] = self.api_key
            import dashscope  # noqa: F401
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def encode(self, texts: Union[str, List[str]]):
        self._ensure_client()
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        # REST 模式（OpenAI兼容）
        if self.base_url:
            import requests
            url = self.base_url.rstrip("/") + "/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model_name, "input": inputs}
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code >= 400:
                raise RuntimeError(f"Embedding REST 调用失败: {resp.status_code} {resp.text}")
            data = resp.json()
            # 期望结构：{"data": [{"embedding": [...]}]}
            items = data.get("data") or []
            vecs = [np.array(item.get("embedding")) for item in items]
            if single:
                return vecs[0]
            return vecs

        # SDK 模式
        from dashscope import TextEmbedding
        rsp = TextEmbedding.call(model=self.model_name, input=inputs)
        embeddings_obj = None
        if isinstance(rsp, dict):
            embeddings_obj = (rsp.get("output") or {}).get("embeddings")
        else:
            embeddings_obj = getattr(getattr(rsp, "output", None), "embeddings", None)
        if not embeddings_obj:
            raise RuntimeError("DashScope 返回为空或格式不匹配")
        vecs = [np.array(item.get("embedding") or item.get("vector")) for item in embeddings_obj]
        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.encode("health_check"))
        return int(self._dimension or 0)


# ==============
# 工厂与回退
# ==============

def create_embedding_model(model_type: str = "local", **kwargs) -> EmbeddingModel:
    """创建嵌入模型实例

    model_type: "dashscope" | "local" | "tfidf"
    kwargs: model_name, api_key
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return LocalTransformerEmbedding(**kwargs)
    elif model_type == "dashscope":
        return DashScopeEmbedding(**kwargs)
    elif model_type == "tfidf":
        return TFIDFEmbedding(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def create_embedding_model_with_fallback(
    preferred_type: str = "dashscope",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> EmbeddingModel:
    """带回退的创建：dashscope -> local -> tfidf。

    每个后端使用「自己的」默认模型名；只有当用户显式配置 model_name 时，
    才把它传给首选后端。这样可避免把 dashscope 的模型名（如 text-embedding-v3）
    误传给本地后端，导致去 HuggingFace 下载一个不存在的模型。
    """
    if preferred_type in ("sentence_transformer", "huggingface"):
        preferred_type = "local"
    fallback = ["dashscope", "local", "tfidf"]
    # 将首选放最前
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0, preferred_type)

    errors = []
    for t in fallback:
        try:
            kwargs = {}
            # 用户指定的模型名只作用于首选后端；其余回退后端各用自身默认模型
            if t == preferred_type and model_name:
                kwargs["model_name"] = model_name
            if t == "dashscope":
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
            model = create_embedding_model(t, **kwargs)
            model.warmup()  # 立即校验可用性，失败则切换到下一个后端
            logger.info("嵌入后端已选定: %s", t)
            return model
        except Exception as e:
            errors.append(f"{t}({type(e).__name__}: {e})")
            logger.warning("嵌入后端 %s 不可用，尝试下一个: %s", t, e)

    raise RuntimeError("所有嵌入模型都不可用，请安装依赖或检查配置: " + " | ".join(errors))


# ==================
# Provider（单例）
# ==================

_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None


def _build_embedder() -> EmbeddingModel:
    preferred = os.getenv("EMBED_MODEL_TYPE", "dashscope").strip()
    # 仅在用户显式配置 EMBED_MODEL_NAME 时传递模型名；
    # 留空则各后端使用自身的默认模型（避免首选的模型名串到回退后端）
    model_name = os.getenv("EMBED_MODEL_NAME", "").strip() or None
    return create_embedding_model_with_fallback(
        preferred_type=preferred,
        model_name=model_name,
        api_key=os.getenv("EMBED_API_KEY"),
        base_url=os.getenv("EMBED_BASE_URL"),
    )


def get_text_embedder() -> EmbeddingModel:
    """获取全局共享的文本嵌入实例（线程安全单例）"""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """获取统一向量维度（失败回退默认值）"""
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception as e:
        logger.warning("获取嵌入维度失败，回退默认值 %s: %s", default, e)
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """强制重建嵌入实例（可用于动态切换环境变量）"""
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder