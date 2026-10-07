"""agent_runtime — 独立可复用的智能体运行时框架。

详见 README.md。
"""

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# 在包导入时加载 .env（任意子模块导入都会先执行本文件，保证配置早于使用生效）
#
# 关键点：huggingface_hub / transformers 等库在 **导入时** 读取 HF_ENDPOINT，
# 因此 .env 里的 HF_ENDPOINT（如镜像 https://hf-mirror.com）必须先于这些库
# 被导入。放在包入口可保证任意导入顺序下都生效。
# ---------------------------------------------------------------------------
import os as _os
from pathlib import Path as _Path

try:
    from dotenv import load_dotenv as _load_dotenv

    # 锚定到 agent_runtime 项目根目录下的 .env，使配置在任意工作目录下都生效
    # （__init__.py → agent_runtime(包) → agent_runtime(项目根)）
    _ENV_FILE = _Path(__file__).resolve().parents[1] / ".env"
    if _ENV_FILE.is_file():
        _load_dotenv(_ENV_FILE)
    # 兼容回退：再按当前工作目录搜索一次（已加载的值不会被覆盖）
    _load_dotenv()
except ImportError:
    pass
