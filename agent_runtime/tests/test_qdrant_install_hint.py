"""守住 qdrant-client 缺失时的报错信息可操作性。

`_install_hint()` 函数生成的提示必须包含：
1. 当前 Python 解释器路径 + 可复制的安装命令
2. `agent-runtime[storage]` 这一可读性更高的安装目标
3. 原始 import 异常（用于排障）

直接调用函数测试——避免 importlib.reload 的 sys.modules 共享陷阱。
"""

import sys

import pytest

from agent_runtime.memory.storage.qdrant_store import _install_hint


class FakeImportError(Exception):
    pass


def test_install_hint_includes_python_executable():
    """报错应自动带上当前 Python 解释器路径 + 一行可复制的安装命令。"""
    hint = _install_hint()
    py = sys.executable.replace("\\", "/")
    assert py in hint, f"应当包含 sys.executable ({py})，实际: {hint}"


def test_install_hint_includes_pip_install_command():
    hint = _install_hint()
    assert "-m pip install qdrant-client" in hint
    # 应包含 PyPI 上一个具体的最低版本号（>=1.6.0 这样的兜底）
    assert "qdrant-client>=1." in hint


def test_install_hint_promotes_agent_runtime_storage_extra():
    """可读性更高的安装路径：`pip install agent-runtime[storage]`。"""
    hint = _install_hint()
    assert "agent-runtime[storage]" in hint


def test_install_hint_includes_installable_python_path():
    """路径中的反斜杠应被替换为正斜杠，方便直接复制粘贴。"""
    hint = _install_hint()
    # 安装命令里应使用正斜杠（Windows 兼容）
    install_line = next(line for line in hint.splitlines() if "-m pip install" in line)
    assert "\\" not in install_line, f"安装命令应使用正斜杠，实际: {install_line!r}"


def test_install_hint_is_multiline_for_readability():
    """提示必须多行，便于人眼阅读：摘要 + 安装命令 + 可选依赖。"""
    hint = _install_hint()
    assert hint.count("\n") >= 2, f"提示应多行，实际只有 {hint.count(chr(10))} 个换行"