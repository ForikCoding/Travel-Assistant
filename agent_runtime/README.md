# agent_runtime

一个独立、可复用的智能体运行时框架，提供：

- **核心抽象**：`Agent`、`Message`、`LLM`、`Config`、`Tool`、`Memory`、`Protocol`
- **多种 Agent**：`SimpleAgent`、`ReActAgent`、`ReflectionAgent`、`PlanSolveAgent`
- **多类型记忆**：工作记忆、情景记忆、语义记忆、感知记忆（含 RAG 检索增强）
- **多存储后端**：SQLite / Qdrant / Neo4j
- **多智能体通信协议**：A2A / ANP / MCP

## 安装

### 依赖矩阵

| 用途 | 安装命令 | 包含 |
|---|---|---|
| 仅核心（嵌入走本地 sentence-transformers） | `pip install -e ./agent_runtime` | pydantic / openai / qdrant-client / numpy / tiktoken |
| + Web UI 与 MCP | `pip install -e "./agent_runtime[full]"` | gradio / fastmcp |
| + Neo4j + spacy（SemanticMemory） | `pip install -e "./agent_runtime[graph]"` | neo4j / spacy |
| + 多模态感知（CLIP + CLAP 真实图像/音频编码） | `pip install -e "./agent_runtime[perceptual]"` | transformers / Pillow / librosa |
| + 全部可选 + 测试 | `pip install -e "./agent_runtime[full,graph,perceptual,dev]"` | 上面所有 |
| 从 requirements.txt（核心） | `pip install -r requirements.txt` | 与第一行等价 |

> **为什么 qdrant-client 是核心依赖？**  
> `agent_runtime.memory.types.episodic` 在模块顶层 `from qdrant_client import ...`，
> 即使是「纯 SQLite 离线模式」也必须装它才能 import 包。运行时通过设置
> `MEMORY_DISABLE_EMBEDDINGS=true` / `MEMORY_DISABLE_VECTOR_STORE=true` 即可不连服务。

> **PerceptualMemory 多模态依赖说明**  
> 启用 `image` / `audio` 模态时，会通过 `transformers` 加载 CLIP / CLAP 模型，
> 并通过 `Pillow` 解码图像、`librosa` 读取音频。未安装这些包时，相应模态会
> **优雅降级**为基于 SHA-256 的确定性哈希向量（同内容一定同向量，但语义不可比）。
> CLIP 模型 ~600 MB、CLAP 模型 ~1.5 GB，首次加载需要联网下载。

### 方式一：构建 wheel 包（推荐，可分发给其他项目）

```bash
cd agent_runtime
pip install build
python -m build
```

这会在 `dist/` 下生成 `.whl` 文件，然后其他项目可以：

```bash
pip install agent_runtime-0.1.0-py3-none-any.whl
```

### 拷贝到其他项目

整个 `agent_runtime/` 目录夹可以原样复制到任意项目的子目录下，然后：

```bash
pip install -e path/to/copied/agent_runtime
```

或者直接将其打成 wheel 后分发。

## 使用示例

```python
from agent_runtime.core.agent import Agent
from agent_runtime.core.llm import AgentsLLM
from agent_runtime.core.message import Message

llm = AgentsLLM(model="gpt-4")
agent = Agent(name="assistant", llm=llm)
reply = agent.run([Message(role="user", content="你好")])
print(reply)
```

## Skills

agent_runtime 支持两种 Skill 类型：

### Python Skill（代码型）

继承 `Skill` 类，实现 `execute` 方法：

```python
from agent_runtime.skills import Skill, SkillMetadata, SkillParameter

class MySkill(Skill):
    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="my-skill",
            description="我的自定义 Skill",
            parameters=[
                SkillParameter(name="input", type="string", description="输入", required=True),
            ],
            tags=["custom"],
        )

    def execute(self, context, **kwargs):
        return {"success": True, "data": {"result": kwargs["input"].upper()}}

# 注册
from agent_runtime.skills import init_skills
registry = init_skills()
registry.register(MySkill())
```

### Markdown Skill（指令型）

使用格式 C 的 `.md` 文件定义，Agent 加载后自动将指令注入 LLM 上下文，由 LLM 自行执行。

**文件格式示例：**

````markdown
# Skill: office-docx

## 元信息
name: office-docx
description: 创建 Word 文档
version: 0.1.0

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
3. 保存到用户指定路径（默认保存在当前目录）

## 使用场景
当用户说"帮我创建一份报告"、"生成一个 Word 文档"时调用此 Skill。
````

**安装外部 Markdown Skill：**

安装后只需将 `.md` 文件所在目录配置到环境变量即可：

```bash
# Windows
set AGENT_RUNTIME_SKILLS_PATHS=D:\path\to\skills_folder

# Linux/macOS
export AGENT_RUNTIME_SKILLS_PATHS=/path/to/skills_folder
```

多个路径用 `;`（Windows）或 `:`（Linux/macOS）分隔。

**初始化时自动扫描：**

```python
from agent_runtime.skills import init_skills

# 自动扫描：
# 1. 内置 builtin/ 目录
# 2. AGENT_RUNTIME_SKILLS_PATHS 环境变量指定的路径
registry = init_skills()
```

### Agent 中使用 Skill

```python
from agent_runtime.skills import init_skills
from agent_runtime.tools.builtin import register_skill_tool
from agent_runtime.agents import SimpleAgent
from agent_runtime.tools.registry import global_registry

# 初始化 Skills 并注册到 ToolRegistry
registry = init_skills()
register_skill_tool(registry)

# 创建带工具调用的 Agent
agent = SimpleAgent(
    name="assistant",
    llm=llm,
    tool_registry=global_registry,
)

# Agent 调用方式（通过 LLM 推理自动选择）：
# skill[{"skill_name": "note", "args": {"action": "create", "title": "会议", "content": "..."}}]
# skill[{"skill_name": "office-docx", "args": {"title": "报告", "content": "..."}}]
```

## 测试

```bash
cd agent_runtime
pytest tests/
```

## 故障排查

### `ImportError: qdrant-client 未安装 ...`

`agent_runtime.memory.storage.qdrant_store` 在初始化时会检查 `qdrant-client`。
未安装时抛出的 `ImportError` 已自动带上：

1. **原始 import 异常**（方便定位是版本冲突还是网络问题）
2. **当前 Python 解释器路径**与一行可复制的安装命令（用 `sys.executable -m pip install`）
3. **`agent_runtime[storage]`** 这一可读性更高的安装目标

直接照提示装即可。

### 想完全不装 qdrant / Neo4j 跑纯 SQLite

```python
from agent_runtime.memory.base import MemoryConfig
from agent_runtime.memory.manager import MemoryManager

cfg = MemoryConfig(
    disable_embeddings=True,
    disable_vector_store=True,
)
manager = MemoryManager(config=cfg)   # 仅工作记忆可用，其它记忆走 SQLite 文本检索
```

或在 `.env` 设置：

```bash
MEMORY_DISABLE_EMBEDDINGS=true
MEMORY_DISABLE_VECTOR_STORE=true
```

### `PerceptualMemory` 加载失败 / 想跳过 CLIP/CLAP

`PerceptualMemory` 在 `__init__` 会无条件尝试加载 CLIP/CLAP，连不上 HuggingFace
或 transformers 抛异常时**不会中断**——`_clip_model` / `_clap_model` 被置为 `None`，
后续 image / audio 模态自动降级为 SHA-256 哈希编码（确定性、可重复、但无语义）。

如需显式降级，仅启用 text 模态即可：

```python
from agent_runtime.memory.base import MemoryConfig

cfg = MemoryConfig(perceptual_memory_modalities=["text"])
```

### `transformers` / `tensorflow` 与 numpy 版本冲突

`transformers` 加载时会拉起 `tensorflow` → `keras`，需要 `numpy<2`。如果遇到：

```
AttributeError: _ARRAY_API not found
```

请降级 numpy：

```bash
pip install "numpy<2"
```

或者安装纯 PyTorch 后端的 transformers（默认就是 torch），跳过 tensorflow 链：

```bash
pip install tensorflow-cpu==2.15.0  # 兼容 numpy 2.x 的最后一个 TF 版本（可选）
```

## 项目结构

```
agent_runtime/
├── agents/        # 各种 Agent 实现
├── core/          # 核心抽象：Agent、LLM、Message、Config
├── context/       # 上下文构建
├── memory/        # 记忆系统（基础/类型/RAG/存储）
├── protocols/     # 通信协议（A2A/ANP/MCP）
├── tools/         # 工具框架与内置工具
└── tests/         # 单元测试
```