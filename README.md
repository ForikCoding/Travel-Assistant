# Travel-Assistant 🌍✈️

基于 Agent 框架构建的智能旅行规划助手，集成高德地图 MCP 服务，提供个性化的旅行计划生成。

## ✨ 功能特点

- 🤖 **AI 驱动的旅行规划**：基于自研 Agent 框架（SimpleAgent），智能生成详细的多日旅程
- 🗺️ **高德地图集成**：通过 MCP 协议接入高德地图服务，支持景点搜索、路线规划、天气查询
- 🧠 **智能工具调用**：Agent 自动调用高德地图 MCP 工具，获取实时 POI、路线和天气信息
- 🔍 **搜索增强**：集成 Google Search Results (SerpApi)，辅助获取旅行相关信息
- 🎨 **现代化前端**：Vue 3 + TypeScript + Vite，响应式设计，流畅的用户体验
- 📱 **完整功能**：包含住宿、交通、餐饮和景点游览时间推荐

## 🏗️ 技术栈

### 后端
- **框架**：FastAPI + Agent Runtime（自研 Agent 框架）
- **Agent**：SimpleAgent、ReActAgent、PlanSolveAgent、ReflectionAgent
- **MCP 工具**：amap-mcp-server（高德地图）
- **LLM**：支持多种 LLM 提供商（OpenAI、DeepSeek、通义千问、智谱等）
- **数据验证**：Pydantic v2 + pydantic-settings
- **搜索服务**：Google Search Results (SerpApi)
- **图片服务**：Unsplash API

### 前端
- **框架**：Vue 3 + TypeScript
- **构建工具**：Vite
- **状态管理**：Pinia
- **路由**：Vue Router 4
- **HTTP 客户端**：Axios

## 📁 项目结构

```
Travel-Assistant/
├── backend/                         # 后端服务
│   ├── app/
│   │   ├── agent_runtime/           # Agent 运行时框架
│   │   │   ├── agents/              # SimpleAgent | ReActAgent | PlanSolveAgent | ReflectionAgent
│   │   │   ├── core/                # Agent | Config | LLM | Message | Exceptions
│   │   │   ├── tools/               # 工具注册表 | MCP 工具 | 内置工具
│   │   │   ├── context/             # 上下文构建器
│   │   │   └── memory/              # 记忆模块（语义记忆 | RAG）
│   │   ├── trip_agents/             # 旅行规划 Agent
│   │   │   └── trip_planner_agent.py
│   │   ├── api/                     # FastAPI 路由
│   │   │   ├── main.py
│   │   │   └── routes/
│   │   │       ├── trip.py          # 旅行规划接口
│   │   │       ├── poi.py           # POI 详情接口
│   │   │       └── map.py           # 地图服务接口
│   │   ├── services/                # 服务层
│   │   │   ├── amap_service.py      # 高德地图服务
│   │   │   ├── llm_service.py       # LLM 服务
│   │   │   └── unsplash_service.py  # Unsplash 图片服务
│   │   ├── models/                  # 数据模型
│   │   │   └── schemas.py
│   │   ├── tools/                   # 可复用工具包
│   │   │   ├── tool_base.py         # Tool / ToolParameter 抽象基类
│   │   │   ├── mcp_tool.py          # MCPTool / MCPWrappedTool
│   │   │   └── llm.py               # AgentsLLM（多提供商 LLM 客户端）
│   │   └── config.py                # 配置管理（pydantic-settings）
│   ├── requirements.txt
│   ├── .env.example
│   └── .gitignore
├── frontend/                        # 前端应用
│   ├── src/
│   │   ├── components/              # Vue 组件
│   │   ├── assets/                  # 静态资源
│   │   ├── App.vue                  # 根组件
│   │   ├── main.ts                  # 应用入口
│   │   └── style.css                # 全局样式
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🚀 快速开始

### 前提条件

- Python 3.10+
- Node.js 16+
- [uv](https://docs.astral.sh/uv/)（提供 `uvx` 命令，用于运行高德地图 MCP 服务器）
- 高德地图 API 密钥（Web 服务 API）
- LLM API 密钥（OpenAI / DeepSeek 等）

### 后端安装

1. 进入后端目录
```bash
cd backend
```

2. 创建虚拟环境
```bash
python -m venv venv

# Git Bash / Linux / macOS:
source venv/Scripts/activate
# 或
source venv/bin/activate

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat
```

3. 安装依赖
```bash
pip install -r requirements.txt
pip install uv  # 安装 uv，提供 uvx 命令用于运行 amap-mcp-server
```

4. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

5. 启动后端服务
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端安装

1. 进入前端目录
```bash
cd frontend
```

2. 安装依赖
```bash
npm install
```

3. 启动开发服务器
```bash
npm run dev
```

4. 打开浏览器访问 `http://localhost:5173`

## 📝 使用指南

1. 在首页填写旅行信息：
   - 目的地城市
   - 旅行日期和天数
   - 交通方式偏好
   - 住宿偏好
   - 旅行风格标签

2. 点击「生成旅行计划」按钮

3. 系统将：
   - 调用 Agent 生成初步计划
   - Agent 自动调用高德地图 MCP 工具搜索景点
   - Agent 获取天气信息和路线规划
   - 整合所有信息生成完整行程

4. 查看结果：
   - 每日详细行程
   - 景点信息与地图标记
   - 交通路线规划
   - 天气预报
   - 餐饮推荐

## 🔧 核心实现

### Agent 集成

```python
from app.agent_runtime.agents.simple_agent import SimpleAgent
from app.agent_runtime.tools.builtin.protocol_tools import MCPTool
from app.services.llm_service import get_llm

# 创建高德地图 MCP 工具
amap_tool = MCPTool(
    name="amap",
    server_command=["uvx", "amap-mcp-server"],
    env={"AMAP_MAPS_API_KEY": "your_api_key"},
    auto_expand=True
)

# 创建旅行规划 Agent
agent = SimpleAgent(
    name="旅行规划助手",
    llm=get_llm(),
    system_prompt="你是一个专业的旅行规划助手..."
)

# 添加工具
agent.add_tool(amap_tool)
```

### MCP 工具调用

Agent 可以自动调用以下高德地图 MCP 工具：
- `maps_text_search`：搜索景点 POI
- `maps_weather`：查询天气
- `maps_direction_walking_by_address`：步行路线规划
- `maps_direction_driving_by_address`：驾车路线规划
- `maps_direction_transit_integrated_by_address`：公共交通路线规划

## 📄 API 文档

启动后端服务后，访问 `http://localhost:8000/docs` 查看完整的 API 文档（Swagger UI）。

主要端点：
- `POST /api/trip/plan` — 生成旅行计划
- `GET /api/map/poi` — 搜索 POI
- `GET /api/map/weather` — 查询天气
- `POST /api/map/route` — 规划路线
- `GET /api/poi/detail/{poi_id}` — 获取 POI 详情
- `GET /api/poi/photo` — 获取景点图片

## 🤝 贡献指南

欢迎提交 Pull Request 或 Issue！

## 📜 开源协议

MIT

## 🙏 致谢

- [HelloAgents](https://github.com/datawhalechina/Hello-Agents) — 智能体教程
- [HelloAgents 框架](https://github.com/jjyaoao/HelloAgents) — 智能体框架
- [高德地图开放平台](https://lbs.amap.com/) — 地图服务
- [amap-mcp-server](https://github.com/sugarforever/amap-mcp-server) — 高德地图 MCP 服务器

---

**Travel-Assistant** — 让旅行计划变得简单而智能 🌈

-y
@fangjunjie/ssh-mcp-server
--host
192.168.204.131
port
2201
--username
lion
--password
th8682087@