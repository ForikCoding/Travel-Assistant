# Travel-Assistant

旅游 AI 助手 —— 基于 Vue 3 + TypeScript 前端与 Python FastAPI 后端的全栈项目。

## 技术栈

| 层级   | 技术                           |
| ------ | ------------------------------ |
| 前端   | Vue 3, TypeScript, Vite        |
| 路由   | Vue Router 4                   |
| 状态管理 | Pinia 4                       |
| 后端   | Python 3.10+, FastAPI, Uvicorn |
| 数据验证 | Pydantic 2                    |
| 搜索服务 | Google Search Results (SerpApi) |

## 项目结构

```
Travel-Assistant/
├── frontend/                  # 前端项目 (Vue 3 + TypeScript)
│   ├── src/
│   │   ├── assets/            # 静态资源
│   │   ├── components/        # 公共组件
│   │   ├── App.vue            # 根组件
│   │   ├── main.ts            # 应用入口
│   │   └── style.css          # 全局样式
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
│
├── backend/                   # 后端项目 (Python FastAPI)
│   ├── app/
│   │   ├── agents/            # Agent 模块
│   │   ├── models/            # 数据模型
│   │   ├── services/          # 服务层
│   │   ├── tools/             # 可复用工具包
│   │   │   ├── tool_base.py   # Tool / ToolParameter 抽象基类
│   │   │   ├── mcp_tool.py    # MCPTool / MCPWrappedTool
│   │   │   └── llm.py         # AgentsLLM (DeepSeek API)
│   │   ├── config.py          # 配置管理
│   │   └── main.py            # API 入口
│   ├── venv/                  # Python 虚拟环境（需自行创建）
│   ├── .env                   # 环境变量配置
│   └── requirements.txt
│
└── .gitignore
```

## 快速开始

### 前端

```bash
cd frontend
npm install
npm run dev        # 启动开发服务器
npm run build      # 生产构建
npm run preview    # 预览生产构建
```

### 后端

```bash
cd backend

# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境（根据终端选择）
# Git Bash / Linux / macOS:
source venv/Scripts/activate
# 或
source venv/bin/activate

# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# 如遇执行策略限制，先运行:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Windows CMD:
venv\Scripts\activate.bat

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量（编辑 .env 填写 API Key）
# 5. 启动服务
uvicorn app.main:app --reload --port 8000
```

启动后访问 http://localhost:8000/docs 查看 API 文档（Swagger UI）。

## License

MIT
