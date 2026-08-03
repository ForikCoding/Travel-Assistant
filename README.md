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
│   │   ├── __init__.py
│   │   └── main.py            # API 入口
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
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

启动后访问 http://localhost:8000/docs 查看 API 文档（Swagger UI）。

## License

MIT
