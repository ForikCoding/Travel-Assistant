"""
MCP (Model Context Protocol) 工具

基于 Python MCP SDK 实现的 MCP 客户端工具，参考 hello-agents 的
tools/builtin/protocol_tools.py 和 tools/builtin/mcp_wrapper_tool.py 实现。

功能：
- 连接外部 MCP 服务器（如 amap-mcp-server）
- 自动发现服务器提供的工具列表
- auto_expand 模式下将每个 MCP 工具展开为独立的 Tool 对象
- 支持 list_tools / call_tool / list_resources / read_resource / list_prompts / get_prompt
"""

import asyncio
import concurrent.futures
import os
from typing import Any, Dict, List, Optional

from .tool_base import Tool, ToolParameter

# ── 可选依赖检查 ──────────────────────────────────────────────
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client as _mcp_stdio_client

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    _mcp_stdio_client = None  # type: ignore


# ── MCP 环境变量映射表 ────────────────────────────────────────
# 用于自动检测常见 MCP 服务器需要的环境变量
MCP_SERVER_ENV_MAP: Dict[str, List[str]] = {
    "server-github": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    "server-slack": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    "server-google-drive": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
    "server-postgres": ["POSTGRES_CONNECTION_STRING"],
    "server-sqlite": [],
    "server-filesystem": [],
    "amap-mcp-server": ["AMAP_MAPS_API_KEY"],
}


# ── MCP 客户端 ────────────────────────────────────────────────

class _MCPClient:
    """MCP stdio 客户端（内部使用）

    封装了与 MCP 服务器的 stdio 通信，提供：
    - list_tools: 列出可用工具
    - call_tool: 调用指定工具
    - list_resources: 列出资源
    - read_resource: 读取资源
    - list_prompts: 列出提示词
    - get_prompt: 获取提示词
    """

    def __init__(
        self,
        server_command: List[str],
        server_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.server_command = server_command
        self.server_args = server_args or []
        self.env = env or {}
        self._session: Optional[Any] = None
        self._transport: Optional[Any] = None

    async def __aenter__(self) -> "_MCPClient":
        if not _MCP_AVAILABLE:
            raise ImportError(
                "MCP 功能需要安装 mcp 库，请执行: pip install mcp"
            )

        # 合并环境变量
        merged_env = {**os.environ, **self.env} if self.env else None

        server_params = StdioServerParameters(
            command=self.server_command[0],
            args=self.server_command[1:] + self.server_args,
            env=merged_env,
        )

        # stdio_client 返回一个上下文管理器
        self._transport_cm = _mcp_stdio_client(server_params)
        read, write = await self._transport_cm.__aenter__()

        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session is not None:
            await self._session.__aexit__(*args)
        if hasattr(self, "_transport_cm"):
            await self._transport_cm.__aexit__(*args)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出 MCP 服务器提供的所有工具"""
        result = await self._session.list_tools()
        tools = []
        for tool in result.tools:
            tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
            })
        return tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """调用 MCP 工具"""
        result = await self._session.call_tool(tool_name, arguments)
        # 提取文本内容
        if result.content:
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    texts.append(block["text"])
                else:
                    texts.append(str(block))
            return "\n".join(texts)
        return str(result)

    async def list_resources(self) -> List[Dict[str, Any]]:
        """列出 MCP 服务器提供的资源"""
        result = await self._session.list_resources()
        return [
            {"uri": r.uri, "name": r.name, "description": getattr(r, "description", "")}
            for r in result.resources
        ]

    async def read_resource(self, uri: str) -> str:
        """读取 MCP 资源"""
        result = await self._session.read_resource(uri)
        if result.content:
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
                else:
                    texts.append(str(block))
            return "\n".join(texts)
        return str(result)

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """列出 MCP 服务器提供的提示词"""
        result = await self._session.list_prompts()
        return [
            {"name": p.name, "description": getattr(p, "description", "")}
            for p in result.prompts
        ]

    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取指定提示词"""
        result = await self._session.get_prompt(name, arguments)
        return [
            {"role": msg.role, "content": msg.content}
            for msg in result.messages
        ]


# ── MCP 工具包装器 ────────────────────────────────────────────

class MCPWrappedTool(Tool):
    """MCP 工具包装器 — 将单个 MCP 工具包装成独立的 Tool 对象

    由 MCPTool 在 auto_expand 模式下自动创建，Agent 调用时只需提供参数，
    无需了解 MCP 的内部结构。
    """

    def __init__(
        self,
        mcp_tool: "MCPTool",
        tool_info: Dict[str, Any],
        prefix: str = "",
    ):
        self.mcp_tool = mcp_tool
        self.tool_info = tool_info
        self.mcp_tool_name = tool_info.get("name", "unknown")

        # 构建工具名：prefix + mcp_tool_name
        tool_name = f"{prefix}{self.mcp_tool_name}" if prefix else self.mcp_tool_name
        description = tool_info.get("description", f"MCP 工具: {self.mcp_tool_name}")

        # 解析参数 schema
        self._parameters = self._parse_input_schema(
            tool_info.get("input_schema", {})
        )

        super().__init__(name=tool_name, description=description)

    def _parse_input_schema(self, input_schema: Dict[str, Any]) -> List[ToolParameter]:
        """将 MCP 的 input_schema（JSON Schema 格式）转换为 ToolParameter 列表"""
        parameters: List[ToolParameter] = []
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])

        for param_name, param_info in properties.items():
            parameters.append(ToolParameter(
                name=param_name,
                type=param_info.get("type", "string"),
                description=param_info.get("description", ""),
                required=param_name in required_fields,
            ))

        return parameters

    def get_parameters(self) -> List[ToolParameter]:
        return self._parameters

    def run(self, params: Dict[str, Any]) -> str:
        """执行 MCP 工具 — 委托给父 MCPTool"""
        mcp_params = {
            "action": "call_tool",
            "tool_name": self.mcp_tool_name,
            "arguments": params,
        }
        return self.mcp_tool.run(mcp_params)


# ── MCP 工具 ──────────────────────────────────────────────────

class MCPTool(Tool):
    """MCP (Model Context Protocol) 工具

    连接到外部 MCP 服务器并调用其提供的工具、资源和提示词。

    功能：
    - 自动发现服务器提供的工具
    - auto_expand 模式下展开为独立工具
    - 支持 list_tools / call_tool / list_resources / read_resource / list_prompts / get_prompt

    使用示例:
        >>> tool = MCPTool(
        ...     name="amap",
        ...     server_command=["uvx", "amap-mcp-server"],
        ...     env={"AMAP_MAPS_API_KEY": "xxx"},
        ...     auto_expand=True,
        ... )
        >>> result = tool.run({"action": "list_tools"})
        >>> result = tool.run({"action": "call_tool", "tool_name": "xxx", "arguments": {...}})
    """

    def __init__(
        self,
        name: str = "mcp",
        description: Optional[str] = None,
        server_command: Optional[List[str]] = None,
        server_args: Optional[List[str]] = None,
        auto_expand: bool = True,
        env: Optional[Dict[str, str]] = None,
        env_keys: Optional[List[str]] = None,
    ):
        """
        初始化 MCP 工具

        Args:
            name: 工具名称（建议为不同服务器指定不同名称）
            description: 工具描述（可选，默认自动生成）
            server_command: 服务器启动命令（如 ["uvx", "amap-mcp-server"]）
            server_args: 服务器参数列表
            auto_expand: 是否自动展开为独立工具（默认 True）
            env: 环境变量字典（优先级最高）
            env_keys: 要从系统环境变量加载的 key 列表

        环境变量优先级（从高到低）：
            1. 直接传递的 env 参数
            2. env_keys 指定的环境变量
            3. 自动检测的环境变量（根据 server_command 匹配 MCP_SERVER_ENV_MAP）
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self._available_tools: List[Dict[str, Any]] = []
        self.auto_expand = auto_expand
        self.prefix = f"{name}_" if auto_expand else ""

        # 环境变量处理
        self.env = self._prepare_env(env, env_keys, server_command)

        # 自动发现工具
        if server_command:
            self._discover_tools()

        # 自动生成描述
        if description is None:
            description = self._generate_description()

        super().__init__(name=name, description=description)

    # ── 环境变量处理 ──────────────────────────────────────

    def _prepare_env(
        self,
        env: Optional[Dict[str, str]],
        env_keys: Optional[List[str]],
        server_command: Optional[List[str]],
    ) -> Dict[str, str]:
        """准备环境变量，优先级：env > env_keys > 自动检测"""
        result_env: Dict[str, str] = {}

        # 1. 自动检测（优先级最低）
        if server_command:
            server_name = None
            for part in server_command:
                if "server-" in part or "mcp-server" in part:
                    server_name = part.split("/")[-1] if "/" in part else part
                    break

            if server_name and server_name in MCP_SERVER_ENV_MAP:
                for key in MCP_SERVER_ENV_MAP[server_name]:
                    value = os.getenv(key)
                    if value:
                        result_env[key] = value
                        print(f"[ENV] auto-loaded env var: {key}")

        # 2. env_keys 指定的环境变量（优先级中等）
        if env_keys:
            for key in env_keys:
                value = os.getenv(key)
                if value:
                    result_env[key] = value
                    print(f"[ENV] loaded from env_keys: {key}")
                else:
                    print(f"[WARN] env var not set: {key}")

        # 3. 直接传递的 env（优先级最高）
        if env:
            result_env.update(env)
            for key in env:
                print(f"[ENV] using direct env var: {key}")

        return result_env

    # ── 异步执行辅助 ──────────────────────────────────────

    @staticmethod
    def _run_async(coro):
        """在同步上下文中运行异步协程

        自动检测当前是否已有运行中的事件循环：
        - 若已有（如 FastAPI/uvicorn），在新线程中运行
        - 若无，直接使用 asyncio.run()
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环
            return asyncio.run(coro)

        # 已有运行中的事件循环，在新线程中执行
        def _run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run_in_new_loop).result()

    # ── 工具发现 ──────────────────────────────────────────

    def _discover_tools(self) -> None:
        """发现 MCP 服务器提供的所有工具"""
        if not self.server_command:
            return

        async def _discover():
            async with _MCPClient(
                self.server_command, self.server_args, env=self.env
            ) as client:
                return await client.list_tools()

        try:
            self._available_tools = self._run_async(_discover())
        except Exception as e:
            # 工具发现失败不影响初始化，可在 run() 时再试
            print(f"[WARN] MCP tool discovery failed: {e}")
            self._available_tools = []

    # ── 描述生成 ──────────────────────────────────────────

    def _generate_description(self) -> str:
        """自动生成工具描述"""
        if not self._available_tools:
            return "连接到 MCP 服务器，调用工具、读取资源和获取提示词。"

        if self.auto_expand:
            return f"MCP 工具服务器，包含 {len(self._available_tools)} 个工具。这些工具会自动展开为独立的工具供 Agent 使用。"
        else:
            lines = [f"MCP 工具服务器，提供 {len(self._available_tools)} 个工具："]
            for tool in self._available_tools[:10]:  # 最多显示前 10 个
                tool_name = tool.get("name", "unknown")
                tool_desc = tool.get("description", "无描述")
                short_desc = tool_desc.split(".")[0] if tool_desc else "无描述"
                lines.append(f"  • {tool_name}: {short_desc}")
            if len(self._available_tools) > 10:
                lines.append(f"  ... 还有 {len(self._available_tools) - 10} 个工具")
            return "\n".join(lines)

    # ── 展开工具 ──────────────────────────────────────────

    def get_expanded_tools(self) -> List[MCPWrappedTool]:
        """获取展开的工具列表

        将 MCP 服务器的每个工具包装成独立的 MCPWrappedTool 对象。

        Returns:
            MCPWrappedTool 列表
        """
        if not self.auto_expand:
            return []

        expanded: List[MCPWrappedTool] = []
        for tool_info in self._available_tools:
            wrapped = MCPWrappedTool(
                mcp_tool=self,
                tool_info=tool_info,
                prefix=self.prefix,
            )
            expanded.append(wrapped)

        return expanded

    # ── 执行 ──────────────────────────────────────────────

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行 MCP 操作

        Args:
            parameters: 包含以下可选参数的字典
                - action: 操作类型 (list_tools / call_tool / list_resources /
                  read_resource / list_prompts / get_prompt)
                  不指定但提供了 tool_name 时自动推断为 call_tool
                - tool_name: 工具名称（call_tool 需要）
                - arguments: 工具参数（call_tool 需要）
                - uri: 资源 URI（read_resource 需要）
                - prompt_name: 提示词名称（get_prompt 需要）
                - prompt_arguments: 提示词参数（get_prompt 可选）

        Returns:
            操作结果字符串
        """
        # 智能推断 action
        action = parameters.get("action", "").lower()
        if not action and "tool_name" in parameters:
            action = "call_tool"

        if not action:
            return "错误：必须指定 action 参数或 tool_name 参数"

        async def _run_operation() -> str:
            async with _MCPClient(
                self.server_command, self.server_args, env=self.env
            ) as client:
                # ── list_tools ──
                if action == "list_tools":
                    tools = await client.list_tools()
                    if not tools:
                        return "没有找到可用的工具"
                    result = f"找到 {len(tools)} 个工具:\n"
                    for tool in tools:
                        result += f"- {tool['name']}: {tool['description']}\n"
                    return result

                # ── call_tool ──
                elif action == "call_tool":
                    tool_name = parameters.get("tool_name")
                    arguments = parameters.get("arguments", {})
                    if not tool_name:
                        return "错误：必须指定 tool_name 参数"
                    result = await client.call_tool(tool_name, arguments)
                    return f"工具 '{tool_name}' 执行结果:\n{result}"

                # ── list_resources ──
                elif action == "list_resources":
                    resources = await client.list_resources()
                    if not resources:
                        return "没有找到可用的资源"
                    result = f"找到 {len(resources)} 个资源:\n"
                    for resource in resources:
                        result += f"- {resource['uri']}: {resource['name']}\n"
                    return result

                # ── read_resource ──
                elif action == "read_resource":
                    uri = parameters.get("uri")
                    if not uri:
                        return "错误：必须指定 uri 参数"
                    content = await client.read_resource(uri)
                    return f"资源 '{uri}' 内容:\n{content}"

                # ── list_prompts ──
                elif action == "list_prompts":
                    prompts = await client.list_prompts()
                    if not prompts:
                        return "没有找到可用的提示词"
                    result = f"找到 {len(prompts)} 个提示词:\n"
                    for prompt in prompts:
                        result += f"- {prompt['name']}: {prompt['description']}\n"
                    return result

                # ── get_prompt ──
                elif action == "get_prompt":
                    prompt_name = parameters.get("prompt_name")
                    prompt_arguments = parameters.get("prompt_arguments", {})
                    if not prompt_name:
                        return "错误：必须指定 prompt_name 参数"
                    messages = await client.get_prompt(prompt_name, prompt_arguments)
                    result = f"提示词 '{prompt_name}':\n"
                    for msg in messages:
                        result += f"[{msg['role']}] {msg['content']}\n"
                    return result

                else:
                    return f"错误：不支持的操作 '{action}'"

        try:
            return self._run_async(_run_operation())
        except Exception as e:
            return f"MCP 操作失败: {str(e)}"

    # ── 参数定义 ──────────────────────────────────────────

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: list_tools, call_tool, list_resources, "
                "read_resource, list_prompts, get_prompt",
                required=True,
            ),
            ToolParameter(
                name="tool_name",
                type="string",
                description="工具名称（call_tool 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="arguments",
                type="object",
                description="工具参数（call_tool 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="uri",
                type="string",
                description="资源 URI（read_resource 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="prompt_name",
                type="string",
                description="提示词名称（get_prompt 操作需要）",
                required=False,
            ),
            ToolParameter(
                name="prompt_arguments",
                type="object",
                description="提示词参数（get_prompt 操作可选）",
                required=False,
            ),
        ]

    @property
    def available_tools(self) -> List[Dict[str, Any]]:
        """获取已发现的工具列表（只读）"""
        return list(self._available_tools)
