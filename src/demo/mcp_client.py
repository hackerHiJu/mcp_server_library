import asyncio
import json
from typing import Optional
from urllib.parse import unquote

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from contextlib import AsyncExitStack

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-max"
KEY = "sk-be221ba48a924603bb8271743ee860a0"


class MCPClient:

    def __init__(self):
        """初始化MCP客户端"""
        self.openai_api_key = KEY
        self.openai_api_base = BASE_URL
        self.model = MODEL_NAME

        if not self.openai_api_key:
            raise ValueError("请设置您的OpenAI API密钥")

        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    # 处理对话请求
    async def process_query(self, query: str) -> str:
        messages = [{
            "role": "system",
            "content": "你是一个智能助手，帮助用户回答问题",
        }, {
            "role": "user",
            "content": query,
        }]

        # 获取到工具列表
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            }
            for tool in response.tools]

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=available_tools,
                )
            )
            content = response.choices[0]
            if content.finish_reason == "tool_calls":
                # 如果使用的是工具，解析工具
                tool_call = content.message.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # 执行工具
                result = await self.session.call_tool(tool_name, tool_args)
                print(f"\n\n工具调用:[{tool_name}]，参数:[{tool_args}]")

                # 将工具返回结果存入message中，model_dump()克隆一下消息
                messages.append(content.message.model_dump())
                messages.append({
                    "role": "tool",
                    "content": result.content[0].text,
                    "tool_call_id": tool_call.id,
                })

                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )

                return response.choices[0].message.content

            # 正常返回
            return content.message.content
        except Exception as e:
            return f"调用OpenAI API错误：{str(e)}"

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器的连接 """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器脚本路径必须以 .py 或 .js 结尾")

        command = "python" if is_python else "node"

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        # 初始化会话
        await self.session.initialize()
        # 列出工具
        tool_response = await self.session.list_tools()
        print("\n已经连接到服务器，支持以下工具：", [tools.name for tools in tool_response.tools])

        # 查询资源
        resources = await self.session.list_resources()
        print("\n已经连接到服务器，支持以下资源：", [resources.name for resources in resources.resources])

        # 查询资源
        templates = await self.session.list_resource_templates()
        print("\n已经连接到服务器，支持以下模板资源：", [resources.name for resources in templates.resourceTemplates])

        resource_result = await self.session.read_resource("echo://hello")
        for content in resource_result.contents:
            print(f"读取资源内容：{unquote(content.text)}")

        resource_result = await self.session.read_resource("echo://张三/18")
        for content in resource_result.contents:
            print(f"读取资源内容：{unquote(content.text)}")

        # 查询提示词
        prompt_result = await self.session.list_prompts()
        print("\n已经连接到服务器，支持以下提示词：", [prompt.name for prompt in prompt_result.prompts])

        get_prompt = await self.session.get_prompt("review_code", { "code": "hello world"})
        for message in get_prompt.messages:
            print(f"提示词内容：{message}")

        # 调用图片工具
        image = await self.session.call_tool("create_thumbnail", {"image_url": "/Users/haijun/Documents/图片/WechatIMG47.jpg"})
        print("读取图片资源：", image)

        # 调用天气工具使用Context对象
        weather = await self.session.call_tool(name="test", arguments={"city": "北京"})
        print("天气信息：", weather)

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\n MCP客户端已启动！输入 ‘quit’ 退出")

        while True:
            try:
                user_input = input("请输入您的问题：").strip()
                if user_input.lower() == "quit":
                    print("退出交互式聊天")
                    break
                response = await self.process_query(user_input)
                print(f"大模型：{response}")
            except Exception as e:
                print(f"发生错误：{str(e)}")

    async def cleanup(self):
        """清理资源"""
        print("Cleaning up resources...")
        await self.exit_stack.aclose()


async def main():
    mcp_client = MCPClient()

    try:
        await mcp_client.connect_to_server("./mcp_lower_server.py")
        # await mcp_client.chat_loop()
    finally:
        await mcp_client.cleanup()


if __name__ == '__main__':
    asyncio.run(main())
