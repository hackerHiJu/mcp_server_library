import asyncio
import json

from fastmcp import Client
from openai import OpenAI

BASE_URL = "http://192.168.2.236:10000/v1"
MODEL_NAME = "Qwen3-32B-AWQ"
KEY = "18e567651c50428c8009f2191aa9773f.A5QNRyfcFveC1RVZ"


class MCPClient:

    def __init__(self, mcp_config):
        """初始化MCP客户端"""
        self.openai_api_key = KEY
        self.openai_api_base = BASE_URL
        self.model = MODEL_NAME

        if not self.openai_api_key:
            raise ValueError("请设置您的OpenAI API密钥")

        self.llm_client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.session = Client(mcp_config)

    # 处理对话请求
    async def process_query(self, query: str) -> str:
        messages = [{
            "role": "system",
            "content": "你是一个工具调用助手，你将根据用户的需求来执行对应的工具来查询数据库数据",
        }, {
            "role": "user",
            "content": query,
        }]

        # 获取到mcp所有的工具信息
        mcp_tools = await self.get_mcp_tools()
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=mcp_tools,
                    tool_choice="auto",
                )
            )
            assistant_message = response.choices[0].message
            tools = assistant_message.tool_calls
            if tools is not None and len(tools) > 0:
                for tool in tools:
                    function_name = tool.function.name
                    arguments = tool.function.arguments
                    result = ""
                    async with self.session:
                        tool_result = await self.session.call_tool(function_name, json.loads(arguments))
                        result = tool_result.content[0].text


                    print(f"\n\n工具调用:[{function_name}]，参数:[{arguments}]")
                    # 将工具返回结果存入message中，model_dump()克隆一下消息
                    messages.append(assistant_message)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool.id,
                    })

            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            # 正常返回
            return response.choices[0].message.content
        except Exception as e:
            return f"调用OpenAI API错误：{str(e)}"

    async def get_mcp_tools(self):
        """ 获取到所有的mcp工具 """
        if self.session is None:
            return []
        # 获取到mcp所有的工具信息
        available_tools = []

        async with self.session:
            mcp_tools = await self.session.list_tools()
            for tool in mcp_tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })
        return available_tools

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


async def main():
    config = {
        "mcpServers": {
            "zh-reservoir-new-mcp-server": {
                "type": "sse",
                "url": "http://192.168.2.240:20410/sse"
            },
        }
    }
    mcp_client = MCPClient(config)
    tools = await mcp_client.get_mcp_tools()
    # async with mcp_client.session:
    #     # await mcp_client.session.call_tool("connect_db", arguments={
    #     #     "host": "mysql.server",
    #     #     "user": "zhxx",
    #     #     "password": "zhxx@123456",
    #     #     "database": "zh_reservoir_new"
    #     # })
    async with mcp_client.session:
        result = await mcp_client.session.call_tool("queryEvent", {
            "eventMcpQuery": {
                "startTime": "2025-07-29 00:00:00",
                "endTime": "2025-07-29 23:59:00",
                "type": 3
            }
        })
        print(result)
    # await mcp_client.chat_loop()


if __name__ == '__main__':
    asyncio.run(main())
