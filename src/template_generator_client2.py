import asyncio
import json
import re
from datetime import datetime
from typing import Dict, List, Any

from fastmcp import Client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall


def intercept_str(target: str, prefix: str, suffix: str):
    """
    截取指定字符串开头和指定字符串结尾的子字符串
    :param target: 目标字符串
    :param prefix: 开头字符串
    :param suffix: 结尾字符串
    :return: 截取后的子字符串，如果没有匹配则返回空字符串
    """
    start_index = target.find(prefix)
    if start_index == -1:
        return ""

    end_index = target.find(suffix, start_index + len(prefix))
    if end_index == -1:
        return ""
    return target[start_index + len(prefix):end_index]


TASK_SPLICE_PROMPT = """
# 角色
你是一个报告任务拆分助手。

# 任务
你将根据用户提供的md模板来对任务进行拆分，逐级提取模板中`{}`占位符所描述的任务，并且形成树形结构的json数据

# 输出结构
```json
[
    {
        "title": "标题",
        "task_desc": "任务描述",
        "children": [
            {
                "title": "标题",
                "task_desc": "任务描述",
                "children": []
            }
        ]
    }
]

# 数据模板
```
"""

CALL_TOOL_PROMPT = """
# 角色
你是一个专业的工具调用助手。

# 任务
你将根据用户提供的json格式任务需求来使用服务器提供的工具来进行数据收集

# 要求
1. 根据任务需求优先选择最合适的工具
2. 按照工具的参数要求提供正确的参数
3. 观察工具的返回结果，并根据结果决定下一步操作
4. 工具可能会发生变化，比如新增工具或现有工具消失
5. 一次只能返回一个工具调用

# 用户需求
%s
"""

NEXT_STEP_PROMPT = """
# 任务
根据已经获取的信息，判断当前获取到的数据是否满足当前任务所需数据：

# 任务要求
- 如果可以解决(满足用户给出的条件和范围)，请输出<finish>
- 如果缺少数据或内容，请继续调用合适的工具获取更多信息

# 任务需求
%s
"""

FINISH_GENETATE = '''
# 任务目标
根据用户提供的需求和模板信息结合已经收集到的数据信息完成报告的生成

# 已收集信息
%s

# 任务要求
1、请根据图片的描述将图片链接插入到合适的位置，如果没有符合要求的图片，请不要插入图片
2、以模板为基础生成对应的报告，数据只需要对占位符进行替换

# 模板数据
%s

# 用户需求
%s
'''

# BASE_URL = "http://192.168.2.236:10000/v1"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# MODEL_NAME = "Qwen2.5-7B-Instruct"
MODEL_NAME = "glm-4-flash"
KEY = "18e567651c50428c8009f2191aa9773f.A5QNRyfcFveC1RVZ"


class MCPClient:
    def __init__(self, mcp_config: Dict[str, Any] = {}, tools: List[Dict[str, Any]] = []):
        """初始化MCP客户端"""
        self.openai_api_key = KEY
        self.openai_api_base = BASE_URL
        self.model = MODEL_NAME

        if not self.openai_api_key:
            raise ValueError("请设置您的OpenAI API密钥")

        self.llmClient = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        if mcp_config is not None:
            self.mcpClient = Client(mcp_config)

        self.tools = tools

    async def generator_report(self, template: str, user_input: str):
        """ 根据指定的模板信息生成对应的报告信息 """
        messages = [
            {
                "role": "system",
                "content": TASK_SPLICE_PROMPT,
            },
            {
                "role": "user",
                "content": f"""
                # 模板信息：
                {template}
                """
            }
        ]
        response = self.llmClient.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        json_content = intercept_str(response.choices[0].message.content, "```json", "```")
        task_list = json.loads(json_content)

        results = []
        for task in task_list:
            task_desc = task['task_desc']
            if task_desc is not None and task_desc != "":
                await self.invoke_task(task_desc, task, messages, results)
                if task['children'] is not None and len(task['children']) > 0:
                    for child in task['children']:
                        await self.invoke_task(child['task_desc'], child, messages, results)

        messages.append({
            "role": "user",
            "content": FINISH_GENETATE % (results, template, user_input)
        })

        response = self.llmClient.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        message = response.choices[0].message.content
        return message

    async def invoke_task(self, task_desc: str, task: Dict, messages: List, results: List):
        mcp_tools = await self.get_mcp_tools()
        sys_tools = self.tools
        messages.append({
            "role": "user",
            "content": CALL_TOOL_PROMPT % task_desc,
        })
        response = self.llmClient.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=[*sys_tools, *mcp_tools],
            tool_choice="auto",
        )
        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls
        tool_result = None
        tool_id = None
        if tool_calls is not None and len(tool_calls) > 0:
            for tool in tool_calls:
                tool_id, tool_result = self.tool_call(tool)

        messages.append(assistant_message)
        if tool_id is not None:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(tool_result)
            })
            results.append(tool_result)
        messages.append({
            "role": "user",
            "content": NEXT_STEP_PROMPT % task_desc
        })
        response = self.llmClient.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        message = response.choices[0].message
        if 'finish' in message.content:
            return
        messages.append({
            "role": "assistant",
            "content": message.content
        })
        return self.invoke_task(task_desc, task, messages, results)

    async def tool_call(self, tool: ChatCompletionMessageToolCall):
        tool_id = tool.id
        function_name = tool.function.name
        params = tool.function.arguments
        if function_name == 'get_current_weather':
            is_sys_tool = [1, 5, 6, 20, 5, 8, 7]
        if function_name == 'get_concurrent_time':
            # 返回当前yyyy-MM-dd HH:mm:ss 的时间
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        async with self.mcpClient:
            mcp_result = await self.mcpClient.call_tool(function_name, json.loads(params))
            tool_result = mcp_result.content[0].text

        return tool_id, tool_result

    async def get_mcp_tools(self):
        if self.mcpClient is None:
            return []
        # 获取到mcp所有的工具信息
        available_tools = []

        async with self.mcpClient:
            mcp_tools = await self.mcpClient.list_tools()
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


async def main():
    config = {
        "mcpServers": {
            "mcp-server-chart": {
                "type": "sse",
                "url": "https://mcp.api-inference.modelscope.net/c77f34d86ffe4d/sse"
            },
            "mcp_drone_order_server": {
                "transport": "stdio",
                "command": "python3",
                "args": ["./mcp_drone_order_server.py", "--verbose"],
            }
        }
    }

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_concurrent_time",
                "description": "获取到当前系统时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                    },
                },
            }
        }
    ]

    mcp_client = MCPClient(mcp_config=config, tools=tools)
    template = get_md_template()
    user_input = "生成本周工作周报"
    data = await mcp_client.generator_report(template, user_input)

    with open("./报告.md", "w", encoding="utf-8") as f:
        f.write(data)


def get_md_template():
    with open("./xxx月工单周报.md", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == '__main__':
    asyncio.run(main())
