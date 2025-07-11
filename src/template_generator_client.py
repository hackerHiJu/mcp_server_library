import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any

from fastmcp import Client
from openai import OpenAI


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
你是一个专业的模板处理助手。你将根据用户提供的模板数据来完成对应数据的提取和处理

# 任务目标
1. 逐级浏览提供的md模板数据，并且将其转换为树形结构
2. 在浏览每一级的过程中，提取#{}占位符中的数据，并且根据其中的描述生成精简的英文的key，生成的key为下划线进行分割
3. 提取了#{}中的描述后，将模板中的原本的描述替换为生成的英文的key。例如：#{请帮我查询一下计划数据}，替换后#{query_plan}
4. 生成key后进行模板检查是否出现了重复，要确保整个模板key是唯一
5. 每个层级的描述只能生成一个key
6. 如果层级没有提取到#{}的数据，则返回空字符串并且不要生成key
7. 标题和任务描述需要原封不动的提取数据，禁止修改标题和任务描述信息

# 输出结构
```json
{
    "template": "对占位符进行替换后的模板字符串",
    "tasks": [
        {
            "title": "每个层级的标题",
            "task_desc": "任务描述提取`#{}`占位符所包裹的信息，没有则为空",
            "key": "替换后的英文key",
            "children": [
                {
                    "title": "标题",
                    "task_desc": "任务描述",
                    "key": "替换后的英文key",
                    "children": []
                }
            ]
        }
    ]
}
```
"""

TOOL_CALL_REPORT = """
# 角色
你是一个工具调用助手，你将根据用的需求任务来选择调用对应的函数工具。

# 工具调用要求
1. 仔细的对任务进行拆分并且向用户解释的每一个步骤和推理过程
2. 如果需要调用工具，那么就在最后加上需要调用的函数，每次只能返回一个工具，以json格式，例如：```json { "function_name": "函数名称", "params": { "param1": "value1", "param2": "value2" }} ```
3. 按照工具的参数要求并且结合提供的任务需求来提供正确的参数
4. 观察工具的返回结果，并根据结果决定下一步操作
5. 工具可能会发生变化，比如新增工具或现有工具消失
6. 如果工具调用出现异常或者为空则需要判断参数是否合理并且重新进行调整，再次调用或者更换其他合适的工具
7. 时间类型的参数优先调用列表中获取当前最新时间的工具

# 可用工具:
%s
"""

NEXT_STEP_PROMPT = """
# 任务目标
根据已经获取的信息，判断当前获取到的数据是否满足当前任务所需数据：

# 任务要求
1. 如果可以解决(满足用户给出的条件和范围)，根据已经提供的数据和上下文的调用过程对用户的需求进行总结和归纳并且报告数据
- 如果数据是图片链接则根据需求内容进行总结，并且总结跟随在图片链接后面
- 如果是多个图片链接则使用md的语法对图片进行排版并且限制图片的大小为相同大小
- 如果是数字类型的数据则对数据进行归纳整理并且对关注数据通过md语法对其进行高亮
2. 如果缺少数据或内容，请继续结合上下文中提供的工具调用合适的工具获取更多信息并且加上需要调用的工具，
每次只能返回一个工具，以json格式，例如：```json { "function_name": "函数名称", "params": { "param1": "value1", "param2": "value2" }} ```
3. 同一个工具如果调用3次还获取不到数据，则使用友好专业的语气返回数据查询为空

# 收集的数据：
%s

# 任务需求
%s
"""

BASE_URL = "http://192.168.2.236:10000/v1"
# BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# MODEL_NAME = "Qwen2.5-7B-Instruct"
# MODEL_NAME = "glm-4-flash"
MODEL_NAME = "Qwen3-8B"
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
                "content": f"# 模板数据\n\n{template}",
            }
        ]
        response = self.llmClient.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            top_p=0.8,
            temperature=0.7,
            extra_body={
                "tok_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )
        messages.append(response.choices[0].message)
        content = response.choices[0].message.content
        if content.startswith("```json"):
            content = intercept_str(response.choices[0].message.content, "```json", "```")
        replace_template = json.loads(content)

        template = replace_template['template']
        task_list = replace_template['tasks']
        results = []
        available_tools = json.dumps([*await self.get_mcp_tools(), *self.tools], ensure_ascii=False)
        tool_call_messages = [
            {
                "role": "system",
                "content": TOOL_CALL_REPORT % available_tools,
            }
        ]
        for task in task_list:
            await self.process_task(task, tool_call_messages, results)

        for result in results:
            key = result["key"]
            # 替换字符串中 #{} 中标识的数据
            template = template.replace(f"#{key}", result["content"])
        return template

    async def process_task(self, task: Dict, messages: List, results: List):
        task_desc = task['task_desc']
        if task_desc is not None and task_desc != "":
            await self.invoke_task(task, messages, results)
        child_task = task['children']
        if child_task is not None and len(child_task) > 0:
            for child in task['children']:
                await self.process_task(child, messages, results)

    async def invoke_task(self, task: Dict, sys_messages: List, results: List):
        task_desc = task['task_desc']
        task_key = task['key']
        messages = [*sys_messages,
                    {
                        "role": "user",
                        "content": f"用户需求:\n{task_desc}",
                    }]

        response = self.llmClient.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            extra_body={
                "tok_k": 20,
                "min_p": 0,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )
        tool_call_result = []
        while True:
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            content = assistant_message.content
            tool = None
            if "```json" in content:
                tool = json.loads(intercept_str(assistant_message.content, "```json", "```"))

            if tool is None:
                results.append({
                    "key": task_key,
                    "content": assistant_message.content
                })
                break
            else:
                tool_result = None
                tool_name = None
                try:
                    tool_name, tool_result = await self.tool_call(tool)
                except Exception as e:
                    tool_result = str(e)
                    pass
                if tool_name is not None:
                    messages.append({
                        "role": "user",
                        "content": f"""
                            工具名称：{tool_name}
                            调用结果: {str({tool_result})}
                            """
                    })
                tool_call_result.append(tool_result)

            messages.append({
                "role": "user",
                "content": NEXT_STEP_PROMPT % (json.dumps(tool_call_result, ensure_ascii=False), task_desc)
            })

            response = self.llmClient.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_body={
                    "tok_k": 20,
                    "min_p": 0,
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                },
            )

    async def tool_call(self, tool):
        function_name = tool['function_name']
        params = tool['params']
        if function_name == 'get_current_weather':
            return function_name, [1, 5, 6, 20, 5, 8, 7]
        if function_name == 'get_concurrent_time':
            # 返回当前yyyy-MM-dd HH:mm:ss 的时间
            return function_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tool_result = None
        async with self.mcpClient:
            mcp_result = await self.mcpClient.call_tool(function_name, params)
            if mcp_result.content is not None and len(mcp_result.content) > 0:
                tool_result = mcp_result.content[0].text

        return function_name, tool_result

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
                "url": "https://mcp.api-inference.modelscope.net/6e05b4133c474e/sse"
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
                "description": "获取到当前时间",
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
