import asyncio
import json
import re
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


TASK_SPLIT_TEMPLATE = """
你是一个任务拆分助手。你将根据用户提供的md模板数据拆分成对应的子任务，仔细理解模板中的标题信息和模板任务信息，根据
提供的内容来拆分任务:

模板数据：
%s

拆分规则
1. 逐级遍历md模板中的每一级菜单，仔细理解标题和上下文中提供的信息
2. 只关注标题下面是否含有`{}`占位符包裹的信息，如果没有包含`{}`的菜单则不提取任何信息置空
3. 输出的格式按照树形json格式进行返回

响应格式
``json
[
    {
        "title": "",
        "task": "",
        "children": [
            {
                "title": "",
                "task": "",
                "children": []
            }
        ]
    }
]
```

注意点
1. 严格按照json格式进行数据的输出
2. 标题内容不要进行任何修改原封不动的输出
"""

TOOL_PROMPT = """
你是一个工具调用助手。你将根据用户提供的函数工具和对应的任务信息来选择调用对应的工具：

在使用工具时，请遵循以下步骤：
- 根据任务需求选择合适的工具
- 按照工具的参数要求提供正确的参数
- 观察工具的返回结果，并根据结果决定下一步操作
- 工具可能会发生变化，比如新增工具或现有工具消失

请遵循以下指南：
- 使用工具时，确保参数符合工具的文档要求
- 如果出现错误，请理解错误原因并尝试用修正后的参数重新调用
- 按照任务需求逐步完成，优先选择最合适的工具
- 如果需要连续调用多个工具，请一次只调用一个工具并等待结果
- 根据返回的结果需要进行判断是否满足任务的需求，如果不满足则继续调用对应的工具
- 如果工具都不支持，则返回空数据

注意：
- 禁止一次性调用多个工具，根据任务只能一次调用一个工具并且等待后续结果
- 禁止输出与工具调用无关的内容
"""

WRITER_PROMPT = """
你是一位专业的数据模板处理引擎，你需要根据用户提供的数据和模板进行对应的数据的填充，严格按照以下规则执行任务：

输入规范：
1. 用户将提供以下格式的数据：
```json
{
    "md_template": "markdown格式的模板数据",
    "user_input": "用户输入信息",
    "json_data": [
        {
            "title": "标题内容",
            "task": "{}中的任务描述",
            "data": []
        }
    ]
}
```
2. 模板中的填充符格式为：`{任务描述}`
3. json_data中为扁平化的数据与任务描述相关联，data则为模板中需要用到的数据信息

填充规则：
1. 逐级扫描模板中的包含`{}`占位符的菜单，并且根据`{}`占位符的标题到`json_data`中提取对应的数据填充到对应的`{}`占位符中，填充格式需要遵循以下要求：
- 标题涉及到 `xxx` 字符的情况下，需要根据上下文信息进行替换为对应的数据，例如：xxx周报，输入信息为2025年7月，则需要替换为2025年7月周报，否则不需要进行替换
- 如果是多张图片链接等数据，在插入模板数据的时候对图片进行合理的布局并且限制每张链接图片的大小并且保持一致
- 如果是数字类型的数据类型，在插入模板时使用加粗标红等语法对数字数据进行重点标注出来
- 如果提供的是json格式的数据，则对提供的json格式数据进行分析和归纳后再进行填充

2. 填充数据时允许对填充的数据进行数据的简单的描述和总结，总结格式需要遵循以下要求：
- 如果是图片链接，总结和描述需要跟随在图片后面，描述和总结必须清晰有逻辑且专业，禁止出现：无法、可能、也许等不确定字符
- 如果是json类型的格式数据，则需要根据任务描述的逻辑对其进行总结，并且在关键的地方引用查询出来的数据进行引用总结

注意：
1. 禁止修改模板中的标题信息，只做数据替换与`{}`占位符数据替换
2. 注意模板格式，并且随意进行数据的填充
3. 
"""

#BASE_URL = "http://192.168.2.236:10000/v1"
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
#MODEL_NAME = "Qwen2.5-7B-Instruct"
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

    async def task_splice(self, md_template: str, user_input: str) -> List[Dict]:
        """
        使用 TASK_SPLIT_TEMPLATE 提示词对模板进行拆分
        :param md_template: markdown模板
        :param user_input: 用户输入信息
        :return: 拆分任务的JSON数据
        """
        messages = [{
            "role": "system",
            "content": TASK_SPLIT_TEMPLATE % md_template,
        }, {
            "role": "user",
            "content": user_input,
        }]
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llmClient.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
            )
            content = response.choices[0].message.content
            if content is None or content == '':
                return []
            content = intercept_str(content, '```json', '```')
            return json.loads(content)
        except Exception as e:
            raise RuntimeError(f"任务拆分失败：{str(e)}") from e

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

    def get_tools(self):
        return self.tools

    async def tools_calling(self, tasks: List[Dict[str, Any]],
                            md_template: str = '', user_input: str = '') -> List[Dict[str, Any]]:
        """
        处理工具调用循环
        :param tasks: task_splice返回的任务列表
        :param md_template: markdown模板
        :param user_input: 用户查询信息
        :return: 组装的数据（符合WRITER_PROMPT格式）
        """
        messages = [{
            "role": "system",
            "content": TOOL_PROMPT,
        }, {
            "role": "user",
            "content": user_input
        }]
        result = []
        for task in tasks:
            await self.call_llm_and_tools(task, messages, result)

        return result

    async def call_llm_and_tools(self, task: Dict, messages: List, result: List):
        if task['children'] is not None and len(task['children']) > 0:
            for child in task['children']:
                await self.call_llm_and_tools(child, messages, result)
        if task['task'] is None or task['task'] == '':
            return
        messages.append(self.build_message(task))
        r = await self.call_model(messages, result)
        result.append({
            'title': task['title'],
            'task': task['task'],
            'data': r,
        })


    async def call_model(self, messages: List, result: List):
        tools = await self.get_mcp_tools()
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.llmClient.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=[*tools, *self.get_tools()],
                tool_choice="auto"
            )
        )
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        tool_calls = assistant_message.tool_calls
        if tool_calls is None or len(tool_calls) == 0:
            return assistant_message.content
        for tool in tool_calls:
            function_name = tool.function.name
            params = tool.function.arguments
            exists_with_name = any(t['function']['name'] == function_name for t in tools)
            if exists_with_name:
                async with self.mcpClient:
                    tool_result = await self.mcpClient.call_tool(function_name, json.loads(params))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "content": str(tool_result.content[0].text)
                    })
            else:
                tool_result = self.invoke_tool(function_name, json.loads(params))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "content": str(tool_result)
                })
        return await self.call_model(messages, result)

    def invoke_tool(self, function_name: str, param: Dict[str, Any]):
        if function_name == 'get_current_weather':
            return [
                1, 5, 6, 20, 5, 8, 7
            ]
        if function_name == 'get_concurrent_time':
            # 返回当前yyyy-MM-dd HH:mm:ss 的时间
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def build_message(self, task: Dict):
        t = task['task']
        return {
            'role': 'user',
            'content': t
        }

    async def writer(self, data: Dict[str, Any]) -> str:
        """
        使用 WRITER_PROMPT 提示词处理数据并生成字符串
        :param data: 需要写入的数据（符合WRITER_PROMPT格式）
        :return: 生成的字符串
        """
        messages = [{
            "role": "system",
            "content": WRITER_PROMPT,
        }, {
            "role": "user",
            "content": f"请根据以下提供数据对模板数据进行处理：\n\n{json.dumps(data, indent=2)}",
        }]

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llmClient.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
            )
            content = response.choices[0].message.content

            # 提取字符串内容
            return content

        except Exception as e:
            raise RuntimeError(f"生成字符串失败：{str(e)}") from e

    async def file_create(self, content: str, user_input: str) -> str:
        """
        创建文件
        :param content: 要写入的内容
        :param user_input: 用户查询信息
        :return: 创建的文件名
        """
        # 让大模型生成文件名
        messages = [{
            "role": "system",
            "content": "你是一个文件命名助手，请根据用户的请求生成一个合适的md文档的文件名称。",
        }, {
            "role": "user",
            "content": f"""
            请为以下内容生成一个简洁且具有描述性的文件名（不要包含任何特殊字符）：
            
            用户输入问题：{user_input}
            
            内容数据：{content}
            """,
        }]

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.llmClient.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
            )
            filename = response.choices[0].message.content.strip()
            # 确保文件名合法
            filename = re.sub(r'[\\/*?:"<>|]', "", filename)
            # 写入文件
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            return filename

        except Exception as e:
            raise RuntimeError(f"创建文件失败：{str(e)}") from e


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
    result = await mcp_client.task_splice(template, user_input)
    data = await mcp_client.tools_calling(result, user_input)
    fill_template = await mcp_client.writer({
        "md_template": template,
        "user_input": user_input,
        "json_data": data
    })
    await mcp_client.file_create(fill_template, user_input)


def get_md_template():
    with open("./xxx月工单周报.md", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == '__main__':
    asyncio.run(main())
