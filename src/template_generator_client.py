import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Iterable, Callable, Self

from fastmcp import Client
from mcp.types import CallToolResult
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessage, ChatCompletionMessageToolCall, ChatCompletionMessageParam


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


def repair_json_str(target: str) -> str:
    """ 修复json中md字符串的格式 """
    return target.replace('\n', '\\n').replace('"', '\\"')


BASE_URL = "http://192.168.2.236:10000/v1"
# BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# MODEL_NAME = "glm-4-flash"
MODEL_NAME = "Qwen3-32B-AWQ"
KEY = "18e567651c50428c8009f2191aa9773f.A5QNRyfcFveC1RVZ"


class SysFunctionDefinition:
    def __init__(self, name: str, description: str, callable: Callable[..., ...]):
        self.name = name
        self.description = description
        self.callable = callable

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)


class FunctionCaller:

    def __init__(self, mcp_client: Client, sys_function: List[SysFunctionDefinition] = None):
        self.mcp_client = mcp_client
        self.sys_function = sys_function

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> CallToolResult:
        """
        调用工具
        :param tool_name: 工具名称
        :param params: 工具参数
        :return: 工具返回结果
        """
        for function in self.sys_function:
            if function.name == tool_name:
                return function(**params)
        async with self.mcp_client:
            result = await self.mcp_client.call_tool(tool_name, params)
            return result.content[0].text

    async def get_tools_list(self):
        """ 获取到所有的mcp工具 """
        if self.mcp_client is None:
            return []
            # 获取到mcp所有的工具信息

        available_tools = []

        for function in self.sys_function:
            available_tools.append(json.dumps(function.description, ensure_ascii=False))

        async with self.mcp_client:
            mcp_tools = await self.mcp_client.list_tools()
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


class Task:
    def __init__(self, title: str, task_desc_list: List[str], children=None):
        self.title = title
        self.task_desc_list = task_desc_list or []
        self.children = children or []


class MessageContext:
    template: str
    replace_template: str
    tasks: List[Task]

    def __init__(self, template: str, tasks: List[Task]):
        self.template = template
        self.tasks = tasks

    def render(self, results: List) -> str:
        for result in results:
            key = result["key"]
            content = result["content"]
            self.replace_template = self.replace_template.replace(f"#{{{key}}}", content)

        return self.replace_template


class BaseLLMClient:
    function_caller: FunctionCaller

    def __init__(self, llm_client: OpenAI, function_caller: FunctionCaller):
        self.llm_client = llm_client
        self.function_caller = function_caller

    async def invoke(self, messages: List[ChatCompletionMessageParam]) -> ChatCompletionMessage:
        tools = await self.get_tools()
        if tools is not None:
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
                tools=tools,
                tool_choice="auto",
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
        else:
            response = self.llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=False,
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
        return response.choices[0].message

    async def get_tools(self) -> Iterable[ChatCompletionToolParam]:
        pass

    async def invoke_tool(self, tools: List[ChatCompletionMessageToolCall]):
        result = []
        for tool in tools:
            tool_name = tool.function.name
            params = tool.function.arguments
            result.append(await self.function_caller.call_tool(tool_name, json.loads(params)))


class TaskSplicer(BaseLLMClient):
    prompt = """
        # 角色
        你是一个专业的模板处理助手。你将根据用户提供的模板数据来完成对应数据的提取和处理
        
        # 任务目标
        1. 逐级浏览提供的md模板数据，并且将其转换为树形结构
        2. 在浏览每一级的过程中，提取#{}占位符中的数据
        3. 每个菜单下可能会有多个`#{}`占位符的出现，注意提取的树形结构
        4. 如果层级没有提取到#{}的数据，则返回空字符串并且不要生成key
        5. 任务描述需要原封不动的提取数据，禁止修改标题和任务描述信息
        6. 提取的数据需要注意树形的结构，每一级标题都需要单独进行提取
        7. 确保输出时为标准并且正确的json格式，注意md模板数据中转义符等格式问题
        
        # 输出结构
        字段说明：
        - title：每个菜单的标题
        - task_desc_list：每一级菜单下面所有的#{}占位符标识的任务列表，#{}占位符中的描述信息，原封不动提取禁止修改，如果当前菜单下面没有#{}占位符则保持为空
        - children：子菜单中包含的菜单数据，其中的title、task_desc_list均为一样，如果没有子级菜单直接返回空数组结构
            
        
        ```json
       [
            {
                "title": "",
                "task_desc_list": [],
                "children": [
                    {
                        "title": "",
                        "task_desc_list": [],
                        "children": []
                    }
                ]
            }
        ]
        ```
        
    # 示例
    ```md
    # 一、xxx计划
    ## 1. 查询数据
    #{查询数据}
    #{计算数据}
    ## 2. 查询历史数据
    #{查询数据}
    ### 2.1 统计历史数据
    #{统计历史数据}
    ```
    输出数据
    ```json
    [
        {
            "title": "一、xxx计划",
            "task_desc_list": [],
            "children": [
                {
                    "title": "1. 查询数据",
                    "task_desc_list": ["查询数据", "计算数据"],
                    "children": []
                },
                {
                    "title": "2. 查询历史数据",
                    "task_desc_list": ["查询数据"],
                    "children" : [
                        "title": "2.1 统计历史数据",
                        "task_desc_list": ["统计历史数据"],
                        "children" : []
                    ]
                }
            ]
        }
    ]
    ```
    """

    def __int__(self, llm_client: OpenAI, function_caller: FunctionCaller):
        super().__init__(llm_client, function_caller)

    async def splice_template(self, template: str) -> MessageContext:
        """ 拆分模板 """
        messages = [
            {
                "role": "system",
                "content": self.prompt,
            },
            {
                "role": "user",
                "content": f"# 模板数据\n\n{template}",
            }
        ]
        assistant_message = await self.invoke(messages)
        content = ''
        if assistant_message.content.startswith("```json"):
            content = intercept_str(assistant_message.content, "```json", "```")
        data = json.loads(content)
        return MessageContext(template=template, tasks=self.recursive_transformations(data))

    def recursive_transformations(self, data: List[Dict]) -> List[Task]:
        results = []
        for d in data:
            task = Task(**d)
            task.children = self.recursive_transformations(d.get('children', []))
            results.append(task)
        return results


class TaskInvoker(BaseLLMClient):
    tool_call_prompt = """
        # 角色
        你是一个工具调用助手，你将根据用的需求任务来选择调用对应的函数工具，拆分任务步骤依据顺序选择优先调用工具。
        
        # 工具调用要求
        1. 仔细的对任务进行拆分并且向用户解释的每一个步骤和推理过程
        2. 判断是否需要调用工具，每次只能返回一个工具，以在返回后面追加json格式，例如：```json { "function_name": "函数名称", "params": { "param1": "value1", "param2": "value2" }} ```
        3. 按照工具的参数要求并且结合提供的任务需求来提供正确的参数
        4. 观察工具的返回结果，并根据结果决定下一步操作
        5. 工具可能会发生变化，比如新增工具或现有工具消失
        
        # 可用工具:
        %s
    """

    next_step_prompt = """
        # 任务目标
        根据已经获取的信息，判断当前获取到的数据是否满足当前任务所需数据：
        
        # 任务要求
        1. 如果可以解决(满足用户给出的条件和范围)，根据已经提供的数据和上下文的调用过程对用户的需求进行总结和归纳并且生成报告数据，报告要求如下：
        1.1 如果数据是图片链接则根据需求内容进行总结，并且总结跟随在图片链接后面
        1.2 如果是多个图片链接则使用md的语法对图片进行排版，同一行放置两张图片，并且限制图片的大小
        1.3 如果是数字类型的数据则对数据进行归纳总结并且对数据通过md语法对其进行高亮
        1.4 总结的内容不要使用多级菜单，只需要直接返回总结内容即可
        1.5 **总结的内容不要出现使用过的函数名称以及参数等信息**

        2. 如果缺少数据或内容，请继续结合上下文中提供的工具调用合适的工具获取更多信息并且加上需要调用的工具，
        每次只能返回一个工具，以json格式，例如：```json { "function_name": "函数名称", "params": { "param1": "value1", "param2": "value2" }} ```
        3. 如果工具调用结果返回为空判断后续的函数工具是否要以当前函数工具为基础，如果需要以当前函数工具为基础就不需要再进行后续的函数调用并且直接返回当前任务
        获取的数据为空即可
        
        # 收集的数据：
        %s
        
        # 任务需求
        %s
    """

    function_counter: Dict[str, int]

    def __int__(self, llm_client: OpenAI, function_caller: FunctionCaller):
        super().__init__(llm_client, function_caller)

    async def invoke_task(self, message_content: MessageContext) -> List[Any]:
        tasks = message_content.tasks
        tools = await self.function_caller.get_tools_list()
        messages = [
            {
                "role": "system",
                "content": self.tool_call_prompt % json.dumps(tools, ensure_ascii=False),
            },
        ]
        results = []
        for task in tasks:
            await self.process_task(task, messages, results)

        return results

    def counter(self, function_name: str):
        """ 函数计数器 """
        if function_name not in self.function_counter:
            self.function_counter[function_name] = 0
        self.function_counter[function_name] += 1

    async def process_task(self, task: Task, messages: List, results: List):
        """ 循环遍历树形结构 """
        task_desc_list = task.task_desc_list
        if task_desc_list is not None and len(task_desc_list) > 0:
            for task_desc in task_desc_list:
                await self.do_invoke_task(task_desc, messages, results)
        child_task = task.children
        if child_task is not None and len(child_task) > 0:
            for child in child_task:
                await self.process_task(child, messages, results)

    async def do_invoke_task(self, task: str, sys_messages: List, results: List):
        """ 执行task任务 """
        # 构建模型的请求消息
        messages = [*sys_messages,
                    {
                        "role": "user",
                        "content": f"用户需求:\n{task}",
                    }]

        assistant_message = await self.invoke(messages)
        tool_call_result = []
        # 循环遍历当前模型是否需要进行工具的调用
        while True:
            # 将模型返回的数据添加到上下文中
            messages.append(assistant_message)
            content = assistant_message.content
            tool = None
            # 提取其中是否进行了工具调用
            if "```json" in content:
                tool = json.loads(intercept_str(assistant_message.content, "```json", "```"))

            # 如果不需要进行工具调用了，则直接将模型总结的数据进行返回
            if tool is None:
                results.append({
                    "key": task,
                    "content": assistant_message.content
                })
                break
            else:
                tool_name = tool['function_name']
                try:
                    # 调用工具，获取到返回的结果
                    tool_result = await self.function_caller.call_tool(tool_name, tool['params'])
                except Exception as e:
                    tool_result = str(e)
                tool_call_result.append({
                    "function_name": tool_name,
                    "result": str(tool_result)
                })

            # 根据下一步的提示词，让大模型判断当前收集的数据是否满足要求了
            messages.append({
                "role": "user",
                "content": self.next_step_prompt % (json.dumps(tool_call_result, ensure_ascii=False), task)
            })

            # 调用大模型后继续遍历进行工具的解析
            assistant_message = await self.invoke(messages)


class MCPClient:
    task_splicer: TaskSplicer

    def __init__(self, mcp_config=None, tools: List[SysFunctionDefinition] = None):
        """初始化MCP客户端"""
        if mcp_config is None:
            mcp_config = {}
        if not KEY:
            raise ValueError("请设置您的OpenAI API密钥")

        self.llm_client = OpenAI(
            api_key=KEY,
            base_url=BASE_URL,
        )
        self.mcp_client = Client(mcp_config)
        self.function_caller = FunctionCaller(self.mcp_client, sys_function=tools)

    async def generator_report(self, template: str, user_input: str):
        """ 根据指定的模板信息生成对应的报告信息 """
        task_splicer = TaskSplicer(self.llm_client, self.function_caller)
        message_context = await task_splicer.splice_template(template)
        invoker = TaskInvoker(self.llm_client, self.function_caller)
        results = await invoker.invoke_task(message_context)

        replace_template = message_context.template
        for result in results:
            key = result["key"]
            content = result["content"]
            replace_template = replace_template.replace(f"#{{{key}}}", content)

        return replace_template


def get_concurrent_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    config = {
        "mcpServers": {
            "mcp-server-chart": {
                "type": "sse",
                "url": "https://mcp.api-inference.modelscope.net/7060e0e92c8d49/sse"
            },
            "mcp_drone_order_server": {
                "transport": "stdio",
                "command": "python3",
                "args": ["./mcp_drone_order_server.py", "--verbose"],
            },
        }
    }

    get_time = SysFunctionDefinition("get_concurrent_time", """
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
    """, get_concurrent_time)

    mcp_client = MCPClient(mcp_config=config, tools=[get_time])
    # tools = await mcp_client.get_mcp_tools()
    # print(tools)
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
