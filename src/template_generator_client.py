import asyncio
import io
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Iterable, Callable, Self

import requests
from fastmcp import Client
from flask import Flask, request
from mcp.types import CallToolResult
from minio import Minio
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam, ChatCompletionMessage, ChatCompletionMessageToolCall, ChatCompletionMessageParam

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# MinIO 配置
MINIO_ENDPOINT = "minio-api.cd.zhinf.com"  # MinIO 服务地址（含端口）
MINIO_ACCESS_KEY = "g1FfoY95nvJLjGcx"  # 替换为你的 Access Key
MINIO_SECRET_KEY = "V3CER7j7uM9zaCsVOkm6cHNOzzVG9c9r"  # 替换为你的 Secret Key
MINIO_BUCKET_NAME = "zhxx-common"  # 存储桶名称
MINIO_SECURE = False  # 是否使用 HTTPS
# 初始化 MinIO 客户端
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)


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


def replace_string(target: str, prefix: str, suffix: str, replacement: str) -> str:
    """
    替换指定字符串开头和指定字符串结尾的子字符串
    :param target: 待替换的字符串
    :param prefix: 开头字符串
    :param suffix: 结尾字符串
    :param replacement: 替换的字符串
    :return: 替换后的字符串
    """
    start_index = target.find(prefix)
    if start_index == -1:
        return target

    end_index = target.find(suffix, start_index + len(prefix))
    if end_index == -1:
        return target

    return (target[:start_index] + replacement + target[end_index + len(suffix):]).strip()


def repair_json_str(target: str) -> str:
    """ 修复json中md字符串的格式 """
    return target.replace('\n', '\\n').replace('"', '\\"')


def upload_minio(template: str, file_name: str) -> str:
    template_bytes = io.BytesIO(bytes(template, encoding="utf-8"))
    file_name = f"10101/{MINIO_BUCKET_NAME}/{datetime.now().strftime('%Y%m%d')}/{file_name}"
    # 上传文件
    client.put_object(MINIO_BUCKET_NAME, file_name, template_bytes, length=template_bytes.getbuffer().nbytes, content_type="md")
    # 返回对象访问路径（如果是 public 策略，可以直接返回 URL）
    return f"http://{MINIO_ENDPOINT}/{MINIO_BUCKET_NAME}/{file_name}"


def remove_minio(file_name: str):
    file_name = f"10101/{MINIO_BUCKET_NAME}/{datetime.now().strftime('%Y%m%d')}/{file_name}"
    client.remove_object(MINIO_BUCKET_NAME, file_name)


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
        self.tool_call_result = []
        self.messages = []
        self.task_result = []

    def append_call_result(self, result: Dict):
        self.tool_call_result.append(result)

    def append_context(self, message: Any):
        self.messages.append(message)

    def create_messages(self, role: str, content: str) -> Any:
        message = {
            "role": role,
            "content": content
        }
        self.append_context(message)
        return message

    def append_task_result(self, result: Dict):
        self.task_result.append(result)


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

    async def invoke(self, messages: List[ChatCompletionMessageParam], enable_thinking: bool = False) -> ChatCompletionMessage:
        tools = await self.get_tools()
        response = self.llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            tools=tools or [],
            tool_choice="auto",
            top_p=0.7,
            temperature=0.7,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
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
        5. 任务描述需要原封不动的提取数据，禁止修改标题和任务描述信息，包括占位符中的换行符等特殊字符
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
        6. 如果提供的工具列表没有合适的工具信息，直接返回<Finish>
        
        # 可用工具:
        %s
    """

    task_prediction = """
        # 任务目标
        根据已经获取的信息，判断当前获取到的数据是否满足当前任务所需数据：
        
        # 任务要求
        1. 根据已经提供的数据判断是否满足用户需求，如果满足用户需求直接返回<Finish>，禁止输出其他数据
        2. 如果缺少数据或内容，请继续结合上下文中提供的工具继续选择合适的工具来获取更多信息，
        每次只能返回一个工具，以json格式，例如：```json { "function_name": "函数名称", "params": { "param1": "value1", "param2": "value2" }} ```
        3. 如果工具调用结果返回为空判断后续的函数工具是否要以当前函数工具为基础，如果需要以当前函数工具为基础就不需要再进行后续的函数调用并且直接返回当前任务
        获取的数据为空即可
        4. 不得使用相同的参数调用同一个工具函数
        
        # 收集的数据：
        %s
        
        # 任务需求
        %s
    """

    finish_prompt = """
        # 角色
        报告生成助手。你将根据收集到的数据和用户需求来生成对应的报告内容，请遵循以下要求：
        
        # 要求
        1. 严格按照用户需求生成报告数据，**生成的报告内容禁止出现推理的过程，禁止出现使用过的工具名称以及工具参数信息**
        2. 仔细浏览收集到的数据，并且在生成内容时充分使用到收集到的数据，不能出现遗漏的数据
        2. 报告内容不得生成多级标题
        3. 如果用户的需求对排版格式有要求，在生成了报告后需要检查报告样式是否满足md语法格式要求
        4. 如果存在图片链接时使用html语法插入图片并且限制图片长宽为320x320，如果数据提供中不存在图片链接则不生成图片相关的数据对应的模板置空即可
        5. 如果收集到的数据为空，则按照用户生成报告，数据内容保持为空
        
        # 收集的数据：
        %s
    """

    function_counter: Dict[str, int]

    def __int__(self, llm_client: OpenAI, function_caller: FunctionCaller):
        super().__init__(llm_client, function_caller)

    async def invoke_task(self, message_content: MessageContext) -> List[Any]:
        tasks = message_content.tasks
        tools = await self.function_caller.get_tools_list()
        results = []
        for task in tasks:
            await self.process_task(task, tools, results)

        return results

    def counter(self, function_name: str):
        """ 函数计数器 """
        if function_name not in self.function_counter:
            self.function_counter[function_name] = 0
        self.function_counter[function_name] += 1

    async def process_task(self, task: Task, tools: List[Dict], results: List):
        """ 循环遍历树形结构 """
        task_desc_list = task.task_desc_list
        if task_desc_list is not None and len(task_desc_list) > 0:
            task.create_messages(
                "system",
                self.tool_call_prompt % json.dumps(tools, ensure_ascii=False),
            )
            for task_desc in task_desc_list:
                await self.do_invoke_task(task, task_desc)
                results.extend(task.task_result)
        child_task = task.children
        if child_task is not None and len(child_task) > 0:
            for child in child_task:
                await self.process_task(child, tools, results)

    async def do_invoke_task(self, task: Task, task_desc: str):
        """ 执行task任务 """
        if task.tool_call_result is not None and len(task.tool_call_result) > 0:
            # 根据下一步的提示词，让大模型判断当前收集的数据是否满足要求了
            task.create_messages(
                "user",
                self.task_prediction % (json.dumps(task.tool_call_result, ensure_ascii=False), task_desc)
            )
        else:
            task.create_messages('user', f"用户需求:\n{task_desc}")

        assistant_message = await self.invoke(task.messages)
        # 循环遍历当前模型是否需要进行工具的调用
        while True:
            try:
                # 将模型返回的数据添加到上下文中
                task.append_context(assistant_message)
                content = assistant_message.content

                # 当前任务已经完成
                if "Finish" in content:
                    finish_messages = [
                        {
                            "role": "system",
                            "content": self.finish_prompt % json.dumps(task.tool_call_result, ensure_ascii=False)
                        },
                        {
                            "role": "user",
                            "content": f"# 用户需求:\n{task_desc}"
                        }
                    ]
                    assistant_message = await self.invoke(finish_messages)
                    task.append_context(assistant_message)
                    task.append_task_result({
                        "key": task_desc,
                        "content": assistant_message.content
                    })
                    break

                tool = None
                # 提取其中是否进行了工具调用
                if "```json" in content:
                    tool = json.loads(intercept_str(assistant_message.content, "```json", "```"))

                # 如果不需要进行工具调用了，则直接将模型总结的数据进行返回
                if tool is not None:
                    tool_name = tool['function_name']
                    params = tool['params']
                    try:
                        # 调用工具，获取到返回的结果
                        tool_result = await self.function_caller.call_tool(tool_name, params)
                    except Exception as e:
                        tool_result = str(e)
                    task.append_call_result({
                        "function_name": tool_name,
                        "params": params,
                        "result": tool_result
                    })

                # 根据下一步的提示词，让大模型判断当前收集的数据是否满足要求了
                task.create_messages(
                    "user",
                    self.task_prediction % (json.dumps(task.tool_call_result, ensure_ascii=False), task_desc)
                )

                # 调用大模型后继续遍历进行工具的解析
                assistant_message = await self.invoke(task.messages)
            except Exception as e:
                task.append_task_result({
                    "key": task_desc,
                    "content": "数据查询异常请重新生成报告"
                })
                break


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


with open("./mcp_server.json", "r") as f:
    config = json.loads(f.read())

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


class ReportGenerator:

    def __init__(self, identifier: str, template: str, callback_url: str):
        self.identifier = identifier
        self.template = template
        self.callback_url = callback_url

    def download_template(self):
        """下载模板信息"""
        try:
            if not self.template.startswith("http://"):
                return self.template
            response = requests.get(self.template)
            response.raise_for_status()  # 如果响应状态码不是200会抛出异常
            # 根据Content-Type判断编码，如果没有则默认使用utf-8
            content_type = response.headers.get('content-type', '')
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[-1]
            else:
                encoding = 'utf-8'
            # 将响应内容解码为字符串
            content = response.content.decode(encoding)
            return content
        except requests.RequestException as e:
            raise Exception(f"获取Markdown文件失败: {str(e)}")


def do_generator_report(request: ReportGenerator):
    error_message = ""
    url = ""
    try:
        template = request.download_template()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(mcp_client.generator_report(template, ""))
        url = upload_minio(result, f"报告{datetime.now().strftime('%Y%m%d%H%M')}.md")
        logging.info(f"报告生成成功，报告地址：{url}")
    except Exception as e:
        logging.error(f"报告生成失败：{str(e)}")
        error_message = "报告生成失败！！！"

    finally:
        response = requests.post(request.callback_url, json={"url": url, "identifier": request.identifier, "errorMsg": error_message})
        logging.info(f"回调结果：{response.text}")


@app.route("/generator/report", methods=["POST"])
def generator_report():
    body = request.json
    report = ReportGenerator(**body)
    thread = threading.Thread(target=do_generator_report, args=[report])
    thread.start()
    return {"message": "success", "code": 200}


async def main():
    with open("./xxx月工单周报.md", "r", encoding="utf-8") as f:
        template = f.read()
        md = await mcp_client.generator_report(template, "")
        with open("./报告.md", "w", encoding="utf-8") as writer:
            writer.write(md)


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=10000)
    # asyncio.run(main())
    with open("./demo.md", "r", encoding="utf-8") as f:
        template = f.read()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(mcp_client.generator_report(template, ""))
    with open("./报告.md", "w", encoding="utf-8") as writer:
        writer.write(result)
