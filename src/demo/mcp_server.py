import json
from typing import Any
from urllib.parse import unquote

from mcp.server.fastmcp import FastMCP, Image, Context
from mcp.server.fastmcp.prompts import base
from PIL import Image as PILImage

mcp = FastMCP("mcp_server")


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    获取天气的api
    :param city: 城市名称（需要使用英文，如Beijing）
    :return: 天气数据字典；若出错返回包含 error信息的字典
    """
    return {
        "城市": f"{city}\n",
        "温度": "25.5°C\n",
        "湿度": "25%\n",
        "风俗": "12 m/s\n",
        "天气": "晴\n",
    }


@mcp.tool()
async def get_weather(city: str) -> dict[str, Any] | None:
    """
    获取天气
    :param city: 城市名称（需要使用英文，如Beijing）
    :return: 天气数据字典；若出错返回包含 error信息的字典
    """
    return await fetch_weather(city)


@mcp.tool()
async def test(city: str, ctx: Context) -> str:
    """
    获取天气
    :param city: 城市名称（需要使用英文，如Beijing）
    :return: 天气描述
    """
    get_weather_city = await ctx.read_resource(f"echo://{city}/25")
    result: str = ""
    for content in get_weather_city:
        result += unquote(content.content)
    return result


@mcp.tool()
def create_thumbnail(image_url: str) -> Image:
    """Create a thumbnail from an image"""
    img = PILImage.open(image_url)
    img.thumbnail((100, 100))
    return Image(data=img.tobytes(), format="jpg")


@mcp.resource(uri="echo://hello")
def resource() -> str:
    """Echo a message as a resource"""
    return f"Resource echo: hello"


@mcp.resource(uri="echo://{message}/{age}")
def message(message: str, age: int) -> str:
    """Echo a message as a resource"""
    return f"你好，{message}，{age}"


@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


@mcp.prompt()
def debug_error(error: str) -> list[base.Message]:
    return [
        base.UserMessage("I'm seeing this error:"),
        base.UserMessage(error),
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),
    ]


if __name__ == '__main__':
    # 使用标准 I/O 方式运行MCP服务器
    mcp.run(transport='stdio')
