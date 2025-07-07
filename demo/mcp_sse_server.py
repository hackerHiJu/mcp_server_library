# 服务端
import asyncio

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

mcp = FastMCP('weather')


@mcp.tool()
async def get_weather(city: str) -> str:
    """获取天气信息"""
    return "天气信息"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """创建一个starlette应用来支持sse协议"""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == '__main__':
    mcp_server = mcp._mcp_server

    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, port=9000)
