from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Any

import mcp.server.stdio
import mcp.types as types
from fastmcp import Context
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage server startup and shutdown lifecycle."""
    # Initialize resources on startup
    try:
        yield {"message": "hello"}
    finally:
        print("最终执行")


class ExampleServer(Server):
    tools: types.Tool = []

    def __init__(self, name: str, lifespan: Any):
        super().__init__(name, lifespan)

    def tool(
            self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if callable(name):
            raise TypeError(
                "The @tool decorator was used incorrectly. "
                "Did you forget to call it? Use @tool() instead of @tool"
            )

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(
                types.Tool(
                    name=name or fn.__name__,
                    description=description or fn.__doc__ or "",
                    inputSchema={
                        "properties": {
                            "city": {
                                "title": "City",
                                "type": "string"
                            }
                        },
                        "required": ["city"],
                        "title": "example-tool",
                        "type": "object"
                    }
                )
            )
            return fn

        return decorator

    def get_tools(self):
        return self.tools


# 指定服务的生命周期函数
server = ExampleServer("example-server", lifespan=server_lifespan)


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="example-prompt",
            description="An example prompt template",
            arguments=[
                types.PromptArgument(
                    name="arg1", description="Example argument", required=True
                )
            ],
        )
    ]


@server.get_prompt()
async def handle_get_prompt(
        name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    if name != "example-prompt":
        raise ValueError(f"Unknown prompt: {name}")

    return types.GetPromptResult(
        description="Example prompt",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text="Example prompt text"),
            )
        ],
    )


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return server.get_tools()


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            name="example-resource",
            description="An example resource",
            content=types.TextContent(type="text", text="Example resource text"),
        )
    ]


@server.tool()
async def test(city: str, ctx: Context) -> str:
    """测试方法"""
    return ctx.lifespan_context["db"]


async def run():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="example",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
