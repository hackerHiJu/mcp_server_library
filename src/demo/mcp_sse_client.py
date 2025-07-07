import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from mcp import ClientSession
from mcp.client.sse import sse_client


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_sse_server(self, server_url: str):
        """Connect to an MCP server running with SSE transport"""

        async with sse_client(url=server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("初始化SSE客户端...")
                response = await session.list_tools()
                tools = response.tools
                print("\n获取到的工具:", [tool.name for tool in tools])

    async def cleanup(self):
        """Properly clean up the session and streams"""


async def main():
    client = MCPClient()
    try:
        await client.connect_to_sse_server(server_url="http://127.0.0.1:9000/sse")
    finally:
        pass


if __name__ == "__main__":
    import sys

    asyncio.run(main())
