# 服务端
from mcp.server import FastMCP
from mcp.server.fastmcp import Context
from mcp.types import SamplingMessage, TextContent

mcp = FastMCP('file_server')


@mcp.tool()
async def delete_file(file_path: str, ctx: Context):
    # 创建 SamplingMessage 用于触发 sampling callback 函数
    result = await ctx.session.create_message(
        messages=[
            SamplingMessage(
                role='user',
                content=TextContent(type='text', text=f'是否要删除文件: {file_path} (Y)')
            )
        ],
        max_tokens=100
    )

    # 获取到 sampling callback 函数的返回值，并根据返回值进行处理
    if result.content.text == 'Y':
        return f'文件 {file_path} 已被删除！！'


if __name__ == '__main__':
    mcp.run(transport='stdio')
