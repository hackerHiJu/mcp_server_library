from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import pymysql
import logging

logging.basicConfig(level=logging.INFO)

mcp = FastMCP(name="Drone Order MCP",
              instructions="""
                提供工单信息数据查询
              """
              )


@mcp.tool(
    name="query_order",
    description="查询指定时间范围指定类型的工单数据",
)
def query_order(start_time: Annotated[str, Field(description="开始时间,格式为yyyy-DD-mm HH:mm:ss")],
                end_time: Annotated[str, Field(description="结束时间,格式为yyyy-DD-mm HH:mm:ss")],
                type: Annotated[int, Field(description="查询工单的类型：1为无人机，2为水印工单")]):
    if (start_time is None or start_time == "") or end_time is None or type is None:
        return []

    """ 查询指定时间范围的工单数据 """
    conn = pymysql.connect(
        host='mysql.server',
        user='zhxx',
        password='zhxx@123456',
        database='zh_reservoir_new'
    )

    cursor = conn.cursor()

    # 查询指定时间段内的订单
    query = """
        select a.name, a.status, a.fact_start_time, a.fact_end_time,a.overdue from `order` a
             join order_extend b on a.id = b.order_id
        where a.create_time between %s and %s
        and b.extend -> '$.workType' = %s
    """
    cursor.execute(query, (start_time, end_time, type))
    logging.info(query % (start_time, end_time, type))

    # 获取查询结果
    query_results = cursor.fetchall()
    result = {
        "filed_description": {
            "name": "工单名称",
            "status": "工单状态(待派发-1，已派发-2，执行中-3，已完成-4，已评价-5，已关闭-6)",
            "fact_start_time": "工单开始时间",
            "fact_end_time": "工单结束时间",
            "overdue": "工单是否超时"
        },
        "data": []
    }
    for row in query_results:
        data = {
            "name": row[0],
            "status": row[1],
            "fact_start_time": row[2],
            "fact_end_time": row[3],
            "overdue": bool(row[4])
        }
        result['data'].append(data)

    cursor.close()
    conn.close()
    return result


if __name__ == '__main__':
    mcp.run()
