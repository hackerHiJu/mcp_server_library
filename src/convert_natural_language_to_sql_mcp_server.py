import json

import pymysql
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


def get_database_table():
    conn = pymysql.connect(
        host='mysql.server',
        user='zhxx',
        password='zhxx@123456',
        database='zh_reservoir_new'
    )
    # 查询数据库所有的表名和描述信息
    query_sql = "SELECT table_name, table_comment FROM information_schema.tables WHERE table_schema = %s"
    with conn.cursor() as cursor:
        cursor.execute(query_sql, (conn.db.decode() if hasattr(conn.db, 'decode') else conn.db,))
        tables = cursor.fetchall()

    # 将结果转换为字典格式
    result = {table[0]: table[1] for table in tables}
    return result


def get_table_ddl(table_names):
    conn = pymysql.connect(
        host='mysql.server',
        user='zhxx',
        password='zhxx@123456',
        database='zh_reservoir_new'
    )
    result = {}
    with conn.cursor() as cursor:
        for table_name in table_names:
            cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
            create_table_sql = cursor.fetchone()[1]
            result[table_name] = create_table_sql
    return result


BASE_URL = "http://192.168.2.236:10000/v1"
MODEL_NAME = "Qwen3-32B-AWQ"
KEY = "18e567651c50428c8009f2191aa9773f.A5QNRyfcFveC1RVZ"

llm_client = OpenAI(
    api_key=KEY,
    base_url=BASE_URL,
)

system_prompt = """
# 角色
你是一个专业的数据库语句生成助手，你将根据用户的需要和数据库表结构来生成对应的sql语句。

# 要求
1. 仔细理解用户的需求以及表结构的关系，根据需求生成sql
2. 生成的sql必须要符合mysql8.x的版本
3. 生成sql语句后需要检查其是否符合语法要求，如果出现语法错误及时进行修改
4. 注意表结构中的json类型的字段为扩展字段，会单独提供json格式的说明，你需要将有json类型字段的表与主表结合进行生成
5. 注意生成的sql语句必须要在表结构中进行定义，禁止生成对应表结构中不存在的字段
"""

extract_tables = """
# 任务目标
根据上下文提供的数据库表与用户的需求描述来提取需要用到的数据库表名，以json格式的数组返回，例如：```json ["order", "plan"] ```

# 要求
1. 提取表名时判断提取的表是否存在扩展表或者附表，如果存在扩展表时，**需要将主表和扩展表名称一起进行提取**

# 数据库表
%s

# 用户需求
%s
"""

next_prompt = """
# 任务目标
根据用户提供的需求与数据库表结构来生成对应的sql。

# 要求
1. 根据用户提供的数据库表结构来判断是否满足当前用户的需求
2. 如果不满足用户的需求请根据上下文提供的数据表名称来继续选择对应的数据表获取数据结构，并且返回的格式为数组json，例如： ```json ["order"] ```
2.1 如果主表数据结构字段不满足要求，优先提取上下文提供的数据表中的扩展表结构，判断扩展表是否有满足需求的字段，如果扩展表中的字段也不满足再考虑提取其他数据表的结构信息
2.2 如果给出的表都不满足用户的需求，那么就直接提示用户没有数据表满足要求
4. 如果已经满足要求则根据对应的数据表结构来生成对应的mysql语句，并且返回sql格式，例如：```sql select * from `order` ```

# 已经提供的表名
%s

# 数据表结构
%s
"""

messages = [
    {
        "role": "system",
        "content": system_prompt,
    },
    {
        "role": "user",
        "content": extract_tables % (json.dumps(get_database_table(), indent=4, ensure_ascii=False), "帮我查询项目合同名称含有广安前锋环卫项目合同的项目面积信息"),
    }
]

response = llm_client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    extra_body={
        "tok_k": 20,
        "min_p": 0,
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

tables = []
ddl = {}
while True:
    assistant_message = response.choices[0].message
    messages.append(assistant_message)
    content = assistant_message.content

    if "```sql" in content:
        sql = intercept_str(content, "```sql", "```")
        print(sql)
        break

    if "```json" in content:
        tables.extend(json.loads(intercept_str(content, "```json", "```")))

    if tables is None or len(tables) == 0:
        break

    ddl.update(get_table_ddl(tables))

    messages.append({
        "role": "user",
        "content": next_prompt % (str(tables), json.dumps(ddl, indent=4, ensure_ascii=False))
    })
    response = llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        extra_body={
            "tok_k": 20,
            "min_p": 0,
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )
