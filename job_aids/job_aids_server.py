from datetime import datetime
from openai import OpenAI

openai_api_base = "http://192.168.2.236:9002/v1"
api_key = "123"
model = "Qwen2.5-7B-Instruct"
prompt = """
你作为专业会议助手，根据用户提供的工作记录或者会议纪要转为结构化记录。要求：
1. 按照会议的标题结构进行分段整理，详细理解每一个标题的具体内容
2. 提取每个标题内容的**核心观点**并且进行内容概述，如果有多条观点那么需要对观点进行总结整理提取核心关键
3. 注意报告点之间的逻辑关系并且在总结时需要注意点与点之间的逻辑关系
4. 保留技术术语但解释专业缩写，对比较少的内容使用专业的术语对其进行润色，一定要保证每一条概述内容都描述的足够清楚
5. 每一段总结完成后检查逻辑是否通顺，是否总结合理
6. 禁止输出与提供内容无关的数据

【示例】
# 2025年2月2日xxx会议纪要
## 业务线条
实现从零到一的突破

## 主线业务
xxxxxx

## 进行方式
- 找到上层总包公司
- 企业与企业之间的合作

## 贸易路线
xxx

【输出格式要求】
年月日xxx总结报告
1.今年的主要业务线条
2.进行的方式
3.xxx
4.xxx

"""


def main(path, name):
    # 获取当前日期并格式化为YYYYMMDD
    today = datetime.now().strftime("%Y%m%d")
    # 拼接文件名与日期后缀
    full_file_name = f"{path}/{name}_{today}.md"
    print(f"尝试读取文件：{full_file_name}")

    # 可以在这里添加读取文件的逻辑
    with open(full_file_name, 'r', encoding='utf-8') as file:
        content = file.read()
        model_chat(content)


def model_chat(content: str):
    client = OpenAI(
        api_key=api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"请帮我总结下面的文档，并生成结构化报告：\n{content}",
                    },
                ]
            }
        ],
    )
    print("Chat response:", chat_response.choices[0].message.content)


if __name__ == '__main__':
    path = "/Users/haijun/Documents/文档/无人设备智慧管理平台/"
    name = "无人设备管理平台思路方案"

    main(path, name)
