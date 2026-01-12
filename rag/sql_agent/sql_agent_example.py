"""
LangChain SQL Agent 完整示例
基于官方文档: https://docs.langchain.com/oss/python/langchain/sql-agent

本示例演示如何：
1. 配置 SQL 数据库连接
2. 创建数据库交互工具
3. 构建能够查询数据库的 Agent
4. 实现基本的查询流程
5. 处理错误和边界情况

注意：本示例不包含 Human-in-the-Loop，完整的 HITL 示例请参考 sql_agent_hitl_example.py
"""

import os
import pathlib
import requests
from pathlib import Path
from typing import List

# 加载环境变量（从 .env 文件）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv 可以自动从 .env 文件加载环境变量")
    print("安装命令: uv pip install python-dotenv")

# ============================================================================
# 辅助函数：等待用户确认
# ============================================================================

def wait_for_user():
    """等待用户按 Enter 键继续"""
    print("\n" + "🔵" * 40)
    input("按 Enter 键继续下一部分...")
    print("🔵" * 40 + "\n")

# ============================================================================
# 第一部分：选择 LLM 模型
# ============================================================================

print("=" * 80)
print("第一部分：初始化 LLM 模型")
print("=" * 80)

# 从环境变量读取 DashScope (Qwen) API 密钥
qwen_api_key = os.environ.get("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError(
        "未找到 QWEN_API_KEY 环境变量！\n"
        "请在 .env 文件中设置: QWEN_API_KEY=your_api_key_here\n"
        "DashScope API 密钥获取: https://dashscope.console.aliyun.com/apiKey"
    )

# 从环境变量读取模型名称，默认使用 qwen-plus
qwen_model = os.environ.get("QWEN_LLM_MODEL", "qwen-plus")

print(f"正在初始化 Qwen 模型: {qwen_model}")

try:
    from langchain_community.chat_models.tongyi import ChatTongyi
    
    # 初始化 Qwen LLM
    # 注意：必须使用支持 Function Calling（函数调用）的模型
    model = ChatTongyi(
        model=qwen_model,
        dashscope_api_key=qwen_api_key,
        temperature=0,  # 设置为 0 使输出更确定
    )
    
    print(f"✅ LLM 模型初始化成功: {qwen_model}")
    
except ImportError:
    raise ImportError(
        "请安装 langchain-community 包: uv pip install langchain-community\n"
    )

# 测试模型是否正常工作
print("\n测试模型调用...")
try:
    response = model.invoke("你好，请简单介绍一下你自己")
    print(f"模型响应: {response.content[:100]}...")
    print("✅ 模型测试成功\n")
except Exception as e:
    print(f"❌ 模型测试失败: {e}")
    print("请检查 API 密钥是否正确")
    exit(1)

wait_for_user()

# ============================================================================
# 第二部分：配置数据库
# ============================================================================

print("=" * 80)
print("第二部分：配置 SQL 数据库")
print("=" * 80)

# 下载 Chinook 示例数据库
# Chinook 是一个代表数字媒体商店的示例数据库
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"✅ {local_path} 已存在，跳过下载")
else:
    print(f"正在下载 Chinook 数据库...")
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"✅ 数据库下载完成: {local_path}")
    else:
        print(f"❌ 下载失败，状态码: {response.status_code}")
        exit(1)

# 使用 LangChain 的 SQLDatabase 包装器连接数据库
from langchain_community.utilities import SQLDatabase

# 创建数据库连接
# SQLite 使用文件路径连接，格式为: sqlite:///path/to/database.db
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"\n数据库信息:")
print(f"  数据库类型: {db.dialect}")
print(f"  可用表数量: {len(db.get_usable_table_names())}")
print(f"  可用表列表: {db.get_usable_table_names()}")

# 测试数据库查询
print("\n测试数据库查询...")
sample_query = "SELECT * FROM Artist LIMIT 5;"
sample_result = db.run(sample_query)
print(f"  测试查询: {sample_query}")
print(f"  查询结果: {sample_result}")

print("\n✅ 数据库配置成功")

wait_for_user()

# ============================================================================
# 第三部分：添加数据库交互工具
# ============================================================================

print("=" * 80)
print("第三部分：创建数据库交互工具")
print("=" * 80)

from langchain_community.agent_toolkits import SQLDatabaseToolkit

# 创建 SQL 数据库工具包
# 工具包会自动创建一组用于与数据库交互的工具
toolkit = SQLDatabaseToolkit(db=db, llm=model)

# 获取所有可用工具
tools = toolkit.get_tools()

print(f"\n可用工具数量: {len(tools)}\n")

# 打印每个工具的信息
for i, tool in enumerate(tools, 1):
    print(f"{i}. {tool.name}")
    print(f"   描述: {tool.description}")
    print()

print("✅ 工具创建成功")

# 工具说明：
# 1. sql_db_query: 执行 SQL 查询并返回结果
#    - 这是最核心的工具，用于实际执行查询
#    - 如果查询有错误，会返回错误信息
# 
# 2. sql_db_schema: 获取指定表的模式信息
#    - 返回表的结构（列名、数据类型）和示例数据
#    - 帮助 Agent 了解表的内容
# 
# 3. sql_db_list_tables: 列出所有可用的表
#    - Agent 通常会先调用这个工具了解数据库结构
# 
# 4. sql_db_query_checker: 检查 SQL 查询的正确性
#    - 使用 LLM 检查查询是否有语法错误或常见问题
#    - 在执行查询前进行双重检查

wait_for_user()

# ============================================================================
# 第四部分：创建 Agent
# ============================================================================

print("=" * 80)
print("第四部分：创建 SQL Agent")
print("=" * 80)

from langchain.agents import create_agent

# 定义 Agent 的系统提示词
# 这个提示词定义了 Agent 的行为规则和工作流程
system_prompt = """
你是一个专门与 SQL 数据库交互的智能代理。
给定一个输入问题，创建一个语法正确的 {dialect} 查询来运行，
然后查看查询结果并返回答案。

除非用户指定了具体的数量，否则始终将查询结果限制在最多 {top_k} 条。

你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。

你必须在执行查询之前双重检查你的查询。如果在执行查询时出错，
请重写查询并重试。

不要对数据库进行任何 DML 语句（INSERT、UPDATE、DELETE、DROP 等）。

开始时，你应该始终查看数据库中有哪些表，以了解可以查询什么。
不要跳过这一步。

然后，你应该查询最相关表的模式。
""".format(
    dialect=db.dialect,
    top_k=5,
)

print("正在创建 Agent...")

# 创建 Agent
# create_agent 是一个高级函数，自动处理了很多底层细节
agent = create_agent(
    model,           # LLM 模型
    tools,           # 可用工具列表
    system_prompt=system_prompt,  # 系统提示词
)

print("✅ Agent 创建成功")
print("\nAgent 配置:")
print(f"  - LLM 模型: {qwen_model}")
print(f"  - 可用工具: {len(tools)} 个")
print(f"  - 数据库类型: {db.dialect}")
print(f"  - 最大返回结果: 5 条")

wait_for_user()

# ============================================================================
# 第五部分：运行 Agent - 示例 1
# ============================================================================

print("=" * 80)
print("第五部分：运行 Agent - 示例查询")
print("=" * 80)

# 示例问题 1：简单的聚合查询
question_1 = "哪个音乐类型平均曲目长度最长？"

print(f"\n问题 1: {question_1}")
print("=" * 80)

# 使用 stream 方法运行 Agent
# stream_mode="values" 会流式返回每一步的消息
for step in agent.stream(
    {"messages": [{"role": "user", "content": question_1}]},
    stream_mode="values",
):
    # 检查步骤中是否包含消息
    if "messages" in step:
        # 获取最后一条消息并打印
        last_message = step["messages"][-1]
        
        # 美化输出
        if hasattr(last_message, 'type'):
            if last_message.type == "human":
                print(f"\n👤 用户: {last_message.content}")
            elif last_message.type == "ai":
                # AI 消息可能包含工具调用或文本回复
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"\n🤖 Agent 决策:")
                    for tool_call in last_message.tool_calls:
                        print(f"  - 调用工具: {tool_call['name']}")
                        # 格式化参数显示
                        if 'args' in tool_call:
                            for key, value in tool_call['args'].items():
                                # 如果是 SQL 查询，格式化显示
                                if key == 'query' and isinstance(value, str):
                                    print(f"    {key}:")
                                    for line in value.strip().split('\n'):
                                        print(f"      {line}")
                                else:
                                    print(f"    {key}: {value}")
                elif last_message.content:
                    print(f"\n🤖 Agent 回复:\n{last_message.content}")
            elif last_message.type == "tool":
                print(f"\n🔧 工具返回 ({last_message.name}):")
                print(f"  {last_message.content[:200]}")
                if len(last_message.content) > 200:
                    print("  ...")

print("\n" + "=" * 80)
print("✅ 查询 1 完成")

wait_for_user()

# ============================================================================
# 第五部分：运行 Agent - 示例 2
# ============================================================================

print("=" * 80)
print("示例查询 2")
print("=" * 80)

# 示例问题 2：多表关联查询
question_2 = "列出购买金额最高的前 5 位客户的姓名和总购买金额"

print(f"\n问题 2: {question_2}")
print("=" * 80)

for step in agent.stream(
    {"messages": [{"role": "user", "content": question_2}]},
    stream_mode="values",
):
    if "messages" in step:
        last_message = step["messages"][-1]
        
        if hasattr(last_message, 'type'):
            if last_message.type == "human":
                print(f"\n👤 用户: {last_message.content}")
            elif last_message.type == "ai":
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"\n🤖 Agent 决策:")
                    for tool_call in last_message.tool_calls:
                        print(f"  - 调用工具: {tool_call['name']}")
                        if 'args' in tool_call:
                            for key, value in tool_call['args'].items():
                                if key == 'query' and isinstance(value, str):
                                    print(f"    {key}:")
                                    for line in value.strip().split('\n'):
                                        print(f"      {line}")
                                else:
                                    print(f"    {key}: {value}")
                elif last_message.content:
                    print(f"\n🤖 Agent 回复:\n{last_message.content}")
            elif last_message.type == "tool":
                print(f"\n🔧 工具返回 ({last_message.name}):")
                print(f"  {last_message.content[:200]}")
                if len(last_message.content) > 200:
                    print("  ...")

print("\n" + "=" * 80)
print("✅ 查询 2 完成")

wait_for_user()

# ============================================================================
# 第五部分：运行 Agent - 示例 3
# ============================================================================

print("=" * 80)
print("示例查询 3 - 复杂查询")
print("=" * 80)

# 示例问题 3：更复杂的业务查询
question_3 = "哪个国家的客户数量最多？列出前 3 个国家及其客户数量"

print(f"\n问题 3: {question_3}")
print("=" * 80)

for step in agent.stream(
    {"messages": [{"role": "user", "content": question_3}]},
    stream_mode="values",
):
    if "messages" in step:
        last_message = step["messages"][-1]
        
        if hasattr(last_message, 'type'):
            if last_message.type == "human":
                print(f"\n👤 用户: {last_message.content}")
            elif last_message.type == "ai":
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    print(f"\n🤖 Agent 决策:")
                    for tool_call in last_message.tool_calls:
                        print(f"  - 调用工具: {tool_call['name']}")
                        if 'args' in tool_call:
                            for key, value in tool_call['args'].items():
                                if key == 'query' and isinstance(value, str):
                                    print(f"    {key}:")
                                    for line in value.strip().split('\n'):
                                        print(f"      {line}")
                                else:
                                    print(f"    {key}: {value}")
                elif last_message.content:
                    print(f"\n🤖 Agent 回复:\n{last_message.content}")
            elif last_message.type == "tool":
                print(f"\n🔧 工具返回 ({last_message.name}):")
                print(f"  {last_message.content[:200]}")
                if len(last_message.content) > 200:
                    print("  ...")

print("\n" + "=" * 80)
print("✅ 查询 3 完成")

wait_for_user()

# ============================================================================
# 总结
# ============================================================================

print("=" * 80)
print("教程总结")
print("=" * 80)
print("""
您已经完成了 LangChain SQL Agent 的基础示例！

本示例涵盖了：
1. ✅ LLM 模型初始化：使用 DashScope (Qwen) 模型
2. ✅ 数据库配置：连接 SQLite Chinook 数据库
3. ✅ 工具创建：使用 SQLDatabaseToolkit 创建数据库交互工具
4. ✅ Agent 创建：使用 create_agent 创建 SQL Agent
5. ✅ 查询执行：运行多个示例查询，观察 Agent 的工作流程

Agent 的工作流程：
1. 接收用户问题
2. 调用 sql_db_list_tables 查看可用表
3. 调用 sql_db_schema 获取相关表的结构
4. 生成 SQL 查询
5. 调用 sql_db_query_checker 检查查询
6. 调用 sql_db_query 执行查询
7. 根据结果生成自然语言回答

下一步建议：
- 🔒 学习 Human-in-the-Loop（人机交互）：运行 sql_agent_hitl_example.py
- 🎯 学习更多中间件用法：查看 README.md 中的"中间件深入讲解"章节
- 🔧 尝试修改系统提示词，观察 Agent 行为的变化
- 📊 尝试连接您自己的数据库

⚠️ 重要提醒：
- 在生产环境中，始终使用 Human-in-the-Loop 机制
- 使用只读数据库账户
- 不要让 Agent 执行 DML 语句（INSERT、UPDATE、DELETE）
- 限制查询结果数量，避免返回过多数据

更多资源：
- Human-in-the-Loop 完整示例: sql_agent_hitl_example.py
- SQL Agent 文档: https://docs.langchain.com/oss/python/langchain/sql-agent
- LangGraph 文档: https://langchain-ai.github.io/langgraph/
""")
