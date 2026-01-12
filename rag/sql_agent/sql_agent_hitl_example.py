"""
LangChain SQL Agent - Human-in-the-Loop（人机交互）完整示例
基于官方文档: https://docs.langchain.com/oss/python/langchain/sql-agent

本示例深入演示：
1. 如何配置 Human-in-the-Loop（HITL）中间件
2. 如何处理中断并进行人工审批
3. 如何批准、拒绝或修改 Agent 的决策
4. 如何使用多个中间件
5. Checkpointer 的作用和使用方法

⭐ 这是 SQL Agent 教程的核心内容！
"""

import os
import pathlib
import requests
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv: uv pip install python-dotenv")

# ============================================================================
# 准备工作：初始化模型和数据库
# ============================================================================

print("=" * 80)
print("准备工作：初始化环境")
print("=" * 80)

# 检查 API 密钥
qwen_api_key = os.environ.get("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError(
        "未找到 QWEN_API_KEY 环境变量！\n"
        "请在 .env 文件中设置: QWEN_API_KEY=your_api_key_here"
    )

qwen_model = os.environ.get("QWEN_LLM_MODEL", "qwen-plus")

# 初始化 LLM
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(
    model=qwen_model,
    dashscope_api_key=qwen_api_key,
    temperature=0,
)

print(f"✅ LLM 模型初始化成功: {qwen_model}")

# 下载并连接数据库
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if not local_path.exists():
    print(f"正在下载 Chinook 数据库...")
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"✅ 数据库下载完成")

from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"✅ 数据库连接成功: {db.dialect}")

# 创建工具
from langchain_community.agent_toolkits import SQLDatabaseToolkit
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

print(f"✅ 工具创建成功: {len(tools)} 个工具")

# 系统提示词
system_prompt = """
你是一个专门与 SQL 数据库交互的智能代理。
给定一个输入问题，创建一个语法正确的 {dialect} 查询来运行，
然后查看查询结果并返回答案。

重要规则：
1. 始终先查看数据库中有哪些表（使用 sql_db_list_tables）
2. 然后查询最相关表的模式（使用 sql_db_schema）
3. 在执行查询前必须双重检查（使用 sql_db_query_checker）
4. 限制结果数量不超过 {top_k} 条
5. 不要执行 DML 语句（INSERT、UPDATE、DELETE、DROP 等）
6. 按相关列排序结果，返回最有趣的示例
""".format(dialect=db.dialect, top_k=5)

print("\n" + "=" * 80)
print("准备工作完成！现在开始 Human-in-the-Loop 演示")
print("=" * 80 + "\n")

input("按 Enter 键开始...")

# ============================================================================
# 示例 1：基础 Human-in-the-Loop - 批准执行
# ============================================================================

print("\n" + "=" * 80)
print("示例 1：基础 Human-in-the-Loop - 批准执行")
print("=" * 80)

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 创建带 HITL 中间件的 Agent
agent_hitl = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            # 指定在哪些工具调用时暂停
            interrupt_on={
                "sql_db_query": True,  # 在执行 SQL 查询时暂停（最重要！）
                "sql_db_schema": False,  # 获取表结构时不暂停
                "sql_db_list_tables": False,  # 列出表时不暂停
                "sql_db_query_checker": False,  # 检查查询时不暂停
            },
            description_prefix="⚠️ 工具执行等待审批",
        ),
    ],
    checkpointer=InMemorySaver(),  # 必须配置 checkpointer 才能使用 HITL
)

print("""
配置说明：
- 中间件类型: HumanInTheLoopMiddleware
- 中断触发器: sql_db_query（执行 SQL 查询时）
- Checkpointer: InMemorySaver（内存存储）

工作流程：
1. Agent 生成 SQL 查询
2. 系统暂停，显示查询内容
3. 等待人工审批（批准/拒绝/修改）
4. 根据审批结果继续或重新规划
""")

input("\n按 Enter 键发送查询...")

# 配置包含 thread_id，用于标识会话
config = {"configurable": {"thread_id": "demo-1"}}

question = "查询购买金额最高的前 3 位客户"
print(f"\n📝 用户问题: {question}")
print("=" * 80)

# 第一步：发送问题，Agent 会在需要执行 SQL 时暂停
for step in agent_hitl.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "__interrupt__" in step:
        # 检测到中断！
        print("\n🛑 检测到中断 - Agent 准备执行 SQL 查询")
        print("=" * 80)
        
        interrupt = step["__interrupt__"][0]
        
        # 获取待审批的操作
        for request in interrupt.value["action_requests"]:
            tool_name = request.get("tool", "未知工具")
            tool_args = request.get("args", {})
            
            print(f"\n工具名称: {tool_name}")
            print(f"工具参数:")
            
            # 格式化显示 SQL 查询
            if "query" in tool_args:
                sql_query = tool_args["query"]
                print("\n" + "─" * 80)
                print("SQL 查询:")
                print("─" * 80)
                # 按行显示 SQL，使其更易读
                for line in sql_query.strip().split('\n'):
                    print(line)
                print("─" * 80)
            
            # 其他参数
            for key, value in tool_args.items():
                if key != "query":
                    print(f"  {key}: {value}")
        
        print("\n" + "=" * 80)
        print("人工审批选项:")
        print("  [y] 批准 - 执行此查询")
        print("  [n] 拒绝 - 不执行，让 Agent 重新生成")
        print("  [e] 修改 - 修改查询后执行")
        print("=" * 80)
        
        # 获取人工审批决策
        decision = input("\n请选择 (y/n/e): ").strip().lower()
        
        if decision == "y":
            # 批准执行
            print("\n✅ 批准执行查询")
            print("正在执行...")
            
            # 使用 Command 恢复执行
            for s in agent_hitl.stream(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config,
                stream_mode="values",
            ):
                if "messages" in s:
                    last_msg = s["messages"][-1]
                    if hasattr(last_msg, 'type'):
                        if last_msg.type == "tool":
                            print(f"\n🔧 工具返回:")
                            print(f"  {last_msg.content[:300]}")
                            if len(last_msg.content) > 300:
                                print("  ...")
                        elif last_msg.type == "ai" and last_msg.content:
                            print(f"\n🤖 Agent 最终回复:")
                            print(f"  {last_msg.content}")
        
        elif decision == "n":
            # 拒绝执行
            print("\n❌ 拒绝执行查询")
            print("Agent 将重新规划...")
            
            for s in agent_hitl.stream(
                Command(resume={"decisions": [{"type": "reject"}]}),
                config,
                stream_mode="values",
            ):
                if "messages" in s:
                    last_msg = s["messages"][-1]
                    if hasattr(last_msg, 'type') and last_msg.type == "ai" and last_msg.content:
                        print(f"\n🤖 Agent 响应:")
                        print(f"  {last_msg.content}")
        
        elif decision == "e":
            # 修改后执行
            print("\n✏️ 修改查询")
            print("请输入修改后的 SQL 查询（可以粘贴原查询并修改）:")
            print("提示: 输入多行查询后，在新行输入 'END' 结束")
            
            lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            
            new_query = "\n".join(lines)
            
            print(f"\n修改后的查询:")
            print("─" * 80)
            print(new_query)
            print("─" * 80)
            
            print("\n正在执行修改后的查询...")
            
            for s in agent_hitl.stream(
                Command(resume={
                    "decisions": [{
                        "type": "edit",
                        "args": {"query": new_query}
                    }]
                }),
                config,
                stream_mode="values",
            ):
                if "messages" in s:
                    last_msg = s["messages"][-1]
                    if hasattr(last_msg, 'type'):
                        if last_msg.type == "tool":
                            print(f"\n🔧 工具返回:")
                            print(f"  {last_msg.content[:300]}")
                        elif last_msg.type == "ai" and last_msg.content:
                            print(f"\n🤖 Agent 最终回复:")
                            print(f"  {last_msg.content}")
        
        else:
            print("\n❌ 无效选择，默认拒绝执行")
            for s in agent_hitl.stream(
                Command(resume={"decisions": [{"type": "reject"}]}),
                config,
                stream_mode="values",
            ):
                pass
    
    elif "messages" in step:
        # 普通消息，显示 Agent 的其他操作
        last_msg = step["messages"][-1]
        if hasattr(last_msg, 'type'):
            if last_msg.type == "ai" and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                print(f"\n🤖 Agent 正在调用工具:")
                for tc in last_msg.tool_calls:
                    tool_name = tc.get('name', '未知')
                    # 只显示非 sql_db_query 的工具调用（sql_db_query 会被中断）
                    if tool_name != "sql_db_query":
                        print(f"  - {tool_name}")

print("\n" + "=" * 80)
print("✅ 示例 1 完成")
print("=" * 80)

input("\n按 Enter 键继续下一个示例...")

# ============================================================================
# 示例 2：多个中间件 - 记录日志 + HITL
# ============================================================================

print("\n" + "=" * 80)
print("示例 2：多个中间件 - 记录日志 + Human-in-the-Loop")
print("=" * 80)

print("""
本示例演示如何使用多个中间件：
1. LoggingMiddleware: 记录所有工具调用
2. HumanInTheLoopMiddleware: 在关键工具调用时暂停

中间件执行顺序：
  工具调用前: Middleware 1 → Middleware 2 → ... → 工具执行
  工具调用后: ... → Middleware 2 → Middleware 1
""")

from langchain.agents.middleware import AgentMiddleware
from typing import Any

# 自定义日志中间件
class LoggingMiddleware(AgentMiddleware):
    """记录所有工具调用的中间件"""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
    
    def before_model(self, state: dict) -> dict[str, Any] | None:
        """在调用模型前执行"""
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            # 检查是否有工具调用
            if hasattr(last_msg, 'type') and last_msg.type == "human":
                self.call_count += 1
                print(f"\n📋 [{self.call_count}] 用户查询")
        return None
    
    def after_model(self, state: dict) -> dict[str, Any] | None:
        """在模型返回后执行"""
        # 记录模型响应中的工具调用
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            # 检查是否有工具调用
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    tool_name = tool_call.get("name", "未知工具")
                    print(f"🔧 工具调用: {tool_name}")
        return None

# 创建带多个中间件的 Agent
agent_multi = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        LoggingMiddleware(),  # 第一个中间件：记录日志
        HumanInTheLoopMiddleware(  # 第二个中间件：人工审批
            interrupt_on={"sql_db_query": True},
            description_prefix="⚠️ SQL 查询等待审批",
        ),
    ],
    checkpointer=InMemorySaver(),
)

input("\n按 Enter 键发送查询...")

config_2 = {"configurable": {"thread_id": "demo-2"}}
question_2 = "列出所有音乐类型的名称"

print(f"\n📝 用户问题: {question_2}")
print("=" * 80)

# 自动批准模式（为了演示日志功能）
print("\n⚙️ 自动批准模式：将自动批准所有 SQL 查询")
print("（您将看到 LoggingMiddleware 记录所有工具调用）\n")

for step in agent_multi.stream(
    {"messages": [{"role": "user", "content": question_2}]},
    config_2,
    stream_mode="values",
):
    if "__interrupt__" in step:
        print("\n🛑 检测到 SQL 查询中断")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            if "query" in request.get("args", {}):
                print(f"SQL: {request['args']['query'][:100]}...")
        
        print("⚙️ 自动批准执行")
        
        # 自动批准
        for s in agent_multi.stream(
            Command(resume={"decisions": [{"type": "approve"}]}),
            config_2,
            stream_mode="values",
        ):
            if "messages" in s:
                last_msg = s["messages"][-1]
                if hasattr(last_msg, 'type') and last_msg.type == "ai" and last_msg.content:
                    print(f"\n🤖 Agent 回复: {last_msg.content}")

print("\n" + "=" * 80)
print("✅ 示例 2 完成")
print("观察：LoggingMiddleware 记录了所有工具调用")
print("=" * 80)

input("\n按 Enter 键继续下一个示例...")

# ============================================================================
# 示例 3：使用 SQLite Checkpointer（持久化）
# ============================================================================

print("\n" + "=" * 80)
print("示例 3：使用持久化 Checkpointer")
print("=" * 80)

print("""
本示例演示 Checkpointer 的概念和使用方式。

Checkpointer 的作用：
- 保存 Agent 的执行状态
- 支持中断后恢复执行
- 记录对话历史

Checkpointer 类型：
- InMemorySaver: 内存存储，程序结束后清空（本示例使用）
- SqliteSaver: SQLite 持久化存储（需要额外配置）
- PostgresSaver: PostgreSQL 持久化存储（生产环境推荐）

💡 持久化存储的优势：
- 重启程序后可以恢复之前的会话
- 支持长时间运行的对话
- 可以查看和分析历史记录
""")

# SQLite Checkpointer 配置
# 需要安装: uv pip install langgraph-checkpoint-sqlite
# 
# 参考官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory
print("正在初始化 SqliteSaver...")

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    use_sqlite = True
    print("✅ SqliteSaver 可用，将使用 with 语句管理连接")
    print(f"📂 数据库文件: checkpoints.db")
    print(f"💡 会话状态将持久化保存，程序重启后可恢复\n")
except ImportError as e:
    use_sqlite = False
    print(f"⚠️ SqliteSaver 不可用: {e}")
    print("   回退到 InMemorySaver")
    print("   安装命令: uv pip install langgraph-checkpoint-sqlite\n")

# 定义运行 Agent 的函数（避免代码重复）
def run_example_3(agent, checkpointer_name):
    """运行示例3的主要逻辑"""
    input("按 Enter 键发送查询...")
    
    config_3 = {"configurable": {"thread_id": "persistent-session-1"}}
    question_3 = "有多少位艺术家？"
    
    print(f"\n📝 用户问题: {question_3}")
    print(f"📂 会话 ID: {config_3['configurable']['thread_id']}")
    print(f"📂 Checkpointer: {checkpointer_name}")
    print("=" * 80)
    
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question_3}]},
        config_3,
        stream_mode="values",
    ):
        if "__interrupt__" in step:
            print("\n🛑 检测到中断")
            interrupt = step["__interrupt__"][0]
            for request in interrupt.value["action_requests"]:
                if "query" in request.get("args", {}):
                    sql = request['args']['query']
                    print(f"\nSQL 查询:\n{sql}")
            
            approve = input("\n批准执行? (y/n): ").strip().lower()
            
            if approve == "y":
                print("\n✅ 批准执行")
                for s in agent.stream(
                    Command(resume={"decisions": [{"type": "approve"}]}),
                    config_3,
                    stream_mode="values",
                ):
                    if "messages" in s:
                        last_msg = s["messages"][-1]
                        if hasattr(last_msg, 'type') and last_msg.type == "ai" and last_msg.content:
                            print(f"\n🤖 Agent 回复: {last_msg.content}")
            else:
                print("\n❌ 拒绝执行")
                for s in agent.stream(
                    Command(resume={"decisions": [{"type": "reject"}]}),
                    config_3,
                    stream_mode="values",
                ):
                    pass

# 根据 SqliteSaver 是否可用选择不同的实现
if use_sqlite:
    # 使用 with 语句（推荐方式）
    # 参考: https://docs.langchain.com/oss/python/langchain/short-term-memory
    with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        agent_persist = create_agent(
            model,
            tools,
            system_prompt=system_prompt,
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={"sql_db_query": True},
                    description_prefix="⚠️ SQL 查询等待审批",
                ),
            ],
            checkpointer=checkpointer,
        )
        
        run_example_3(agent_persist, "SqliteSaver (with 语句管理)")
        
        print("\n💾 会话状态已保存到 checkpoints.db")
        print("   下次运行时，使用相同的 thread_id 可以恢复此会话")
        # with 块结束，连接自动关闭
else:
    # 回退到 InMemorySaver
    checkpointer = InMemorySaver()
    agent_persist = create_agent(
        model,
        tools,
        system_prompt=system_prompt,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"sql_db_query": True},
                description_prefix="⚠️ SQL 查询等待审批",
            ),
        ],
        checkpointer=checkpointer,
    )
    
    run_example_3(agent_persist, "InMemorySaver (内存存储)")
    
    print("\n💡 使用内存存储，会话状态在程序结束后会丢失")

print("\n" + "=" * 80)
print("✅ 示例 3 完成")
print("=" * 80)

input("\n按 Enter 键继续下一个示例...")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("Human-in-the-Loop 教程总结")
print("=" * 80)
print("""
您已经完成了 Human-in-the-Loop 的完整示例！

本示例涵盖了：
1. ✅ 基础 HITL 配置和使用
   - HumanInTheLoopMiddleware 配置
   - interrupt_on 参数设置
   - 处理中断和审批

2. ✅ 三种审批方式
   - 批准（approve）: 直接执行
   - 拒绝（reject）: 让 Agent 重新生成
   - 修改（update）: 修改后执行

3. ✅ 多个中间件组合
   - 自定义 LoggingMiddleware
   - 中间件执行顺序
   - before_tool_call 和 after_tool_call

4. ✅ Checkpointer 的使用
   - InMemorySaver: 内存存储（适合开发）
   - SqliteSaver: 持久化存储（适合生产）
   - thread_id 会话隔离

核心概念：
- 🛡️ HITL 提供了安全控制层，防止危险操作
- 🔍 中间件是灵活的拦截机制，可以自定义行为
- 💾 Checkpointer 保存状态，支持暂停和恢复
- 🎯 可以针对不同工具设置不同的中断策略

最佳实践：
1. 在生产环境中，始终对 sql_db_query 启用 HITL
2. 使用持久化 Checkpointer（SqliteSaver）
3. 为每个用户会话使用唯一的 thread_id
4. 记录所有审批决策，用于审计
5. 可以添加自定义中间件进行日志记录、监控等

下一步：
- 将 HITL 集成到您的应用中
- 根据业务需求自定义中间件
- 实现更复杂的审批工作流
- 添加用户界面，使审批过程更友好

文档资源：
- SQL Agent 完整文档: README.md
- LangGraph HITL 指南: https://langchain-ai.github.io/langgraph/
- Middleware 文档: https://docs.langchain.com/
""")

print("\n🎉 教程完成！感谢学习！")
