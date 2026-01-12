# LangChain SQL Agent 教程

本教程将指导您使用 LangChain 构建一个能够查询 SQL 数据库的智能代理（Agent）。该代理可以理解自然语言问题，自动生成 SQL 查询，并返回结果。

## 概述

SQL Agent 的工作流程：

1. 从数据库获取可用的表和模式
2. 根据问题决定哪些表是相关的
3. 获取相关表的模式信息
4. 基于问题和模式信息生成 SQL 查询
5. 使用 LLM 双重检查查询是否有常见错误
6. 执行查询并返回结果
7. 如果出错，修正错误直到查询成功
8. 基于结果形成最终回答

## ⚠️ 安全警告

构建 SQL 数据库的问答系统需要执行模型生成的 SQL 查询，这存在固有风险。请确保：
- 数据库连接权限尽可能窄，仅授予代理必需的权限
- 使用只读账户连接数据库
- 在生产环境中始终使用人机交互（Human-in-the-Loop）审查机制

## 核心概念

本教程涵盖以下核心概念：

1. **SQL 数据库工具** - 用于读取和查询 SQL 数据库的工具
2. **LangChain Agents** - 智能代理，能够使用工具解决复杂问题
3. **Human-in-the-Loop（人机交互）** - 在关键步骤暂停，等待人工审批
4. **Middleware（中间件）** - 在工具调用前后添加自定义逻辑
5. **记忆系统** - 短期和长期记忆管理（[详细文档](./记忆系统说明.md)）

## 快速开始

### 使用 uv 创建项目环境

```bash
# 1. 创建虚拟环境
uv venv

# 2. 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 3. 激活虚拟环境（Linux/macOS）
source .venv/bin/activate

# 4. 从 requirements.txt 安装依赖
uv pip install -r requirements.txt

# 5. 配置环境变量（创建 .env 文件）
# QWEN_API_KEY=your-dashscope-api-key
# QWEN_LLM_MODEL=qwen-plus  # 或 qwen-max, qwen-turbo

# 6. 运行示例
python sql_agent_example.py
```

## 前置要求

### 环境要求

- Python 3.8+
- uv 包管理器（[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)）
- DashScope (Qwen) API 密钥

### 使用 uv 安装依赖

```bash
# 从 requirements.txt 安装所有依赖
uv pip install -r requirements.txt
```

### 环境变量设置

在项目根目录创建 `.env` 文件：

```env
# DashScope (Qwen) API 密钥（必填）
QWEN_API_KEY=your-dashscope-api-key-here

# Qwen LLM 模型名称（可选，默认使用 qwen-plus）
# 可选值: qwen-turbo, qwen-plus, qwen-max
QWEN_LLM_MODEL=qwen-plus
```

**获取 DashScope API 密钥**：
- 访问 [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)
- 注册/登录阿里云账号并创建 API 密钥

## 1. 选择 LLM 模型

本教程使用 DashScope (Qwen) 作为 LLM 模型。Qwen 支持函数调用（Function Calling），这是构建 Agent 的关键能力。

```python
from langchain_community.chat_models.tongyi import ChatTongyi

# 初始化 Qwen LLM
model = ChatTongyi(
    model="qwen-plus",  # 或 qwen-turbo, qwen-max
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)
```

### Qwen 模型选择

- **qwen-turbo**: 快速响应，适合简单查询
- **qwen-plus**: 平衡性能和成本（推荐）
- **qwen-max**: 最强性能，适合复杂查询

## 2. 配置数据库

本教程使用 SQLite 数据库示例 - Chinook 数据库，它代表一个数字媒体商店。

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"数据库类型: {db.dialect}")
print(f"可用表: {db.get_usable_table_names()}")
```

### Chinook 数据库结构

Chinook 数据库包含以下表：
- `Album`: 专辑信息
- `Artist`: 艺术家信息
- `Customer`: 客户信息
- `Employee`: 员工信息
- `Genre`: 音乐类型
- `Invoice`: 发票信息
- `InvoiceLine`: 发票明细
- `MediaType`: 媒体类型
- `Playlist`: 播放列表
- `PlaylistTrack`: 播放列表曲目
- `Track`: 曲目信息

## 3. 添加数据库交互工具

`SQLDatabaseToolkit` 提供了与 SQL 数据库交互的工具集：

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()
```

### 可用工具

1. **sql_db_query**: 执行 SQL 查询并返回结果
2. **sql_db_schema**: 获取指定表的模式和示例行
3. **sql_db_list_tables**: 列出数据库中的所有表
4. **sql_db_query_checker**: 双重检查 SQL 查询的正确性

## 4. 创建 Agent

使用 `create_agent` 函数创建 SQL Agent：

```python
from langchain.agents import create_agent

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
""".format(dialect=db.dialect, top_k=5)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)
```

## 5. 运行 Agent

```python
question = "哪个音乐类型平均曲目长度最长？"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
```

## 6. 人机交互（Human-in-the-Loop）⭐

**这是本教程的重点内容！**

在生产环境中，让 Agent 直接执行 SQL 查询可能存在风险。通过人机交互机制，可以在执行查询前暂停并等待人工审批。

### 6.1 什么是 Human-in-the-Loop？

Human-in-the-Loop（HITL）是一种设计模式，允许在 AI 系统执行关键操作前插入人工审查步骤。

**优势**：
- 🛡️ **安全性**: 防止执行危险或错误的 SQL 查询
- 🔍 **可控性**: 人工可以修改或拒绝 Agent 的决策
- 📊 **透明性**: 清楚地看到 Agent 准备执行的操作
- 🎯 **精准性**: 及时纠正 Agent 的错误

### 6.2 使用中间件实现 HITL

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},  # 在执行查询前暂停
            description_prefix="工具执行等待审批",
        ),
    ],
    checkpointer=InMemorySaver(),  # 必需，用于保存状态
)
```

### 6.3 HITL 工作流程

1. **Agent 生成 SQL 查询**
2. **系统暂停并显示查询** - 等待人工审批
3. **人工审查查询**:
   - ✅ **批准（approve）**: 执行查询
   - ❌ **拒绝（reject）**: 不执行，让 Agent 重新生成
   - ✏️ **修改（edit）**: 修改查询后执行
4. **继续执行或重新规划**

**决策类型说明**：

| 决策类型 | 用户输入 | 说明 | Command 格式 |
|---------|---------|------|-------------|
| `"approve"` | `y` | 批准执行原始查询 | `{"type": "approve"}` |
| `"reject"` | `n` | 拒绝执行，Agent 重新规划 | `{"type": "reject"}` |
| `"edit"` | `e` | 修改查询后执行 | `{"type": "edit", "args": {"query": "..."}}` |

⚠️ **注意**: 使用 `"edit"` 而不是 ~~`"update"`~~！

### 6.4 处理中断

```python
config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "__interrupt__" in step:
        # 检测到中断，显示待审批的操作
        print("⚠️ 检测到中断 - 等待人工审批")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(f"工具: {request['tool']}")
            print(f"参数: {request['args']}")
```

### 6.5 恢复执行

使用 `Command` 恢复执行：

```python
from langgraph.types import Command

# 批准执行
agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config,
    stream_mode="values",
)

# 拒绝执行
agent.stream(
    Command(resume={"decisions": [{"type": "reject"}]}),
    config,
    stream_mode="values",
)

# 修改后执行
agent.stream(
    Command(resume={
        "decisions": [{
            "type": "edit",  # 使用 "edit" 而不是 "update"
            "args": {"query": "修改后的 SQL"}
        }]
    }),
    config,
    stream_mode="values",
)
```

## 7. 中间件（Middleware）深入讲解⭐

**中间件是拦截和修改工具调用的强大机制！**

### 7.1 什么是中间件？

中间件（Middleware）是在工具调用前后执行的代码层，可以：
- 🔍 **检查**: 查看工具调用的参数
- 🛑 **拦截**: 暂停执行等待审批
- ✏️ **修改**: 修改工具调用的参数或结果
- 📝 **记录**: 记录所有工具调用

### 7.2 中间件类型

#### HumanInTheLoopMiddleware

用于人机交互审批：

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "sql_db_query": True,      # 在查询执行前暂停
        "sql_db_schema": False,    # 获取模式时不暂停
    },
    description_prefix="等待审批",
)
```

**配置选项**：
- `interrupt_on`: 字典，指定哪些工具需要暂停
- `description_prefix`: 中断消息的前缀
- `interrupt_before`: 是否在工具调用前暂停（默认 True）
- `interrupt_after`: 是否在工具调用后暂停（默认 False）

#### 自定义中间件

可以创建自定义中间件：

```python
from langchain.agents.middleware import AgentMiddleware
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    """记录所有工具调用的中间件"""
    
    def __init__(self):
        super().__init__()
    
    def before_model(self, state: dict) -> dict[str, Any] | None:
        """在调用模型前执行"""
        print(f"📋 模型调用前")
        return None
    
    def after_model(self, state: dict) -> dict[str, Any] | None:
        """在模型返回后执行"""
        messages = state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    tool_name = tool_call.get("name", "未知工具")
                    print(f"✅ 工具调用: {tool_name}")
        return None
```

### 7.3 多个中间件

可以同时使用多个中间件，按顺序执行：

```python
agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        LoggingMiddleware(),           # 第一个：记录日志
        ValidationMiddleware(),        # 第二个：验证参数
        HumanInTheLoopMiddleware(...), # 第三个：人工审批
    ],
    checkpointer=InMemorySaver(),
)
```

### 7.4 中间件的执行顺序

```
用户输入
  ↓
Agent 决策
  ↓
中间件 1 - before_tool_call
  ↓
中间件 2 - before_tool_call
  ↓
中间件 3 - before_tool_call
  ↓
[可能的中断点]
  ↓
执行工具
  ↓
中间件 3 - after_tool_call
  ↓
中间件 2 - after_tool_call
  ↓
中间件 1 - after_tool_call
  ↓
返回结果
```

## 8. Checkpointer（检查点）与短期记忆

Checkpointer 用于保存 Agent 的执行状态，这是实现**短期记忆**的核心机制。

> 💡 **深入了解记忆系统**：查看 [记忆系统完整指南](./记忆系统说明.md) 了解短期和长期记忆的详细使用方法。

### 8.1 使用 InMemorySaver（开发/测试）

```python
from langgraph.checkpoint.memory import InMemorySaver

# 内存检查点（适合开发）
checkpointer = InMemorySaver()
```

**特点：**
- ✅ 无需配置，开箱即用
- ✅ 快速（纯内存操作）
- ❌ 程序结束后数据丢失

### 8.2 使用 SqliteSaver（持久化）✨

**推荐使用 `with` 语句管理连接：**

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 方式1: with 语句（推荐）
# 参考官方文档: https://docs.langchain.com/oss/python/langchain/short-term-memory
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    agent = create_agent(
        model,
        tools,
        checkpointer=checkpointer,
    )
    
    # 使用 agent...
    agent.invoke(
        {"messages": [{"role": "user", "content": "查询数据"}]},
        {"configurable": {"thread_id": "session-1"}},
    )
    # with 结束，连接自动关闭
```

**安装要求：**
```bash
uv pip install langgraph-checkpoint-sqlite
```

**特点：**
- ✅ 数据持久化到磁盘
- ✅ 程序重启后可恢复会话
- ✅ 适合单机应用、原型开发

### 8.3 Checkpointer 的作用（短期记忆）

Checkpointer 实现了 Agent 的**短期记忆**：

| 功能 | 说明 |
|------|------|
| **对话历史** | 记住当前会话的所有消息 |
| **中断状态** | 保存 HITL 中断点的状态 |
| **恢复执行** | 从中断点恢复执行 |
| **多轮对话** | 支持上下文感知的多轮对话 |
| **会话隔离** | 通过 `thread_id` 隔离不同会话 |

### 8.4 短期记忆 vs 长期记忆

| 维度 | 短期记忆（Checkpointer） | 长期记忆（Store） |
|------|----------------------|-----------------|
| **作用域** | 单个会话/线程 | 跨会话，全局 |
| **存储内容** | 对话历史、状态 | 用户偏好、知识 |
| **实现方式** | Checkpointer | Store |
| **访问方式** | 自动注入 state | 通过 store 显式访问 |

📚 **更多信息**：参考 [记忆系统完整指南](./记忆系统说明.md) 了解长期记忆的使用方法。

## 9. 完整的 HITL 示例

```python
# 配置 Agent
config = {"configurable": {"thread_id": "user-123"}}

# 第一步：发送问题
for step in agent.stream(
    {"messages": [{"role": "user", "content": "查询前5名客户"}]},
    config,
    stream_mode="values",
):
    if "__interrupt__" in step:
        interrupt = step["__interrupt__"][0]
        request = interrupt.value["action_requests"][0]
        
        # 显示 SQL 查询
        sql_query = request["args"]["query"]
        print(f"待审批 SQL:\n{sql_query}")
        
        # 人工审查
        decision = input("批准(y)/拒绝(n)/修改(e): ")
        
        if decision == "y":
            # 批准执行
            for s in agent.stream(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config,
                stream_mode="values",
            ):
                if "messages" in s:
                    print(s["messages"][-1].content)
        
        elif decision == "n":
            # 拒绝执行
            for s in agent.stream(
                Command(resume={"decisions": [{"type": "reject"}]}),
                config,
                stream_mode="values",
            ):
                if "messages" in s:
                    print(s["messages"][-1].content)
        
        elif decision == "e":
            # 修改后执行
            new_query = input("输入修改后的 SQL: ")
            for s in agent.stream(
                Command(resume={
                    "decisions": [{
                        "type": "edit",  # 使用 "edit" 而不是 "update"
                        "args": {"query": new_query}
                    }]
                }),
                config,
                stream_mode="values",
            ):
                if "messages" in s:
                    print(s["messages"][-1].content)
```

## 使用场景

完成本教程后，您可以：

1. **自然语言数据库查询**: 用户用自然语言提问，Agent 自动生成并执行 SQL
2. **数据分析助手**: 帮助非技术人员查询和分析数据
3. **商业智能（BI）**: 构建智能报表系统
4. **数据库管理**: 自动化数据库运维任务（需谨慎使用）

## 最佳实践

1. **安全性**:
   - ✅ 始终使用只读数据库账户
   - ✅ 在生产环境启用 Human-in-the-Loop
   - ✅ 限制查询结果数量
   - ❌ 永远不要让 Agent 执行 DML 语句（INSERT、UPDATE、DELETE）

2. **性能优化**:
   - 为常用表添加索引
   - 限制返回结果数量
   - 缓存常见查询结果

3. **用户体验**:
   - 提供清晰的错误消息
   - 显示生成的 SQL 查询（透明性）
   - 支持查询历史记录

4. **测试**:
   - 测试各种类型的查询
   - 测试错误处理
   - 测试 HITL 流程

## 下一步

- 查看 [记忆系统完整指南](./记忆系统说明.md) 学习短期和长期记忆管理
- 查看 [LangGraph 文档](https://langchain-ai.github.io/langgraph/) 了解更高级的 Agent 构建
- 查看 [SQL Agent 深度定制教程](https://docs.langchain.com/) 学习如何直接使用 LangGraph 原语
- 学习 LangSmith Observability（Tracing）：见 `rag/langsmith_observability/`

## 注意事项

1. **API 密钥**: 确保正确设置 DashScope (Qwen) API 密钥
2. **数据库文件**: 首次运行会自动下载 Chinook.db 数据库文件
3. **依赖安装**: 使用 `uv pip install -r requirements.txt` 安装所有依赖
4. **Function Calling**: Qwen 模型必须支持函数调用功能
5. **Checkpointer**: 使用 HITL 时必须配置 Checkpointer
6. **线程 ID**: 每个会话使用唯一的 thread_id 来隔离状态
