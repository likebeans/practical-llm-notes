# Skills 模式：SQL 助理系统

> 基于 LangChain 官方文档：[Build a SQL assistant with skills](https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant)

## 目录

- [什么是 Skills 模式](#什么是-skills-模式)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [工作流程](#工作流程)
- [实现方式](#实现方式)
- [使用场景](#使用场景)
- [快速开始](#快速开始)

## 什么是 Skills 模式

**Skills（技能）模式**是一种多代理架构，其中多个代理共享一组通用的技能（工具）。每个代理可以专注于不同的任务，但都能使用相同的底层能力。

### 关键特点

- 🔧 **技能共享**：多个代理使用相同的工具集
- 🎯 **专注任务**：每个代理有不同的系统提示和目标
- 🔄 **能力复用**：避免重复实现相同的功能
- 📊 **统一接口**：所有代理通过标准接口访问数据

## 核心概念

### Shared Skills（共享技能）

共享技能是所有代理都可以使用的通用工具：
- SQL 查询执行
- 数据库模式查询
- 查询验证
- 结果格式化

### Specialized Agents（专门化代理）

每个代理有不同的提示和目标：
- **分析代理**：回答数据分析问题
- **报告代理**：生成格式化报告
- **监控代理**：检测异常和趋势
- **优化代理**：提供查询优化建议

## 架构设计

### SQL 助理系统架构

```
┌─────────────────────────────────────────┐
│          Shared SQL Skills              │
│  ┌─────────────────────────────────┐    │
│  │ • sql_db_query                  │    │
│  │ • sql_db_schema                 │    │
│  │ • sql_db_list_tables            │    │
│  │ • sql_db_query_checker          │    │
│  └─────────────────────────────────┘    │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┬──────────┐
    │             │             │          │
    ↓             ↓             ↓          ↓
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│Analysis │  │ Report  │  │ Monitor │  │Optimize │
│ Agent   │  │ Agent   │  │ Agent   │  │ Agent   │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
 (数据分析)    (生成报告)    (监控异常)    (优化查询)
```

### 与其他模式的区别

| 特性 | Skills 模式 | Subagents 模式 | Router 模式 |
|------|------------|---------------|------------|
| 工具共享 | ✅ 完全共享 | ❌ 各自独立 | ❌ 各自独立 |
| 代理职责 | 不同任务 | 不同领域 | 路由选择 |
| 数据源 | 同一数据库 | 多个数据源 | 多个知识库 |
| 协作方式 | 并行独立 | 层级协调 | 智能路由 |

## 工作流程

### 示例：多角度数据分析

```
共享技能层:
  ├─ sql_db_query: 执行 SQL 查询
  ├─ sql_db_schema: 获取表结构
  ├─ sql_db_list_tables: 列出所有表
  └─ sql_db_query_checker: 验证 SQL 正确性

用户请求: "分析上个月的销售情况"

┌────────────────────────┐
│   Analysis Agent       │ → 使用共享技能
│  "分析销售趋势"          │    • 查询销售数据
└──────────┬─────────────┘    • 计算统计指标
           │                  • 生成分析结果
           ↓
      分析报告: "销售额增长 15%..."

┌────────────────────────┐
│   Report Agent         │ → 使用共享技能  
│  "生成销售报表"          │    • 查询详细数据
└──────────┬─────────────┘    • 格式化结果
           │                  • 生成 Markdown 表格
           ↓
      格式化报表: "| 产品 | 销量 |..."

┌────────────────────────┐
│   Monitor Agent        │ → 使用共享技能
│  "检测异常情况"          │    • 查询历史数据
└──────────┬─────────────┘    • 比较基线
           │                  • 识别异常
           ↓
      监控报告: "产品 X 销量异常下降..."
```

## 实现方式

### 1. 定义共享 SQL 技能

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# 连接数据库
db = SQLDatabase.from_uri("sqlite:///sales.db")

# 创建共享的 SQL 工具集
toolkit = SQLDatabaseToolkit(db=db, llm=model)
shared_sql_skills = toolkit.get_tools()

# 工具包括:
# - sql_db_query: 执行 SQL 查询
# - sql_db_schema: 获取表结构
# - sql_db_list_tables: 列出表
# - sql_db_query_checker: 验证 SQL
```

### 2. 创建专门化的代理

```python
from langchain.agents import create_agent

# 分析代理：专注数据分析
analysis_agent = create_agent(
    model,
    tools=shared_sql_skills,  # 使用共享技能
    system_prompt="""
    你是一个数据分析专家。
    使用 SQL 查询分析数据，提供洞察和趋势。
    
    分析步骤:
    1. 使用 sql_db_list_tables 查看可用表
    2. 使用 sql_db_schema 了解表结构
    3. 编写并执行 SQL 查询分析数据
    4. 解释分析结果，提供洞察
    
    重点: 深入分析数据背后的含义，而不仅仅是数字。
    """
)

# 报告代理：专注生成报表
report_agent = create_agent(
    model,
    tools=shared_sql_skills,  # 使用相同的共享技能
    system_prompt="""
    你是一个报表生成专家。
    查询数据并生成格式化的报表。
    
    报表要求:
    1. 使用 Markdown 表格格式
    2. 包含标题和摘要
    3. 数据清晰易读
    4. 添加必要的说明
    
    重点: 让报表专业、清晰、易于理解。
    """
)

# 监控代理：专注异常检测
monitor_agent = create_agent(
    model,
    tools=shared_sql_skills,  # 使用相同的共享技能
    system_prompt="""
    你是一个数据监控专家。
    监控数据异常，发现潜在问题。
    
    监控内容:
    1. 与历史数据对比
    2. 识别异常值
    3. 检测突然变化
    4. 评估数据质量
    
    重点: 快速识别问题，提供预警。
    """
)
```

### 3. 使用专门化的代理

```python
# 使用不同代理完成不同任务

# 任务 1: 数据分析
print("【数据分析】")
for step in analysis_agent.stream(
    {"messages": [{"role": "user", "content": "分析上个月的销售趋势"}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            if message.type == 'ai':
                print(message.content)

# 任务 2: 生成报表
print("\n【生成报表】")
for step in report_agent.stream(
    {"messages": [{"role": "user", "content": "生成上个月销售额前10的产品报表"}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            if message.type == 'ai':
                print(message.content)

# 任务 3: 监控异常
print("\n【异常监控】")
for step in monitor_agent.stream(
    {"messages": [{"role": "user", "content": "检测最近一周是否有销量异常的产品"}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            if message.type == 'ai':
                print(message.content)
```

### 4. 创建协调器（可选）

如果需要同时使用多个代理：

```python
@tool
def analyze_data(query: str) -> str:
    """使用分析代理分析数据"""
    result = analysis_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

@tool
def generate_report(query: str) -> str:
    """使用报告代理生成报表"""
    result = report_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

@tool
def monitor_anomalies(query: str) -> str:
    """使用监控代理检测异常"""
    result = monitor_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# 创建协调代理
coordinator = create_agent(
    model,
    tools=[analyze_data, generate_report, monitor_anomalies],
    system_prompt="""
    你是一个 SQL 助理协调器。
    根据用户需求，调用合适的专家代理：
    - analyze_data: 深入分析数据
    - generate_report: 生成格式化报表
    - monitor_anomalies: 检测异常情况
    
    可以同时调用多个代理来提供全面的答案。
    """
)
```

## 高级功能

### 1. 添加自定义技能

除了基础 SQL 技能，还可以添加自定义技能：

```python
from langchain.tools import tool

@tool
def export_to_csv(query: str, filename: str) -> str:
    """执行查询并导出结果到 CSV 文件"""
    result = db.run(query)
    # 导出逻辑...
    return f"数据已导出到 {filename}"

@tool
def visualize_data(query: str, chart_type: str) -> str:
    """执行查询并生成可视化图表"""
    result = db.run(query)
    # 生成图表逻辑...
    return f"已生成 {chart_type} 图表"

# 扩展技能集
extended_skills = shared_sql_skills + [export_to_csv, visualize_data]

# 使用扩展技能创建代理
advanced_agent = create_agent(
    model,
    tools=extended_skills,
    system_prompt="你可以查询数据、导出 CSV 和生成图表..."
)
```

### 2. 添加人机交互（HITL）

为敏感操作添加审批：

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# 为所有使用共享技能的代理添加 HITL
analysis_agent = create_agent(
    model,
    tools=shared_sql_skills,
    system_prompt=ANALYSIS_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="SQL 查询等待审批",
        )
    ],
    checkpointer=InMemorySaver(),
)
```

### 3. 技能组合

不同代理可以使用技能的不同子集：

```python
# 只读代理：只能查询，不能修改
readonly_skills = [
    tool for tool in shared_sql_skills 
    if 'query' in tool.name or 'schema' in tool.name
]

readonly_agent = create_agent(
    model,
    tools=readonly_skills,
    system_prompt="你只能查询数据，不能修改..."
)

# 管理员代理：拥有所有技能
admin_skills = shared_sql_skills + custom_admin_tools

admin_agent = create_agent(
    model,
    tools=admin_skills,
    system_prompt="你拥有完整的数据库访问权限..."
)
```

## 使用场景

### 适合 Skills 模式的场景

✅ **推荐使用**：
- **数据分析平台**：多个分析视角共享数据访问
- **BI 系统**：不同类型的报表共用查询能力
- **监控系统**：多个监控维度使用相同的数据源
- **团队协作**：不同角色使用相同的底层工具

### 典型应用

1. **商业智能（BI）**
   - 销售分析代理
   - 财务报表代理
   - 库存监控代理
   - KPI 追踪代理

2. **数据科学平台**
   - 探索性分析代理
   - 统计建模代理
   - 数据清洗代理
   - 可视化代理

3. **运维监控**
   - 性能监控代理
   - 日志分析代理
   - 异常检测代理
   - 报警生成代理

4. **客户分析**
   - 行为分析代理
   - 细分分群代理
   - 流失预警代理
   - 价值评估代理

### 不适合的场景

❌ **不推荐**：
- **不同数据源**：各代理访问完全不同的数据 → 使用 Router 模式
- **工具互斥**：代理需要完全不同的工具 → 使用 Subagents 模式
- **顺序依赖**：任务必须按特定顺序执行 → 使用 Handoffs 模式

## 最佳实践

### 1. 技能设计

- 保持技能通用和可复用
- 提供清晰的工具描述
- 合理的权限控制
- 添加必要的验证和安全检查

### 2. 代理专门化

- 每个代理有明确的职责
- 提示词要体现代理的专业性
- 避免职责重叠和冲突
- 测试每个代理的独立功能

### 3. 安全性

- 使用只读账户（查询代理）
- 敏感操作添加 HITL
- 限制查询复杂度和结果数量
- 记录所有数据库操作

### 4. 性能优化

- 缓存常用查询结果
- 为常用表添加索引
- 限制并发查询数量
- 使用异步处理长查询

### 5. 监控和维护

- 追踪每个代理的使用情况
- 监控查询性能
- 定期审查和优化提示词
- 收集用户反馈改进代理

## 快速开始

### 前置要求

```bash
# 安装依赖
uv pip install langchain langchain-community
uv pip install python-dotenv
```

### 环境配置

创建 `.env` 文件：

```env
# Qwen API 密钥
QWEN_API_KEY=your_api_key_here

# LangSmith 追踪（可选）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
```

### 运行示例

```bash
# 进入目录
cd mutil-agents/skills-sql-assistant

# 运行基础示例
uv run sql_skills_example.py

# 运行协调器示例
uv run sql_skills_coordinator.py
```

## 关键要点

1. **技能共享**：多个代理使用相同的工具集
2. **专注提示**：每个代理有不同的系统提示和目标
3. **能力复用**：避免重复实现相同功能
4. **灵活组合**：可以为不同代理分配不同的技能子集
5. **统一管理**：技能的更新会影响所有使用它的代理

## 下一步

完成本教程后，你可以：

1. ✅ 尝试运行 `sql_skills_example.py`
2. ✅ 创建自定义技能工具
3. ✅ 设计不同职责的专门化代理
4. ✅ 实现代理协调器
5. ✅ 添加人机交互审批
6. ✅ 使用 LangSmith 追踪代理行为
7. ✅ 探索其他多代理模式：
   - [Subagents 模式](../subagents-personal-assistant/) - 用于领域专家协调
   - [Handoffs 模式](../handoffs-customer-support/) - 用于状态机工作流
   - [Router 模式](../router-knowledge-base/) - 用于知识库路由

## 更多资源

- [LangChain 官方文档](https://docs.langchain.com/)
- [Multi-Agent 概述](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [SQL Agent 教程](https://docs.langchain.com/oss/python/langchain/sql-agent)
- [SQL Database Toolkit](https://docs.langchain.com/oss/python/integrations/toolkits/sql_database)

## 总结

Skills 模式提供了一种优雅的方式来构建共享底层能力的多代理系统：

- **技能共享**：所有代理使用相同的工具集
- **专注任务**：每个代理通过提示词专注不同任务
- **能力复用**：避免重复开发相同功能
- **灵活扩展**：轻松添加新代理或新技能

这是构建数据分析平台、BI 系统和监控系统的理想模式！
