# 多代理系统（Multi-Agent Systems）

> 基于 LangChain 官方文档的多代理系统学习指南

## 📚 概述

多代理系统是构建复杂 AI 应用的强大架构模式。本项目包含 4 种核心多代理模式的完整实现和详细文档。

## 🎯 四种核心模式

### 1. [Subagents 模式](./subagents-personal-assistant/) - 个人助理系统

**监督者协调专家**

```
Supervisor Agent
    ├─ Calendar Agent (日历专家)
    ├─ Email Agent (邮件专家)
    └─ ... (其他专家)
```

**特点**：
- 🎯 中央监督者协调多个专门化子代理
- 🔧 每个子代理有自己的工具和提示
- 📊 适合多领域任务协调

**适用场景**：
- 个人助理系统
- 企业工作流自动化
- 多领域任务协调

**示例**：
```bash
cd subagents-personal-assistant
uv run personal_assistant_example.py
```

---

### 2. [Handoffs 模式](./handoffs-customer-support/) - 客户支持系统

**状态机工作流**

```
保修验证 → 问题分类 → 解决方案
   (状态1)    (状态2)     (状态3)
```

**特点**：
- 🔄 单一代理，通过状态改变行为
- 📋 每个状态有专门的提示和工具
- 🎯 适合顺序信息收集

**适用场景**：
- 客户支持流程
- 表单填写向导
- 诊断和排查系统

**示例**：
```bash
cd handoffs-customer-support
uv run customer_support_example.py
uv run customer_support_interactive.py  # 交互式
```

---

### 3. [Router 模式](./router-knowledge-base/) - 多源知识库路由

**智能路由检索**

```
User Query
    ↓
Router Agent
    ├─ Knowledge Base 1
    ├─ Knowledge Base 2
    └─ Knowledge Base 3
```

**特点**：
- 🎯 根据查询内容智能路由
- 📚 统一访问多个独立知识库
- 🔀 每个知识库可独立优化

**适用场景**：
- 企业知识管理
- 技术文档系统
- 多源数据检索

**示例**：
```bash
cd router-knowledge-base
uv run router_example.py
```

---

### 4. [Skills 模式](./skills-sql-assistant/) - SQL 助理系统

**技能共享**

```
Shared SQL Skills
    ├─ Analysis Agent (分析)
    ├─ Report Agent (报表)
    └─ Monitor Agent (监控)
```

**特点**：
- 🔧 多个代理共享相同工具集
- 🎯 每个代理专注不同任务
- 📊 避免重复实现相同功能

**适用场景**：
- 数据分析平台
- BI 报表系统
- 数据监控系统

**示例**：
```bash
cd skills-sql-assistant
uv run sql_skills_example.py
```

## 📊 模式对比

| 特性 | Subagents | Handoffs | Router | Skills |
|------|-----------|----------|--------|--------|
| **架构** | 多个代理 | 单代理多状态 | 单代理路由 | 多代理共享工具 |
| **协调方式** | 层级协调 | 状态转换 | 智能路由 | 独立并行 |
| **工具管理** | 各自独立 | 状态相关 | 各自独立 | 完全共享 |
| **适用场景** | 多领域任务 | 顺序流程 | 多源检索 | 共享能力 |
| **复杂度** | 中-高 | 中 | 低-中 | 低-中 |
| **扩展性** | ✅ 高 | ⚠️ 中等 | ✅ 高 | ✅ 高 |

## 🚀 快速开始

### 前置要求

```bash
# 安装 uv 包管理器
pip install uv

# 或通过 pipx 安装
pipx install uv
```

### 安装依赖

```bash
# 进入项目目录
cd mutil-agents

# 安装依赖
uv pip install langchain langchain-community
uv pip install langgraph langgraph-checkpoint
uv pip install dashscope
uv pip install python-dotenv
```

### 环境配置

在项目根目录创建 `.env` 文件：

```env
# Qwen API 密钥（必需）
QWEN_API_KEY=your_api_key_here

# Qwen 模型选择（可选）
QWEN_LLM_MODEL=qwen-plus  # 或 qwen-turbo, qwen-max

# LangSmith 追踪（可选）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
```

**获取 API 密钥**：
- [DashScope (Qwen)](https://dashscope.console.aliyun.com/apiKey)
- [LangSmith](https://smith.langchain.com/)

### 运行示例

```bash
# 1. Subagents 模式
cd subagents-personal-assistant
uv run personal_assistant_example.py

# 2. Handoffs 模式
cd handoffs-customer-support
uv run customer_support_example.py

# 3. Router 模式
cd router-knowledge-base
uv run router_example.py

# 4. Skills 模式
cd skills-sql-assistant
uv run sql_skills_example.py
```

## 📖 学习路径

### 初学者路径

1. **开始**: [Skills 模式](./skills-sql-assistant/) - 最简单，理解技能共享
2. **进阶**: [Router 模式](./router-knowledge-base/) - 学习智能路由
3. **深入**: [Handoffs 模式](./handoffs-customer-support/) - 掌握状态机
4. **精通**: [Subagents 模式](./subagents-personal-assistant/) - 理解层级协调

### 实战路径

根据你的应用场景选择：

- **构建客服系统** → Handoffs 模式
- **多领域助手** → Subagents 模式
- **文档问答** → Router 模式
- **数据分析** → Skills 模式

## 🎓 核心概念

### 什么是多代理系统？

多代理系统是由多个 AI 代理协作完成复杂任务的架构。每个代理可以：
- 专注于特定领域或任务
- 使用专门的工具和知识
- 与其他代理协作或独立工作

### 为什么使用多代理系统？

**优势**：
- ✅ **关注点分离**：每个代理职责清晰
- ✅ **独立优化**：可以单独改进每个代理
- ✅ **易于扩展**：添加新能力不影响现有功能
- ✅ **更好的性能**：专门化代理表现更好

**挑战**：
- ⚠️ **复杂度增加**：需要协调多个组件
- ⚠️ **调试困难**：问题可能出现在多个层级
- ⚠️ **成本增加**：可能需要更多 LLM 调用

### 何时使用多代理？

**推荐使用**：
- 任务涉及多个不同领域
- 需要不同的专业知识
- 工具集较大且可以逻辑分组
- 需要独立优化不同部分

**不推荐使用**：
- 简单的单一任务
- 所有操作在同一领域
- 工具数量少（< 5 个）

## 🛠️ 高级功能

### 人机交互（HITL）

所有模式都支持人机交互审批：

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    tools=tools,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sensitive_tool": True},
            description_prefix="等待审批",
        )
    ],
    checkpointer=InMemorySaver(),
)
```

### 记忆管理

使用 Checkpointer 实现状态持久化：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    agent = create_agent(
        model,
        tools=tools,
        checkpointer=checkpointer,
    )
```

### LangSmith 追踪

启用 LangSmith 追踪所有代理交互：

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="your_key"
```

## 📂 项目结构

```
mutil-agents/
├── README.md                          # 本文件
├── .env.example                       # 环境变量示例
│
├── subagents-personal-assistant/      # Subagents 模式
│   ├── README.md                      # 详细文档
│   ├── personal_assistant_example.py  # 基础示例
│   └── personal_assistant_hitl_example.py  # HITL 示例
│
├── handoffs-customer-support/         # Handoffs 模式
│   ├── README.md                      # 详细文档
│   ├── customer_support_example.py    # 基础示例
│   └── customer_support_interactive.py  # 交互式示例
│
├── router-knowledge-base/             # Router 模式
│   ├── README.md                      # 详细文档
│   └── router_example.py              # 路由示例
│
└── skills-sql-assistant/              # Skills 模式
    ├── README.md                      # 详细文档
    └── sql_skills_example.py          # SQL 技能示例
```

## 🔧 常见问题

### Q: 如何选择合适的模式？

**决策树**：
1. 任务是否需要按顺序收集信息？→ **Handoffs 模式**
2. 是否需要从多个知识库检索？→ **Router 模式**
3. 多个代理是否使用相同工具？→ **Skills 模式**
4. 是否需要协调多个领域专家？→ **Subagents 模式**

### Q: 可以组合多种模式吗？

可以！实际应用中经常组合使用：
- Subagents 中的子代理可以使用 Handoffs 模式
- Router 可以路由到使用 Skills 模式的代理
- Handoffs 的某个状态可以调用 Subagents

### Q: 性能如何优化？

**优化建议**：
1. **缓存**：缓存常见查询和嵌入
2. **并行**：使用异步 API 并行调用
3. **批处理**：批量处理嵌入和查询
4. **模型选择**：根据任务选择合适大小的模型
5. **提示优化**：简化提示减少 token 使用

### Q: 如何调试多代理系统？

**调试工具**：
1. **LangSmith**：追踪完整的执行流程
2. **日志**：添加详细的日志记录
3. **分层测试**：先测试单个代理，再测试集成
4. **可视化**：绘制代理交互图

## 📚 更多资源

### 官方文档
- [LangChain 官方文档](https://docs.langchain.com/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [Multi-Agent 概述](https://docs.langchain.com/oss/python/langchain/multi-agent)

### 相关教程
- [RAG Agent 教程](https://docs.langchain.com/oss/python/langchain/rag)
- [SQL Agent 教程](https://docs.langchain.com/oss/python/langchain/sql-agent)
- [LangSmith 可观测性](https://docs.langchain.com/oss/python/langchain/langsmith-observability)

### 社区资源
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Academy](https://academy.langchain.com/)
- [LangChain Discord](https://discord.gg/langchain)

## 🤝 贡献

欢迎贡献！如果你发现问题或有改进建议：
1. Fork 本项目
2. 创建特性分支
3. 提交 Pull Request

## 📄 许可证

本项目遵循 MIT 许可证。

## 🎉 总结

多代理系统是构建复杂 AI 应用的强大工具。通过本项目，你将掌握：

- ✅ 4 种核心多代理模式
- ✅ 每种模式的适用场景
- ✅ 完整的代码实现
- ✅ 最佳实践和优化技巧

选择合适的模式，构建强大的 AI 应用！🚀

---

**快速链接**：
- [Subagents 模式](./subagents-personal-assistant/)
- [Handoffs 模式](./handoffs-customer-support/)
- [Router 模式](./router-knowledge-base/)
- [Skills 模式](./skills-sql-assistant/)
