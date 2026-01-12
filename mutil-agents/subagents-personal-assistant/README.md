# Subagents 模式：个人助理系统

> 基于 LangChain 官方文档：[Build a personal assistant with subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)

## 目录

- [什么是 Subagents 模式](#什么是-subagents-模式)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [为什么使用 Supervisor](#为什么使用-supervisor)
- [工作流程](#工作流程)
- [实现方式](#实现方式)
- [使用场景](#使用场景)
- [快速开始](#快速开始)

## 什么是 Subagents 模式

**Subagents（子代理）模式**是一种多代理架构，其中一个**中央 supervisor（监督者）代理**协调多个**专门化的 worker（工作者）代理**。这种方法在任务需要不同类型的专业知识时表现出色。

与其构建一个管理跨域工具选择的代理，不如创建由监督者协调的专注专家，监督者理解整体工作流程。

## 核心概念

### Supervisor Agent（监督者代理）

- **职责**：理解用户意图，将任务路由到合适的子代理
- **工具**：每个子代理都被包装为一个高级工具
- **决策**：在领域级别做出路由决策，而非单个 API 级别

### Worker Agents（工作者代理）

- **专门化**：每个子代理专注于一个特定领域
- **工具集**：拥有该领域的所有相关工具
- **独立性**：可以独立完成领域内的复杂任务

## 架构设计

### 三层架构

```
┌─────────────────────────────────────────┐
│      Supervisor Agent (监督者层)         │
│  - 理解用户意图                          │
│  - 协调子代理                            │
│  - 综合结果                              │
└─────────────────────────────────────────┘
              │         │
      ┌───────┴─────┬───┴──────┐
      │             │          │
┌─────▼─────┐ ┌────▼─────┐ ┌──▼──────┐
│ Calendar  │ │  Email   │ │  Other  │
│  Agent    │ │  Agent   │ │  Agent  │
└───────────┘ └──────────┘ └─────────┘
   (中间层：子代理 - 接受自然语言，返回自然语言)
      │             │          │
┌─────▼─────┐ ┌────▼─────┐ ┌──▼──────┐
│ Calendar  │ │  Email   │ │  Other  │
│   APIs    │ │   APIs   │ │   APIs  │
└───────────┘ └──────────┘ └─────────┘
   (底层：刚性 API - 需要精确格式)
```

### 层级职责

**顶层（Supervisor）**：
- 接收用户请求
- 路由到高级能力
- 综合多个子代理的结果

**中间层（Sub-agents）**：
- 接受自然语言输入
- 转换为结构化 API 调用
- 返回自然语言确认

**底层（APIs）**：
- 执行实际操作
- 要求精确格式
- 返回结构化数据

## 为什么使用 Supervisor

### 单一代理的问题

假设一个代理直接访问所有日历和电子邮件 API：

❌ **问题**：
- 必须从众多相似工具中选择
- 理解每个 API 的精确格式
- 同时处理多个域
- 性能下降时难以调试

### Supervisor 模式的优势

✅ **好处**：
- **工具分区**：将相关工具和提示逻辑分组
- **专注提示**：每个子代理有针对性的指令
- **可扩展性**：轻松添加新领域而不影响现有功能
- **独立测试**：可以独立测试和改进每一层
- **清晰职责**：每层有明确的责任边界

## 工作流程

### 示例：复杂的多领域请求

```
用户: "安排下周二下午2点与设计团队的会议，并发送邮件提醒他们查看新原型"

Supervisor Agent:
  ↓ 识别需要两个领域：日历 + 邮件
  
1. 调用 schedule_event("下周二下午2点与设计团队的会议，1小时")
   ↓ Calendar Agent:
     - 解析 "下周二下午2点" → "2024-06-18T14:00:00"
     - 调用 create_calendar_event(...)
     - 返回: "已安排会议：6月18日 2:00 PM - 3:00 PM"

2. 调用 manage_email("提醒设计团队查看新原型")
   ↓ Email Agent:
     - 生成邮件主题和正文
     - 调用 send_email(...)
     - 返回: "已发送邮件提醒给设计团队"

Supervisor Agent:
  ↓ 综合两个结果
  
回复用户: "已完成！会议已安排在6月18日下午2-3点，并已发送邮件提醒。"
```

## 实现方式

### 1. 定义底层 API 工具

```python
from langchain.tools import tool

@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO: "2024-01-15T14:00:00"
    end_time: str,
    attendees: list[str],
    location: str = ""
) -> str:
    """创建日历事件。需要精确的 ISO 时间格式。"""
    # 实际应用中调用 Google Calendar API, Outlook API 等
    return f"事件已创建: {title} 从 {start_time} 到 {end_time}"

@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """通过电子邮件 API 发送邮件。"""
    # 实际应用中调用 SendGrid, Gmail API 等
    return f"邮件已发送给 {', '.join(to)}"
```

### 2. 创建专门化的子代理

```python
from langchain.agents import create_agent

# 日历代理：理解时间表达，转换为 ISO 格式
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt="""
    你是日历调度助手。
    解析自然语言的调度请求（如"下周二下午2点"）为正确的 ISO 时间格式。
    使用 get_available_time_slots 检查可用性。
    使用 create_calendar_event 安排事件。
    始终在最终回复中确认已安排的内容。
    """
)

# 邮件代理：生成专业邮件内容
email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt="""
    你是邮件助手。
    基于自然语言请求撰写专业邮件。
    提取收件人信息，生成合适的主题和正文。
    使用 send_email 发送消息。
    始终在最终回复中确认已发送的内容。
    """
)
```

### 3. 将子代理包装为工具

```python
@tool
def schedule_event(request: str) -> str:
    """使用自然语言安排日历事件。
    
    当用户想要创建、修改或检查日历约会时使用此工具。
    处理日期/时间解析、可用性检查和事件创建。
    
    输入: 自然语言调度请求
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

@tool
def manage_email(request: str) -> str:
    """使用自然语言发送电子邮件。
    
    当用户想要发送通知、提醒或任何邮件通信时使用此工具。
    处理收件人提取、主题生成和邮件撰写。
    
    输入: 自然语言邮件请求
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
```

**关键点**：
- 工具描述帮助 supervisor 决定何时使用每个工具
- 我们只返回子代理的最终响应（文本）
- Supervisor 不需要看到中间推理或工具调用

### 4. 创建 Supervisor Agent

```python
supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt="""
    你是一个有用的个人助理。
    你可以安排日历事件和发送电子邮件。
    将用户请求分解为适当的工具调用并协调结果。
    当请求涉及多个操作时，按顺序使用多个工具。
    """
)
```

### 5. 使用 Supervisor

```python
# 示例：需要协调多个领域的请求
query = (
    "安排下周二下午2点与设计团队的会议，1小时，"
    "并发送邮件提醒他们查看新原型。"
)

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

## 添加人机交互（HITL）

对于敏感操作，可以添加人工审批：

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# 为子代理配置 HITL
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="日历事件等待审批",
        ),
    ],
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="出站邮件等待审批",
        ),
    ],
)

# 注意：只需要在顶层代理添加 checkpointer
supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),  # 必需用于 HITL
)
```

**处理中断**：

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "session-1"}}

# 第一步：开始执行
interrupts = []
for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    config,
):
    if isinstance(update, tuple):
        interrupt = update[0]
        interrupts.append(interrupt)
        print(f"\n中断: {interrupt.id}")

# 第二步：审查并决策
resume = {}
for interrupt in interrupts:
    request = interrupt.value["action_requests"][0]
    
    # 显示待审批的操作
    print(f"工具: {request['tool']}")
    print(f"参数: {request['args']}")
    
    # 决策
    decision = input("批准(y)/拒绝(n)/修改(e): ")
    
    if decision == "y":
        resume[interrupt.id] = {"decisions": [{"type": "approve"}]}
    elif decision == "n":
        resume[interrupt.id] = {"decisions": [{"type": "reject"}]}
    elif decision == "e":
        edited_args = {...}  # 修改参数
        edited_action = request.copy()
        edited_action["arguments"] = edited_args
        resume[interrupt.id] = {
            "decisions": [{"type": "edit", "edited_action": edited_action}]
        }

# 第三步：恢复执行
for step in supervisor_agent.stream(Command(resume=resume), config):
    # 处理结果...
```

## 高级：控制信息流

### 传递额外的上下文给子代理

```python
from langchain.tools import ToolRuntime

@tool
def schedule_event(
    request: str,
    runtime: ToolRuntime
) -> str:
    """使用自然语言安排日历事件。"""
    # 获取原始用户消息
    original_message = next(
        msg for msg in runtime.state["messages"]
        if msg.type == "human"
    )
    
    # 自定义子代理接收的上下文
    prompt = (
        f"原始用户询问:\n{original_message.text}\n\n"
        f"你的子任务:\n{request}"
    )
    
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return result["messages"][-1].text
```

### 控制 Supervisor 接收的信息

```python
import json

@tool
def schedule_event(request: str) -> str:
    """使用自然语言安排日历事件。"""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    
    # 选项1：只返回确认消息
    return result["messages"][-1].text
    
    # 选项2：返回结构化数据
    # return json.dumps({
    #     "status": "success",
    #     "event_id": "evt_123",
    #     "summary": result["messages"][-1].text
    # })
```

## 使用场景

### 适合 Supervisor 模式的场景

✅ **推荐使用**：
- 多个不同的领域（日历、邮件、CRM、数据库）
- 每个领域有多个工具或复杂逻辑
- 需要集中的工作流控制
- 子代理不需要直接与用户对话

### 不适合的场景

❌ **不推荐**：
- 只有少量工具的简单场景 → 使用单一代理
- 需要代理与用户对话 → 使用 [Handoffs 模式](../handoffs-customer-support/)
- 代理之间需要点对点协作 → 使用其他多代理模式

## 最佳实践

### 1. 清晰的领域边界

- 每个子代理应该有明确的职责范围
- 避免职责重叠或模糊
- 工具应该逻辑分组

### 2. 专注的提示

- 为每个子代理编写针对性的系统提示
- 明确说明工具的使用场景
- 强调最终消息应包含所有相关信息

### 3. 清晰的工具描述

- Supervisor 的工具描述应该清晰易懂
- 说明何时使用每个工具
- 提供输入格式示例

### 4. 独立测试

- 先单独测试每个子代理
- 验证 API 工具的功能
- 最后测试完整的集成

### 5. 监控和调试

- 使用 LangSmith 追踪完整的执行流程
- 检查每一层的输入和输出
- 识别性能瓶颈

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
cd mutil-agents/subagents-personal-assistant

# 运行示例
uv run personal_assistant_example.py

# 运行 HITL 示例
uv run personal_assistant_hitl_example.py
```

## 技术对比

| 特性 | Supervisor 模式 | 单一代理 | Handoffs 模式 |
|------|----------------|---------|--------------|
| 工具组织 | 按领域分组 | 所有工具混在一起 | 按对话阶段 |
| 提示管理 | 每个子代理独立 | 单一复杂提示 | 每个阶段独立 |
| 可扩展性 | ✅ 高 | ⚠️ 中等 | ✅ 高 |
| 调试难度 | ⚠️ 中等 | ✅ 简单 | ⚠️ 中等 |
| 用户交互 | ❌ 间接 | ✅ 直接 | ✅ 直接 |
| LLM 调用次数 | 多次 | 少 | 多次 |

## 关键要点

1. **分层抽象**：每层有清晰的职责
2. **领域专家**：子代理专注于特定领域
3. **集中协调**：Supervisor 理解整体工作流
4. **独立测试**：每层可以独立开发和测试
5. **清晰接口**：子代理接受自然语言，返回自然语言

## 下一步

完成本教程后，你可以：

1. ✅ 尝试运行 `personal_assistant_example.py`
2. ✅ 添加新的领域（如任务管理、文件操作）
3. ✅ 实现人机交互审批机制
4. ✅ 使用 LangSmith 追踪和优化性能
5. ✅ 探索其他多代理模式：
   - [Handoffs 模式](../handoffs-customer-support/) - 用于对话式交互
   - [Router 模式](../router-knowledge-base/) - 用于知识库路由
   - [Skills 模式](../skills-sql-assistant/) - 用于技能共享

## 更多资源

- [LangChain 官方文档](https://docs.langchain.com/)
- [Multi-Agent 概述](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Context Engineering](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [LangSmith 可观测性](https://smith.langchain.com/)

## 总结

Supervisor 模式通过创建分层抽象来管理复杂性：

- **顶层**：理解意图，路由任务
- **中间层**：专注专家，处理领域逻辑
- **底层**：刚性 API，执行操作

这种架构提供了清晰的职责分离、易于扩展和独立测试的能力，是构建复杂多代理系统的强大模式！
