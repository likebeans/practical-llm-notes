# Handoffs 模式：客户支持系统

> 基于 LangChain 官方文档：[Build customer support with handoffs](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support)

## 目录

- [什么是 Handoffs 模式](#什么是-handoffs-模式)
- [核心概念](#核心概念)
- [状态机架构](#状态机架构)
- [工作流程](#工作流程)
- [实现方式](#实现方式)
- [使用场景](#使用场景)
- [快速开始](#快速开始)

## 什么是 Handoffs 模式

**Handoffs（交接）模式**也称为**状态机模式（State Machine Pattern）**，描述了代理的行为随着任务的不同状态而变化的工作流程。

这个模式通过使用工具调用来动态改变单个代理的配置——基于当前状态更新可用工具和指令。

### 关键特点

- 🔄 **单一代理**：不是多个代理，而是一个代理在不同状态下有不同配置
- 📋 **状态驱动**：工具调用更新状态，触发配置变化
- 🎯 **专注指令**：每个状态有针对性的提示和工具
- 🔗 **顺序流程**：适合需要按顺序收集信息的场景

## 核心概念

### 状态（State）

状态是工作流的当前阶段，决定了：
- 代理的系统提示
- 可用的工具集
- 下一步可能的转换

### 转换（Transition）

通过工具调用来触发状态转换：
```python
@tool
def record_warranty_status(status: str, runtime: ToolRuntime) -> Command:
    """记录保修状态并转换到下一步"""
    return Command(update={
        "warranty_status": status,
        "current_step": "issue_classifier"  # 状态转换！
    })
```

### 中间件（Middleware）

中间件根据当前状态动态应用配置：
```python
@wrap_model_call
def apply_step_config(request: ModelRequest, handler) -> ModelResponse:
    """基于当前步骤配置代理行为"""
    current_step = request.state.get("current_step", "warranty_collector")
    config = STEP_CONFIG[current_step]
    
    request = request.override(
        system_prompt=config["prompt"],
        tools=config["tools"]
    )
    return handler(request)
```

## 状态机架构

### 客户支持工作流示例

```
┌──────────────────┐
│  warranty_       │
│  collector       │  ← 初始状态：收集保修信息
│  (保修验证)       │
└────────┬─────────┘
         │ record_warranty_status()
         ↓
┌──────────────────┐
│  issue_          │
│  classifier      │  ← 分类问题类型
│  (问题分类)       │
└────────┬─────────┘
         │ record_issue_type()
         ↓
┌──────────────────┐
│  resolution_     │
│  specialist      │  ← 提供解决方案
│  (解决方案)       │
└──────────────────┘
```

### 状态配置结构

```python
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": "询问保修状态...",
        "tools": [record_warranty_status],
        "requires": [],  # 无前置要求
    },
    "issue_classifier": {
        "prompt": "分类问题类型...",
        "tools": [record_issue_type],
        "requires": ["warranty_status"],  # 需要先有保修状态
    },
    "resolution_specialist": {
        "prompt": "提供解决方案...",
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],  # 需要前两步的信息
    },
}
```

## 工作流程

### 完整对话流程

```
【轮次 1】保修验证阶段
用户: "你好，我的手机屏幕碎了"
代理: "抱歉听到这个消息。请问您的设备还在保修期内吗？"
  状态: current_step = "warranty_collector"
  工具: [record_warranty_status]

【轮次 2】记录保修信息
用户: "是的，还在保修期内"
代理: 调用 record_warranty_status("in_warranty")
  ↓ 状态转换
  状态: current_step = "issue_classifier"
  工具: [record_issue_type]

【轮次 3】问题分类阶段
代理: "请描述您遇到的具体问题"
用户: "屏幕因摔落而破裂"
代理: 调用 record_issue_type("hardware")
  ↓ 状态转换
  状态: current_step = "resolution_specialist"
  工具: [provide_solution, escalate_to_human]

【轮次 4】解决方案阶段
代理: "由于这是硬件问题且在保修期内，我们可以为您免费维修..."
  调用 provide_solution("保修维修流程...")
```

## 实现方式

### 1. 定义自定义状态

```python
from langchain.agents import AgentState
from typing_extensions import NotRequired
from typing import Literal

# 定义可能的工作流步骤
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """客户支持工作流的状态"""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]
```

### 2. 创建管理工作流状态的工具

```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录客户的保修状态并转换到问题分类"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态已记录为: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",  # 状态转换！
        }
    )

@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录问题类型并转换到解决方案专家"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型已记录为: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",  # 状态转换！
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """升级到人工客服"""
    return f"已升级到人工客服。原因: {reason}"

@tool
def provide_solution(solution: str) -> str:
    """提供解决方案给客户"""
    return f"解决方案已提供: {solution}"
```

### 3. 定义步骤配置

```python
# 为每个步骤定义提示
WARRANTY_COLLECTOR_PROMPT = """你是客户支持代理，正在处理设备问题。

当前阶段: 保修验证

在此步骤，你需要：
1. 热情问候客户
2. 询问他们的设备是否在保修期内
3. 使用 record_warranty_status 记录回复并进入下一步

保持对话自然友好。不要一次问多个问题。"""

ISSUE_CLASSIFIER_PROMPT = """你是客户支持代理，正在处理设备问题。

当前阶段: 问题分类
客户信息: 保修状态为 {warranty_status}

在此步骤，你需要：
1. 询问客户描述他们的问题
2. 判断是硬件问题（物理损坏、部件损坏）还是软件问题（应用崩溃、性能问题）
3. 使用 record_issue_type 记录分类并进入下一步

如果不确定，先询问澄清性问题。"""

RESOLUTION_SPECIALIST_PROMPT = """你是客户支持代理，正在处理设备问题。

当前阶段: 解决方案
客户信息: 保修状态为 {warranty_status}，问题类型为 {issue_type}

在此步骤，你需要：
1. 对于软件问题: 使用 provide_solution 提供故障排除步骤
2. 对于硬件问题:
   - 如果在保修期内: 使用 provide_solution 说明保修维修流程
   - 如果过保: 使用 escalate_to_human 升级以获取付费维修选项

在解决方案中要具体和有帮助。"""

# 步骤配置: 将步骤名称映射到配置
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}
```

### 4. 创建基于步骤的中间件

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """基于当前步骤配置代理行为"""
    # 获取当前步骤（首次交互默认为 warranty_collector）
    current_step = request.state.get("current_step", "warranty_collector")
    
    # 查找步骤配置
    step_config = STEP_CONFIG[current_step]
    
    # 验证所需状态是否存在
    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"在到达 {current_step} 之前必须设置 {key}")
    
    # 使用状态值格式化提示（支持 {warranty_status}, {issue_type} 等）
    system_prompt = step_config["prompt"].format(**request.state)
    
    # 注入系统提示和步骤特定工具
    request = request.override(
        system_prompt=system_prompt,
        tools=step_config["tools"],
    )
    
    return handler(request)
```

### 5. 创建代理

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 收集所有步骤配置中的工具
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# 创建带有基于步骤配置的代理
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),  # 必需，用于跨轮次维护状态
)
```

### 6. 测试工作流

```python
from langchain.messages import HumanMessage
import uuid

# 此对话线程的配置
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# 轮次 1: 初始消息 - 从 warranty_collector 步骤开始
print("=== 轮次 1: 保修收集 ===")
result = agent.invoke(
    {"messages": [HumanMessage("你好，我的手机屏幕碎了")]},
    config
)
print(f"当前步骤: {result.get('current_step')}")

# 轮次 2: 用户回复保修问题
print("\n=== 轮次 2: 保修回复 ===")
result = agent.invoke(
    {"messages": [HumanMessage("是的，还在保修期内")]},
    config
)
print(f"当前步骤: {result.get('current_step')}")

# 轮次 3: 用户描述问题
print("\n=== 轮次 3: 问题描述 ===")
result = agent.invoke(
    {"messages": [HumanMessage("屏幕因摔落而物理破裂")]},
    config
)
print(f"当前步骤: {result.get('current_step')}")

# 轮次 4: 解决方案
print("\n=== 轮次 4: 解决方案 ===")
result = agent.invoke(
    {"messages": [HumanMessage("我应该怎么做？")]},
    config
)
```

## 理解状态转换

### 状态如何流转

```python
# 轮次 1: 初始状态
{
    "messages": [HumanMessage("你好，我的手机屏幕碎了")],
    "current_step": "warranty_collector"  # 默认值
}
# 中间件应用:
# - 系统提示: WARRANTY_COLLECTOR_PROMPT
# - 工具: [record_warranty_status]

# 轮次 2: 保修已记录
# 工具调用: record_warranty_status("in_warranty") 返回:
Command(update={
    "warranty_status": "in_warranty",
    "current_step": "issue_classifier"  # 状态转换！
})

# 下一轮次，中间件应用:
# - 系统提示: ISSUE_CLASSIFIER_PROMPT (格式化为 warranty_status="in_warranty")
# - 工具: [record_issue_type]

# 轮次 3: 问题已分类
# 工具调用: record_issue_type("hardware") 返回:
Command(update={
    "issue_type": "hardware",
    "current_step": "resolution_specialist"  # 状态转换！
})

# 下一轮次，中间件应用:
# - 系统提示: RESOLUTION_SPECIALIST_PROMPT (格式化为 warranty_status 和 issue_type)
# - 工具: [provide_solution, escalate_to_human]
```

## 添加灵活性：返回上一步

有时需要允许用户返回前一步来更正信息：

```python
@tool
def go_back_to_warranty() -> Command:
    """返回保修验证步骤"""
    return Command(update={"current_step": "warranty_collector"})

@tool
def go_back_to_classification() -> Command:
    """返回问题分类步骤"""
    return Command(update={"current_step": "issue_classifier"})

# 更新 resolution_specialist 配置以包含这些工具
STEP_CONFIG["resolution_specialist"]["tools"].extend([
    go_back_to_warranty,
    go_back_to_classification
])
```

更新提示以提及这些工具：

```python
RESOLUTION_SPECIALIST_PROMPT = """...

如果客户表示任何信息有误，使用:
- go_back_to_warranty 更正保修状态
- go_back_to_classification 更正问题类型

..."""
```

## 消息历史管理

随着代理在步骤中推进，消息历史会增长。使用汇总中间件压缩早期消息：

```python
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )
    ],
    checkpointer=InMemorySaver(),
)
```

## 使用场景

### 适合 Handoffs 模式的场景

✅ **推荐使用**：
- **顺序信息收集**：需要按特定顺序收集信息
- **多步骤流程**：每一步都需要前一步的结果
- **状态依赖行为**：代理行为严重依赖当前状态
- **对话式工作流**：直接与用户交互，逐步推进

### 典型应用

1. **客户支持**
   - 收集问题信息
   - 分类问题
   - 提供解决方案或升级

2. **表单填写**
   - 逐步收集字段
   - 验证输入
   - 提交完整表单

3. **诊断流程**
   - 收集症状
   - 分类问题类型
   - 提供诊断结果

4. **预订系统**
   - 选择服务
   - 收集个人信息
   - 确认和预订

### 不适合的场景

❌ **不推荐**：
- **无状态查询**：每个查询独立，不需要上下文
- **并行任务**：任务可以同时执行，不需要顺序
- **复杂分支**：有大量可能的路径和转换

对于这些场景，考虑：
- 单一代理（简单查询）
- Supervisor 模式（并行任务）
- Router 模式（基于内容路由）

## 最佳实践

### 1. 清晰的状态定义

- 为每个状态使用描述性名称
- 使用类型提示（Literal）限制可能的值
- 记录状态转换条件

### 2. 专注的提示

- 每个步骤的提示应该清晰明了
- 明确说明当前阶段和目标
- 包含可用工具的使用指南

### 3. 验证前置条件

- 在进入新步骤前检查所需状态
- 提供有用的错误消息
- 考虑添加"返回"工具用于更正

### 4. 状态持久化

- 始终使用 checkpointer 维护状态
- 为每个会话使用唯一的 thread_id
- 考虑使用持久化 checkpointer（如 SqliteSaver）

### 5. 测试状态转换

- 测试正常的状态流
- 测试边缘情况（缺失信息、无效转换）
- 测试"返回"功能

## 快速开始

### 前置要求

```bash
# 安装依赖
uv pip install langchain langchain-community
uv pip install langgraph langgraph-checkpoint
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
cd mutil-agents/handoffs-customer-support

# 运行基础示例
uv run customer_support_example.py

# 运行交互式示例
uv run customer_support_interactive.py
```

## 技术对比

| 特性 | Handoffs 模式 | Supervisor 模式 | 单一代理 |
|------|--------------|----------------|---------|
| 代理数量 | 1个（多配置） | 多个 | 1个 |
| 状态管理 | ✅ 显式 | ⚠️ 隐式 | ❌ 无 |
| 工作流控制 | ✅ 强 | ⚠️ 中等 | ❌ 弱 |
| 用户交互 | ✅ 直接 | ❌ 间接 | ✅ 直接 |
| 适合顺序流程 | ✅ 是 | ⚠️ 可以 | ❌ 否 |
| 适合并行任务 | ❌ 否 | ✅ 是 | ⚠️ 可以 |

## 关键要点

1. **单一代理，多种配置**：通过状态改变行为
2. **工具驱动转换**：工具调用更新 `current_step`
3. **中间件响应**：基于状态应用配置
4. **显式状态**：`current_step` 字段控制工作流
5. **需要 Checkpointer**：跨轮次维护状态

## 下一步

完成本教程后，你可以：

1. ✅ 尝试运行 `customer_support_example.py`
2. ✅ 添加新的步骤和状态转换
3. ✅ 实现更复杂的状态机
4. ✅ 添加"返回"功能用于更正
5. ✅ 使用 LangSmith 追踪状态转换
6. ✅ 探索其他多代理模式：
   - [Subagents 模式](../subagents-personal-assistant/) - 用于领域专家协调
   - [Router 模式](../router-knowledge-base/) - 用于知识库路由
   - [Skills 模式](../skills-sql-assistant/) - 用于技能共享

## 更多资源

- [LangChain 官方文档](https://docs.langchain.com/)
- [Multi-Agent 概述](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Middleware 指南](https://docs.langchain.com/oss/python/langchain/middleware)
- [LangSmith 可观测性](https://smith.langchain.com/)

## 总结

Handoffs（状态机）模式提供了一种优雅的方式来构建具有明确步骤和状态转换的对话式工作流：

- **单一代理**通过中间件在状态间转换
- **工具调用**驱动状态变化
- **专注配置**确保每个阶段的清晰性
- **Checkpointer** 维护跨轮次的状态

这是构建客户支持、表单填写、诊断系统等顺序工作流的理想模式！
