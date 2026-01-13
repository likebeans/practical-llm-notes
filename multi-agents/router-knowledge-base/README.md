# Router 模式：多源知识库路由

> 基于 LangChain 官方文档：[Build a multi-source knowledge base with routing](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base)

## 目录

- [概述](#概述)
- [为什么使用 Router](#为什么使用-router)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [实现步骤](#实现步骤)
- [理解架构](#理解架构)
- [高级功能](#高级功能)
- [使用场景](#使用场景)

## 概述

**Router 模式**是一种多智能体架构，其中路由步骤对输入进行分类，并将其定向到专门化的 agents，然后将结果综合成组合响应。

当你的组织知识分布在不同的**垂直领域（verticals）**——每个都需要自己的专门化工具和提示的独立知识域时，这种模式表现出色。

### 示例场景

构建一个协调三个专家的多源知识库路由器：

- **GitHub agent**：搜索代码、issues 和 pull requests
- **Notion agent**：搜索内部文档和 wikis
- **Slack agent**：搜索相关线程和讨论

当用户问"如何验证 API 请求？"时，路由器将查询分解为针对特定源的子问题，并行路由到相关 agents，然后综合结果成为一致的答案。

## 为什么使用 Router

Router 模式提供几个优势：

- ✅ **并行执行**：同时查询多个源，相比顺序方法减少延迟
- ✅ **专门化 agents**：每个垂直领域有针对其域优化的工具和提示
- ✅ **选择性路由**：并非每个查询都需要每个源——路由器智能选择相关垂直领域
- ✅ **针对性子问题**：每个 agent 接收针对其域定制的问题，提高结果质量
- ✅ **清晰综合**：来自多个源的结果被组合成单一、连贯的响应

## 核心概念

### 三个关键阶段

```
1. 分类阶段 (Classification)
   - 分析查询
   - 确定相关的知识源
   - 为每个源生成针对性子问题

2. 并行路由 (Parallel Routing)
   - 使用 Send API 并行调用
   - 每个专门化 agent 处理其子问题
   - Reducer 收集所有结果

3. 综合阶段 (Synthesis)
   - 组合多源结果
   - 消除冗余
   - 生成连贯的最终答案
```

### 关键技术

- **StateGraph**：编排整个工作流
- **Send API**：实现并行执行
- **Reducer**：收集并行结果
- **Structured Output**：分类决策

### Router vs Subagents

- **Router 模式**：需要专门化预处理、自定义路由逻辑，或显式控制并行执行时使用
- **Subagents 模式**：希望 LLM 动态决定调用哪些 agents 时使用

## 架构设计

### 完整工作流

```
用户查询: "如何验证 API 请求？"
        ↓
┌─────────────────────────────────────┐
│   分类阶段 (classify_query)          │
│  - 分析查询内容                      │
│  - 确定相关源: github, notion        │
│  - 生成子问题:                       │
│    • github: "API 认证实现代码"      │
│    • notion: "API 认证文档"          │
└─────────────┬───────────────────────┘
              │
        ┌─────┴─────┐  (Send API - 并行执行)
        ↓           ↓
┌──────────────┐ ┌──────────────┐
│ GitHub Agent │ │ Notion Agent │
│ 搜索代码/PR  │ │ 搜索文档    │
└──────┬───────┘ └──────┬───────┘
       │                 │
       └────────┬────────┘
                ↓
    (Reducer: operator.add 收集结果)
                ↓
┌─────────────────────────────────────┐
│   综合阶段 (synthesize_results)      │
│  - 组合多源信息                      │
│  - 消除冗余                          │
│  - 生成最终答案                      │
└─────────────┬───────────────────────┘
              ↓
      最终答案返回给用户
```

### 状态定义

```python
class Classification(TypedDict):
    """单个路由决策"""
    source: Literal["github", "notion", "slack"]
    query: str

class RouterState(TypedDict):
    """主工作流状态"""
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  # Reducer
    final_answer: str
```

## 实现步骤

### 1. 定义状态

```python
from typing import Annotated, Literal, TypedDict
import operator

class AgentInput(TypedDict):
    """传递给每个子 agent 的简单状态"""
    query: str

class AgentOutput(TypedDict):
    """每个子 agent 返回的结果"""
    source: str
    result: str

class Classification(TypedDict):
    """单个路由决策：调用哪个 agent 以及什么查询"""
    source: Literal["github", "notion", "slack"]
    query: str

class RouterState(TypedDict):
    """主工作流状态"""
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]  # Reducer 收集并行结果
    final_answer: str
```

**关键点**：`results` 字段使用 **reducer** (`operator.add`) 来将并行 agent 执行的输出收集到单个列表中。

### 2. 为每个垂直领域定义工具

```python
from langchain.tools import tool

# GitHub 工具
@tool
def search_code(query: str, repo: str = "main") -> str:
    """在 GitHub 仓库中搜索代码"""
    return f"在 {repo} 中找到匹配 '{query}' 的代码: src/auth.py 中的认证中间件"

@tool
def search_issues(query: str) -> str:
    """搜索 GitHub issues 和 pull requests"""
    return f"找到 3 个匹配 '{query}' 的 issues: #142 (API 认证文档), #89 (OAuth 流程)"

@tool
def search_prs(query: str) -> str:
    """搜索 pull requests 以获取实现细节"""
    return f"PR #156 添加了 JWT 认证，PR #201 改进了令牌刷新逻辑"

# Notion 工具
@tool
def search_notion(query: str) -> str:
    """搜索 Notion 工作空间中的文档"""
    return f"在 Notion 中找到: '认证指南' 页面详细说明了 JWT 和 OAuth 流程"

@tool
def get_page(page_id: str) -> str:
    """获取特定 Notion 页面的完整内容"""
    return f"页面内容: API 认证最佳实践..."

# Slack 工具
@tool
def search_slack(query: str, days: int = 30) -> str:
    """搜索最近的 Slack 消息"""
    return f"在 #engineering 中找到讨论: Sarah 分享了认证设置，Mike 报告了令牌过期问题"

@tool
def get_thread(thread_id: str) -> str:
    """获取完整的 Slack 线程"""
    return f"线程讨论: 关于实现 JWT 认证的 10 条消息..."
```

### 3. 创建专门化的 Agents

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

# GitHub Agent
github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "你是一个 GitHub 专家。通过搜索代码、issues 和 pull requests "
        "来回答关于实现细节、API 参考和技术决策的问题。"
    ),
)

# Notion Agent
notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "你是一个 Notion 专家。通过搜索组织的 Notion 工作空间来回答 "
        "关于内部流程、政策和团队文档的问题。"
    ),
)

# Slack Agent
slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "你是一个 Slack 专家。通过搜索相关线程和讨论来回答问题，"
        "团队成员在其中分享了知识和解决方案。"
    ),
)
```

### 4. 构建路由工作流

#### 4.1 分类节点

```python
from langchain_core.pydantic_v1 import BaseModel, Field

class ClassificationResult(BaseModel):
    """分类结果的结构化输出"""
    classifications: list[Classification] = Field(
        description="要查询的源列表及其针对性子问题"
    )

def classify_query(state: RouterState) -> dict:
    """分类查询并确定要调用哪些 agents"""
    structured_llm = model.with_structured_output(ClassificationResult)
    
    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """分析此查询并确定要咨询哪些知识库。
为每个相关源生成一个针对该源优化的目标子问题。

可用源:
- github: 代码、API 参考、实现细节、issues、pull requests
- notion: 内部文档、流程、政策、团队 wikis
- slack: 团队讨论、非正式知识分享、最近对话

只返回与查询相关的源。"""
        },
        {"role": "user", "content": state["query"]}
    ])
    
    return {"classifications": result.classifications}
```

#### 4.2 路由到 Agents（并行执行）

```python
from langgraph.types import Send

def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类扇出到 agents（并行执行）"""
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]
```

**关键点**：`Send` API 实现并行执行。每个 `Send` 对象指定目标节点和输入。

#### 4.3 查询节点

```python
def query_github(state: AgentInput) -> dict:
    """查询 GitHub agent"""
    result = github_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {
        "results": [{
            "source": "github",
            "result": result["messages"][-1].content
        }]
    }

def query_notion(state: AgentInput) -> dict:
    """查询 Notion agent"""
    result = notion_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {
        "results": [{
            "source": "notion",
            "result": result["messages"][-1].content
        }]
    }

def query_slack(state: AgentInput) -> dict:
    """查询 Slack agent"""
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {
        "results": [{
            "source": "slack",
            "result": result["messages"][-1].content
        }]
    }
```

#### 4.4 综合节点

```python
def synthesize_results(state: RouterState) -> dict:
    """将所有 agents 的结果组合成连贯的答案"""
    if not state["results"]:
        return {"final_answer": "未从任何知识源找到结果。"}
    
    formatted = [
        f"**来自 {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]
    
    synthesis_response = model.invoke([
        {
            "role": "system",
            "content": f"""综合这些搜索结果以回答原始问题: "{state['query']}"

- 组合来自多个源的信息，避免冗余
- 突出最相关和可操作的信息
- 注意源之间的任何差异
- 保持响应简洁且组织良好"""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])
    
    return {"final_answer": synthesis_response.content}
```

### 5. 编译工作流

```python
from langgraph.graph import StateGraph, START, END

workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges(
        "classify",
        route_to_agents,
        ["github", "notion", "slack"]
    )
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

### 6. 使用路由器

```python
result = workflow.invoke({
    "query": "如何验证 API 请求？"
})

print("原始查询:", result["query"])
print("\n分类:")
for c in result["classifications"]:
    print(f"  {c['source']}: {c['query']}")
print("\n" + "=" * 60 + "\n")
print("最终答案:")
print(result["final_answer"])
```

**输出示例**：

```
原始查询: 如何验证 API 请求？

分类:
  github: 在我们的代码库中查找 API 认证实现
  notion: 查找 API 认证文档和指南

============================================================

最终答案:
要验证 API 请求，我们的系统使用 JWT 令牌：

从 GitHub 代码库:
- 认证中间件在 src/auth.py 中实现
- PR #156 添加了 JWT 认证支持
- 令牌验证逻辑在 src/middleware/auth.py

从 Notion 文档:
- '认证指南' 详细说明了 JWT 和 OAuth 流程
- 令牌应该在请求头中以 'Authorization: Bearer <token>' 发送
- 令牌过期时间设置为 24 小时

实施步骤：
1. 从 /auth/login 端点获取 JWT 令牌
2. 在后续请求中包含令牌
3. 后端中间件会验证令牌签名和过期时间
```

## 理解架构

### 分类阶段

分类节点使用 **structured output** 来决定：
- 哪些知识源是相关的
- 为每个源提出什么子问题

```python
# 输入
{"query": "如何验证 API 请求？"}

# 输出（structured）
{
    "classifications": [
        {"source": "github", "query": "API 认证实现代码"},
        {"source": "notion", "query": "API 认证文档"}
    ]
}
```

### 并行执行与 Send

`Send` API 实现真正的并行执行：

```python
def route_to_agents(state: RouterState) -> list[Send]:
    return [
        Send("github", {"query": "API 认证实现代码"}),
        Send("notion", {"query": "API 认证文档"})
    ]
```

这会**并行**调用 `query_github` 和 `query_notion` 节点。

### 结果收集与 Reducers

Reducer 自动收集并行结果：

```python
class RouterState(TypedDict):
    results: Annotated[list[AgentOutput], operator.add]  # Reducer
```

**工作原理**：
1. `query_github` 返回 `{"results": [{"source": "github", ...}]}`
2. `query_notion` 返回 `{"results": [{"source": "notion", ...}]}`
3. Reducer (`operator.add`) 将它们组合：`results = [...github...] + [...notion...]`

### 综合阶段

综合节点将多源结果组合成单一答案：
- 消除冗余信息
- 突出最相关内容
- 注意源之间的差异
- 生成连贯的响应

## 高级功能

### 1. 有状态路由器

基础路由器是**无状态的**——每个请求独立处理。对于多轮对话，有两种方法：

#### 工具包装方法（推荐）

```python
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """跨多个知识源搜索（GitHub、Notion、Slack）"""
    result = workflow.invoke({"query": query})
    return result["final_answer"]

conversational_agent = create_agent(
    model,
    tools=[search_knowledge_base],
    system_prompt=(
        "你是一个有用的助手，回答关于我们组织的问题。"
        "使用 search_knowledge_base 工具在我们的代码、文档和团队讨论中查找信息。"
    ),
    checkpointer=MemorySaver(),
)

# 多轮对话
config = {"configurable": {"thread_id": "user-123"}}

result = conversational_agent.invoke(
    {"messages": [{"role": "user", "content": "如何验证 API 请求？"}]},
    config
)

result = conversational_agent.invoke(
    {"messages": [{"role": "user", "content": "这些端点的速率限制如何？"}]},
    config
)
```

**优势**：
- 保持路由器无状态
- 会话 agent 处理记忆和上下文
- 清晰的责任分离

#### 完整持久化方法

如果需要路由器本身维护状态：

```python
from langgraph.checkpoint.memory import MemorySaver

workflow = (
    StateGraph(RouterState)
    # ... 添加节点 ...
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "conversation-1"}}
result1 = workflow.invoke({"query": "如何验证 API 请求？"}, config)
result2 = workflow.invoke({"query": "那些端点呢？"}, config)
```

**注意**：有状态路由器增加了复杂性。如果跨轮次路由到不同的 agents，对话可能会感觉不一致。考虑使用 handoffs 模式或 subagents 模式。

### 2. 自定义路由逻辑

添加业务规则到路由决策：

```python
def classify_query(state: RouterState) -> dict:
    """带自定义规则的分类"""
    # 获取 LLM 建议
    result = structured_llm.invoke(...)
    
    # 应用业务规则
    classifications = result.classifications
    
    # 规则: 代码相关查询始终包含 GitHub
    if any(word in state["query"].lower() for word in ["code", "api", "function"]):
        if not any(c["source"] == "github" for c in classifications):
            classifications.append({
                "source": "github",
                "query": state["query"]
            })
    
    # 规则: 政策查询始终包含 Notion
    if "policy" in state["query"].lower() or "process" in state["query"].lower():
        if not any(c["source"] == "notion" for c in classifications):
            classifications.append({
                "source": "notion",
                "query": state["query"]
            })
    
    return {"classifications": classifications}
```

### 3. 动态工具选择

根据查询复杂度调整每个 agent 的工具：

```python
def query_github(state: AgentInput) -> dict:
    """带动态工具选择的 GitHub 查询"""
    # 简单查询 - 只搜索代码
    if len(state["query"].split()) < 5:
        tools = [search_code]
    # 复杂查询 - 所有工具
    else:
        tools = [search_code, search_issues, search_prs]
    
    agent = create_agent(model, tools=tools, system_prompt=...)
    result = agent.invoke({"messages": [{"role": "user", "content": state["query"]}]})
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}
```

## 使用场景

### 适合 Router 模式的场景

✅ **推荐使用**：

1. **不同的垂直领域**
   - 多个独立的知识域
   - 每个需要专门化的工具和提示
   - 例如：代码 + 文档 + 讨论

2. **并行查询需求**
   - 问题受益于同时查询多个源
   - 需要低延迟响应
   - 可以独立处理的查询

3. **综合需求**
   - 来自多个源的结果需要组合
   - 需要消除冗余
   - 需要统一的答案格式

### 典型应用

1. **企业知识管理**
   - GitHub（代码）+ Notion（文档）+ Slack（讨论）
   - Jira（任务）+ Confluence（文档）+ Email（通信）

2. **技术支持系统**
   - 产品文档 + FAQ 数据库 + 历史工单
   - API 文档 + 代码示例 + 社区论坛

3. **研究助手**
   - 学术论文 + 专利数据库 + 行业报告
   - 内部研究 + 外部文献 + 实验数据

### 不适合的场景

❌ **不推荐**：

- **单一数据源**：只有一个知识库 → 使用简单的 RAG Agent
- **动态工具选择**：希望 LLM 决定调用哪些工具 → 使用 Subagents 模式
- **顺序工作流**：Agents 需要按特定顺序对话 → 使用 Handoffs 模式

## 最佳实践

### 1. 分类质量

- 在系统提示中提供清晰的源描述
- 为每个源包含示例查询
- 使用 structured output 确保一致的分类
- 测试边缘案例（模糊查询、多域查询）

### 2. 子问题优化

- 为每个源生成针对性的子问题
- 避免直接复制原始查询
- 考虑每个源的优势和限制
- 测试子问题是否能获得相关结果

### 3. 并行执行

- 确保 agents 真正独立（无共享状态）
- 使用 reducer 正确收集结果
- 处理部分失败（一些 agents 可能失败）
- 考虑超时设置

### 4. 综合质量

- 提供清晰的综合指令
- 突出最相关信息
- 消除冗余
- 保持答案简洁和可操作

### 5. 监控和调试

- 使用 LangSmith 追踪每个阶段
- 监控分类准确性
- 追踪每个源的使用频率
- 收集用户反馈改进路由逻辑

## 关键要点

1. **三个阶段**：分类 → 并行路由 → 综合
2. **Send API**：实现真正的并行执行
3. **Reducer**：自动收集并行结果
4. **Structured Output**：确保一致的分类决策
5. **专门化 Agents**：每个垂直领域有优化的工具和提示

## 下一步

完成本教程后，你可以：

1. ✅ 实现自己的多源路由器
2. ✅ 添加更多专门化的 agents
3. ✅ 实现自定义路由逻辑
4. ✅ 添加有状态对话支持
5. ✅ 使用 LangSmith 监控和优化
6. ✅ 探索其他多 agent 模式：
   - [Handoffs 模式](../handoffs-customer-support/) - 用于 agent-to-agent 对话
   - [Subagents 模式](../subagents-personal-assistant/) - 用于集中编排
   - [Skills 模式](../skills-sql-assistant/) - 用于渐进式披露

## 更多资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base)
- [Multi-Agent 概述](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [Send API 文档](https://docs.langchain.com/oss/python/langgraph/send-api)
- [StateGraph 文档](https://docs.langchain.com/oss/python/langgraph/state-graph)

## 总结

Router 模式通过三个阶段提供强大的多源查询能力：

- 📋 **分类**：分析查询并生成针对性子问题
- ⚡ **并行路由**：使用 Send API 同时查询多个专家
- 🎯 **综合**：组合多源结果成连贯答案

这是构建企业级知识管理系统、技术支持平台和研究助手的理想模式！
