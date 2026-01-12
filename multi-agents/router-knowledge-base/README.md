# Router 模式：多源知识库路由

> 基于 LangChain 官方文档：[Build a multi-source knowledge base with routing](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base)

## 目录

- [什么是 Router 模式](#什么是-router-模式)
- [核心概念](#核心概念)
- [架构设计](#架构设计)
- [工作流程](#工作流程)
- [实现方式](#实现方式)
- [使用场景](#使用场景)
- [快速开始](#快速开始)

## 什么是 Router 模式

**Router（路由）模式**是一种多代理架构，其中一个路由代理根据查询内容智能地将请求分发到不同的专门化检索系统或知识库。

### 关键特点

- 🎯 **智能路由**：根据查询内容选择最合适的知识源
- 📚 **多源整合**：统一访问多个独立的知识库
- 🔀 **动态选择**：每个查询可以路由到不同的数据源
- 🎨 **独立配置**：每个知识源可以有自己的配置和优化

## 核心概念

### Router Agent（路由代理）

路由代理的职责：
- 分析用户查询的意图和主题
- 决定哪个知识库最合适
- 将查询转发到选定的检索系统
- 返回检索结果

### 知识库专家

每个知识库专家：
- 专注于特定领域或文档集合
- 有自己的向量存储和检索配置
- 可以使用不同的嵌入模型或检索策略
- 独立优化和维护

## 架构设计

### 多源知识库架构

```
┌─────────────────────────────────────┐
│         用户查询                     │
│   "如何使用 LangChain 构建代理?"     │
└─────────────┬───────────────────────┘
              │
              ↓
┌─────────────────────────────────────┐
│      Router Agent (路由层)           │
│  - 分析查询主题                      │
│  - 选择最佳知识源                    │
│  - 调用相应检索工具                  │
└─────────────┬───────────────────────┘
              │
       ┌──────┼──────┬─────────┐
       │      │      │         │
       ↓      ↓      ↓         ↓
  ┌────────┐ ┌────┐ ┌─────┐ ┌─────┐
  │LangChain│ │API │ │Blog │ │FAQ  │
  │  Docs  │ │Docs│ │Posts│ │ DB  │
  └────────┘ └────┘ └─────┘ └─────┘
  (知识库专家层 - 每个专注特定领域)
```

### 与其他模式的区别

| 特性 | Router 模式 | Subagents 模式 | Handoffs 模式 |
|------|------------|---------------|--------------|
| 代理交互 | 并行可选 | 顺序/并行 | 顺序必需 |
| 路由依据 | 查询内容 | 任务类型 | 工作流状态 |
| 知识库 | 多个独立 | 单个共享 | 单个共享 |
| 适用场景 | 多源检索 | 多领域任务 | 顺序收集 |

## 工作流程

### 示例：技术文档查询

```
用户查询: "LangChain 的 create_agent 函数如何使用?"

Router Agent 分析:
  - 主题: LangChain API
  - 领域: 技术文档
  - 决策: 路由到 "LangChain 官方文档" 知识库
  
  ↓ 调用 search_langchain_docs("create_agent 函数使用方法")

LangChain Docs Expert:
  - 在 LangChain 文档向量存储中搜索
  - 检索相关文档片段
  - 返回: "create_agent() 用于创建代理..."

Router Agent:
  ↓ 基于检索结果生成回答
  
返回用户: "要使用 create_agent 函数，你需要..."
```

### 示例：跨领域查询

```
用户查询: "最近的 AI 新闻有哪些?"

Router Agent 分析:
  - 主题: 新闻/博客
  - 时间: 最近
  - 决策: 路由到 "AI 博客文章" 知识库
  
  ↓ 调用 search_blog_posts("最近 AI 新闻")

Blog Posts Expert:
  - 在博客文章向量存储中搜索
  - 按时间排序
  - 返回最新文章摘要

Router Agent:
  ↓ 综合多篇文章
  
返回用户: "最近的 AI 新闻包括..."
```

## 实现方式

### 1. 创建多个向量存储

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)

# 知识库 1: LangChain 官方文档
langchain_docs_store = Chroma(
    collection_name="langchain_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db/langchain_docs"
)

# 知识库 2: API 参考文档
api_docs_store = Chroma(
    collection_name="api_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db/api_docs"
)

# 知识库 3: 博客文章
blog_posts_store = Chroma(
    collection_name="blog_posts",
    embedding_function=embeddings,
    persist_directory="./chroma_db/blog_posts"
)
```

### 2. 创建检索工具

```python
from langchain.tools import tool

@tool
def search_langchain_docs(query: str) -> str:
    """在 LangChain 官方文档中搜索。
    
    用于查询 LangChain 的概念、教程、指南等。
    适合: 概念解释、使用教程、最佳实践
    
    示例查询:
    - "如何创建 RAG 应用?"
    - "什么是 Agent?"
    - "LangChain 的架构是什么?"
    """
    docs = langchain_docs_store.similarity_search(query, k=3)
    return "\n\n".join([
        f"来源: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    ])

@tool
def search_api_docs(query: str) -> str:
    """在 API 参考文档中搜索。
    
    用于查询具体的函数、类、方法的使用方式。
    适合: API 参数、返回值、代码示例
    
    示例查询:
    - "create_agent 函数参数"
    - "ChatTongyi 类如何初始化?"
    - "AgentState 的字段有哪些?"
    """
    docs = api_docs_store.similarity_search(query, k=3)
    return "\n\n".join([
        f"API: {doc.metadata.get('api', 'unknown')}\n{doc.page_content}"
        for doc in docs
    ])

@tool
def search_blog_posts(query: str) -> str:
    """在博客文章中搜索。
    
    用于查询实战案例、技术分享、新功能介绍。
    适合: 实际应用、问题解决、趋势分析
    
    示例查询:
    - "多代理系统的实战案例"
    - "如何优化 RAG 性能?"
    - "最新的 LangChain 功能"
    """
    docs = blog_posts_store.similarity_search(query, k=3)
    return "\n\n".join([
        f"标题: {doc.metadata.get('title', 'unknown')}\n{doc.page_content}"
        for doc in docs
    ])
```

### 3. 创建 Router Agent

```python
from langchain.agents import create_agent

ROUTER_PROMPT = """你是一个智能路由助手，帮助用户从多个知识库中获取信息。

你有权访问三个专门的知识库：

1. **search_langchain_docs**: LangChain 官方文档
   - 用于: 概念解释、教程、指南、最佳实践
   - 示例: "什么是 RAG?", "如何创建 Agent?"

2. **search_api_docs**: API 参考文档
   - 用于: 函数用法、参数说明、代码示例
   - 示例: "create_agent 的参数", "ChatTongyi 类的方法"

3. **search_blog_posts**: 技术博客文章
   - 用于: 实战案例、问题解决、新功能介绍
   - 示例: "多代理系统案例", "性能优化技巧"

根据用户的查询，选择最合适的知识库进行搜索，然后基于检索结果回答问题。
如果需要，可以搜索多个知识库来获得更全面的答案。
"""

router_agent = create_agent(
    model,
    tools=[search_langchain_docs, search_api_docs, search_blog_posts],
    system_prompt=ROUTER_PROMPT,
)
```

### 4. 使用 Router Agent

```python
# 示例 1: 概念查询 → 路由到文档
query1 = "什么是 LangChain 的 Agent?"
for step in router_agent.stream(
    {"messages": [{"role": "user", "content": query1}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()

# 示例 2: API 查询 → 路由到 API 文档
query2 = "create_agent 函数的参数有哪些?"
for step in router_agent.stream(
    {"messages": [{"role": "user", "content": query2}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()

# 示例 3: 实战查询 → 路由到博客
query3 = "有哪些多代理系统的实际应用案例?"
for step in router_agent.stream(
    {"messages": [{"role": "user", "content": query3}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

## 高级功能

### 1. 多源融合

有时一个查询需要从多个知识库获取信息：

```python
ROUTER_PROMPT_MULTI = """...

当查询需要跨多个领域的信息时：
1. 先搜索最相关的知识库
2. 如果需要补充信息，再搜索其他知识库
3. 综合所有检索结果给出完整答案

例如: "如何使用 create_agent 构建 RAG 应用?"
- 先查 API 文档了解 create_agent 的用法
- 再查官方文档了解 RAG 的概念
- 可能还要查博客获取实战案例
"""
```

### 2. 元数据过滤

为检索添加过滤条件：

```python
@tool
def search_langchain_docs(
    query: str,
    version: str = "latest"
) -> str:
    """在 LangChain 官方文档中搜索，可以指定版本"""
    # 使用元数据过滤
    docs = langchain_docs_store.similarity_search(
        query,
        k=3,
        filter={"version": version}
    )
    return "\n\n".join([doc.page_content for doc in docs])
```

### 3. 重排序

使用更强大的模型对检索结果重排序：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

@tool
def search_with_reranking(query: str) -> str:
    """使用重排序的高级搜索"""
    # 基础检索器
    base_retriever = langchain_docs_store.as_retriever(search_kwargs={"k": 10})
    
    # 添加 LLM 重排序
    compressor = LLMChainExtractor.from_llm(model)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    docs = compression_retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])
```

### 4. 混合检索

结合向量搜索和关键词搜索：

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

@tool
def hybrid_search(query: str) -> str:
    """使用混合检索（向量 + 关键词）"""
    # 向量检索器
    vector_retriever = langchain_docs_store.as_retriever(search_kwargs={"k": 5})
    
    # BM25 关键词检索器
    # 注意: 需要预先加载文档
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 5
    
    # 组合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    docs = ensemble_retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])
```

## 使用场景

### 适合 Router 模式的场景

✅ **推荐使用**：
- **多源文档库**：公司有多个独立的文档系统
- **跨领域查询**：用户查询可能涉及不同专业领域
- **异构数据源**：不同类型的数据（文档、API、数据库）
- **独立维护**：各知识库由不同团队维护

### 典型应用

1. **企业知识管理**
   - 技术文档库
   - 产品文档库
   - 政策法规库
   - 历史案例库

2. **技术支持系统**
   - 产品手册
   - API 文档
   - 常见问题
   - 故障排查

3. **研究辅助工具**
   - 学术论文库
   - 实验数据库
   - 专利文档库
   - 行业报告库

4. **客户服务**
   - 产品信息库
   - 服务流程库
   - 用户反馈库
   - 解决方案库

### 不适合的场景

❌ **不推荐**：
- **单一数据源**：只有一个知识库 → 使用简单的 RAG Agent
- **固定路由**：查询总是访问相同的源 → 不需要路由
- **实时协作**：知识库之间需要交互 → 使用 Subagents 模式

## 最佳实践

### 1. 清晰的工具描述

- 明确说明每个知识库的覆盖范围
- 提供典型查询示例
- 说明适用和不适用的场景

### 2. 知识库优化

- 为每个领域使用专门的文档分割策略
- 考虑使用领域特定的嵌入模型
- 为不同知识库设置不同的检索参数（k 值、相似度阈值）

### 3. 路由策略

- 提供清晰的路由规则
- 允许跨知识库搜索
- 实现回退机制（如果首选知识库无结果）

### 4. 性能优化

- 缓存常见查询
- 并行搜索多个知识库（如适用）
- 使用异步 API 提高吞吐量

### 5. 监控和改进

- 追踪路由决策的准确性
- 监控各知识库的使用率
- 基于用户反馈优化路由逻辑

## 快速开始

### 前置要求

```bash
# 安装依赖
uv pip install langchain langchain-community langchain-chroma
uv pip install dashscope
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
cd mutil-agents/router-knowledge-base

# 运行示例
uv run router_example.py

# 运行交互式示例
uv run router_interactive.py
```

## 关键要点

1. **智能路由**：根据查询内容选择最佳知识源
2. **独立优化**：每个知识库可以独立配置和优化
3. **清晰描述**：工具描述是路由决策的关键
4. **多源整合**：统一接口访问多个数据源
5. **灵活扩展**：轻松添加新的知识库

## 下一步

完成本教程后，你可以：

1. ✅ 尝试运行 `router_example.py`
2. ✅ 添加新的知识库和检索工具
3. ✅ 实现混合检索或重排序
4. ✅ 优化路由决策逻辑
5. ✅ 使用 LangSmith 追踪路由行为
6. ✅ 探索其他多代理模式：
   - [Subagents 模式](../subagents-personal-assistant/) - 用于领域专家协调
   - [Handoffs 模式](../handoffs-customer-support/) - 用于状态机工作流
   - [Skills 模式](../skills-sql-assistant/) - 用于技能共享

## 更多资源

- [LangChain 官方文档](https://docs.langchain.com/)
- [Multi-Agent 概述](https://docs.langchain.com/oss/python/langchain/multi-agent)
- [RAG 教程](https://docs.langchain.com/oss/python/langchain/rag)
- [向量存储](https://docs.langchain.com/oss/python/integrations/vectorstores)

## 总结

Router 模式提供了一个优雅的解决方案来统一访问多个独立的知识库：

- **智能分发**：根据查询选择最合适的数据源
- **独立管理**：每个知识库可以独立维护和优化
- **灵活扩展**：轻松添加新的数据源
- **统一接口**：为用户提供无缝的查询体验

这是构建企业级知识管理系统和多源检索应用的理想模式！
