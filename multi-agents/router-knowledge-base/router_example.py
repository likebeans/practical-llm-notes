"""
Router 模式：多源知识库路由示例

这个示例演示了如何使用路由模式智能地从多个知识库中检索信息。
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool
from langchain.agents import create_agent

# 加载环境变量
load_dotenv()

# 初始化 Qwen LLM
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)

# ============================================================================
# 步骤 1: 创建检索工具（模拟多个知识库）
# ============================================================================

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
    # 模拟检索结果
    mock_results = {
        "agent": "Agent 是 LangChain 中的智能代理，能够使用工具自主决策完成任务。",
        "rag": "RAG (检索增强生成) 结合了信息检索和大语言模型，先检索相关文档，再基于文档生成答案。",
        "create": "使用 create_agent() 函数可以快速创建代理，需要指定模型、工具和系统提示。"
    }
    
    # 简单的关键词匹配
    for key, value in mock_results.items():
        if key in query.lower():
            return f"📚 LangChain 文档:\n{value}"
    
    return "📚 LangChain 文档:\n未找到完全匹配的结果，建议查看官方文档。"


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
    # 模拟 API 文档
    mock_api = {
        "create_agent": """
        📖 API 文档: create_agent
        
        函数签名:
        def create_agent(
            model: BaseLanguageModel,
            tools: List[Tool],
            system_prompt: str,
            state_schema: Type[AgentState] = None,
            middleware: List[Middleware] = None,
            checkpointer: Checkpointer = None
        ) -> Agent
        
        参数说明:
        - model: 使用的 LLM 模型
        - tools: 代理可用的工具列表
        - system_prompt: 系统提示，定义代理行为
        - state_schema: 自定义状态模式（可选）
        - middleware: 中间件列表（可选）
        - checkpointer: 状态检查点（可选）
        
        返回: Agent 实例
        """,
        "chattongyi": """
        📖 API 文档: ChatTongyi
        
        类初始化:
        ChatTongyi(
            model="qwen-plus",
            dashscope_api_key="your_api_key",
            temperature=0.7,
            top_p=0.9
        )
        
        常用参数:
        - model: 模型名称 (qwen-turbo, qwen-plus, qwen-max)
        - dashscope_api_key: API 密钥
        - temperature: 温度参数，控制随机性
        - top_p: 核采样参数
        """
    }
    
    for key, value in mock_api.items():
        if key.replace("_", "") in query.lower().replace("_", "").replace(" ", ""):
            return value
    
    return "📖 API 文档:\n未找到匹配的 API，建议查看 API 参考手册。"


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
    # 模拟博客文章
    mock_blogs = {
        "多代理": """
        📝 博客文章: 多代理系统实战案例
        
        在企业应用中，多代理系统被广泛用于：
        
        1. 客户服务自动化
           - 使用 Handoffs 模式处理客户咨询
           - 自动分类问题并路由到合适的处理流程
        
        2. 技术文档助手
           - 使用 Router 模式从多个文档库检索
           - Supervisor 模式协调文档生成和代码示例
        
        3. 数据分析平台
           - Skills 模式共享 SQL 查询能力
           - 多个分析代理协作生成报告
        """,
        "优化": """
        📝 博客文章: RAG 性能优化指南
        
        提升 RAG 系统性能的关键技巧：
        
        1. 文档处理优化
           - 合适的块大小 (500-1000 字符)
           - 添加块重叠以保持上下文
        
        2. 检索优化
           - 使用混合检索（向量 + 关键词）
           - 实现重排序提高相关性
           - 调整 k 值和相似度阈值
        
        3. 缓存策略
           - 缓存常见查询
           - 使用嵌入缓存减少 API 调用
        """
    }
    
    for key, value in mock_blogs.items():
        if key in query:
            return value
    
    return "📝 博客文章:\n暂无相关博客文章，建议查看最新技术博客。"


# ============================================================================
# 步骤 2: 创建 Router Agent
# ============================================================================

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
如果查询涉及多个方面，可以搜索多个知识库来获得更全面的答案。

重要: 始终先调用工具获取信息，然后基于检索结果回答问题。
"""

router_agent = create_agent(
    model,
    tools=[search_langchain_docs, search_api_docs, search_blog_posts],
    system_prompt=ROUTER_PROMPT,
)

# ============================================================================
# 步骤 3: 测试 Router Agent
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Router 模式：多源知识库路由示例")
    print("="*80)
    
    # 测试用例
    test_queries = [
        ("概念查询", "什么是 LangChain 的 Agent?"),
        ("API 查询", "create_agent 函数的参数有哪些?"),
        ("实战查询", "有哪些多代理系统的实际应用案例?"),
    ]
    
    for category, query in test_queries:
        print(f"\n【{category}】")
        print("-"*80)
        print(f"用户: {query}\n")
        
        for step in router_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            for update in step.values():
                for message in update.get("messages", []):
                    if hasattr(message, 'content') and message.content and message.type == 'ai':
                        print(f"助理: {message.content}")
        
        print("\n" + "="*80)
    
    print("\n✅ 示例完成！")
    print("\n💡 观察:")
    print("- Router Agent 根据查询内容自动选择了合适的知识库")
    print("- 概念查询 → LangChain 文档")
    print("- API 查询 → API 参考文档")
    print("- 实战查询 → 博客文章")
