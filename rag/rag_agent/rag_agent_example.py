"""
LangChain RAG Agent 完整示例
基于官方文档: https://docs.langchain.com/oss/python/langchain/rag

本示例演示如何：
1. 构建知识库索引（复用语义搜索的步骤）
2. 创建 RAG Agent（带检索工具）
3. 创建 RAG Chain（固定检索流程）
4. 对比两种方式的效果
"""

import os
from pathlib import Path
from typing import List, Any

# 加载环境变量
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

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LangChain RAG Agent 完整教程                               ║
║                                                                              ║
║  本教程将展示如何构建智能 RAG 应用：                                              ║
║  1. 索引：加载和索引知识库                                                       ║
║  2. RAG Agent：让 LLM 自主决策何时检索                                          ║
║  3. RAG Chain：固定流程的快速检索                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# 第一部分：索引 - 构建知识库
# ============================================================================

print("=" * 80)
print("第一部分：索引 - 构建知识库")
print("=" * 80)

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 加载 PDF 文档
pdf_name = "./test.pdf"
file_path = str(Path(__file__).parent.parent / pdf_name)

print(f"\n正在加载文档: {pdf_name}")

try:
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"✅ 成功加载 PDF，共 {len(docs)} 页")
    print(f"\n第一页内容预览:\n{docs[0].page_content[:200]}...\n")
except FileNotFoundError:
    print(f"⚠️ 找不到文件 {file_path}，使用示例文档")
    docs = [
        Document(
            page_content="任务分解（Task Decomposition）是将复杂任务分解为更小、更简单的步骤的过程。"
            "这是一个重要的技术，可以让代理更好地理解和完成复杂任务。常见的方法包括 Chain of Thought (CoT)。",
            metadata={"source": "ai-concepts", "page": 1},
        ),
        Document(
            page_content="Chain of Thought (CoT) 是任务分解的标准方法。"
            "它通过让模型一步步思考来分解问题。常见的扩展包括 Tree of Thoughts 和 Self-Consistency。"
            "Tree of Thoughts 探索多个推理路径，Self-Consistency 通过多次采样来提高准确性。",
            metadata={"source": "ai-concepts", "page": 2},
        ),
        Document(
            page_content="人工智能代理（AI Agent）是可以自主决策和执行任务的系统。"
            "它们可以使用工具、访问外部资源，并根据反馈调整行为。"
            "LangChain 提供了强大的框架来构建这样的代理。",
            metadata={"source": "ai-concepts", "page": 3},
        ),
        Document(
            page_content="检索增强生成（RAG）结合了信息检索和语言生成。"
            "它允许 LLM 访问外部知识库，从而提供更准确、更新的信息。"
            "RAG 在问答系统、聊天机器人和知识助手中广泛应用。",
            metadata={"source": "ai-concepts", "page": 4},
        ),
    ]
    print(f"✅ 使用 {len(docs)} 个示例文档进行演示")

# 分割文档
print("\n正在分割文档...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"✅ 文档分割完成，共生成 {len(all_splits)} 个文档块")

# 设置嵌入模型
print("\n正在初始化嵌入模型...")
qwen_api_key = os.environ.get("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError(
        "未找到 QWEN_API_KEY 环境变量！\n"
        "请在 .env 文件中设置: QWEN_API_KEY=your_api_key_here"
    )

try:
    from langchain_community.embeddings import DashScopeEmbeddings
except ImportError:
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings

qwen_embedding_model = os.environ.get("QWEN_EMBEDDING_MODEL", "text-embedding-v1")
embeddings = DashScopeEmbeddings(
    model=qwen_embedding_model,
    dashscope_api_key=qwen_api_key
)
print(f"✅ 嵌入模型初始化完成（{qwen_embedding_model}）")

# 创建向量存储
print("\n正在创建向量存储...")
from langchain_chroma import Chroma
import tempfile
import shutil

# 方案一：使用临时目录（程序结束后自动清理）
# 创建临时目录用于存储 Chroma 数据
temp_dir = tempfile.mkdtemp()
print(f"📁 使用临时目录: {temp_dir}")

try:
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=temp_dir  # 使用临时目录
    )
    print("✅ 向量存储创建完成（临时模式，程序结束后会自动清理）")
except Exception as e:
    # 如果创建失败，清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)
    raise e

print(f"""
📊 索引统计:
  - 文档数量: {len(docs)} 页
  - 文档块数量: {len(all_splits)} 个
  - 嵌入模型: {qwen_embedding_model}
  - 向量存储: Chroma
""")

wait_for_user()

# ============================================================================
# 第二部分：初始化聊天模型
# ============================================================================

print("=" * 80)
print("第二部分：初始化聊天模型")
print("=" * 80)

# 设置 Qwen 聊天模型
print("\n正在初始化聊天模型...")

try:
    from langchain_community.chat_models.tongyi import ChatTongyi
except ImportError:
    raise ImportError(
        "请安装 dashscope 包: uv pip install dashscope"
    )

qwen_chat_model = os.environ.get("QWEN_CHAT_MODEL", "qwen-max")
llm = ChatTongyi(
    model=qwen_chat_model,
    dashscope_api_key=qwen_api_key,
    temperature=0.7
)
print(f"✅ 聊天模型初始化完成（{qwen_chat_model}）")

# 测试聊天模型
print("\n测试聊天模型...")
test_messages = [{"role": "user", "content": "你好，简单介绍一下自己"}]
response = llm.invoke(test_messages)
print(f"模型响应: {response.content[:100]}...")

wait_for_user()

# ============================================================================
# 第三部分：RAG Agent - 智能检索代理
# ============================================================================

print("=" * 80)
print("第三部分：RAG Agent - 智能检索代理")
print("=" * 80)

print("""
RAG Agent 的特点：
✅ 自主决策是否需要检索
✅ 可以执行多次检索
✅ 基于上下文生成更好的检索查询
⚠️ 需要多次 LLM 调用，延迟较高
""")

from langchain.tools import tool
from langchain.agents import create_agent

# 创建检索工具
@tool
def retrieve_context(query: str) -> str:
    """
    从知识库检索相关信息。
    
    参数:
        query: 要检索的查询字符串
        
    返回:
        检索到的相关文档内容
    """
    print(f"\n🔍 执行检索，查询: {query}")
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    if not retrieved_docs:
        return "未找到相关文档。"
    
    serialized = "\n\n---\n\n".join(
        f"📄 来源: {doc.metadata.get('source', '未知')}, "
        f"页码: {doc.metadata.get('page', '未知')}\n"
        f"内容: {doc.page_content}"
        for doc in retrieved_docs
    )
    
    print(f"✅ 检索到 {len(retrieved_docs)} 个相关文档")
    return serialized

# 创建工具列表
tools = [retrieve_context]

# 创建系统提示
system_prompt = """你是一个有用的 AI 助手。你可以使用检索工具从知识库中获取信息。

使用指南：
- 如果用户的问题需要特定的知识或事实，使用 retrieve_context 工具检索相关信息
- 对于简单的问候、闲聊或常识性问题，可以直接回答，无需检索
- 如果需要更多信息才能完整回答问题，可以多次使用检索工具
- 基于检索到的上下文提供准确、有帮助的答案
- 如果检索到的信息与问题无关，告知用户并尝试其他检索策略

请用中文回答。"""

# 创建 Agent（使用新的 create_agent API）
agent = create_agent(llm, tools, system_prompt=system_prompt)

print("\n✅ RAG Agent 创建完成")

# 测试 RAG Agent
print("\n" + "=" * 80)
print("测试 RAG Agent")
print("=" * 80)

test_queries = [
    "你好！",  # 简单问候，不需要检索
    "什么是任务分解？",  # 需要检索
    "Chain of Thought 的常见扩展有哪些？",  # 需要检索
]

for i, query in enumerate(test_queries, 1):
    print(f"\n{'─' * 80}")
    print(f"查询 {i}: {query}")
    print('─' * 80)
    
    try:
        # 使用新的 API，通过 stream 方式获取响应
        print("\n🤖 Agent 思考中...\n")
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            # 打印最后一条消息
            last_message = event["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                # 只在有实际内容时打印
                if last_message.type == "ai":
                    print(f"\n💬 Agent: {last_message.content}")
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
    
    if i < len(test_queries):
        print("\n⏸️  暂停 2 秒...")
        import time
        time.sleep(2)

wait_for_user()

# ============================================================================
# 第四部分：RAG Chain - 固定检索流程
# ============================================================================

print("=" * 80)
print("第四部分：RAG Chain - 固定检索流程")
print("=" * 80)

print("""
RAG Chain 的特点：
✅ 每个查询只需一次 LLM 调用
✅ 低延迟，快速响应
✅ 行为可预测，总是执行检索
⚠️ 不能跳过检索或多次检索
""")

from langchain.agents.middleware import dynamic_prompt, ModelRequest

# 使用 dynamic_prompt 中间件创建 RAG Chain
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """将检索到的上下文注入到提示中"""
    # 获取最后一条用户消息
    last_message = request.state["messages"][-1]
    last_query = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    # 执行检索
    print(f"\n🔍 正在检索相关文档...")
    retrieved_docs = vector_store.similarity_search(last_query, k=2)
    print(f"✅ 检索到 {len(retrieved_docs)} 个文档")
    
    # 格式化文档内容
    docs_content = "\n\n---\n\n".join(
        f"📄 来源: {doc.metadata.get('source', '未知')}, "
        f"页码: {doc.metadata.get('page', '未知')}\n"
        f"内容: {doc.page_content}"
        for doc in retrieved_docs
    )
    
    # 构建系统消息
    system_message = (
        "你是一个有用的 AI 助手。请基于以下上下文回答用户的问题。\n\n"
        f"上下文信息：\n{docs_content}\n\n"
        "请提供准确、有帮助的答案。如果上下文中没有足够的信息来回答问题，请诚实地说明。"
    )
    
    return system_message

# 创建 RAG Chain Agent（不使用工具，只使用 middleware）
rag_chain_agent = create_agent(llm, tools=[], middleware=[prompt_with_context])

print("\n✅ RAG Chain 创建完成（使用 dynamic_prompt 中间件）")

# 测试 RAG Chain
print("\n" + "=" * 80)
print("测试 RAG Chain")
print("=" * 80)

test_queries_chain = [
    "什么是任务分解？",
    "Chain of Thought 方法有什么扩展？",
    "解释一下什么是 RAG",
]

for i, query in enumerate(test_queries_chain, 1):
    print(f"\n{'─' * 80}")
    print(f"查询 {i}: {query}")
    print('─' * 80)
    
    try:
        # RAG Chain 使用 middleware 自动检索并注入上下文
        print("🤔 正在处理查询...")
        
        # 使用 stream 方式获取响应
        for event in rag_chain_agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            last_message = event["messages"][-1]
            # 只打印最终的 AI 回复
            if hasattr(last_message, 'type') and last_message.type == "ai":
                if hasattr(last_message, 'content') and last_message.content:
                    print(f"\n💬 回答:\n{last_message.content}\n")
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
    
    if i < len(test_queries_chain):
        print("\n⏸️  暂停 2 秒...")
        import time
        time.sleep(2)

wait_for_user()

# ============================================================================
# 第五部分：对比与总结
# ============================================================================

print("=" * 80)
print("第五部分：RAG Agent vs RAG Chain 对比")
print("=" * 80)

print("""
📊 性能对比总结：

┌─────────────────┬──────────────────┬──────────────────┐
│   特性          │   RAG Agent      │   RAG Chain      │
├─────────────────┼──────────────────┼──────────────────┤
│ LLM 调用次数    │   2+ 次          │   1 次           │
│ 响应延迟        │   较高           │   较低           │
│ 检索灵活性      │   高（按需）     │   低（总是检索） │
│ 多次检索        │   ✅ 支持        │   ❌ 不支持      │
│ 上下文感知查询  │   ✅ 支持        │   ❌ 不支持      │
│ 简单对话处理    │   ✅ 好          │   ⚠️ 会浪费检索  │
│ 行为可控性      │   ⚠️ 中等        │   ✅ 高          │
│ 适合场景        │   复杂多轮对话   │   简单文档查询   │
└─────────────────┴──────────────────┴──────────────────┘

🎯 选择建议：

RAG Agent 适用于：
  ✅ 需要多轮对话的应用
  ✅ 问题复杂，需要多次检索
  ✅ 需要处理多种类型的查询（包括不需要检索的）
  ✅ 对延迟不太敏感的场景

RAG Chain 适用于：
  ✅ 简单的文档问答
  ✅ 每个查询都需要检索的场景
  ✅ 对响应速度要求高的应用
  ✅ 需要可预测行为的系统

💡 实践建议：
  1. 从 RAG Chain 开始，如果发现局限性再切换到 RAG Agent
  2. 使用 LangSmith 追踪和分析两种方式的性能差异
  3. 根据实际用户查询模式选择合适的方案
  4. 可以在同一个应用中同时使用两种方式（例如，简单查询用 Chain，复杂查询用 Agent）
""")

print("\n" + "=" * 80)
print("教程总结")
print("=" * 80)

print("""
🎉 恭喜！你已经完成了 RAG Agent 完整教程！

本教程涵盖了：
1. ✅ 索引：加载文档、分割、向量化、存储
2. ✅ RAG Agent：使用工具的智能检索代理
3. ✅ RAG Chain：固定流程的快速检索链
4. ✅ 性能对比：理解两种方式的优劣

🚀 下一步建议：
  - 尝试使用自己的文档构建知识库
  - 实验不同的检索参数（k、相似度阈值）
  - 添加对话记忆以支持多轮对话
  - 实现查询重写和文档评分
  - 使用 LangSmith 优化性能
  - 部署到生产环境

📚 更多资源：
  - README.md: 详细的原理介绍和最佳实践
  - LangChain 官方文档: https://docs.langchain.com/
  - LangGraph Agentic RAG: 更高级的实现方式
  
Happy Building! 🎈
""")

# ============================================================================
# 清理临时文件
# ============================================================================

print("\n" + "=" * 80)
print("清理临时文件")
print("=" * 80)

try:
    print(f"\n🧹 正在清理临时目录: {temp_dir}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("✅ 临时文件清理完成")
except Exception as e:
    print(f"⚠️ 清理临时文件时出错: {e}")

print("\n程序结束。感谢使用！👋")
