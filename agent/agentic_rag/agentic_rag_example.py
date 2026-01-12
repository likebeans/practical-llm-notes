"""
Agentic RAG - 自定义 RAG Agent 示例
使用 LangGraph 构建的智能检索增强生成系统

功能特点:
1. 自主决策是否需要检索
2. 文档相关性自动评分
3. 查询自动优化重写
4. 完整的检索增强生成流程
"""

import os
import getpass
from typing import Literal

# LangChain 核心组件
from langchain_core.messages import HumanMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# LangChain 社区工具
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# LangGraph 图构建
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Pydantic 用于结构化输出
from pydantic import BaseModel, Field


# ============================================================================
# 1. 环境配置
# ============================================================================

def setup_environment():
    """设置 API 密钥"""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("请输入 OPENAI_API_KEY: ")


# ============================================================================
# 2. 文档预处理
# ============================================================================

def load_and_split_documents():
    """
    加载并分割文档
    
    步骤:
    1. 从网页加载文档
    2. 使用 RecursiveCharacterTextSplitter 分割文档
    3. 返回分割后的文档块
    """
    print("📄 正在加载文档...")
    
    # 文档来源: Lilian Weng 的技术博客
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]
    
    # 加载所有文档
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    print(f"✅ 已加载 {len(docs_list)} 篇文档")
    
    # 分割文档为小块
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100,      # 每块约 100 tokens
        chunk_overlap=50     # 块之间重叠 50 tokens
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    print(f"✅ 文档已分割为 {len(doc_splits)} 个块")
    
    return doc_splits


# ============================================================================
# 3. 创建检索工具
# ============================================================================

def create_retriever_tool(doc_splits):
    """
    创建向量存储和检索工具
    
    参数:
        doc_splits: 分割后的文档块列表
    
    返回:
        retriever_tool: 可被 Agent 调用的检索工具
    """
    print("🔍 正在创建向量存储...")
    
    # 创建内存向量存储
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    print("✅ 向量存储创建完成")
    
    # 使用 @tool 装饰器定义检索工具
    @tool
    def retrieve_blog_posts(query: str) -> str:
        """搜索并返回 Lilian Weng 博客相关内容"""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    
    return retrieve_blog_posts


# ============================================================================
# 4. 定义 Agent 节点
# ============================================================================

# 初始化模型（在节点函数外部，避免重复初始化）
response_model = None
grader_model = None

def initialize_models():
    """初始化 LLM 模型"""
    global response_model, grader_model
    response_model = init_chat_model("gpt-4o", temperature=0)
    grader_model = init_chat_model("gpt-4o", temperature=0)
    print("✅ 模型初始化完成")


def create_generate_query_or_respond(retriever_tool):
    """
    节点 1: 生成查询或直接回复
    
    功能:
    - 根据用户问题决定是否需要检索
    - 如果需要检索，生成 tool_call
    - 如果不需要，直接生成回复
    """
    def generate_query_or_respond(state: MessagesState):
        """调用模型生成响应或工具调用"""
        print("\n🤖 [generate_query_or_respond] 正在分析问题...")
        
        response = (
            response_model
            .bind_tools([retriever_tool])
            .invoke(state["messages"])
        )
        
        # 检查是否有工具调用
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"   → 决策: 需要检索 (查询: {response.tool_calls[0]['args']['query']})")
        else:
            print("   → 决策: 直接回复")
        
        return {"messages": [response]}
    
    return generate_query_or_respond


def create_grade_documents():
    """
    节点 2: 评估文档相关性
    
    功能:
    - 评估检索到的文档是否与问题相关
    - 返回路由决策: "generate_answer" 或 "rewrite_question"
    """
    # 定义评分模式
    class GradeDocuments(BaseModel):
        """文档相关性评分"""
        binary_score: str = Field(
            description="相关性评分: 'yes' 表示相关, 'no' 表示不相关"
        )
    
    GRADE_PROMPT = (
        "你是一个评估检索文档相关性的评分员。\n"
        "这是检索到的文档: \n\n {context} \n\n"
        "这是用户问题: {question} \n"
        "如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n"
        "给出二元评分 'yes' 或 'no' 来表示文档是否相关。"
    )
    
    def grade_documents(
        state: MessagesState,
    ) -> Literal["generate_answer", "rewrite_question"]:
        """评估检索文档的相关性"""
        print("\n📊 [grade_documents] 正在评估文档相关性...")
        
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = (
            grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )
        score = response.binary_score
        
        if score == "yes":
            print("   → 评估结果: 文档相关 ✓")
            return "generate_answer"
        else:
            print("   → 评估结果: 文档不相关 ✗ (将重写查询)")
            return "rewrite_question"
    
    return grade_documents


def create_rewrite_question():
    """
    节点 3: 重写查询
    
    功能:
    - 优化原始问题以提高检索效果
    - 返回改进后的查询
    """
    REWRITE_PROMPT = (
        "查看输入并尝试推理潜在的语义意图/含义。\n"
        "这是初始问题:\n"
        "------- \n"
        "{question}"
        "\n ------- \n"
        "请重新表述一个改进的问题:"
    )
    
    def rewrite_question(state: MessagesState):
        """重写原始用户问题"""
        print("\n✏️ [rewrite_question] 正在重写查询...")
        
        messages = state["messages"]
        question = messages[0].content
        
        prompt = REWRITE_PROMPT.format(question=question)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        
        print(f"   → 新查询: {response.content}")
        
        return {"messages": [HumanMessage(content=response.content)]}
    
    return rewrite_question


def create_generate_answer():
    """
    节点 4: 生成答案
    
    功能:
    - 基于检索到的上下文生成最终答案
    - 确保答案简洁准确
    """
    GENERATE_PROMPT = (
        "你是一个问答任务助手。"
        "使用以下检索到的上下文来回答问题。"
        "如果你不知道答案，就说不知道。"
        "最多使用三句话，保持答案简洁。\n"
        "问题: {question} \n"
        "上下文: {context}"
    )
    
    def generate_answer(state: MessagesState):
        """生成最终答案"""
        print("\n💡 [generate_answer] 正在生成答案...")
        
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = response_model.invoke([{"role": "user", "content": prompt}])
        
        print("   → 答案已生成 ✓")
        
        return {"messages": [response]}
    
    return generate_answer


# ============================================================================
# 5. 构建 LangGraph 图
# ============================================================================

def build_graph(retriever_tool):
    """
    构建 Agentic RAG 图
    
    图结构:
    START → generate_query_or_respond → [条件分支]
                                        ├→ 需要检索 → retrieve → grade_documents
                                        │                          ├→ 相关 → generate_answer → END
                                        │                          └→ 不相关 → rewrite_question → (回到开始)
                                        └→ 直接回答 → END
    """
    print("\n🔨 正在构建 Agent 图...")
    
    # 初始化模型
    initialize_models()
    
    # 创建所有节点
    generate_query_or_respond = create_generate_query_or_respond(retriever_tool)
    grade_documents = create_grade_documents()
    rewrite_question = create_rewrite_question()
    generate_answer = create_generate_answer()
    
    # 创建状态图
    workflow = StateGraph(MessagesState)
    
    # 添加节点
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    
    # 添加边
    workflow.add_edge(START, "generate_query_or_respond")
    
    # 条件边 1: 决定是否检索
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",      # 需要检索
            END: END,                  # 直接回答
        },
    )
    
    # 条件边 2: 评估文档相关性
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        # grade_documents 返回 "generate_answer" 或 "rewrite_question"
    )
    
    # 无条件边
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    
    # 编译图
    graph = workflow.compile()
    
    print("✅ Agent 图构建完成")
    
    return graph


# ============================================================================
# 6. 运行示例
# ============================================================================

def run_example(graph):
    """运行示例查询"""
    print("\n" + "="*70)
    print("🚀 开始运行 Agentic RAG 示例")
    print("="*70)
    
    # 示例问题
    questions = [
        "你好！今天天气怎么样？",  # 不需要检索的问题
        "Lilian Weng 提到了哪些类型的奖励黑客攻击？",  # 需要检索的问题
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"❓ 问题 {i}: {question}")
        print(f"{'='*70}")
        
        # 运行图并流式输出
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node != "__start__":  # 跳过开始节点
                    pass  # 节点内部已经打印信息
        
        # 获取最终结果
        final_state = graph.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        final_answer = final_state["messages"][-1].content
        
        print(f"\n✨ 最终答案:")
        print(f"   {final_answer}")
        print()


# ============================================================================
# 7. 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("  Agentic RAG - 自定义 RAG Agent 示例")
    print("="*70)
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 加载并分割文档
    doc_splits = load_and_split_documents()
    
    # 3. 创建检索工具
    retriever_tool = create_retriever_tool(doc_splits)
    
    # 4. 构建图
    graph = build_graph(retriever_tool)
    
    # 5. 运行示例
    run_example(graph)
    
    print("\n" + "="*70)
    print("✅ 示例运行完成！")
    print("="*70)


if __name__ == "__main__":
    main()
