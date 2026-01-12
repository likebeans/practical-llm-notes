"""
LangChain 语义搜索引擎完整示例
基于官方文档: https://docs.langchain.com/oss/python/langchain/knowledge-base

本示例演示如何：
1. 加载 PDF 文档
2. 分割文档
3. 创建嵌入向量
4. 存储到向量数据库
5. 进行语义搜索
"""

import os
from pathlib import Path
from typing import List

# 加载环境变量（从 .env 文件）
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

# ============================================================================
# 第一部分：文档和文档加载器
# ============================================================================

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

print("=" * 80)
print("第一部分：文档加载和分割")
print("=" * 80)

# 加载 PDF 文档
pdf_name = "test.pdf"
file_path = str(Path(__file__).parent.parent / pdf_name)

try:
    # 创建 PDF 加载器
    loader = PyPDFLoader(file_path)
    
    # 加载文档，每个 PDF 页面生成一个 Document 对象
    docs = loader.load()
    
    print(f"成功加载 PDF，共 {len(docs)} 页")
    
    # 查看第一个文档的内容（前 200 个字符）
    print(f"\n第一页内容预览:\n{docs[0].page_content[:200]}\n")
    
    # 查看第一个文档的元数据
    print(f"第一页元数据: {docs[0].metadata}")
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {file_path}")
    print("请将 'example.pdf' 替换为您实际的 PDF 文件路径")
    # 使用示例文档演示（如果没有实际 PDF 文件）
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc", "page": 1},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc", "page": 2},
        ),
        Document(
            page_content="The company reported strong financial results in 2023 with revenue growth of 10%.",
            metadata={"source": "mammal-pets-doc", "page": 3},
        ),
    ]
    docs = documents
    print(f"\n使用示例文档进行演示，共 {len(docs)} 个文档")

# ============================================================================
# 文档分割
# ============================================================================

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建文本分割器
# chunk_size: 每个块的大小（字符数）
# chunk_overlap: 块之间的重叠字符数，有助于保持上下文
# add_start_index: 保留每个分割文档在原始文档中的起始字符索引
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 每个块 1000 个字符
    chunk_overlap=200,    # 块之间重叠 200 个字符
    add_start_index=True  # 保留起始索引作为元数据
)

# 分割文档
all_splits = text_splitter.split_documents(docs)

print(f"\n文档分割完成，共生成 {len(all_splits)} 个文档块")

# 查看分割后的第一个文档块
if len(all_splits) > 0:
    print(f"\n第一个文档块内容预览:\n{all_splits[0].page_content[:200]}\n")
    print(f"第一个文档块元数据: {all_splits[0].metadata}")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 第二部分：嵌入模型
# ============================================================================

print("\n" + "=" * 80)
print("第二部分：嵌入模型设置")
print("=" * 80)

# 从环境变量读取 DashScope (Qwen) API 密钥
# DashScope 是阿里云提供的大模型服务，支持 Qwen 系列模型
# 请在 .env 文件中设置: QWEN_API_KEY=your_api_key_here
qwen_api_key = os.environ.get("QWEN_API_KEY")
if not qwen_api_key:
    raise ValueError(
        "未找到 QWEN_API_KEY 环境变量！\n"
        "请在 .env 文件中设置: QWEN_API_KEY=your_api_key_here\n"
        "DashScope API 密钥获取: https://dashscope.console.aliyun.com/apiKey"
    )

try:
    from langchain_community.embeddings import DashScopeEmbeddings
except ImportError:
    # 如果导入失败，尝试使用 dashscope 包
    try:
        from langchain_community.embeddings.dashscope import DashScopeEmbeddings
    except ImportError:
        raise ImportError(
            "请安装 dashscope 包: uv pip install dashscope\n"
            "DashScope API 密钥获取: https://dashscope.console.aliyun.com/apiKey"
        )

# 创建 DashScope (Qwen) 嵌入模型实例
# model: 指定使用的嵌入模型，从环境变量读取，如果没有则使用默认值
# - "text-embedding-v1": Qwen 文本嵌入模型 v1（默认）
# - "text-embedding-v2": Qwen 文本嵌入模型 v2（如果可用）
# 可以在 .env 文件中设置 QWEN_EMBEDDING_MODEL=text-embedding-v1
qwen_model = os.environ.get("QWEN_EMBEDDING_MODEL", "text-embedding-v1")
embeddings = DashScopeEmbeddings(
    model=qwen_model,  # 从环境变量读取，默认使用 text-embedding-v1
    dashscope_api_key=qwen_api_key
)

print(f"嵌入模型初始化完成（使用 DashScope {qwen_model}）")

# 测试嵌入：将单个查询转换为向量
test_query = "测试查询"
try:
    test_embedding = embeddings.embed_query(test_query)
    print(f"测试嵌入成功，向量维度: {len(test_embedding)}")
except Exception as e:
    print(f"嵌入测试失败: {e}")
    print("请检查您的 QWEN_API_KEY 是否正确设置")
    print("DashScope API 密钥获取地址: https://dashscope.console.aliyun.com/apiKey")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 第三部分：向量存储
# ============================================================================

print("\n" + "=" * 80)
print("第三部分：向量存储")
print("=" * 80)

from langchain_chroma import Chroma

# 创建向量存储并添加文档
# from_documents 方法会：
# 1. 为每个文档生成嵌入向量
# 2. 将文档和向量存储到 Chroma 数据库中
# 3. 返回一个 VectorStore 对象用于后续查询
print("正在创建向量存储并生成嵌入向量，这可能需要一些时间...")

try:
    vector_store = Chroma.from_documents(
        documents=all_splits,  # 要存储的文档列表
        embedding=embeddings   # 使用的嵌入模型
    )
    print("向量存储创建完成")
except Exception as e:
    print(f"向量存储创建失败: {e}")
    exit(1)

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 相似度搜索示例
# ============================================================================

print("\n" + "=" * 80)
print("相似度搜索示例")
print("=" * 80)

# 示例查询
query = "What are the main topics discussed?"

print(f"\n查询: {query}")
print("-" * 80)

# 方法 1: 基于查询字符串的相似度搜索
# similarity_search: 返回与查询最相似的 k 个文档
# k: 返回的文档数量
print("\n方法 1: 使用 similarity_search 进行搜索")
retrieved_docs = vector_store.similarity_search(query, k=4)

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n结果 {i}:")
    print(f"内容: {doc.page_content[:200]}...")  # 只显示前 200 个字符
    print(f"元数据: {doc.metadata}")

# 方法 2: 基于嵌入向量的相似度搜索
# 首先将查询转换为嵌入向量，然后进行搜索
print("\n" + "-" * 80)
print("\n方法 2: 使用 similarity_search_by_vector 进行搜索")
embedding = embeddings.embed_query(query)
results = vector_store.similarity_search_by_vector(embedding)

for i, doc in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"内容: {doc.page_content[:200]}...")
    print(f"元数据: {doc.metadata}")

# 方法 3: 带相似度分数的搜索
# similarity_search_with_score 返回文档和对应的相似度分数
print("\n" + "-" * 80)
print("\n方法 3: 使用 similarity_search_with_score 进行搜索（带分数）")
results_with_scores = vector_store.similarity_search_with_score(query, k=4)

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"\n结果 {i} (相似度分数: {score:.4f}):")
    print(f"内容: {doc.page_content[:200]}...")
    print(f"元数据: {doc.metadata}")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 最大边际相关性（MMR）搜索
# ============================================================================

print("\n" + "=" * 80)
print("最大边际相关性（MMR）搜索")
print("=" * 80)

# MMR 搜索在保证结果相关性的同时，优化结果的多样性
# k: 最终返回的文档数量
# fetch_k: 初始获取的候选文档数量（通常比 k 大）
# lambda_mult: 多样性权重，0 表示完全多样性，1 表示完全相关性
print(f"\n查询: {query}")
print("-" * 80)

retrieved_docs_mmr = vector_store.max_marginal_relevance_search(
    query, 
    k=2,      # 返回 2 个文档
    fetch_k=10  # 从 10 个最相似的候选中选择
)

for i, doc in enumerate(retrieved_docs_mmr, 1):
    print(f"\nMMR 结果 {i}:")
    print(f"内容: {doc.page_content[:200]}...")
    print(f"元数据: {doc.metadata}")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 第四部分：检索器
# ============================================================================

print("\n" + "=" * 80)
print("第四部分：检索器")
print("=" * 80)

# 从向量存储创建检索器
# as_retriever 方法将 VectorStore 转换为 Retriever 对象
# search_type: 搜索类型，可以是 "similarity"、"mmr" 或 "similarity_score_threshold"
# search_kwargs: 搜索的额外参数
retriever = vector_store.as_retriever(
    search_type="similarity",        # 使用相似度搜索
    search_kwargs={"k": 1}           # 返回 1 个最相似的文档
)

print("检索器创建完成")

# 单个查询检索
print("\n单个查询检索示例:")
single_query = "What is discussed in the document?"
results = retriever.invoke(single_query)
print(f"\n查询: {single_query}")
print(f"返回 {len(results)} 个结果:")
for i, doc in enumerate(results, 1):
    print(f"\n结果 {i}:")
    print(f"内容: {doc.page_content[:200]}...")
    print(f"元数据: {doc.metadata}")

# 批量查询检索
print("\n" + "-" * 80)
print("\n批量查询检索示例:")
queries = [
    "What are the main topics?",
    "When was the document created?",
]

# 使用 batch 方法进行批量检索
batch_results = retriever.batch(queries)

for query, results in zip(queries, batch_results):
    print(f"\n查询: {query}")
    print(f"返回 {len(results)} 个结果:")
    for i, doc in enumerate(results, 1):
        print(f"  结果 {i}: {doc.page_content[:150]}...")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 使用自定义函数创建检索器
# ============================================================================

print("\n" + "=" * 80)
print("使用自定义函数创建检索器")
print("=" * 80)

from langchain_core.runnables import chain

# 使用 @chain 装饰器创建自定义检索器
# 这个检索器包装了 vector_store.similarity_search 方法
@chain
def custom_retriever(query: str) -> List[Document]:
    """自定义检索器函数，返回与查询最相似的 1 个文档"""
    return vector_store.similarity_search(query, k=1)

# 使用自定义检索器进行批量查询
print("\n使用自定义检索器进行批量查询:")
custom_results = custom_retriever.batch(queries)

for query, results in zip(queries, custom_results):
    print(f"\n查询: {query}")
    for i, doc in enumerate(results, 1):
        print(f"  结果 {i}: {doc.page_content[:150]}...")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 使用相似度分数阈值检索器
# ============================================================================

print("\n" + "=" * 80)
print("使用相似度分数阈值检索器")
print("=" * 80)

# 创建基于相似度分数阈值的检索器
# 只返回相似度分数高于阈值的文档
threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.5}  # 分数阈值 0.5
)

print("相似度分数阈值检索器创建完成（阈值: 0.5）")

# 测试阈值检索器
test_query = "What are the main topics?"
threshold_results = threshold_retriever.invoke(test_query)

print(f"\n查询: {test_query}")
print(f"返回 {len(threshold_results)} 个结果（分数 >= 0.5）")

# 等待用户确认后继续
wait_for_user()

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("教程总结")
print("=" * 80)
print("""
您已经完成了 LangChain 语义搜索引擎的完整示例！

本示例涵盖了：
1. ✅ 文档加载：使用 PyPDFLoader 加载 PDF 文档
2. ✅ 文档分割：使用 RecursiveCharacterTextSplitter 分割文档
3. ✅ 嵌入生成：使用 DashScope (Qwen) 嵌入模型将文本转换为向量
4. ✅ 向量存储：使用 Chroma 存储和检索向量
5. ✅ 相似度搜索：多种搜索方法（相似度、MMR、带分数）
6. ✅ 检索器：创建和使用不同类型的检索器

下一步建议：
- 尝试使用您自己的 PDF 文档
- 调整 chunk_size 和 chunk_overlap 参数观察效果
- 尝试不同的嵌入模型
- 构建完整的 RAG 应用，将检索结果用于 LLM 问答

更多资源：
- 文档加载器：https://docs.langchain.com/docs/use_cases/document_loaders
- 嵌入模型：https://docs.langchain.com/docs/integrations/text_embedding
- 向量存储：https://docs.langchain.com/docs/integrations/vectorstores
- RAG 教程：https://docs.langchain.com/docs/use_cases/question_answering
""")
