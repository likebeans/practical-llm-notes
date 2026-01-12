# LangChain 语义搜索引擎教程

本教程将指导您使用 LangChain 构建一个基于 PDF 文档的语义搜索引擎。该搜索引擎可以根据输入查询检索文档中相似的内容段落。

## 概述

本教程涵盖了以下核心概念：

1. **文档和文档加载器** - 如何加载和表示文档
2. **文本分割器** - 如何将文档分割成更小的块
3. **嵌入模型** - 如何将文本转换为向量表示
4. **向量存储** - 如何存储和检索向量数据
5. **检索器** - 如何创建和使用检索器

## 快速开始

### 使用 uv 创建项目环境

```bash
# 1. 创建虚拟环境
uv venv

# 2. 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 3. 激活虚拟环境（Linux/macOS）
source .venv/bin/activate

# 4. 从 requirements.txt 安装依赖（推荐）
uv pip install -r requirements.txt

# 如果你的包源只有 1.x 的预发布（例如 1.0.0a1），需要允许预发布：
uv pip install --prerelease=allow -r requirements.txt

# 5. 设置 API 密钥并运行示例
$env:QWEN_API_KEY="your-qwen-api-key"  # Qwen API 密钥
python semantic_search_example.py

# 或者创建 .env 文件，在文件中设置：
# QWEN_API_KEY=your-qwen-api-key
# QWEN_EMBEDDING_MODEL=text-embedding-v1  # 可选，默认使用 text-embedding-v1
```

## 前置要求

### 环境要求

- Python 3.8+
- uv 包管理器（[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)）
- LangChain（建议使用与你的包源可用版本一致的“稳定版”；如必须 1.x，请使用预发布安装方式）

### 使用 uv 安装依赖

本教程使用 `uv` 管理 Python 环境和依赖包。首先确保已安装 uv，然后使用以下命令安装所需依赖：

**方式一：使用 requirements.txt（稳定版，推荐）**

```bash
# 从 requirements.txt 安装所有依赖
uv pip install -r requirements.txt
```

**如果你安装时报 “No solution found / 只有 1.0.0a1 可用”**

> 你终端里的报错 `langchain-community<=1.0.0a1 is available` 说明包源里没有稳定的 `1.0.0`，只有 alpha（`a1`）版本。  
> 由于 `1.0.0a1 < 1.0.0`，所以 `>=1.0.0` 会被判定为不可满足。  
> 这时需要 **允许预发布**：

```bash
uv pip install --prerelease=allow -r requirements.txt
```

**方式二：手动安装**

```bash
# 使用 uv 安装依赖包
uv pip install langchain-community pypdf langchain-text-splitters langchain-openai chromadb langchain-chroma

# 或者使用 uv 创建虚拟环境并安装
uv venv
uv pip install langchain-community pypdf langchain-text-splitters langchain-openai chromadb langchain-chroma
```

### 版本说明

本教程的示例代码实际依赖这些包（注意：代码里用的是 `langchain-core`，并不需要显式安装 `langchain` 元包）：

- `langchain-core`
- `langchain-community`
- `langchain-openai`
- `langchain-text-splitters`
- `langchain-chroma`

如果你的包源提供稳定版本，直接使用：

```bash
uv pip install -r requirements.txt
```

如果你要走 1.x 预发布（alpha）路线，使用：

```bash
uv pip install --prerelease=allow -r requirements.txt
```

### 环境变量设置

在使用 DashScope (Qwen) 嵌入模型之前，需要设置 API 密钥：

**方式一：通过环境变量设置**

```bash
# Windows PowerShell
$env:QWEN_API_KEY="your-qwen-api-key-here"

# Linux/macOS
export QWEN_API_KEY="your-qwen-api-key-here"
```

**方式二：使用 .env 文件（推荐）**

在项目根目录创建 `.env` 文件：

```env
# Qwen API 密钥（必填）
# 获取地址: https://dashscope.console.aliyun.com/apiKey
QWEN_API_KEY=your-qwen-api-key-here

# Qwen 嵌入模型名称（可选，默认使用 text-embedding-v1）
# 可选值: text-embedding-v1, text-embedding-v2, text-embedding-v3
QWEN_EMBEDDING_MODEL=text-embedding-v1
```

代码会自动从 `.env` 文件加载环境变量（需要安装 `python-dotenv`）。

**获取 DashScope API 密钥**：
- 访问 [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)
- 注册/登录阿里云账号并创建 API 密钥

**注意**：如果使用 OpenAI 嵌入模型，环境变量名称为 `OPENAI_API_KEY`。

## 1. 文档和文档加载器

### 1.1 Document 抽象

LangChain 实现了 `Document` 抽象，用于表示一个文本单元及其关联的元数据。它包含三个属性：

- `page_content`: 表示内容的字符串
- `metadata`: 包含任意元数据的字典
- `id`: （可选）文档的字符串标识符

### 1.2 加载 PDF 文档

使用 `PyPDFLoader` 可以将 PDF 文件加载为一系列 `Document` 对象。每个 PDF 页面会生成一个 `Document` 对象。

```python
from langchain_community.document_loaders import PyPDFLoader

file_path = "example.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
```

每个文档包含：
- 页面的字符串内容
- 包含文件名和页码的元数据

### 1.3 文档分割

对于信息检索和后续的问答任务，单个页面可能过于粗糙。我们需要将文档分割成更小的块。

使用 `RecursiveCharacterTextSplitter`：
- 基于字符进行分割
- 可以设置块大小和重叠大小
- 重叠有助于避免将语句与其重要上下文分离
- 会递归地使用常见分隔符（如换行符）进行分割，直到每个块达到合适的大小

推荐参数：
- `chunk_size=1000`: 每个块 1000 个字符
- `chunk_overlap=200`: 块之间有 200 个字符的重叠
- `add_start_index=True`: 保留每个分割文档在原始文档中的起始字符索引

## 2. 嵌入模型

向量搜索是存储和搜索非结构化数据（如文本）的常见方法。核心思想是：
- 将文本转换为数值向量
- 给定查询时，将其嵌入为相同维度的向量
- 使用向量相似度指标（如余弦相似度）来识别相关文本

LangChain 支持来自数十个提供商的嵌入模型。本教程使用 DashScope (Qwen) 的嵌入模型。

### 设置 DashScope (Qwen) 嵌入

DashScope 是阿里云提供的大模型服务，支持 Qwen 系列嵌入模型。

```python
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",  # Qwen 文本嵌入模型 v1
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)
```

**获取 API 密钥**：
- 访问 [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)
- 注册/登录阿里云账号
- 创建 API 密钥

**注意**：如果你想使用 OpenAI 嵌入模型，可以改用：

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

## 3. 向量存储

向量存储用于存储文档的嵌入向量，并支持相似度搜索。

本教程使用 Chroma 作为向量存储。主要操作包括：

### 3.1 创建向量存储并添加文档

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings
)
```

### 3.2 相似度搜索

- `similarity_search(query, k=4)`: 基于查询字符串进行相似度搜索，返回 k 个最相似的文档
- `similarity_search_by_vector(embedding)`: 基于嵌入向量进行搜索
- `similarity_search_with_score(query, k=4)`: 返回文档及其相似度分数

### 3.3 最大边际相关性（MMR）搜索

MMR 在确保结果相关性的同时，优化结果的多样性，避免返回过于相似的文档。

```python
retrieved_docs = vector_store.max_marginal_relevance_search(
    "query", k=2, fetch_k=10
)
```

## 4. 检索器

检索器（Retriever）是 LangChain 中的 `Runnable` 对象，实现了标准的同步和异步方法。

### 4.1 从向量存储创建检索器

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)
```

### 4.2 检索器类型

- `"similarity"` (默认): 基于相似度的搜索
- `"mmr"`: 最大边际相关性搜索
- `"similarity_score_threshold"`: 基于相似度分数阈值的搜索

### 4.3 批量检索

检索器支持批量操作：

```python
results = retriever.batch([
    "查询1",
    "查询2"
])
```

## 使用场景

完成上述步骤后，您可以：

1. **语义搜索**: 在文档中查找与查询语义相似的内容
2. **RAG 应用**: 将检索到的文档作为上下文，结合 LLM 进行问答
3. **文档问答**: 基于特定文档回答用户问题

## 下一步

- 查看 [文档加载器概述](https://docs.langchain.com/docs/use_cases/document_loaders)
- 查看 [嵌入模型集成](https://docs.langchain.com/docs/integrations/text_embedding)
- 查看 [向量存储集成](https://docs.langchain.com/docs/integrations/vectorstores)
- 学习 [构建 RAG 应用](https://docs.langchain.com/docs/use_cases/question_answering)
- 学习 LangSmith Observability（Tracing）：见 `rag/langsmith_observability/langsmith_observability_quickstart.md`
- 学习 Voice Agent（语音代理）：见 `rag/voice_agent/README.md`

## 注意事项

1. **uv 环境管理**: 本教程使用 uv 管理依赖，确保已正确安装 uv 并激活虚拟环境
2. **版本选择**: 如果你的包源没有稳定的 `1.0.0`，只有 `1.0.0a1` 这类预发布，请在安装时加 `--prerelease=allow`
3. **API 密钥**: 确保正确设置 DashScope (Qwen) API 密钥，获取地址：https://dashscope.console.aliyun.com/apiKey
4. **PDF 文件**: 需要准备一个 PDF 文件用于测试，或使用示例代码中的示例文档
5. **依赖安装**: 建议使用 `uv pip install -r requirements.txt`；如果解析失败，再用 `uv pip install --prerelease=allow -r requirements.txt`
6. **内存使用**: 大型文档可能占用较多内存，注意调整 chunk_size 参数
7. **Chroma 数据库**: Chroma 会创建本地数据库文件，存储位置与运行方式/版本有关；如需固定路径，可在代码里显式设置 `persist_directory`
