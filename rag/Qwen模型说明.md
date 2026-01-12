# Qwen (通义千问) 模型配置说明

本项目的所有示例都使用阿里云的 **Qwen（通义千问）** 模型系列。

## 为什么选择 Qwen？

✅ **中文支持优秀**：专为中文优化，理解更准确  
✅ **性能强大**：在各类任务上表现优异  
✅ **价格实惠**：相比国际模型更经济  
✅ **本地化服务**：国内访问速度快，无需特殊网络  
✅ **官方支持**：阿里云提供稳定的 API 服务

## 使用的 Qwen 模型

### 1. 嵌入模型（Embedding Models）

用于将文本转换为向量，支持语义搜索。

| 模型名称 | 维度 | 特点 | 使用场景 |
|---------|------|------|---------|
| **text-embedding-v1** | 1536 | 通用，速度快 | ⭐ 推荐用于一般场景 |
| **text-embedding-v2** | 1536 | 性能更好 | 高精度需求 |
| **text-embedding-v3** | 2048 | 最新版本 | 最新特性 |

**代码示例：**

```python
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key="your_api_key"
)
```

### 2. 聊天模型（Chat Models）

用于对话和文本生成，支持 RAG Agent。

| 模型名称 | 特点 | 适用场景 |
|---------|------|---------|
| **qwen-max** | 最强能力 | ⭐ 推荐用于复杂推理、RAG Agent |
| **qwen-plus** | 平衡性能 | 一般对话场景 |
| **qwen-turbo** | 速度快 | 低延迟需求 |
| **qwen-long** | 长上下文 | 长文档处理 |

**代码示例：**

```python
from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(
    model="qwen-max",
    dashscope_api_key="your_api_key",
    temperature=0.7
)
```

## 环境配置

### 步骤 1：获取 API Key

1. 访问 [DashScope 控制台](https://dashscope.console.aliyun.com/apiKey)
2. 注册/登录阿里云账号
3. 创建 API Key
4. 复制 API Key 备用

### 步骤 2：配置环境变量

**方式一：创建 .env 文件（推荐）**

在项目根目录创建 `.env` 文件：

```env
# Qwen API 密钥（必填）
QWEN_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx

# 嵌入模型（可选，默认 text-embedding-v1）
QWEN_EMBEDDING_MODEL=text-embedding-v1

# 聊天模型（可选，默认 qwen-max）
QWEN_CHAT_MODEL=qwen-max
```

**方式二：设置系统环境变量**

Windows PowerShell:
```powershell
$env:QWEN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxx"
```

Linux/macOS:
```bash
export QWEN_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxx"
```

### 步骤 3：安装依赖

```bash
# 安装 dashscope 包（Qwen 模型支持）
uv pip install dashscope

# 或安装完整依赖
uv pip install -r requirements.txt
```

## 项目中的使用

### semantic_search（语义搜索）

使用 Qwen 嵌入模型进行语义搜索：

```bash
cd rag/semantic_search
uv run semantic_search_example.py
```

**使用的模型：**
- 嵌入模型：`text-embedding-v1`（默认）

### rag_agent（RAG 代理）

使用 Qwen 嵌入 + 聊天模型构建智能问答：

```bash
cd rag/rag_agent
uv run rag_agent_example.py
```

**使用的模型：**
- 嵌入模型：`text-embedding-v1`（默认）
- 聊天模型：`qwen-max`（默认）

## 成本估算

Qwen 模型定价（参考价格，以官网为准）：

### 嵌入模型
- text-embedding-v1: ~¥0.0007 / 1K tokens
- text-embedding-v2: ~¥0.0007 / 1K tokens

### 聊天模型
- qwen-max: ~¥0.04 / 1K tokens（输入），~¥0.12 / 1K tokens（输出）
- qwen-plus: ~¥0.004 / 1K tokens（输入），~¥0.012 / 1K tokens（输出）
- qwen-turbo: ~¥0.002 / 1K tokens（输入），~¥0.006 / 1K tokens（输出）

**示例成本：**
- 处理 1 万个文档块（每个 500 字符）≈ ¥5-10
- 100 次对话交互（包含检索）≈ ¥5-15

💡 **省钱技巧：**
1. 开发测试时使用 `qwen-turbo`
2. 生产环境根据需求选择合适模型
3. 合理设置文档块大小减少嵌入次数
4. 使用缓存避免重复嵌入

## 常见问题

### Q: 为什么不使用 OpenAI 模型？

**答：** 

1. **中文支持**：Qwen 专为中文优化，理解更准确
2. **访问便利**：国内访问无需代理，速度快
3. **价格优势**：比 OpenAI 便宜很多
4. **本地化服务**：数据处理在国内，更符合合规要求

当然，如果你需要使用 OpenAI，代码很容易切换：

```python
# 切换到 OpenAI 嵌入模型
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="your_openai_key"
)

# 切换到 OpenAI 聊天模型
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key="your_openai_key"
)
```

### Q: Qwen 模型支持哪些语言？

**答：** Qwen 主要优化中文和英文，也支持其他主流语言，但表现最好的是中英文。

### Q: 如何选择合适的模型？

**答：** 

**嵌入模型：**
- 一般场景：`text-embedding-v1`（速度快，性能好）
- 高精度需求：`text-embedding-v2` 或 `v3`

**聊天模型：**
- 复杂推理、RAG Agent：`qwen-max`
- 一般对话：`qwen-plus`
- 低延迟需求：`qwen-turbo`
- 超长文档：`qwen-long`

### Q: API Key 无效怎么办？

**答：** 检查：

1. ✅ API Key 是否正确复制（无多余空格）
2. ✅ API Key 是否已激活
3. ✅ 账号是否有足够余额
4. ✅ 环境变量名称是否正确（`QWEN_API_KEY`）

### Q: 如何查看 API 使用情况？

**答：** 访问 [DashScope 控制台](https://dashscope.console.aliyun.com/) 查看：
- API 调用次数
- Token 消耗量
- 费用统计
- 调用日志

## 更多资源

- 🌐 [DashScope 官方文档](https://help.aliyun.com/zh/dashscope/)
- 🔑 [获取 API Key](https://dashscope.console.aliyun.com/apiKey)
- 💰 [价格详情](https://help.aliyun.com/zh/dashscope/product-overview/billing-details)
- 📚 [Qwen 模型介绍](https://qwenlm.github.io/)
- 🛠️ [LangChain DashScope 集成](https://python.langchain.com/docs/integrations/providers/dashscope)

## 技术支持

遇到问题？

1. 查看本项目的 README 文档
2. 阅读 [DashScope 官方文档](https://help.aliyun.com/zh/dashscope/)
3. 查看 [LangChain 文档](https://python.langchain.com/)
4. 在项目中提交 Issue

---

**祝你使用愉快！** 🎉

如果觉得 Qwen 模型好用，欢迎分享给更多人！
