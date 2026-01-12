# LangSmith Observability（可观测性）

> 基于 LangSmith 官方 Tracing Quickstart：[`https://docs.langchain.com/langsmith/observability-quickstart`](https://docs.langchain.com/langsmith/observability-quickstart)

## 概述

LangSmith Observability（可观测性）是一套强大的追踪和监控工具，可以帮助你调试和监控 LangChain 应用程序。通过 LangSmith，你可以：

- 📊 追踪完整的 LLM 调用链
- 🔍 查看每个步骤的输入输出
- ⏱️ 监控性能和延迟
- 🐛 快速定位和调试问题
- 📈 分析应用程序的使用模式

## 核心概念

### Tracing（追踪）

Tracing 是记录应用程序执行过程的技术。在 LangChain 中，tracing 可以捕获：
- LLM 调用及其响应
- 检索器查询和结果
- Agent 的决策过程
- 整个 RAG pipeline 的执行流程

### Workspace（工作区）

LangSmith 的工作区是组织和管理 traces 的容器，可以为不同的项目或环境创建不同的工作区。

## 依赖说明

本实例需要以下依赖包（已配置在项目根目录的 pyproject.toml）：

- **langsmith**：LangSmith SDK，用于追踪和监控
- **langchain**：LangChain 核心库
- **openai** 或 **dashscope**：LLM 提供商
- **python-dotenv**：环境变量管理

安装方式：

```bash
uv sync
```

## 1. 前置条件

- 需要一个 LangSmith 账号与 API Key（在 LangSmith 控制台创建）。
- 需要一个模型 Provider 的 API Key（官方示例用 OpenAI；如果你用 Qwen/DashScope 的 OpenAI 兼容模式，也同样适用）。

## 2. 环境变量（.env）

官方推荐的最小配置如下（请把占位符替换为你自己的值）：

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY="<your-langsmith-api-key>"
OPENAI_API_KEY="<your-llm-provider-api-key>"
# 可选：当你的 LangSmith API Key 关联多个 workspace 时指定
LANGSMITH_WORKSPACE_ID="<your-workspace-id>"
```

说明：
- `LANGSMITH_TRACING=true`：开启 tracing。
- `LANGSMITH_API_KEY`：LangSmith 的 key。
- `OPENAI_API_KEY`：模型 provider 的 key（示例里用 OpenAI；也可以是 Qwen/DashScope 兼容 OpenAI 的 key）。
- `LANGSMITH_WORKSPACE_ID`：可选项。

## 3. Python 最小示例：追踪 LLM 调用

官方思路是：用 `wrap_openai` 包一层 OpenAI 客户端，这样后续的模型调用会自动写入 LangSmith。

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai

client = wrap_openai(OpenAI())

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello!"},
    ],
)
print(resp.choices[0].message.content)
```

## 4. Python 最小示例：追踪整个 RAG 流程（推荐）

在追踪 LLM 调用的基础上，再用 `@traceable` 包住你的主流程函数（例如 `rag()`），这样可以在 LangSmith UI 里看到完整 pipeline（检索 → 组装 prompt → LLM）。

```python
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable

def retriever(query: str):
    return ["Harrison worked at Kensho"]

client = wrap_openai(OpenAI())

@traceable
def rag(question: str) -> str:
    docs = retriever(question)
    system_message = (
        "Answer the user's question using only the provided information below:\n"
        + "\n".join(docs)
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    print(rag("Where did Harrison work?"))
```

## 5. 在 LangSmith UI 查看 Trace

运行应用后，进入 LangSmith 的 Tracing Project（默认是 `default`，或你指定的 workspace）即可看到 traces。

