"""
LangSmith Observability 示例
演示如何使用 LangSmith 追踪和监控 LangChain 应用程序

运行前确保设置环境变量：
- LANGSMITH_TRACING=true
- LANGSMITH_API_KEY=your-langsmith-api-key
- DASHSCOPE_API_KEY=your-dashscope-api-key（如使用 Qwen）
或
- OPENAI_API_KEY=your-openai-api-key（如使用 OpenAI）
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================================================
# 示例 1：使用 OpenAI 追踪简单 LLM 调用
# ============================================================================

def example_1_simple_llm_tracing():
    """示例 1：追踪简单的 LLM 调用"""
    print("\n=== 示例 1：追踪简单的 LLM 调用 ===\n")
    
    try:
        from openai import OpenAI
        from langsmith.wrappers import wrap_openai
        
        # 使用 LangSmith wrapper 包装 OpenAI 客户端
        client = wrap_openai(OpenAI())
        
        # 调用 LLM
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hello! What is LangSmith?"},
            ],
        )
        
        print(f"回答：{resp.choices[0].message.content}")
        print("\n✅ 请访问 LangSmith UI 查看 trace")
        
    except ImportError:
        print("❌ 未安装 OpenAI 库，请运行: uv add openai")
    except Exception as e:
        print(f"❌ 错误：{e}")


# ============================================================================
# 示例 2：使用 Qwen 追踪 LLM 调用
# ============================================================================

def example_2_qwen_tracing():
    """示例 2：使用 Qwen（通义千问）进行追踪"""
    print("\n=== 示例 2：使用 Qwen 追踪 LLM 调用 ===\n")
    
    try:
        from langsmith import traceable
        import dashscope
        from dashscope.aigc.generation import Generation
        
        # 设置 API Key
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        @traceable(name="qwen_chat")
        def chat_with_qwen(message: str) -> str:
            """使用 Qwen 进行对话"""
            response = Generation.call(
                model="qwen-turbo",
                messages=[
                    {"role": "user", "content": message}
                ]
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                return f"Error: {response.message}"
        
        # 调用
        question = "什么是 LangSmith？请用中文简短回答。"
        answer = chat_with_qwen(question)
        
        print(f"问题：{question}")
        print(f"回答：{answer}")
        print("\n✅ 请访问 LangSmith UI 查看 trace")
        
    except ImportError as e:
        print(f"❌ 未安装必要的库：{e}")
    except Exception as e:
        print(f"❌ 错误：{e}")


# ============================================================================
# 示例 3：追踪完整的 RAG 流程
# ============================================================================

def example_3_rag_tracing():
    """示例 3：追踪完整的 RAG 流程"""
    print("\n=== 示例 3：追踪完整的 RAG 流程 ===\n")
    
    try:
        from openai import OpenAI
        from langsmith.wrappers import wrap_openai
        from langsmith import traceable
        
        # 模拟检索器（实际应用中会连接向量数据库）
        @traceable(name="retriever")
        def retriever(query: str) -> list[str]:
            """模拟文档检索"""
            # 这里返回模拟数据，实际应用中会查询向量数据库
            knowledge_base = {
                "harrison": ["Harrison worked at Kensho", "Harrison is a developer"],
                "langsmith": ["LangSmith is an observability platform", "LangSmith helps debug LLM applications"],
                "default": ["No relevant information found"]
            }
            
            # 简单的关键词匹配
            for key in knowledge_base:
                if key.lower() in query.lower():
                    return knowledge_base[key]
            
            return knowledge_base["default"]
        
        # 包装 OpenAI 客户端
        client = wrap_openai(OpenAI())
        
        @traceable(name="rag_pipeline")
        def rag(question: str) -> str:
            """RAG 主流程"""
            # 1. 检索相关文档
            docs = retriever(question)
            
            # 2. 构建系统提示词
            system_message = (
                "Answer the user's question using only the provided information below:\n\n"
                + "\n".join(f"- {doc}" for doc in docs)
            )
            
            # 3. 调用 LLM 生成答案
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": question},
                ],
            )
            
            return resp.choices[0].message.content
        
        # 运行 RAG
        question = "Where did Harrison work?"
        answer = rag(question)
        
        print(f"问题：{question}")
        print(f"回答：{answer}")
        print("\n✅ 请访问 LangSmith UI 查看完整的 RAG trace")
        print("   你可以看到：retriever → rag_pipeline → LLM call 的完整流程")
        
    except ImportError:
        print("❌ 未安装 OpenAI 库，请运行: uv add openai")
    except Exception as e:
        print(f"❌ 错误：{e}")


# ============================================================================
# 示例 4：使用 LangChain 和 LangSmith
# ============================================================================

def example_4_langchain_tracing():
    """示例 4：使用 LangChain 与 LangSmith 集成"""
    print("\n=== 示例 4：LangChain 与 LangSmith 集成 ===\n")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langsmith import traceable
        
        # 创建 prompt 模板
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer briefly."),
            ("user", "{question}")
        ])
        
        # 创建 LLM
        llm = ChatOpenAI(model="gpt-4o-mini")
        
        # 创建输出解析器
        output_parser = StrOutputParser()
        
        # 构建链
        chain = prompt | llm | output_parser
        
        @traceable(name="langchain_qa")
        def ask_question(question: str) -> str:
            """使用 LangChain 链回答问题"""
            return chain.invoke({"question": question})
        
        # 运行
        question = "What is the capital of France?"
        answer = ask_question(question)
        
        print(f"问题：{question}")
        print(f"回答：{answer}")
        print("\n✅ 请访问 LangSmith UI 查看 trace")
        print("   LangChain 会自动追踪 prompt、LLM 调用和解析器")
        
    except ImportError as e:
        print(f"❌ 未安装必要的库：{e}")
        print("   请运行: uv add langchain-openai")
    except Exception as e:
        print(f"❌ 错误：{e}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("LangSmith Observability 示例程序")
    print("=" * 70)
    
    # 检查环境变量
    if not os.getenv("LANGSMITH_API_KEY"):
        print("\n⚠️  警告：未设置 LANGSMITH_API_KEY 环境变量")
        print("   请在 .env 文件中配置或设置环境变量")
        return
    
    if os.getenv("LANGSMITH_TRACING") != "true":
        print("\n⚠️  警告：未启用 LANGSMITH_TRACING")
        print("   请设置 LANGSMITH_TRACING=true")
        return
    
    print("\n✅ LangSmith 配置已就绪\n")
    
    # 运行示例
    # example_1_simple_llm_tracing()  # 需要 OpenAI API Key
    example_2_qwen_tracing()  # 需要 DashScope API Key
    # example_3_rag_tracing()  # 需要 OpenAI API Key
    # example_4_langchain_tracing()  # 需要 OpenAI API Key 和 langchain-openai
    
    print("\n" + "=" * 70)
    print("📊 如何查看 Traces：")
    print("1. 访问 https://smith.langchain.com")
    print("2. 登录你的账号")
    print("3. 进入对应的 Project")
    print("4. 查看 Traces 列表，点击查看详情")
    print("=" * 70)


if __name__ == "__main__":
    main()
