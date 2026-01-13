"""
Router 模式：多源知识库路由示例

演示如何使用 Router 模式构建多源知识库系统，包括：
- 分类阶段：将查询分解为针对性子问题
- 并行路由：使用 Send API 同时查询多个 agents
- 综合阶段：组合多源结果成连贯答案
"""

import os
import operator
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_community.chat_models.tongyi import ChatTongyi
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# 加载环境变量
load_dotenv()

# 初始化模型
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY", ""),
)

# 路由器使用的模型（用于分类）
router_llm = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY", ""),
)


# ============================================================================
# 步骤 1: 定义状态
# ============================================================================

class AgentInput(TypedDict):
    """传递给每个子 agent 的简单输入状态"""
    query: str


class AgentOutput(TypedDict):
    """每个子 agent 返回的输出"""
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


class ClassificationResult(BaseModel):
    """分类结果的结构化输出"""
    classifications: list[Classification] = Field(
        description="要查询的源列表及其针对性子问题"
    )


# ============================================================================
# 步骤 2: 为每个垂直领域定义工具
# ============================================================================

# GitHub 工具
@tool
def search_code(query: str, repo: str = "main") -> str:
    """在 GitHub 仓库中搜索代码"""
    return f"在 {repo} 中找到匹配 '{query}' 的代码: src/auth.py 中的认证中间件使用 JWT 令牌"


@tool
def search_issues(query: str) -> str:
    """搜索 GitHub issues 和 pull requests"""
    return (
        f"找到 3 个匹配 '{query}' 的 issues:\n"
        "- #142: API 认证文档需要更新\n"
        "- #89: 实现 OAuth 2.0 流程\n"
        "- #203: 令牌刷新机制优化"
    )


@tool
def search_prs(query: str) -> str:
    """搜索 pull requests 以获取实现细节"""
    return (
        f"相关 PRs:\n"
        "- PR #156: 添加 JWT 认证支持，实现了令牌生成和验证\n"
        "- PR #201: 改进令牌刷新逻辑，支持自动续期"
    )


# Notion 工具
@tool
def search_notion(query: str) -> str:
    """搜索 Notion 工作空间中的文档"""
    return (
        f"在 Notion 中找到关于 '{query}' 的文档:\n"
        "- '认证指南': 详细说明了 JWT 和 OAuth 流程\n"
        "- 'API 安全最佳实践': 包含令牌管理和过期策略"
    )


@tool
def get_page(page_id: str) -> str:
    """获取特定 Notion 页面的完整内容"""
    return (
        f"页面 {page_id} 内容:\n"
        "# API 认证最佳实践\n"
        "1. 使用 JWT 令牌进行无状态认证\n"
        "2. 令牌应在请求头中以 'Authorization: Bearer <token>' 发送\n"
        "3. 令牌过期时间设置为 24 小时\n"
        "4. 实现令牌刷新机制以提供无缝用户体验"
    )


# Slack 工具
@tool
def search_slack(query: str, days: int = 30) -> str:
    """搜索最近的 Slack 消息"""
    return (
        f"在 #engineering 频道中找到关于 '{query}' 的讨论:\n"
        "- Sarah (2天前): 分享了新的 JWT 认证设置，包含配置示例\n"
        "- Mike (1周前): 报告了令牌过期导致的问题，已在 PR #201 中修复\n"
        "- Alex (2周前): 讨论了安全最佳实践，建议使用短期令牌"
    )


@tool
def get_thread(thread_id: str) -> str:
    """获取完整的 Slack 线程"""
    return (
        f"线程 {thread_id} 的完整讨论 (10 条消息):\n"
        "主要讨论了如何实现 JWT 认证，包括令牌生成、验证和刷新策略。\n"
        "团队决定使用 24 小时过期时间，并实现自动刷新机制。"
    )


# ============================================================================
# 步骤 3: 创建专门化的 Agents
# ============================================================================

# GitHub Agent
github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "你是一个 GitHub 专家。通过搜索代码、issues 和 pull requests "
        "来回答关于实现细节、API 参考和技术决策的问题。"
        "\n\n重要：提供简洁但完整的答案，包括相关的代码位置和 PR 引用。"
    ),
)

# Notion Agent
notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "你是一个 Notion 专家。通过搜索组织的 Notion 工作空间来回答 "
        "关于内部流程、政策和团队文档的问题。"
        "\n\n重要：从文档中提取最相关的指南和最佳实践。"
    ),
)

# Slack Agent
slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "你是一个 Slack 专家。通过搜索相关线程和讨论来回答问题，"
        "团队成员在其中分享了知识和解决方案。"
        "\n\n重要：总结团队讨论的关键见解和决策。"
    ),
)


# ============================================================================
# 步骤 4: 构建路由工作流
# ============================================================================

def classify_query(state: RouterState) -> dict:
    """分类查询并确定要调用哪些 agents"""
    structured_llm = router_llm.with_structured_output(ClassificationResult)
    
    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """分析此查询并确定要咨询哪些知识库。
为每个相关源生成一个针对该源优化的目标子问题。

可用源:
- github: 代码、API 参考、实现细节、issues、pull requests
- notion: 内部文档、流程、政策、团队 wikis、最佳实践
- slack: 团队讨论、非正式知识分享、最近对话、问题解决

**重要规则**:
1. 只返回与查询真正相关的源
2. 为每个源生成针对性的子问题（不要直接复制原始查询）
3. 子问题应该针对该源的特点优化
4. 如果查询明确只涉及一个源，只返回那个源

**示例**:
查询: "如何验证 API 请求？"
→ github: "查找 API 认证的实现代码和相关 PRs"
→ notion: "查找 API 认证的文档和配置指南"
→ slack: "查找团队关于 API 认证的讨论和实际经验"

查询: "PR #156 做了什么？"
→ github: "获取 PR #156 的详细信息和变更"
（只查询 GitHub，因为这是明确的 PR 查询）"""
        },
        {"role": "user", "content": state["query"]}
    ])
    
    print(f"\n📋 分类结果:")
    for c in result.classifications:
        print(f"  → {c['source']}: {c['query']}")
    print()
    
    return {"classifications": result.classifications}


def route_to_agents(state: RouterState) -> list[Send]:
    """根据分类扇出到 agents（并行执行）"""
    print(f"⚡ 并行路由到 {len(state['classifications'])} 个 agents...")
    return [
        Send(c["source"], {"query": c["query"]})
        for c in state["classifications"]
    ]


def query_github(state: AgentInput) -> dict:
    """查询 GitHub agent"""
    print(f"  🔍 GitHub agent 正在处理: {state['query']}")
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
    print(f"  📚 Notion agent 正在处理: {state['query']}")
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
    print(f"  💬 Slack agent 正在处理: {state['query']}")
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {
        "results": [{
            "source": "slack",
            "result": result["messages"][-1].content
        }]
    }


def synthesize_results(state: RouterState) -> dict:
    """将所有 agents 的结果组合成连贯的答案"""
    print(f"\n🎯 综合 {len(state['results'])} 个结果...")
    
    if not state["results"]:
        return {"final_answer": "未从任何知识源找到结果。"}
    
    formatted = [
        f"**来自 {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]
    
    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"""综合这些搜索结果以回答原始问题: "{state['query']}"

**综合指南**:
1. 组合来自多个源的信息，避免冗余
2. 突出最相关和可操作的信息
3. 如果源之间有差异，注明
4. 保持响应简洁且组织良好
5. 使用清晰的结构（如编号列表、要点）
6. 如果适用，提供实施步骤或示例

**输出格式**:
- 开始时简要直接回答问题
- 然后提供来自各源的支持细节
- 结束时提供可操作的下一步或建议"""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])
    
    return {"final_answer": synthesis_response.content}


# ============================================================================
# 步骤 5: 编译工作流
# ============================================================================

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


# ============================================================================
# 步骤 6: 测试路由器
# ============================================================================

def test_router():
    """测试路由器的不同场景"""
    
    print("=" * 80)
    print("Router 模式：多源知识库路由示例")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "场景 1: 跨源查询（需要多个源）",
            "query": "如何验证 API 请求？",
            "description": "预期路由到: GitHub（代码）、Notion（文档）、可能 Slack（讨论）"
        },
        {
            "name": "场景 2: 单源查询（只需要 GitHub）",
            "query": "PR #156 做了什么改动？",
            "description": "预期只路由到: GitHub"
        },
        {
            "name": "场景 3: 文档查询（只需要 Notion）",
            "query": "我们的 API 安全最佳实践是什么？",
            "description": "预期主要路由到: Notion，可能 Slack"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"{test_case['name']}")
        print("=" * 80)
        print(f"\n📝 用户查询: {test_case['query']}")
        print(f"💡 {test_case['description']}\n")
        
        try:
            result = workflow.invoke({
                "query": test_case["query"]
            })
            
            print(f"\n{'=' * 80}")
            print("📊 最终答案:")
            print("=" * 80)
            print(result["final_answer"])
            print()
            
        except Exception as e:
            print(f"❌ 错误: {e}\n")
        
        if i < len(test_cases):
            input("\n⏸️  按 Enter 继续下一个场景...")
    
    print("\n" + "=" * 80)
    print("✅ 所有示例完成！")
    print("=" * 80)
    
    print("\n💡 关键观察:")
    print("1. 分类阶段将查询分解为针对性子问题")
    print("2. Send API 实现了真正的并行执行")
    print("3. Reducer (operator.add) 自动收集并行结果")
    print("4. 综合阶段将多源结果组合成连贯答案")
    print("5. 不是每个查询都路由到所有源（智能选择）")
    
    print("\n📈 优势:")
    print("- 并行执行减少延迟")
    print("- 专门化 agents 提供高质量结果")
    print("- 选择性路由避免不必要的查询")
    print("- 综合阶段消除冗余并提供统一答案")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    test_router()


if __name__ == "__main__":
    main()
