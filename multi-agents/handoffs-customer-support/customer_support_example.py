"""
客户支持状态机（Handoffs）模式示例

这个示例演示了状态机模式，其中单个代理根据 current_step 状态
动态改变其行为，创建顺序信息收集的状态机。
"""

import os
import uuid
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool, ToolRuntime
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Callable, Literal
from typing_extensions import NotRequired

# 加载环境变量
load_dotenv()

# 初始化 Qwen LLM
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)

# ============================================================================
# 步骤 1: 定义自定义状态
# ============================================================================

# 定义可能的工作流步骤
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """客户支持工作流的状态"""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]

# ============================================================================
# 步骤 2: 创建管理工作流状态的工具
# ============================================================================

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录客户的保修状态并转换到问题分类"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"保修状态已记录为: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )

@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """记录问题类型并转换到解决方案专家"""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"问题类型已记录为: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )

@tool
def escalate_to_human(reason: str) -> str:
    """升级到人工客服"""
    return f"已升级到人工客服。原因: {reason}"

@tool
def provide_solution(solution: str) -> str:
    """提供解决方案给客户"""
    return f"解决方案已提供: {solution}"

# ============================================================================
# 步骤 3: 定义步骤配置
# ============================================================================

# 为每个步骤定义提示
WARRANTY_COLLECTOR_PROMPT = """你是客户支持代理，正在处理设备问题。

当前阶段: 保修验证

在此步骤，你需要：
1. 热情问候客户
2. 询问他们的设备是否在保修期内
3. 使用 record_warranty_status 记录回复并进入下一步

保持对话自然友好。不要一次问多个问题。"""

ISSUE_CLASSIFIER_PROMPT = """你是客户支持代理，正在处理设备问题。

当前阶段: 问题分类
客户信息: 保修状态为 {warranty_status}

在此步骤，你需要：
1. 询问客户描述他们的问题
2. 判断是硬件问题（物理损坏、部件损坏）还是软件问题（应用崩溃、性能问题）
3. 使用 record_issue_type 记录分类并进入下一步

如果不确定，先询问澄清性问题。"""

RESOLUTION_SPECIALIST_PROMPT = """你是客户支持代理，正在处理设备问题。

当前阶段: 解决方案
客户信息: 保修状态为 {warranty_status}，问题类型为 {issue_type}

在此步骤，你需要：
1. 对于软件问题: 使用 provide_solution 提供故障排除步骤
2. 对于硬件问题:
   - 如果在保修期内: 使用 provide_solution 说明保修维修流程
   - 如果过保: 使用 escalate_to_human 升级以获取付费维修选项

在解决方案中要具体和有帮助。"""

# 步骤配置: 将步骤名称映射到配置
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}

# ============================================================================
# 步骤 4: 创建基于步骤的中间件
# ============================================================================

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """基于当前步骤配置代理行为"""
    # 获取当前步骤（首次交互默认为 warranty_collector）
    current_step = request.state.get("current_step", "warranty_collector")
    
    # 查找步骤配置
    step_config = STEP_CONFIG[current_step]
    
    # 验证所需状态是否存在
    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"在到达 {current_step} 之前必须设置 {key}")
    
    # 使用状态值格式化提示（支持 {warranty_status}, {issue_type} 等）
    system_prompt = step_config["prompt"].format(**request.state)
    
    # 注入系统提示和步骤特定工具
    request = request.override(
        system_prompt=system_prompt,
        tools=step_config["tools"],
    )
    
    return handler(request)

# ============================================================================
# 步骤 5: 创建代理
# ============================================================================

# 收集所有步骤配置中的工具
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# 创建带有基于步骤配置的代理
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

# ============================================================================
# 步骤 6: 测试工作流
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("客户支持状态机（Handoffs）模式示例")
    print("="*80)
    
    # 此对话线程的配置
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 轮次 1: 初始消息 - 从 warranty_collector 步骤开始
    print("\n【轮次 1】保修收集阶段")
    print("-"*80)
    result = agent.invoke(
        {"messages": [HumanMessage("你好，我的手机屏幕碎了")]},
        config
    )
    for msg in result['messages']:
        if hasattr(msg, 'content') and msg.content and msg.type != 'tool':
            print(f"助理: {msg.content}")
    print(f"\n状态: current_step = {result.get('current_step')}")
    
    # 轮次 2: 用户回复保修问题
    print("\n" + "="*80)
    print("\n【轮次 2】保修回复")
    print("-"*80)
    result = agent.invoke(
        {"messages": [HumanMessage("是的，还在保修期内")]},
        config
    )
    for msg in result['messages']:
        if hasattr(msg, 'content') and msg.content and msg.type != 'tool':
            print(f"助理: {msg.content}")
    print(f"\n状态: current_step = {result.get('current_step')}")
    print(f"      warranty_status = {result.get('warranty_status')}")
    
    # 轮次 3: 用户描述问题
    print("\n" + "="*80)
    print("\n【轮次 3】问题描述")
    print("-"*80)
    result = agent.invoke(
        {"messages": [HumanMessage("屏幕因摔落而物理破裂")]},
        config
    )
    for msg in result['messages']:
        if hasattr(msg, 'content') and msg.content and msg.type != 'tool':
            print(f"助理: {msg.content}")
    print(f"\n状态: current_step = {result.get('current_step')}")
    print(f"      warranty_status = {result.get('warranty_status')}")
    print(f"      issue_type = {result.get('issue_type')}")
    
    # 轮次 4: 解决方案
    print("\n" + "="*80)
    print("\n【轮次 4】解决方案")
    print("-"*80)
    result = agent.invoke(
        {"messages": [HumanMessage("我应该怎么做？")]},
        config
    )
    for msg in result['messages']:
        if hasattr(msg, 'content') and msg.content and msg.type != 'tool':
            print(f"助理: {msg.content}")
    
    print("\n" + "="*80)
    print("✅ 工作流完成！")
    print("="*80)
    print("\n📊 状态转换摘要:")
    print("  1. warranty_collector → 收集保修信息")
    print("  2. issue_classifier → 分类问题类型")
    print("  3. resolution_specialist → 提供解决方案")
