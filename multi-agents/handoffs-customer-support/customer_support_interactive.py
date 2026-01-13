"""
客户支持状态机（Handoffs）模式 - 交互式示例

这个示例允许你与客户支持代理进行真实的交互式对话，
体验状态机模式如何在不同阶段改变代理行为。
"""

import sys
import os

# 设置 UTF-8 编码（解决 Windows 中文/emoji 输出问题）
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import uuid
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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
try:
    model = init_chat_model(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    model_provider="openai",
    openai_api_key=os.environ.get("QWEN_API_KEY"),
    openai_api_base=os.environ.get("QWEN_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)
except Exception as e:
    print(f"❌ 初始化 Qwen LLM 失败: {e}")
    exit(1)

# ============================================================================
# 定义状态和工具（与 customer_support_example.py 相同）
# ============================================================================

SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

class SupportState(AgentState):
    """客户支持工作流的状态"""
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]

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
                    content=f"✅ 保修状态已记录为: {status}",
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
                    content=f"✅ 问题类型已记录为: {issue_type}",
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
    return f"📞 已升级到人工客服。原因: {reason}"

@tool
def provide_solution(solution: str) -> str:
    """提供解决方案给客户"""
    return f"✅ 解决方案: {solution}"

@tool
def go_back_to_warranty() -> Command:
    """返回保修验证步骤"""
    return Command(update={"current_step": "warranty_collector"})

@tool
def go_back_to_classification() -> Command:
    """返回问题分类步骤"""
    return Command(update={"current_step": "issue_classifier"})

# ============================================================================
# 步骤配置
# ============================================================================

WARRANTY_COLLECTOR_PROMPT = """你是客户支持代理，正在处理设备问题。

🔹 当前阶段: 保修验证

在此步骤，你需要：
1. 热情问候客户
2. 询问他们的设备是否在保修期内
3. 使用 record_warranty_status 记录回复并进入下一步

保持对话自然友好。不要一次问多个问题。"""

ISSUE_CLASSIFIER_PROMPT = """你是客户支持代理，正在处理设备问题。

🔹 当前阶段: 问题分类
📋 客户信息: 保修状态为 {warranty_status}

在此步骤，你需要：
1. 询问客户描述他们的问题
2. 判断是硬件问题（物理损坏、部件损坏）还是软件问题（应用崩溃、性能问题）
3. 使用 record_issue_type 记录分类并进入下一步

如果不确定，先询问澄清性问题。"""

RESOLUTION_SPECIALIST_PROMPT = """你是客户支持代理，正在处理设备问题。

🔹 当前阶段: 解决方案
📋 客户信息: 保修状态为 {warranty_status}，问题类型为 {issue_type}

在此步骤，你需要：
1. 对于软件问题: 使用 provide_solution 提供故障排除步骤
2. 对于硬件问题:
   - 如果在保修期内: 使用 provide_solution 说明保修维修流程
   - 如果过保: 使用 escalate_to_human 升级以获取付费维修选项

如果客户表示任何信息有误，使用:
- go_back_to_warranty 更正保修状态
- go_back_to_classification 更正问题类型

在解决方案中要具体和有帮助。"""

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
        "tools": [provide_solution, escalate_to_human, go_back_to_warranty, go_back_to_classification],
        "requires": ["warranty_status", "issue_type"],
    },
}

# ============================================================================
# 中间件
# ============================================================================

@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """基于当前步骤配置代理行为"""
    current_step = request.state.get("current_step", "warranty_collector")
    step_config = STEP_CONFIG[current_step]
    
    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"在到达 {current_step} 之前必须设置 {key}")
    
    system_prompt = step_config["prompt"].format(**request.state)
    
    request = request.override(
        system_prompt=system_prompt,
        tools=step_config["tools"],
    )
    
    return handler(request)

# ============================================================================
# 创建代理
# ============================================================================

all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
    go_back_to_warranty,
    go_back_to_classification,
]

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

# ============================================================================
# 交互式会话
# ============================================================================

def print_state_info(state):
    """打印当前状态信息"""
    step_names = {
        "warranty_collector": "保修验证",
        "issue_classifier": "问题分类",
        "resolution_specialist": "解决方案"
    }
    
    # 如果 current_step 是 None，使用默认值
    current_step = state.get('current_step') or 'warranty_collector'
    print(f"\n📊 当前阶段: {step_names.get(current_step, current_step)}")
    
    if state.get('warranty_status'):
        print(f"   保修状态: {state.get('warranty_status')}")
    if state.get('issue_type'):
        print(f"   问题类型: {state.get('issue_type')}")
    print()

def run_interactive_session():
    """运行交互式会话"""
    print("="*80)
    print("🤖 客户支持代理 - 交互式会话")
    print("="*80)
    print("\n欢迎！这是一个智能客户支持系统的演示。")
    print("输入 'quit' 或 'exit' 退出，输入 'status' 查看当前状态。\n")
    print("-"*80)
    
    # 创建会话
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 初始化状态
    state = {"current_step": "warranty_collector"}
    
    turn = 0
    while True:
        turn += 1
        
        # 获取用户输入
        user_input = input(f"\n[轮次 {turn}] 你: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n👋 感谢使用！再见！")
            break
        
        if user_input.lower() == 'status':
            print_state_info(state)
            turn -= 1
            continue
        
        # 调用代理
        try:
            result = agent.invoke(
                {"messages": [HumanMessage(user_input)]},
                config
            )
            
            # 更新状态
            state = {
                "current_step": result.get('current_step'),
                "warranty_status": result.get('warranty_status'),
                "issue_type": result.get('issue_type')
            }
            
            # 打印代理回复（取最后一条 AI 消息）
            print("\n助理:", end=" ")
            ai_messages = [msg for msg in result['messages'] if hasattr(msg, 'content') and msg.content and msg.type == 'ai']
            if ai_messages:
                print(ai_messages[-1].content)  # 打印最后一条
            else:
                print("(没有回复)")
            
            # 显示状态变化
            print_state_info(state)
            
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            turn -= 1

if __name__ == "__main__":
    run_interactive_session()
