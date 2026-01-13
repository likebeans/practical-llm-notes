"""
个人助理 Supervisor 模式 - 带人机交互（HITL）示例

这个示例演示了如何在 Supervisor 模式中添加人机交互审批机制。
敏感操作（如创建日历事件和发送邮件）在执行前会暂停等待人工审批。
"""

import os
import uuid
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# 加载环境变量
load_dotenv()

# ============================================================================
# 步骤 1: 定义底层 API 工具（存根实现）
# ============================================================================

@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO 格式: "2024-01-15T14:00:00"
    end_time: str,    # ISO 格式: "2024-01-15T15:00:00"
    attendees: list[str],  # 邮箱地址
    location: str = ""
) -> str:
    """创建日历事件。需要精确的 ISO 日期时间格式。"""
    return f"✅ 事件已创建: {title} 从 {start_time} 到 {end_time}，{len(attendees)} 位参与者"


@tool
def send_email(
    to: list[str],      # 邮箱地址
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """通过电子邮件 API 发送邮件。需要格式正确的邮箱地址。"""
    return f"✅ 邮件已发送给 {', '.join(to)} - 主题: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO 格式: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """检查指定日期给定参与者的日历可用性。"""
    return ["09:00", "14:00", "16:00"]


# ============================================================================
# 步骤 2: 创建带 HITL 的专门化子代理
# ============================================================================

# 初始化 Qwen LLM
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)

# 日历代理 - 在创建事件前暂停审批
calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "你是一个日历调度助手。"
        "解析自然语言的调度请求（例如，'下周二下午2点'）为正确的 ISO 日期时间格式。"
        "需要时使用 get_available_time_slots 检查可用性。"
        "使用 create_calendar_event 安排事件。"
        "始终在最终回复中确认已安排的内容。"
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="📅 日历事件等待审批",
        ),
    ],
)

# 邮件代理 - 在发送邮件前暂停审批
email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "你是一个邮件助手。"
        "基于自然语言请求撰写专业邮件。"
        "提取收件人信息，生成合适的主题和正文。"
        "使用 send_email 发送消息。"
        "始终在最终回复中确认已发送的内容。"
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="📧 出站邮件等待审批",
        ),
    ],
)

# ============================================================================
# 步骤 3: 将子代理包装为 supervisor 的工具
# ============================================================================

@tool
def schedule_event(request: str) -> str:
    """使用自然语言安排日历事件。
    
    当用户想要创建、修改或检查日历约会时使用此工具。
    处理日期/时间解析、可用性检查和事件创建。
    
    输入: 自然语言调度请求（例如，'下周二下午2点与设计团队开会'）
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


@tool
def manage_email(request: str) -> str:
    """使用自然语言发送电子邮件。
    
    当用户想要发送通知、提醒或任何电子邮件通信时使用此工具。
    处理收件人提取、主题生成和电子邮件撰写。
    
    输入: 自然语言电子邮件请求（例如，'提醒他们关于会议'）
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].content


# ============================================================================
# 步骤 4: 创建带 checkpointer 的 supervisor 代理
# ============================================================================

# 注意：只需要在顶层代理添加 checkpointer
supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "你是一个有用的个人助理。"
        "你可以安排日历事件和发送电子邮件。"
        "将用户请求分解为适当的工具调用并协调结果。"
        "当请求涉及多个操作时，按顺序使用多个工具。"
    ),
    checkpointer=InMemorySaver(),  # 必需用于 HITL
)

# ============================================================================
# 步骤 5: 使用带 HITL 的 supervisor
# ============================================================================

def print_interrupt_info(interrupt):
    """打印中断信息"""
    print("\n" + "🔔"*40)
    print("⚠️  检测到中断 - 等待人工审批")
    print("🔔"*40)
    
    # 调试：打印实际的数据结构
    print("\n🔍 调试信息:")
    print(f"Interrupt ID: {interrupt.id}")
    print(f"Interrupt value keys: {interrupt.value.keys()}")
    
    if "action_requests" in interrupt.value:
        for i, request in enumerate(interrupt.value["action_requests"], 1):
            print(f"\n📋 请求 {i}:")
            print(f"Request keys: {request.keys()}")
            print(f"Request content: {request}")
            
            # 尝试不同的键名
            if 'tool' in request:
                print(f"   工具: {request['tool']}")
            if 'description' in request:
                print(f"   描述: {request['description']}")
            if 'args' in request:
                print(f"   参数: {request['args']}")
            if 'arguments' in request:
                print(f"   参数: {request['arguments']}")


def get_user_decision():
    """获取用户决策"""
    print("\n" + "-"*40)
    print("请选择操作:")
    print("  y - 批准（approve）")
    print("  n - 拒绝（reject）")
    print("  e - 修改（edit）")
    print("-"*40)
    
    while True:
        decision = input("你的选择 (y/n/e): ").strip().lower()
        if decision in ['y', 'n', 'e']:
            return decision
        print("❌ 无效输入，请输入 y、n 或 e")


if __name__ == "__main__":
    print("="*80)
    print("个人助理 Supervisor 模式 - 带人机交互（HITL）示例")
    print("="*80)
    
    # 配置会话
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    # 用户请求
    user_request = (
        "安排明天下午2点与设计团队的会议，1小时，"
        "并发送邮件提醒他们准备设计稿。"
    )
    
    print(f"\n用户请求: {user_request}")
    print("\n" + "="*80 + "\n")
    
    # 第一步：开始执行，收集中断
    interrupts = []
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": user_request}]},
        config,
    ):
        for update in step.values():
            if isinstance(update, dict):
                for message in update.get("messages", []):
                    if hasattr(message, 'content') and message.content:
                        print(f"🤖 助理: {message.content}")
            else:
                # 检测到中断
                interrupt = update[0]
                interrupts.append(interrupt)
                print_interrupt_info(interrupt)
    
    # 第二步：如果有中断，处理审批
    if interrupts:
        print("\n" + "="*80)
        print(f"检测到 {len(interrupts)} 个待审批操作")
        print("="*80)
        
        resume = {}
        for i, interrupt in enumerate(interrupts, 1):
            print(f"\n【审批 {i}/{len(interrupts)}】")
            request = interrupt.value["action_requests"][0]
            
            # 显示详细信息
            print(f"Request keys: {request.keys()}")
            if 'description' in request:
                print(f"描述: {request['description']}")
            if 'tool' in request:
                print(f"工具: {request['tool']}")
            if 'args' in request:
                for key, value in request['args'].items():
                    print(f"  {key}: {value}")
            elif 'arguments' in request:
                for key, value in request['arguments'].items():
                    print(f"  {key}: {value}")
            
            # 获取用户决策
            decision = get_user_decision()
            
            if decision == "y":
                # 批准
                resume[interrupt.id] = {"decisions": [{"type": "approve"}]}
                print("✅ 已批准")
            elif decision == "n":
                # 拒绝
                resume[interrupt.id] = {"decisions": [{"type": "reject"}]}
                print("❌ 已拒绝")
            elif decision == "e":
                # 修改
                print("\n修改参数（直接回车保持原值）:")
                edited_action = request.copy()
                
                # 支持 'args' 或 'arguments'
                args_key = 'args' if 'args' in edited_action else 'arguments'
                edited_args = edited_action[args_key].copy()
                
                for key, value in edited_args.items():
                    new_value = input(f"  {key} [{value}]: ").strip()
                    if new_value:
                        # 保持类型
                        if isinstance(value, list):
                            edited_args[key] = [x.strip() for x in new_value.split(',')]
                        else:
                            edited_args[key] = new_value
                
                # 使用正确的键名更新
                edited_action['arguments' if args_key == 'args' else 'arguments'] = edited_args
                resume[interrupt.id] = {
                    "decisions": [{"type": "edit", "edited_action": edited_action}]
                }
                print("✏️  已修改")
        
        # 第三步：恢复执行
        print("\n" + "="*80)
        print("恢复执行...")
        print("="*80 + "\n")
        
        for step in supervisor_agent.stream(
            Command(resume=resume),
            config,
        ):
            for update in step.values():
                if isinstance(update, dict):
                    for message in update.get("messages", []):
                        if hasattr(message, 'content') and message.content:
                            print(f"🤖 助理: {message.content}")
    
    print("\n" + "="*80)
    print("✅ 示例完成！")
    print("="*80)
    print("\n💡 提示:")
    print("- 你刚刚体验了人机交互（HITL）审批流程")
    print("- 在生产环境中，这可以防止执行危险或错误的操作")
    print("- 你可以批准、拒绝或修改 AI 代理的决策")
