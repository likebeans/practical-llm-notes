"""
个人助理 Supervisor 模式示例

这个示例演示了多代理系统中的工具调用模式。
一个 supervisor 代理协调专门的子代理（日历和邮件），
这些子代理被包装为工具。
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool
from langchain.agents import create_agent

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
    return f"事件已创建: {title} 从 {start_time} 到 {end_time}，{len(attendees)} 位参与者"


@tool
def send_email(
    to: list[str],      # 邮箱地址
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """通过电子邮件 API 发送邮件。需要格式正确的邮箱地址。"""
    return f"邮件已发送给 {', '.join(to)} - 主题: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO 格式: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """检查指定日期给定参与者的日历可用性。"""
    return ["09:00", "14:00", "16:00"]


# ============================================================================
# 步骤 2: 创建专门化的子代理
# ============================================================================

# 初始化 Qwen LLM
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "你是一个日历调度助手。"
        "解析自然语言的调度请求（例如，'下周二下午2点'）为正确的 ISO 日期时间格式。"
        "需要时使用 get_available_time_slots 检查可用性。"
        "使用 create_calendar_event 安排事件。"
        "始终在最终回复中确认已安排的内容。"
    )
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "你是一个邮件助手。"
        "基于自然语言请求撰写专业邮件。"
        "提取收件人信息，生成合适的主题和正文。"
        "使用 send_email 发送消息。"
        "始终在最终回复中确认已发送的内容。"
    )
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
# 步骤 4: 创建 supervisor 代理
# ============================================================================

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "你是一个有用的个人助理。"
        "你可以安排日历事件和发送电子邮件。"
        "将用户请求分解为适当的工具调用并协调结果。"
        "当请求涉及多个操作时，按顺序使用多个工具。"
    )
)

# ============================================================================
# 步骤 5: 使用 supervisor
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("个人助理 Supervisor 模式示例")
    print("="*80)
    
    # 示例 1：简单的单领域请求
    print("\n【示例 1】简单的日历请求")
    print("-"*80)
    query1 = "明天上午9点安排团队站会"
    print(f"用户: {query1}\n")
    
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query1}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                if hasattr(message, 'content') and message.content:
                    print(f"助理: {message.content}")
    
    # 示例 2：复杂的多领域请求
    print("\n" + "="*80)
    print("\n【示例 2】复杂的多领域请求")
    print("-"*80)
    query2 = (
        "安排下周二下午2点与设计团队的会议，1小时，"
        "并发送邮件提醒他们查看新原型。"
    )
    print(f"用户: {query2}\n")
    
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query2}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                if hasattr(message, 'content') and message.content:
                    print(f"助理: {message.content}")
    
    print("\n" + "="*80)
    print("示例完成！")
    print("="*80)
