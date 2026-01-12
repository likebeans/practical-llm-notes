"""
Skills 模式：SQL 助理示例

演示多个代理共享相同的 SQL 技能，但有不同的专门化任务。
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent

# 加载环境变量
load_dotenv()

# 初始化 Qwen LLM
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY")
)

# ============================================================================
# 步骤 1: 创建共享 SQL 技能
# ============================================================================

# 连接到 Chinook 示例数据库（音乐商店数据库）
db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# 创建共享的 SQL 工具集
toolkit = SQLDatabaseToolkit(db=db, llm=model)
shared_sql_skills = toolkit.get_tools()

print("="*80)
print("共享 SQL 技能")
print("="*80)
print(f"\n可用技能: {[tool.name for tool in shared_sql_skills]}")
print(f"数据库: {db.dialect}")
print(f"可用表: {db.get_usable_table_names()}")

# ============================================================================
# 步骤 2: 创建专门化的代理（使用相同的共享技能）
# ============================================================================

# 分析代理：专注数据分析
analysis_agent = create_agent(
    model,
    tools=shared_sql_skills,
    system_prompt=f"""
    你是一个数据分析专家，专门分析 {db.dialect} 数据库。
    数据库包含这些表: {', '.join(db.get_usable_table_names())}
    
    分析步骤:
    1. 使用 sql_db_list_tables 查看可用表
    2. 使用 sql_db_schema 了解表结构
    3. 使用 sql_db_query_checker 验证 SQL 正确性
    4. 使用 sql_db_query 执行查询分析数据
    5. 解释分析结果，提供洞察
    
    重点: 深入分析数据背后的含义，提供有价值的洞察。
    限制: 查询结果不超过 5 条。
    """
)

# 报告代理：专注生成报表
report_agent = create_agent(
    model,
    tools=shared_sql_skills,
    system_prompt=f"""
    你是一个报表生成专家，专门生成 {db.dialect} 数据库的报表。
    数据库包含这些表: {', '.join(db.get_usable_table_names())}
    
    报表要求:
    1. 查询数据并生成格式化的 Markdown 表格
    2. 包含清晰的标题和摘要
    3. 数据要易于阅读和理解
    4. 添加必要的说明和注释
    
    重点: 让报表专业、清晰、易于理解。
    限制: 查询结果不超过 10 条。
    """
)

# 监控代理：专注异常检测
monitor_agent = create_agent(
    model,
    tools=shared_sql_skills,
    system_prompt=f"""
    你是一个数据监控专家，专门监控 {db.dialect} 数据库。
    数据库包含这些表: {', '.join(db.get_usable_table_names())}
    
    监控内容:
    1. 检查数据完整性
    2. 识别异常值或模式
    3. 发现潜在的数据质量问题
    4. 提供预警和建议
    
    重点: 快速识别问题，提供可操作的预警。
    限制: 查询结果不超过 5 条。
    """
)

# ============================================================================
# 步骤 3: 使用不同的代理完成不同任务
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Skills 模式：SQL 助理示例")
    print("="*80)
    
    # 测试任务
    tasks = [
        ("分析代理", analysis_agent, "分析哪个国家的客户最多"),
        ("报告代理", report_agent, "生成销售额前5的专辑报表"),
        ("监控代理", monitor_agent, "检查是否有异常的发票数据"),
    ]
    
    for agent_name, agent, query in tasks:
        print(f"\n{'='*80}")
        print(f"【{agent_name}】")
        print("-"*80)
        print(f"任务: {query}\n")
        
        try:
            for step in agent.stream(
                {"messages": [{"role": "user", "content": query}]}
            ):
                for update in step.values():
                    for message in update.get("messages", []):
                        if hasattr(message, 'content') and message.content and message.type == 'ai':
                            print(f"回复: {message.content}")
                            break
        except Exception as e:
            print(f"错误: {e}")
    
    print("\n" + "="*80)
    print("✅ 示例完成！")
    print("="*80)
    print("\n💡 关键观察:")
    print("- 三个代理使用相同的 SQL 技能工具集")
    print("- 每个代理通过不同的系统提示专注不同的任务")
    print("- 分析代理关注数据洞察")
    print("- 报告代理关注格式化输出")
    print("- 监控代理关注异常检测")
    print("- 这就是 Skills 模式的核心：技能共享，任务专注")
