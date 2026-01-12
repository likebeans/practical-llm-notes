"""
SQL Agent - 智能 SQL 查询助手示例
使用 LangGraph 构建的自然语言到 SQL 的智能转换系统

功能特点:
1. 自然语言转 SQL 查询
2. 自动执行 SQL 并获取结果
3. 错误检测和 SQL 自动修复
4. 查询结果的自然语言解释
5. 安全防护（只允许 SELECT 查询）
"""

import os
import getpass
import sqlite3
from typing import TypedDict, Optional, Literal

# LangChain 核心组件
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase

# LangGraph 图构建
from langgraph.graph import StateGraph, START, END

# ============================================================================
# 1. 状态定义
# ============================================================================

class SQLAgentState(TypedDict):
    """SQL Agent 状态定义"""
    messages: list                    # 消息历史
    sql_query: Optional[str]          # 生成的 SQL 查询
    query_result: Optional[str]       # 查询执行结果
    error_message: Optional[str]      # 错误信息
    retry_count: int                  # 重试次数


# ============================================================================
# 2. 环境配置
# ============================================================================

def setup_environment():
    """设置 API 密钥"""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("请输入 OPENAI_API_KEY: ")


# ============================================================================
# 3. 数据库初始化
# ============================================================================

def create_sample_database():
    """
    创建示例数据库
    
    包含三个表:
    - customers: 客户信息
    - products: 产品信息
    - orders: 订单信息
    """
    print("📊 正在创建示例数据库...")
    
    # 连接数据库（如果不存在会自动创建）
    conn = sqlite3.connect("sample_store.db")
    cursor = conn.cursor()
    
    # 创建 customers 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            city TEXT
        )
    """)
    
    # 创建 products 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL NOT NULL,
            stock INTEGER DEFAULT 0
        )
    """)
    
    # 创建 orders 表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            order_date TEXT,
            total_amount REAL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)
    
    # 插入示例数据（如果表为空）
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        # 插入客户数据
        customers = [
            (1, "张三", "zhangsan@email.com", "北京"),
            (2, "李四", "lisi@email.com", "上海"),
            (3, "王五", "wangwu@email.com", "广州"),
            (4, "赵六", "zhaoliu@email.com", "深圳"),
        ]
        cursor.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)", 
            customers
        )
        
        # 插入产品数据
        products = [
            (1, "笔记本电脑", "电子产品", 5999.00, 50),
            (2, "无线鼠标", "电子产品", 99.00, 200),
            (3, "机械键盘", "电子产品", 399.00, 100),
            (4, "显示器", "电子产品", 1299.00, 80),
            (5, "办公椅", "家具", 899.00, 30),
        ]
        cursor.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?)", 
            products
        )
        
        # 插入订单数据
        orders = [
            (1, 1, 1, 2, "2024-01-15", 11998.00),
            (2, 2, 2, 3, "2024-01-16", 297.00),
            (3, 1, 3, 1, "2024-01-17", 399.00),
            (4, 3, 4, 2, "2024-01-18", 2598.00),
            (5, 2, 5, 1, "2024-01-19", 899.00),
            (6, 4, 1, 1, "2024-01-20", 5999.00),
        ]
        cursor.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", 
            orders
        )
    
    conn.commit()
    conn.close()
    
    print("✅ 示例数据库创建完成: sample_store.db")
    return "sqlite:///sample_store.db"


def get_database_schema(db_uri: str) -> str:
    """获取数据库架构信息"""
    db = SQLDatabase.from_uri(db_uri)
    return db.get_table_info()


# ============================================================================
# 4. 初始化模型
# ============================================================================

llm = None

def initialize_model():
    """初始化 LLM 模型"""
    global llm
    llm = init_chat_model("gpt-4o", temperature=0)
    print("✅ 模型初始化完成")


# ============================================================================
# 5. 节点函数
# ============================================================================

def generate_sql(state: SQLAgentState) -> dict:
    """
    节点 1: 根据用户问题生成 SQL 查询
    
    功能:
    - 分析用户的自然语言问题
    - 结合数据库架构生成合适的 SQL
    - 只生成 SELECT 查询以确保安全
    """
    print("\n🤖 [generate_sql] 正在生成 SQL 查询...")
    
    # 获取用户问题
    user_question = state["messages"][-1].content
    
    # 获取数据库架构
    schema = get_database_schema("sqlite:///sample_store.db")
    
    # 构建提示词
    prompt = f"""你是一个 SQL 查询生成专家。
    
数据库架构信息:
{schema}

用户问题: {user_question}

请生成一个 SQL 查询来回答这个问题。要求:
1. 只生成 SELECT 查询（不允许 INSERT、UPDATE、DELETE 等操作）
2. 确保 SQL 语法正确
3. 只返回 SQL 语句，不要包含任何解释或标记
4. 如果需要，添加适当的 LIMIT 子句

SQL 查询:
"""
    
    # 调用 LLM 生成 SQL
    response = llm.invoke([HumanMessage(content=prompt)])
    sql = response.content.strip()
    
    # 清理 SQL（移除可能的代码块标记）
    sql = sql.replace("```sql", "").replace("```", "").strip()
    
    print(f"   ✓ 生成的 SQL: {sql}")
    
    return {
        "sql_query": sql,
        "error_message": None
    }


def execute_sql(state: SQLAgentState) -> dict:
    """
    节点 2: 执行 SQL 查询
    
    功能:
    - 安全检查 SQL 语句
    - 连接数据库并执行查询
    - 返回查询结果或错误信息
    """
    print("\n⚡ [execute_sql] 正在执行 SQL 查询...")
    
    sql = state["sql_query"]
    
    # 安全检查：只允许 SELECT 查询
    if not sql.strip().upper().startswith("SELECT"):
        print("   ✗ 安全检查失败: 只允许 SELECT 查询")
        return {
            "query_result": None,
            "error_message": "安全限制: 只允许 SELECT 查询"
        }
    
    try:
        # 连接数据库
        conn = sqlite3.connect("sample_store.db")
        cursor = conn.cursor()
        
        # 执行查询
        cursor.execute(sql)
        
        # 获取列名
        column_names = [description[0] for description in cursor.description]
        
        # 获取结果（最多 100 行）
        rows = cursor.fetchmany(100)
        
        conn.close()
        
        # 格式化结果
        if rows:
            result_str = f"列名: {', '.join(column_names)}\n\n"
            for i, row in enumerate(rows, 1):
                result_str += f"行 {i}: {row}\n"
            
            print(f"   ✓ 查询成功，返回 {len(rows)} 行数据")
            
            return {
                "query_result": result_str,
                "error_message": None
            }
        else:
            print("   ✓ 查询成功，但没有返回数据")
            return {
                "query_result": "查询执行成功，但没有匹配的数据。",
                "error_message": None
            }
    
    except Exception as e:
        error_msg = str(e)
        print(f"   ✗ 查询失败: {error_msg}")
        
        return {
            "query_result": None,
            "error_message": error_msg
        }


def check_result(state: SQLAgentState) -> Literal["generate_response", "fix_sql", "error_response"]:
    """
    条件边: 检查查询结果并决定下一步
    
    返回:
    - "generate_response": 查询成功，生成自然语言回复
    - "fix_sql": 查询失败但可以重试，修复 SQL
    - "error_response": 查询失败且无法重试，返回错误信息
    """
    print("\n📊 [check_result] 正在检查查询结果...")
    
    # 如果有错误
    if state["error_message"]:
        # 检查是否可以重试
        if state["retry_count"] < 2:  # 最多重试 2 次
            print(f"   → 决策: 尝试修复 SQL (第 {state['retry_count'] + 1} 次重试)")
            return "fix_sql"
        else:
            print("   → 决策: 超过最大重试次数，返回错误")
            return "error_response"
    
    # 如果查询成功
    if state["query_result"]:
        print("   → 决策: 查询成功，生成回复")
        return "generate_response"
    
    print("   → 决策: 未知错误")
    return "error_response"


def fix_sql(state: SQLAgentState) -> dict:
    """
    节点 3: 修复错误的 SQL 查询
    
    功能:
    - 分析错误信息
    - 重新生成正确的 SQL
    - 增加重试计数
    """
    print("\n🔧 [fix_sql] 正在修复 SQL 查询...")
    
    original_sql = state["sql_query"]
    error = state["error_message"]
    question = state["messages"][-1].content
    
    # 获取数据库架构
    schema = get_database_schema("sqlite:///sample_store.db")
    
    # 构建提示词
    prompt = f"""你是一个 SQL 调试专家。

数据库架构:
{schema}

用户问题: {question}

之前生成的 SQL（有错误）: {original_sql}

错误信息: {error}

请分析错误原因并生成正确的 SQL 查询。要求:
1. 只生成 SELECT 查询
2. 修复语法错误
3. 确保表名和列名正确
4. 只返回修正后的 SQL 语句

修正后的 SQL:
"""
    
    # 调用 LLM 修复 SQL
    response = llm.invoke([HumanMessage(content=prompt)])
    fixed_sql = response.content.strip()
    
    # 清理 SQL
    fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
    
    print(f"   ✓ 修复后的 SQL: {fixed_sql}")
    
    return {
        "sql_query": fixed_sql,
        "retry_count": state["retry_count"] + 1,
        "error_message": None
    }


def generate_response(state: SQLAgentState) -> dict:
    """
    节点 4: 生成自然语言回复
    
    功能:
    - 将查询结果转换为用户友好的自然语言
    - 总结和解释数据
    """
    print("\n💡 [generate_response] 正在生成自然语言回复...")
    
    question = state["messages"][-1].content
    sql = state["sql_query"]
    result = state["query_result"]
    
    # 构建提示词
    prompt = f"""你是一个数据分析助手。

用户问题: {question}

执行的 SQL 查询: {sql}

查询结果:
{result}

请用自然语言总结查询结果，回答用户的问题。要求:
1. 语言简洁清晰
2. 突出关键数据
3. 如果有多行数据，提供简要的总结
4. 使用中文回答

回答:
"""
    
    # 调用 LLM 生成回复
    response = llm.invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()
    
    print("   ✓ 回复已生成")
    
    # 添加到消息历史
    return {
        "messages": state["messages"] + [AIMessage(content=answer)]
    }


def generate_error_response(state: SQLAgentState) -> dict:
    """
    节点 5: 生成错误回复
    
    功能:
    - 向用户解释为什么无法完成查询
    - 提供建议
    """
    print("\n❌ [generate_error_response] 正在生成错误回复...")
    
    error = state["error_message"]
    
    error_response = f"""抱歉，无法完成您的查询。

错误信息: {error}

建议:
1. 请确保您的问题描述清楚
2. 尝试用不同的方式表述您的问题
3. 检查您要查询的数据是否存在于数据库中

如需帮助，您可以问:
- "有哪些客户？"
- "产品的平均价格是多少？"
- "显示所有订单信息"
"""
    
    return {
        "messages": state["messages"] + [AIMessage(content=error_response)]
    }


# ============================================================================
# 6. 构建 LangGraph 图
# ============================================================================

def build_sql_agent():
    """
    构建 SQL Agent 图
    
    图结构:
    START → generate_sql → execute_sql → check_result
                                            ├→ 成功 → generate_response → END
                                            ├→ 失败(可重试) → fix_sql → execute_sql
                                            └→ 失败(不可重试) → error_response → END
    """
    print("\n🔨 正在构建 SQL Agent 图...")
    
    # 初始化模型
    initialize_model()
    
    # 创建状态图
    workflow = StateGraph(SQLAgentState)
    
    # 添加节点
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("fix_sql", fix_sql)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("error_response", generate_error_response)
    
    # 添加边
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    
    # 条件边: 检查查询结果
    workflow.add_conditional_edges(
        "execute_sql",
        check_result,
        {
            "generate_response": "generate_response",
            "fix_sql": "fix_sql",
            "error_response": "error_response"
        }
    )
    
    # 修复后重新执行
    workflow.add_edge("fix_sql", "execute_sql")
    
    # 结束边
    workflow.add_edge("generate_response", END)
    workflow.add_edge("error_response", END)
    
    # 编译图
    graph = workflow.compile()
    
    print("✅ SQL Agent 图构建完成")
    
    return graph


# ============================================================================
# 7. 运行示例
# ============================================================================

def run_example(graph):
    """运行示例查询"""
    print("\n" + "="*70)
    print("🚀 开始运行 SQL Agent 示例")
    print("="*70)
    
    # 示例问题
    questions = [
        "有多少个客户？",
        "价格最高的产品是什么？",
        "显示所有来自北京的客户",
        "统计每个产品类别的产品数量",
        "订单总金额最高的客户是谁？",
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"❓ 问题 {i}: {question}")
        print(f"{'='*70}")
        
        # 运行图
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "sql_query": None,
            "query_result": None,
            "error_message": None,
            "retry_count": 0
        })
        
        # 获取最终回复
        final_answer = result["messages"][-1].content
        
        print(f"\n✨ 最终答案:")
        print(f"   {final_answer}")
        print()


# ============================================================================
# 8. 交互式查询
# ============================================================================

def interactive_mode(graph):
    """交互式查询模式"""
    print("\n" + "="*70)
    print("💬 进入交互式查询模式")
    print("="*70)
    print("输入您的问题（输入 'exit' 或 'quit' 退出）")
    print("示例: 有多少个客户？")
    print("="*70 + "\n")
    
    while True:
        # 获取用户输入
        question = input("您的问题 > ").strip()
        
        # 检查退出命令
        if question.lower() in ['exit', 'quit', '退出']:
            print("\n👋 再见!")
            break
        
        if not question:
            continue
        
        print()
        
        # 运行查询
        result = graph.invoke({
            "messages": [HumanMessage(content=question)],
            "sql_query": None,
            "query_result": None,
            "error_message": None,
            "retry_count": 0
        })
        
        # 显示结果
        final_answer = result["messages"][-1].content
        print(f"✨ 答案: {final_answer}\n")


# ============================================================================
# 9. 主函数
# ============================================================================

def main():
    """主函数"""
    print("="*70)
    print("  SQL Agent - 智能 SQL 查询助手示例")
    print("="*70)
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 创建示例数据库
    db_uri = create_sample_database()
    
    # 3. 构建 Agent 图
    graph = build_sql_agent()
    
    # 4. 运行示例查询
    run_example(graph)
    
    # 5. 进入交互式模式（可选）
    print("\n" + "="*70)
    print("是否进入交互式查询模式？(y/n)")
    choice = input("> ").strip().lower()
    
    if choice == 'y':
        interactive_mode(graph)
    
    print("\n" + "="*70)
    print("✅ 程序结束")
    print("="*70)


if __name__ == "__main__":
    main()
