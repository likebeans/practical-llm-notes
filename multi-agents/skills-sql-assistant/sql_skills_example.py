"""
Skills 模式：SQL 助理示例（渐进式披露）

演示如何使用 progressive disclosure 技术，让 agent 按需加载 skills。
"""

import os
import uuid
from typing import TypedDict, Callable
from dotenv import load_dotenv

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
load_dotenv()

# 初始化模型
model = ChatTongyi(
    model=os.environ.get("QWEN_LLM_MODEL", "qwen-plus"),
    dashscope_api_key=os.environ.get("QWEN_API_KEY", ""),
)


# ============================================================================
# 步骤 1: 定义 Skills
# ============================================================================

class Skill(TypedDict):
    """可以渐进式披露给 agent 的技能"""
    name: str         # 唯一标识符
    description: str  # 系统提示中显示的简短描述
    content: str      # 按需加载的完整内容


# 定义两个示例 skills
SKILLS = [
    {
        "name": "sales_analytics",
        "description": "销售分析数据库的模式、业务逻辑和示例查询",
        "content": """
# 销售分析技能

## 数据库模式

```sql
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    country TEXT,
    join_date DATE,
    lifetime_value DECIMAL(10,2)
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    total DECIMAL(10,2) NOT NULL,
    status TEXT CHECK(status IN ('pending', 'completed', 'cancelled'))
);

CREATE TABLE order_items (
    item_id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL
);
```

## 业务规则

1. **销售额计算**: 
   - 总销售额 = SUM(total) WHERE status = 'completed'
   - 只计算已完成的订单

2. **客户分类**:
   - 高价值客户: lifetime_value > $10,000
   - 活跃客户: 最近 90 天内有订单
   - 流失客户: 超过 180 天没有订单

3. **订单状态**:
   - pending: 待处理
   - completed: 已完成（计入销售额）
   - cancelled: 已取消（不计入销售额）

## 常用查询模式

### 查询销售额最高的客户
```sql
SELECT 
    c.customer_id,
    c.name,
    SUM(o.total) as total_revenue,
    COUNT(o.order_id) as order_count
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
GROUP BY c.customer_id, c.name
ORDER BY total_revenue DESC
LIMIT 10;
```

### 查询特定时间段的销售额
```sql
SELECT 
    DATE(o.order_date) as date,
    SUM(o.total) as daily_revenue,
    COUNT(DISTINCT o.customer_id) as unique_customers
FROM orders o
WHERE o.status = 'completed'
    AND o.order_date >= DATE('now', '-30 days')
GROUP BY DATE(o.order_date)
ORDER BY date DESC;
```

### 查询不活跃客户
```sql
SELECT 
    c.customer_id,
    c.name,
    c.email,
    MAX(o.order_date) as last_order_date,
    JULIANDAY('now') - JULIANDAY(MAX(o.order_date)) as days_since_order
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.name, c.email
HAVING MAX(o.order_date) < DATE('now', '-180 days')
    OR MAX(o.order_date) IS NULL
ORDER BY last_order_date DESC;
```

## 注意事项

- 始终过滤 `status = 'completed'` 以确保只计算完成的订单
- 使用日期函数时注意 SQLite 的语法
- 考虑添加 LIMIT 子句以避免返回过多数据
        """
    },
    {
        "name": "inventory_management",
        "description": "库存管理数据库的模式、补货逻辑和监控查询",
        "content": """
# 库存管理技能

## 数据库模式

```sql
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    unit_cost DECIMAL(10,2) NOT NULL,
    reorder_point INTEGER NOT NULL,
    discontinued BOOLEAN DEFAULT 0
);

CREATE TABLE inventory (
    inventory_id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    warehouse_location TEXT NOT NULL,
    quantity_on_hand INTEGER NOT NULL,
    last_updated DATE DEFAULT CURRENT_DATE
);

CREATE TABLE suppliers (
    supplier_id INTEGER PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    product_id INTEGER REFERENCES products(product_id),
    lead_time_days INTEGER,
    minimum_order_quantity INTEGER
);
```

## 业务规则

1. **库存状态判断**:
   - 库存充足: quantity_on_hand > reorder_point * 1.5
   - 需要关注: reorder_point < quantity_on_hand <= reorder_point * 1.5
   - 库存不足: quantity_on_hand <= reorder_point
   - 缺货: quantity_on_hand = 0

2. **补货计算**:
   - 需补货量 = reorder_point - current_stock + safety_stock
   - 安全库存 = reorder_point * 0.5
   - 考虑供应商的最小订货量

3. **产品状态**:
   - 只监控 discontinued = 0 的产品
   - 已停产产品不需要补货

## 常用查询模式

### 查询需要补货的产品
```sql
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    SUM(i.quantity_on_hand) as total_stock,
    p.reorder_point,
    p.reorder_point - SUM(i.quantity_on_hand) + 
        CAST(p.reorder_point * 0.5 AS INTEGER) as units_to_reorder,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand) + 
        CAST(p.reorder_point * 0.5 AS INTEGER)) * p.unit_cost as reorder_cost
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = 0
GROUP BY p.product_id, p.product_name, p.category, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;
```

### 查询各仓库的库存状态
```sql
SELECT 
    i.warehouse_location,
    COUNT(DISTINCT p.product_id) as product_count,
    SUM(i.quantity_on_hand) as total_units,
    SUM(i.quantity_on_hand * p.unit_cost) as total_value,
    SUM(CASE 
        WHEN i.quantity_on_hand <= p.reorder_point THEN 1 
        ELSE 0 
    END) as low_stock_products
FROM inventory i
JOIN products p ON i.product_id = p.product_id
WHERE p.discontinued = 0
GROUP BY i.warehouse_location
ORDER BY total_value DESC;
```

### 查询缺货产品
```sql
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    s.supplier_name,
    s.lead_time_days,
    s.minimum_order_quantity,
    p.unit_cost * s.minimum_order_quantity as minimum_order_cost
FROM products p
LEFT JOIN inventory i ON p.product_id = i.product_id
LEFT JOIN suppliers s ON p.product_id = s.product_id
WHERE p.discontinued = 0
GROUP BY p.product_id
HAVING COALESCE(SUM(i.quantity_on_hand), 0) = 0
ORDER BY s.lead_time_days ASC;
```

## 注意事项

- 库存查询要跨所有仓库聚合 (使用 SUM 和 GROUP BY)
- 只监控未停产的产品 (discontinued = 0)
- 补货量要考虑安全库存
- 关注供应商的交付时间和最小订货量
        """
    }
]


# ============================================================================
# 步骤 2: 创建 Skill 加载工具
# ============================================================================

@tool
def load_skill(skill_name: str) -> str:
    """加载 skill 的完整内容到 agent 的上下文中。
    
    当你需要关于如何处理特定类型请求的详细信息时使用此工具。
    这将为你提供该技能领域的全面指令、数据库模式、业务规则和示例查询。
    
    Args:
        skill_name: 要加载的 skill 名称
                   可用值: "sales_analytics", "inventory_management"
    
    Returns:
        完整的 skill 内容，包括数据库模式、业务规则和示例查询
    """
    # 查找并返回请求的 skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"✅ 已成功加载技能: {skill_name}\n\n{skill['content']}"
    
    # Skill 未找到
    available = ", ".join(s["name"] for s in SKILLS)
    return f"❌ 技能 '{skill_name}' 未找到。\n\n可用技能: {available}"


# ============================================================================
# 步骤 3: 构建 Skill 中间件
# ============================================================================

class SkillMiddleware(AgentMiddleware):
    """将 skill 描述注入系统提示的中间件"""
    
    # 注册 load_skill 工具为类变量
    tools = [load_skill]
    
    def __init__(self):
        """从 SKILLS 列表生成 skills 提示"""
        # 构建 skills 列表
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)
    
    def wrap_model_call(
        self,
        request,
        handler: Callable,
    ):
        """同步：注入 skill 描述到系统提示"""
        # 构建 skills 附录
        skills_addendum = (
            f"\n\n## 可用技能 (Skills)\n\n"
            f"{self.skills_prompt}\n\n"
            "**重要提示**: 当你需要编写 SQL 查询时，首先使用 `load_skill` 工具加载相关技能。\n"
            "这将为你提供完整的数据库模式、业务规则和示例查询，确保你的查询准确无误。"
        )
        
        # 附加到系统消息内容
        current_content = request.system_message.content
        if isinstance(current_content, str):
            new_content = current_content + skills_addendum
        elif isinstance(current_content, list):
            new_content = list(current_content) + [
                {"type": "text", "text": skills_addendum}
            ]
        else:
            new_content = skills_addendum
        
        # 创建新的系统消息
        new_system_message = SystemMessage(content=new_content)
        
        # 使用 override 方法修改请求
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


# ============================================================================
# 步骤 4: 创建带 Skill 支持的 Agent
# ============================================================================

def create_sql_assistant():
    """创建带 skill 支持的 SQL 助理 agent"""
    
    agent = create_agent(
        model,
        system_prompt="""
你是一个专业的 SQL 查询助手，帮助用户编写准确的数据库查询。

**工作流程**:
1. 理解用户的查询需求
2. 确定需要哪个技能（sales_analytics 或 inventory_management）
3. 使用 load_skill 工具加载相关技能的完整内容
4. 根据加载的数据库模式和业务规则编写 SQL 查询
5. 解释查询的逻辑和预期结果

**注意事项**:
- 始终先加载相关技能，再编写查询
- 遵循业务规则（如只计算已完成的订单）
- 使用提供的示例查询作为参考
- 添加适当的注释说明查询逻辑
        """.strip(),
        middleware=[SkillMiddleware()],
        checkpointer=MemorySaver(),
    )
    
    return agent


# ============================================================================
# 步骤 5: 测试渐进式披露
# ============================================================================

def test_progressive_disclosure():
    """测试 progressive disclosure 功能"""
    
    print("=" * 80)
    print("Skills 模式：SQL 助理示例（渐进式披露）")
    print("=" * 80)
    
    # 创建 agent
    agent = create_sql_assistant()
    
    # 测试场景
    test_cases = [
        {
            "name": "场景 1: 销售分析查询",
            "query": "查询上个月订单金额超过 $1000 的所有客户，包括他们的订单数量"
        },
        {
            "name": "场景 2: 库存管理查询",
            "query": "查询所有需要补货的产品，并计算补货成本"
        },
        {
            "name": "场景 3: 混合查询",
            "query": "查询销售额最高的前5个产品，同时显示它们的库存状态"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"{test_case['name']}")
        print("=" * 80)
        print(f"\n📝 用户查询: {test_case['query']}\n")
        
        # 创建新的对话线程
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 流式输出 agent 的响应
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": test_case["query"]}]},
                config,
                stream_mode="updates"
            ):
                for node_name, update in chunk.items():
                    if "messages" in update:
                        for message in update["messages"]:
                            # 打印 AI 消息
                            if hasattr(message, "content") and message.content:
                                if message.type == "ai":
                                    print(f"🤖 Agent: {message.content}\n")
                            # 打印工具调用
                            elif hasattr(message, "tool_calls") and message.tool_calls:
                                for tool_call in message.tool_calls:
                                    print(f"🔧 调用工具: {tool_call['name']}")
                                    print(f"   参数: {tool_call['args']}\n")
                            # 打印工具响应
                            elif message.type == "tool":
                                # 只显示前 200 个字符
                                content_preview = message.content[:200]
                                if len(message.content) > 200:
                                    content_preview += "..."
                                print(f"📦 工具响应: {content_preview}\n")
        
        except Exception as e:
            print(f"❌ 错误: {e}\n")
        
        if i < len(test_cases):
            input("\n⏸️  按 Enter 继续下一个场景...")
    
    print("\n" + "=" * 80)
    print("✅ 所有示例完成！")
    print("=" * 80)
    
    print("\n💡 关键观察:")
    print("1. Agent 首先看到系统提示中的简短 skill 描述")
    print("2. Agent 根据查询需求决定需要哪个 skill")
    print("3. Agent 调用 load_skill 工具按需加载完整内容")
    print("4. 只有相关的 skill 被加载，而非所有 skills")
    print("5. 这就是 Progressive Disclosure（渐进式披露）的核心")
    print("\n📈 优势:")
    print("- 减少上下文使用：只加载需要的 skills")
    print("- 可扩展：可以添加数十个 skills 而不压垮上下文")
    print("- 团队自治：不同团队可以独立维护各自的 skills")


# ============================================================================
# 示例 6: 直接查看系统提示（调试用）
# ============================================================================

def show_system_prompt():
    """显示注入 skill 描述后的系统提示"""
    
    print("=" * 80)
    print("查看系统提示（包含 Skill 描述）")
    print("=" * 80)
    
    # 创建中间件实例
    middleware = SkillMiddleware()
    
    print("\n📋 注入的 Skills 附录:\n")
    print("-" * 80)
    
    skills_addendum = (
        f"\n\n## 可用技能 (Skills)\n\n"
        f"{middleware.skills_prompt}\n\n"
        "**重要提示**: 当你需要编写 SQL 查询时，首先使用 `load_skill` 工具加载相关技能。\n"
        "这将为你提供完整的数据库模式、业务规则和示例查询，确保你的查询准确无误。"
    )
    
    print(skills_addendum)
    print("-" * 80)
    
    print("\n💡 说明:")
    print("- Agent 在系统提示中只能看到简短的 skill 描述")
    print("- 完整的数据库模式、业务规则和示例查询需要通过 load_skill 工具加载")
    print("- 这样可以避免将所有信息一次性塞入上下文")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print("\n" + "=" * 80)
    print("Skills 模式：渐进式披露 (Progressive Disclosure)")
    print("=" * 80)
    
    print("\n选择运行模式:")
    print("1. 运行完整测试（推荐）")
    print("2. 仅查看系统提示")
    
    choice = input("\n请选择 (1 或 2): ").strip()
    
    if choice == "2":
        show_system_prompt()
    else:
        test_progressive_disclosure()


if __name__ == "__main__":
    main()
