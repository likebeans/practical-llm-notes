# Skills 模式：SQL 助理系统（渐进式披露）

> 基于 LangChain 官方文档：[Build a SQL assistant with on-demand skills](https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant)

## 目录

- [什么是 Skills 模式](#什么是-skills-模式)
- [核心概念](#核心概念)
- [渐进式披露](#渐进式披露)
- [工作流程](#工作流程)
- [实现步骤](#实现步骤)
- [使用场景](#使用场景)
- [最佳实践](#最佳实践)

## 什么是 Skills 模式

**Skills（技能）模式**是一种使用 **Progressive Disclosure（渐进式披露）** 的上下文管理技术。Agent 通过工具调用**按需加载**信息，而不是在系统提示中预先加载所有内容。

### 关键特点

- 📦 **按需加载**：只加载当前任务需要的 skills
- 🎯 **减少上下文**：避免将所有信息塞入系统提示
- 🔧 **Prompt-based**：Skills 主要是专门化的提示指令
- 🏢 **团队自治**：不同团队可以独立开发和维护自己的 skills

### 什么是 Skills

如 Claude Code 所示，**Skills 是自包含的、基于提示的专门化指令单元**：

- 主要通过**提示**引导行为
- 可以包含工具使用说明
- 可以包含示例代码（对编码 agent）
- 存储为文件系统中的目录或数据库中的记录

> **注意**：Skills 不是工具本身，而是关于如何处理特定业务任务的**指令和知识**。

## 核心概念

### 1. Progressive Disclosure（渐进式披露）

渐进式披露由 Anthropic 推广，是一种三层架构：

```
元数据 → 核心内容 → 详细资源
```

Agent 只在需要时加载信息：

1. **系统提示**：显示 skill 的简短描述（元数据）
2. **工具调用**：加载完整的 skill 内容（核心内容）
3. **进一步工具**：访问详细资源（可选）

### 2. Skill 结构

每个 skill 包含三个部分：

```python
class Skill(TypedDict):
    name: str         # 唯一标识符
    description: str  # 1-2 句话描述（显示在系统提示中）
    content: str      # 完整内容（按需加载）
```

**示例**：

```python
{
    "name": "sales_analytics",
    "description": "销售分析技能，提供销售数据库模式和业务逻辑",
    "content": """
    # 销售分析技能
    
    ## 数据库模式
    - customers 表: customer_id, name, email, country
    - orders 表: order_id, customer_id, order_date, total
    - order_items 表: item_id, order_id, product_id, quantity
    
    ## 业务规则
    - 销售额 = SUM(quantity * unit_price)
    - 高价值客户: 总消费 > $10,000
    
    ## 示例查询
    SELECT customer_id, SUM(total) as revenue
    FROM orders
    GROUP BY customer_id
    ORDER BY revenue DESC;
    """
}
```

### 3. 工作流程

```
用户请求: "查询上个月销售额最高的客户"
        ↓
系统提示显示:
  - sales_analytics: 销售分析技能
  - inventory_management: 库存管理技能
        ↓
Agent 决定: "我需要 sales_analytics"
        ↓
调用工具: load_skill("sales_analytics")
        ↓
加载完整内容:
  - 数据库模式
  - 业务规则
  - 示例查询
        ↓
Agent 生成 SQL 查询
```

## 渐进式披露

### 为什么使用渐进式披露

**场景**：在大型企业中，你可能有：
- 数百个业务垂直领域
- 每个领域有独立的数据库或数千张表
- 无法将所有模式放入上下文窗口

**解决方案**：
1. 系统提示：显示轻量级的 skill 描述
2. 按需加载：只加载与查询相关的 skill
3. 独立维护：产品负责人独立维护各自领域的 skills

### 优势

- ✅ **减少上下文使用**：只加载 2-3 个需要的 skills，而非所有
- ✅ **团队自治**：不同团队独立开发专门化的 skills
- ✅ **高效扩展**：添加数十或数百个 skills 而不压垮上下文
- ✅ **简化对话历史**：单个 agent，一个对话线程

### 权衡

- ⚠️ **延迟**：按需加载需要额外的工具调用
- ⚠️ **工作流控制**：基础实现依赖提示引导，无法强制硬约束

### Skills vs RAG

Skills with progressive disclosure 可以看作是 **RAG 的一种形式**：
- 每个 skill 是一个检索单元
- 不一定需要嵌入或关键词搜索
- 可以通过直接查找、文件操作或 API 调用检索

## 实现步骤

### 1. 定义 Skills

```python
from typing import TypedDict

class Skill(TypedDict):
    """可以渐进式披露给 agent 的技能"""
    name: str         # 唯一标识符
    description: str  # 系统提示中显示的简短描述
    content: str      # 按需加载的完整内容

# 定义 skills
SKILLS = [
    {
        "name": "sales_analytics",
        "description": "销售分析数据库的模式和业务逻辑",
        "content": """
        # 销售分析技能
        
        ## 数据库模式
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            country TEXT,
            join_date DATE
        );
        
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers,
            order_date DATE NOT NULL,
            total DECIMAL(10,2),
            status TEXT
        );
        
        ## 业务规则
        - 销售额 = SUM(total) WHERE status = 'completed'
        - 高价值客户: 总消费 > $10,000
        - 活跃客户: 最近 90 天内有订单
        
        ## 常用查询模式
        -- 查询销售额前 10 的客户
        SELECT c.name, SUM(o.total) as revenue
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        WHERE o.status = 'completed'
        GROUP BY c.customer_id, c.name
        ORDER BY revenue DESC
        LIMIT 10;
        """
    },
    {
        "name": "inventory_management",
        "description": "库存管理数据库的模式和补货逻辑",
        "content": """
        # 库存管理技能
        
        ## 数据库模式
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            category TEXT,
            unit_cost DECIMAL(10,2),
            reorder_point INTEGER
        );
        
        CREATE TABLE inventory (
            inventory_id INTEGER PRIMARY KEY,
            product_id INTEGER REFERENCES products,
            warehouse_location TEXT,
            quantity_on_hand INTEGER,
            last_updated DATE
        );
        
        ## 业务规则
        - 库存不足: quantity_on_hand <= reorder_point
        - 需要补货量 = reorder_point - quantity_on_hand + 安全库存
        - 安全库存 = reorder_point * 0.5
        
        ## 常用查询模式
        -- 查询需要补货的产品
        SELECT p.product_name, 
               SUM(i.quantity_on_hand) as total_stock,
               p.reorder_point,
               p.reorder_point - SUM(i.quantity_on_hand) as units_to_reorder
        FROM products p
        JOIN inventory i ON p.product_id = i.product_id
        GROUP BY p.product_id
        HAVING SUM(i.quantity_on_hand) <= p.reorder_point
        ORDER BY units_to_reorder DESC;
        """
    }
]
```

### 2. 创建 Skill 加载工具

```python
from langchain.tools import tool

@tool
def load_skill(skill_name: str) -> str:
    """加载 skill 的完整内容到 agent 的上下文中。
    
    当你需要关于如何处理特定类型请求的详细信息时使用。
    这将为你提供该技能领域的全面指令、策略和指南。
    
    Args:
        skill_name: 要加载的 skill 名称（如 "sales_analytics", "inventory_management"）
    """
    # 查找并返回请求的 skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"已加载技能: {skill_name}\n\n{skill['content']}"
    
    # Skill 未找到
    available = ", ".join(s["name"] for s in SKILLS)
    return f"技能 '{skill_name}' 未找到。可用技能: {available}"
```

### 3. 构建 Skill 中间件

使用 middleware 将 skill 描述注入系统提示：

```python
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import SystemMessage
from typing import Callable

class SkillMiddleware(AgentMiddleware):
    """将 skill 描述注入系统提示的中间件"""
    
    # 注册 load_skill 工具
    tools = [load_skill]
    
    def __init__(self):
        """从 SKILLS 列表生成 skills 提示"""
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)
    
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """注入 skill 描述到系统提示"""
        # 构建 skills 附录
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "当你需要关于如何处理特定类型请求的详细信息时，"
            "使用 load_skill 工具。"
        )
        
        # 附加到系统消息内容块
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)
```

### 4. 创建带 Skill 支持的 Agent

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# 初始化模型
model = ChatOpenAI(model="gpt-4")

# 创建带 skill 支持的 agent
agent = create_agent(
    model,
    system_prompt=(
        "你是一个 SQL 查询助手，帮助用户编写业务数据库查询。"
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)
```

### 5. 测试渐进式披露

```python
import uuid

# 配置对话线程
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# 请求 SQL 查询
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "查询上个月订单超过 $1000 的所有客户"
            }
        ]
    },
    config
)

# 打印对话
for message in result["messages"]:
    if hasattr(message, 'pretty_print'):
        message.pretty_print()
```

**预期行为**：
1. Agent 看到系统提示中的 skill 描述
2. Agent 识别需要 `sales_analytics` skill
3. Agent 调用 `load_skill("sales_analytics")`
4. 完整的数据库模式和业务规则被加载
5. Agent 使用加载的信息编写 SQL 查询

### 6. 高级：使用自定义状态添加约束

使用自定义状态来追踪和约束 skill 加载：

```python
from typing import TypedDict, Annotated
from langchain.agents import AgentState

class SkillState(AgentState):
    """带 skill 追踪的自定义状态"""
    loaded_skills: Annotated[list[str], "已加载的 skills"]
    max_skills: Annotated[int, "最多加载的 skills 数量"]

def validate_skill_load(state: SkillState) -> dict:
    """验证是否可以加载更多 skills"""
    if len(state.get("loaded_skills", [])) >= state.get("max_skills", 3):
        return {
            "error": f"已达到最大 skill 数量 ({state['max_skills']})"
        }
    return {}

# 在 middleware 中使用自定义状态
class ConstrainedSkillMiddleware(SkillMiddleware):
    def wrap_tool_call(self, request, handler):
        # 在加载 skill 前检查约束
        if request.tool_name == "load_skill":
            validation = validate_skill_load(request.state)
            if "error" in validation:
                return ToolResponse(error=validation["error"])
        
        response = handler(request)
        
        # 追踪已加载的 skills
        if request.tool_name == "load_skill" and response.success:
            loaded = request.state.get("loaded_skills", [])
            loaded.append(request.tool_arguments["skill_name"])
            response.state_update = {"loaded_skills": loaded}
        
        return response
```

## 实现变体

### 存储后端

- **内存**（本教程）：Python 数据结构，快速访问
- **文件系统**（Claude Code 方法）：目录和文件，通过文件操作发现
- **远程存储**：S3、数据库、Notion 或 API

### Skill 发现

- **系统提示列表**（本教程）：在系统提示中显示 skill 描述
- **文件系统**：扫描目录发现 skills
- **注册表**：查询 skill 注册服务
- **动态查找**：通过工具调用列出可用 skills

### 披露策略

- **单次加载**（本教程）：一次工具调用加载整个 skill
- **分页**：分多页/块加载大型 skills
- **基于搜索**：在特定 skill 内搜索相关部分
- **层级**：先加载概览，再深入特定子部分

### 大小考虑

- **小 skills**（< 1K tokens）：可直接包含在系统提示中，使用提示缓存
- **中 skills**（1-10K tokens）：按需加载（本教程）
- **大 skills**（> 10K tokens）：使用分页、搜索或层级探索

## 使用场景

### 适合 Skills 模式的场景

✅ **推荐使用**：

1. **大型企业数据仓库**
   - 多个业务垂直领域
   - 数百或数千张表
   - 无法将所有模式放入上下文

2. **多团队协作**
   - 不同产品负责人维护各自领域
   - 独立开发和更新 skills
   - 无需重新部署即可添加新 skills

3. **复杂业务逻辑**
   - 每个领域有特定的业务规则
   - 需要详细的指令和示例
   - 上下文窗口有限

4. **动态 Few-shot 提示**
   - 根据查询加载相关示例
   - 结合模式和示例模式

### 典型应用

1. **企业 BI 系统**
   - 销售分析 skill
   - 财务报表 skill
   - 库存管理 skill
   - HR 分析 skill

2. **多租户 SaaS**
   - 每个客户有自定义模式
   - 按需加载客户特定的数据模型
   - 动态适应不同的数据结构

3. **合规和审计**
   - 不同法规领域的 skills
   - 加载特定法规的规则和检查
   - 审计追踪 skill 使用

### 不适合的场景

❌ **不推荐**：

- **小型数据库**：所有模式可以放入系统提示
- **单一领域**：不需要多个专门化的 skills
- **实时性要求高**：额外的工具调用会增加延迟
- **简单查询**：不需要复杂的业务逻辑

## 结合 Few-shot 提示

Progressive disclosure 可以与 few-shot 提示结合：

### 动态加载示例

```python
{
    "name": "customer_analysis",
    "description": "客户分析数据库和示例查询",
    "content": """
    ## 数据库模式
    ...
    
    ## Few-shot 示例
    
    ### 示例 1: 查找不活跃客户
    用户: "查找 6 个月没有订单的客户"
    SQL:
    SELECT c.customer_id, c.name
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id 
        AND o.order_date > DATE('now', '-6 months')
    WHERE o.order_id IS NULL;
    
    ### 示例 2: 日期范围过滤
    用户: "查询上个季度的订单"
    SQL:
    SELECT *
    FROM orders
    WHERE order_date BETWEEN 
        DATE('now', 'start of month', '-3 months') 
        AND DATE('now', 'start of month', '-1 day');
    """
}
```

### 语义搜索示例

```python
# 扩展 load_skill 工具以支持示例搜索
@tool
def load_skill_with_examples(skill_name: str, query: str) -> str:
    """加载 skill 并检索相关示例"""
    skill = get_skill(skill_name)
    
    # 使用嵌入搜索相关示例
    relevant_examples = search_examples(skill["examples"], query)
    
    return f"""
    {skill['content']}
    
    ## 相关示例
    {relevant_examples}
    """
```

## 最佳实践

### 1. Skill 设计

- **保持 skills 自包含**：每个 skill 应该独立完整
- **提供清晰的描述**：让 agent 能够决定是否需要加载
- **包含示例**：Few-shot 示例提高查询质量
- **文档业务规则**：明确说明计算逻辑和约束

### 2. 性能优化

- **使用提示缓存**：缓存系统提示以降低成本
- **限制 skill 数量**：避免在单个请求中加载太多 skills
- **监控加载模式**：识别常用 skills 进行优化
- **考虑预加载**：对常用 skills 组合可以预加载

### 3. 安全性

- **验证 skill 访问**：确保用户有权访问特定 skills
- **审计 skill 使用**：记录哪些 skills 被加载和使用
- **限制敏感信息**：不要在 skills 中包含密码或密钥
- **版本控制**：追踪 skill 的变更历史

### 4. 维护性

- **模块化 skills**：每个业务领域一个 skill
- **团队所有权**：指定每个 skill 的维护者
- **测试 skills**：验证 skill 内容的准确性
- **定期更新**：保持 skills 与数据库模式同步

### 5. 监控和调试

- **使用 LangSmith**：追踪 skill 加载和使用
- **记录失败**：当 agent 无法找到正确的 skill
- **A/B 测试**：测试不同的 skill 描述和内容
- **收集反馈**：从用户那里了解哪些 skills 有用

## 快速开始

### 前置要求

```bash
# 安装依赖
pip install langchain langchain-openai langgraph
```

### 环境配置

创建 `.env` 文件：

```env
# OpenAI API 密钥
OPENAI_API_KEY=sk-...

# LangSmith 追踪（可选）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
```

### 运行示例

```bash
# 进入目录
cd multi-agents/skills-sql-assistant

# 运行示例
python sql_skills_example.py
```

## 关键要点

1. **Progressive Disclosure**：按需加载信息，而非预先全部加载
2. **Prompt-based Skills**：Skills 主要是专门化的提示指令
3. **三层架构**：元数据（系统提示）→ 核心内容（工具加载）→ 详细资源
4. **减少上下文**：只加载当前任务需要的 2-3 个 skills
5. **团队自治**：不同团队独立维护各自领域的 skills
6. **可扩展**：添加数十或数百个 skills 而不压垮上下文

## 与其他模式的对比

| 特性 | Skills 模式 | Subagents 模式 | Router 模式 |
|------|------------|---------------|------------|
| 核心技术 | Progressive Disclosure | 层级协调 | 智能路由 |
| 信息加载 | 按需加载 | 预先配置 | 路由到专家 |
| 上下文管理 | 动态控制 | 分布到子 agents | 分区隔离 |
| 团队协作 | Skills 独立维护 | Sub-agents 独立 | Knowledge bases 独立 |
| 适用场景 | 大型模式、多领域 | 复杂工作流 | 多知识库 |

## 下一步

完成本教程后，你可以：

1. ✅ 实现自己的 skills 系统
2. ✅ 探索不同的存储后端（文件系统、数据库）
3. ✅ 添加语义搜索以在大型 skill 集合中查找
4. ✅ 结合 few-shot 提示提高输出质量
5. ✅ 使用 LangSmith 监控 skill 加载模式
6. ✅ 探索其他上下文工程技术
7. ✅ 查看其他多 agent 模式：
   - [Subagents 模式](../subagents-personal-assistant/) - 层级任务协调
   - [Handoffs 模式](../handoffs-customer-support/) - 顺序工作流
   - [Router 模式](../router-knowledge-base/) - 知识库路由

## 更多资源

- [LangChain 官方文档](https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant)
- [Anthropic: Equipping agents with Agent Skills](https://www.anthropic.com/news/agent-skills)
- [Context Engineering 技术](https://docs.langchain.com/oss/python/langchain/context-engineering)
- [Middleware 文档](https://docs.langchain.com/oss/python/langchain/middleware)

## 总结

Skills 模式通过 progressive disclosure 提供了一种优雅的方式来管理大规模上下文：

- 📦 **按需加载**：只加载需要的 skills
- 🎯 **减少上下文**：避免压垮上下文窗口
- 🔧 **Prompt-based**：通过提示引导行为
- 🏢 **团队自治**：独立开发和维护 skills
- 📈 **可扩展**：支持数百个 skills

这是构建大型企业数据分析系统、多租户 SaaS 平台的理想模式！
