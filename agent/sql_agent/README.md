# SQL Agent - 智能 SQL 查询助手

## 概述

SQL Agent 是一个使用 LangGraph 构建的智能数据库查询助手。它能够理解自然语言问题，自动生成 SQL 查询，执行查询并返回结果，同时具备错误处理和查询优化能力。

### 应用场景

- **商业智能**：将自然语言转换为 SQL 查询进行数据分析
- **数据探索**：快速查询数据库而无需编写 SQL
- **报表生成**：自动化数据提取和报告
- **数据库管理**：智能化数据库操作和维护

## 核心概念

### 什么是 SQL Agent？

SQL Agent 是一个能够将自然语言转换为 SQL 查询的智能系统。与传统的硬编码 SQL 生成不同，SQL Agent：

1. **理解意图**：准确理解用户的查询意图
2. **生成 SQL**：自动生成正确的 SQL 语句
3. **执行查询**：安全地执行 SQL 并获取结果
4. **错误修复**：自动检测和修复 SQL 错误
5. **结果解释**：用自然语言解释查询结果

### LangGraph 基础

#### 状态管理（State）
使用自定义状态管理查询过程中的信息：
- **messages**: 对话历史
- **sql_query**: 生成的 SQL 查询
- **query_result**: 查询执行结果
- **error_message**: 错误信息（如有）

#### 节点（Nodes）
- **generate_sql**：生成 SQL 查询
- **execute_sql**：执行 SQL 查询
- **check_result**：检查查询结果
- **fix_sql**：修复错误的 SQL
- **generate_response**：生成自然语言回复

#### 边（Edges）
- **条件边**：根据查询结果决定下一步（成功/失败）
- **循环边**：支持 SQL 错误修复的迭代过程

## 依赖说明

本实例需要以下依赖包（已配置在项目根目录的 pyproject.toml 中）：

### 主要依赖
- **langchain** (>=0.3.18): LLM 调用和工具管理
- **langgraph** (>=0.2.62): 构建 Agent 图结构
- **langchain-openai**: OpenAI 模型集成
- **langchain-community**: SQL 工具包等社区工具

### 数据库依赖
- **sqlite3**: Python 内置，用于 SQLite 数据库
- **sqlalchemy**: 数据库抽象层（可选，用于其他数据库）
- **pymysql** / **psycopg2**: MySQL / PostgreSQL 连接器（可选）

### 安装方式

```bash
# 在项目根目录运行
uv sync

# 如需连接 MySQL
uv add pymysql

# 如需连接 PostgreSQL
uv add psycopg2-binary
```

## Agent 架构设计

### 状态定义

```python
from typing import TypedDict, Optional
from langchain_core.messages import BaseMessage

class SQLAgentState(TypedDict):
    """SQL Agent 状态"""
    messages: list[BaseMessage]           # 消息历史
    sql_query: Optional[str]              # 生成的 SQL
    query_result: Optional[str]           # 查询结果
    error_message: Optional[str]          # 错误信息
    retry_count: int                      # 重试次数
```

### 节点说明

#### 1. generate_sql
**功能**：根据用户问题生成 SQL 查询
- 输入：用户的自然语言问题 + 数据库架构信息
- 输出：SQL 查询语句
- 逻辑：使用 LLM 理解问题并生成 SQL

**提示词模板**：
```python
GENERATE_SQL_PROMPT = """
给定以下数据库架构:
{schema}

用户问题: {question}

请生成一个 SQL 查询来回答这个问题。
只返回 SQL 语句，不要包含其他解释。
"""
```

#### 2. execute_sql
**功能**：执行生成的 SQL 查询
- 输入：SQL 查询语句
- 输出：查询结果或错误信息
- 逻辑：连接数据库并安全执行 SQL

**安全措施**：
- 只允许 SELECT 查询（防止数据修改）
- 设置查询超时限制
- 结果行数限制

#### 3. check_result
**功能**：检查查询执行结果
- 输入：查询结果或错误信息
- 输出：路由决策（成功/失败/重试）
- 逻辑：判断是否需要修复 SQL 或直接返回结果

#### 4. fix_sql
**功能**：修复错误的 SQL 查询
- 输入：原始 SQL + 错误信息
- 输出：修复后的 SQL
- 逻辑：使用 LLM 分析错误并生成正确的 SQL

#### 5. generate_response
**功能**：将查询结果转换为自然语言
- 输入：查询结果
- 输出：用户友好的自然语言回复
- 逻辑：总结和解释查询结果

### 工作流程

```
START
  ↓
generate_sql (生成 SQL)
  ↓
execute_sql (执行 SQL)
  ↓
check_result (检查结果)
  ↓
[条件分支]
  ├→ 成功 → generate_response → END
  ├→ 失败 (可重试) → fix_sql → execute_sql
  └→ 失败 (超过重试) → generate_error_response → END
```

### 流程图示

```mermaid
graph TD
    A[START] --> B[generate_sql]
    B --> C[execute_sql]
    C --> D[check_result]
    D -->|成功| E[generate_response]
    D -->|失败_可重试| F[fix_sql]
    D -->|失败_超限| G[generate_error_response]
    F --> C
    E --> H[END]
    G --> H
```

## 配置说明

### 环境变量设置

```bash
# .env 文件
OPENAI_API_KEY=your_openai_api_key_here

# 数据库连接（如需要）
DATABASE_URL=sqlite:///path/to/database.db
# 或 MySQL
DATABASE_URL=mysql+pymysql://user:password@localhost/dbname
# 或 PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/dbname
```

### 数据库配置

#### SQLite（本地开发）

```python
import sqlite3

conn = sqlite3.connect("example.db")
```

#### MySQL

```python
from sqlalchemy import create_engine

engine = create_engine("mysql+pymysql://user:password@localhost/dbname")
```

#### PostgreSQL

```python
from sqlalchemy import create_engine

engine = create_engine("postgresql://user:password@localhost/dbname")
```

### Agent 配置

```python
# 最大重试次数
MAX_RETRIES = 3

# 查询超时时间（秒）
QUERY_TIMEOUT = 30

# 最大返回行数
MAX_ROWS = 100

# 允许的 SQL 操作
ALLOWED_OPERATIONS = ["SELECT"]
```

## 使用示例

### 基本使用

```python
from langgraph.graph import StateGraph

# 1. 构建图
workflow = StateGraph(SQLAgentState)
workflow.add_node("generate_sql", generate_sql)
workflow.add_node("execute_sql", execute_sql)
# ... 添加其他节点和边
graph = workflow.compile()

# 2. 运行查询
result = graph.invoke({
    "messages": [
        {"role": "user", "content": "显示销售额最高的 5 个产品"}
    ],
    "retry_count": 0
})

# 3. 获取结果
print(result["messages"][-1].content)
```

### 流式输出

```python
for chunk in graph.stream({
    "messages": [{"role": "user", "content": "统计每个类别的产品数量"}],
    "retry_count": 0
}):
    for node, update in chunk.items():
        print(f"节点: {node}")
        if "sql_query" in update:
            print(f"SQL: {update['sql_query']}")
```

### 批量查询

```python
questions = [
    "有多少客户？",
    "最贵的产品是什么？",
    "2024 年的总销售额是多少？"
]

for question in questions:
    result = graph.invoke({
        "messages": [{"role": "user", "content": question}],
        "retry_count": 0
    })
    print(f"问题: {question}")
    print(f"答案: {result['messages'][-1].content}\n")
```

## 运行说明

### 1. 准备数据库

```bash
# 下载示例数据库（Chinook 音乐数据库）
wget https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite

# 或创建自己的数据库
sqlite3 mydb.db < schema.sql
```

### 2. 设置环境变量

```bash
# Windows
set OPENAI_API_KEY=your_key

# Linux/Mac
export OPENAI_API_KEY=your_key
```

### 3. 运行示例

```bash
python agent/sql_agent/sql_agent_example.py
```

### 预期输出

```
🔍 正在生成 SQL 查询...
生成的 SQL: SELECT Name, UnitPrice FROM Track ORDER BY UnitPrice DESC LIMIT 5

⚡ 正在执行 SQL 查询...
✅ 查询执行成功

💡 正在生成回复...
答案: 价格最高的 5 首歌曲是:
1. Name: Symphony No. 5, Price: $1.99
2. Name: Piano Concerto No. 2, Price: $1.99
...
```

## 核心代码片段

### 1. 生成 SQL

```python
from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate

def generate_sql(state: SQLAgentState):
    """根据问题生成 SQL 查询"""
    question = state["messages"][-1].content
    
    # 获取数据库架构
    db = SQLDatabase.from_uri("sqlite:///database.db")
    schema = db.get_table_info()
    
    # 构建提示词
    prompt = PromptTemplate.from_template("""
    数据库架构:
    {schema}
    
    用户问题: {question}
    
    生成 SQL 查询（只返回 SQL 语句）:
    """)
    
    # 生成 SQL
    response = llm.invoke(prompt.format(schema=schema, question=question))
    sql = response.content.strip()
    
    return {"sql_query": sql}
```

### 2. 执行 SQL

```python
import sqlite3

def execute_sql(state: SQLAgentState):
    """安全执行 SQL 查询"""
    sql = state["sql_query"]
    
    # 安全检查
    if not sql.strip().upper().startswith("SELECT"):
        return {
            "error_message": "只允许 SELECT 查询",
            "query_result": None
        }
    
    try:
        # 连接数据库
        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        
        # 执行查询
        cursor.execute(sql)
        results = cursor.fetchmany(100)  # 限制返回行数
        
        conn.close()
        
        return {
            "query_result": str(results),
            "error_message": None
        }
    except Exception as e:
        return {
            "query_result": None,
            "error_message": str(e)
        }
```

### 3. 检查结果并路由

```python
from typing import Literal

def check_result(
    state: SQLAgentState
) -> Literal["generate_response", "fix_sql", "error_response"]:
    """检查查询结果并决定下一步"""
    
    # 如果有错误
    if state["error_message"]:
        # 检查是否可以重试
        if state["retry_count"] < 3:
            return "fix_sql"
        else:
            return "error_response"
    
    # 如果成功
    if state["query_result"]:
        return "generate_response"
    
    return "error_response"
```

### 4. 修复 SQL

```python
def fix_sql(state: SQLAgentState):
    """修复错误的 SQL 查询"""
    original_sql = state["sql_query"]
    error = state["error_message"]
    question = state["messages"][0].content
    
    prompt = f"""
    原始问题: {question}
    
    生成的 SQL: {original_sql}
    
    错误信息: {error}
    
    请修复这个 SQL 查询，只返回修正后的 SQL:
    """
    
    response = llm.invoke(prompt)
    fixed_sql = response.content.strip()
    
    return {
        "sql_query": fixed_sql,
        "retry_count": state["retry_count"] + 1,
        "error_message": None
    }
```

## 进阶功能

### 1. 多表查询

```python
def generate_sql_with_joins(state: SQLAgentState):
    """支持多表 JOIN 查询"""
    prompt = """
    数据库有以下表:
    - customers (id, name, email)
    - orders (id, customer_id, total, date)
    - products (id, name, price)
    
    问题: {question}
    
    生成包含必要 JOIN 的 SQL 查询:
    """
    # ... 实现
```

### 2. 查询优化

```python
def optimize_query(sql: str) -> str:
    """优化 SQL 查询性能"""
    optimizations = {
        "SELECT *": "SELECT specific_columns",
        "NO LIMIT": "LIMIT 1000",
        # 更多优化规则...
    }
    # 应用优化
    return optimized_sql
```

### 3. 查询缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def execute_sql_cached(sql: str):
    """缓存频繁查询的结果"""
    return execute_query(sql)
```

### 4. 人机协作（Human-in-the-Loop）

```python
from langgraph.checkpoint.memory import MemorySaver

def approve_sql(state: SQLAgentState):
    """在执行前请求用户确认 SQL"""
    sql = state["sql_query"]
    print(f"将要执行: {sql}")
    approval = input("是否执行？ (y/n): ")
    
    if approval.lower() == 'y':
        return {"approved": True}
    else:
        return {"approved": False}

# 在图中添加此节点
workflow.add_node("approve_sql", approve_sql)
```

## 安全最佳实践

### 1. SQL 注入防护

```python
def sanitize_sql(sql: str) -> bool:
    """检查 SQL 是否安全"""
    dangerous_keywords = [
        "DROP", "DELETE", "UPDATE", "INSERT",
        "ALTER", "CREATE", "TRUNCATE", "EXEC"
    ]
    
    sql_upper = sql.upper()
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return False
    return True
```

### 2. 权限控制

```python
def check_table_access(table_name: str, user_role: str) -> bool:
    """检查用户是否有权访问表"""
    allowed_tables = {
        "analyst": ["products", "sales", "customers"],
        "manager": ["products", "sales", "customers", "employees"],
        "admin": ["*"]
    }
    
    if user_role == "admin":
        return True
    
    return table_name in allowed_tables.get(user_role, [])
```

### 3. 结果数据脱敏

```python
def mask_sensitive_data(results: list) -> list:
    """脱敏敏感数据"""
    sensitive_fields = ["email", "phone", "ssn"]
    
    for row in results:
        for field in sensitive_fields:
            if field in row:
                row[field] = "***masked***"
    
    return results
```

## 性能优化建议

1. **连接池**：使用数据库连接池避免频繁连接
2. **查询限制**：始终设置 LIMIT 避免大结果集
3. **索引建议**：分析慢查询并建议添加索引
4. **异步执行**：对于长查询使用异步执行
5. **结果缓存**：缓存常见查询结果

## 常见问题

### Q: 如何连接远程数据库？

```python
from sqlalchemy import create_engine

# MySQL
engine = create_engine(
    "mysql+pymysql://user:pass@host:3306/dbname",
    pool_size=10,
    max_overflow=20
)

# PostgreSQL
engine = create_engine(
    "postgresql://user:pass@host:5432/dbname",
    pool_pre_ping=True
)
```

### Q: 如何处理大结果集？

```python
def execute_with_pagination(sql: str, page_size: int = 100):
    """分页执行大查询"""
    offset = 0
    while True:
        paginated_sql = f"{sql} LIMIT {page_size} OFFSET {offset}"
        results = execute_query(paginated_sql)
        
        if not results:
            break
        
        yield results
        offset += page_size
```

### Q: 如何添加查询日志？

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def execute_sql_with_logging(sql: str):
    """记录 SQL 执行日志"""
    logger.info(f"Executing SQL: {sql}")
    start_time = time.time()
    
    try:
        result = execute_query(sql)
        duration = time.time() - start_time
        logger.info(f"Query succeeded in {duration:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
```

## 参考资料

### 官方文档
- [LangGraph SQL Agent 教程](https://docs.langchain.com/oss/python/langgraph/sql-agent)
- [LangChain SQL 工具包](https://python.langchain.com/docs/integrations/toolkits/sql_database)
- [SQLDatabase 工具](https://python.langchain.com/docs/integrations/tools/sql_database)

### 数据库资源
- [Chinook 示例数据库](https://github.com/lerocha/chinook-database)
- [SQLite 文档](https://www.sqlite.org/docs.html)
- [SQL 最佳实践](https://www.sqlstyle.guide/)

### 安全指南
- [SQL 注入防护](https://owasp.org/www-community/attacks/SQL_Injection)
- [数据库安全最佳实践](https://cheatsheetseries.owasp.org/cheatsheets/Database_Security_Cheat_Sheet.html)

---

**版本**：v1.0  
**最后更新**：2026-01-12  
**作者**：项目团队
