# Multi-Agents 架构规范

本文档定义了 `multi-agents` 目录下多代理系统实例的标准架构和组织方式，用于保持项目的一致性和可维护性。

## 📋 目录结构规范

### 整体架构

```
multi-agents/
├── AGENTS.md                           # 本架构规范文档
├── subagents-personal-assistant/       # 子代理模式：个人助理实例
│   ├── README.md                       # 知识讲解文档（必需）
│   └── subagents_personal_assistant_example.py  # 代码示例文件（必需）
├── handoffs-customer-support/          # 切换模式：客户支持实例
│   ├── README.md
│   └── handoffs_customer_support_example.py
├── router-knowledge-base/              # 路由模式：知识库实例
│   ├── README.md
│   └── router_knowledge_base_example.py
├── skills-sql-assistant/               # 技能模式：SQL 助手实例
│   ├── README.md
│   └── skills_sql_assistant_example.py
└── ...
```

### 文件命名规范

1. **实例目录名**：使用小写字母和连字符，命名应清晰表达多代理模式和应用场景
   - ✅ 推荐：`subagents-personal-assistant`, `handoffs-customer-support`, `router-knowledge-base`
   - ❌ 避免：`agent1`, `test-multi`, `demo`

2. **知识讲解文档**：统一命名为 `README.md`
   
3. **代码示例文件**：命名格式为 `{实例目录名}_example.py`（使用下划线）
   - 例如：`subagents_personal_assistant_example.py`, `handoffs_customer_support_example.py`
   - 如有多个示例，可添加后缀：`subagents_simple.py`, `subagents_advanced.py`

## 🏗️ 多代理架构模式

本目录实现了四种核心的多代理协作模式，每种模式适用于不同的应用场景：

### 1. Subagents（子代理模式）

**核心概念**：Supervisor Pattern - 由一个中央监督者代理协调多个专业化的工作代理

**适用场景**：
- 任务需要不同类型的专业知识
- 每个领域有多个工具或复杂逻辑
- 需要集中式工作流控制
- 子代理不需要直接与用户对话

**示例应用**：`subagents-personal-assistant/`
- 日历代理处理日程安排
- 邮件代理管理通信
- 监督者协调整体工作流

### 2. Handoffs（切换模式）

**核心概念**：Agent-to-Agent Conversations - 代理之间可以进行对话并移交控制权

**适用场景**：
- 代理需要与用户进行对话
- 不同阶段需要不同专家介入
- 任务需要在代理之间传递上下文

**示例应用**：`handoffs-customer-support/`
- 初级支持代理处理常见问题
- 专家代理处理复杂问题
- 计费代理处理付款相关事务

### 3. Router（路由模式）

**核心概念**：智能路由 - 根据查询类型将请求路由到最合适的专业代理

**适用场景**：
- 有明确分类的查询类型
- 每个类别有专门的处理逻辑
- 需要快速分发请求

**示例应用**：`router-knowledge-base/`
- 根据问题类型路由到不同知识库
- 技术问题、产品问题、账户问题分别处理

### 4. Skills（技能模式）

**核心概念**：可组合的技能单元 - 将复杂能力封装为可重用的技能模块

**适用场景**：
- 需要组合多个技能解决问题
- 技能需要在不同场景中复用
- 能力需要模块化管理

**示例应用**：`skills-sql-assistant/`
- SQL 查询技能
- 数据分析技能
- 结果可视化技能

## 📄 必需文件说明

### 1. README.md（知识讲解文档）

每个多代理实例必须包含一个 `README.md` 文档，用于讲解该多代理系统的架构和使用方法。

#### 推荐内容结构：

```markdown
# 多代理系统名称

## 概述
- 简要介绍该多代理系统的用途和应用场景
- 说明采用的多代理模式（Subagents/Handoffs/Router/Skills）

## 架构设计

### 代理角色
列出系统中的所有代理及其职责：
- **Agent 1 名称**：职责描述
- **Agent 2 名称**：职责描述
- **Supervisor/Router**：协调逻辑说明

### 协作流程
描述代理之间的协作流程：
1. 用户请求如何被接收
2. 如何分发/路由到具体代理
3. 代理之间如何通信
4. 如何返回最终结果

### 架构图（可选）
使用 Mermaid 或文字描述展示系统架构

## 核心概念
- 解释所采用的多代理模式的核心原理
- 说明为什么选择这种模式
- 与其他模式的对比

## 依赖说明
说明所需依赖包（注：实际依赖已配置在项目根目录的 pyproject.toml 中）

主要依赖：
- langchain: 用于构建代理
- langgraph: 用于状态管理和工作流编排
- 其他依赖...

安装方式：
\`\`\`bash
uv sync
\`\`\`

## 配置说明
- API Keys 配置
- 环境变量设置
- 模型选择建议
- 其他必要配置项

## 使用示例
展示基本使用方法和代码片段

## 运行说明
如何运行示例代码

## 进阶特性

### Human-in-the-Loop（可选）
如果支持人工审核，说明如何实现

### 信息流控制（可选）
如何控制代理之间的信息传递

### 错误处理
如何处理代理执行失败的情况

## 最佳实践
- 提示词工程建议
- 性能优化技巧
- 常见问题和解决方案

## 何时使用此模式
明确说明：
- ✅ 适合使用的场景
- ❌ 不适合使用的场景
- 🔄 可替代的其他模式

## 参考资料
- LangChain 官方文档链接
- 相关论文或博客
- 其他学习资源
```

### 2. *_example.py（代码示例文件）

每个多代理实例必须包含至少一个可运行的代码示例文件。

#### 代码文件要求：

- **完整可运行**：代码应该是完整的、可直接运行的多代理系统示例
- **注释清晰**：关键代码段需要有中文注释说明
- **结构清晰**：
  - 明确区分各个代理的定义
  - 清晰展示代理协作逻辑
  - 易于理解和学习
- **最佳实践**：体现该多代理模式的最佳使用方式

#### 推荐代码结构：

```python
"""
多代理系统名称示例
简要说明该示例的功能和用途
采用的模式：Subagents/Handoffs/Router/Skills
"""

# ============================================
# 导入必要的库
# ============================================
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ============================================
# 配置部分
# ============================================
# API Keys 和模型配置
# ...

# ============================================
# 定义工具
# ============================================
# 各个代理使用的工具定义
# ...

# ============================================
# 创建子代理
# ============================================
# Agent 1: 职责描述
def create_agent_1():
    """创建第一个专业代理"""
    pass

# Agent 2: 职责描述
def create_agent_2():
    """创建第二个专业代理"""
    pass

# ============================================
# 将代理封装为工具（仅限 Subagents 模式）
# ============================================
# 将子代理包装为监督者可调用的工具
# ...

# ============================================
# 创建监督者/路由器
# ============================================
# 创建中央协调者
# ...

# ============================================
# 构建多代理系统
# ============================================
# 使用 StateGraph 构建整体工作流
workflow = StateGraph(...)
# 添加节点
workflow.add_node("supervisor", ...)
workflow.add_node("agent_1", ...)
workflow.add_node("agent_2", ...)
# 定义边和路由逻辑
# ...

# ============================================
# 编译和运行
# ============================================
if __name__ == "__main__":
    # 创建可执行图
    graph = workflow.compile(
        checkpointer=MemorySaver()  # 启用记忆功能
    )
    
    # 示例 1：简单单领域请求
    print("=== 示例 1 ===")
    result = graph.invoke({
        "messages": [HumanMessage(content="...")]
    })
    
    # 示例 2：复杂多领域请求
    print("\n=== 示例 2 ===")
    result = graph.invoke({
        "messages": [HumanMessage(content="...")]
    })
    
    # 示例 3：Human-in-the-Loop（如适用）
    print("\n=== 示例 3 ===")
    # ...
```

## 🚫 禁止包含的文件

为保持项目统一管理和简洁性，每个多代理实例目录中**不应包含**以下文件：

### ❌ requirements.txt
- **原因**：依赖已统一管理在项目根目录的 `pyproject.toml` 中
- **替代方案**：在 README.md 中简要说明所需依赖即可

### ❌ 独立配置文件
- 避免在实例目录中创建独立的 `.env` 或配置文件
- 配置说明应在 README.md 中体现

### ❌ 测试数据文件
- 避免存放大型测试数据文件
- 如需示例数据，应使用轻量级的内置示例或在代码中生成

## 📦 依赖管理

### 统一依赖管理原则

所有多代理实例的依赖包统一配置在项目根目录的 `pyproject.toml` 文件中。

```toml
[project]
name = "vedio-teach"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.3.18",
    "langgraph>=0.2.62",
    "langchain-openai>=0.3.18",
    "langchain-anthropic>=0.3.18",
    # ... 其他依赖
]
```

### 添加新依赖

当创建新的多代理实例需要额外依赖时：

1. 在项目根目录运行：
   ```bash
   uv add package-name
   ```

2. 在该实例的 README.md 中说明所需依赖：
   ```markdown
   ## 依赖说明
   本实例需要以下依赖包（已配置在 pyproject.toml）：
   - package-name: 用于...
   ```

### 安装依赖

用户只需在项目根目录运行一次：
```bash
uv sync
```

## 📝 创建新多代理实例的步骤

1. **确定多代理模式**
   - 分析任务需求
   - 选择最适合的模式（Subagents/Handoffs/Router/Skills）
   - 设计代理角色和协作流程

2. **创建实例目录**
   ```bash
   mkdir multi-agents/new-pattern-name
   ```

3. **创建 README.md**
   - 按照推荐结构编写知识讲解文档
   - 详细说明架构设计和代理协作流程
   - 说明所需依赖（无需创建 requirements.txt）

4. **创建代码示例文件**
   - 命名为 `new_pattern_name_example.py`（注意使用下划线）
   - 编写完整可运行的多代理系统示例
   - 包含多个使用场景的演示

5. **添加依赖（如需要）**
   ```bash
   cd ../..  # 回到项目根目录
   uv add new-package
   ```

6. **测试运行**
   - 确保所有代理可以正常协作
   - 验证不同场景下的系统行为
   - 检查文档说明清晰准确

7. **更新本文档**
   - 如果引入了新的多代理模式，在本文档中添加说明

## 🎯 规划实例

当前计划实现的多代理系统实例：

### 1. subagents-personal-assistant/
**模式**：Supervisor Pattern（监督者模式）  
**说明**：个人助理系统，包含日历代理和邮件代理，由监督者协调  
**参考**：https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

### 2. handoffs-customer-support/
**模式**：Handoffs Pattern（切换模式）  
**说明**：客户支持系统，支持代理之间的对话和任务移交  
**参考**：https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

### 3. router-knowledge-base/
**模式**：Router Pattern（路由模式）  
**说明**：知识库系统，根据查询类型智能路由到专业代理  
**参考**：https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

### 4. skills-sql-assistant/
**模式**：Skills Pattern（技能模式）  
**说明**：SQL 助手系统，展示可组合的技能单元  
**参考**：https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

## 🔍 模式选择指南

### 何时使用 Subagents（子代理）模式？
- ✅ 多个不同领域（如日历、邮件、CRM、数据库）
- ✅ 每个领域有多个工具或复杂逻辑
- ✅ 需要集中式工作流控制
- ✅ 子代理不需要直接与用户对话

### 何时使用 Handoffs（切换）模式？
- ✅ 代理需要与用户进行对话
- ✅ 任务在不同阶段需要不同专家
- ✅ 需要在代理之间传递丰富的上下文
- ✅ 用户可能需要与不同专家互动

### 何时使用 Router（路由）模式？
- ✅ 查询类型明确可分类
- ✅ 每个类别有独立的处理流程
- ✅ 需要快速分发和并行处理
- ✅ 不需要代理之间协作

### 何时使用 Skills（技能）模式？
- ✅ 需要组合多个技能解决问题
- ✅ 技能需要在不同上下文中复用
- ✅ 能力需要模块化管理
- ✅ 想要构建灵活的能力组合

## 📌 注意事项

1. **模式选择**：根据实际需求选择合适的多代理模式，避免过度设计
2. **代理边界**：明确定义每个代理的职责范围，避免职责重叠
3. **通信设计**：
   - 仔细设计代理之间的信息流
   - 控制传递给子代理的上下文量
   - 确保监督者收到足够的信息做决策
4. **提示词工程**：
   - 每个代理都需要清晰的系统提示词
   - 监督者的提示词要明确说明如何选择代理
   - 子代理的提示词要强调返回完整信息
5. **错误处理**：
   - 处理代理执行失败的情况
   - 实现重试机制
   - 提供有意义的错误信息
6. **性能考虑**：
   - 监控 LLM 调用次数
   - 优化代理协作流程
   - 考虑使用缓存机制
7. **测试策略**：
   - 独立测试每个代理
   - 测试代理协作流程
   - 测试边界情况和错误场景
8. **文档维护**：
   - 保持文档与代码同步
   - 记录设计决策和权衡
   - 提供清晰的使用示例

## 🔗 相关资源

### LangChain 官方文档
- [Multi-Agent Systems Overview](https://docs.langchain.com/oss/python/langchain/multi-agent/)
- [Subagents: Personal Assistant](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [Handoffs: Customer Support](https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support)
- [Router: Knowledge Base](https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base)
- [Skills: SQL Assistant](https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant)

### LangGraph 文档
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [Checkpointers](https://langchain-ai.github.io/langgraph/reference/checkpoints/)

### 学习资源
- LangChain Academy - Multi-Agent Systems 课程
- LangSmith - 用于调试和监控多代理系统

---

**版本**：v1.0  
**最后更新**：2026-01-12  
**维护者**：项目团队
