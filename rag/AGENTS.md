# RAG 实例架构规范

本文档定义了 `rag` 目录下每个 Agent 实例的标准架构和组织方式，用于保持项目的一致性和可维护性。

## 📋 目录结构规范

### 整体架构

```
rag/
├── AGENTS.md                    # 本架构规范文档
├── agent_name_1/                # Agent 实例目录
│   ├── README.md                # 知识讲解文档（必需）
│   └── agent_name_1_example.py  # 代码示例文件（必需）
├── agent_name_2/
│   ├── README.md
│   └── agent_name_2_example.py
└── ...
```

### 文件命名规范

1. **实例目录名**：使用小写字母和下划线，命名应清晰表达 Agent 功能
   - ✅ 推荐：`rag_agent`, `sql_agent`, `semantic_search`
   - ❌ 避免：`Agent1`, `test`, `demo`

2. **知识讲解文档**：统一命名为 `README.md`
   
3. **代码示例文件**：命名格式为 `{实例目录名}_example.py`
   - 例如：`rag_agent_example.py`, `sql_agent_example.py`
   - 如有多个示例，可添加后缀：`sql_agent_simple.py`, `sql_agent_advanced.py`

## 📄 必需文件说明

### 1. README.md（知识讲解文档）

每个 Agent 实例必须包含一个 `README.md` 文档，用于讲解该 Agent 的相关知识和使用方法。

#### 推荐内容结构：

```markdown
# Agent 名称

## 概述
简要介绍该 Agent 的用途和应用场景

## 核心概念
解释相关的核心技术概念和原理

## 依赖说明
说明所需依赖包（注：实际依赖已配置在项目根目录的 pyproject.toml 中）

主要依赖：
- langchain: 用于...
- langgraph: 用于...
- 其他依赖...

安装方式：
\`\`\`bash
uv sync
\`\`\`

## 配置说明
- API Keys 配置
- 环境变量设置
- 其他必要配置项

## 使用示例
展示基本使用方法和代码片段

## 运行说明
如何运行示例代码

## 参考资料
相关文档和学习资源链接
```

### 2. *_example.py（代码示例文件）

每个 Agent 实例必须包含至少一个可运行的代码示例文件。

#### 代码文件要求：

- **完整可运行**：代码应该是完整的、可直接运行的示例
- **注释清晰**：关键代码段需要有中文注释说明
- **结构清晰**：代码组织合理，易于理解和学习
- **最佳实践**：体现该 Agent 的最佳使用方式

#### 推荐代码结构：

```python
"""
Agent 名称示例
简要说明该示例的功能和用途
"""

# 导入必要的库
from langchain_core import ...
from langgraph import ...

# 配置部分
# 说明配置项的作用

# 主要功能实现
# 使用注释说明每个关键步骤

# 使用示例
if __name__ == "__main__":
    # 演示如何使用
    pass
```

## 🚫 禁止包含的文件

为保持项目统一管理和简洁性，每个 Agent 实例目录中**不应包含**以下文件：

### ❌ requirements.txt
- **原因**：依赖已统一管理在项目根目录的 `pyproject.toml` 中
- **替代方案**：在 README.md 中简要说明所需依赖即可

### ❌ 其他配置文件（可选禁止）
- 避免在实例目录中创建独立的配置文件
- 如需特定配置，应在 README.md 中说明或使用示例代码中的注释说明

## 📦 依赖管理

### 统一依赖管理原则

所有 Agent 实例的依赖包统一配置在项目根目录的 `pyproject.toml` 文件中。

```toml
[project]
name = "vedio-teach"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "langchain>=0.3.18",
    "langgraph>=0.2.62",
    "dashscope>=1.25.7",
    # ... 其他依赖
]
```

### 添加新依赖

当创建新的 Agent 实例需要额外依赖时：

1. 在项目根目录运行：
   ```bash
   uv add package-name
   ```

2. 在该 Agent 的 README.md 中说明所需依赖：
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

## 📝 创建新 Agent 实例的步骤

1. **创建实例目录**
   ```bash
   mkdir rag/new_agent_name
   ```

2. **创建 README.md**
   - 按照推荐结构编写知识讲解文档
   - 说明所需依赖（无需创建 requirements.txt）

3. **创建代码示例文件**
   - 命名为 `new_agent_name_example.py`
   - 编写完整可运行的示例代码

4. **添加依赖（如需要）**
   ```bash
   cd ../..  # 回到项目根目录
   uv add new-package
   ```

5. **测试运行**
   - 确保代码可以正常运行
   - 验证文档说明清晰准确

## 🎯 现有实例参考

当前项目中的标准实例示例：

- **semantic_search/**：语义搜索 Agent 实例
- **sql_agent/**：SQL 查询 Agent 实例
- **rag_agent/**：RAG 检索增强生成 Agent 实例
- **voice_agent/**：语音交互 Agent 实例

在创建新实例时，可以参考这些现有实例的组织方式。

## 📌 注意事项

1. **保持一致性**：所有实例都应遵循相同的架构规范
2. **文档先行**：先完善 README.md 再编写代码
3. **代码质量**：示例代码应体现最佳实践，而非快速原型
4. **定期更新**：当依赖或技术栈更新时，及时更新文档和代码
5. **简洁原则**：每个实例只保留必需的文件，避免冗余


---

**版本**：v1.0  
**最后更新**：2026-01-12  
**维护者**：项目团队
