# Chroma 向量数据库管理说明

本文档说明如何管理 Chroma 向量数据库，包括临时使用和持久化使用两种模式。

## 📊 两种模式对比

| 特性 | 临时模式 | 持久化模式 |
|------|---------|-----------|
| 数据存储 | 临时目录 | 指定目录 |
| 程序结束后 | 自动删除 | 保留在磁盘 |
| 适用场景 | 演示、测试 | 生产环境 |
| 启动速度 | 慢（每次重建） | 快（直接加载） |
| 磁盘占用 | 无（自动清理） | 持续占用 |

## 🔧 使用方式

### 方案一：临时模式（当前默认）✨

**适用场景：**
- 教学演示
- 开发测试
- 不需要保存向量数据
- 文档频繁变化

**实现方式：**

```python
import tempfile
import shutil
from langchain_chroma import Chroma

# 创建临时目录
temp_dir = tempfile.mkdtemp()

try:
    # 创建向量存储
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=temp_dir
    )
    
    # 使用 vector_store...
    
finally:
    # 程序结束时清理
    shutil.rmtree(temp_dir, ignore_errors=True)
```

**优点：**
- ✅ 自动清理，不留垃圾文件
- ✅ 不占用磁盘空间
- ✅ 适合频繁测试

**缺点：**
- ⚠️ 每次运行都要重新生成向量（耗时）
- ⚠️ 不能重用已有数据

### 方案二：持久化模式 💾

**适用场景：**
- 生产环境
- 数据量大
- 文档不常变化
- 需要快速启动

**实现方式：**

```python
from langchain_chroma import Chroma
from pathlib import Path

# 指定持久化目录
persist_dir = "./chroma_db"

# 检查是否已存在数据库
if Path(persist_dir).exists():
    print("📂 加载已有向量数据库...")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
else:
    print("📦 创建新的向量数据库...")
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
```

**优点：**
- ✅ 只需生成一次向量
- ✅ 后续启动快速
- ✅ 可以增量添加文档

**缺点：**
- ⚠️ 占用磁盘空间
- ⚠️ 需要手动管理和清理
- ⚠️ 文档更新需要重建索引

## 🗂️ 持久化模式的目录结构

使用持久化模式后，会创建如下目录结构：

```
your_project/
├── rag/
│   └── rag_agent/
│       ├── rag_agent_example.py
│       ├── README.md
│       └── chroma_db/              # Chroma 数据库目录
│           ├── chroma.sqlite3      # SQLite 数据库
│           ├── *.bin              # 向量数据文件
│           └── ...
```

**磁盘占用估算：**
- 1000 个文档块 ≈ 10-50 MB
- 10000 个文档块 ≈ 100-500 MB

## 🧹 如何清理 Chroma 数据库

### 方法一：手动删除

**Windows PowerShell:**
```powershell
Remove-Item -Recurse -Force .\chroma_db
```

**Linux/macOS:**
```bash
rm -rf ./chroma_db
```

### 方法二：Python 脚本

创建 `cleanup_chroma.py`：

```python
import shutil
from pathlib import Path

# 要删除的数据库目录
db_dirs = [
    "./chroma_db",
    "./rag/semantic_search/chroma_db",
    "./rag/rag_agent/chroma_db",
]

for db_dir in db_dirs:
    if Path(db_dir).exists():
        print(f"🧹 删除 {db_dir}")
        shutil.rmtree(db_dir, ignore_errors=True)
        print(f"✅ {db_dir} 已删除")
    else:
        print(f"⏭️ {db_dir} 不存在，跳过")

print("\n清理完成！")
```

运行：
```bash
python cleanup_chroma.py
```

### 方法三：在 .gitignore 中排除

如果使用 Git，确保将 Chroma 数据库排除：

```gitignore
# Chroma 向量数据库
chroma_db/
**/chroma_db/
*.chroma
chroma.sqlite3
```

## 🔄 如何更新向量数据库

### 完全重建（推荐）

```python
import shutil
from pathlib import Path

persist_dir = "./chroma_db"

# 删除旧数据库
if Path(persist_dir).exists():
    print("🗑️ 删除旧数据库...")
    shutil.rmtree(persist_dir)

# 创建新数据库
print("📦 创建新数据库...")
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=persist_dir
)
```

### 增量添加（高级）

```python
from langchain_chroma import Chroma

# 加载已有数据库
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 添加新文档
new_docs = [...]  # 新的文档列表
vector_store.add_documents(new_docs)

print("✅ 新文档已添加到数据库")
```

## 📝 最佳实践

### 1. 开发阶段

**推荐：使用临时模式**

```python
import tempfile
import shutil

temp_dir = tempfile.mkdtemp()
try:
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=temp_dir
    )
    # 测试代码...
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
```

### 2. 生产环境

**推荐：使用持久化模式 + 定期更新**

```python
from pathlib import Path
import os

# 从环境变量读取配置
persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

# 检查是否强制重建
force_rebuild = os.environ.get("CHROMA_FORCE_REBUILD", "false") == "true"

if force_rebuild or not Path(persist_dir).exists():
    vector_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
else:
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
```

### 3. 环境变量配置

在 `.env` 文件中：

```env
# Chroma 配置
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_FORCE_REBUILD=false
```

## 🔍 监控和调试

### 查看数据库大小

**Python 代码：**

```python
from pathlib import Path

def get_dir_size(path):
    """计算目录大小（字节）"""
    total = 0
    for file in Path(path).rglob('*'):
        if file.is_file():
            total += file.stat().st_size
    return total

# 使用
db_size = get_dir_size("./chroma_db")
print(f"数据库大小: {db_size / 1024 / 1024:.2f} MB")
```

### 查看文档数量

```python
# 获取集合信息
collection = vector_store._collection
doc_count = collection.count()
print(f"向量数量: {doc_count}")
```

## ⚠️ 常见问题

### Q: 为什么每次运行都要等很久？

**A:** 如果使用临时模式，每次运行都要重新生成向量。

**解决方案：**
- 开发时：使用小文档测试
- 生产时：切换到持久化模式

### Q: 磁盘空间不够怎么办？

**A:** 持久化模式会占用磁盘空间。

**解决方案：**
1. 定期清理不用的数据库
2. 优化文档分割（减少块数量）
3. 使用更小的嵌入维度

### Q: 文档更新了怎么办？

**A:** 向量数据库不会自动更新。

**解决方案：**
- 删除旧数据库，重新创建
- 或使用增量添加（需要追踪已添加的文档）

### Q: 如何在多个脚本间共享数据库？

**A:** 使用持久化模式，指定相同的目录。

```python
# 脚本 A 创建
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./shared_chroma_db"
)

# 脚本 B 加载
vector_store = Chroma(
    persist_directory="./shared_chroma_db",
    embedding_function=embeddings
)
```

## 🎯 推荐配置

根据你的使用场景选择：

### 教学/演示（当前配置）✨
```python
# 使用临时模式，自动清理
temp_dir = tempfile.mkdtemp()
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=temp_dir
)
# 使用完毕后清理
shutil.rmtree(temp_dir, ignore_errors=True)
```

### 开发/调试
```python
# 使用本地目录，手动管理
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="./chroma_db_dev"
)
```

### 生产环境
```python
# 使用配置化的持久化目录
persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=persist_dir
)
```

## 📚 更多资源

- [Chroma 官方文档](https://docs.trychroma.com/)
- [LangChain Chroma 集成](https://python.langchain.com/docs/integrations/vectorstores/chroma)
- [向量数据库最佳实践](https://www.pinecone.io/learn/vector-database/)

---

**更新时间：** 2026-01-12  
**版本：** 1.0
