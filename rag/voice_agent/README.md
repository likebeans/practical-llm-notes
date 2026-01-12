# LangChain Voice Agent 教程

本教程将指导您使用 LangChain 构建一个语音代理（Voice Agent）。语音代理可以通过语音与用户进行自然对话，而不是传统的文本输入方式。

> 基于官方文档：[Build a voice agent with LangChain](https://docs.langchain.com/oss/python/langchain/voice-agent)

## 目录

1. [概述](#概述)
2. [什么是语音代理](#什么是语音代理)
3. [语音代理的工作原理](#语音代理的工作原理)
4. [核心技术详解](#核心技术详解)
5. [架构设计](#架构设计)
6. [快速开始](#快速开始)
7. [三大核心组件](#三大核心组件)
8. [完整实现](#完整实现)

---

## 概述

近年来，多模态 AI 的突破性进展为人机交互开辟了激动人心的新可能性。高质量的生成模型和富有表现力的文本转语音（TTS）系统，使得构建更像对话伙伴而非工具的代理成为可能。

语音代理（Voice Agent）就是这样一个例子。您可以使用口语与其交互，而不是依赖键盘和鼠标输入，这是一种更自然、更有吸引力的交互方式。

---

## 什么是语音代理

**语音代理**是能够与用户进行自然语音对话的智能代理。这些代理结合了以下技术：

- 🎤 **语音识别**（Speech Recognition）
- 🧠 **自然语言处理**（NLP）
- 🤖 **生成式 AI**（Generative AI）
- 🔊 **文本转语音**（Text-to-Speech）

### 适用场景

语音代理特别适合以下使用场景：

1. **客户支持** - 自动化客服系统
2. **个人助理** - 语音助手
3. **免手操作界面** - 驾驶、烹饪等场景
4. **教练和培训** - 语言学习、技能培训

---

## 语音代理的工作原理

在高层次上，每个语音代理需要处理三个核心任务：

```
┌─────────┐      ┌─────────┐      ┌─────────┐
│  倾听   │  →   │  思考   │  →   │  说话   │
│ Listen  │      │  Think  │      │  Speak  │
└─────────┘      └─────────┘      └─────────┘
  捕获音频         解释意图          生成音频
  转录文本         推理规划          流式返回
```

1. **倾听（Listen）** - 捕获音频并转录为文本
2. **思考（Think）** - 理解意图、推理、规划
3. **说话（Speak）** - 生成音频并流式传输给用户

---

## 核心技术详解

### 1. 语音转文本（STT - Speech-to-Text）

#### 什么是 STT？

STT 是将人类语音转换为文本的技术。它是语音代理的"耳朵"。

#### STT 的工作流程

```
原始音频波形 → 音频特征提取 → 声学模型 → 语言模型 → 文本输出
     ↓              ↓              ↓           ↓          ↓
  PCM/WAV      梅尔频谱图        CTC/RNN    N-gram    "Hello World"
```

#### 关键概念

**音频格式**
- **PCM (Pulse Code Modulation)**: 原始音频格式，未压缩
- **采样率**: 每秒采样次数，常见值：16kHz、24kHz、48kHz
- **位深度**: 每个采样的比特数，常见值：16-bit

**转录模式**
- **实时流式转录**: 边说边转录，适合对话场景
- **批量转录**: 等待完整音频后转录，准确度更高

**事件类型**
- `stt_chunk`: 部分转录结果（实时反馈）
- `stt_output`: 最终转录结果（触发 Agent 处理）

#### 常见 STT 提供商

| 提供商 | 特点 | 延迟 | 准确度 |
|--------|------|------|--------|
| **AssemblyAI** | 实时流式，支持多语言 | 低 | 高 |
| **Deepgram** | 超低延迟，支持方言 | 极低 | 高 |
| **Google Cloud STT** | 成熟稳定，多语言支持 | 中 | 高 |
| **Whisper (OpenAI)** | 开源，多语言，高准确度 | 中-高 | 极高 |
| **Azure Speech** | 企业级，自定义模型 | 中 | 高 |

#### STT 实现要点

```python
# 生产者-消费者模式
async def stt_stream(audio_stream):
    """
    音频流 → STT 事件流
    
    并发处理：
    - 生产者：发送音频块到 STT 服务
    - 消费者：接收转录事件
    """
    stt = AssemblyAISTT(sample_rate=16000)
    
    # 后台任务：持续发送音频
    async def send_audio():
        async for audio_chunk in audio_stream:
            await stt.send_audio(audio_chunk)
        await stt.close()
    
    send_task = asyncio.create_task(send_audio())
    
    # 主流程：接收转录结果
    async for event in stt.receive_events():
        yield event
```

### 2. 文本转语音（TTS - Text-to-Speech）

#### 什么是 TTS？

TTS 是将文本转换为人类语音的技术。它是语音代理的"嘴巴"。

#### TTS 的演进历程

```
第一代：拼接式 TTS           → 机械、不自然
    ↓
第二代：参数化 TTS           → 更流畅，但缺乏表现力
    ↓
第三代：神经网络 TTS         → 自然、富有情感
 (Tacotron, WaveNet)
    ↓
第四代：Transformer TTS      → 高质量、低延迟
 (FastSpeech, VALL-E)
```

#### TTS 关键技术

**声学模型（Acoustic Model）**
- 将文本特征转换为声学特征（梅尔频谱图）
- 代表：Tacotron 2、FastSpeech

**声码器（Vocoder）**
- 将声学特征转换为音频波形
- 代表：WaveNet、HiFi-GAN、Vocoder

**韵律控制（Prosody）**
- 控制语音的节奏、重音、语调
- 使语音更自然、更有表现力

#### TTS 音频参数

**编码格式**
- **PCM**: 未压缩，高质量
- **MP3/AAC**: 压缩格式，节省带宽
- **Opus**: 低延迟，适合实时通信

**质量参数**
- **采样率**: 24kHz（电话质量）、48kHz（高质量）
- **比特率**: 影响音频质量和文件大小
- **声道**: 单声道（mono）适合语音

#### 常见 TTS 提供商

| 提供商 | 特点 | 延迟 | 质量 | 情感表现 |
|--------|------|------|------|---------|
| **Cartesia** | 流式合成，低延迟 | 极低 | 高 | 优秀 |
| **ElevenLabs** | 声音克隆，高质量 | 低 | 极高 | 极佳 |
| **OpenAI TTS** | 多种声音，稳定 | 中 | 高 | 良好 |
| **Google Cloud TTS** | 多语言，WaveNet | 中 | 高 | 良好 |
| **Azure Neural TTS** | 企业级，自定义声音 | 中 | 高 | 优秀 |

#### TTS 流式合成

```python
async def tts_stream(event_stream):
    """
    事件流 → 音频流
    
    并发处理：
    1. 上游处理：传递事件，发送文本到 TTS
    2. 音频接收：接收合成的音频块
    """
    tts = CartesiaTTS()
    
    async def process_upstream():
        async for event in event_stream:
            yield event  # 传递所有事件
            if event.type == "agent_chunk":
                # 将 Agent 生成的文本发送到 TTS
                await tts.send_text(event.text)
    
    # 合并两个异步流
    async for event in merge_async_iters(
        process_upstream(),      # 上游事件
        tts.receive_events()     # TTS 音频
    ):
        yield event
```

### 3. LangChain Agent（中间层）

#### Agent 的角色

Agent 是语音代理的"大脑"，负责：

- 🧠 **理解意图** - 解析用户需求
- 🤔 **推理规划** - 决定如何响应
- 🔧 **调用工具** - 执行具体操作
- 💾 **记忆管理** - 维护对话上下文

#### 流式响应

```python
# Agent 配置
agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[add_to_order, confirm_order],
    system_prompt="""你是一个友好的三明治店助手。
    你的目标是接收用户订单。保持简洁友好。
    不要使用表情符号、特殊字符或 markdown。
    你的回复将被文本转语音引擎读取。""",
    checkpointer=InMemorySaver(),  # 对话记忆
)

async def agent_stream(event_stream):
    thread_id = str(uuid4())  # 对话线程 ID
    
    async for event in event_stream:
        yield event  # 传递上游事件
        
        if event.type == "stt_output":
            # 流式处理 Agent 响应
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )
            
            async for message, _ in stream:
                if message.text:
                    yield AgentChunkEvent.create(message.text)
```

---

## 架构设计

### 两种主流架构

#### 架构一：三明治架构（STT → Agent → TTS）

```
用户语音 → [STT] → 文本 → [Agent] → 文本 → [TTS] → 合成语音
```

**优点** ✅
- 完全控制每个组件（可随时替换 STT/TTS 提供商）
- 访问最新的文本模态模型能力
- 透明行为，组件边界清晰

**缺点** ❌
- 需要协调多个服务
- 管道复杂度增加
- 语音→文本转换会丢失信息（如语调、情感）

#### 架构二：端到端语音架构（S2S）

```
用户语音 → [多模态模型] → 合成语音
```

**优点** ✅
- 更简单的架构，组件更少
- 简单交互的延迟更低
- 直接处理音频，保留语调等细节

**缺点** ❌
- 模型选择有限，供应商锁定风险高
- 功能可能落后于文本模态模型
- 音频处理透明度较低
- 可控性和定制性降低

### 本教程选择：三明治架构

我们选择三明治架构的原因：

1. **性能与控制的平衡** - 可实现低于 700ms 的延迟
2. **灵活性** - 可独立选择和优化每个组件
3. **可扩展性** - 易于添加新功能和集成

---

## 快速开始

### 环境要求

- Python 3.10+
- uv 包管理器（[安装 uv](https://docs.astral.sh/uv/getting-started/installation/)）
- LangChain 0.3+

### 安装依赖

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 激活虚拟环境（Linux/macOS）
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements.txt
```

### 环境变量设置

在项目根目录的 `.env` 文件中设置以下环境变量：

```bash
# LangSmith（可选，用于追踪）
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key

# STT 提供商
ASSEMBLYAI_API_KEY=your_assemblyai_api_key

# TTS 提供商
CARTESIA_API_KEY=your_cartesia_api_key

# LLM 提供商
ANTHROPIC_API_KEY=your_anthropic_api_key
# 或使用 OpenAI
OPENAI_API_KEY=your_openai_api_key

# 或使用 Qwen
QWEN_API_KEY=your_qwen_api_key
```

### 运行示例

```bash
# 运行完整示例
python voice_agent_example.py

# 运行简化版本（仅核心逻辑）
python voice_agent_simple.py
```

---

## 三大核心组件

### 组件 1：STT 流（Speech-to-Text Stream）

```python
async def stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    转换流：音频字节 → 语音事件
    
    使用生产者-消费者模式：
    - 生产者：读取音频块并发送到 AssemblyAI
    - 消费者：从 AssemblyAI 接收转录事件
    """
    stt = AssemblyAISTT(sample_rate=16000)
    
    async def send_audio():
        """后台任务：向 AssemblyAI 发送音频块"""
        try:
            async for audio_chunk in audio_stream:
                await stt.send_audio(audio_chunk)
        finally:
            await stt.close()
    
    # 在后台启动音频发送任务
    send_task = asyncio.create_task(send_audio())
    
    try:
        # 接收并生成转录事件
        async for event in stt.receive_events():
            yield event
    finally:
        send_task.cancel()
        await stt.close()
```

**关键点**：
- ⚡ 异步处理，边发送边接收
- 🔄 生产者-消费者模式，提高吞吐量
- 📝 支持部分转录（实时反馈）和最终转录

### 组件 2：Agent 流（Agent Stream）

```python
async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    转换流：语音事件 → 语音事件（附带 Agent 响应）
    
    传递所有上游事件，并在处理 STT 转录时添加 agent_chunk 事件。
    """
    thread_id = str(uuid4())  # 对话记忆的线程 ID
    
    async for event in event_stream:
        yield event  # 传递所有上游事件
        
        if event.type == "stt_output":
            # 流式处理 Agent 响应
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )
            
            async for message, _ in stream:
                if message.text:
                    yield AgentChunkEvent.create(message.text)
```

**关键点**：
- 🧠 流式响应，无需等待完整回复
- 💾 对话记忆，跨轮次维护状态
- 🔧 工具调用，执行具体操作

### 组件 3：TTS 流（Text-to-Speech Stream）

```python
async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    转换流：语音事件 → 语音事件（附带音频）
    
    合并两个并发流：
    1. process_upstream(): 传递事件并发送文本到 Cartesia
    2. tts.receive_events(): 从 Cartesia 生成音频块
    """
    tts = CartesiaTTS()
    
    async def process_upstream():
        """处理上游事件并发送 Agent 文本到 Cartesia"""
        async for event in event_stream:
            yield event
            if event.type == "agent_chunk":
                await tts.send_text(event.text)
    
    try:
        # 合并上游事件和 TTS 音频事件
        async for event in merge_async_iters(
            process_upstream(),
            tts.receive_events()
        ):
            yield event
    finally:
        await tts.close()
```

**关键点**：
- 🎵 流式合成，边生成边播放
- 🔀 并发处理，降低端到端延迟
- 📤 事件传递，便于监控和调试

---

## 完整实现

### 管道组合

```python
from langchain_core.runnables import RunnableGenerator

# 将三个阶段串联成完整管道
pipeline = (
    RunnableGenerator(stt_stream)      # 音频 → STT 事件
    | RunnableGenerator(agent_stream)  # STT 事件 → Agent 事件
    | RunnableGenerator(tts_stream)    # Agent 事件 → TTS 音频
)
```

### WebSocket 端点

```python
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点，用于实时双向通信"""
    await websocket.accept()
    
    async def websocket_audio_stream():
        """从 WebSocket 生成音频字节流"""
        while True:
            data = await websocket.receive_bytes()
            yield data
    
    # 通过管道转换音频
    output_stream = pipeline.atransform(websocket_audio_stream())
    
    # 将 TTS 音频发送回客户端
    async for event in output_stream:
        if event.type == "tts_chunk":
            await websocket.send_bytes(event.audio)

# 提供简单的 HTML 客户端
@app.get("/")
async def get_client():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Voice Agent</title>
        </head>
        <body>
            <h1>Voice Agent Demo</h1>
            <button id="start">Start Recording</button>
            <button id="stop">Stop Recording</button>
            <script src="/static/client.js"></script>
        </body>
    </html>
    """)
```

---

## 性能优化

### 延迟优化策略

1. **流式处理** 🌊
   - 每个组件边接收边处理，无需等待完整输入
   - 可实现低于 700ms 的端到端延迟

2. **并发处理** ⚡
   - STT、Agent、TTS 同时工作
   - 使用 asyncio 实现真正的并发

3. **提供商选择** 🎯
   - STT: AssemblyAI、Deepgram（超低延迟）
   - TTS: Cartesia、ElevenLabs（流式合成）

4. **网络优化** 🌐
   - WebSocket 保持长连接
   - 音频块大小平衡（100-200ms）
   - CDN 加速静态资源

### 质量优化策略

1. **音频质量** 🎵
   - 采样率：16kHz（STT）、24kHz+（TTS）
   - 降噪：客户端预处理
   - 编码：PCM（质量）vs Opus（带宽）

2. **转录准确度** 📝
   - 自定义词汇表（领域术语）
   - 上下文提示
   - 多候选结果

3. **对话质量** 💬
   - System Prompt 优化
   - 去除 Markdown 和特殊字符
   - 简洁自然的表达

---

## 高级主题

### 1. 多语言支持

```python
# STT 配置
stt = AssemblyAISTT(
    sample_rate=16000,
    language_code="zh",  # 中文
)

# TTS 配置
tts = CartesiaTTS(
    language="zh",
    voice_id="chinese-voice-id",
)
```

### 2. 情感识别

```python
# 从 STT 提取情感
async for event in stt.receive_events():
    if event.type == "stt_output":
        sentiment = event.sentiment  # 积极/消极/中性
        # 根据情感调整 Agent 行为
```

### 3. 打断处理

```python
# 检测用户打断
if user_is_speaking and agent_is_speaking:
    # 停止 Agent 输出
    await agent.cancel()
    await tts.stop()
    # 开始处理新的用户输入
```

### 4. 对话历史管理

```python
# 使用持久化 Checkpointer
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("conversations.db")

agent = create_agent(
    model="...",
    tools=[...],
    checkpointer=checkpointer,
)
```

---

## 测试和调试

### 1. 单元测试

```python
import pytest

@pytest.mark.asyncio
async def test_stt_stream():
    """测试 STT 流"""
    audio_stream = generate_test_audio()
    events = []
    
    async for event in stt_stream(audio_stream):
        events.append(event)
    
    assert len(events) > 0
    assert any(e.type == "stt_output" for e in events)
```

### 2. 集成测试

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    """测试完整管道"""
    audio_stream = load_test_audio("test_order.wav")
    
    output_events = []
    async for event in pipeline.atransform(audio_stream):
        output_events.append(event)
    
    # 验证包含所有阶段的事件
    assert any(e.type == "stt_output" for e in output_events)
    assert any(e.type == "agent_chunk" for e in output_events)
    assert any(e.type == "tts_chunk" for e in output_events)
```

### 3. 性能测试

```python
import time

async def measure_latency():
    """测量端到端延迟"""
    start_time = time.time()
    
    # 发送音频
    audio_stream = ...
    
    # 等待第一个 TTS 音频块
    async for event in pipeline.atransform(audio_stream):
        if event.type == "tts_chunk":
            latency = (time.time() - start_time) * 1000
            print(f"首字节延迟: {latency:.0f}ms")
            break
```

---

## 部署建议

### 1. 本地开发

```bash
# 使用 uvicorn 运行
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 生产部署

```bash
# 使用 Gunicorn + Uvicorn workers
gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000
```

### 3. Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 4. 扩展性考虑

- 负载均衡：Nginx/HAProxy
- 会话粘性：基于用户 ID 路由
- 监控：Prometheus + Grafana
- 日志：ELK Stack

---

## 常见问题

### Q1: 如何降低延迟？

**A**: 
1. 选择低延迟的 STT/TTS 提供商
2. 使用流式处理，避免等待完整输入
3. 优化音频块大小（100-200ms）
4. 部署在靠近用户的区域

### Q2: 如何处理网络断线？

**A**:
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        # ... 正常处理 ...
    except WebSocketDisconnect:
        # 清理资源
        await cleanup_session(session_id)
```

### Q3: 如何支持多用户并发？

**A**:
- 每个 WebSocket 连接独立处理
- 使用异步 I/O，支持数千并发连接
- 考虑使用队列系统（如 Redis）分发任务

### Q4: 音频质量差怎么办？

**A**:
1. 提高采样率（24kHz → 48kHz）
2. 客户端降噪预处理
3. 使用更高质量的 TTS 声音
4. 检查网络带宽和稳定性

---

## 扩展资源

### 官方文档
- [LangChain Voice Agent Guide](https://docs.langchain.com/oss/python/langchain/voice-agent)
- [LangChain Agents](https://docs.langchain.com/docs/concepts/agents)
- [LangGraph Documentation](https://langgraph.readthedocs.io/)

### STT/TTS 提供商文档
- [AssemblyAI Real-time API](https://www.assemblyai.com/docs/api-reference/realtime)
- [Cartesia TTS API](https://docs.cartesia.ai/)
- [Deepgram Streaming](https://developers.deepgram.com/docs/streaming)
- [ElevenLabs API](https://docs.elevenlabs.io/)

### 相关项目
- [voice-sandwich-demo](https://github.com/langchain-ai/voice-sandwich-demo) - 官方示例
- [Whisper](https://github.com/openai/whisper) - 开源 STT 模型
- [Coqui TTS](https://github.com/coqui-ai/TTS) - 开源 TTS 引擎

---

## 下一步

完成本教程后，您可以：

1. ✅ 理解语音代理的工作原理
2. ✅ 掌握 STT、TTS 核心技术
3. ✅ 实现三明治架构的语音代理
4. ✅ 优化性能和质量
5. ✅ 部署到生产环境

**继续学习**：
- [构建 RAG Agent](../rag_agent/README.md)
- [语义搜索引擎](../semantic_search/README.md)
- [LangSmith Observability](../langsmith_observability/langsmith_observability_quickstart.md)

---

## 总结

语音代理代表了人机交互的未来方向。通过结合 STT、LangChain Agent 和 TTS 技术，我们可以创建自然、高效的对话体验。

**关键要点**：
- 🎯 **三明治架构**提供最佳的控制性和灵活性
- ⚡ **流式处理**是实现低延迟的关键
- 🧠 **LangChain Agent**提供强大的推理和工具调用能力
- 🎵 **音频质量**直接影响用户体验

开始构建您自己的语音代理吧！🚀
