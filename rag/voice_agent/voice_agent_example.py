"""
LangChain Voice Agent 完整示例
基于官方文档: https://docs.langchain.com/oss/python/langchain/voice-agent

本示例演示如何：
1. 实现 STT（语音转文本）流
2. 创建 LangChain Agent 处理流
3. 实现 TTS（文本转语音）流
4. 组合完整的语音代理管道

注意：这是一个教学示例，展示核心概念和架构
完整的生产实现请参考：https://github.com/langchain-ai/voice-sandwich-demo
"""

import os
import asyncio
from typing import AsyncIterator, Optional, List
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装 python-dotenv 可以自动从 .env 文件加载环境变量")
    print("安装命令: uv pip install python-dotenv")

# ============================================================================
# 第一部分：事件系统定义
# ============================================================================

print("=" * 80)
print("第一部分：Voice Agent 事件系统")
print("=" * 80)

class EventType(str, Enum):
    """事件类型枚举"""
    STT_CHUNK = "stt_chunk"          # STT 部分转录结果
    STT_OUTPUT = "stt_output"        # STT 最终转录结果
    AGENT_CHUNK = "agent_chunk"      # Agent 响应块
    TTS_CHUNK = "tts_chunk"          # TTS 音频块

@dataclass
class VoiceAgentEvent:
    """语音代理事件基类"""
    type: EventType
    timestamp: float
    
@dataclass
class STTChunkEvent(VoiceAgentEvent):
    """STT 部分转录事件"""
    partial_transcript: str
    confidence: Optional[float] = None
    
    @classmethod
    def create(cls, text: str, confidence: Optional[float] = None):
        import time
        return cls(
            type=EventType.STT_CHUNK,
            timestamp=time.time(),
            partial_transcript=text,
            confidence=confidence,
        )

@dataclass
class STTOutputEvent(VoiceAgentEvent):
    """STT 最终转录事件"""
    transcript: str
    confidence: Optional[float] = None
    
    @classmethod
    def create(cls, text: str, confidence: Optional[float] = None):
        import time
        return cls(
            type=EventType.STT_OUTPUT,
            timestamp=time.time(),
            transcript=text,
            confidence=confidence,
        )

@dataclass
class AgentChunkEvent(VoiceAgentEvent):
    """Agent 响应块事件"""
    text: str
    
    @classmethod
    def create(cls, text: str):
        import time
        return cls(
            type=EventType.AGENT_CHUNK,
            timestamp=time.time(),
            text=text,
        )

@dataclass
class TTSChunkEvent(VoiceAgentEvent):
    """TTS 音频块事件"""
    audio: bytes
    
    @classmethod
    def create(cls, audio: bytes):
        import time
        return cls(
            type=EventType.TTS_CHUNK,
            timestamp=time.time(),
            audio=audio,
        )

print("""
[OK] 定义了 4 种事件类型：
  1. STT_CHUNK - 实时部分转录（边说边显示）
  2. STT_OUTPUT - 最终转录结果（触发 Agent）
  3. AGENT_CHUNK - Agent 响应文本块（流式生成）
  4. TTS_CHUNK - 合成的音频块（流式播放）

这些事件在管道中流动，连接 STT → Agent → TTS
""")

# ============================================================================
# 第二部分：STT（语音转文本）实现
# ============================================================================

print("\n" + "=" * 80)
print("第二部分：STT（Speech-to-Text）实现")
print("=" * 80)

class MockAssemblyAISTT:
    """
    模拟 AssemblyAI STT 客户端
    
    在真实实现中，这会连接到 AssemblyAI 的 WebSocket API
    文档：https://www.assemblyai.com/docs/api-reference/realtime
    """
    
    def __init__(self, sample_rate: int = 16000, api_key: Optional[str] = None):
        self.sample_rate = sample_rate
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "mock_key")
        self.is_closed = False
        
    async def send_audio(self, audio_chunk: bytes) -> None:
        """发送音频块到 STT 服务"""
        # 在真实实现中，这会通过 WebSocket 发送 PCM 音频数据
        print(f"  [Out] 发送音频块: {len(audio_chunk)} 字节")
        await asyncio.sleep(0.01)  # 模拟网络延迟
        
    async def receive_events(self) -> AsyncIterator[VoiceAgentEvent]:
        """接收 STT 转录事件"""
        # 模拟转录过程
        # 真实实现会从 WebSocket 接收 JSON 消息并解析
        
        # 模拟部分转录
        partial_texts = ["我想", "我想要", "我想要一个"]
        for text in partial_texts:
            yield STTChunkEvent.create(text, confidence=0.85)
            await asyncio.sleep(0.1)
        
        # 模拟最终转录
        yield STTOutputEvent.create("我想要一个金枪鱼三明治", confidence=0.95)
        
    async def close(self):
        """关闭 STT 连接"""
        self.is_closed = True
        print("  [Close] 关闭 STT 连接")


async def stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    STT 流转换：音频字节 → 语音事件
    
    使用生产者-消费者模式：
    - 生产者：持续发送音频块到 STT 服务
    - 消费者：接收并生成转录事件
    
    这种模式允许边发送音频边接收转录结果，降低延迟
    """
    print("\n[MIC] 启动 STT 流...")
    
    stt = MockAssemblyAISTT(sample_rate=16000)
    
    async def send_audio():
        """后台任务：持续发送音频到 STT"""
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
            if event.type == EventType.STT_CHUNK:
                print(f"  [Text] 部分转录: {event.partial_transcript}")
            elif event.type == EventType.STT_OUTPUT:
                print(f"  [OK] 最终转录: {event.transcript}")
            yield event
    finally:
        # 清理
        send_task.cancel()
        try:
            await send_task
        except asyncio.CancelledError:
            pass
        await stt.close()

print("""
[OK] STT 流实现要点：

1. 生产者-消费者模式
   - 音频发送和转录接收并发进行
   - 提高吞吐量，降低延迟

2. 事件类型
   - STT_CHUNK: 实时部分转录（用户体验反馈）
   - STT_OUTPUT: 最终转录（触发 Agent 处理）

3. 配置参数
   - sample_rate: 音频采样率（通常 16kHz）
   - language_code: 语言代码（如 "zh" 中文）
   - punctuate: 是否自动添加标点
""")

# ============================================================================
# 第三部分：Agent（智能代理）实现
# ============================================================================

print("\n" + "=" * 80)
print("第三部分：LangChain Agent 实现")
print("=" * 80)

# 定义 Agent 工具
def add_to_order(item: str, quantity: int = 1) -> str:
    """添加商品到订单"""
    return f"已添加 {quantity} 个 {item} 到订单"

def confirm_order() -> str:
    """确认订单"""
    return "订单已确认，正在准备中"

def cancel_order() -> str:
    """取消订单"""
    return "订单已取消"

print("""
[Tool] 定义了 3 个工具函数：
  1. add_to_order - 添加商品到订单
  2. confirm_order - 确认订单
  3. cancel_order - 取消订单

这些工具让 Agent 能够执行具体操作
""")

class MockAgent:
    """
    模拟 LangChain Agent
    
    在真实实现中，使用 create_agent() 或 LangGraph 创建
    """
    
    def __init__(self, tools: List, system_prompt: str):
        self.tools = tools
        self.system_prompt = system_prompt
        self.conversation_history = {}
        
    async def astream(
        self,
        input_dict: dict,
        config: dict,
        stream_mode: str = "messages",
    ):
        """流式处理 Agent 响应"""
        thread_id = config.get("configurable", {}).get("thread_id")
        messages = input_dict.get("messages", [])
        
        if not messages:
            return
        
        user_message = messages[0].content
        print(f"  [Brain] Agent 收到: {user_message}")
        
        # 模拟 Agent 思考和响应
        # 真实实现会调用 LLM 并可能使用工具
        response_text = f"好的，我已经记录了您的需求：{user_message}。请问还需要其他的吗？"
        
        # 流式生成响应（模拟逐词生成）
        words = response_text.split()
        for i, word in enumerate(words):
            # 模拟 LangChain 的消息格式
            class MockMessage:
                def __init__(self, text):
                    self.text = text
            
            yield (MockMessage(word + " "), None)
            await asyncio.sleep(0.05)  # 模拟生成延迟


async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Agent 流转换：语音事件 → 语音事件（附带 Agent 响应）
    
    传递所有上游事件，并在收到 STT_OUTPUT 时处理并生成 Agent 响应
    """
    print("\n[Brain] 启动 Agent 流...")
    
    # 创建 Agent（在真实实现中使用 LangChain）
    from langchain_core.messages import HumanMessage
    
    agent = MockAgent(
        tools=[add_to_order, confirm_order, cancel_order],
        system_prompt="""你是一个友好的三明治店助手。
        你的目标是帮助客户点餐。保持简洁友好。
        不要使用表情符号、特殊字符或 markdown。
        你的回复将被文本转语音引擎读取。""",
    )
    
    # 为这个对话创建唯一线程 ID（用于记忆）
    thread_id = str(uuid4())
    print(f"  [Doc] 对话线程 ID: {thread_id[:8]}...")
    
    async for event in event_stream:
        # 传递所有上游事件
        yield event
        
        # 只处理最终转录结果
        if event.type == EventType.STT_OUTPUT:
            print(f"  [Msg] 处理用户输入: {event.transcript}")
            
            # 构造消息
            message = HumanMessage(content=event.transcript)
            
            # 流式处理 Agent 响应
            stream = agent.astream(
                {"messages": [message]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )
            
            # 生成 Agent 响应块
            async for message, _ in stream:
                if message.text:
                    print(f"  [Msg] Agent 响应: {message.text}", end="", flush=True)
                    yield AgentChunkEvent.create(message.text)
            
            print()  # 换行

print("""
[OK] Agent 流实现要点：

1. 流式响应
   - 边生成边发送，无需等待完整回复
   - 用户体验更好，延迟更低

2. 对话记忆
   - 使用 thread_id 维护对话状态
   - checkpointer 持久化对话历史

3. 工具调用
   - Agent 可以调用预定义的工具函数
   - 执行具体操作（如添加订单）

4. System Prompt 优化
   - 明确角色和任务
   - 去除不适合 TTS 的格式（emoji、markdown）
   - 保持简洁自然的表达
""")

# ============================================================================
# 第四部分：TTS（文本转语音）实现
# ============================================================================

print("\n" + "=" * 80)
print("第四部分：TTS（Text-to-Speech）实现")
print("=" * 80)

class MockCartesiaTTS:
    """
    模拟 Cartesia TTS 客户端
    
    在真实实现中，这会连接到 Cartesia 的 WebSocket API
    文档：https://docs.cartesia.ai/
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "default",
        model_id: str = "sonic-3",
        sample_rate: int = 24000,
    ):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY", "mock_key")
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.is_closed = False
        
    async def send_text(self, text: str) -> None:
        """发送文本到 TTS 服务进行合成"""
        if not text or not text.strip():
            return
        
        print(f"  [Out] 发送文本到 TTS: {text.strip()}")
        await asyncio.sleep(0.01)  # 模拟网络延迟
        
    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        """接收 TTS 合成的音频块"""
        # 模拟音频合成过程
        # 真实实现会从 WebSocket 接收 base64 编码的音频数据
        
        for i in range(3):
            # 模拟音频数据（在真实实现中是 PCM 音频字节）
            mock_audio = f"audio_chunk_{i}".encode()
            print(f"  [Speaker] 接收音频块 {i+1}: {len(mock_audio)} 字节")
            yield TTSChunkEvent.create(mock_audio)
            await asyncio.sleep(0.1)  # 模拟合成延迟
    
    async def close(self):
        """关闭 TTS 连接"""
        self.is_closed = True
        print("  [Close] 关闭 TTS 连接")


async def merge_async_iters(*iters: AsyncIterator) -> AsyncIterator:
    """
    合并多个异步迭代器
    
    这是一个工具函数，用于同时从多个异步流中接收数据
    """
    import asyncio
    from typing import Any
    
    queue: asyncio.Queue = asyncio.Queue()
    
    async def producer(it: AsyncIterator):
        """将迭代器的项放入队列"""
        try:
            async for item in it:
                await queue.put(item)
        finally:
            await queue.put(None)  # 结束标记
    
    # 启动所有生产者
    tasks = [asyncio.create_task(producer(it)) for it in iters]
    active_count = len(tasks)
    
    try:
        while active_count > 0:
            item = await queue.get()
            if item is None:
                active_count -= 1
            else:
                yield item
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    TTS 流转换：语音事件 → 语音事件（附带音频）
    
    合并两个并发流：
    1. 上游事件处理 - 传递事件并发送 Agent 文本到 TTS
    2. TTS 音频接收 - 接收合成的音频块
    
    这种并发处理方式使得 TTS 可以边接收文本边合成音频
    """
    print("\n[Speaker] 启动 TTS 流...")
    
    tts = MockCartesiaTTS()
    
    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        """处理上游事件并发送文本到 TTS"""
        async for event in event_stream:
            # 传递所有事件
            yield event
            
            # 将 Agent 文本发送到 TTS
            if event.type == EventType.AGENT_CHUNK:
                await tts.send_text(event.text)
    
    try:
        # 合并上游事件流和 TTS 音频流
        # 两个流并发运行
        async for event in merge_async_iters(
            process_upstream(),
            tts.receive_events()
        ):
            if event.type == EventType.TTS_CHUNK:
                print(f"  [OK] 音频块就绪")
            yield event
    finally:
        await tts.close()

print("""
[OK] TTS 流实现要点：

1. 流式合成
   - 边接收文本边合成音频
   - 无需等待完整文本，降低延迟

2. 并发处理
   - 上游事件处理和音频接收同时进行
   - 使用 merge_async_iters 合并流

3. 音频格式
   - 采样率：24kHz（高质量）或 16kHz（节省带宽）
   - 编码：PCM（未压缩）或 Opus（压缩）
   - 声道：单声道（mono）适合语音

4. 质量优化
   - 选择合适的声音模型
   - 调整语速和音调
   - 支持情感表达
""")

# ============================================================================
# 第五部分：完整管道组合
# ============================================================================

print("\n" + "=" * 80)
print("第五部分：完整 Voice Agent 管道")
print("=" * 80)

async def run_voice_agent_pipeline():
    """
    运行完整的 Voice Agent 管道
    
    管道流程：
    音频流 → STT 流 → Agent 流 → TTS 流 → 音频输出
    """
    
    print("\n[Start] 启动完整 Voice Agent 管道...\n")
    
    # 模拟音频输入流
    async def mock_audio_stream() -> AsyncIterator[bytes]:
        """模拟从麦克风捕获的音频流"""
        print("[MIC] 开始接收音频...")
        # 模拟 5 个音频块
        for i in range(5):
            audio_chunk = f"audio_chunk_{i}".encode()
            print(f"  [In] 接收音频块 {i+1}")
            yield audio_chunk
            await asyncio.sleep(0.1)
        print("  [Stop] 音频输入结束\n")
    
    # 构建管道：STT → Agent → TTS
    audio_input = mock_audio_stream()
    
    # 第一阶段：STT
    stt_events = stt_stream(audio_input)
    
    # 第二阶段：Agent
    agent_events = agent_stream(stt_events)
    
    # 第三阶段：TTS
    tts_events = tts_stream(agent_events)
    
    # 收集输出
    print("\n[Stats] 处理管道输出...")
    print("-" * 80)
    
    event_counts = {
        EventType.STT_CHUNK: 0,
        EventType.STT_OUTPUT: 0,
        EventType.AGENT_CHUNK: 0,
        EventType.TTS_CHUNK: 0,
    }
    
    async for event in tts_events:
        event_counts[event.type] += 1
        
        # 这里可以处理最终输出
        if event.type == EventType.TTS_CHUNK:
            # 在真实应用中，这里会将音频发送到客户端播放
            pass
    
    print("-" * 80)
    print("\n[Chart] 管道统计：")
    print(f"  STT 部分转录: {event_counts[EventType.STT_CHUNK]} 个")
    print(f"  STT 最终转录: {event_counts[EventType.STT_OUTPUT]} 个")
    print(f"  Agent 响应块: {event_counts[EventType.AGENT_CHUNK]} 个")
    print(f"  TTS 音频块: {event_counts[EventType.TTS_CHUNK]} 个")
    
    print("\n[OK] 管道执行完成！")

print("""
[OK] 完整管道要点：

1. 管道架构
   音频输入 → STT → Agent → TTS → 音频输出
   
2. 流式处理
   - 每个阶段边接收边处理
   - 无需等待上游完成
   - 端到端延迟最小化

3. 事件流动
   - STT_CHUNK: 实时反馈
   - STT_OUTPUT: 触发 Agent
   - AGENT_CHUNK: 流式生成
   - TTS_CHUNK: 音频播放

4. 并发执行
   - STT、Agent、TTS 同时工作
   - 使用 asyncio 实现真正的并发
""")

# ============================================================================
# 第六部分：运行示例
# ============================================================================

print("\n" + "=" * 80)
print("第六部分：运行示例")
print("=" * 80)

if __name__ == "__main__":
    print("\n" + "===" * 40)
    print("开始运行 Voice Agent 完整示例")
    print("===" * 40 + "\n")
    
    # 运行异步管道
    asyncio.run(run_voice_agent_pipeline())
    
    print("\n" + "=" * 80)
    print("教程总结")
    print("=" * 80)
    print("""
您已经完成了 LangChain Voice Agent 的完整示例！

本示例涵盖了：
1. [OK] 事件系统 - 定义管道中的数据流
2. [OK] STT 流 - 语音转文本（生产者-消费者模式）
3. [OK] Agent 流 - 智能处理和响应（流式生成）
4. [OK] TTS 流 - 文本转语音（并发合成）
5. [OK] 完整管道 - 端到端的语音对话系统

核心概念：
- === 三明治架构：STT → Agent → TTS
- [Fast] 流式处理：边接收边处理，降低延迟
- [Loop] 异步并发：充分利用 I/O 等待时间
- [Stats] 事件驱动：清晰的数据流动和状态管理

下一步建议：
- 集成真实的 STT 提供商（AssemblyAI、Deepgram）
- 集成真实的 TTS 提供商（Cartesia、ElevenLabs）
- 实现 WebSocket 服务器端点
- 构建前端客户端界面
- 优化性能和延迟
- 添加错误处理和重试机制

完整生产实现参考：
https://github.com/langchain-ai/voice-sandwich-demo

更多资源：
- LangChain Agents: https://docs.langchain.com/docs/concepts/agents
- LangGraph: https://langgraph.readthedocs.io/
- AssemblyAI: https://www.assemblyai.com/docs/
- Cartesia: https://docs.cartesia.ai/
    """)
