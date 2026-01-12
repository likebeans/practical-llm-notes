"""
LangChain Voice Agent 简化示例
展示核心概念，去除复杂的实现细节

核心思想：
音频 → STT（语音转文本）→ Agent（处理）→ TTS（文本转语音）→ 音频
"""

import asyncio
from typing import AsyncIterator

# ============================================================================
# 简化的三个核心函数
# ============================================================================

async def speech_to_text(audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """
    第一步：语音转文本（STT）
    输入：音频流
    输出：文本流
    """
    print("\n[STT] 开始语音识别...")
    
    # 模拟：收集音频 → 转换为文本
    audio_chunks = []
    async for chunk in audio_stream:
        audio_chunks.append(chunk)
        print(f"   接收音频块: {len(chunk)} 字节")
    
    # 模拟转录结果
    transcripts = [
        "你好",
        "我想要一个三明治",
        "谢谢"
    ]
    
    for text in transcripts:
        print(f"   [OK] 转录: {text}")
        yield text
        await asyncio.sleep(0.5)


async def agent_process(text_stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    第二步：智能处理（Agent）
    输入：文本流
    输出：响应文本流
    """
    print("\n[Agent] 开始智能处理...")
    
    async for text in text_stream:
        print(f"   收到用户输入: {text}")
        
        # 模拟 Agent 思考和响应
        if "你好" in text:
            response = "你好！欢迎来到我们的三明治店。"
        elif "三明治" in text:
            response = "好的，我们有金枪鱼、火腿和素食三明治，您想要哪一种？"
        elif "谢谢" in text:
            response = "不客气，祝您用餐愉快！"
        else:
            response = "请问还有什么需要帮助的吗？"
        
        # 流式生成响应（逐词）
        words = response.split()
        for word in words:
            print(f"   [Gen] 生成: {word}", end=" ", flush=True)
            yield word + " "
            await asyncio.sleep(0.1)
        print()


async def text_to_speech(text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """
    第三步：文本转语音（TTS）
    输入：文本流
    输出：音频流
    """
    print("\n[TTS] 开始语音合成...")
    
    async for text in text_stream:
        # 模拟：文本 → 音频
        audio = f"[语音:{text.strip()}]".encode()
        print(f"   [Audio] 合成音频: {len(audio)} 字节")
        yield audio
        await asyncio.sleep(0.05)


# ============================================================================
# 组合成完整管道
# ============================================================================

async def voice_agent_pipeline():
    """完整的语音代理管道"""
    
    print("=" * 80)
    print("语音代理管道 - 简化示例")
    print("=" * 80)
    
    # 模拟音频输入
    async def mock_audio_input():
        """模拟从麦克风接收音频"""
        print("\n[Input] 开始接收音频...")
        for i in range(3):
            audio = f"audio_chunk_{i}".encode()
            yield audio
            await asyncio.sleep(0.2)
        print("   [Stop] 音频输入结束")
    
    # 构建管道
    audio_input = mock_audio_input()
    text_output = speech_to_text(audio_input)     # STT
    response_text = agent_process(text_output)     # Agent
    audio_output = text_to_speech(response_text)   # TTS
    
    # 处理最终输出
    print("\n[Output] 输出音频...")
    audio_count = 0
    async for audio_chunk in audio_output:
        audio_count += 1
        # 在真实应用中，这里会播放音频
        print(f"   [Play] 播放音频块 {audio_count}")
    
    print("\n" + "=" * 80)
    print("[OK] 管道执行完成！")
    print(f"共生成 {audio_count} 个音频块")
    print("=" * 80)


# ============================================================================
# 运行示例
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Voice Agent 简化示例")
    print("=" * 80)
    
    asyncio.run(voice_agent_pipeline())
    
    print("""
    
核心概念总结：

1. 三个阶段
   音频 → STT → Agent → TTS → 音频
   
2. 流式处理
   每个阶段边接收边处理，不需要等待上游完成
   
3. 异步并发
   使用 async/await 实现高效的 I/O 处理
   
4. 管道组合
   简单地将三个函数串联起来

下一步：
- 查看 voice_agent_example.py 了解详细实现
- 阅读 README.md 了解 STT、TTS 等技术细节
- 集成真实的 API（AssemblyAI、Cartesia 等）
    """)
