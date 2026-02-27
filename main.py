from llm_client import Message
from llm_client.client import LLMClient









llm_client = LLMClient.from_env()

kwargs = {
    "temperature": 0.3,
    "max_tokens": 4096,
    "response_format": {"type": "json_object"}
}

messages = [
    Message(role="system", content="你是一个智能助手。"),
    Message(role="user", content="给我讲一个笑话")
]




async def call_llm_example():
    response = await llm_client.chat(
        messages=messages,
        model="glm-4.7",
        provider="zhipu",
        tools=None,
        thinking=None,
        **kwargs,
    )
    print(response)

async def call_llm_stream_example():


    # 流式调用
    async for chunk in llm_client.chat_stream(
            messages=messages,
            model="glm-4.7",
            temperature=0.7
    ):
        # 如果有思考内容
        if chunk.thinking_delta:
            print(chunk.thinking_delta, end="", flush=True)
        # 打印内容片段（不换行）
        if chunk.delta:
            print(chunk.delta, end="", flush=True)








if __name__ == "__main__":

    import asyncio
    # asyncio.run(call_llm_example())
    asyncio.run(call_llm_stream_example())













