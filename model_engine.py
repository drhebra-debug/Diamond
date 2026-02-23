import httpx
import asyncio


LOCAL_CLAUDE_URL = "http://localhost:11434/v1/messages"


async def run_model_stream(model, system, messages, tools=None):

    payload = {
        "model": model,
        "system": system,
        "messages": messages,
        "stream": True
    }

    if tools:
        payload["tools"] = tools

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", LOCAL_CLAUDE_URL, json=payload) as response:
            async for line in response.aiter_lines():
                if line:
                    yield line
