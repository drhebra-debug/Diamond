import asyncio
import json
import uuid
from typing import AsyncGenerator, Dict, Any

from model_engine import run_model_stream
from tool_registry import resolve_tools


class ClaudeStreamFormatter:

    @staticmethod
    async def stream_text(generator: AsyncGenerator[str, None]):
        message_id = str(uuid.uuid4())

        yield {
            "type": "message_start",
            "message": {
                "id": message_id,
                "role": "assistant"
            }
        }

        yield {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": ""
            }
        }

        async for chunk in generator:
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": chunk
                }
            }

        yield {
            "type": "content_block_stop",
            "index": 0
        }

        yield {
            "type": "message_stop"
        }


class AgentExecutor:

    def __init__(self, model_name="diamond"):
        self.model_name = model_name

    async def run(self, agent_config: Dict[str, Any], messages):
        tools = resolve_tools(agent_config)

        async def generator():
            async for token in run_model_stream(
                model=self.model_name,
                system=agent_config["prompt"],
                messages=messages,
                tools=tools
            ):
                yield token

        return ClaudeStreamFormatter.stream_text(generator())
