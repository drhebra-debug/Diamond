import json
from tool_registry import TOOLS


async def execute_tool_call(tool_name, arguments):
    if tool_name not in TOOLS:
        raise Exception(f"Tool {tool_name} not found")

    tool_fn = TOOLS[tool_name]["function"]
    return await tool_fn(**arguments)


async def tool_execution_loop(model_stream, send_to_model):

    async for event in model_stream:

        if event.get("type") == "tool_use":

            tool_name = event["name"]
            args = event["input"]

            result = await execute_tool_call(tool_name, args)

            await send_to_model({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result)
            })

        else:
            yield event
