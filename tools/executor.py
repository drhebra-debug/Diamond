from concurrent.futures import ThreadPoolExecutor
from .registry import TOOL_REGISTRY
from .schemas import validate_tool_input

def execute_tool_call(tool_call):
    name = tool_call["name"]
    input_data = tool_call["input"]

    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return {"error": f"Tool {name} not found"}

    validate_tool_input(tool["schema"], input_data)

    return tool["function"](**input_data)


def execute_parallel(tool_calls):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(execute_tool_call, tool_calls))
