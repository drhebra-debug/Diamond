import inspect
import json
from typing import Callable, Dict


TOOLS = {}


def tool(func: Callable):
    schema = {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    sig = inspect.signature(func)

    for name, param in sig.parameters.items():
        schema["input_schema"]["properties"][name] = {
            "type": "string"
        }
        schema["input_schema"]["required"].append(name)

    TOOLS[func.__name__] = {
        "schema": schema,
        "function": func
    }

    return func


def resolve_tools(agent_config):
    if "tools" not in agent_config:
        return None

    return [
        TOOLS[t]["schema"]
        for t in agent_config["tools"]
        if t in TOOLS
    ]
