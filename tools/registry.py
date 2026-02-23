import inspect
import pkgutil
import importlib

TOOL_REGISTRY = {}

def register_tool(name, schema):
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "function": fn,
            "schema": schema
        }
        return fn
    return decorator


def auto_discover_tools(package="tools"):
    for _, module_name, _ in pkgutil.iter_modules([package]):
        importlib.import_module(f"{package}.{module_name}")
