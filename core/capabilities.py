MODEL_CAPABILITIES = {
    "default": {
        "tools": True,
        "streaming": True,
        "vision": False,
        "max_tokens": 8192
    }
}

def get_capabilities(model_name):
    return MODEL_CAPABILITIES.get(model_name, MODEL_CAPABILITIES["default"])
