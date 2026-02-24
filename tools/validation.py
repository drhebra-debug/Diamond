from typing import Dict, Any
from pathlib import Path
from fastapi import HTTPException

ALLOWED_MODEL_SUFFIXES = {".gguf", ".bin", ".pt", ".pth"}


def is_model_path_valid(name: str, model_paths: Dict[str, str]) -> bool:
    # named model
    if name in model_paths:
        return True
    p = Path(name)
    return p.is_file() and p.suffix in ALLOWED_MODEL_SUFFIXES


def validate_request_payload(payload: Dict[str, Any], model_paths: Dict[str, str]):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    model = payload.get('model')
    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model' in payload")

    if not is_model_path_valid(model, model_paths):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found or invalid")

    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="'messages' must be a non-empty list")

    # Basic message shape check
    for m in messages:
        if not isinstance(m, dict) or 'role' not in m or 'content' not in m:
            raise HTTPException(status_code=400, detail="Each message must be an object with 'role' and 'content'")

    max_tokens = payload.get('max_tokens', 4096)
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise HTTPException(status_code=400, detail="Invalid 'max_tokens' value")

    temperature = payload.get('temperature', 0.7)
    if not (isinstance(temperature, (int, float)) and 0.0 <= float(temperature) <= 2.0):
        raise HTTPException(status_code=400, detail="Invalid 'temperature' value")

    return True
