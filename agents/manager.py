import time
import uuid
import json
import logging
import threading
import os
import re
from collections import OrderedDict
from typing import List, Dict, Optional
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)

# ====================== MODEL PATHS (example) ======================
MODEL_PATHS: Dict[str, str] = {
    "claude-3.7-sonnet-reasoning-gemma3-12b": "/mnt/storage/webui/models/claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0_2.gguf",
    "phi-4-mini-reasoning": "/mnt/storage/webui/models/Phi-4-mini-reasoning-Q4_K_M.gguf",
    "codegeex4-all-9b": "/mnt/storage/webui/models/codegeex4-all-9b-Q4_K_M.gguf",
}

MAX_LOADED_MODELS = int(os.getenv("MAX_LOADED_MODELS", 3))
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", 900))

model_cache: OrderedDict = OrderedDict()
model_last_used: Dict[str, float] = {}
lock = threading.Lock()

def unload_model(name: str):
    if name in model_cache:
        logging.info(f"🔄 Unloading model: {name}")
        del model_cache[name]
        del model_last_used[name]

def cleanup_models():
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            for name in list(model_last_used.keys()):
                if now - model_last_used[name] > MODEL_TIMEOUT:
                    unload_model(name)

def load_model(name: str) -> Llama:
    if name not in MODEL_PATHS:
        raise ValueError(f"Model '{name}' not found")
    with lock:
        if name in model_cache:
            model_last_used[name] = time.time()
            model_cache.move_to_end(name)
            return model_cache[name]
        if len(model_cache) >= MAX_LOADED_MODELS:
            oldest = next(iter(model_cache))
            unload_model(oldest)
        logging.info(f"🚀 Loading model: {name}")
        llm = Llama(
            model_path=MODEL_PATHS[name],
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=1024,
            verbose=False,
        )
        model_cache[name] = llm
        model_last_used[name] = time.time()
        model_cache.move_to_end(name)
        return llm

threading.Thread(target=cleanup_models, daemon=True).start()
