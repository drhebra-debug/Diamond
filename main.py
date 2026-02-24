from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import json
import uuid
import time

from diamond.core.usage import UsageTracker
from diamond.core.headers import apply_standard_headers
from diamond.core.errors import claude_error
from diamond.core.fingerprint import generate_fingerprint
from diamond.core.capabilities import get_capabilities

from diamond.tools.router import tool_execution_loop
from diamond.tools.registry import auto_discover_tools



from autonomy.agent_loop import AutonomousCoder
from collaboration.orchestrator import MultiAgentOrchestrator
import json
import uuid
import time
import logging
import threading
import subprocess
import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from collections import OrderedDict
import hashlib
import csv
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
import psycopg2
import redis
from pydantic import BaseModel
try:
    from llama_cpp import Llama
except Exception:
    Llama = None
auto_discover_tools()
# ==========================================================
# CONFIG
# ==========================================================
HOST = "0.0.0.0"
PORT = 11434
N_CTX = 8192
N_THREADS = 12
RAG_TOP_K = 8

LOG_FOLDER = Path("conversation_logs")
LOG_FOLDER.mkdir(exist_ok=True)
PROJECT_ROOT = Path(__file__).parent.absolute()

MAIN_MODEL_NAME = os.getenv("MAIN_MODEL_NAME", "diamond")
MAIN_MODEL_PATH = os.getenv("MAIN_MODEL_PATH", "/mnt/storage/webui/models/codegeex4-all-9b-Q4_K_M.gguf")
OPTIMIZER_MODEL_NAME = os.getenv("OPTIMIZER_MODEL_NAME", "phi-4-mini-reasoning")
OPTIMIZER_MODEL_PATH = os.getenv("OPTIMIZER_MODEL_PATH", "/mnt/storage/webui/models/Phi-4-mini-reasoning-Q4_K_M.gguf")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-large-en-v1.5")
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/mnt/storage/webui/embeddings/bge-large-en-v1.5.Q4_K_M.gguf")

CLEAN_DATASET_PATH = "cleaned_dataset.csv"

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", 15432))
PG_USER = os.getenv("PG_USER", "diamond")
PG_PASSWORD = os.getenv("PG_PASSWORD")
PG_DB = os.getenv("PG_DB", "diamond_rag")

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



from agents import router as agents_router
from tools.sanitizer import sanitize_system_context
from skills.selector import select_skill_from_flags


# ==========================================================
# FULL MODEL LIBRARY + AGENTS (inline)
# ==========================================================
MODEL_PATHS: Dict[str, str] = {
    MAIN_MODEL_NAME: MAIN_MODEL_PATH,
    OPTIMIZER_MODEL_NAME: OPTIMIZER_MODEL_PATH,
    "claude-3.7-sonnet-reasoning-gemma3-12b": "/mnt/storage/webui/models/claude-3.7-sonnet-reasoning-gemma3-12B.Q8_0_2.gguf",
    "deepseek-coder-v2-lite-instruct": "/mnt/storage/webui/models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
    "qwen2.5-coder-14b": "/mnt/storage/webui/models/Qwen2.5-Coder-14B-Q4_K_M.gguf",
    "codegeex4-all-9b": "/mnt/storage/webui/models/codegeex4-all-9b-Q4_K_M.gguf",
    "fingpt-forecaster-llama2-7b": "/mnt/storage/webui/models/fingpt-forecaster_llama2-7b.gguf",
    # Add any other models you want here
}

AGENTS: Dict[str, Dict] = {
    "code-reviewer": {"name": "Code Reviewer", "model": "deepseek-coder-v2-lite-instruct", "description": "Elite bug/security/performance reviewer", "system_prompt": "You are an elite Code Reviewer. Always use ### Issues, ### Severity, ### Suggested Fix, ### Score."},
    "code-simplifier": {"name": "Code Simplifier", "model": "qwen2.5-coder-14b", "description": "Turns complex code clean & Pythonic", "system_prompt": "You are a Code Simplifier master. Output only the simplified file in a code block."},
    "security-reviewer": {"name": "Security Reviewer", "model": "codegeex4-all-9b", "description": "Hardcore OWASP/security auditor", "system_prompt": "You are a world-class Security Engineer. Give exact patches and CVE-like severity."},
    "tech-lead": {"name": "Tech Lead", "model": "claude-3.7-sonnet-reasoning-gemma3-12b", "description": "Senior architect & roadmap planner", "system_prompt": "You are the Tech Lead. Use mermaid diagrams and strategic trade-offs."},
    "ux-reviewer": {"name": "UX Reviewer", "model": "claude-3.7-sonnet-reasoning-gemma3-12b", "description": "UI/UX accessibility & delight expert", "system_prompt": "You are a senior UX Designer. Review for clarity, WCAG, modern patterns."},
    "bash": {"name": "Bash", "model": "phi-4-mini-reasoning", "description": "Safe terminal executor", "system_prompt": "You are a safe bash executor."},
    "plan": {"name": "Plan", "model": "claude-3.7-sonnet-reasoning-gemma3-12b", "description": "Perfect step-by-step planner", "system_prompt": "You create flawless step-by-step plans."},
}

# ==========================================================
# LAZY MODEL CACHE
# ==========================================================
MAX_LOADED_MODELS = 3
model_cache: OrderedDict = OrderedDict()
model_last_used: Dict[str, float] = {}
lock = threading.Lock()

def unload_model(name: str):
    if name in model_cache:
        logging.info(f"🔄 Unloading {name}")
        del model_cache[name]
        del model_last_used[name]

def cleanup_models():
    while True:
        time.sleep(60)
        now = time.time()
        with lock:
            for name in list(model_last_used.keys()):
                if now - model_last_used[name] > 900:
                    unload_model(name)

def load_model(name: str) -> Llama:
    # Allow either a registered model name or an absolute / relative GGUF path.
    model_path = None
    cache_key = name

    # Prefer named models from MODEL_PATHS first (safe for names like "diamond").
    if name in MODEL_PATHS:
        model_path = MODEL_PATHS[name]
        cache_key = name
    else:
        # Only treat `name` as a filesystem path if it points to an actual file
        # and looks like a model file (e.g. .gguf, .bin). This prevents passing
        # literal model names (like "diamond") directly to llama_cpp.
        maybe_path = Path(name)
        if maybe_path.is_file() and maybe_path.suffix in (".gguf", ".bin", ".pt", ".pth"):
            model_path = str(maybe_path)
            cache_key = model_path
        else:
            raise HTTPException(status_code=404, detail=f"Model {name} not found")

    with lock:
        if cache_key in model_cache:
            model_last_used[cache_key] = time.time()
            model_cache.move_to_end(cache_key)
            return model_cache[cache_key]
        if len(model_cache) >= MAX_LOADED_MODELS:
            oldest = next(iter(model_cache))
            unload_model(oldest)
        logging.info(f"🚀 Loading model: {cache_key} -> {model_path}")
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=N_CTX,
                n_batch=1024,
                n_threads=N_THREADS,
                verbose=False,
            )
        except Exception as e:
            logging.error(f"Failed to initialize Llama model at {model_path}: {e}")
            # Ensure we don't leave a partially-constructed object around
            try:
                del llm
            except Exception:
                pass
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
        model_cache[cache_key] = llm
        model_last_used[cache_key] = time.time()
        model_cache.move_to_end(cache_key)
        return llm

threading.Thread(target=cleanup_models, daemon=True).start()

# ==========================================================
# SESSION LOGGING
# ==========================================================
session_log_files: Dict[str, Path] = {}

def get_or_create_log_file(session_id: str) -> Path:
    if session_id in session_log_files:
        return session_log_files[session_id]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_FOLDER / f"conversation_{timestamp}.jsonl"
    session_log_files[session_id] = log_file
    logging.info(f"📝 New conversation log started: {log_file.name}")
    return log_file

def log_conversation(session_id: str, user_message: str, assistant_response: str, model: str):
    log_file = get_or_create_log_file(session_id)
    entry = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "user": user_message, "assistant": assistant_response, "model": model}
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ==========================================================
# DB + RAG (robust)
# ==========================================================
pg_conn = None
embedder = None

def reset_db_transaction():
    if pg_conn:
        try: pg_conn.rollback()
        except: pass

def safe_embed(text: str):
    try:
        emb = embedder.embed(text)
        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
        return emb_list if len(emb_list) == 1024 else None
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return None

def is_dataset_embedded(source_name: str) -> bool:
    reset_db_transaction()
    if not pg_conn: return False
    try:
        with pg_conn.cursor() as cur:
            cur.execute("SELECT 1 FROM rag_chunks WHERE source = %s LIMIT 1", (source_name,))
            return cur.fetchone() is not None
    except:
        reset_db_transaction()
        return False

def embed_cleaned_dataset(csv_path: str, source_name: str = "cleaned_dataset"):
    if not Path(csv_path).exists(): return
    if is_dataset_embedded(source_name):
        logging.info("✅ Dataset already embedded.")
        return
    logging.info("🚀 Embedding cleaned dataset...")
    reset_db_transaction()
    inserted = 0
    batch_size = 30
    batch = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if not row: continue
            text = row[0].strip()
            if len(text) < 30: continue
            content_hash = hash_text(text)
            embedding = safe_embed(text)
            if embedding is None: continue
            batch.append((source_name, text, content_hash, embedding))
            if len(batch) >= batch_size:
                try:
                    reset_db_transaction()
                    with pg_conn.cursor() as cur:
                        cur.executemany("INSERT INTO rag_chunks (source, chunk, content_hash, embedding) VALUES (%s,%s,%s,%s) ON CONFLICT (content_hash) DO NOTHING;", batch)
                    pg_conn.commit()
                    inserted += len(batch)
                except Exception as e:
                    logging.error(f"Batch failed: {e}")
                    reset_db_transaction()
                batch = []
    if batch:
        try:
            reset_db_transaction()
            with pg_conn.cursor() as cur:
                cur.executemany("INSERT INTO rag_chunks (source, chunk, content_hash, embedding) VALUES (%s,%s,%s,%s) ON CONFLICT (content_hash) DO NOTHING;", batch)
            pg_conn.commit()
            inserted += len(batch)
        except Exception as e:
            logging.error(f"Final batch failed: {e}")
            reset_db_transaction()
    logging.info(f"✅ Inserted {inserted} embeddings.")

def retrieve_rag_context(query: str) -> str:
    if not pg_conn or not embedder: return ""
    reset_db_transaction()
    try:
        q_emb = safe_embed(query)
        if not q_emb: return ""
        with pg_conn.cursor() as cur:
            cur.execute("SELECT source, chunk FROM rag_chunks ORDER BY embedding <=> %s::vector LIMIT %s", (q_emb, RAG_TOP_K))
            rows = cur.fetchall()
        return "\n\n".join([f"📄 {source}\n{chunk[:1500]}" for source, chunk in rows])
    except Exception as e:
        logging.warning(f"RAG failed: {e}")
        reset_db_transaction()
        return ""

# ==========================================================
# CLEANING (anti-repetition)
# ==========================================================
def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts).strip()

    return ""

# ==========================================================
# AGENT LOOP
# ==========================================================
MAX_CONTEXT = 8192
RESERVED_OUTPUT_TOKENS = 1024

def safe_token_count(llm: Llama, messages: List[Dict]) -> int:
    try:
        text = "\n\n".join(m.get("content", "") for m in messages)
        return len(llm.tokenize(text.encode("utf-8", errors="ignore")))
    except:
        return sum(len(m.get("content", "")) // 4 for m in messages)

def trim_messages_to_fit(llm: Llama, messages: List[Dict], max_tokens: int):
    while True:
        if safe_token_count(llm, messages) + max_tokens <= MAX_CONTEXT:
            return messages
        for i, m in enumerate(messages):
            if m["role"] != "system":
                del messages[i]
                break
        else:
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = messages[0]["content"][:2000]
            return messages

def run_agent_loop(llm: Llama, messages: List[Dict], temperature=0.7, max_tokens=1024) -> str:
    try:
        if max_tokens > RESERVED_OUTPUT_TOKENS: max_tokens = RESERVED_OUTPUT_TOKENS
        messages = trim_messages_to_fit(llm, messages, max_tokens)
        input_tokens = safe_token_count(llm, messages)
        if input_tokens + max_tokens > MAX_CONTEXT:
            max_tokens = max(256, MAX_CONTEXT - input_tokens - 50)
        if max_tokens <= 0: return "⚠️ Context overflow"
        output = llm.create_chat_completion(messages=messages, temperature=temperature, max_tokens=max_tokens, stream=False)
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Generation error: {e}")
        return f"🚨 Generation error: {str(e)}"

COMMANDER_SYSTEM_PROMPT = """You are DIAMOND, the supreme local coding commander.
Never repeat yourself. Never output generic CLAUDE.md content unless the user explicitly asks for it.
You have full access to the codebase. Be direct and precise."""

# ==========================================================
# LIFESPAN
# ==========================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, pg_conn
    logging.info("Loading embedder (GPU)...")
    try:
        embedder = Llama(
            model_path=EMBEDDING_MODEL_PATH,
            embedding=True,
            n_gpu_layers=-1,
            n_ctx=512,
            n_batch=256,
            verbose=False,
            logits_all=False
        )
    except Exception as e:
        logging.error(f"Failed to initialize embedder: {e}")
        embedder = None

    logging.info("✅ Diamond v5.2 with Agents ready!")
    yield
    if pg_conn: pg_conn.close()

app = FastAPI(lifespan=lifespan)

# Register agent package startup/shutdown handlers at the application level
try:
    # Use application-level event handlers to avoid router.on_event deprecation warnings
    import asyncio as _asyncio
    def _agents_startup():
        # schedule the coroutine
        _asyncio.create_task(agents_router.context.connect())

    def _agents_shutdown():
        _asyncio.create_task(agents_router.context.close())

    app.add_event_handler("startup", _agents_startup)
    app.add_event_handler("shutdown", _agents_shutdown)
except Exception:
    pass

# ==========================================================
# REQUEST MODELS
# ==========================================================
class AnthropicRequest(BaseModel):
    model: str
    messages: List[Dict]
    max_tokens: int = 4096
    temperature: float = 0.7
    session_id: Optional[str] = None
    agent: Optional[str] = None

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    options: Optional[Dict] = {}

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict]
    options: Optional[Dict] = {}

class EmbeddingRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]

# ==========================================================
# ENDPOINTS
# ==========================================================
@app.get("/agents")
async def list_agents():
    app.include_router(agents_router)
    return {
        "agents": [{"id": k, "name": v["name"], "description": v["description"], "model": v["model"]} for k, v in AGENTS.items()],
        "built_in": ["bash", "plan"]
    }

@app.post("/autonomous/execute")
async def autonomous_execute(task: str):
    llm = load_model(MAIN_MODEL_NAME)
    agent = AutonomousCoder(llm, PROJECT_ROOT)
    result = agent.execute_task(task)
    return result


@app.post("/collaboration/run")
async def collaboration_run(task: str):
    llm = load_model(MAIN_MODEL_NAME)
    orchestrator = MultiAgentOrchestrator(llm)
    result = orchestrator.run(task)
    return result


@app.post("/v1/messages")
async def anthropic_messages(req: AnthropicRequest):

    session_id = req.session_id or str(uuid.uuid4())
    agent_key = req.agent

    if agent_key and agent_key in AGENTS:
        agent = AGENTS[agent_key]
        model_name = agent["model"]
        diamond_system = agent["system_prompt"]
    else:
        model_name = req.model or MAIN_MODEL_NAME
        diamond_system = COMMANDER_SYSTEM_PROMPT

    # -------- CLEAN MESSAGE EXTRACTION --------
    cleaned = []
    # collect system hints
    system_hints = []
    for m in req.messages:
        raw = m.get("content")
        if m.get("role") == "system":
            s, flags = sanitize_system_context(raw)
            system_hints.append((s, flags))
            text = extract_text_content(s)
        else:
            text = extract_text_content(raw)
        if not text:
            continue
        cleaned.append({
            "role": m.get("role"),
            "content": text
        })

    last_query = next(
        (m["content"] for m in reversed(cleaned) if m["role"] == "user"),
        ""
    )

    rag = retrieve_rag_context(last_query)

    if rag.strip():
        diamond_system += (
            "\n\nUse the following project context only if relevant:\n\n"
            f"{rag}\n"
        )

    # Server-side skill selection: inspect system_hints and cleaned user text
    all_flags = []
    for s, flags in system_hints:
        all_flags.extend(flags)
    user_text = "\n".join(m.get("content", "") for m in cleaned if m.get("role") == "user")
    selected_skill = select_skill_from_flags(all_flags, user_text)
    if selected_skill == "claude-developer-platform":
        # use concise skill instruction rather than full policy blocks
        diamond_system = "You are Claude Developer Platform assistant. Answer concisely and act on codebase context when applicable." \
                        + "\n\n" + ("Use the following project context only if relevant:\n\n" + rag + "\n" if rag.strip() else "")

    messages = [{"role": "system", "content": diamond_system}]
    messages.extend(cleaned)

    # Validate requested model early
    try:
        llm = load_model(model_name)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Model load failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    def stream():
        """Synchronous generator that yields SSE events for the client.

        Notes:
        - Uses the `llm.create_chat_completion(..., stream=True)` iterator.
        - After the generation completes we scan the final text for any
          JSON tool-invocation blocks with `"type": "tool_use"` and
          execute them via the local tools registry/executor.
        """
        message_id = f"msg_{uuid.uuid4().hex}"

        # ---- MESSAGE START ----
        yield f"event: message_start\ndata: {json.dumps({'id': message_id, 'type': 'message', 'role': 'assistant', 'model': model_name})}\n\n"

        # ---- CONTENT BLOCK START ----
        yield f"event: content_block_start\ndata: {json.dumps({'index': 0, 'type': 'text'})}\n\n"

        full_text = ""

        try:
            for chunk in llm.create_chat_completion(
                messages=messages,
                temperature=req.temperature,
                max_tokens=min(req.max_tokens, RESERVED_OUTPUT_TOKENS),
                stream=True
            ):
                delta = chunk.get("choices", [])[0].get("delta", {})
                token = delta.get("content") or delta.get("text") or ""
                if token:
                    full_text += token
                    yield (
                        "event: content_block_delta\n"
                        f"data: {json.dumps({'index': 0, 'delta': {'type': 'text', 'text': token}})}\n\n"
                    )
        except Exception as e:
            yield f"event: message_delta\ndata: {json.dumps({'error': str(e)})}\n\n"
            yield f"event: message_stop\ndata: {{}}\n\n"
            return

        # ---- CONTENT BLOCK STOP ----
        yield f"event: content_block_stop\ndata: {json.dumps({'index': 0})}\n\n"

        # ---- TOOL DETECTION & EXECUTION (post-process) ----
        try:
            import re as _re, json as _json
            matches = _re.findall(r'(\{.*?"type"\s*:\s*"tool_use".*?\})', full_text, flags=_re.DOTALL)
            for mjson in matches:
                try:
                    tool_call = _json.loads(mjson)
                    tool_name = tool_call.get('name')
                    yield f"event: tool_start\ndata: {json.dumps({'name': tool_name})}\n\n"
                    # try the tools.executor first, fallback to root `tool_registry` or tools registry
                    tool_result = None
                    try:
                        from tools.executor import execute_tool_call as _exec_tool
                        tool_result = _exec_tool(tool_call)
                    except Exception:
                        try:
                            from tool_registry import TOOLS as _TOOLS
                            t = _TOOLS.get(tool_name)
                            if t:
                                tool_result = t['function'](**(tool_call.get('input', {})))
                            else:
                                tool_result = {'error': 'tool not found'}
                        except Exception as ex:
                            tool_result = {'error': str(ex)}

                    yield f"event: tool_result\ndata: {json.dumps({'name': tool_name, 'result': tool_result})}\n\n"

                    # inject the tool output into the stream as assistant text
                    yield (
                        "event: content_block_delta\n"
                        f"data: {json.dumps({'index': 0, 'delta': {'type': 'text', 'text': str(tool_result)}})}\n\n"
                    )
                except Exception as e:
                    yield f"event: tool_result\ndata: {json.dumps({'error': str(e)})}\n\n"
        except Exception:
            # non-fatal, continue
            pass

        # ---- FINAL DELTA & STOP ----
        yield f"event: message_delta\ndata: {json.dumps({'stop_reason': 'end_turn'})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({})}\n\n"

        log_conversation(session_id, last_query, full_text, model_name)

    return StreamingResponse(stream(), media_type="text/event-stream")
@app.get("/health")
@app.get("/v1/health")
async def health():
    return {"status": "ok", "version": "5.2-ULTIMATE", "agents": len(AGENTS)}

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": name, "object": "model"} for name in MODEL_PATHS]}

# Your other endpoints (/api/chat, /api/generate, /api/embeddings, /api/tags) can be added here exactly as before — they now use load_model.

if __name__ == "__main__":
    import uvicorn
    print(f"🌟 Diamond v5.2 with full Agents running on http://{HOST}:{PORT}")
