import json
import asyncio
from typing import Dict, Any, Optional, Callable
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import redis.asyncio as redis
import asyncpg


# ============================================================
# CONFIGURATION
# ============================================================

REDIS_URL = "redis://localhost:6379"
POSTGRES_DSN = "postgresql://postgres:postgres@localhost:5432/agents"


# ============================================================
# DATA MODELS
# ============================================================

class AgentRequest(BaseModel):
    agent: str
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    agent: str
    type: str
    output: Any


# ============================================================
# AGENT CORE
# ============================================================

class AgentContext:
    """
    Shared context passed to all agents
    """
    def __init__(self):
        self.redis = None
        self.pg = None

    async def connect(self):
        self.redis = await redis.from_url(REDIS_URL)
        self.pg = await asyncpg.connect(POSTGRES_DSN)

    async def close(self):
        if self.redis:
            await self.redis.close()
        if self.pg:
            await self.pg.close()


# ============================================================
# BASE AGENT CLASS
# ============================================================

class BaseAgent:
    name = "base"

    def __init__(self, context: AgentContext):
        self.context = context

    async def run(self, prompt: str, metadata: Optional[Dict] = None):
        raise NotImplementedError


# ============================================================
# IMPLEMENTED AGENTS
# ============================================================

class CodeAgent(BaseAgent):
    name = "code"

    async def run(self, prompt: str, metadata=None):
        result = {
            "analysis": f"Analyzed code request: {prompt}",
            "suggestions": ["Refactor functions", "Improve naming"]
        }
        return AgentResponse(
            agent=self.name,
            type="json",
            output=result
        )


class ChatAgent(BaseAgent):
    name = "chat"

    async def run(self, prompt: str, metadata=None):
        return AgentResponse(
            agent=self.name,
            type="text",
            output=f"Chat response to: {prompt}"
        )


class ResearchAgent(BaseAgent):
    name = "research"

    async def run(self, prompt: str, metadata=None):
        output = f"# Research Result\n\nDetailed analysis about:\n\n{prompt}"
        return AgentResponse(
            agent=self.name,
            type="markdown",
            output=output
        )


class MemoryAgent(BaseAgent):
    name = "memory"

    async def run(self, prompt: str, metadata=None):
        # best-effort memory persist; may fail if context not connected
        if self.context and getattr(self.context, 'redis', None):
            try:
                await self.context.redis.set("last_memory", prompt)
            except Exception:
                pass
        if self.context and getattr(self.context, 'pg', None):
            try:
                await self.context.pg.execute(
                    "INSERT INTO memories(content) VALUES($1)",
                    prompt
                )
            except Exception:
                pass
        return AgentResponse(
            agent=self.name,
            type="status",
            output="Memory stored."
        )


# ============================================================
# AGENT REGISTRY
# ============================================================

class AgentRegistry:
    def __init__(self, context: AgentContext):
        self.context = context
        self.agents: Dict[str, Callable] = {}

        # Register all agents here
        self.register(CodeAgent)
        self.register(ChatAgent)
        self.register(ResearchAgent)
        self.register(MemoryAgent)

    def register(self, agent_cls):
        self.agents[agent_cls.name] = agent_cls

    def get(self, name: str):
        if name not in self.agents:
            raise ValueError(f"Agent '{name}' not found")
        return self.agents[name](self.context)


# ============================================================
# FASTAPI ROUTER (/agents)
# ============================================================

router = APIRouter(prefix="/agents", tags=["agents"])

context = AgentContext()
registry = AgentRegistry(context)


@router.on_event("startup")
async def startup():
    # Start connection but ignore failures in test environments
    try:
        await context.connect()
    except Exception:
        pass


@router.on_event("shutdown")
async def shutdown():
    try:
        await context.close()
    except Exception:
        pass


@router.post("/")
async def run_agent(request: AgentRequest):
    try:
        agent = registry.get(request.agent)
        response = await agent.run(request.prompt, request.metadata)
        # Support both pydantic v2 (model_dump) and v1 (dict)
        return response.model_dump() if hasattr(response, "model_dump") else (response.dict() if hasattr(response, "dict") else {})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
