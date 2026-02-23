from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from agents_runtime import AgentExecutor

router = APIRouter(prefix="/agents")

executor = AgentExecutor()


@router.post("/stream")
async def run_agent(payload: dict):

    agent_config = payload["agent_config"]
    messages = payload["messages"]

    stream = await executor.run(agent_config, messages)

    async def event_stream():
        async for event in stream:
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
