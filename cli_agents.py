import argparse
import asyncio
import json
from agents_runtime import AgentExecutor


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent")
    parser.add_argument("prompt")
    args = parser.parse_args()

    executor = AgentExecutor()

    agent_config = {
        "prompt": f"You are agent {args.agent}"
    }

    messages = [{"role": "user", "content": args.prompt}]

    stream = await executor.run(agent_config, messages)

    async for event in stream:
        print(json.dumps(event))


if __name__ == "__main__":
    asyncio.run(main())
