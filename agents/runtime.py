class AgentRuntime:

    def __init__(self, registry, model_pool, context_manager):
        self.registry = registry
        self.model_pool = model_pool
        self.context_manager = context_manager

    async def run(self, agent_name, messages):

        messages = await self.context_manager.compact(
            messages,
            model_call=self.raw_model_call
        )

        endpoint = self.model_pool.next()

        stream = self.raw_model_stream(endpoint, messages)

        async for event in tool_execution_loop(stream, self.inject_message):
            yield event
