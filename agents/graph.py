class AgentNode:
    def __init__(self, name, executor):
        self.name = name
        self.executor = executor
        self.children = []

    def connect(self, agent_node):
        self.children.append(agent_node)


class AgentOrchestrator:

    def __init__(self, registry):
        self.registry = registry

    async def execute(self, root_agent, messages):

        agent = self.registry.get(root_agent)
        result = await agent.run(messages)

        # Autonomous delegation trigger
        if "DELEGATE:" in result:
            next_agent = result.split("DELEGATE:")[1].strip()
            return await self.execute(next_agent, messages)

        return result
