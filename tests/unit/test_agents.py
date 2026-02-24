import asyncio

def test_agent_registry_and_chat():
    from agents import registry

    # Ensure registry returns a chat agent and it runs without external services
    agent = registry.get('chat')
    assert agent.name == 'chat'

    result = asyncio.run(agent.run('Hello agent'))
    assert result.type == 'text'
    assert 'Chat response' in result.output
