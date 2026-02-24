from fastapi.testclient import TestClient
import main


class MockLLM:
    def create_chat_completion(self, messages=None, temperature=None, max_tokens=None, stream=False):
        if stream:
            # Yield two small chunks to simulate streaming
            yield {"choices": [{"delta": {"content": "Hello "}}]}
            yield {"choices": [{"delta": {"content": "world"}}]}
        else:
            return {"choices": [{"message": {"content": "Hello world"}}]}


def test_agents_list():
    client = TestClient(main.app)
    r = client.get("/agents")
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data


def test_v1_messages_stream(monkeypatch):
    # Patch load_model to avoid loading real GGUF models during tests
    monkeypatch.setattr(main, "load_model", lambda name: MockLLM())

    client = TestClient(main.app)

    payload = {
        "model": "mock",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 32,
        "temperature": 0.2
    }

    with client.stream("POST", "/v1/messages", json=payload) as resp:
        assert resp.status_code == 200
        body = ""
        for line in resp.iter_lines(decode_unicode=True):
            body += line + "\n"

    # Expect SSE event names in stream
    assert "message_start" in body or "content_block_delta" in body
