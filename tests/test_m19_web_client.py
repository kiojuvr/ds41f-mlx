import json

from ds41f_mlx.web_client import StatefulToolChatClient
from ds41f_mlx.web_tools import ToolRegistry, ToolResult, ToolError, FetchURLTool


class FakeSearch:
    name = "fake"
    def search(self, query, *, max_results):
        return [{"title": "Example", "url": "https://example.com", "snippet": f"hit for {query}"}]


class WebSearchOnly:
    name = "web_search"
    schema = {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}
    def run(self, arguments):
        return ToolResult(json.dumps({"results": [{"title": "T", "url": "https://u", "snippet": arguments["query"]}]}), {"tool": "web_search", "results": [{"title": "T", "url": "https://u", "snippet": arguments["query"]}]})


class FakeRuntime:
    def __init__(self):
        self.calls = []
    def chat(self, session_id, body):
        self.calls.append((session_id, body))
        if len(self.calls) == 1:
            return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\":\"ds41f\"}"}}]}}]}
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_call_id"] == "call_1"
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Final answer with source."}}]}


def test_m19_generic_tool_loop_returns_result_to_same_stateful_session():
    runtime = FakeRuntime()
    client = StatefulToolChatClient(runtime, ToolRegistry([WebSearchOnly()]))
    out = client.run_turn(session_id="sess_1", transcript=[], user_message="search")
    assert [c[0] for c in runtime.calls] == ["sess_1", "sess_1"]
    assert len(runtime.calls) == 2
    assert out.messages[0] == {"role": "user", "content": "search"}
    assert out.messages[1]["tool_calls"][0]["id"] == "call_1"
    assert out.messages[2]["role"] == "tool"
    assert out.messages[-1]["content"] == "Final answer with source."


def test_m19_tool_registry_rejects_duplicate_or_unknown_calls():
    reg = ToolRegistry([WebSearchOnly()])
    calls = [
        {"id": "x", "function": {"name": "web_search", "arguments": "{\"query\":\"a\"}"}},
        {"id": "x", "function": {"name": "web_search", "arguments": "{\"query\":\"b\"}"}},
    ]
    try:
        reg.execute_calls(calls)
    except ToolError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate tool id accepted")


def test_m19_fetch_url_blocks_loopback_before_network(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 0))])
    try:
        FetchURLTool().run({"url": "http://localhost/private"})
    except ToolError as exc:
        assert "local/private" in str(exc)
    else:
        raise AssertionError("loopback URL accepted")
