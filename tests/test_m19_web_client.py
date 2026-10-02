import json
import urllib.error

import pytest

from ds41f_mlx.web_client import StatefulToolChatClient, RuntimeHTTPError
from ds41f_mlx.web_tools import AutoSearchProvider, HostedMCPClient, ToolRegistry, ToolResult, ToolError, FetchURLTool, WebSearchTool


class WebSearchOnly:
    name = "web_search"
    schema = {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}
    def run(self, arguments, context=None):
        return ToolResult(json.dumps({"results": [{"title": "T", "url": "https://u", "snippet": arguments["query"]}], "session": context.session_id}), {"tool": "web_search", "session": context.session_id})


class FakeRuntime:
    def __init__(self, always_tool=False):
        self.calls = []; self.always_tool = always_tool
    def chat(self, session_id, body):
        self.calls.append((session_id, body))
        if self.always_tool or len(self.calls) == 1:
            return {"choices": [{"finish_reason": "tool_calls", "message": {"content": "", "tool_calls": [{"id": f"call_{len(self.calls)}", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\":\"ds41f\"}"}}]}}]}
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_call_id"] == "call_1"
        return {"choices": [{"finish_reason": "stop", "message": {"content": "Final answer with source."}}]}


class Provider:
    def __init__(self, name, fail=False):
        self.name = name; self.fail = fail; self.calls = []
    def search(self, query, *, max_results, session_id=None):
        self.calls.append((query, max_results, session_id))
        if self.fail:
            raise TimeoutError(f"{self.name} timeout")
        return {"provider": self.name, "query": query, "results": [{"title": self.name, "url": "https://example.com", "snippet": query}]}


def test_m19_generic_tool_loop_returns_result_to_same_stateful_session():
    runtime = FakeRuntime()
    client = StatefulToolChatClient(runtime, ToolRegistry([WebSearchOnly()]))
    out = client.run_turn(session_id="sess_1", transcript=[], user_message="search")
    assert [c[0] for c in runtime.calls] == ["sess_1", "sess_1"]
    assert len(runtime.calls) == 2
    assert out.messages[0] == {"role": "user", "content": "search"}
    assert out.messages[1]["tool_calls"][0]["id"] == "call_1"
    assert out.messages[2]["role"] == "tool"
    assert json.loads(out.messages[2]["content"])["session"] == "sess_1"
    assert out.messages[-1]["content"] == "Final answer with source."


def test_m19_tool_registry_rejects_duplicate_unknown_and_malformed_calls():
    reg = ToolRegistry([WebSearchOnly()])
    with pytest.raises(ToolError, match="duplicate"):
        reg.execute_calls([
            {"id": "x", "function": {"name": "web_search", "arguments": "{\"query\":\"a\"}"}},
            {"id": "x", "function": {"name": "web_search", "arguments": "{\"query\":\"b\"}"}},
        ])
    with pytest.raises(ToolError, match="unknown"):
        reg.execute_calls([{"id": "x", "function": {"name": "nope", "arguments": "{}"}}])
    with pytest.raises(ToolError, match="valid JSON"):
        reg.execute_calls([{"id": "x", "function": {"name": "web_search", "arguments": "{"}}])


def test_m19_tool_round_limit_is_bounded():
    client = StatefulToolChatClient(FakeRuntime(always_tool=True), ToolRegistry([WebSearchOnly()]))
    with pytest.raises(ToolError, match="maximum client tool rounds"):
        client.run_turn(session_id="sess", transcript=[], user_message="loop", max_tool_rounds=1)


def test_m19_auto_provider_selection_is_deterministic_and_falls_back():
    exa = Provider("exa", fail=True); parallel = Provider("parallel")
    auto = AutoSearchProvider({"exa": exa, "parallel": parallel})
    assert auto._order("sess_A") == auto._order("sess_A")
    out = auto.search("q", max_results=1, session_id="sess_A")
    assert out["provider"] == "parallel"
    assert out["fallback_errors"]


def test_m19_explicit_provider_override(monkeypatch):
    monkeypatch.setenv("DS41F_WEB_SEARCH_PROVIDER", "exa")
    from ds41f_mlx.web_tools import _provider_from_env
    assert _provider_from_env().name == "exa"
    monkeypatch.setenv("DS41F_WEB_SEARCH_PROVIDER", "parallel")
    assert _provider_from_env().name == "parallel"


def test_m19_web_search_malformed_provider_payload_is_tool_error():
    class Bad:
        name = "bad"
        def search(self, query, *, max_results, session_id=None):
            return {"provider": "bad", "results": "not-list"}
    with pytest.raises(ToolError, match="malformed"):
        WebSearchTool(Bad()).run({"query": "x"})


def test_m19_mcp_sse_error_is_tool_error(monkeypatch):
    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self, *a): return b'event: message\ndata: {"jsonrpc":"2.0","id":1,"error":{"message":"bad"}}\n\n'
    monkeypatch.setattr("ds41f_mlx.web_tools.urlopen", lambda *a, **k: R())
    with pytest.raises(ToolError, match="MCP"):
        HostedMCPClient("https://example.com").call("tools/list", {})


def test_m19_fetch_url_blocks_loopback_before_network(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("127.0.0.1", 0))])
    with pytest.raises(ToolError, match="local/private"):
        FetchURLTool().run({"url": "http://localhost/private"})


def test_m19_fetch_url_blocks_public_to_private_redirect(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda host, *a, **k: [(None, None, None, None, ("93.184.216.34" if host == "public.example" else "127.0.0.1", 0))])
    class Resp:
        status = 302
        def getheader(self, name, default=None): return "http://localhost/private" if name == "Location" else default
        def read(self, *a): return b""
    class Conn:
        def __init__(self, *a, **k): pass
        def request(self, *a, **k): pass
        def getresponse(self): return Resp()
        def close(self): pass
    monkeypatch.setattr("http.client.HTTPConnection", Conn)
    with pytest.raises(ToolError, match="local/private"):
        FetchURLTool().run({"url": "http://public.example/start"})


def test_m19_fetch_url_rejects_content_type_and_size(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *a, **k: [(None, None, None, None, ("93.184.216.34", 0))])
    class Resp:
        status = 200
        def __init__(self, ctype, data): self.ctype=ctype; self.data=data
        def getheader(self, name, default=None): return self.ctype if name == "content-type" else default
        def read(self, n=-1): return self.data[:n]
    class ConnBadType:
        def __init__(self, *a, **k): pass
        def request(self, *a, **k): pass
        def getresponse(self): return Resp("image/png", b"x")
        def close(self): pass
    monkeypatch.setattr("http.client.HTTPSConnection", ConnBadType)
    with pytest.raises(ToolError, match="non-text"):
        FetchURLTool().run({"url": "https://example.com/img"})
    class ConnHuge(ConnBadType):
        def getresponse(self): return Resp("text/plain", b"x" * 80001)
    monkeypatch.setattr("http.client.HTTPSConnection", ConnHuge)
    with pytest.raises(ToolError, match="exceeds"):
        FetchURLTool().run({"url": "https://example.com/huge"})


def test_m19_unknown_closed_session_surfaces_runtime_error():
    class ClosedRuntime:
        def chat(self, session_id, body):
            raise RuntimeHTTPError(404, '{"error":{"message":"unknown or closed session"}}')
    client = StatefulToolChatClient(ClosedRuntime(), ToolRegistry([WebSearchOnly()]))
    with pytest.raises(RuntimeHTTPError) as exc:
        client.run_turn(session_id="closed", transcript=[], user_message="hi")
    assert exc.value.status == 404
