"""Client-side tool runtime for the ds41f local web client.

This module deliberately lives above the qualified model runtime.  It declares
function tools through the public Chat Completions API, validates model-emitted
calls, executes local-client implementations, and returns tool results as
ordinary tool messages to the same stateful ds41f session.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from time import monotonic
from threading import Timer
import hashlib
import http.client
import ipaddress
import json
import os
import socket
from typing import Any, Protocol
from urllib.parse import quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen


class ToolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolResult:
    content: str
    display: dict[str, Any]


@dataclass(frozen=True)
class ToolContext:
    session_id: str | None = None


class ClientTool(Protocol):
    name: str
    schema: dict[str, Any]
    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult: ...


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _require_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ToolError(f"tool arguments are not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ToolError("tool arguments must be a JSON object")
    return value


def _validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("only http(s) URLs with a host are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"could not resolve URL host {parsed.hostname!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ToolError("refusing to fetch local/private network URL")
    return url


def readable_text(text: str, content_type: str) -> str:
    if 'html' not in content_type.lower(): return ' '.join(text.split())
    class Reader(HTMLParser):
        def __init__(self): super().__init__(convert_charrefs=True); self.parts = []; self.hidden = []
        def handle_starttag(self, tag, attrs):
            if tag in ('script', 'style', 'noscript', 'template'): self.hidden.append(tag)
            if not self.hidden and tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'tr'): self.parts.append(' ')
        def handle_endtag(self, tag):
            if self.hidden and tag == self.hidden[-1]: self.hidden.pop()
            if not self.hidden: self.parts.append(' ')
        def handle_data(self, value):
            if not self.hidden: self.parts.append(value)
    reader = Reader(); reader.feed(text); reader.close()
    return ' '.join(''.join(reader.parts).split())


def _read_public_url(url: str, *, timeout: float = 45, max_bytes: int = 8 * 1024 * 1024, max_redirects: int = 8) -> tuple[str, str, bytes]:
    current = _validate_public_url(url)
    deadline = monotonic() + timeout
    for _ in range(max_redirects + 1):
        parsed = urlparse(current)
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        host = parsed.hostname or ""
        port = parsed.port
        remaining = deadline - monotonic()
        if remaining <= 0: raise ToolError('fetch network timeout ceiling reached')
        conn = conn_cls(host, port=port, timeout=remaining)
        path = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
        deadline_timer = None
        try:
            conn.connect()
            connection_socket = conn.sock
            peer = ipaddress.ip_address(connection_socket.getpeername()[0])
            if not peer.is_global: raise ToolError('refusing actual local/private network peer')
            def set_remaining_timeout():
                remaining = deadline - monotonic()
                if remaining <= 0: raise ToolError('fetch network timeout ceiling reached')
                connection_socket.settimeout(remaining)
                return remaining
            remaining = set_remaining_timeout()
            def expire_transfer():
                # Inactivity timeouts alone do not bound trickled headers.
                try: connection_socket.shutdown(socket.SHUT_RDWR)
                except OSError: pass
            deadline_timer = Timer(remaining, expire_transfer)
            deadline_timer.daemon = True
            deadline_timer.start()
            conn.request("GET", path, headers={"User-Agent": "ds41f-local-web-client/0.1", "Accept-Encoding": "identity", "Accept": "text/html,text/plain,application/json;q=0.9,*/*;q=0.1"})
            set_remaining_timeout()
            resp = conn.getresponse()
            if resp.status in {301, 302, 303, 307, 308}:
                loc = resp.getheader("Location")
                if not loc:
                    raise ToolError("redirect response missing Location header")
                current = _validate_public_url(urljoin(current, loc))
                continue  # finally closes connection; never wait on a redirect body
            if resp.status >= 400:
                raise ToolError(f"fetch_url HTTP status {resp.status}")
            ctype = resp.getheader("content-type", "")
            allowed = any(x in ctype.lower() for x in ("text/", "html", "json", "xml"))
            if not allowed:
                raise ToolError(f"refusing non-text content type {ctype!r}")
            if resp.getheader('content-encoding', 'identity').lower() not in ('identity', ''):
                raise ToolError('compressed fetch bodies are not supported (resource/decompression boundary)')
            chunks = bytearray()
            while len(chunks) <= max_bytes:
                set_remaining_timeout()
                chunk = resp.read1(min(65536, max_bytes + 1 - len(chunks)))
                if not chunk: break
                chunks.extend(chunk)
            data = bytes(chunks)
            # Keep usable partial data at the resource ceiling, not an error
            # that discards an otherwise ordinary large page.
            return current, ctype, data
        finally:
            if deadline_timer is not None: deadline_timer.cancel()
            conn.close()
    raise ToolError("too many redirects")


class SearchProvider(Protocol):
    name: str
    def search(self, query: str, *, max_results: int, session_id: str | None = None) -> dict[str, Any]: ...


class HostedMCPClient:
    def __init__(self, endpoint: str, *, headers: dict[str, str] | None = None, timeout: float = 20) -> None:
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self._next_id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        rid = self._next_id; self._next_id += 1
        body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "User-Agent": "ds41f-local-web-client/0.1", **self.headers}
        req = Request(self.endpoint, data=json.dumps(body).encode(), method="POST", headers=headers)
        with urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read(1_000_000).decode("utf-8", "replace")
        payloads: list[str] = []
        if raw.lstrip().startswith("{"):
            payloads = [raw]
        else:
            for line in raw.splitlines():
                if line.startswith("data:"):
                    data = line[5:].strip()
                    if data and data != "[DONE]":
                        payloads.append(data)
        last: Any = None
        for payload in payloads:
            msg = json.loads(payload)
            if msg.get("id") == rid or "result" in msg or "error" in msg:
                if msg.get("error"):
                    raise ToolError(f"MCP {method} error: {msg['error']}")
                last = msg.get("result")
        return last


def _extract_text_from_mcp_result(result: Any) -> str:
    parts: list[str] = []
    for item in (result or {}).get("content", []) if isinstance(result, dict) else []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n\n".join(parts)


def _normalize_text_results(text: str, *, provider: str, max_results: int) -> list[dict[str, str]]:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            raw = data.get("results") if isinstance(data, dict) else None
            if isinstance(raw, list):
                out = []
                for item in raw[:max_results]:
                    if not isinstance(item, dict):
                        continue
                    excerpts = item.get("excerpts")
                    snippet = " ".join(str(x) for x in excerpts) if isinstance(excerpts, list) else str(item.get("snippet") or item.get("description") or "")
                    out.append({"title": str(item.get("title", ""))[:200], "url": str(item.get("url", ""))[:500], "snippet": " ".join(snippet.split())[:900], "provider_metadata": provider})
                usable = [item for item in out if item['url'].startswith(('http://', 'https://'))]
                if usable:
                    return usable
        except Exception:
            pass
    results: list[dict[str, str]] = []
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    for block in blocks:
        title = ""; url = ""; lines = []
        for line in block.splitlines():
            if line.startswith("Title:"):
                title = line.split(":", 1)[1].strip()
            elif line.startswith("URL:"):
                url = line.split(":", 1)[1].strip()
            elif not line.startswith(("Published:", "Author:", "Highlights:")):
                lines.append(line.strip())
        snippet = " ".join(" ".join(lines).split())
        if url.startswith(('http://', 'https://')):
            results.append({"title": title[:200], "url": url[:500], "snippet": snippet[:900], "provider_metadata": provider})
        if len(results) >= max_results:
            break
    usable = [item for item in results if item['url'].startswith(('http://', 'https://'))]
    if not usable:
        # Hosted MCP quota/auth notices can arrive as ordinary text, not isError.
        # They are provider failures, never invented successful search sources.
        raise ToolError(f"{provider} returned no usable source URLs: " + " ".join(text.split())[:180])
    return usable[:max_results]


class ExaMCPProvider:
    name = "exa"
    def __init__(self, endpoint: str = "https://mcp.exa.ai/mcp", api_key: str | None = None) -> None:
        if api_key:
            sep = "&" if "?" in endpoint else "?"
            endpoint = f"{endpoint}{sep}exaApiKey={quote_plus(api_key)}"
        self.client = HostedMCPClient(endpoint)

    def search(self, query: str, *, max_results: int, session_id: str | None = None) -> dict[str, Any]:
        objective = f"Find concise, authoritative sources for: {query}"
        result = self.client.call("tools/call", {"name": "web_search_exa", "arguments": {"query": query, "objective": objective, "type": "auto", "livecrawl": "fallback", "numResults": max_results}})
        text = _extract_text_from_mcp_result(result)
        if not text.strip():
            raise ToolError("Exa returned no usable text content")
        return {"provider": self.name, "query": query, "results": _normalize_text_results(text, provider=self.name, max_results=max_results)}


class ParallelMCPProvider:
    name = "parallel"
    def __init__(self, endpoint: str = "https://search.parallel.ai/mcp", api_key: str | None = None) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers.update({"Authorization": f"Bearer {api_key}", "x-api-key": api_key})
        self.client = HostedMCPClient(endpoint, headers=headers)
        self.tool_name = os.environ.get("DS41F_PARALLEL_MCP_TOOL")

    def _tool(self) -> dict[str, Any]:
        listing = self.client.call("tools/list", {})
        tools = listing.get("tools", []) if isinstance(listing, dict) else []
        if self.tool_name:
            for t in tools:
                if t.get("name") == self.tool_name:
                    return t
            raise ToolError(f"Parallel MCP tool {self.tool_name!r} not found")
        for t in tools:
            name = str(t.get("name", "")).lower()
            desc = str(t.get("description", "")).lower()
            if "search" in name or "search" in desc:
                return t
        raise ToolError("Parallel MCP exposed no search tool")

    def search(self, query: str, *, max_results: int, session_id: str | None = None) -> dict[str, Any]:
        name = self.tool_name or "web_search"
        args: dict[str, Any] = {"objective": query, "search_queries": [query], "session_id": session_id or "ds41f"}
        result = self.client.call("tools/call", {"name": name, "arguments": args})
        text = _extract_text_from_mcp_result(result)
        if not text.strip():
            raise ToolError("Parallel returned no usable text content")
        return {"provider": self.name, "query": query, "results": _normalize_text_results(text, provider=self.name, max_results=max_results), "provider_metadata": {"tool": name}}


class AutoSearchProvider:
    name = "auto"
    def __init__(self, providers: dict[str, SearchProvider]) -> None:
        self.providers = providers

    def _order(self, session_id: str | None) -> list[str]:
        sid = session_id or "default"
        first = "exa" if int(hashlib.sha256(sid.encode()).hexdigest(), 16) % 2 == 0 else "parallel"
        second = "parallel" if first == "exa" else "exa"
        return [first, second]

    def search(self, query: str, *, max_results: int, session_id: str | None = None) -> dict[str, Any]:
        errors: list[str] = []
        for name in self._order(session_id):
            provider = self.providers[name]
            try:
                payload = provider.search(query, max_results=max_results, session_id=session_id)
                payload["selected_provider"] = name
                payload["fallback_errors"] = errors
                return payload
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise ToolError("all search providers failed: " + "; ".join(errors))


class WebSearchTool:
    name = "web_search"
    schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web via Exa or Parallel. Returns bounded results with provider, title, URL, and concise context; does not fetch full pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Number of results to return (default 10, maximum 20)."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, provider: SearchProvider) -> None:
        self.provider = provider

    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("web_search.query must be a non-empty string")
        max_results = arguments.get("max_results", 10)
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1 or max_results > 20:
            raise ToolError("web_search.max_results must be an integer from 1 to 20")
        payload = self.provider.search(query.strip(), max_results=max_results, session_id=None if context is None else context.session_id)
        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list):
            raise ToolError("search provider returned malformed results")
        bounded = []
        for r in raw_results[:max_results]:
            if not isinstance(r, dict):
                raise ToolError("search provider returned malformed result item")
            bounded.append({"title": str(r.get("title", ""))[:200], "url": str(r.get("url", ""))[:500], "snippet": str(r.get("snippet", ""))[:900], "provider_metadata": r.get("provider_metadata")})
        out = {"query": query.strip(), "provider": payload.get("provider", self.provider.name), "selected_provider": payload.get("selected_provider", payload.get("provider", self.provider.name)), "results": bounded, "fallback_errors": payload.get("fallback_errors", [])}
        if payload.get("provider_metadata"):
            out["provider_metadata"] = payload["provider_metadata"]
        return ToolResult(content=_json_dumps(out), display={"tool": self.name, **out})


class FetchURLTool:
    name = "fetch_url"
    schema = {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch readable public http(s) text. Network resource ceiling 8 MiB; model-facing text is fitted by runtime context admission. Truncation and next_offset are explicit; use offset for an intentional continuation. Private/local URLs and redirects are refused.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}}, "required": ["url"], "additionalProperties": False},
        },
    }

    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolError("fetch_url.url must be a non-empty string")
        offset = arguments.get('offset', 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError('fetch_url.offset must be a nonnegative integer')
        final_url, ctype, data = _read_public_url(url.strip())
        network_limit = 8 * 1024 * 1024
        network_truncated = len(data) > network_limit
        text = readable_text(data[:network_limit].decode('utf-8', 'replace'), ctype)
        # A separate memory/serialization ceiling, not a model token estimate.
        selected = text[offset:].encode('utf-8')
        excerpt = selected[:1024 * 1024].decode('utf-8', 'ignore')
        payload = {"url": final_url, "content_type": ctype, "excerpt": excerpt,
                   'offset': offset, 'total_chars': len(text), 'returned_chars': len(excerpt),
                   'next_offset': offset + len(excerpt), 'network_truncated': network_truncated,
                   'resource_truncated': len(selected) > 1024 * 1024,
                   'context_truncated': False,
                   'termination_reason': 'network_resource_ceiling' if network_truncated else 'model_facing_resource_ceiling' if len(selected) > 1024 * 1024 else None}
        return ToolResult(content=_json_dumps(payload), display={"tool": self.name, **payload})


class ToolRegistry:
    def __init__(self, tools: list[ClientTool] | None = None) -> None:
        self.tools = {tool.name: tool for tool in (tools or [WebSearchTool(_provider_from_env()), FetchURLTool()])}

    @property
    def declarations(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self.tools.values()]

    def execute_calls(self, calls: list[dict[str, Any]], *, session_id: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seen: set[str] = set(); messages: list[dict[str, Any]] = []; displays: list[dict[str, Any]] = []
        context = ToolContext(session_id=session_id)
        for call in calls:
            call_id = call.get("id")
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name")
            if not isinstance(call_id, str) or not call_id:
                raise ToolError("tool call id must be a non-empty string")
            if call_id in seen:
                raise ToolError(f"duplicate tool call id {call_id!r}")
            seen.add(call_id)
            if not isinstance(name, str) or name not in self.tools:
                raise ToolError(f"unknown tool {name!r}")
            args = _require_object(fn.get("arguments") if isinstance(fn.get("arguments"), str) else "{}")
            try:
                result = self.tools[name].run(args, context)
            except Exception as exc:
                payload = {"error": str(exc), "tool": name}
                result = ToolResult(content=_json_dumps(payload), display={"tool": name, "error": str(exc)})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result.content})
            displays.append({"id": call_id, **result.display})
        return messages, displays


def _provider_from_env() -> SearchProvider:
    provider_name = os.environ.get("DS41F_WEB_SEARCH_PROVIDER", "auto").lower()
    exa = ExaMCPProvider(api_key=os.environ.get("DS41F_EXA_API_KEY"))
    parallel = ParallelMCPProvider(api_key=os.environ.get("DS41F_PARALLEL_API_KEY"))
    if provider_name == "auto":
        return AutoSearchProvider({"exa": exa, "parallel": parallel})
    if provider_name == "exa":
        return exa
    if provider_name == "parallel":
        return parallel
    raise RuntimeError(f"unsupported DS41F_WEB_SEARCH_PROVIDER {provider_name!r}; expected auto, exa, or parallel")


def registry_from_env() -> ToolRegistry:
    return ToolRegistry([WebSearchTool(_provider_from_env()), FetchURLTool()])
