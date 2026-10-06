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
    def __init__(self, message: str, *, code: str = 'tool_error') -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AcquiredResource:
    """Bounded original-byte transfer, not model/history authority.

    data excludes the overflow probe byte. Binary consumers must not interpret
    a truncated resource as a complete file.
    """
    source_url: str
    url: str
    content_type: str
    data: bytes
    truncated: bool
    redirects: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


FETCH_TEXT_BYTES = 8 * 1024 * 1024
FETCH_TRANSFER_SECONDS = 45
FETCH_REDIRECTS = 8
FETCH_CONTENT_TYPE_CHARS = 4096
TOOL_BATCH_SECONDS = 180
MCP_RESPONSE_BYTES = 1_000_000
TOOL_CALL_BYTES = 1024 * 1024
TOOL_RESULT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ToolResult:
    content: str | list[dict[str, Any]]
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
    if not isinstance(url, str) or len(url) > 8192:
        raise ToolError('URL exceeds 8192-character security/resource boundary', code='security_rejection')
    try:
        parsed = urlparse(url)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise ToolError('invalid URL host or port', code='security_rejection') from exc
    if (parsed.scheme not in {"http", "https"} or not host
            or parsed.username is not None or parsed.password is not None):
        raise ToolError("only public http(s) URLs without credentials are allowed", code='security_rejection')
    try:
        infos = socket.getaddrinfo(host, port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"could not resolve URL host {host!r}", code='acquisition_failure') from exc
    if not infos:
        raise ToolError('URL host resolved to no addresses', code='acquisition_failure')
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise ToolError("refusing to fetch local/private network URL", code='security_rejection')
    return url


def readable_text(text: str, content_type: str) -> str:
    media = content_type.split(';', 1)[0].strip().lower()
    if media not in {'text/html', 'application/xhtml+xml'}: return ' '.join(text.split())
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


def _content_type_allowed(content_type: str, kind: str) -> bool:
    # Parameters never change a media type. image/png;note=html is not text.
    media = content_type.split(';', 1)[0].strip().lower()
    if kind == 'text':
        return (media.startswith('text/') or media in {'application/json', 'application/xml', 'application/xhtml+xml'}
                or media.startswith('application/') and media.endswith(('+json', '+xml')))
    if kind == 'image':
        return media in {'image/png', 'image/jpeg', 'image/webp'}
    if kind == 'pdf':
        return media == 'application/pdf'
    raise ValueError('unknown acquisition content kind')


def acquire_public_resource(url: str, *, kind: str, max_bytes: int,
                            timeout: float = FETCH_TRANSFER_SECONDS,
                            max_redirects: int = FETCH_REDIRECTS) -> AcquiredResource:
    """Shared public transport; interpretation/admission belongs to consumers.

    No retries, decompression or proxy headers. DNS validation and actual peer
    validation both precede every GET. Initial DNS elapsed time is included in
    the deadline; the OS DNS call itself is not cancellable.
    """
    if kind not in {'text', 'image', 'pdf'} or max_bytes <= 0 or timeout <= 0 or max_redirects < 0:
        raise ValueError('invalid acquisition policy')
    deadline = monotonic() + timeout
    current = _validate_public_url(url)
    for redirect_count in range(max_redirects + 1):
        parsed = urlparse(current)
        conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        host = parsed.hostname or ""
        port = parsed.port
        remaining = deadline - monotonic()
        if remaining <= 0: raise ToolError('fetch network timeout ceiling reached', code='resource_ceiling')
        conn = conn_cls(host, port=port, timeout=remaining)
        path = (parsed.path or "/") + (("?" + parsed.query) if parsed.query else "")
        deadline_timer = None
        try:
            conn.connect()
            connection_socket = conn.sock
            peer = ipaddress.ip_address(connection_socket.getpeername()[0])
            if not peer.is_global: raise ToolError('refusing actual local/private network peer', code='security_rejection')
            def set_remaining_timeout():
                remaining = deadline - monotonic()
                if remaining <= 0: raise ToolError('fetch network timeout ceiling reached', code='resource_ceiling')
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
            accept = {'text': 'text/html,text/plain,application/json,application/xml',
                      'image': 'image/png,image/jpeg,image/webp', 'pdf': 'application/pdf'}[kind]
            conn.request("GET", path, headers={"User-Agent": "ds41f-local-web-client/0.1", "Accept-Encoding": "identity", "Accept": accept})
            set_remaining_timeout()
            resp = conn.getresponse()
            if resp.status in {301, 302, 303, 307, 308}:
                loc = resp.getheader("Location")
                if not loc:
                    raise ToolError("redirect response missing Location header")
                if redirect_count == max_redirects:
                    raise ToolError('too many redirects', code='resource_ceiling')
                current = _validate_public_url(urljoin(current, loc))
                continue  # finally closes connection; never wait on a redirect body
            if resp.status >= 400:
                raise ToolError(f"fetch HTTP status {resp.status}", code='acquisition_failure')
            ctype = resp.getheader("content-type", "")
            if len(ctype) > FETCH_CONTENT_TYPE_CHARS:
                raise ToolError('Content-Type metadata exceeds 4096-character resource ceiling', code='resource_ceiling')
            if not _content_type_allowed(ctype, kind):
                raise ToolError(f"refusing non-{kind} content type {ctype!r}", code='unsupported_content')
            if resp.getheader('content-encoding', 'identity').strip().lower() not in ('identity', ''):
                raise ToolError('compressed fetch bodies are not supported (resource/decompression boundary)', code='unsupported_content')
            chunks = bytearray()
            while len(chunks) <= max_bytes:
                # Content-Length completion may already have closed the socket;
                # never set a timeout on its now-closed descriptor.
                if getattr(resp, 'length', None) == 0:
                    break
                set_remaining_timeout()
                chunk = resp.read1(min(65536, max_bytes + 1 - len(chunks)))
                if monotonic() >= deadline:
                    raise ToolError('fetch network timeout ceiling reached', code='resource_ceiling')
                if not chunk:
                    if getattr(resp, 'length', None) not in (None, 0):
                        raise ToolError('HTTP body ended before declared Content-Length', code='acquisition_failure')
                    break
                chunks.extend(chunk)
            # The extra byte establishes overflow; it is not an artifact byte.
            return AcquiredResource(url, current, ctype, bytes(chunks[:max_bytes]),
                                    len(chunks) > max_bytes, redirect_count)
        except (OSError, http.client.HTTPException) as exc:
            code = 'resource_ceiling' if monotonic() >= deadline else 'acquisition_failure'
            raise ToolError('fetch network timeout ceiling reached' if code == 'resource_ceiling' else 'network acquisition failed: ' + str(exc), code=code) from exc
        finally:
            if deadline_timer is not None: deadline_timer.cancel()
            conn.close()
    raise ToolError("too many redirects", code='resource_ceiling')


def _read_public_url(url: str, *, timeout: float = FETCH_TRANSFER_SECONDS,
                     max_bytes: int = FETCH_TEXT_BYTES, max_redirects: int = FETCH_REDIRECTS) -> AcquiredResource:
    """Text specialization of the shared acquisition boundary."""
    return acquire_public_resource(url, kind='text', timeout=timeout,
                                   max_bytes=max_bytes, max_redirects=max_redirects)


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
            data = resp.read(MCP_RESPONSE_BYTES + 1)
            if len(data) > MCP_RESPONSE_BYTES:
                raise ToolError('Hosted MCP response exceeds 1,000,000-byte resource ceiling', code='resource_ceiling')
            raw = data.decode("utf-8", "replace")
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
                    out.append({"title": str(item.get("title", ""))[:200], "url": str(item.get("url", "")), "snippet": " ".join(snippet.split())[:900], "provider_metadata": provider})
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
            results.append({"title": title[:200], "url": url, "snippet": snippet[:900], "provider_metadata": provider})
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

    def validate(self, arguments):
        if set(arguments) - {'query', 'max_results'}:
            raise ToolError('unknown web_search arguments')
        query = arguments.get('query')
        if not isinstance(query, str) or not query.strip():
            raise ToolError('web_search.query must be a non-empty string')
        maximum = arguments.get('max_results', 10)
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 20:
            raise ToolError('web_search.max_results must be an integer from 1 to 20')

    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        self.validate(arguments)
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("web_search.query must be a non-empty string")
        max_results = arguments.get("max_results", 10)
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1 or max_results > 20:
            raise ToolError("web_search.max_results must be an integer from 1 to 20")
        try:
            payload = self.provider.search(query.strip(), max_results=max_results, session_id=None if context is None else context.session_id)
        except Exception as exc:
            raise ToolError(str(exc), code='search_provider_failure') from exc
        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list):
            raise ToolError("search provider returned malformed results")
        bounded = []
        for r in raw_results[:max_results]:
            if not isinstance(r, dict):
                raise ToolError("search provider returned malformed result item")
            bounded.append({"title": str(r.get("title", ""))[:200], "url": str(r.get("url", "")), "snippet": str(r.get("snippet", ""))[:900], "provider_metadata": r.get("provider_metadata")})
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
            "description": "Fetch readable public http(s) text. Network resource ceiling 8 MiB; model-facing text is fitted by runtime context admission. Default discovery excerpt is 4096 characters (not a capacity ceiling); length can explicitly select more. Use artifact_id and offset for retained continuation without another download. For mixed Vision research keep early text focused: full-history Vision context is 8192 expanded positions. Supply exactly one of url or artifact_id. Private/local URLs and redirects are refused.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}, "artifact_id": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "length": {"type": "integer", "minimum": 1}}, "additionalProperties": False},
        },
    }

    def __init__(self, store=None):
        self.store = store

    def validate(self, arguments):
        from ds41f_mlx.web_binary_tools import validate_acquisition
        validate_acquisition(arguments, {'url', 'artifact_id', 'offset', 'length'})
        offset = arguments.get('offset', 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolError('fetch_url.offset must be a nonnegative integer')

        length = arguments.get('length', 4096)
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ToolError('fetch_url.length must be a positive integer')

    def run(self, arguments: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        self.validate(arguments)
        offset = arguments.get('offset', 0)
        artifact = None
        if self.store is not None:
            from ds41f_mlx.web_binary_tools import artifact_for
            artifact = artifact_for(arguments, self.store, 'text')
            resource = artifact.resource
        else:
            # Explicitly constructed transport-only clients retain compatibility.
            url = arguments.get('url')
            if not isinstance(url, str) or not url.strip():
                raise ToolError('fetch_url.url must be a non-empty string')
            resource = _read_public_url(url.strip())
        network_truncated = resource.truncated
        text = readable_text(resource.data.decode('utf-8', 'replace'), resource.content_type)
        # The acquired-byte and aggregate batch serialization bounds own memory;
        # runtime admission, not a fixed excerpt quota, selects consumed text.
        excerpt = text[offset:offset + arguments.get('length', 4096)]
        payload = {"url": resource.url, "source_url": resource.source_url,
                   "content_type": resource.content_type, "excerpt": excerpt,
                   'acquired_bytes': len(resource.data), 'sha256': resource.sha256,
                   'hash_scope': 'retained_original_bytes', 'redirects': resource.redirects,
                   'offset': offset, 'total_chars': len(text), 'returned_chars': len(excerpt),
                   'next_offset': offset + len(excerpt), 'has_more': offset + len(excerpt) < len(text), 'network_truncated': network_truncated,
                   'resource_truncated': False,
                   'context_truncated': False,
                   'termination_reason': 'network_resource_ceiling' if network_truncated else None}
        if artifact is not None:
            payload.update(artifact.metadata())
        return ToolResult(content=_json_dumps(payload), display={"tool": self.name, **payload})


def _tool_result_size(call_id, result):
    return len(_json_dumps({'message': {'role': 'tool', 'tool_call_id': call_id, 'content': result.content},
                            'display': {'id': call_id, **result.display}}).encode('utf-8')) + 2


def _bounded_tool_outcome(name, retained=None, executed=True):
    payload = {**(retained or {}), 'error_code': 'tool_result_resource_ceiling' if executed else 'resource_deferred',
               'error': 'Acquisition/result completed but serialization ceiling reached. Use retained artifact_id for focused local interpretation; no automatic download retry.' if executed else 'Batch resource/elapsed-time circuit breaker; this call was NOT executed.'}
    return ToolResult(_json_dumps(payload), {'tool': name, **payload})


def _outcome_reservations(prepared):
    return [_tool_result_size(call_id, _bounded_tool_outcome(name, {'artifact_id': '0' * 64, 'sha256': '0' * 64}))
            for call_id, name, _ in prepared]


class ToolRegistry:
    def __init__(self, tools: list[ClientTool] | None = None) -> None:
        self.tools = {tool.name: tool for tool in (tools if tools is not None else _default_tools())}

    @property
    def declarations(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self.tools.values()]

    def validate_calls(self, calls):
        seen = set()
        prepared = []
        if not calls or len(_json_dumps(calls).encode('utf-8')) > TOOL_CALL_BYTES:
            raise ToolError('nonempty tool batch exceeds 1 MiB call serialization ceiling', code='resource_ceiling')
        # Validate the WHOLE batch before any external effect. A malformed later
        # call must not abandon an already-completed earlier acquisition receipt.
        for call in calls:
            if not isinstance(call, dict):
                raise ToolError('tool call must be an object')
            call_id = call.get("id")
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = fn.get("name")
            if not isinstance(call_id, str) or not call_id or len(call_id) > 256:
                raise ToolError("tool call id must be a non-empty string of at most 256 characters")
            if call_id in seen:
                raise ToolError(f"duplicate tool call id {call_id!r}")
            seen.add(call_id)
            if not isinstance(name, str) or name not in self.tools:
                raise ToolError(f"unknown tool {name!r}")
            raw = fn.get('arguments', '{}')
            if not isinstance(raw, str):
                raise ToolError('tool arguments must be a JSON string')
            args = _require_object(raw)
            validator = getattr(self.tools[name], 'validate', None)
            if validator is not None:
                validator(args)  # local argument checks only; no DNS/network
            prepared.append((call_id, name, args))
        if sum(_outcome_reservations(prepared)) > TOOL_RESULT_BYTES:
            raise ToolError('batch outcome serialization reservation exceeds resource ceiling', code='resource_ceiling')
        return prepared

    def execute_calls(self, calls: list[dict[str, Any]], *, session_id: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prepared = self.validate_calls(calls)
        messages, displays = [], []
        context = ToolContext(session_id=session_id)
        size_of, bounded_outcome = _tool_result_size, _bounded_tool_outcome

        # Reserve actual worst-case classified receipt envelopes for all remaining
        # calls before the first effect, not a fixed call-count or token estimate.
        reserves = _outcome_reservations(prepared)
        remaining = sum(reserves)
        if remaining > TOOL_RESULT_BYTES:
            raise ToolError('batch outcome serialization reservation exceeds resource ceiling', code='resource_ceiling')
        used = 0
        deadline = monotonic() + TOOL_BATCH_SECONDS
        for (call_id, name, args), reserved in zip(prepared, reserves):
            remaining -= reserved
            if used + remaining + reserved >= TOOL_RESULT_BYTES or monotonic() >= deadline:
                result = bounded_outcome(name, executed=False)
            else:
                try:
                    result = self.tools[name].run(args, context)
                except Exception as exc:
                    code = exc.code if isinstance(exc, ToolError) else 'execution_failure'
                    payload = {'error': str(exc), 'error_code': code, 'tool': name}
                    result = ToolResult(_json_dumps(payload), {'tool': name, 'error': str(exc), 'error_code': code})
            size = size_of(call_id, result)
            if used + size + remaining > TOOL_RESULT_BYTES:
                retained = {key: result.display[key] for key in ('artifact_id', 'sha256') if key in result.display}
                result = bounded_outcome(name, retained)
                size = size_of(call_id, result)
            used += size
            messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': result.content})
            displays.append({'id': call_id, **result.display})
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


def _default_tools():
    from pathlib import Path
    from ds41f_mlx.web_artifacts import ArtifactStore
    from ds41f_mlx.web_binary_tools import FetchImageTool, FetchPDFTool, pdf_dependency_identity
    # Fail configuration BEFORE declaring/starting PDF acquisition capability.
    pdf_dependency_identity()
    path = os.environ.get('DS41F_WEB_ARTIFACT_DB', str(Path.home() / '.local/share/ds41f/web-artifacts.sqlite3'))
    store = ArtifactStore(path)
    return [WebSearchTool(_provider_from_env()), FetchURLTool(store), FetchImageTool(store), FetchPDFTool(store)]


def registry_from_env() -> ToolRegistry:
    return ToolRegistry()
