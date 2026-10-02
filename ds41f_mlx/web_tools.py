"""Client-side tool runtime for the ds41f local web client.

This module deliberately lives above the qualified model runtime.  It declares
function tools through the public Chat Completions API, validates model-emitted
calls, executes local-client implementations, and returns tool results as
ordinary tool messages to the same stateful ds41f session.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import os
import socket
from html import unescape
from html.parser import HTMLParser
from typing import Any, Callable, Protocol
from urllib.parse import quote_plus, urlparse
from urllib.request import Request, urlopen


class ToolError(ValueError):
    pass


@dataclass(frozen=True)
class ToolResult:
    content: str
    display: dict[str, Any]


class ClientTool(Protocol):
    name: str
    schema: dict[str, Any]
    def run(self, arguments: dict[str, Any]) -> ToolResult: ...


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


def _public_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ToolError("only http(s) URLs with a host are allowed")
    host = parsed.hostname
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolError(f"could not resolve URL host {host!r}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ToolError("refusing to fetch local/private network URL")
    return url


class SearchProvider(Protocol):
    name: str
    def search(self, query: str, *, max_results: int) -> list[dict[str, str]]: ...


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_a = False
        self._href: str | None = None
        self._title: list[str] = []
        self._snippet_next = False
        self._snippet: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = dict(attrs)
        cls = attrs_d.get("class") or ""
        if tag == "a" and "result__a" in cls:
            self._in_a = True; self._href = attrs_d.get("href"); self._title = []
        elif tag in {"a", "div"} and "result__snippet" in cls:
            self._snippet_next = True; self._snippet = []

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._title.append(data)
        elif self._snippet_next:
            self._snippet.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_a:
            title = " ".join("".join(self._title).split())
            if title and self._href:
                self.results.append({"title": title, "url": self._href, "snippet": ""})
            self._in_a = False; self._href = None; self._title = []
        elif tag in {"a", "div"} and self._snippet_next:
            snippet = " ".join("".join(self._snippet).split())
            if snippet and self.results and not self.results[-1].get("snippet"):
                self.results[-1]["snippet"] = snippet
            self._snippet_next = False; self._snippet = []


class WikipediaSearchProvider:
    """Reliable no-key discovery provider using the public MediaWiki API."""
    name = "wikipedia"

    def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit={max_results}&srsearch={quote_plus(query)}"
        req = Request(url, headers={"User-Agent": "ds41f-local-web-client/0.1"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read(512_000).decode("utf-8", "replace"))
        out: list[dict[str, str]] = []
        for item in (data.get("query", {}).get("search") or [])[:max_results]:
            title = str(item.get("title", ""))
            page = title.replace(" ", "_")
            snippet = unescape(" ".join(str(item.get("snippet", "")).replace("<span class=\"searchmatch\">", "").replace("</span>", "").split()))
            out.append({"title": title, "url": f"https://en.wikipedia.org/wiki/{quote_plus(page)}", "snippet": snippet})
        return out


class BraveSearchProvider:
    name = "brave"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        url = f"https://api.search.brave.com/res/v1/web/search?q={quote_plus(query)}&count={max_results}"
        req = Request(url, headers={"User-Agent": "ds41f-local-web-client/0.1", "Accept": "application/json", "X-Subscription-Token": self.api_key})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read(512_000).decode("utf-8", "replace"))
        return [{"title": str(r.get("title", "")), "url": str(r.get("url", "")), "snippet": str(r.get("description", ""))} for r in (data.get("web", {}).get("results") or [])[:max_results]]


class DuckDuckGoHTMLSearchProvider:
    """Optional no-key provider; may be blocked by upstream anti-bot controls."""
    name = "duckduckgo_html"

    def search(self, query: str, *, max_results: int) -> list[dict[str, str]]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        req = Request(url, headers={"User-Agent": "ds41f-local-web-client/0.1"})
        with urlopen(req, timeout=10) as resp:
            html = resp.read(512_000).decode("utf-8", "replace")
        parser = DuckDuckGoHTMLParser(); parser.feed(html)
        return parser.results[:max_results]


class WebSearchTool:
    name = "web_search"
    schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web for current information. Returns bounded results with title, URL, and snippet; does not fetch full pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Number of results to return."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, provider: SearchProvider | None = None) -> None:
        self.provider = provider or WikipediaSearchProvider()

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolError("web_search.query must be a non-empty string")
        max_results = arguments.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 5:
            raise ToolError("web_search.max_results must be an integer from 1 to 5")
        results = self.provider.search(query.strip(), max_results=max_results)
        bounded = [{"title": str(r.get("title", ""))[:200], "url": str(r.get("url", ""))[:500], "snippet": str(r.get("snippet", ""))[:500]} for r in results[:max_results]]
        payload = {"query": query.strip(), "provider": self.provider.name, "results": bounded}
        return ToolResult(content=_json_dumps(payload), display={"tool": self.name, **payload})


class FetchURLTool:
    name = "fetch_url"
    schema = {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a small text excerpt from a public http(s) URL discovered by search. Local/private network addresses are refused.",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"], "additionalProperties": False},
        },
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ToolError("fetch_url.url must be a non-empty string")
        safe_url = _public_url(url.strip())
        req = Request(safe_url, headers={"User-Agent": "ds41f-local-web-client/0.1"})
        with urlopen(req, timeout=10) as resp:
            ctype = resp.headers.get("content-type", "")
            if "text" not in ctype and "html" not in ctype and "json" not in ctype:
                raise ToolError(f"refusing non-text content type {ctype!r}")
            text = resp.read(80_000).decode("utf-8", "replace")
        excerpt = " ".join(text.split())[:4000]
        payload = {"url": safe_url, "excerpt": excerpt}
        return ToolResult(content=_json_dumps(payload), display={"tool": self.name, **payload})


class ToolRegistry:
    def __init__(self, tools: list[ClientTool] | None = None) -> None:
        self.tools = {tool.name: tool for tool in (tools or [WebSearchTool(), FetchURLTool()])}

    @property
    def declarations(self) -> list[dict[str, Any]]:
        return [tool.schema for tool in self.tools.values()]

    def execute_calls(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seen: set[str] = set(); messages: list[dict[str, Any]] = []; displays: list[dict[str, Any]] = []
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
                result = self.tools[name].run(args)
            except Exception as exc:
                payload = {"error": str(exc), "tool": name}
                result = ToolResult(content=_json_dumps(payload), display={"tool": name, "error": str(exc)})
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result.content})
            displays.append({"id": call_id, **result.display})
        return messages, displays


def registry_from_env() -> ToolRegistry:
    provider_name = os.environ.get("DS41F_WEB_SEARCH_PROVIDER", "wikipedia")
    if provider_name == "wikipedia":
        provider: SearchProvider = WikipediaSearchProvider()
    elif provider_name == "duckduckgo_html":
        provider = DuckDuckGoHTMLSearchProvider()
    elif provider_name == "brave":
        key = os.environ.get("DS41F_BRAVE_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("DS41F_BRAVE_SEARCH_API_KEY is required for DS41F_WEB_SEARCH_PROVIDER=brave")
        provider = BraveSearchProvider(key)
    else:
        raise RuntimeError(f"unsupported DS41F_WEB_SEARCH_PROVIDER {provider_name!r}")
    return ToolRegistry([WebSearchTool(provider), FetchURLTool()])
