# M19 local web client

M19 adds a separate local browser client above the qualified ds41f runtime API.

```text
browser -> ds41f_mlx.web -> qualified ds41f HTTP API -> stateful Chat Completions session
                         -> client tool executor -> Exa/Parallel/fetch_url -> tool result to same session
```

The model runtime remains the inference/session authority. `ds41f_mlx.web` is a client application: it serves static HTML/CSS/JS, proxies operator actions to the public HTTP API, keeps only a protocol/presentation transcript needed to submit exact stateful continuations, and executes tools outside the runtime process.

## Launch

Start the qualified runtime and the web client as separate processes:

```bash
python -m ds41f_mlx.serve --host 127.0.0.1 --port 8000
python -m ds41f_mlx.web --host 127.0.0.1 --port 8080 --runtime-url http://127.0.0.1:8000
```

Open <http://127.0.0.1:8080/>.

Environment knobs:

- `DS41F_RUNTIME_URL` (default `http://127.0.0.1:8000`)
- `DS41F_WEB_HOST` / `DS41F_WEB_PORT`
- `DS41F_WEB_SEARCH_PROVIDER=auto` (default), `exa`, or `parallel`
- `DS41F_EXA_API_KEY` optional; Exa hosted MCP anonymous access was observed working for `web_search_exa` during M19 integration
- `DS41F_PARALLEL_API_KEY` optional/usually required; Parallel hosted MCP returned HTTP 401 without credentials during M19 integration
- `DS41F_PARALLEL_MCP_TOOL` optional override when Parallel exposes multiple search tools

## OpenCode comparison

Current OpenCode behavior inspected for M19 uses hosted MCP search services rather than embedding a scraper: Exa and Parallel are the primary web-search providers, provider override is supported, and automatic provider choice is stable for a session instead of changing randomly between adjacent searches. ds41f adopts that architecture in Python without copying OpenCode code.

For `DS41F_WEB_SEARCH_PROVIDER=auto`, ds41f deterministically hashes the ds41f session id to choose Exa or Parallel as the first provider. If that provider fails, times out, is unauthorized/rate-limited, or returns unusable content, one bounded failover attempt is made to the other provider. The tool result records the provider that actually served the request and any fallback errors.

## Boundary

Do not add filesystem, shell, network, or other external actions to `ds41f_mlx.serve` for client convenience. New tools belong in the client/tool layer and must be declared through Chat Completions `tools`, validated, executed outside the runtime process, and returned as tool messages to the same stateful session.

The browser stores the session id and presentation transcript in localStorage so refresh can continue when the ds41f session still exists. If the runtime reports an unknown/closed session, the UI surfaces that state; it does not silently create a fresh prefill as a substitute for lost model state.

## Tools

`ds41f_mlx.web_tools.ToolRegistry` provides generic dispatch:

1. declare function schemas;
2. detect `finish_reason == "tool_calls"` from the qualified API response;
3. validate non-empty unique tool-call ids, known tool names, and JSON-object arguments;
4. execute client-side implementations;
5. append `role: tool` results to the same transcript/session;
6. repeat until the model returns an ordinary assistant completion;
7. fail clearly after a bounded number of tool rounds.

Initial tools:

- `web_search`: bounded public web discovery through Exa hosted MCP or Parallel hosted MCP, returning JSON with provider, selected provider, query, title, URL, snippet/context, and bounded provider metadata (max 5 results).
- `fetch_url`: bounded text excerpt fetch for public `http(s)` URLs.

`web_search` normalizes hosted MCP text output into a compact source list. It does not expose raw MCP responses to the model.

## Public URL retrieval boundary

`fetch_url` is not a general browser. It validates the initial URL and every redirect target before fetching. Destinations resolving to loopback, private, link-local, multicast, reserved, or unspecified address space are refused. Redirects are bounded. Timeout, response size, accepted content type, and returned excerpt size are bounded.

## Persistence

The UI exposes explicit save/restore actions mapped directly to the existing same-backend KV persistence API. Browser metadata and server-side KV artifacts remain distinct; restored sessions start with an empty browser presentation transcript unless the operator separately retained it.

## Current limitations

- UI token streaming is not implemented.
- Parallel anonymous hosted MCP access was not observed; use `DS41F_PARALLEL_API_KEY` where needed.
- No filesystem, coding, shell, multimodal, multi-user auth, batching, MTP, DSpark, or speculative decoding features are added.
