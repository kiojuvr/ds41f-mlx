# M19 local web client

M19 adds a separate local browser client above the qualified ds41f runtime API.

```text
browser -> ds41f_mlx.web -> qualified ds41f HTTP API -> stateful Chat Completions session
                         -> client tool executor -> tool result to same session
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
- `DS41F_WEB_SEARCH_PROVIDER=wikipedia` (default), `brave`, or `duckduckgo_html`
- `DS41F_BRAVE_SEARCH_API_KEY` when using `brave`

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
6. repeat until the model returns an ordinary assistant completion.

Initial tools:

- `web_search`: bounded public web discovery, returning JSON with provider, title, URL, and snippet (max 5 results).
- `fetch_url`: bounded text excerpt fetch for public `http(s)` URLs. DNS results resolving to loopback/private/link-local/multicast/reserved/unspecified addresses are refused.

The default provider is the no-key public Wikipedia API because it is reliable for local zero-configuration discovery. A configured Brave Search provider is available for broader web search (`DS41F_WEB_SEARCH_PROVIDER=brave` plus `DS41F_BRAVE_SEARCH_API_KEY`). A DuckDuckGo HTML adapter is also available for local experimentation but may be blocked by upstream anti-bot controls. The provider interface is narrow so providers can change without changing the model-runtime contract.

## Persistence

The UI exposes explicit save/restore actions mapped directly to the existing same-backend KV persistence API. Browser metadata and server-side KV artifacts remain distinct; restored sessions start with an empty browser presentation transcript unless the operator separately retained it.
