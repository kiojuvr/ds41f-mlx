# M19 local web client

M19 added the original separate local browser client above the qualified ds41f runtime API.
This document retains M19's historical acceptance. For the current streaming,
Vision, recovery and save/restore application, see [Web application](web-application.md).

```text
browser -> ds41f_mlx.web -> qualified ds41f HTTP API -> stateful Chat Completions session
                         -> client tool executor -> Exa/Parallel/fetch_url -> tool result to same session
```

The model runtime remains the inference/session authority. `ds41f_mlx.web` is a client application: it serves static HTML/CSS/JS, proxies operator actions to the public HTTP API, keeps only a protocol/presentation transcript needed to submit exact stateful continuations, and executes tools outside the runtime process.

## Launch

Start the qualified runtime and the web client as separate processes. Select the
runtime interpreter from the existing runtime qualification, **not** automatically
from the repository's development `.venv`. The runtime interpreter must contain
an installed `deepseek-recipe` package with an importable native extension; the
recipe source checkout alone is insufficient. The client may use a different
interpreter (FastAPI/uvicorn and the client dependencies, no model/native package).

```bash
# Set these to your qualified runtime and client interpreters.
export DS41F_RUNTIME_PYTHON=/path/to/qualified/environment/bin/python
export DS41F_CLIENT_PYTHON=/path/to/client/environment/bin/python

# Run in the same environment/working directory as the launch below.
"$DS41F_RUNTIME_PYTHON" - <<'PY'
import sys, importlib.util
from ds41f_mlx.config import load_runtime_config
load_runtime_config().apply_import_paths()
spec = importlib.util.find_spec("deepseek_recipe")
print("Python:", sys.executable, flush=True)
print("deepseek_recipe:", spec.origin if spec else "NOT FOUND", flush=True)
import deepseek_recipe._native as native
print("native:", native.__file__)
PY
# Stop here if the import fails; compare the interpreter to qualification evidence
# before attempting a build or installing build-only dependencies.
"$DS41F_RUNTIME_PYTHON" -m ds41f_mlx.serve --print-config
"$DS41F_RUNTIME_PYTHON" -m ds41f_mlx.serve --host 127.0.0.1 --port 8000
# In another terminal:
"$DS41F_CLIENT_PYTHON" -m ds41f_mlx.web --host 127.0.0.1 --port 8080 --runtime-url http://127.0.0.1:8000
```

`--print-config` includes the Python executable and resolved package origins.
If an installed recipe package is absent, the launcher falls back to the source
checkout's Python directory, which may lack `_native`. A successful path check
or client unit test is not proof that this interpreter can run the model.

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

The original M19 browser stored the session id and presentation transcript in localStorage so refresh could continue when the ds41f session still existed. If the runtime reports an unknown/closed session, the UI surfaces that state; it does not silently create a fresh prefill as a substitute for lost model state.

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

## Historical M19 persistence

The original UI exposed explicit save/restore actions mapped directly to the existing same-backend KV persistence API. Browser metadata and server-side KV artifacts remain distinct; restored sessions start with an empty browser presentation transcript unless the operator separately retained it.

## Final real-model acceptance (2026-10-02)

**PASS**: the canonical runtime and separate web client completed a real Exa
hosted-MCP tool loop with DeepSeek-V4.1-Flash, then an ordinary following turn on
`sess_3301fc00405a430a8f5e8a218dd9fa70`. The model generated
`web_search({"query":"DeepSeek V4.1 Flash official announcement release date"})`
(call id `call_6bb0829a-4557-4957-9a06-0d24dcddb647_0`). Exa returned the official
announcement with a September 10, 2026 snippet; the model cited it, then repeated
the date on the following turn without another tool call. Runtime request count
advanced from 2 (search + tool continuation) to 3. Session close and SIGTERM
shutdown of both services completed cleanly.

Evidence:

- [Environment reconciliation](../artifacts/m19/m19-environment-reconciliation.json)
- [Final bounded real-model acceptance](../artifacts/m19/m19-real-model-acceptance.json)
- [Earlier environment failure and independent provider checks](../artifacts/m19/m19-client-acceptance.json) (unchanged historical evidence)
- [First qualified-runtime attempt](../artifacts/m19/m19-real-model-attempt-1.json): broad question exceeded the bounded tool-round limit after search/fetch calls.
- [Second attempt](../artifacts/m19/m19-real-model-attempt-2.json): search loop passed; disabling tools on the following turn changed the recipe prompt and continuation was rejected. The final attempt kept declarations unchanged.

The original `_native` failure used the repository `.venv/bin/python`, which had
no installed recipe package and fell back to the recipe source checkout without
its native extension. The M18-qualified interpreter still imported the installed
package/native extension and started the real canonical server. The earlier
maturin/OpenCV attempt was an unnecessary build attempt in the wrong/incomplete
environment, not a blocker in the qualified runtime. No rebuild or system
dependency installation was performed for final acceptance.

M18 qualification remains valid: runtime-source digest, checkpoint fingerprint,
oMLX identity, recipe identity, installed versions/origins, and production
selectors match M18. Installed package RECORD hashes verify and file change times
predate M18; there is no evidence of installed package/native changes. M18 did not
capture a binary hash, so historical byte-for-byte verification is unavailable;
the current native hash is now recorded. No expensive runtime requalification was
rerun. Only documentation and acceptance artifacts changed.

## Historical M19 limitations (superseded by current application documentation)

- Keep tool declarations/settings unchanged within a live stateful session; disabling tools mid-session changes the recipe prefix and may reject continuation. Close/create a session to change settings. An ordinary conversational turn can leave tools enabled without using them.
- UI token streaming is not implemented.
- Parallel anonymous hosted MCP access was not observed; use `DS41F_PARALLEL_API_KEY` where needed.
- No filesystem, coding, shell, multimodal, multi-user auth, batching, MTP, DSpark, or speculative decoding features are added.
