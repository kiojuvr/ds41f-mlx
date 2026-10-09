# Ordinary MTP serving: state ownership

## Enable explicitly

`standard-off` remains the default. `mtp-singleton-v1` retains its existing
session/fence/sequence contract. The separate **`mtp-serving-v1`** opt-in exposes
ordinary `POST /v1/chat/completions`, `/v1/models`, and `/health`, without public
sessions, sequence headers, certificates, or a dedicated client.

Use the existing normal-local MTP environment and asset admission, not a new
package or dependency stack:

```sh
export DS41F_CHECKPOINT=/absolute/official/DeepSeek-V4.1-Flash
/path/to/mtp-env/bin/ds41f inspect --profile mtp-serving-v1
/path/to/mtp-env/bin/ds41f start --profile mtp-serving-v1

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"Reply with exactly OK."}],"max_tokens":16}'
```

The source admission inventory in `third_party/mtp/normal-local.json` is refreshed
for this implementation. After updating an existing source clone, refresh its
local seal with the **unchanged admitted recipe wheel**:

```sh
/path/to/mtp-env/bin/python -m ds41f_mlx.mtp_identity seal \
  --wheel third_party/mtp/deepseek_recipe-0.1.1-cp310-abi3-macosx_11_0_arm64.whl
```

Sealing verifies repository-admitted source/dependencies; it is not a new release
qualification. Native recipe, oMLX archive, dependency locks, checkpoint admission,
hardware envelope, and native topology are unchanged.

### Direct trusted private-LAN serving

```sh
/Volumes/SDXC-512/m53r-local-venv/bin/python -m ds41f_mlx.ops start \
  --profile mtp-serving-v1 --host 0.0.0.0 --port 8000
```

From another trusted LAN client use `http://<Mac-Studio-LAN-IP>:8000/v1`.
Local `http://127.0.0.1:8000/v1` clients continue to work with the wildcard bind.
A concrete LAN interface address (or IPv6 bind) can be supplied instead; clients
must then use a reachable bound interface. The default remains `127.0.0.1`.
Ordinary Host admission accepts one syntactically valid IP/IPv6/DNS authority,
with optional port, rather than requiring `127.0.0.1:<port>` or matching the bind
address. Missing, duplicate and malformed Host headers and Origin remain rejected.
No proxy, SSH tunnel, VPN or additional transport daemon is required.

`mtp-serving-v1` targets **trusted private LAN / single-operator deployments**.
It provides no authentication, TLS, adversarial multi-user isolation, or Internet
exposure support. Wildcard binding listens on all interfaces; select a private
interface and/or configure the host firewall as appropriate for your deployment.
`mtp-singleton-v1` retains its literal `127.0.0.1` bind and exact Host contract;
`standard-off` is unchanged. This removes historical singleton transport
constraint leakage, not an execution or dependency admission requirement.

This path keeps one worker, one executable native singleton and greedy/thinking-off
text. Ordinary prompt + output is bounded by **1,048,576 tokens** (or a smaller
checkpoint-configured range); maximum output is **393,216 tokens**. The old
8192 total / 768 output ordinary bounds were historical bounded serving scope,
not the model/runtime capability.
`serving/capacity.py` owns the established qualified text envelope; recipe capacity
admission, ordinary admission, Scheduler paired-cache sizing/publication and
`/v1/models` use that authority. `ds41f inspect --profile mtp-serving-v1` reports
these as `serving_limits`; its nested identity's historical singleton limits are
not ordinary admission policy. There is no unlimited-context claim. Ordinary
JSON body/main text ingress is bounded to **16 MiB**, rather than singleton's
qualification-era 1 MiB, so normal long text can reach token admission. This is
resource headroom, not a byte-to-token approximation; tool-result/argument bounds
remain unchanged. Ordinary generic function tools are supported, including
multiple declarations, arbitrary names and recipe-representable JSON Schema-style
parameters. Missing `max_tokens`
uses 128 (unchanged default generation budget). The maximum output is a capability
ceiling, not mandatory generation or a KV/DSpark/output allocation reservation.
`1 <= requested output <= 393,216` and `prompt + requested output <= 1,048,576`
are separate admission rules: 600,000 + 393,216 fits; 900,000 + 393,216 does not.
Actual state grows with prompt and tokens actually executed; EOS, recipe semantic
stop, tool calls and application termination can finish far below the maximum.
Thought/runaway/agent-loop policy, cancellation budgets and operator cost policy
are separate responsibilities, not reasons to lower this capability ceiling.
No new runaway-protection framework is introduced. Unsupported controls are rejected, not silently implemented. Applications
own their messages and branches. Independent requests may queue; they do not join
a shared native MTP batch. No persistence, crash recovery, exactly-once effects, or
implicit HTTP retry identity is promised.

## Ordinary function tools and OpenCode

DeepSeek recipe is the authority for declaration conversion, conversation rendering,
argument/tool-call parsing, tool result semantics and JSON/SSE response projection.
Serving admission enforces the bounded request surface and binds `role: tool`
results to prior assistant `tool_call_id` values; it does not validate arbitrary
parameter schemas or implement a second tool grammar. Multiple completed calls
are supported within recipe capabilities and the existing body/context budgets.
OpenCode's empty assistant `reasoning_content` is accepted through this same recipe
path; generation remains thinking-off.

**Tool execution is the client's responsibility.** ds41f does not execute file,
shell, browser, Web or MCP tools. After a canonical recipe-completed assistant
call, send its assistant message and matching tool results in the next ordinary
conversation request. No session ID or ds41f certificate protocol is required.
The same Scheduler settlement/publication and paired prefix rules apply to tool
round trips and changed-result branches. Partial speculative calls do not publish
application history or reusable generated state.

`mtp-singleton-v1` retains its pinned weather declaration, bounded call/effect
semantics and certificate contract. The weather schema is qualification evidence,
not the ordinary serving tool API. Its own 8192-total / 768-output ceiling remains
unchanged; only ordinary serving promotes the existing long-context capability.

OpenCode provider-local model key and display `name` need not match the model ID.
Only the wire request's `model` participates in ds41f admission. For OpenCode
1.18.30, configure `id` (the wire modelID) separately from presentation metadata:

```json
{
  "model": "local/deepseek-v4.1-flash",
  "enabled_providers": ["local"],
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "local"},
      "models": {
        "deepseek-v4.1-flash": {
          "id": "deepseek-v4.1-flash",
          "name": "DeepSeek V4.1 Flash",
          "tool_call": true,
          "limit": {"context": 1048576, "output": 393216}
        }
      }
    }
  }
}
```

Keep agent temperature zero and the complete encoded prompt plus output reservation
within the advertised envelope. No reduced agent/tool schema is required merely
to fit the obsolete 8K serving scope. Keep the provider local-only and fail closed;
`enabled_providers` prevents fallback to an Internet provider.

## Owners and production boundaries

- `ProductionScheduler` **is a subclass of pinned oMLX Scheduler**, not a second
  scheduler beside it. Upstream Request, waiting/running registries, queue cap,
  UID mappings, deferred abort admission, counters, and PagedCacheManager remain
  the generic owners. Its external-prefill/decode/completion hooks adapt the
  qualified core instead of executing generic prefill, stop parsing, completion
  storage, or cache-corruption replay recovery.
- The existing recipe HTTP conversion/encoding/projection remains authoritative.
  `ProductionMTPBackend` bridges asynchronous delivery to Scheduler steps on the
  existing shielded worker. A bounded HTTP preparation semaphore protects recipe
  conversion and is released **before** executable admission; it is not an
  execution queue. JSON and SSE disconnect cleanup targets one Scheduler ID.
- P5 still revokes the same-live DwarfStar producer. An injected batch factory
  makes its physical adapter use **the Scheduler's sole BatchGenerator**.
  `OMLXMTPGenerationSession` is a subordinate physical/semantic adapter, with no
  independent waiting queue, scheduling policy, or reusable cache authority.
- `PairedCheckpointAuthority` owns paired payload integrity. Upstream PagedCache
  owns hash lookup, acquisition references, capacity and LRU eviction. Its root-tail
  block size follows the authoritative context envelope, so long complete pairs
  remain discoverable without splitting/repacking state. Four complete checkpoints
  are retained (plus upstream's reserved null block), with no independent prefix
  index or eviction policy. `total_tokens_cached` counts actual retained frontiers,
  not four times the block-size capacity. Hash-drop/clear hooks drop the
  associated paired payload. Target-only SSD restoration is deliberately absent.

The existing thin HTTP boundary is retained. Importing the pinned full oMLX HTTP
server requires its unrelated, unadmitted `openai_harmony` dependency chain;
restoring that entire server is not necessary to use Scheduler/PagedCache and
would introduce an additional protocol/dependency surface. No parallel DeepSeek
request, tool, or response grammar is introduced.

## Paired capture and restore

A checkpoint binds admitted execution/tokenizer/recipe identity, complete encoded
IDs through C, all forty target layers' seven physical slots and layout/padding
metadata, and all three committed DSpark rings at **the same C**. Ring capacity is
128; keys retain absolute-modulo physical order, not chronological rotation.
Draft KV, proposal/verify stashes, queued future predictions, semantic horizon and
RNG state are never cached.

Capture creates frozen slot/ring metadata and independent array/cache handles.
Pinned MLX arrays are functional values, but backing sharing is **not** a retention
budget assumption: materialized `mx.array` snapshots can allocate a full copy.
Mutable cache containers and ring objects are never shared. Acquisition
constructs fresh target and ring objects/array handles and materializes them before
use. Native array indexed updates and wrapped ring appends were regression-tested
not to change the retained snapshot. There is no numeric pack/unpack, whole-cache
repack, model forward, or prompt replay in capture/restore.

Lookup considers only checkpoints through N-2. For exact repeated prompts this
leaves one genuine suffix-prefill token plus P5's held-out terminal. A fresh
prefill seals at N-2, captures a **pending, undiscoverable** pair, then continues
via the existing P6 committed-owner transfer to N-1. P5 consumes terminal N-1
once. This avoids both recurrent N-to-N-1 trimming and synthetic P5 certificates
for a zero-length append. Settled generated checkpoints can serve append requests
when their complete encoded prefix matches. Publication requires room for a genuine
suffix-prefill token, P5 terminal and at least one reserved output token: C+3 must
fit the authoritative context ceiling. Near the ceiling only the earlier eligible
prompt pair is retained. An unusable ceiling checkpoint is not published.

Incomplete, damaged, incompatible, or missing paired payloads are invalidated
and become a shorter upstream hit or miss. A target hash hit alone is never an
MTP hit. Edits, shortened histories and branches never rewind live rings.

## Lifecycle and publication

```text
recipe-prepared request -> Scheduler waiting
  -> paired acquisition (fresh mutable handles) or fresh state
  -> DwarfStar/P6 sealed append + pending immutable prompt capture
  -> P5 producer revocation / Scheduler-owned native batch + DSpark install
  -> qualified generation + consuming recipe guard
  -> canonical quiescence, prediction retirement, semantic settlement
  -> immutable paired publication or burn
  -> request/UID retirement and immutable JSON/SSE delivery projection
```

Only Scheduler completion after qualified settlement publishes either pending
prompt or generated state. Backend finish and SSE delivery do not authorize it.
Cancellation waits for in-flight worker mutation, then settles the exact request
or burns state. A mutation/settlement/retirement failure burns its aliases, clears
reuse and fails the engine closed; it never silently replays or retries mutable
state. Eviction releases only immutable snapshots, never an active request or
application history. Native owner removal precedes publication; request delivery
can finish later. The worker can progress independently of a stalled client.

`/health` exposes bounded queue/activity and upstream cache counters; official
recipe usage reports the **actual paired restored token count**, independently of
raw hash-index hit counters. Ordinary settlement logs record request ID, reused
frontier, canonical frontier, publication and cancellation disposition.

## Long-context promotion evidence and retention

This is integration of existing capability, **not a new long-context qualification**.
The [very-long-context evidence](very-long-context-production-qualification.md)
establishes DwarfStar/P5/P6 same-list execution, deferred append, ownership,
resource lifetime and cancellation through **1,048,576 consumed tokens**.
[M25 native MTP evidence](milestone-25-mtp-decision.md) includes real 200,000-token
prefix execution; current canonical state/settlement rules reuse M50/M51R/M52R
and their regression evidence. These components are composed through the existing
ProductionScheduler, not numerically re-proved. No model-execution or MTP algorithm
code changes are needed for this serving promotion. The serving prefill bridge
needed one integration repair: its short-context hidden-tap wrapper is not a
qualified suffix-math layer and the deferred decoder intentionally computes only
terminal cones. Bulk prefill now runs the original admitted layers, then the
existing ordinary P6 append produces the final 128 rows at all DSpark tap layers.
Native absolute-offset ring initialization consumes those rows once, fills every
physical slot and reaches the prompt-capture frontier. Earlier passive ring
handles can be replaced before P5; no live native state is rewound. Reused short
suffixes still append to restored rings directly. No token is replayed, and no
late-layer full-prompt hidden tensor or new prefill executor is introduced. Task A's recipe-authoritative
generic declarations, tool calls/results and SSE remain unchanged.

The existing four-pair retention bound is kept deliberately. Historical measured
near-1M packed target state is about 946 MB per complete state, plus three bounded
128-row DSpark rings and encoded-token metadata. Even full independent array copies
therefore retain roughly 4 GB of target/ring state, not four models or four prefill
arenas, within the admitted 512 GiB host's established headroom. A pinned-MLX
16 MiB array snapshot check confirms materialized copying (not zero-cost sharing)
and immutable retained contents after mutable-handle updates. Active execution is
not evictable; acquisition clones before releasing its upstream lease. Hash drop,
clear and failure burn release the paired immutable payload through existing
PagedCache callbacks. No second cache, byte index, replay or whole-cache repack is
introduced. This bounded policy does not promise multi-user memory admission.

## Acceptance and regression

### Long-context production-path acceptance (2026-10-09)

Actual admitted normal-local environment, unchanged official asset and native
packages, ordinary HTTP and OpenCode. Compact receipt:
`artifacts/long-mtp-serving/acceptance.json`; raw server/client receipts remain in
that directory. Only the production delta was exercised, not a context ladder:

- Ordinary maintenance prompt **17,110 tokens** executed. Repeat restored paired
  **17,108**, appended turn **17,114**, and changed suffix **17,114** tokens.
- One long SSE disconnect settled at canonical **17,130**, with cached **17,114**,
  publication after settlement, no error and no live request left. Subsequent JSON
  completed. The earlier exact prompt had been evicted under four-pair LRU: its
  safe miss is intentional, not a cancellation failure. The collector's mistaken
  guaranteed-hit assertion was removed; cancellation was not repeated.
- Generic `read_file`/`stat_file` declarations used a **17,426-token** prompt.
  Model `read_file` arguments drove a real client-side file read. Continuation
  returned `GENERIC_TOOL_SUCCESS`, restoring paired **17,474** tokens; edited
  result returned `BRANCH_SUCCESS` at the same safe **17,474** frontier. Generic
  SSE and existing malformed-declaration/result-ID checks also passed.
- OpenCode **1.18.30**, canonical **`local/deepseek-v4.1-flash`**, local-only
  provider, normal **build** agent and all **10** normal tool declarations, with
  this project's serving document configured through ordinary `instructions`:
  initial prompt **12,051 tokens**; actual client `read` completed; continuation
  restored paired **12,116** tokens and returned `OPENCODE_LONG_TOOL_SUCCESS`.
  Build/title temperature was zero; no reduced agent prompt or tool schema,
  serving-layer tool special case, or Internet fallback. A capture-only localhost
  forwarder recorded unchanged wire requests. Text-part file attachments remain
  outside the unchanged string-message surface; they were not normalized by it.
- Every settled execution logged **replay=0, repack=0**, including cancellation;
  final health had no fatal error and zero queued/active requests. PagedCache
  eviction occurred, and counters counted the actual retained paired frontiers.
- Relevant ordinary/generic/cache/lifecycle/resource regression: **61 passed**;
  singleton/profile/transport/recovery: **107 passed**; selected existing P5/P6
  ownership/append/handoff: **38 passed, 26 subtests**. Existing capacity tests:
  **6 passed** in their standard-OFF pin (the MTP native recipe is intentionally
  a different admission identity). Total **212 passed, 26 subtests**. Final
  normal-local seal and ordinary `inspect` admission passed with the advertised
  **1,048,576 / 768** limits at that time (output subsequently promoted below). No full-repository qualification was run.

### Generic function acceptance (2026-10-09)

`tools/accept_generic_mtp_tools.py` ran against the actual admitted normal-local
`mtp-serving-v1` HTTP server, official checkpoint, P5/native MTP and Scheduler:

- Two declarations (`read_file`, `stat_file`) admitted; model returned ordinary
  `read_file({"path":"/tmp/example.txt"})` with a recipe-generated call ID.
- Matching client result completed with `GENERIC_TOOL_SUCCESS`; paired reuse
  restored **374 tokens**, the settled assistant-call frontier.
- Changed result completed with `BRANCH_SUCCESS`, safely restoring the same earlier
  **374-token** prefix, not the prior result/final-answer state. No rewind or fault.
- Ordinary SSE returned `read_file` deltas, `finish_reason: tool_calls` and `[DONE]`.
  Malformed declaration and foreign result ID returned HTTP 400, not weather errors.
- OpenCode **1.18.30**, reduced read-only agent, arbitrary local key `local-alias`,
  distinct display name and canonical `id`, sent its unmodified generic `read`
  declaration via a capture-only forwarding proxy. ds41f returned a tool call;
  OpenCode executed the file read and completed `OPENCODE_GENERIC_SUCCESS`.
  Continuation restored **853 tokens** (also reported in OpenCode's cache usage).
- Relevant admission, singleton/profile, recovery, paired cache, lifecycle,
  bind/resource and recipe boundary/transport regressions in the pinned normal-local
  environment: **161 passed**. No native/recipe dependency change.

### Direct-LAN transport fix checks (2026-10-09)

- Admitted normal-local startup with `mtp-serving-v1 --host 0.0.0.0 --port 8000`
  succeeded on the Mac Studio. `/v1/models` returned 200 through both
  `127.0.0.1:8000` and `192.168.68.56:8000`.
- A localhost Chat Completions request returned `OK`, fresh 0/9 cached tokens.
  Through the LAN-IP authority, its repeat returned `OK`, 7/9 paired tokens
  restored; appended history returned `YES`, 11/20 restored. These requests
  used ordinary JSON, without public sessions or sequence headers.
- LAN-IP SSE disconnected after content; the engine logged canonical settlement
  with `cancelled=True`. Subsequent JSON completed; its repeat restored 7/9.
  Final health had zero queued/active requests and no fatal error.
- `mtp-singleton-v1 --host 0.0.0.0` still failed before startup with the literal
  loopback requirement. Profile bind/worker/port, ordinary HTTP, paired cache,
  lifecycle, singleton boundary and MTP resource tests: **111 passed**.
  Additional M29/M34/M47/multimodal cancellation/Web private-LAN checks:
  **44 passed, 4 failed**. All four M47 identity failures also reproduce on the
  unmodified parent in this MTP environment (26 passed, same 4 failures);
  its native dependency is not the standard-OFF resource pin.

**Pending acceptance:** both IP clients above ran on the same Mac. No separate
LAN client execution facility was available; this is not evidence of packets
crossing the LAN or of host-firewall reachability. From another trusted machine,
GET `http://192.168.68.56:8000/v1/models`, POST the example Chat Completions body
twice, and confirm `OK` plus positive paired `cached_tokens` on the repeat.
The direct-LAN acceptance remains incomplete until those checks actually run.

### Existing ordinary execution acceptance

Actual normal-local startup on the admitted M3 Ultra/512 GB host, official asset,
and unchanged native dependencies exercised ordinary HTTP clients:

- Fresh `OK`: zero hits; repeat: 7/9 prompt tokens reused, same answer.
- Append `OK -> YES`: 11/20 reused, with no public conversation ID.
- Last-turn edit and branch: safe misses, producing `NO` and `MAYBE`.
- Wrapped rings: fresh and repeated 174-token prompt produced identical `WRAPPED`;
  repeat restored 172 tokens, beyond ring capacity 128.
- SSE disconnect after first content delta, and JSON socket disconnect during
  generation: canonical settlement logged with cancellation; subsequent requests
  restored 18/20 prompt tokens and completed normally. No cache/engine fault.
- Two independent concurrent JSON requests completed as `FIRST` and `SECOND`.
  Final health reported zero waiting/running requests and no fatal error;
  capacity eviction occurred while subsequent reuse remained coherent.
- Existing singleton HTTP fresh/append retained `OK`/`YES`, idle recoverable
  frontier 22, sequence 3, and normal DELETE retirement.

Affected current lifecycle, transport, recovery, resource, P5/P6/P7, profile and
new ordinary-path regressions pass. Existing frozen R1 tests have five obsolete
fixture failures; M40 has three historical receipt assertions. These same failures
were reproduced on unmodified master `71d21d1`; no receipts/tests were rewritten.
The current profile-origin tests pass in their pinned-origin process (older
lifecycle doubles otherwise contaminate import origins in a combined invocation).
No soak, benchmark campaign, qualification framework or new artifact schema was
created.

B1/B2 are implemented for this bounded ordinary serving path. This is **not**
closure of every general-runtime production-release requirement: broader API and
control domains beyond the supported ordinary function tools, transparent profile/resource policy and full production
observability in `production-serving-parity-closure.md` still need implementation.
Portable packaging/promotion, above-qualified context, Web, new hosts and
default-profile changes remain outside this task. M54R/M55R's existing bounded execution approval
is not replaced by a new campaign or extended to those domains.

## Ordinary output-capability promotion

The sole ordinary output authority is `serving/capacity.py:ORDINARY_OUTPUT_CEILING`
(**393,216**). Validation, recipe capacity preparation, production admission,
Scheduler request construction, `/v1/models` and `ds41f inspect` agree. The
singleton `mtp_profile.LIMITS` and internal qualification route remain 8192/768.
No decode, sampling, settlement or paired-cache algorithm changed.

Allocation ownership was inspected along OpenAI `max_tokens` -> recipe inference
options -> ProductionMTPBackend -> SamplingParams/Request -> Scheduler-owned
BatchGenerator -> native MTP. BatchGenerator stores a scalar/list termination
budget; its insertion creates state from actual prompt IDs/cache, not that budget.
Native MTP compares the generated count with the ceiling and clamps its existing
small speculative block to the remaining budget. Target state in
`model_execution/cache.py` and `language.py` grows from executed rows (bounded
window, appended compressed KV/index state). DSparkContextCache appends committed
rows to model-sized rings, independent of `max_tokens`. TokenBuffer grows in
256-token increments; recipe delivery/history accumulate actual events/tokens.
No max-output-sized preallocation exists or was introduced.

Finite real production HTTP results are in `artifacts/output-capability/`:

- `max_tokens=32768` and `393216`: accepted; both returned `OUTPUT_OK` in **3**
  completion tokens, `finish_reason=stop` (4 native canonical tokens including EOS).
- One **1024-token** completion crossed the old 768 boundary and normally ended
  with `finish_reason=length`. Canonical target and all three DSpark frontiers
  settled at **1065** and published. Follow-up reused **1065** tokens.
- One SSE disconnect after >800 visible tokens cancelled at **802** native
  generated tokens, settled/published at **843**, with no engine fault; a subsequent
  huge-ceiling short request completed normally. Existing fail-closed/burn tests
  remain passing; error handling was not relaxed.
- Generic model-generated `read_file` -> actual client file read -> continuation,
  changed-result branch and existing SSE behavior passed. JSON tool round trip
  used a **393216** allowance and terminated at the tool call/answer, not the budget.
- Every observed settlement retained same-frontier paired state, **0 prompt
  replay / 0 whole-cache repack**, and no new settlement proposal/verify cycle.
- Temporary external observers around existing enqueue/retire methods sampled
  MLX memory, without changing allocation. For the short 393216 request, active
  memory was **309,151,981,340 bytes** both before and after enqueue; cache memory
  was **2,503,408 bytes** both times. After EOS active memory remained unchanged,
  and peak stayed at the model-load peak **311,556,974,436 bytes**. State did not
  reserve 393K output positions. This is a diagnostic sample, not a memory suite.
- Actual HTTP total overflow returned `prompt + requested output exceeds
  1,048,576 total context tokens`; 393217 returned a distinct output-capability
  violation. Boundary tests accept 600000 + 393216 and reject 900000 + 393216.
- OpenCode **1.18.30**, local-only provider configured as **1048576/393216**, read
  `marker.txt` via its real read tool, continued with **7413** cached tokens and
  returned `OPENCODE_OUTPUT_PROMOTION_OK`. No external provider or 384K generation.
- The existing `accept_long_mtp_serving.py` regression also passed unchanged:
  **17110**-token fresh prompt, **17108** repeat reuse, **17114** append/branch
  reuse, disconnect recovery and long-context generic tool continuation.
  Receipt: `long-context-regression.json`; replay/repack remained zero.
- Relevant independent-process regressions: **158 passed + 14 subtests** across
  production serving/context, generic tools, ordinary capacity, singleton profile,
  lifecycle, bind, cache release and P5. The unrelated standard-off budget HTTP
  test was excluded for its normal-local recipe-native identity mismatch; this is
  not a full-repository qualification. Mixed-process runs expose existing
  environment/import contamination; affected profile suites pass independently.

The normal-local source inventory/local seal was refreshed using the unchanged
admitted recipe wheel. No new framework, endurance matrix, runaway policy,
packaging or release promotion was added.
