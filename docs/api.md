# Historical Standard-OFF API evidence

> Standard-OFF is retired. This release-surface-referenced account is retained
> as historical evidence, not a supported API or startup contract. Current
> Chat/Messages contracts belong to [ordinary MTP](mtp-production-serving.md).

At its recorded baseline, this page owned the **standard-OFF HTTP contract and client boundary**. Explicit
ordinary MTP shares recipe-facing Chat Completions but has different scheduling,
state and recovery semantics: see [ordinary serving](mtp-production-serving.md),
[admission](ordinary-api-boundary.md) and [production sessions](production-sessions.md).
The singleton has its separate [bounded contract](mtp-local-release-candidate.md).
[Capabilities](capabilities.md) owns profile comparisons, limits and defaults;
[operations](operations.md) owns installed startup and network configuration.
An implemented endpoint is not fresh release qualification of later HEAD.

## Protocol authority and readiness

Official DeepSeek recipe owns request conversion, prompt rendering/tokenization,
tool/reasoning grammar, response parsing and JSON/SSE formatting. ds41f supplies
execution, capability/resource/security admission and lifecycle safeguards.
Supported execution is narrower than everything upstream recipe can parse.
Configuration is resolved through `RuntimeConfig`; clients do not manipulate KV,
prompt-token replay, native parser state or oMLX objects.

`GET /health` distinguishes `alive` (process responding), `ready` (backend model
loaded) and `unavailable` (fatal runtime error). Process liveness alone is not
inference readiness; wait for `ready` with `model_ready=true`. Advertised generation
policy/defaults describe the selected runtime, not a cross-profile guarantee.
`GET /v1/models` reports the model and aliases: `deepseek-v4.1-flash`,
`deepseek-v41-flash` and `deepseek-flash`.

## Standard-OFF endpoints

| Endpoint | Contract |
| --- | --- |
| `POST /v1/chat/completions` | Independent Chat Completions, JSON or SSE |
| `POST /v1/responses` | Independent text Responses, retained smoke-qualified scope |
| `POST /v1/messages` | Independent text Messages, supported admitted recipe scope |
| `POST /v1/sessions` | Create a retained Chat Completions session; optional body `id` |
| `GET /v1/sessions/{id}` | Observe bounded metadata, active request and latest canonical turn; never execute |
| `DELETE /v1/sessions/{id}` | Close and release session/target ownership; active-work conflicts require reconciliation |
| `POST /v1/sessions/{id}/chat/completions` | Exact retained continuation, JSON or SSE |
| `POST /v1/sessions/{id}/budget` | Recipe tokenization/capacity preview and continuation validation without generation; optional expected-count check |
| `POST /v1/sessions/{id}/cancel` | Cancel/reconcile the identified stateful stream using body `request_id` |
| `POST /v1/sessions/{id}/persist` | Persist an idle artifact; optional body `artifact_root` |
| `POST /v1/sessions/restore` | Restore an idle same-backend artifact; required `artifact_path`, optional `id` |

The retained historical release scope and original endpoint account are preserved
in the earlier API edition (`c1d1f716:docs/archive/editorial-baseline/api.md.txt` in Git). Current budget,
stream cancellation and generation controls are not inferred from old release
receipts. Sessionized Responses/Messages are unsupported.

## Runtime contract behind the API

Standard-OFF selects DENSE_P0_P7/P7 dense prefill. Recipe tokens except the final
input are committed; P5 transfers the same live state and held-out terminal once
to first-party target generation. MTP, DSpark and speculative decode are OFF on
this path. See [architecture](architecture.md) and [generation](generation.md).
The HTTP/client layer owns neither native state nor tool effects.

## Stateful behavior

Retained Chat Completions require exact full canonical conversation and encoded
prefix extension. There is one executable cache authority per session, no hidden
history replay, no whole-cache repack/reconstruction, and all 40 layer frontiers
agree at an idle boundary. Independent requests instead prefill fresh; editing a
retained transcript does not authorize rewinding its live recurrent cache.

OFF's live-session limit is configurable through `DS41F_MAX_LIVE_SESSIONS`
(default four). Native execution is single-flight; active response/session
reservations and conflicts prevent competing mutation. Close retires its ownership.
Sampling is request-owned under `generation-policy-v1`; retained reasoning cannot
change silently. See historical frontend controls (`c1d1f716:docs/generation-policy-frontends.md` in Git).

Chat Completions reasoning defaults are configured at Runtime launch via
`--default-reasoning-effort {none,low,high,xhigh,max}` or
`DS41F_DEFAULT_REASONING_EFFORT`; the CLI option takes precedence. Headless launches
fall back to `none`; TUI launches use a saved preference, initially `max`.
Only omitted reasoning controls receive the default: explicit `reasoning_effort`
or `thinking` controls, including Off, remain authoritative. `/health` publishes
`chat_defaults.reasoning_effort`, `generation_policy.reasoning.default_reasoning_effort`
and the corresponding `default_thinking_mode`. Retained-session mode/effort still
cannot change mid-session. Responses/Messages conversion is unchanged; the
singleton profile only permits `none`. See [operations](operations.md) for setup.

Streaming returns `X-DS41F-Request-ID`. Optional expected-count/application-UUID
headers on OFF streaming provide observation/correlation; **they are not the
mandatory reserved-UUID retry fence of production MTP logical sessions**.
After ambiguous delivery, inspect the canonical `last_turn` and active identity;
do not infer rollback or permission to resend from a lost socket.

## Bounded multimodal development/core extension

OFF Chat Completions accepts recipe text/image content parts with inline image
data, for example `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...","detail":"high"}}`.
This is a separately qualified bounded core extension, not universal Vision,
multimodal Responses/Messages, MTP images or release/R1 promotion.

[Capabilities](capabilities.md#bounded-inline-images) owns format/count/byte/pixel/
aspect/context limits; [image qualification](multimodal-production-qualification.md)
owns measured evidence. Invalid or unsupported images reject before native cache
mutation. Expanded image positions count toward capacity and reserved output.

Continuation revalidates identical old encoded image bytes and absolute
span/grid/order before appending new content. Identity is not merely an image ID.
New images encode once; retained spans are never re-encoded. Idle persistence/
restore does not require resupplying old images; later conversation requests
still revalidate their CPU input identities.

## Persistence/restore

Only idle same-backend artifacts are supported: manifest, safetensors and commit
marker with checkpoint/schema/shape/dtype/frontier/provenance validation. Storage
is dormant, never a second executable authority. Mismatch/corruption fails closed
without fresh-history prefill. Cross-runtime/backend KV portability is unsupported.
See [session state](session-state.md) for live versus dormant ownership.

## Function tools

Recipe-authoritative client function tools and repeated tool/result continuations
are supported within the OFF contract. Completed model calls precede client tool
execution; matching ordered results extend canonical history. Invalid/foreign/
duplicate results reject before mutation. The runtime does not execute arbitrary
tools, plugins, MCP, shell or web search. Application-layer acquisition/effect
policies do not become native server features.

## Streaming and interruption

Stateless projection uses official recipe chunks in the supported protocol scope.
Current OFF stateful SSE is **demand-driven live delivery**, with one worker-owned
recipe cursor and a bounded pending protocol batch; it is not simply replay of a
whole completed turn. Backpressure stops further native demand. Protected worker
operations drain before cancellation settles or retires their ownership.

Canonical completion/cancellation publishes `last_turn` and request count before
releasing the response reservation. Unrepresentable interrupted/tool history can
retire executable authority and become unrecoverable. Yields, `[DONE]` and socket
writes are not canonical acknowledgements. A native fault must not invent a
successful partial turn or publish uncertain idle state.

## Termination and stateful `stop` policy

OFF target generation retains DeepSeek EOS token ID `1` in native history; recipe
hides it from text and projects `stop`. Length projects `length` under recipe
semantics. Arbitrary non-empty/non-null request stop strings reject on stateful
endpoints before mutation: detokenized truncation is not a generic recurrent-KV
rollback operation. Independent requests retain official recipe stop behavior.

## Error behavior

Admission/validation rejection does not mutate native session state. Unknown
sessions return not-found; overlap/max-session conflicts return conflict.
Execution faults are different: they may follow committed or uncertain native
work and require conservative settlement/retirement, not a promise that every
error rolls back all tokens. Transport failure alone proves neither nonadmission
nor canonical completion.

## Rust client boundary

`rust/ds41f_api` is an HTTP/JSON/SSE client and optional child-process control
boundary, not a Rust model runtime, FFI layer, KV owner or recipe parser.
`wait_process_alive()` proves liveness; `wait_model_ready()`/`wait_ready()` are
required for inference. `shutdown()` distinguishes graceful exit from forced kill.
Dropping an SSE iterator closes transport; server-owned canonical settlement is
not token-exact rollback or permission to replay.

Client persistence methods do not grant profile parity. No Rust MTP application
qualification follows from a shared endpoint name. The earlier API edition above
retains the original M21 evidence pointer; use current operations for startup.

## Diagnostics and unsupported scope

`DS41F_ENABLE_DIAGNOSTIC_ENDPOINTS=1` enables bounded operational metadata, not a
public native-cache API. Unsupported scope includes above-envelope images,
shared batching, distributed sessions, cross-backend KV, arbitrary stateful stop
rollback, server-side arbitrary effects and Internet/multi-user authentication.
Explicit ordinary MTP is supported in its own scope; “MTP unsupported” is not a
blanket statement about the whole repository. No default/profile/release promotion
follows from this API compilation.
