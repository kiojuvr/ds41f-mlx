# M35 — internal single-flight HTTP/SSE transport qualification

## Decision

**HTTP_SSE_QUALIFIED_BOUNDED_INTERNAL_SINGLETON**. Machine-readable authority:
`artifacts/m35/qualification.json`, with raw sockets/protocol/frontiers in
`http.json`. This qualifies the M34 guarded singleton through the actual local
recipe HTTP server for the bounded workloads below. It does **not** promote MTP
to the release API, public selection, or a default. `docs/api.md` is unchanged.

Source base: `2d6e313` (clean master at M34). Final commit resolves with
`git log -1 --format=%H -- artifacts/m35/qualification.json`. Exact new sources,
harnesses, gates and evidence hashes are in the artifact. M33/M34 runtime sources,
upstream candidates and historical evidence are unchanged.

## Identities and admission

- oMLX: `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`, base
  `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`, `/tmp/ds41f-m33-omlx`.
- Recipe: `29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`, base
  `8cadfede7063c896b944e7bae05daa3549ae97ea`, `/tmp/ds41f-m32-recipe`.
- Full M32 ARM64/OpenCV recipe native SHA256:
  `454413afdcee2916795e1c5f7ce1b94a8346e76bffd6b45024f28f69f8d0a73f`.
- Official checkpoint remains
  `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.
  Config/tokenizer, actual imports, Python/packages and preserved OFF release
  artifacts are freshly checked in `identities.json`. Stock MLX/source-JIT model
  kernels, not borrowed release C++ kernels; portable wheels are not qualified.

The experiment explicitly constructs `InternalMTPQualificationBackend` and
injects it into the existing `serving.server.create_app`. No environment,
request-body, normal launcher or release selector enables it. Actual uvicorn
serves an ephemeral **127.0.0.1** TCP port; exact port/version/config are recorded.
The seam allows one logical live session, one response lease, text stateful Chat
Completions only, maximum 768 responses and 8192 prompt-plus-budget tokens.
Overlap rejects rather than queues. Stateless inference, persistence/restore,
oversized requests and extra live sessions are denied. Arbitrary stateful user
stops remain rejected by the existing pre-conversion policy.

Run the explicitly internal experiment with:

```sh
DS41F_OMLX_PATH=/tmp/ds41f-m33-omlx /tmp/ds41f-m32-qual/bin/python tools/run_m35_http_qualification.py
```

This command is qualification tooling, **not** a release launch instruction.

## Streaming architecture and ownership

The ordinary M12 stateful route still buffers its complete committed assistant
turn before sending recipe SSE. That route is unchanged for normal backends.
Its transport can test post-commit loss, but cannot exercise active MTP ownership
on disconnect. The injected backend therefore supplies a narrower live response
bridge through the same session routes and official recipe request/chunk APIs.

1. Native synchronous P6/P5/MTP phases run on the existing single-worker executor.
   P6 layer-input taps prime DSpark from the same target forwards, exactly as in
   M34. The M33 guard alone advances the official native recipe parser for emitted
   or safely drained canonical tokens. Preview never advances it.
2. Active canonical progress belongs to the M34 generation owner's history.
   The record's committed history is its previous idle boundary until settlement;
   GET reports **busy** during that interval. It is not a second active frontier.
3. Only then are newly generated recipe events serialized and yielded as SSE.
   Events are associated with their canonical input ordinal **upper-bound
   watermark**. Protocol output is not a one-token/one-chunk representation:
   role, usage, DSML buffering, decoder-pending IDs and backend controls matter.
   No transport-specific token truth or grammar is introduced.
4. A semantic terminal is settled before its event frames are exposed. Cancellation
   after terminal emission but before this step follows the same shielded cleanup.
   The final canonical token may receive the existing one-token target/DSpark
   repair, never successor sampling or verification.
5. Server event production and iterator yields are recorded separately from real
   client-received recipe events. Every received sequence is checked against the
   canonical native event prefix. A yielded frame or socket write is **not** an
   application acknowledgement. There is no application ACK protocol here; M34
   delivered-token metadata deliberately stays at response ordinal zero. Client
   observations carry input-watermark bounds, not fabricated exact token ACKs.
6. Disconnect keeps canonical authority, including undelivered text/events and
   any bounded safe native drain. It discards only un-emitted future ownership.
   Native quiescence transfers executable cache wrappers through the retained
   M34 `extract_cache(0)` correction. After owner close, H=T=R across all forty
   target and three DSpark offsets; prediction/queue ownership is retired.
7. Internal GET exposes the settled canonical recipe response for recovery. The
   controller re-enters using ordinary full-message recipe conversion and exact
   prefix admission, not a raw-ID splice, prompt replay or cache reconstruction.
   Each next turn uses a fresh recipe processor on the retained native caches.

All native mutation and cleanup are shielded; unshielded AnyIO checkpoints admit
cancellation **between** coherent synchronous boundaries. This is bounded
response-boundary cancellation, not immediate/token-exact abort. A protected
native or protocol failure poisons the record, releases the lease and requires
DELETE; it cannot advertise an ambiguous idle/resumable state. The response also
owns admission before its iterator starts, so header-send failure cannot leak a
busy reservation. DELETE while busy returns 409; DELETE after cleanup holds the
same singleton lease while retiring cache/ring/prompt-priming references. A
continuation cannot race retirement; failed retirement leaves the singleton
poisoned rather than permitting reuse. Static weights remain resident for the next
fresh logical session.

## Real transport matrix

Final evidence: **18 HTTP turns / 1359 returned model responses**, plus one
490-response same-request synchronous direct control; eleven fresh target-cache
allocations (ten HTTP logical sessions and the direct control). Maximum observed
HTTP canonical frontier is 580, not an 8K or longer-context operating claim.

- Ordinary non-streaming request and exact-prefix next user turn.
- Forced `lookup_weather(Paris)` SSE call, deterministic **client-side** tool
  execution, ordinary tool-result SSE re-entry.
- Forced Berlin call disconnected after canonical DSML terminal/quiescence but
  before all transport output; GET recovers the complete call and ordinary
  tool-result re-entry succeeds.
- A fresh detailed eight-section guide completes normally: 490 responses, 491
  SSE events including DONE. A fresh identical guide is consumed with a 600 ms
  pause every seven events. Canonical progress leads the received event's input
  watermark by as much as **259** while the client catches up.
- Ordinary long-stream socket shutdown after forty received events. Corrected
  bridge stops at **41** returned responses, settles frontier **89**, and
  ordinary next-turn admission/continuation succeeds at frontier **126**.
- Controlled 800 ms server delivery delay: shutdown after one canonical response
  but **zero yielded/received events**. Canonical frontier **49** survives; next
  ordinary turn reaches **89**. No delivered frontier is invented.
- A short EOS completion is disconnected at its settled terminal delivery window;
  retained frontier **13** continues to **35**. The separate Berlin case above
  proves actual **recipe semantic-terminal** ownership, not only EOS handling.
- Existing Rust `Ds41fClient.session_chat_completions_stream` receives forty
  events, then iterator Drop closes its real socket. Runtime settles after 41
  responses at frontier **89**; Python-authoritative ordinary continuation reaches
  **197** (96-response length boundary). No cache/model authority moves to Rust.
- GET/DELETE and fresh-session reuse follow completion and interruption. Thirteen
  admission/rejection results include overlap, second session, busy DELETE,
  persistence/restore, stateful stops, bad model/budget, excessive context,
  malformed request shape, unsupported stateless inference and idle non-prefix
  history. Rejections preserve retained history; idle non-prefix rejection also
  compares the complete session snapshot before/after.

All received events match the canonical recipe prefix; incomplete delivery never
invalidates canonical progress. All exercised recoveries use exact-prefix
re-encoding. No HTTP cancellation case naturally required a safe queue drain;
the undelivered canonical suffix still exists and is recovered. Fresh unchanged
M33/M34 native interruption gates separately exercise safe drains, predicted but
un-emitted terminals, decoder-pending/alignment fixtures and protected failures.
Those controlled fixtures are not relabelled natural HTTP sampler outcomes.

Slow-client output in this bounded matrix fits socket buffers: it proves real
buffered delivery lag, **not sustained socket-buffer saturation**. Controlled
server delay is labelled separately from client stalls. There is no claim of
arbitrary partially decoded Unicode/stop-string client re-entry; broader pending
byte recovery and saturating backpressure need another gate.

## Defects and corrections

1. **Real disconnect starvation:** exploratory ordinary and Rust drops after
   forty events still generated all 490 responses. Shielded worker calls admitted
   no cancellation checkpoint between boundaries; uvicorn sends need not suspend.
   Fix: explicit unshielded checkpoints after priming and each canonical response,
   retaining shields around mutation/cleanup. Real reruns now stop at 41, preserve
   coherent caches and continue ordinarily. The regression cancels an AnyIO scope
   inside a shielded response and proves no subsequent chunk/generation begins.
2. **Native protocol transfer / double settlement:** evidence serialization after
   `response.append` accessed consumed native recipe chunks. Exception cleanup
   then retried an already-drained runtime, masking the original error with an
   unknown-queue-topology failure. Fix the ownership order: serialize before native
   chunk transfer; retire runtime ownership exactly once before protocol
   accumulation. Protocol faults fail closed rather than retry quiescence.
   Real-native chunk-transfer and post-quiescence fault regressions cover this.
3. **Pre-iteration admission:** a synthetic header-send failure established that
   closing an unstarted async generator does not execute its finally block.
   Response-level cleanup covers that reservation using exact per-response lease
   retirement, not the mutable session busy bit: a stale response cannot release a
   subsequent request's lease. Failed retirement also releases admission while
   keeping the logical singleton poisoned and unavailable for reuse. These have
   synthetic lifecycle regressions, not claimed sampled checkpoint disconnects.

Failed and exploratory evidence remains classified in the artifact. A pre-model
client-helper import shadowing error is a harness mistake, not a model defect.
No M33 horizon or M34 native cache-transfer code was changed; no copying containers
to evade capabilities, flag clearing, history replay or hidden regeneration was
introduced.

## Performance decomposition

The same resident model, fresh prompt, greedy sampler and exact **490 canonical
IDs** run both through normal real SSE and a synchronous guarded direct loop:

- HTTP model/canonical phase **24.44 tok/s**, direct **24.43 tok/s**, ratio
  approximately **1.0004**; no material model-phase regression.
- Normal SSE admission-to-first canonical **377 ms**, including **315 ms**
  suffix/prompt prefill and handoff; first iterator-visible frame follows in under
  a millisecond. Actual first client event is **378 ms**, recorded separately.
- Model phase **20.050 s**, event serialization **1.9 ms**, cleanup **53.8 ms**;
  normal request-to-client-completion **20.495 s**. Residual executor/event-loop
  dispatch and protocol-loop time is separately derived in `qualification.json`,
  not mislabelled GPU time. The first request's approximately 68-second model load
  and initial compilation/prefill are excluded from steady decode comparisons.
- Stalled consumption: model phase still **24.51 tok/s**, client completion
  **42.944 s**. This is delivery lag, not a model decode slowdown.
- Ordinary/Rust disconnect-to-observed-idle **62/49 ms**; runtime cleanup about
  **49 ms**, including the permitted canonical-token repair. Zero-event delayed
  disconnect cleanup **1.08 ms**, observed idle about **2 ms**.
- Ordinary disconnect next-turn prefill/handoff **223 ms**, first canonical from
  admission **284 ms**. Rust next-turn equivalents **187/250 ms**. Compilation
  shapes can cost seconds (e.g. the slow-guide continuation); they are retained,
  not smoothed away as network delay.

Acceptance uses the unchanged upstream **considered-draft** denominator. Normal
and slow guides both measure **63.87%**; completed-owner stats and per-case scope
are explicit. Active-cancellation owner stats are not reported as a complete
acceptance sample. M34's 42–44 tok/s controls had roughly 94–95% acceptance and
longer/repetitive prompts; its 33.73 weighted soak rate is not a network baseline.
M35's matched direct comparison isolates server cost instead of attributing
workload predictability differences to transport. M34 acceleration evidence and
its unchanged runtime identities remain intact; M35 does not claim a fresh OFF
HTTP speedup measurement or a universal acceleration guarantee.

Replay/reconciliation and full packing/admission are instrumented forbidden;
observed counts remain **zero**. Target-cache construction is permitted only for
fresh allocation and forbidden during handoff/continuation. Quiescence counters
prove no new verification/proposals, replay or repack. All corrected HTTP/SSE and
tool-result re-entries consume the retained cache wrappers after owner release.

## Gates and next milestone

Fresh gates: **98 Python tests + 169 subtests**, ten transport lease/native
transfer regressions, **29 real-native horizon fixtures**, all **16** checkpoint
interruption/fault cases, eleven OFF policy/config tests, five preserved release
native/cache ABI tests (one existing Pydantic warning), and `cargo test`.
Identity checks pass. Artifact/frontier/prefix/defect/performance gates additionally
validate the canonical M35 result. Pre-existing repository-wide authority-label
findings from M34 are not silently promoted to passing audit claims.

Still unsupported: public/default/release MTP, persistence/restore or restart,
token-exact abort, concurrent/shared execution, batching, 200K/long-context HTTP
MTP, distributed serving and portable native wheels. Bound checks and singleton
admission remain internal and do not authorize release promotion.

Recommended **M36: client/agent canonical recovery and admission qualification**:
exercise automated recovery through client boundaries, pending-byte/Unicode
partial responses, and larger workloads with actual sustained socket backpressure.
The M35 controller's internal GET recovery is not yet a documented public client
contract. Longer-context HTTP operation and a separate release/admission gate
remain unproven. Do not choose public/default promotion merely because M35 passes.
