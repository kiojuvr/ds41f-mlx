# M40 — bounded MTP release/admission/provenance readiness

## Decision and authority

**NOT_READY_PUBLIC_ADMISSION_AND_REPRODUCIBLE_DELIVERY**.
The M39 architecture is sufficient for a bounded local capability; the repository
has not yet implemented or qualified its external admission and distribution
contract. This is neither MTP release qualification nor an architectural recovery
blocker. Production/default/release MTP remains **OFF**. No runtime changes are
made in M40.

Evaluation base: clean `master`, `ebccd8877d22e00ecbd765b67a665de2c04571de`.
Canonical machine record: `artifacts/m40/readiness.json`; observed identities,
source/evidence hashes and audit logs are adjacent. Final commit resolver:
`git log -1 --format=%H -- artifacts/m40/readiness.json`.
This document specifies a **proposed**, not currently available, release profile.
M40 performs source/evidence inspection and cheap consistency/regression checks,
not another checkpoint campaign, clean-install qualification or performance run.

Primary inspected authorities: `release/ds41f-release.json`, `config.py`, `ops.py`,
`serve.py`, `provenance.py`, `qualify.py`, `release_acceptance.py`, `build_release.py`,
`pyproject.toml`, `docs/api.md`, `operations.md`, `m22-release-packaging.md`,
`m23-relocatable-release.md`, `serving/server.py`, `internal_mtp.py`,
`request_policy.py`, `request_fence.py`, `recovery_certificate.py`,
`internal_local_client.py`, `web_client.py`, `web.py`, `web_tools.py`, Rust
`src/lib.rs` and transport tests. M33–M39 canonical decisions are individually
recorded in the machine record, including M36's negative result and M38's
retention blocker (resolved, not rewritten, by M39).

## 1. Capability profiles: separate process authority

Keep **standard-off** as the existing default, unchanged: DENSE_P0_P7 prefill,
GenerationBatch decode, MTP/DSpark/speculation OFF, stateless text protocols,
stateful Chat Completions, configured live sessions (default 4), requested IDs,
idle same-backend persistence/restore and the existing Rust/operations bundle.

Propose explicit **mtp-singleton-v1**: one process, one backend, one live session,
one active response lease, guarded DSpark/MTP depth 5, DENSE_P0_P7/P5 terminal
handoff, exact-prefix continuation, certified ordinary-protocol outcomes and
server-issued process-lifetime IDs. No parallel OFF backend in that process.
Stock source-JIT model kernels are part of this profile identity. No shared model
configuration, implicit fallback or request-selected acceleration. Mode changes
require orderly shutdown and restart into the other dependency/profile set;
no dynamic switching or backend recreation within a session lifetime.

| Capability | Proposed MTP classification / contract |
|---|---|
| `/health`, `/v1/models` | Supported narrower: retain alive/ready distinction; add profile, limits, dependency identity and qualification status (no private paths in public health). Only fixed qualified aliases advertised. |
| Stateless Chat Completions | Explicitly unavailable, whether or not a session is owned. |
| Stateful Chat Completions | Supported narrower: singleton, text-only, exact-prefix, bounded envelope below, required request fence. |
| Responses / Messages | Explicitly unavailable; sessionized variants remain unsupported in both profiles. |
| Streaming | Supported narrower: canonical recipe SSE, display-only; terminal settlement precedes terminal publication, lease lasts through response cleanup; cancellation is coherent-boundary cleanup, not token-exact abort. |
| Client-side function tools | Supported narrower candidate subset: pinned weather declaration below and certified completed calls only; general schemas are unqualified, not inherited from OFF. |
| Session create / GET / DELETE | Supported narrower: create empty `{}` only, server-issued ID, one live record; GET sanitized status/latest outcome, DELETE shielded retirement. No enumeration/discovery/replacement authority. |
| Requested session IDs | Explicitly unavailable, including null/empty/retired requested fields; reject the field, not just its non-null value. |
| Persistence / restore | Explicitly unavailable before path parsing or I/O; no request-reachable filesystem parameters. |
| Diagnostic endpoint | Optional narrower sanitized bounded metadata, default disabled; no raw canonical IDs, witness/token stream, caches, parser state, internal trace payload or filesystem paths. Existing full diagnostic output is not promoted. |
| Rust raw transport | Health/lifecycle can remain raw; current crate cannot attach sequence headers. Required raw fenced transport addition and installed boundary test if advertised as an MTP request client; no Rust parser/recovery authority. |
| Python local client | Stabilize a small supported facade over InternalLocalClient states, frozen identity, reconcile, stream close and effect ledger; preserve its conservative semantics, do not promote internal diagnostics. |
| Browser / existing StatefulToolChatClient | Explicitly unavailable in initial MTP profile. Existing browser persistence/tool workflow must not connect silently; profile handshake must refuse. Dedicated browser integration is optional parity. |
| M36R/M37 recovery | Supported narrower process/living-client contract; positive certificate AND real continuation exact-prefix required. Negative/manual terminal outcomes are part of support, not failures to auto-repair. |

Unsupported known routes return JSON `unsupported_capability` (400) **before body
conversion/model/session mutation**; unknown routes remain 404. MTP route mounting
must not inherit stateless/persistence handlers just because a backend overrides
methods. Static OpenAPI/docs routes must be disabled or intentionally accounted
for; no incidental endpoint expands the supported contract.

## 2. Canonical proposed external envelope

This is a deliberately conservative v1 allowlist. It is release-candidate input,
not a statement that current deserialization already enforces it. A stricter
subset is meaningful for short local text/tool agents; arbitrary tools/reasoning
are not necessary to support that capability. General tool-schema support would
require a separately expanded admission/acceptance gate, not a silent claim.

| Field / semantics | v1 rule | Evidence / remaining gate |
|---|---|---|
| Body | UTF-8 JSON object; duplicate keys, invalid JSON/nonfinite numbers, wrong types, unknown fields rejected; JSON content type required | Current route lacks strict allowlist and duplicate-key policy. |
| `model` | Required, one of `deepseek-v4.1-flash`, `deepseek-v41-flash`, `deepseek-flash`; normalize only model selection, never frozen body bytes | Alias backend checks exist; test all on packaged profile. No configurable unqualified model name. |
| `messages` | Required nonempty ordinary Chat conversation; system/user string content, certified assistant content/tool calls, tool string results with exact IDs; no image/audio/content parts, raw token IDs, raw special-token transcripts, arbitrary assistant injection or extra role fields | Official conversion plus exact-prefix remains authority; initial application transcript must satisfy ordinary recipe conversion. Unsupported roles/forms reject. |
| `temperature` | Omitted or numeric 0 (not bool); v1 deterministic sampler only | M37–M39 HTTP workflows use 0. M33 nonzero matched decode evidence does not qualify all HTTP sampling knobs. |
| `top_p`, `top_k`, penalties, seed, logprobs, `n`, logit bias | Unavailable, reject even if upstream accepts them | Current sampler accepts top_p and does not establish full public envelope. |
| `reasoning_effort` | Omitted (explicitly converted as `none`) or `none`; no other thinking/reasoning knobs | Exact-form reasoning-transition fixtures are not general admission qualification. |
| `tools` | Omitted, or exactly the `lookup_weather` declaration in `tools/run_m11_tool_boundary_qualification.py:tool_def`: function, strict true, description `Return deterministic weather for a city.`, object parameters, single string `city`, required city, additionalProperties false | M37 real auto/required calls and result loops; freezing this declaration avoids claiming arbitrary recipe-parsable schemas. Schema bytes/semantic identity must be published in RC profile. |
| `tool_choice` | Absent with no tools; with that declaration, `auto` or named function choice `lookup_weather` | Reject built-in/server tools, forced unknown names, parallel policy options and arbitrary custom tools. |
| Tool results | One ordinary real stored result per completed certified call, same order/IDs, no missing/duplicate/wrong-ID result; result string <=64 KiB each, living-client ledger <=128 effects without eviction | Converter/prefix/ledger checks; actual completed one/two-call fixtures inherited; repeat on installed helper. Tool execution remains application-side. |
| `stream` | Omitted/false or JSON boolean true only | SSE is not durable delivery acknowledgement. |
| `stream_options` | Omitted or `{include_usage: boolean}` only, and only with stream true | Recipe usage machinery exists; fresh positive/negative RC tests required. Can remove this optional field if its gate fails. |
| `max_tokens` | Required integer 1..768, not bool; no alternate token-budget field | Make the request budget explicit rather than relying on the backend's implicit 128-token default. Test min/max and overshoot before reservation. |
| `stop` | Omitted only in v1 (null/empty also rejected for simplicity) | Stateful arbitrary stop already prohibited; M33 stop fixtures are not a public stop-string qualification. |
| Context | Official encoded prompt + requested max_tokens <=8192 on every fresh/retained request; no truncation or rollover; prompt strictly extends retained canonical prefix | Existing internal bound; packaged boundary tests needed. |
| Other fields | Reject, including response_format/JSON mode, user metadata, persistence paths, MTP/DSpark switches | Upstream deserialization is not qualification. |
| Sequence header | Required canonical decimal integer 1..2^64-1, starting at 1; reject duplicate/malformed header; identity = (server ID, sequence, SHA256 exact body bytes) | Internal path still allows unfenced legacy work; released profile must not. |
| Create | Empty body or `{}` only, no ID/other fields; same bounded ingress machinery | Existing body `.get` can accept malformed/ignored content or raise an incidental error. |

Preserve raw bytes for fencing; any default expansion is only a conversion option,
never a rewritten retry body. For a retained lifetime keep tools/reasoning envelope
unchanged. Changed schema/settings, edited history and reordered results cannot
bypass the independently checked prefix. An empty/pending/unrepresentable assistant
may settle negatively even for an admitted request; no universal success promise.

### Pre-mutation admission order and resource policy

RC must enforce, in order: profile/route/Host/Origin/content-type; bounded transport
read; strict JSON/types/allowlist; live identity and sequence syntax; official
conversion/tokenization under a bounded preparation slot; body/token/output budget;
retry observation; exact prefix and certificate/state; singleton lease/fence
reservation; model/native work. Validation may tokenize, but may not consume
sequence, issue IDs, allocate model/cache authority, publish chunks or mutate parser.
Latest settled same-body retry returns original JSON outcome without generation;
active retry/overlap returns 409. Changed-body, gap, expired request and poison or
unrecoverable continuation return explicit 409 codes; malformed/envelope errors
400, retired/foreign/never-valid session 404, resource/preparation conflict 409.
Ingress byte excess 413. No waiting queue of generation requests.

Current server calls `await request.body()` **before** the internal 1 MiB check.
Create/restore/stateless routes have no corresponding byte check, conversion runs
before singleton request reservation, malformed create can fail through `.get`,
and arbitrary recipe fields are not rejected. M39 proves model admission atomicity,
not bounded transport ingress. Required RC policy: <=1 MiB cumulative received
bytes on every body-bearing MTP route (including chunked bodies; reject excessive
Content-Length early), reject Content-Encoding, one bounded body/preparation slot
per process, bounded connection count (initial candidate 8), no work queue,
finite incomplete-body timeout (candidate 30 s). These new numeric ingress limits
need actual h11 socket tests; they are not inherited qualification.

Pull-based SSE and response cleanup are qualified, but indefinite stalled peers
are not. Add a finite send-stall policy (candidate 30 s), bounded uvicorn transport
buffers and socket-pressure gate. Stall/disconnect causes existing coherent cleanup
and fenced reconciliation, not early release of native ownership. Timeout values
are release-policy candidates, not model latency SLAs. Shielded native calls can
outlast transport deadlines; document wait/poison/process termination escalation.

## 3. Singleton / lifetime contract

A live empty session owns the singleton too. A second create or an overlapping
active request returns 409 without issuance/reservation/model mutation. DELETE
while busy returns 409; after settlement it holds the lease through synchronized
retirement. Failed retirement retains a poisoned singleton and blocks fresh create.
No stateless work is available at any time in the MTP process. No cache-sharing,
batching, concurrent clients, backend recreation or fallback to OFF.

Server ID is `mtp_<128-bit namespace>_<128-bit serial>` (canonical lowercase);
contiguous issuance starts at 1, never wraps, rejects at exhaustion. Live map is
sole authority; issued absent IDs are permanently retired in this process;
foreign namespace is stale/foreign, not proof of prior issuance. Diagnostics never
restore validity. One live record, one latest fenced outcome/reconstruction body,
16 retirement summaries, fixed 32-slot trace deques internally, <=64-bit counters.
Old outcome expires on the next admitted sequence, not on a timeout; latest settled
observation remains possible at sequence ceiling. No inexhaustible identity claim.
Restart discards authority; old sessions/outcomes are not restored.

## 4. Required local security boundary

MTP profile must **fail startup** for non-loopback bind or ambiguous hostname.
Initially accept literal `127.0.0.1` only (IPv6 ::1 may be added with an explicit
socket gate); do not trust DNS resolution of arbitrary host names. Enforce both
CLI and configuration paths, single uvicorn worker, no inherited external sockets,
no trusted proxy headers or advertised remote/proxy deployment. OFF configuration
remains unchanged. Current config defaults loopback but does not validate override.
M40 does not add an exposed selector, so this requirement is enforced at promotion,
not misreported as present today.

No authentication exists. Loopback is not protection against other local processes
or malicious web origins. Treat all local processes as trusted operator peers; IDs
are visible in lifecycle/status responses and **not secrets/access capabilities**.
Require Host matching configured loopback authority; reject nonapproved Origin on
state-changing routes, require application/json, no permissive CORS. With browser
support deferred, reject browser Origin entirely for MTP initially. Test DNS-rebind
Host, cross-origin/simple POST and malformed ingress denial. A future remote or
multi-user service needs authentication/access-control design; not generic M40
hardening or a deployment supported by a bind override.

GET/latest outcome contains sensitive conversation/tool data; diagnostics default
OFF, sanitized, bounded and loopback only. Local logs/evidence/bundle records may
contain paths/transcripts: operator file permissions and explicit retention apply;
no network diagnostic export. Persist/restore routes unavailable, so no supported
request accepts filesystem paths. Checkpoint/recipe/oMLX paths are trusted startup
configuration, not model-selected inputs. Server performs no tools, URL fetch or
SSRF-capable network action in the profile. Optional web client fetch tools resolve
public addresses and check redirects, but validate-then-connect has DNS-rebinding
risk; search tools make outbound provider calls and expose credentials/network
policy. Excluding that client avoids inheriting an unrelated network security
claim. Do not advertise it as MTP safe because weather stubs were qualified.

Byte, context, output, live-session, prep and connection bounds plus bounded SSE
are release requirements. General internet flood defense, TLS, distributed quotas,
sandboxed autonomous tools and remote RBAC are deferred, not release blockers for
this trusted single-operator local profile.

## 5. Application workflow (support includes terminal states)

The RC must version a public certificate/outcome projection: sequence/body digest,
state, version, frontier/hash evidence, positive predicates and bound ordinary
assistant response. Keep native IDs and internal witness/trace payloads private.
The helper currently validates internal certificate witness/message bindings;
changing to this public projection requires fresh helper validation/fault tests,
not a claim that raw internal GET is already a stable release surface.

Stabilize a small Python local helper/facade, rather than requiring every operator
to reimplement M36R races. One thread, one living client, one owned stream, one
frozen body/identity and one copied latest certified outcome. Retain raw HTTP
contract for deliberate application implementations; do not expose native token,
cache, witness/parser authority. Rust stays raw, with header support if promoted.

| Outcome | Required application action |
|---|---|
| Ordinary JSON or SSE completion | Reconcile settled fenced outcome; replace assistant from positive certificate, never treat `[DONE]`/HTTP success as recovery authority. Advance only then. |
| Transport ambiguity | Close owned stream, discard partial UTF-8/SSE, GET until settled; re-observe frozen same sequence/body. Not-admitted absence is not proof that delayed work cannot arrive. No new request identity. |
| Recoverable | Copy settled result before new sequence; ordinary continuation independently proves prefix. Completed tool execution also requires certified DSML terminal + tool_calls finish. |
| Unrecoverable protocol state | No more generation/effects; explicit DELETE then confirmed fresh create. Fresh prefill uses only legitimate ordinary application history, omits unfinished native fragments, never invents results. |
| Poisoned | No continuation; explicit DELETE; retirement failure blocks create. Operator may stop process if safe retirement cannot be confirmed. |
| DELETE/create transport outcome unknown | Sticky stopped/manual-reconciliation; no guessed ID, session discovery, retry create or destructive automatic replacement. Operator can deliberately stop the owned process and start a new lifetime after application effect review; this is not session restore. |
| External tool effect unknown | Sticky tool_ambiguous; reservation precedes execution. Never auto-repeat. Application decides using external effect evidence; process restart does not resolve effect uncertainty. |
| Expired identity | Reject replay; use an already copied current certified conversation or explicit fresh-session decision. No older outcome archive. |

Effect ledger keyed by session/sequence/index/call ID and bound call contents;
<=128 non-evicted reservations, <=64 KiB result each. At capacity stop before effect;
manual application boundary/fresh living-client lifetime only after reconciliation.
No crash-safe exactly-once, automatic partial DSML repair or distributed effects.
These limitations do not require another ownership/recovery micro-milestone.

## 6. Provenance delta: OFF manifest cannot identify MTP

| Component | OFF release | M39-required MTP identity |
|---|---|---|
| ds41f | Version 0.23.0 manifest; current source must be independently hashed | M39 commit above + current runtime tree/source hashes; internal backend, guard, lifecycle, fence, certificates and client are part of owning identity. |
| oMLX | 0.7.0 base `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40` | Candidate `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc` on same base; `artifacts/m33/omlx-semantic-horizon.patch`, SHA256 `fb120b1ddc9120e69d71ce70f128db7e3bf19a7444b0f237d7661388c9b318d5`; four file hashes in M33 identity. |
| Recipe | 0.1.1 base `8cadfede7063c896b944e7bae05daa3549ae97ea` | Candidate `29dabb5a55b7b2c6a68e18bbb3eb14495623e81a` on same base; preview/native diagnostic binding; patch SHA256 `dea4826afcc7561bbf5c719c0f7e40ec11037b3e7d4fc95f07fc117b02c266b9` and lock `0897f68c7220e3fa11b11b03fe45245ab88bd2515586af057d00bf64111c9328`. |
| Python / MLX / mlx-lm | 3.13.15 / 0.32.2 / 0.31.4.dev132+g94cdcae13 | Unchanged; mlx-lm revision `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`. |
| Native recipe | Preserved release binary SHA256 `a41596ca12503efa1b0a25b16d9d52293378528b3aec4f060c3814f488fdcd43` | Full ARM64 candidate `_native.abi3.so` SHA256 `454413afdcee2916795e1c5f7ce1b94a8346e76bffd6b45024f28f69f8d0a73f`; M32 local wheel SHA256 `7adbd29a366ee59ba03d4df4b181d75a29071cdcf8a482ec37235597c100b496`. |
| Native build/link | Release environment record | Rust/Python tokenizers 0.23.2, OpenCV 4.14.0 (`opencv@4` host dylibs), Rust 1.98.1, macOS 26.5.2 arm64; not DOCS_RS/protocol-only. M32 identities record dylib paths and digests. |
| Model kernels | Production GLM compiled hashes recorded by provenance | Candidate optional C++ DSA absent; pinned stock MLX/source-JIT kernels, no borrowing release DSA binary. Record generated kernels/cache policy separately; do not equate clean Git to compiled identity. |
| Checkpoint | External official index/config/tokenizer fingerprints | Same official checkpoint; index `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`, config `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`, checkpoint tokenizer `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`. Recipe V41 tokenizer separately `81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99`. |
| Hardware / selector | Mac Studio M3 Ultra 512GB class; DENSE_P0_P7, OFF | Same hardware class, P7 overlap/full resident backbone + SSD Engram, P5 terminal handoff; guarded MTP/DSpark ON depth 5, singleton v1/envelope/certificate version identity. No broader platform claim. |

M39 identities gate checks clean candidate commits, native actual import and
unchanged packages; M40 observed snapshots independently check those paths now.
Qualification paths `/tmp/ds41f-m33-omlx`, `/tmp/ds41f-m32-recipe` and
`/private/tmp/ds41f-m32-qual` are evidence locations, **not delivery contracts**.

Existing base+local-content machinery is conceptually reusable but not sufficient
unchanged: it pins one OFF set, approves only historical unreachable oMLX patches,
warns on revision drift, and hashes oMLX GLM binaries but not the imported recipe
extension and transitive dylibs. Package version 0.1.1/0.7.0 alone conflates both
sets. Installed recipe preference can import a different native wheel than the
configured tokenizer checkout. Runtime tree digest excludes web_client even
though MTP helper depends on it; bundle record's source runtime projection misses
`native/` copied source (builder excludes that tree). Separate profile identity,
actual module origins, complete helper/transport hashes, build/link identity and
manifest/envelope digests must be bound to acceptance. Unknown executable drift
must fail MTP startup/acceptance, not remain a successful inspect WARNING.

Selected delivery policy: **vendor/export exact candidate source content plus
patch/base identities in the local distribution**, with locked dependency/build
inputs. Upstreaming is welcome but not a prerequisite; a new upstream commit
requires identity reconciliation. Do not rely on candidate commits existing on a
public remote (not established here). Approved local-content identity is acceptable
only when the content and installation recipe are actually delivered and verified.
A rebuilt native binary is a new artifact identity with fresh native parity/import
and package acceptance; it need not reproduce old binary bytes bit-for-bit.
Checkpoint fingerprints are cheap metadata identity, not a full weight-shard hash:
RC must record shard inventory/size and checkpoint source revision or full asset
hash manifest for independent verification, without claiming M40 rehashed weights.

## 7. Packaging / clean install: remaining concrete work

M23 is a relocatable **runtime bundle with externally provisioned assets**, not a
from-zero dependency installer. Its clean-room test relocated ds41f but reused the
existing qualified Python/dependency environment. It is valid OFF evidence, not
proof that a fresh machine obtains candidate recipe/oMLX or host libraries.
`pyproject.toml` has no dependencies and no explicit build/package-data layout;
ordinary pip installation is not the qualified resource-delivery contract.
Builder copies Python, manifest, Rust source and local m21 acceptance binary;
it excludes dependency/native qualification machinery and copies only OFF docs.
`release_acceptance`/m21 acceptance exercises OFF stateless/persistence assumptions,
not MTP fenced certification. `bundle-record.json` recomputes hashes including its
previous self-content then rewrites itself: its self-entry is not independently
verifiable. RC should exclude the record from its own digest and verify all payloads.

Retain local Apple Silicon tarball distribution; portable native wheels are **not**
required. Provide locked explicit provisioning from delivered source exports and
host dependencies, building recipe natively on the target (Cargo/CMake/OpenCV at
provision time only), or deliver a host-specific binary plus verified link closure.
Chosen first RC: target-native build with recorded toolchain/link dependencies,
then installed operation without Cargo/CMake/Git. No ad hoc PYTHONPATH: wrapper
may deliberately set isolated bundle/import roots, but must not inherit arbitrary
PYTHONPATH shadowing or prefer unrelated installed packages. Use a fresh venv,
record actual module origins and hashes. Delivery must include supported helper,
MTP docs/schema and standalone installed acceptance driver plus Rust binary/header
boundary if claimed. Do not retain development paths in configuration defaults.

Clean-install gate: build/provision from recorded exports, unpack elsewhere, hide
original tree and qualification `/tmp` checkouts, clear PYTHONPATH/profile hidden
env, run inspect/start/accept, relocate once more, negative missing/modified module,
wrong native wheel, missing dylib, wrong tokenizer/checkpoint/profile tests. Model
checkpoint and SSD assets remain explicit external provisioned inputs. No universal
Mac compatibility or public registry publication follows.

## 8. Operator model (proposed, not implemented commands)

Explicit process flag/config `--profile mtp-singleton-v1`; omitted profile is
standard-off. Unknown/contradictory profile/limits fail, no request body or hidden
MTP environment switch. Inspect/print-config/start/accept all resolve the same
profile authority **before imports/model load**, avoiding module-global default
app creation for the wrong backend. Inspection prints requested/resolved profile,
ON/OFF/depth, selectors, supported routes/envelope version, limits, loopback policy,
source/dependency/native actual origins/digests and qualification artifact match.
Distinguish configured identity, environment-valid, installed candidate accepted,
and release-qualified: no inspect PASS implies checkpoint qualification.

`ds41f inspect --profile ...`, `start --profile ...`, and `accept --profile ...`
should work identically from source/bundle. Start fails for unapproved provenance,
non-loopback or limits inconsistent with M39. Standard-off remains default and
uses its unchanged dependencies; operator returns by stopping MTP and restarting
without explicit profile, selecting the preserved OFF environment. Existing MTP
sessions are lost, not converted. `MAX_LIVE_SESSIONS=4` must not silently override
MTP's one-session contract; inspect reports 1, explicit conflicting setting rejects.

## 9. Minimal composed release-candidate acceptance

All fresh acceptance gates bind installed runtime, helper, Rust boundary, manifest,
envelope and dependency/build/native identities. Fresh means the actual final
package, not simply rerunning a historical harness against a checkout.

| Gate | Inherit / rerun |
|---|---|
| M33 horizon, native preview parity; M34 transfer; M35 lease; M36 counterexamples; M36R cert/fence; M37–M39 workflow/lifetime invariants | Inherit owning-source/dependency proof only after exact identity comparison; rerun cheap current-source fixtures. Rebuild recipe: rerun native preview/parity + 28 representation fixtures. Any owning-source change requires affected gates, not blanket inheritance. |
| Provenance/profile | Fresh inspect without model, payload hashes, actual imports/build/link checks; wrong/unknown identities fail; no OFF-manifest inheritance. |
| Clean startup/readiness | Fresh installed process alive->ready, selectors/ON state/native patches, shutdown and restart to OFF; no injected experiment launcher. |
| Admission | Fresh all allowlisted fields and rejected alternatives, malformed/duplicate JSON/headers, ID lifecycle, 1 MiB chunked/content-length/compressed body, min/max token/context bounds, unknown routes, overlap/capacity. Snapshot counters/cache/fence/issuance to prove rejection before forbidden mutation. |
| Local security/resource ingress | Fresh bind/Host/Origin/content-type/proxy/worker negative tests, bounded prep/connection/body slots and deadlines. |
| Ordinary text and streaming | Fresh new/retained text, Unicode, usage, length/EOS; canonical 40 target/3 DSpark frontiers, zero retained replay/repack, no chunks beyond canonical ownership. |
| Completed tools | Fresh one/two calls and real result continuation with pinned declaration; repeated observation executes once; malformed result order/ID/capacity rejection. |
| Certified disconnect recovery | Fresh byte-loss inside UTF-8 and completed-call delivery loss; replace certified assistant, same-identity observation and next ordinary turn. |
| Unrecoverable/fresh + poison | Fresh partial DSML loss -> negative certificate/zero effects, DELETE/fresh real prefill; controlled poison/retirement failure containment; no universal partial repair. |
| Lifecycle/effect uncertainty | Cheap helper fault matrix + installed socket create/DELETE outcome loss; sticky stopped/tool_ambiguous, no extra requests/effects; documented manual path. |
| Stale/finite lifetime | Inherit M39 20,000-lifetime proof; cheap exhaustion/old-ID-after-eviction/64-bit ceiling tests + short installed lifecycle churn. Do not rerun weeks of history. |
| Cancellation/backpressure/bounds | Fresh real socket drop, finite stalled-send test, settled/active retries, lease release, native sync/clear and CPU diagnostic plateau; model bytes and client ledger measured separately. |
| Performance | Small matched OFF/MTP representative 4096-prompt/128-output pair, plus short text/tool HTTP observations; separate prefill/decode/acceptance/HTTP/reconcile/DELETE/fresh costs. Same sampling/workload, no universal speedup threshold. |
| Supported client boundary | Installed stable Python helper complete workflow; Rust raw fenced header/send/drop/error tests if advertised. Browser excluded. |
| OFF regression | Fresh omitted-profile startup with existing OFF identities, stateless Chat/Responses/Messages, stateful tool/stream, requested IDs/multiple sessions and idle persist/restore; existing cheap OFF/native/Rust suite. No default profile drift. |

Final acceptance artifact references the canonical current-source matrix and
explicit inherited scopes. Historical hashes/labels are evidence, not fresh package
qualification. If admission implementation alters native ownership, stop and
rerun the owning proof; no new blocker has been found requiring that redesign.

## 10. Context and conservative performance class

Supportable candidate bound: **8192 total encoded prompt + requested response,
<=768 output tokens**, every turn. Exhausted context explicitly requires a new
application session/history decision; no hidden truncation/persistence. This is
useful bounded local operation, not a meaningful claim for arbitrary long-running
agents. Greater context is optional post-release qualification, not M40 blocker;
**200K HTTP MTP remains unsupported**.

M33 matched model evidence: OFF 19.37 tok/s, guarded ON 42.84 tok/s, native
considered-draft acceptance 94.5%, guard/no-guard ratio 0.9966 on its controlled
4096/128 workload. M35 transport and M37–M39 client/recovery evidence separate
prefill and cleanup from decode. Define an **experimental opt-in accelerated,
workload-dependent local class**, not guaranteed 2x or prompt-universal speedup.
Final package needs the small matched gate above (stock-JIT, newly built recipe
and HTTP admission can change timing). Recovery is bounded correctness work, not
decode acceleration; fresh-session prefill/model load must never be reported as
retained decode throughput. No M40 optimization or 200K campaign.

## 11. Repository audit / housekeeping classification

M40 read-only scanner logs are retained even on failure; no historical artifact
is edited to make global checks green.

| Finding | Classification / action |
|---|---|
| M34 baseline authority labels in architecture plan, omlx_suffix_math.py and old M4 trajectory tool | Documentation/source-label debt, not proof of current runtime math failure. Preserve historical explanation; current operator docs must point to official model/recipe authority and scoped compatibility evidence. |
| Four legacy candidate-consumer/compressed-KV fixture/validation hash labels | Historical metadata and tooling debt: missing hash methods plus a real source-span digest mismatch at model.py 739–763. Do not assume every mismatch is only a final-newline difference or rewrite to inherit new evidence. Independently compare current owning sources before inheritance. |
| Historical HEAD/source hash assertions and M35/M38 named-ID harnesses | Harmless historical evidence; tooling needs explicit historical/current modes. M39's named-ID contract changed; do not run an obsolete harness as release evidence. |
| M33–M39 failed/superseded attempts with PASS/RUNNING inner labels | Historical evidence, canonical decision/manifest is authority; no directory-wide PASS aggregation. |
| pyproject authority candidate-doc link missing; archive/current docs links | Documentation debt, not an executable dependency. Current MTP bundle must include its own canonical API/operations/schema docs. |
| OFF-only manifest/package versions, candidate `/tmp` origins, native recipe not in inspection, helper excluded by runtime projection | **Release provenance ambiguity and required release blocker** until separate profile/build/client identities exist. |
| Self-containing bundle digest, missing dependency delivery/provisioning, OFF installed acceptance only | **Release tooling/distribution blockers**, not a native ownership defect. |
| Remaining global scanner failures | Not a passed release gate. Scope fresh acceptance to explicit current authority while retaining findings; any new current executable drift is blocking. |

### M40 checks actually run

Candidate namespace: 65 Python tests + 8 subtests pass (M39 evidence/admission,
M37 client, M38 faults, configuration, metadata, stateful policy and MTP boundary).
Five M40 evaluation tests pass. `cargo test`: seven boundary tests pass. Five
preserved release-OFF dependency/native-ABI tests pass in the OFF environment
(one Pydantic deprecation warning). Read-only
repository-wide authority-label and source-identity scans fail on the preserved
findings above. These are not suppressed PASS gates. Commands/log identities are
in the machine record. No checkpoint, clean-install or performance gate was run.

## 12. Ranked blockers and one next milestone

Required for bounded opt-in release, in priority order:

1. **Public admission/local-security boundary**: strict profile route/field allowlist,
   mandatory fencing, sanitized outcome/diagnostics, bounded body/preparation/socket
   ingress and SSE deadlines, loopback/Host/Origin enforcement and rejection-atomic
   tests. Current internal backend alone is insufficient.
2. **Reproducible provenance/delivery**: exported/pinned candidates, native target
   build/link closure and actual imports, profile-aware manifest/identity fail-closed
   inspection; correct independently verifiable bundle record and clean provisioning.
3. **Supported application/operator boundary**: explicit process selector and
   handshake, stable local helper/manual states, raw Rust fence support if claimed,
   docs/config and startup/inspection/return-to-OFF workflow.
4. **Fresh installed composed acceptance**: matrix above and conservative matched
   performance observations; no promotion from development-checkout evidence.

Optional OFF parity (not blockers): arbitrary tool declarations, browser integration,
stateless protocols, nonzero sampling/reasoning envelopes, more context. Explicitly
deferred: default/implicit MTP, persistence/restart recovery, distributed identity,
crash-safe external exactly-once, concurrent/shared/batched MTP, immediate token-exact
abort, unlimited backpressure, portable native wheels, universal partial DSML and
remote unauthenticated multi-user service.

Recommend **M41: implement and qualify the complete explicit bounded local MTP
release candidate**. One cohesive change set: finalize this allowlist/schema,
profile authority + ingress/security policy, vendored dependency exports/target
provisioning + provenance, supported helper/raw transport + operator docs, bundle
and clean-installed composed acceptance, with OFF regression. No selector-only,
packaging-only or recovery micro-milestones. A failed fresh gate is evidence for
revising the candidate or identifying a genuinely new blocker; M40 does not
preauthorize a PASS or label the candidate release-qualified.
