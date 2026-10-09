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

This path keeps the bounded execution capabilities: one worker,
one executable native singleton, greedy/thinking-off text, prompt + output budget
<=8192, output <=768, and the existing weather-tool subset. Missing `max_tokens`
uses 128. Unsupported controls are rejected, not silently implemented. Applications
own their messages and branches. Independent requests may queue; they do not join
a shared native MTP batch. No persistence, crash recovery, exactly-once effects, or
implicit HTTP retry identity is promised.

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
  owns hash lookup, acquisition references, capacity and LRU eviction. The bounded
  context fits a root tail; four complete checkpoints are retained, with no
  independent prefix index or eviction policy. Hash-drop/clear hooks drop the
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
Pinned MLX arrays are functional values: immutable backing may be shared until an
update; mutable cache containers and ring objects are never shared. Acquisition
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
when their complete encoded prefix matches. At the context ceiling only the
earlier prompt pair is retained: no admitted future request can extend C=8192.

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

## Acceptance and regression

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
control/tool domains, transparent profile/resource policy and full production
observability in `production-serving-parity-closure.md` still need implementation.
Portable packaging/promotion, broader context, Web, new hosts and default-profile
changes remain outside this task. M54R/M55R's existing bounded execution approval
is not replaced by a new campaign or extended to those domains.
