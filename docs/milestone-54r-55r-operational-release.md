# M54R / M55R — operational sufficiency and normal-local release decision

## Decision and source boundary

Baseline: **`30710aa7a397196f6609ee978e51fa118181c07b`**, master after M53R.

- **M54R: PASS — evidence-complete by composition.** A separate operational
  campaign is unnecessary; no concrete supported-scope operational gap remains.
- **Remaining release blockers: zero.**
- **M55R: RELEASE APPROVED, PROMOTION DEFERRED.**
- **Release approved for normal-local source-clone profile; portable runtime
  projection deferred.** This is an operational release, not merely another
  candidate milestone. `standard-off` remains default and independently scoped.

This decision changes documentation only. Runtime, native, dependencies, admission,
reference material and executable capability policy are unchanged. No additional
operational run, soak, test campaign, probe, benchmark, schema, extraction or package
was started. The decision commit is discoverable with
`git log -1 --format=%H -- docs/milestone-54r-55r-operational-release.md`;
qualification remains bound to the baseline executable bytes, not a new model run.
This document governs the current release disposition over historical candidate
labels in milestone prose and health metadata; those labels do not broaden scope.

## Evidence sufficiency audit

The question is whether supported local singleton operation has a **specific
undecidable operational risk**, not whether a longer run might increase confidence.

| Operational risk | Existing evidence and composition | Disposition |
| --- | --- | --- |
| Physical state/topology drift, wrong accepted prefix or rollback | [M50R](milestone-50r-omlx-candidate-baseline.md): independent byte oracle (16 primitive + 2 live cases), native/Metal boundaries, real rejection/rollback, ring wraps, 18 measured sessions across five loaded processes. [M51R](milestone-51r-canonical-lifecycle-connection.md): identical five-workload topology/IDs, repeated oracle and bridge parity. M52R retains the physical graph; M53R changes setup/admission only. | Closed for unchanged qualified graph, not a changed kernel or broader workload. |
| Cancellation, ambiguous mutation, stale aliases, unsafe idle reuse | M51R real publication fault burns aliases and poisons retirement; affected failure/removal/foreign-response tests, shielded coherent settlement and retained lifecycle gates. M52R pending-delivery and consuming-terminal ownership preserve cancellation/publication ordering. M53R real stream cancellation and retained continuation exercise the admitted installation. | Closed; failed retirement/uncertainty blocks reuse, not silent recovery. |
| Application-specific leak or premature effect/turn closure | [M52R](milestone-52r-application-integration.md): consuming EOF, immutable call certificates, separate effect permission/full-turn completion, actual two-call turn, duplicate reuse and stored-result re-entry. M53R fresh real application exercises the same source/native; four effects, completed and partial tool loss, actual summary. | Closed for one/two weather calls; no arbitrary-tool claim. |
| Repeated lifetime retention or resurrected retired identities | [M34](milestone-34-operational-qualification.md) bounded 90-turn native workload plus recovery; [M38](milestone-38-client-operational-soak.md) passing 76-request living-client workload is supporting historical evidence, **not** a current-source soak PASS. Its actual closed-record blocker is closed by [M39](milestone-39-lifetime-admission.md): 20,000 structural lifetimes, collectible payloads, bounded diagnostics, no-wrap identity/sequence exhaustion, checkpoint create/retire/post-eviction integration. Current source retains those bounds and successful DELETE payload clearing; fresh M53R R1 includes M39 lifetime gates. | Closed by structural bounds plus affected fresh evidence. Historical counts are not extrapolated into unlimited uptime. |
| Fresh setup creates different lifecycle or bypassed admission | [M53R](milestone-53r-normal-local.md): absent venv/empty scratch, ordinary repository setup, strict wheel/native/package/linked-library/checkpoint admission, actual source/version drift rejection by inspect **and** reseal, operation without build scratch. Fresh full R1 uses normal operator startup/shutdown, no source allowance or forced shutdown. | Closed; different closure rejects rather than inherits support. |
| JSON/SSE loss, reconnect, retry or duplicate effects | M52R full R1 and M53R full R1: retained JSON, frozen exact retry, UTF-8/SSE/argument loss, completed-tool loss, negative partial call, duplicate reuse, result re-entry, cancellation/continuation and concurrency/unsupported rejection. Same frozen outcome and living effect ledger, not regenerated authority. | Closed within one living process/client; no crash-safe effects claim. |
| Performance/resource regression | M50R bounded residency/reload/co-residency evidence, M51R bracketed controls, M52R bracketed controls (-0.2994% decode, +0.2065% backbone), M53R matched sanity (+0.1531%, -0.1364%), identical request/IDs/frontier/depth and successful retirement, zero replay/repack. | No material regression or unexplained operational resource finding. Not a latency SLA or severe-pressure qualification. |

### Receipt checks performed during this audit

Read existing receipts only; these checks did not execute the runtime or qualify a
new workload:

- `artifacts/m53r/r1.json`: full `mtp-singleton-v1`, **PASS / CONFORMANT**, all
  **24 gates** return zero. `qualification_adaptation.source_allowance` is null;
  the five existing synthetic-fixture adapters are disclosed, not a production
  admission bypass.
- Every file in its `candidate_source` inventory matches the current file SHA256:
  **zero mismatches**. Its `candidate_commit` is the pre-commit M52R baseline;
  actual recorded source hashes include M53R setup/admission and bind the completed
  run. The commit field alone must not be mistaken for an older executable run.
- Decompressed `r1-gates/mtp-model.json.gz` and retained preview/recovery receipts
  match R1's recorded artifact digests. Model receipt: **18 cases, 27 admission
  assertions, four effects**, with both one-call and two-call result continuation.
- `artifacts/m53r/performance-comparison.json`: PASS, same request/IDs/depth,
  frontier 283, three measured sessions plus excluded warmup, all retired.
- `artifacts/m53r/negative-identities.json`: drift rejected, restoration PASS,
  installation seal unchanged. Setup/admission and shutdown provenance are recorded
  in M53R's final receipt inventory, excluding its development attempts.

M53R does not change the application or physical owner whose repeated operation
was previously bounded. Fresh strict-installation application/lifecycle evidence
connects that retained policy to normal-local delivery. Neither reading old
receipts nor absence of a new long-duration run is itself a qualification gap.
No supported-specific shutdown path, resource leak or post-setup inconsistency
remains uncovered by this composition. Therefore no minimum additional check is
needed. Unsupported capabilities below are not release prerequisites.

## Exact supported release envelope

- **Host:** Mac Studio Apple **M3 Ultra 512 GB class** (qualified Mac15,14),
  **macOS 26.5.2 / 25F84, arm64**, exact admitted host-linked closure under
  `/opt/homebrew`. Not general Apple Silicon/new-host/OS support.
- **Delivery:** location-bound `ds41f-mlx` source-clone/editable install via existing
  setup/operator paths. Repository-delivered exact M52R recipe 0.1.1 wheel/native,
  oMLX 0.7.0 stock source-JIT substrate, Python 3.13.15, MLX/MLX Metal 0.32.2,
  mlx-lm `0.31.4.dev132+g94cdcae13`; complete payload/dylib/package pins in
  `third_party/mtp/normal-local.json` remain admission authority. Not a generic
  wheel, tarball or independently qualified portable distribution.
- **Assets/execution:** official DeepSeek-V4.1-Flash revision
  `dba1be0a40aa45a94ad051997016db3960a90277`, all 48 byte-verified shards and pinned
  tokenizer/configuration; SSD-backed Engram, DENSE_P0_P7/P5 prefill/handoff,
  unchanged oMLX parallel proposal/causal block/prefix rollback topology and mixed
  precision. Requested depth metadata is 5; **actual native maximum is 3 drafts**,
  width up to 4, as frozen by M50R, not a newly qualified depth-5 graph.
- **Operation/security:** explicit `--profile mtp-singleton-v1`, trusted literal
  `127.0.0.1`, fixed port, exact Host/no Origin, no authentication; trusted operator,
  local peers, repository/interpreter/cache/resources static for the running
  lifetime. One process, worker, backend, live session, response lease, preparation
  slot and one-thread living Python `LocalMTPClient`/`RuntimeClient` application.
  Eight bounded sockets are ingress capacity, **not** concurrent inference support.
- **Workload:** greedy text, temperature 0, reasoning none, exact canonical retained
  continuation; encoded prompt/history plus requested output **<=8192**, output
  **1..768**, request body **<=1 MiB**, no truncation/replay/repack/context rollover.
  Sessionized Chat Completions plus health/models/create/get/delete only, as
  enumerated in [the operator/API contract](mtp-local-release-candidate.md).
- **Application:** ordinary text or the exact published `WEATHER` schema with
  **one/two lookup_weather calls per assistant turn**, ordered real string results
  and final continuation/summary. Consuming completion, not SSE deltas, grants
  effect permission. Exactly-once reservation and duplicate stored-result reuse
  apply **within the living client ledger** (128 non-evicted effects, <=64 KiB per
  result), not across process/client crashes or distributed effect owners.
- **Transport/lifecycle:** JSON/SSE, frozen exact-body/session/sequence retry and
  whole-outcome reconnect, no regeneration; shielded settlement, explicit DELETE
  and deliberate fresh creation, owned SIGINT/SIGTERM shutdown. Ambiguous effects
  or lifecycle loss stop; poison/failed retirement prohibits reuse. Finite no-wrap
  lifetime/sequence authority, 16 retired summaries, fixed 32-slot traces and one
  latest outcome. Exhaustion fails closed, never promises indefinite successful
  operation. Unsupported capabilities reject before request/model-state mutation;
  no fallback to OFF and no in-process profile switch.

## Release versus promotion/package

The supported audience is operators on the exact admitted host closure using the
source-clone profile. That reproducible installation is an acceptable formal
operational delivery now; a standalone `ds41f-runtime` release was not requested.
M43 extraction/clean-room packaging is **not necessary for this release decision**
and is not started. Existing runtime promotion receipts qualify only their own
historical trees; this decision does not update or relabel those packages.

If standalone distribution is separately requested later, select a source snapshot
and perform M43 deterministic projection, independent setup/profile acceptance and
promotion receipt in a **separate explicit promotion task**. Source-clone approval
is not a waiver of those rules and is not itself PROMOTION APPROVED.

## Future-only / explicitly unsupported

Vision/content parts, 1M or other unqualified long-context MTP, stochastic sampling,
persistence/restore/process-restart recovery, concurrency/batching, remote/browser/
Web gateway/Rust application use, arbitrary tools/schema or more than two weather
calls, stateless other APIs, diagnostics, inherited OFF features, crash-safe or
distributed exactly-once effects, universal partial DSML repair, token-exact
immediate abort, severe-pressure/unbounded-lifetime SLAs and portable/new-host/OS
packaging are not released here. Capability expansion, optimization, research and
portable packaging are independent future choices, not remaining release blockers.
