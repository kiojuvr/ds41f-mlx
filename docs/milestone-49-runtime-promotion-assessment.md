# M49 — runtime promotion assessment (BLOCKED / NO)

Base: `5e784d9` on `ds41f-mlx` master. **This is a stopped promotion attempt,
not a completed M49 or a newly qualified production MTP envelope.** No production
selector, capability, dependency pin, reference fixture or runtime implementation
was changed. No `ds41f-runtime`, extraction, packaging or release work was done.
This historical assessment is now committed with the user's authorization;
committing its record does not authorize runtime promotion.

**Historical scope:** all descriptions of missing transactions below refer to
base `5e784d9`, not the current tree. M51 subsequently implemented accepted-prefix
settlement and M52 canonical speculative generation. The remaining proposal-side
boundary is recorded in [M53](milestone-53-first-party-dspark-proposal-boundary.md),
which remains NOT PASS. The original M49 evidence is preserved, not relabelled as
a fresh qualification of M51–M53.

## Decision

**NO:** ordinary `ds41f` cannot yet safely use MTP inside the canonical runtime
while retaining the M44–M48 ownership contract. Existing candidate performance
and isolated lifecycle qualification are not evidence of that integration.

The blocker is not a slow kernel, the weather schema, the profile name, or the
8K constant. It is the absence of an **owned speculative target transaction**
under the canonical generation/resource lease. Current production commits one
consumed token across all 40 layers before publishing history or sampling state.
Current MTP verifies multiple positions, rolls state back to an accepted prefix,
and retains a sampled anchor/response queue through a different scheduler. There
is no production transaction interface connecting those two state machines.

This is an engineering boundary to implement, not proof that promotion is
impossible. The negative probes below demonstrate the current boundary; removing
the guards would not implement the missing contract. This attempt stops rather
than relabeling the candidate as normal operation.

## Executable architecture audit

| Authority | Canonical standard runtime | Guarded candidate |
|---|---|---|
| Model/resource lifetime | `OmlxRuntime` → M47 `prepare_resources`/`bind_model` → first-party loader/model/SSD stores | `InternalMTPQualificationBackend.load` loads donor model and installs native MTP patches; separate executable identity |
| P5 | Same packed list to `TargetGenerationSession`, terminal once | Same logical target authority to native `BatchGenerator` through `OMLXMTPGenerationSession` |
| Scheduling/sampling/history | `TargetGenerationSession._consume`, publish only after target commit | Native `GenerationBatch` verification/acceptance/queue plus wrapper canonical/transport history |
| Target mutation | `TargetForwardTransaction` → `DecodeStateProducer`; exact admitted cache class, all-layer materialization/commit, burn on failure | Native captured backbone and rollback machinery; donor state-producing block execution |
| Idle | Exact-list return; retire old P6 capability metadata without tensor conversion | Native singleton extraction, semantic horizon and protected queue settlement |
| Application | Standard server, Chat/Web `RuntimeClient`/`StatefulToolChatClient` | Separate exact-byte fenced `LocalMTPClient`; Web and standard tool helper explicitly refuse it |

Neither isolated lane currently has two executable target truths. The danger is
**composing both active owners** or bypassing the canonical owners to make the
candidate appear standard. Native extraction is an established candidate
operation, not a full-cache repack; nevertheless it is not M44's exact-list
handoff/return contract and cannot silently replace it.

Specific executable boundaries:

- `model_execution/loading.py:load` rejects `preserve_mtp=True` before loading.
  `LanguageModel` deliberately does not construct DSpark stages/rings or native
  committed-context transfer hooks. Some diagnostic `_forward` capture/verify
  code exists; it is not the admitted production transaction and extracts rows.
- `resource_admission.py:bind_model` requires the exact first-party model, OFF
  configuration, checkpoint-derived configuration, admitted numerical classes and
  bound SSD descriptors. Donor MTP cannot inherit this capability.
- `target_generation.py:TargetGenerationSession.__post_init__` rejects speculative
  configuration. Its pending sample is unconsumed lookahead, not a native queue.
- `target_forward.py:TargetForwardTransaction.validate` accepts shape `(1,)`,
  rejects `_mtp_verify_state`, and requires all layers at the committed frontier.
  `execute` commits only `frontier + 1`; no accepted-prefix rollback operation
  exists. `DecodeStateProducer` has no speculative compressor-tail/window/index
  journal or accepted-prefix publication contract.
- `mtp_lifecycle.py` constructs native `BatchGenerator`, calls native generation,
  and uses native context ownership hooks/idle extraction. Native
  `_run_verify_cycle_chain` owns verification logits, acceptance sampling,
  rollback and queue publication, not just draft proposals. Guarded failure is
  fail-closed; this assessment does not characterize it as unsafe OFF fallback.
- `web.py` and `web_client.py` reject the candidate before state-changing effects.
  Removing that refusal alone loses the supported client's certified
  reconciliation/effect boundary. Arbitrary tools require the standard ordinary
  message/certificate/effect contract, not a copied weather-only predicate.

Source hashes, including the actually installed native Python implementations,
are in `artifacts/m49/source-audit.json`. Earlier authority/evidence:
[M41](milestone-41-local-mtp-release-candidate.md),
[M44](milestone-44-target-generation-ownership.md),
[M45](milestone-45-target-forward-ownership.md),
[M46](milestone-46-state-production-ownership.md),
[M47](milestone-47-execution-resource-admission.md),
[M48](milestone-48-first-party-model-execution.md),
[startup](mtp-startup-investigation.md) and
[verification compute](mtp-verification-compute-investigation.md).
The startup resource-placement fix does not supply a speculative commit interface.

## Required coherent implementation boundary (not implemented)

Retain one standard operator/server/application API and one target lifetime.
Acceleration should be selected explicitly at model/session creation, never by
request content, a failed verify, or an ambient experimental flag.

1. Extend the first-party admitted model lifetime with DSpark checkpoint modules,
   stateless proposal math and derived bounded rings. Bind those concrete modules,
   checkpoint parameters and resource handles through M47. Preserve the OFF
   lifetime allocator/wired policy and SSD prefetch drain/retirement; the native
   request wiring lease is not a replacement for that lifetime policy.
2. Extend the owned target transaction and M46 producer with a bounded in-place
   speculative block lease. It must distinguish tentative physical advancement
   from committed frontier, retain only bounded undo/publication information,
   materialize all 280 slots and settle an accepted prefix without replay/full
   repack. Do not use diagnostic model `_forward` as the production transaction
   or maintain a shadow executable cache. Any uncertain failure burns all aliases.
3. One canonical generation owner must own target sampling, RNG/filter state,
   accepted history, lookahead, terminal previews and response publication.
   DSpark proposes; it cannot independently authorize committed tokens or tools.
   Retain the existing semantic-horizon and no-proposal/no-verify protected idle
   settlement invariants, with cancellation only at coherent transaction edges.
4. Integrate that owner into the existing standard session/stream/application
   lifecycle. Exact-byte retry certificates and the living effect ledger must
   survive disconnect/re-entry without authorizing partial DSML effects. Native
   owners may be implementation donors, not a parallel live scheduler.

This is a coordinated generation/transaction/state-production/admission change,
not a profile alias or dependency-installation fix. It needs all-layer
accept/reject/rollback comparison, fault injection at each publication/materialize
boundary, real HTTP Chat/Web tool workloads and full unchanged R1 after integration.

## Envelope and OFF selection

**No canonical MTP context or capability is newly qualified here.** The candidate
remains 8,192 prompt-plus-output positions / 768 output tokens, deterministic
text-only singleton Chat, pinned weather tools and its explicit Python client.
That ceiling is historical candidate policy, **not a demonstrated implementation
or physical-memory limit**. It was not raised because the canonical transaction
is absent, not because 8K is assumed to be the desired final specification.
OFF's measured 1M text core cannot be inherited as MTP evidence.

The initial promotion should qualify local single-flight ordinary dialogue,
stateful continuation, JSON/SSE, declared tool/result loops through existing
Chat/Web, disconnect/cancel/reconcile/re-entry and retirement as one envelope.
Longer MTP contexts need actual context/ring wrap, compression-boundary rejection,
long P6 append, lifecycle and resource evidence on that same implementation.
No final context number is asserted before those runs.

Persistence/restore, Vision, Responses/Messages, stochastic sampling, concurrent
sessions and any other unqualified mode must reject **before mutation** in an MTP
lifetime until independently established. An operator may stop/retire the loaded
lifetime and start an explicitly selected OFF lifetime for those capabilities.
It is never an implicit same-session fallback or a KV conversion/restore promise.

Current ordinary startup is unchanged:

```sh
.venv/bin/python -m ds41f_mlx.ops start --profile standard-off
# Use the normally provisioned candidate environment ONLY for explicit candidate use:
"$HOME/.venvs/ds41f-mtp-investigation/bin/ds41f" start --profile mtp-singleton-v1
```

Set `DS41F_CHECKPOINT` to the official checkpoint when needed. The two existing
interpreters have different admitted native artifacts; swapping their binaries or
loosening identity pins is not an integration. Standard OFF remains the normal
control path; candidate selection remains explicit and is not Web support.

## Fresh evidence

See `artifacts/m49/README.md` for final run results and commands. The real ownership
probe loads the official checkpoint through the actual standard backend and M47
admission, executes 4K dense/P5/OFF decode, and tests unsupported interfaces
without speculative mutation or guard bypass. It compares every physical slot
before/after rejection and tests resource retirement. This is **blocker evidence,
not promotion qualification**. Existing owners' regression tests and fresh full
R1 lanes establish the unchanged baseline only: **full R1 OFF and candidate MTP both CONFORMANT / PASS, 24 gates
each**, unchanged reference identity; **96 owner regression tests PASS**.
Candidate HTTP passes 18 cases, with aligned idle frontiers, zero replay/repack
and zero settlement proposals/verify cycles. Candidate weather decode is
39.949/38.458 tok/s (one/two calls), result continuation 37.812/30.258 tok/s;
4K owned OFF control is 19.916 tok/s. These are separate workload measurements,
not token-identical speedup evidence. An initial MTP launch omitted mandatory
explicit checkpoint configuration and failed before model admission; its excluded
receipt is retained separately from the corrected passing run.

No integrated application MTP workload, expanded MTP context or promotion PASS
is claimed.
