# M46 — owned subordinate decode state production

## Boundary selected

M42 R1, M43 promotion, M44 generation and M45 target-forward remain closed
baselines. Beneath `TargetForwardTransaction`, standard-off now uses ds41f
`runtime/state_production.py::DecodeStateProducer` on **one unpadded token and
the original P7-compatible 40-layer packed list**:

```text
ds41f TargetGenerationSession
  → ds41f TargetForwardTransaction (pending lease / whole-list commit or burn)
    → ds41f DecodeStateProducer (block and packed state production)
      → attributed numerical / packed storage / kernel primitives
```

The selected coherent boundary is block HC/attention/FFN sequencing plus the
attention/compressor/index state producers, not an arbitrary dependency removal.
No donor `Block`, `Attention`, `Compressor` or `Indexer` call executes on the OFF
decode path. Loaded modules provide the existing weights and stateless operations;
there is no new module tree, cache class, tensor format or execution selector.

## Repository evidence and authority transfer

The donor `language.py` seams are:

- `Compressor.__call__`: writes partial KV/gates (slots 4/5), pools completed
  groups and optionally writes `_mtp_verify_state['compressor']`.
- `Attention.__call__`: overwrites packed local window (slot 1), initializes and
  appends compressed KV (slot 2), and publishes/reuses `shared['kv']`/`['idx']`.
  It also optionally records a verification window side channel.
- `Indexer.__call__`: truncates/appends packed index keys (slot 3), publishes
  `index_k`, creates candidate blocks and reuses candidate publications.
- `Block.__call__`: owns the order in which these state producers run; its CED
  branch can clear the window and alter execution origin. None of CED/replay or
  verification is part of the qualified one-token OFF transaction.
- `DeepseekV41Cache`: ordinary slot assignment does not advance a hidden offset;
  offset is slot 0. Creation/extraction/merge/reconstruction methods exist but are
  not called by this decode producer. Inherited `ArraysCache.advance` mutates
  admission `lengths` and `left_padding`, not just numerical cache arrays.

M46 moves these OFF state-producing decisions into ds41f. The existing packing,
RoPE, projection, pooling, chronological index selection and attention operations
are retained in the same order. The producer manages source/reuse and candidate
publication on the parent's token-local dictionary, including no-completion
compressor steps and completed groups. Writes go directly to the authoritative
packed objects while the entire lease is pending: they are **tentative mutations,
not independently publishable commits**. No rollback/retry/recovery is retained.

M45 still owns embedding/hash entry, Engram/block order, slot 0 and history
publication, absent-slot initialization, head projection and all-layer commit.
Admission-metadata subtraction is now explicit ds41f code rather than an external
`advance` call. Completion materializes these metadata arrays along with all seven
slots, then synchronizes. New structural completion checks every window,
compressed/index prefix, compressor remainder and Engram-history length,
including empty non-source slots, before pending is cleared. Preflight rejects
verification side-channel state and a non-seven-slot layout. The producer requires
a pending lease and a one-token input. Exact object identities are checked at the
barrier; a replacement fault burns both original and replacement aliases.

Failure before mutation, during any producer, in SSD reads, after all mutations,
or at completion burns the **whole original lease**. Generation history does not
advance and no continuation can be extracted or readmitted. No clone, shadow
production cache, hidden conversion, prompt replay or full repack is introduced.

## Engram and retained attributed substrate

Engram does **not** need to move with the packed producers:

- `NgramHash.__call__` returns hashes and lookback history without mutating the
  caller's history. M45 decides when to invoke it and when slot 6 is written.
- `Engram.__call__` receives activation tensors/hash IDs, not request cache; it
  reads checkpoint rows and returns a residual activation. Tables are not updated
  by decode. No external Engram continuation publication was found.
- `DiskEngramEmbedding` holds checkpoint file/mapping/residency resources and a
  transient prefetched row future. `TensorFile._seen_pages` tracks I/O page touches,
  not model frontiers. `EngramPrefetch.submit/drain/forward` owns the mechanics of
  one pending read, not a token/cache commit. M45 owns submission timing, read
  participation and scope retirement; exceptions invalidate the transaction.
  Real SSD-future tests prove `_pending` and `_prefetched` are cleared on both
  success and read failure. These resource mechanisms remain subordinate.

Checkpoint loading, raw packed cache storage, quantization/dequantization,
RoPE/HC/index/attention kernels, norms, linear/quantized projections, MoE math,
Ngram hashing and Engram numerical/SSD read mechanisms remain attributed/external.
They receive no mutable authoritative request-cache argument on this path.
The retained projection helper `_input_projections(x)` only projects/quantizes its
input; it has no cache, frontier or publication lifecycle. None of these primitives
creates continuation caches, advances hidden decode frontiers or recovers failed
state. General donor prefill/batched/MTP state producers remain executable only
on those separate, unchanged paths, not as an OFF fallback.

The owned code is MIT-derived, not claimed as independently invented model math.
`artifacts/m46/provenance.json` binds donor source, license, owned sources and the
byte-identical frozen M45 transaction from authoritative `e09fb69`. Qualification
uses that control rather than M44's external LanguageModel. Final control arrays
are passive comparison evidence from independent fresh P7 prefills, never a
correctness crutch or a second production authority.

## Qualification

- Affected producer/P5/P6/P7/generation/continuation/tool/termination/transport/
  request lifecycle tests: **112 passed, 32 subtests**. Producer tests compare
  real loaded numerical modules against M45 producer semantics at every prefix
  for ratios 0/1/2/4, compressor completion/remainder, packed window eviction,
  source/reuse, candidate generation and chronological index selection. They also
  forbid all four donor state-producer calls. Explicit TinyBlock lifecycle doubles
  are test-only injection below the changed seam, never production fallbacks.
- Failure injection covers before block mutation, after window write, after
  compressor-tail write, after index publication, after compressed publication,
  after block execution, structural completion, object replacement, SSD-future
  failure, plus retained M45 layer/sampler/synchronization/frontier failures.
  Exact-object cancellation/resume and whole-lease alias rejection remain passing.
- Final matched real-checkpoint **4096 and 32768 context / 128 generated tokens**:
  **PASS**, identical tokens and every final **40 × 7** packed/tail/history slot.
  All 40 frontiers are checked per step; one terminal P5 handoff, one-token calls,
  exact live-list idle transfer, zero replay/repack and stale-prefill rejection.
  Donor LanguageModel/extract/merge and donor state-producer calls are forbidden
  during owned decode (unchanged dense prefill may use them).
- Median throughput M45 → owned: **19.883 → 19.844 tok/s at 4K**,
  **19.624 → 19.617 tok/s at 32K**, ratios **0.99801 / 0.99962**. Practical
  >=15 tok/s floor passes; no architectural regression or speedup claim.
  Exact same-backend tokens/cache are regression evidence, not a new global
  semantic requirement.
- Full immutable R1 standard-off: **PASS / CONFORMANT**, all **24** isolated
  seam/protocol/preview/recovery/real-model gates. Manifest remains
  `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
  The real OFF gate exercises all three protocols, capacity/tool/result/SSE
  re-entry, restore/continuation, corruption rejection and shutdown. Results are
  in `artifacts/m46/r1-final.json`; final decision is bound in `decision.json`.
  The official checkpoint, backend-local fidelity policy, dense P0-P7/P5,
  SSD-backed Engram, standard-off default and bounded MTP are unchanged.

Initial instrumentation accidentally prohibited unchanged prefill's donor blocks;
that failed bounded probe is retained, not qualification evidence. A subsequent
passing matched run predates the added structural-completion gate; only
`matched-final.json` qualifies the final production source. No environments,
promotion projections/copies or receipts were produced. Immutable R1 verification
uses its existing historical substrate fixture in the development environment;
this is not release promotion. `release/runtime-surface.json` only includes the
new source/test fixture for a future explicit M43 checkpoint.

## Decision and next frontier

**M46: PASS — qualified ds41f-mlx development state only. No runtime promotion.**

No irreversible hidden continuation-state ownership remains at the selected OFF
producer seam. SSD read-future/page/resource state is intentionally not confused
with model continuation state; general prefill/MTP authority is outside this
milestone. Release promotion remains a separate explicit M43 checkpoint.

The remaining architectural coupling is **loaded numerical-module/checkpoint and
SSD-resource binding**: the producer consumes loader-created configuration,
weights, projection methods and SSD handles, rather than a separately admitted
immutable primitive bundle. Resource lifetime/residency remains in the attributed
storage implementation, under existing model-owner close and per-step drained
read scopes. Evidence does not require a representation rewrite or replacement
of Engram; any next transfer should address an actual primitive-admission/resource
lifecycle need at that seam, not merely remove oMLX names or dependencies.
