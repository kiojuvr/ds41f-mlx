# Very-long-context first-party standard-OFF production qualification

This extends the closed [200K baseline](standard-off-200k-production-qualification.md)
from `31a7e1d`, without redesigning it. Evidence is under
`artifacts/very-long-context/`. This is development/core qualification, not a
release, R1 regression, Vision work or `ds41f-runtime` promotion.

## Production path and workload

The same admitted official checkpoint, tokenizer, native profiles, numerical
implementation and first-party SSD Engram resources execute:

```text
recipe backend load/infer -> DENSE_P0_P7 -> P5 terminal exactly once
-> TargetGenerationSession / all-40-layer transaction
-> idle same-list P6 suffix append -> cancellation / re-entry
-> M9 save -> process exit -> fresh admission/load -> exact restore / more turns
```

Displaced donor imports are forbidden. MTP/DSpark/speculation, oracle executors,
custom kernels, P8 and alternative KV authorities are not execution selectors.
The fixture is tokenizer-encoded repository architecture/state/Engram/code and
operator documents with numbered maintenance-journal sections, extending the
200K workload. It is deliberately a bounded maintenance/document workload, not
an all-prompt quality or worst-case random-Engram-page qualification.

Each frontier uses a fresh prefill, 128 initial decoded tokens, eight suffix turns,
one save/32-token live-branch probe, then a **separate process** restore with exact
history and all **280 physical slots** (shape, dtype, raw-byte hash), a matched
32-token probe, four more turns, another save and probe. Suffixes alternate
1, 17, 257 and 2,049 tokens. Two cancellations after seven committed tokens per
frontier are followed by ordinary re-entry. Long turns exercise the actual core
seams, not fully reconstructed million-token recipe conversations or long HTTP
client/tool recovery.

## Exposed production defect and implementation repair

The initial 524,288-token attempt and a diagnostic repeat were **killed by the OS
in prefill**. Kernel records identify Python as the largest compressed process,
with 157,910 / 157,090 MB reported at kill. The diagnostic receipt remains RUNNING;
its supervisor records exit **137**, not a recoverable Python exception or PASS.
The durable ten-second resource stream shows MLX peak active allocation only
320.07 GB and bounded free-cache retention. The existing 32 GiB allocator fix was
working; raising that budget or discarding executable state would not solve this.

The residency lifetime was deficient: MLX's default **wired budget is zero**;
`TargetGenerationSession` acquired the recommended GPU residency budget only
while decoding and restored it on close. Initial dense prefill and idle P6 could
therefore run without that protection. Longer prefill exposed system compression
and jetsam, despite misleadingly large `psutil` available-memory readings. RSS
and MLX active bytes alone are insufficient pressure diagnostics on this machine.

`AdmittedResources` now acquires the device-recommended wired budget
(**498,216,206,336 bytes**) before model allocation and keeps it through prefill,
decode and idle continuation, under the **existing exclusive resource-policy
owner**. It is a budget, not a 498 GB allocation. Generation's existing lease
nests with the same setting and restores to the admitted lifetime setting.
Retirement synchronizes and restores wired and allocator settings exactly;
restoration failure revokes execution and retains ownership until retry. Load
failure restores acquired policy. No new state owner, cache format, eviction,
per-turn cache clear, precision change, replay or repack was introduced.

The **same 512K token digest** was requalified after this fix, then higher
frontiers were exercised with the same implementation. Corrected VM samples
record actual wired pages and small system compressor occupancy; no further
jetsam or swap growth was observed in the corrected finite runs. Tests cover
nondefault caller budgets, acquisition failure, synchronization/restoration
failure, retry, overlapping/concurrent owners, and the existing M44–M48 and P6
failure/ownership boundaries.

### Passive certificate lifetime defect

Very-long per-turn resource samples also exposed old source-publication retention:
at 768K, idle active allocation rose by about **2.22 GB** across the short finite
turn sequence; at near-1M it rose then dropped when cyclic GC ran. Fresh restored
processes stayed much smaller. This was not the consumed KV authority growing
by gigabytes for a few thousand new tokens.

`LivePrefillResult.from_committed` created a **setup → P6 certificate → final
setup** reference cycle. The setup's immutable old publication handles retained
large source arrays after the runner was revoked. They waited for Python cyclic
GC rather than the existing P5 retirement boundary.

P5 transfer/burn now removes **only the passive setup-to-certificate admission
backlink**, in `finally`. It is needed while ready, not after transfer or failure.
The live packed list is untouched; explicit external diagnostic aliases remain
passive. All runner revocation, one-shot result, stale certificate and failed
bootstrap burn guards remain intact. No live state is discarded, reconstructed,
replayed or repacked, and no per-turn `gc.collect`/cache clear is added.
GC-disabled tests verify immediate certificate reclamation after success **and**
failed bootstrap, same-list transfer and stale-certificate rejection.

All three frontiers were then rerun with the **identical corpus token fixtures**,
including fresh prefill and process restore. The Git corpus revision is an
**evidence-only fixture selector**, never a production executor selector. Every
generated token, consumed-history digest and all 280 persisted/post-probe physical
slots match the preceding wired-policy receipts exactly. This final campaign,
not the intermediate GC-dependent receipts, governs current support. The final
receipts cover **seven sequential processes**, **four fresh restores**, **46
continuation/probe turns**, **4,586 decoded tokens** and **six cancellations** in
**6,651.59 s (110.9 minutes)** including fresh verification/load. It is finite
qualification, not an overnight/unbounded soak. Affected tests: **189 passed,
26 subtests**; full R1 was not run.

## Measured frontiers

Decision: **QUALIFIED_BOUNDED_VERY_LONG_FIRST_PARTY_OFF_THROUGH_1M**.
**512K-class is production-qualified**, after the residency repair. Actual
qualified maximum: **1,048,576 total consumed tokens**. Largest fresh initial
context is **1,040,090**, with **1,040,089** tokens in prefix prefill; the held-out
terminal, decode and suffix turns reach the full frontier. The last artifact is
saved at **1,048,530** and a further fresh-process exact restore/probe reaches
**1,048,576** again. This is not a claim of a fresh 1,048,576-token prompt plus
unbounded generation.

The practical supported ceiling for this **single-flight text maintenance/core
envelope** is **1,048,576**, bounded by the admitted official checkpoint's 1M
context range. Its configured `max_seq_len=1048576` had been present at 200K and
was not qualification; only actual operation now establishes support through
that frontier. No configured limit was increased. The two numbers coincide
**after measurement**, not by treating admission as success. This is not an
absolute physical-memory exhaustion ceiling for the Mac: memory/SSD/state
representation did not fail within the checkpoint-supported range. No
above-checkpoint extrapolation or model-quality/correctness guarantee is inferred.
`max_seq_len` is not presented as a newly implemented hard overlength request
guard; operators must keep prompt + output + future suffix history in the
supported envelope.

Timings separate model verification/load, prefix prefill, terminal bootstrap,
decode and continuation. The existing 15 tok/s practical decode floor is retained;
admission or prefill alone cannot pass the evidence gate.

| Initial context | Prefix prefill | Initial decode | Sustained decode | 2,049-token suffix bootstrap |
| --- | ---: | ---: | ---: | ---: |
| 200,000 (closed baseline) | 248.10 s / 806.13 tok/s | 19.12 tok/s | ~19.13 tok/s median | 8.37–8.87 s |
| 524,288 | 762.91 s / 687.22 tok/s | 17.07 tok/s | 17.08–17.56 tok/s | 7.59–7.63 s |
| 786,432 | 1,310.01 s / 600.32 tok/s | 16.51 tok/s | 16.50–16.83 tok/s | 9.12–9.19 s |
| 1,040,090 → 1,048,576 | 1,947.39 s / 534.09 tok/s | 16.03 tok/s | 15.99–16.30 tok/s | 10.42–10.51 s |

| Frontier class | Peak MLX active | Max sampled active + free cache | Min sampled system headroom | Idle artifact / save | Fresh restore after model ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512K | 320.07 GB | 342.20 GB | 99.29 GB | 479.69–482.16 MB / 0.81–0.82 s | 0.206 s |
| 768K | 320.26 GB | 343.63 GB | 98.04 GB | 715.58–718.05 MB / 1.20 s | 0.306 s |
| near 1M | 320.44 GB | 348.60 GB | 96.41 GB | 943.83–946.30 MB / 1.59–1.61 s | 0.402 / 0.403 s |

GB/MB above are decimal. Free allocation cache stays at approximately 32 GiB;
small sampled accounting transients are not executable cache growth. System swap
start/end is unchanged at **284,819,456 bytes** for 512K, **284,819,456 →
276,430,848** for 768K, and unchanged at **276,430,848** for near-1M. VM samples, individual turn and
token latencies, idle active allocation/cache, RSS, threads and descriptors are
retained in receipts. Context growth increases the physical persisted state but
has no retention runaway; there is substantial measured headroom, not a claim
that macOS available memory is identical to allocatable GPU memory.

Cold resource verification/load is additional (roughly five minutes per process).
Long initial ingestion is batch/maintenance-class, not interactive TTFT. P5
terminal bootstrap is **5.52 / 5.57 / 11.27 s**, additional to prefix prefill;
decode medians are **17.30 / 16.57 / 16.07 tok/s**. Idle active allocation now
grows by only **4.82 / 4.82 / 4.91 MB** over the initial process turns, instead
of old publication buffers waiting for GC. Initial/final idle levels are
301.574→301.579 / 301.807→301.812 / 302.033→302.038 GB. Descriptors stay at 13 and
fall to 9 on close; sampled threads reach 96 in this finite evidence.
Decode remains usable once ready. Seconds-scale 2K continuation is observed, not an
independent optimization campaign; initial ingestion is the larger latency cost.

## Lifecycle and SSD interpretation

Every observed idle boundary checks one unchanged packed cache list, all 40
frontiers equal consumed history, exact-prefix admission and zero replay/repack.
Invalid prefixes reject before mutation; there is no fresh-prefill retry.
Cancellation settles between complete transactions, discards sampled lookahead,
returns the same authority, and restores generation's wired setting. No
in-transaction recovery or changed burn semantics is claimed.

Fresh-process restore checks every saved slot and consumed-history digest, then
reproduces all next 32 greedy tokens and every post-probe slot exactly against
the live branch. Saves, restores, subsequent appends, cancellation and a second
save all succeed. Backend active-generation count is zero on close; both global
resource settings restore exactly. Passive model references retained by the
collector mean its final memory sample is **not** proof of physical deallocation;
processes exit before the next model loads. The extra ceiling restart uses
`--restore-probe-only`: restore the final near-capacity artifact, reproduce its
exact saved live-branch probe to the full frontier, then close without appending
an artificial beyond-ceiling turn or pretending a new artifact was saved.

P7 telemetry checks SSD prefetch/consume/logical-match counts and **zero
foreground fallback**. SSD backing remains authoritative. Device-wide I/O deltas
separate admission/load from production: hundreds of GB are read from the
checkpoint SSD at fresh admission, while warm selected-row production reads are
small for this repeated-document corpus. Artifact writes land on USB SSD RAID.
These are global OS counters, not isolated logical read-byte attribution. OS page
cache is permitted; no cold-all-pages or high-diversity SSD latency guarantee is
inferred. Disk/volume identities and raw VM/I/O samples are retained.

## Evidence and reproduction

Use the exact admitted packages recorded in `environment.json`; directory names
or package version strings alone do not satisfy M47 admission. In that environment,
run the existing collector sequentially, never overlapping model processes:

```bash
python tools/qualify_standard_off_long_session.py --context 524288 --fixture-revision 31a7e1d \
  --turns 8 --decode-tokens 128 --output artifacts/very-long-context/512k-wired.json \
  --artifact-root "$DS41F_KV_ROOT"
python tools/qualify_standard_off_long_session.py \
  --restore artifacts/very-long-context/512k-wired.json --turns 4 --decode-tokens 128 \
  --output artifacts/very-long-context/512k-restored.json --artifact-root "$DS41F_KV_ROOT"
python tools/summarize_very_long_context.py \
  artifacts/very-long-context/{512k-wired,512k-restored}.json \
  --output artifacts/very-long-context/512k-summary.json
```

The collector now persists VM/MLX/swap/I/O samples even if the OS kills it. Receipt
gates refuse incomplete restore, short decode, nonproduction prefill, SSD
fallback, ownership/frontier/replay violations and failed policy retirement.
The old 200K summarizer remains defaulted to its historical scope. The final
closeout changes operator-document corpus text; use `--fixture-revision 31a7e1d`
for the recorded fixtures, or treat current-corpus runs as fresh workloads.
`tools/requalify_very_long_context.sh` records the complete sequential matched
rerun, including `--compare` against `before-certificate-retirement/` receipts;
`tools/close_very_long_context.py` verifies actual lifecycle gates, current runtime
source hashes and all same-fixture comparisons before writing `summary.json`.

## Remaining constraints

Finite single-flight core evidence only: no unbounded lifetime/leak proof,
concurrency, crash durability, active-state persistence, cross-backend restore,
immediate protected-phase abort, universal long-context model quality or full
long-HTTP/client-tool recovery claim. No Vision, MTP/DSpark/speculation, full R1,
release/packaging/clean-room or runtime promotion was performed.
