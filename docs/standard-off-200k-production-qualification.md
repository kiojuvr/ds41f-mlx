# Post-M48 standard-OFF 200K production qualification

Decision: **QUALIFIED_BOUNDED_200K_FIRST_PARTY_OFF** on Mac Studio M3 Ultra,
512 GiB unified memory. This is fresh development/core production evidence,
not a new release, R1 regression, runtime promotion or practical context ceiling.

## What actually ran

The official checkpoint and admitted first-party M48 loader/model/math/cache/SSD
Engram implementation ran through the real runtime:

```text
DeepSeekRecipeRuntimeBackend.load / infer (short serving integration)
DwarfStarMLXPrefillSession -> DENSE_P0_P7 -> P5 terminal holdout
-> TargetGenerationSession / all-40-layer transaction
-> M8 exact-prefix P6 continuation -> M9 idle save
-> process exit -> fresh model admission/load -> exact restore -> more turns
```

No oracle executor, donor loading fallback, alternate KV authority, profiler,
custom kernel or speculative switch was used. Displaced donor model imports
were forbidden. Numerical modules, checkpoint bytes, native profiles, tokenizer
and dispatch remained admitted. M44–M48 ownership boundaries were preserved.
The M6/M20/M24/M10 evidence was reviewed, not rerun as an end in itself. Their
short HTTP/tool and corruption semantics remain bounded prior evidence.

The initial **200,000-token** context is tokenizer-encoded repository
architecture/state/Engram/code/operator text interleaved with numbered maintenance
journal sections; it is not the historical synthetic-ID endpoint fixture.
The prefix prefill consumes **199,999** tokens; P5 consumes the one held-out
terminal token. After the initial 256-token decode, repeated follow-ups alternate
1, 17, 257 and 2,049-token exact-prefix suffixes, representing short questions and
larger tool/document results. These are token-level core turns, **not** claims of
41 fully reconstructed recipe conversations or client-side tool executions.
The recipe backend itself is also exercised in the same initial process.

The final qualification runs **three sequential processes**, with **two fresh
restores**, **41 continuation/probe turns**, **8,387 actually decoded tokens**,
and **five cancellations after seven committed tokens** followed by reuse.
Final frontier: **229,281**. This is growth from a 200K baseline, not a fresh
229K prefill qualification. Corrected-run wall time totals **1,747.23 s (29.1 min)**,
including model verification/load; it is finite evidence, not an overnight soak.

## Structural production defect and fix

The pre-policy run succeeded numerically but exposed a resource-policy gap:
MLX free-buffer cache rose from **96.19 to 114.43 GB** while live state stayed
near **301.5 GB**. The measured prior allocator cache limit was
**522,268,023,193 bytes**: free-cache budget plus the resident model could exceed
physical RAM. The OS swap counter rose by about 43 MB during that preliminary
process; that global counter alone does not attribute every swapped page to ds41f.
This was allocator retention, not executable KV growth or hidden cache repacking.

`AdmittedResources` now owns a **32 GiB maximum free allocation cache budget**
for the entire OFF model lifetime, acquired before allocation and retained across
prefill, decode and idle P6. A smaller caller budget is preserved. Retirement/load
failure restores the prior setting; synchronized exclusive policy acquisition
rejects overlapping model lifetimes rather than nesting global restorations.
Failed restoration revokes execution but retains ownership until cleanup retry.
Runtime close still closes model resources and drops its references if restoration
fails. No per-token/turn `clear_cache`, live-state eviction, numerical change,
replay, repack, or new model-state owner was introduced.

The same initial 200K workload was rerun after this fix. Against the pre-policy
receipt, **every generated token**, consumed-history digest, persisted **280-slot**
inventory and post-probe **280-slot** inventory remained exactly equal (shape,
dtype and raw-byte SHA256, including BF16). The fix is inside the correctness
boundary, not a fidelity/performance tradeoff.

## Performance and resources after the fix

| Measurement | Result |
| --- | ---: |
| 199,999-token prefix prefill | 248.10 s / **806.13 tok/s** |
| Held-out terminal bootstrap | 9.05 s |
| Initial 256-token decode, excluding bootstrap/prefill | **19.12 tok/s** |
| Sustained non-cancel decode batches (32–256 tokens) | **19.03–19.34 tok/s**, median **19.13** |
| Early / middle / late batch medians | **19.09 / 19.21 / 19.11 tok/s** |
| One-token suffix bootstrap | about **0.071–0.078 s** |
| 17-token suffix bootstrap | about **0.31–0.34 s** |
| 257-token suffix bootstrap | about **1.00–1.02 s** |
| 2,049-token suffix bootstrap | about **8.37–8.87 s** |
| Model verification/load, per fresh process | about **315–317 s**, separate from inference |
| MLX peak active allocation | **319.79 GB (297.83 GiB)** |
| Maximum sampled active + free allocation cache | **339.97 GB (316.62 GiB)** |
| Peak sampled process RSS | **22.43 GB**, not total unified-memory usage |
| System swap used, corrected-run start / end | **293,797,888 / 293,797,888 bytes** |

The initial request's prefill + terminal bootstrap + first decode latency is about
**257.2 s after model readiness**, not a 50 ms long-prompt TTFT claim. Fresh-load
verification is additional. The first restored probe's bootstrap is about 3.1 s;
later short follow-ups are warm. Timings include the production completion barriers;
there is no profiling observer in the decode loop.

The pre-policy prefill was 247.76 s / 807.24 tok/s and initial decode 19.18 tok/s:
there is no meaningful prefill/decode regression after bounding retention. Its
first 2,049-token append took 25.14 s, versus 8.87 s after the fix; no isolated
kernel speedup is claimed. Corrected 2K suffix bootstrap remains seconds, not
instantaneous, and is a practical interactive-latency limitation.

During the corrected initial process, idle active allocation moves only from
301.54 to 301.72 GB with context growth; free cache stays near **32 GiB** instead
of continuing the preliminary accumulation. Fresh restored processes begin with
smaller free pools and grow within the same policy. File descriptors stay at
13 through measured idle turns and fall to 9 on backend/model close. Thread counts
are recorded (up to 97 in these finite runs), not an arbitrary-thread leak proof.
Close restores the prior allocator limit and retires SSD resources; the collector
still holds passive model references when taking its final memory sample. Physical
weight deallocation is therefore not inferred from that sample; processes exit
before the next model is loaded.

## State, persistence and recovery

- One live packed **40-layer list**, unchanged across generation/idle/P6 admission;
  all 40 offsets equal consumed token history after every observed boundary.
- P5 transfers exactly once. Initial prefill telemetry records 196 SSD Engram
  prefetch/consume events, logical matches, and **zero foreground fallback**.
- Prompt replay, full-cache repack and continuation reconstruction: **zero**.
- Every turn checks exact-prefix extension. Invalid-prefix requests reject before
  mutation, preserving the existing list/frontier; there is no fresh-prefill retry.
- Cancellation settles between complete transactions, discards sampled lookahead,
  closes generation/wired-limit ownership, and resumes through ordinary suffix
  append. It is not in-transaction abort or a universal transport recovery claim.
- Saves at frontiers **213,150 / 226,090 / 229,235** produce artifacts of
  **194.59 / 206.24 / 209.08 MB**, in **0.347 / 0.365 / 0.360 s**.
- Fresh-process restores take **0.087 / 0.091 s after model load**. Each retains
  exact history and every physical slot, reconstructs first-party cache classes,
  and reproduces the next 32 greedy tokens and all 280 post-decode slots exactly
  against the original live branch. Subsequent repeated turns and another save
  succeed without replay/repack.
- Backend generation-session count is zero on cleanup; the process-global
  allocator setting is restored exactly in all three corrected receipts.

Affected tests: **154 passed, 24 subtests**. These include first-party numerical
and resource admission, all-layer transaction failures, state production, P5/P6/P7,
continuation, new budget acquisition/retirement/load-failure/concurrent-owner
negatives, and evidence gates that refuse prefill-only, short-context, incomplete
restore, replay/repack, invalid P5 and cleanup receipts. No full R1 was run.

## Evidence and reproduction

Canonical summary: `artifacts/standard-off-200k/summary.json`.
Full corrected receipts: `initial.json`, `restored.json`, `restored-again.json` in
that directory. They retain individual token latencies, turn boundaries, memory
samples, exact slot inventories, source hashes and admitted identity. Supporting
pre-policy receipts are under `before-policy/`; they are **not** the final healthy
resource qualification. `environment.json` and `final-tests.txt` identify the
actual packages/device and affected tests.

The machine's normally selected recipe/NumPy/Transformers had drifted from M47–M48
admitted bytes. Startup correctly refused recipe and then NumPy before allocation.
The run selected existing hash-matching local cached packages explicitly, without
changing the shared venv or relaxing pins. The first rejected attempt also exposed
an incidental collector cleanup bug (`len()` on the integer active-session count),
fixed before any real-model qualification; it was not a runtime/model defect.
The environment receipt records those
origins: NumPy 2.5.3, Transformers 5.18.0, recipe 0.1.1 with the admitted wrapper
and native binary, MLX 0.32.2 and the admitted mlx-lm/native profiles. Operators
still need those exact resources, not merely the historical venv directory name.

With an admitted environment, run sequentially (never overlap model processes):

```bash
python tools/qualify_standard_off_long_session.py \
  --output artifacts/standard-off-200k/initial.json --artifact-root "$DS41F_KV_ROOT"
python tools/qualify_standard_off_long_session.py \
  --restore artifacts/standard-off-200k/initial.json \
  --output artifacts/standard-off-200k/restored.json --artifact-root "$DS41F_KV_ROOT"
python tools/qualify_standard_off_long_session.py \
  --restore artifacts/standard-off-200k/restored.json --turns 4 \
  --output artifacts/standard-off-200k/restored-again.json --artifact-root "$DS41F_KV_ROOT"
python tools/summarize_standard_off_long_session.py \
  artifacts/standard-off-200k/{initial,restored,restored-again}.json \
  --output artifacts/standard-off-200k/summary.json
```

The optional `--compare` checks a same-fixture pre-policy receipt; it is evidence
only, not an execution selector. The recorded initial fixture was built from the
corpus files at parent `227949c`, before the operator-document updates in this
closeout; its token digest is recorded. A future changed corpus is a fresh workload,
not automatically the identical-token comparison fixture.

## Remaining limits

This closes the first practical **200K first-party OFF core** baseline, not all
production implementation. Bounded single-flight maintenance/document-result
turns and same-backend idle artifacts are qualified here. It does not establish
unlimited session/process lifetime, all-prompt model behavior, loop freedom,
200K HTTP/client tool-loop recovery, crash durability, concurrent execution,
active-state persistence, cross-backend portability, immediate protected-phase
abort, or operation beyond this tested envelope. Existing fail-closed burn and
ownership semantics remain authoritative; no partial transaction is repaired.

**512K-class / practical context ceiling remains a later explicit task.** No Vision,
MTP/DSpark/speculation promotion, full R1, release/packaging/clean-room work or
`ds41f-runtime` promotion was undertaken.
