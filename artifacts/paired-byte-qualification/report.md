# Paired-KV byte qualification bugfix

**PASS — Production promoted and verified, 2026-10-28.** Independent branch
`fix/paired-byte-qualification`, from verified clean
`3444471c142b584279411dc338118bb0fcfe69c9`. Not a reopening of Multimodal-Runtime.

## Incident and decision

The Windows Harness incident's last success restored 25,423 positions and
published canonical frontier 25,633. Its next approximately 27,521-position
request repeatedly failed **before native start**, at baseline
`production_mtp.admit:90`: the combined `roots > 192 MiB OR projected > 12 MiB`
paired-byte qualification rejection. Runtime remained READY. Six source-matching
failures establish this diagnosis; `unexpected_exception` is not a competing
cause. **Which original operand exceeded its limit remains unknown.** No missing
incident measurements are reconstructed here, and no numerical corruption,
malformed-image, memory-exhaustion or network diagnosis is asserted.

Both constants are **redundant qualification-era restrictions**, not physical or
numerical invariants. The [original report](../cached-image-admission/report.md)
explicitly chose 12 MiB to bound each projected complete pair, including output
reservation and five speculative positions; 192 MiB bounded all retained roots.
Together they limited qualification coexistence to 192 + 3×12 = 228 MiB logical
arrays (roots, active restore, two captures). The preceding
[visual-memory report](../visual-history-memory/report.md) asked for that
coexistence/queue qualification. Neither report derives an algorithmic threshold,
an MLX allocation limit or a physical-memory cliff at either number.

The root sum includes **unrelated** indexed/pinned checkpoints, so unrelated
residency alone could reject a valid selected pair. It does not measure the
selected pair's compatibility or available system memory. The projection's
**layout validation remains necessary**: singleton dimensions, supported
compression ratios, typed compressor tails, shared placeholders and 128-position
DSpark rings must still pass. It remains called before restoration; only the
comparison against an arbitrary byte ceiling is removed.

## Upstream-first audit and constraints

Official `https://github.com/deepseek-ai/DeepInfer.git`, explicit clean revision
`553741557a57df44a46fb3ab72b6aafeaf945337`; see [source receipt](upstream.txt).
`ve/multimodal_processor.py:219–260` skips only images wholly covered by the
actual cached token frontier. `model_executor/models/deepseek_vl.py:41–72`
preserves full cached segment lengths and refuses fractional segment skipping.
This agrees with local exact image identities/full-span paired coverage, **not**
metadata-age-only discount or a fabricated shorter prefix.

Upstream resource management is distinct: `worker/base.py:211–268` profiles
Torch peaks and non-Torch allocations, reserves state/copy buffers, and sizes the
KV block pool from device utilization. `csrc/engine/block_manager.hpp:128` and
`scheduler.hpp:195,386–419` enforce that finite pool. It has no equivalent of
these local image-discount byte constants. This CUDA/NPU pool is **not** an MLX
resource reservation and has not been transplanted.

Locally, upstream oMLX PagedCache retains lookup/hash/refcount/LRU authority;
paired capture/restore clones cache containers and array values, validates all
40 Target/3 DSpark frontiers and publishes only settled immutable pairs. Its
18-root capacity and exact logical leases remain unchanged. One executable
Scheduler request, bounded queue/session count, context/output reservation,
source/container/identity checks, body/image bytes, all-history source pixels,
fresh Vision patches, P5/MTP handoff, settlement, cancellation and fatal fences
are unchanged. No cache format/topology, history owner, replay, offload or disk KV
mechanism is added.

These mechanisms bound different resources: cardinality/context are not physical
byte reservations; MLX/platform allocation enforcement is not a host-headroom
SLA. No claim that they prevent every possible exhaustion is made. Removing this
image-only qualification gate does not remove an allocator safeguard: the same
long text/state-growth path already exists independently. The affected finite
path below supplies physical evidence, not universal context qualification. No
actual evidence requires a replacement resource gate, and no new fixed limit is
substituted. Snapshot/restore can allocate; logical `nbytes`, MLX active/peak/cache
and physical process footprint remain separate and are never summed together.

## Minimal correction and bounded measurements

- `production_mtp.py`: retain profile, coverage, fresh-work and layout checks;
  remove only the combined fixed-byte rejection.
- `paired_checkpoint.py`: remove the two admission constants; retain estimator
  and all integrity/ownership operations.
- `capacity.py`: advertise complete-pair/layout conditions, not obsolete ceilings.
- `observability.py`: extend the existing scalar allowlist. The existing active/
  last-request projection records total planned/fresh patches, selected/terminal
  frontiers, selected/root/projected logical bytes, root/free/evictable block
  counts and booleans for each old condition. No prompts, image/tool payloads,
  native objects, exception strings, persistent Runtime storage or second index.

[Model-free measurements](final-tests.log), all with 38,272 total
patches, zero fresh patches, selected frontier 4,625 and terminal reservation
4,696:

| Fixture | Selected bytes | All-root bytes | Projected bytes | Old rejection |
|---|---:|---:|---:|---|
| Pair only | 20,984,512 | 20,984,512 | 20,984,512 | 12 MiB |
| Unrelated root only | 14,912 | 202,404,944 | 135,872 | 192 MiB |
| Both | 20,984,512 | 223,374,544 | 20,984,512 | both |
| After unrelated root eviction | 14,912 | 14,912 | 135,872 | neither |

The same three boundary fixtures fail their new acceptance assertions against
the exact baseline ([red regression](baseline-rejections.log)); the candidate
accepts them without changing constants, estimators, lookup or frontier length.
[Final tests](final-tests.log): **51 passed** (20 admission, 31 documentation-authority).
[Focused regressions](focused-tests.log):
**135 passed**; [lifecycle/numerical regressions](lifecycle-tests.log): **47 passed,
14 subtests passed** (overlapping admission tests, not additive unique counts).
These cover identity/miss/partial spans, missing paired payload, unsupported
compressor/ring/tail layouts, real fresh preparation budgets, changed/compacted
projections, cancellation/retirement, diagnostic privacy and READY versus fatal
health semantics. No historical label alone authorizes native discount.
The initial developer environment correctly failed installed admission; counted
runs used the independently admitted `/private/tmp/ds41f-step4ar1-tests` closure.

## Isolated native evidence

Before evidence is the retained baseline-era native `cached-image-admission/run-03`
receipt, not a replay of the Windows incident: 201,235,196 retained bytes and
12,411,424 projected bytes were accepted; 212,732,400 retained bytes then caused
pre-clone rejection, and a separate longer suffix hit the pair-byte fallback.
Those historical receipts remain unchanged. The new red/green model-free tests
isolate each operand; the following new native run crosses both in real state.

A copied disposable closure `/private/tmp/ds41f-paired-byte-candidate` received a
non-editable wheel built by the established owner with `--checkpoint-variant
flash-1017 --glm-profile native`; no dependency/checkpoint changes. Wheel SHA256:
`e52c27d6ad96bfe29a272c1b006266f1a1d9091f345334ee80896821f82f9e31`.
[Installed/checkpoint admission](native-01/identity.json) preceded loading; every
Runtime Python source matched the candidate ([source proof](native-01/sources.json)).
Build provenance is baseline 3444471 plus the working changes, not a falsely
claimed clean commit. Operational Flash-1017 revision remains
`1bae0e1088993b9b82de08f59de5be5a34e42170`, execution identity
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.

`tools/qualify_paired_byte_continuation.py` uses one permitted 1280×720 dashboard
fixture, eight positioned occurrences (38,272 planned patches), then twelve
legitimate 1,809-position text appends. No original Windows history, tools or
external effects. **20 native Messages generations, eight fresh encodes and 124
covered-occurrence reuses**;
subsequent continuations encode zero images. All 40 Target/3 DSpark offsets align
at each settlement and capture; publication succeeds; replay/repack are zero.
The first old-pair crossing restores C=10,173, projects 13,806,944 bytes, and
settles C=11,985. The first all-root crossing has 220,950,476 retained bytes,
restores C=17,421, and settles C=19,233 (both old conditions true).
The final prompt is **26,478**, selected C=24,669, settled **C=26,481**;
selected actual logical bytes 25,063,924, projected bytes 26,708,384, retained
before restore 334,891,836. Final active-pair logical bytes are 26,676,604,
measured by its shape-preserving capture, not inferred physical allocation.
Forty distinct prompt/final publications leave at most 18 roots, exercising
bounded upstream replacement, then shutdown clears roots and requests.

[Raw samples](native-01/samples.jsonl), [host observations](native-01/host.jsonl),
[requests](native-01/requests.json), [recomputed summary](native-summary.json):

| Resource | Observed maximum/minimum |
|---|---:|
| Process sampled footprint / lifetime physical high-water | 297.263 / **300.535 GiB** |
| MLX sampled active / per-request peak active | 290.630 / 290.745 GiB |
| MLX allocator cache | 8.627 GiB |
| Retained logical arrays / one complete pair | 363,920,076 / 26,676,604 bytes |
| Minimum free+inactive+speculative host reserve proxy | **195.059 GiB** |
| Pressure free percentage | 40–98% |
| Swap usage / swapins / swapouts | **0 / 0 / 0** |

Finite controller: 1,800 seconds, <28K prompt positions, 16 output tokens, <8 MiB
bodies, 360 GiB process/MLX-active+cache stop, 100 GiB host reserve proxy, zero
swap and pressure floor 25%. All pass; the reserve proxy is not guaranteed
allocatable memory. Shutdown uses the serving owner, not exit as settlement;
weights remain loaded until process exit. The retained executed controller is in
`native-01/executed-probe.py`; the maintained driver additionally avoids safety
exceptions inside observational lifecycle hooks so cleanup is never interrupted.
No max-context, 600-image, endurance or arbitrary-host-load guarantee follows.

[Candidate HTTP acceptance](candidate-http/result.json) separately exercises the
installed server, native Messages, `/health`, scalar local diagnostics, recoverable
failure with READY, corrected pair boundary, and graceful exit 0. It restores
C=10,157, projects 13,792,704 bytes, and settles C=11,969 with aligned state and
publication, zero replay/repack. Stdout/stderr were `/dev/null`.

## Production delivery and rollback

Before mutation: [no Runtime/listeners/active external work](production-ownership-before.txt),
so no owned Runtime needed stopping. [Read-only verification](production-before.json)
matched all 124 installed package members to the diagnostic wheel with provenance
`623c35c8db28582525a98ceaa73a3fc12d066a16`; this is the expected installed
bugfix underlying the requested baseline, not an unknown installation.

Rollback directory:
`/Volumes/SDXC-512/ds41f-1017-paired-byte-rollback-3444471/`.
Its archive contains **136 RECORD-owned files**, all independently rehashed:
`previous-package.tar.gz` SHA256
`cc6637bc17d98d4a06382f9e0ba12f6b5f4542f4d7fddbda727fc7d201b1801c`.
The diagnostic wheel and older rollback remain untouched. The qualified candidate
wheel is retained in this directory's `promoted-wheel/`.

Installed with `uv pip install --python /Volumes/SDXC-512/ds41f-1017/bin/python
--no-deps --reinstall <retained-wheel>`. [Installed verification](production-installed-verification.json)
checks every wheel package member and unchanged dependency distributions,
versions and direct origins. [Production native HTTP acceptance](production-http/result.json)
**PASS**: 12 bounded Messages generations, selected C=10,157, projected pair
13,792,704 bytes (>12 MiB), settled C=11,969, complete alignment/publication,
zero replay/repack, READY health and local scalar diagnostics. A recoverable
preparation rejection remained READY. Graceful shutdown exited 0, no unsafe
conditions or remaining listeners/Runtime; [post-check](production-ownership-after.txt).
Production lifetime physical high-water was 300.571 GiB, minimum observed host
reserve proxy 189.440 GiB; swap remained zero. Candidate and Production HTTP
summaries are [here](http-summary.json). The retained
`executed-http-probe.py` is the exact acceptance driver; the maintained tool adds
finite-request and failed-start cleanup hardening (syntax checked, not assigned
additional native evidence). Installed source and dependencies were reverified
after shutdown. No rollback was needed.

Rollback only after graceful owned shutdown and confirming no requests/listeners:
verify the archive digest; `uv pip uninstall --python <target>/bin/python ds41f-mlx`,
then `tar -xpf <rollback>/previous-package.tar.gz -C <target>`. Rehash each file
against `previous-package.json`, run installed ordinary/checkpoint admission with
explicit `DS41F_CHECKPOINT` before restart. Archive recovery is verified; no
rollback execution is claimed unless recorded here.

Reproduce only in an independently admitted non-editable environment: build via
`tools/build_execution_package.py`, install with `--no-deps`, then run the new
native driver with `-I --source <checkout> --output <new-directory> --image
<permitted-render.png>` and explicit operational `DS41F_CHECKPOINT`. HTTP driver
`tools/accept_paired_byte_http.py --env <candidate> --image <fixture> --output
<new-directory>` refuses an existing listener and shuts down only its own process.
Old cached-image qualification tools/receipts describe the baseline limits, not
this changed boundary. Harness history/offload/compaction and Recipe protocol
ownership, all other independent limits, and completed campaign status remain
unchanged. No push, merge, tag or original Windows agent turn.
