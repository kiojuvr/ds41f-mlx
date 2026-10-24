# Cached-image admission decision — PARTIAL

## Decision

**Separate accounting is semantically sound, but a safe replacement resource
envelope is not established. Keep production's 36,864 all-occurrence patch
guard. No execution candidate or inactive selector was added.**

The existing integration already skips fully cached images. The smallest plausible
change would retain early all-history security/metadata admission and enforce the
same 36,864 ceiling on whole-image active work after validated paired restoration,
before `_start`. Merely moving the existing check is unsafe as a resource argument:
it exposes substantial **pre-restoration quadratic identity-key allocation**, in
addition to unqualified restored/captured state peaks. This is PARTIAL, not an
architectural impossibility; no new image processor, cache/session owner, replay,
compaction, or arbitrary replacement threshold is justified here.

## Scope and provenance

- Starting checkout clean, branch `Visual-Feedback-Loop`, expected HEAD
  `2d55b3c15c85f000280ed03909180765806c2573` verified. Process inspection found
  no Runtime/Python/test worker; unrelated LM Studio helper and pi left untouched.
- Used existing `.venv`: Python 3.13.14, MLX 0.32.2, NumPy 2.5.3,
  Pillow 12.3.0, pytest 9.1.1, oMLX 0.7.0, mlx-lm
  `0.31.4.dev132+g94cdcae13`, Recipe 0.1.1. This environment is **not** the
  qualified Recipe 0.2.0 installed service. No dependencies or installations changed.
- Reverified clean read-only official tree `/tmp/ds41f-upstream-accounting`,
  DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
  Reused the [previous audit](upstream-cached-image-accounting-audit.md),
  [Q1/Q2 envelope](../../docs/flash-1017-visual-endurance-envelope.md), and
  implementation-plan responsibility policy; no Q1/Q2/F0 repetition or promotion.
- No checkpoint weights loaded, service launched, or numerical/model/memory
  qualification performed. Python allocation measurements below are not process
  RSS, Metal, wired-memory, or full-request high-water measurements.

## Upstream/local boundary confirmed

Official `MultimodalProcessor` strips only wholly covered binaries when supplied
with a nonzero frontier. `DeepSeekVL._split_cached_prefix` preserves token lengths;
its mapper uses zero cached crops and zero patch increments.
`prepare_input_metadata_vl` sums only nonzero active crop patches. The ordinary
`llm_engine.py:625` caller still omits that frontier, so the pinned ordinary engine
is **not evidence of an operational end-to-end cache-aware handoff**. This caveat
is resolved as a source fact, not a prerequisite to repairing official upstream or
a reason to duplicate its engine locally.

Locally `ProductionScheduler._schedule_waiting` already acquires/restores both
Target and DSpark, verifies logical canonical requirements, validates whole spans,
and requires coverage of every historical image before `_start`.
`PairedCheckpoint.restore` supplies the independent saved-image identity check;
the scheduler's request-derived `validate_prefix` argument alone does not.
`encode_new` uses this actual frontier and existing MLX Vision, not wire freshness,
to select work. No additional numerical mechanism is needed to skip old images.

## New concrete blocker: occurrence amplification before active admission

`serving/paired_checkpoint.py:image_ranges` calls
`image_keys(identities[:i+1])` for every occurrence. Each call reconstructs every
preceding `(start,length,digest,grid)` tuple, including a new grid tuple. Thus it
retains **N(N+1)/2 key entries**, not N. Ordinary `acquire` constructs this argument
**before** upstream `get_computed_blocks`, even on a complete miss, and reconstructs
it when seeking another eligible candidate. A post-restore active-work check
cannot protect this allocation. Retained-session direct acquisition avoids this
particular ordinary lookup cost, but cannot exempt ordinary Messages traffic.

With default checkpoint geometry, a valid **1×1 source** plans a **39×39 grid,
1,521 patches, 184 expanded positions**. Today's patch guard admits 24 such
occurrences (36,504 patches) and rejects 25. If only that guard is removed,
5,698 occurrences fit a 1,048,576-position context, with three surrounding text
tokens: 1,048,435 positions. Repeated valid 69-byte PNGs total just **393,162
encoded occurrence bytes and 5,698 source pixels**. Even inline base64/JSON is
well below ingress capacity for this synthetic prompt. Source-byte and pixel
bounds therefore do not replace occurrence/key admission. This is a preparation
counterexample, not a successful historical session or Recipe conversation.

Measured existing `image_ranges` in `.venv` with `tracemalloc`, tracing started
*after* constructing input identities (shared digest string):

| Occurrences | Retained key entries | Traced peak bytes | Elapsed seconds |
| ---: | ---: | ---: | ---: |
| 24 | 300 | 41,232 | 0.0001 |
| 256 | 32,896 | 4,463,504 | 0.0151 |
| 1,024 | 524,800 | 71,234,820 | 0.2302 |
| 5,698 | 16,236,451 | 2,208,546,724 | 7.1945 |

Allocator reuse affects small results; the quadratic entry count is exact.
No extrapolation of these timings into a production SLA is intended. The largest
case was actually measured, not extrapolated. Compact reproduction (no model):

```python
import time, tracemalloc
from ds41f_mlx.serving.paired_checkpoint import image_ranges
for n in (24, 256, 1024, 5698):
    images = [dict(start=1+i*184, length=184, sha256='a'*64,
                   grid=[39, 39]) for i in range(n)]
    tracemalloc.start()
    t = time.perf_counter()
    ranges = image_ranges(images)
    elapsed = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(n, sum(len(k) for _, k in ranges), peak, elapsed)
    del ranges
```

Changing key representation is not automatically safe: preserve upstream prefix
hash selection, image identity and existing published-pair keys. Alternatively,
an explicit history metadata envelope could bound the existing representation,
but its acceptable cost must be justified, not selected as an unexplained count.
This task stops rather than inventing that policy or expanding into a cache rewrite.

## Whole lifecycle: what still coexists

| Stage | Bound/ownership that must remain accounted for |
| --- | --- |
| Ingress/preparation | Whole-body 16 MiB; all-image encoded-byte, pixel, format, aspect, frame and digest checks; original Recipe/request source references, decoded transient pixels, expanded lists/tuples, kinds, span/identity dictionaries. Cached status is no security exemption. |
| Waiting | Installed upstream Scheduler admits at most `max(max_num_seqs*4,32)` waiting requests (32 here), plus one running. Preparation is serialized, but its lease is released after conversion; it is not an executable queue-memory lease. Each waiting request retains prepared tokens and sources. Session body/reconstruction references also survive independently of `_sources`. |
| Lookup/restore | Quadratic keys above, canonical token copies, immutable retained pairs and newly cloned/evaluated Target/DSpark handles. Immutable backing may share until update; independent handles do not prove zero physical copying. There are 18 retained pair slots plus null, not a global byte budget. |
| Vision/append | Only uncovered whole new spans normalize: per-image CPU float32 RGB patches, MLX bf16 conversion, encoder intermediates, cumulative sparse fresh embeddings; active/restored KV and suffix growth coexist. The patch count is a work proxy, not total-memory bytes. |
| Capture/publication | Prompt capture at N−2 occurs while fresh rows/active state remain; final capture occurs before publication can evict old roots. Retained pairs + active pair + unpublished prompt/final captures must be counted without assuming sharing. Canonical settlement remains publication authority. |
| Failure/cancellation | Queued abort clears sources and queue ownership; scheduler admission `finally` releases sources; encoder `finally` drops source/patch references; worker cancellation drains its consumer before cleanup. Running retirement settles/burns as existing rules require, clears active owners, and invalidates cache on failed native mutation. Request/Recipe/session references and allocator residency are distinct lifetimes; cleanup is not proof of an acceptable transient peak. |

A viable envelope must cover the maximum of preparation/queue, lookup/restore,
Vision/append/prompt-capture, and settlement/final-capture peaks, including retained
roots at each stage. Counting only fresh patches omits the first two peaks and
part of the latter two. Existing context and finite pair counts prove finiteness,
not acceptable byte peaks. Q1/Q2 do not measure these enlarged envelopes.

## Cases and focused evidence

- Whole cached history: zero active Vision work, but full validation, metadata,
  identity lookup and paired-state costs remain.
- Mixed/text-prefix hit: discount only whole spans covered by the actual restored
  pair; count all uncovered **new** grids. Never prorate an intersected image.
- Miss: all fresh images must still fit the same active ceiling; any uncovered
  historical image rejects before encoding. A miss must not evade metadata bounds.
- Partial span, changed digest/order/prompt, absent DSpark, lost session lease:
  preserve existing rejection/eligible-shorter-pair behavior; shorter coverage
  cannot authorize historical replay. Bytes are not recovery state.
- Cancellation: an eventual active-budget rejection should occur before
  `native_started`, retire restored owners, release sources and publish nothing;
  no such new path was implemented or qualified here.

Direct model-free assertions using actual Pillow and checkout preparation passed:
24 tiny images admitted / 25 rejected by unchanged production guard; fully cached
24 images skipped with patch preparation/encoder replaced by assertion failures;
partial frontier rejected and sources cleared. A scoped in-process test-only patch
to `MAX_REQUEST_PATCHES` allowed actual **5,698-image metadata preparation** with
patch materialization forbidden; the production constant was restored and checked
as 36,864. The key-allocation probe above used actual unchanged `image_ranges`.

Attempted existing focused pytest modules: `test_metadata_first_vision`,
`test_expanded_vision_capacity`, `test_production_mtp`,
`test_multimodal_cancellation`, `test_multimodal_restore_ownership`, excluding the
tiny-random-ViT numerical adapter test. **85 setup errors, one deselected; no tests
executed.** The session fixture's isolated `python -I` installed-payload inspection
fails `ModuleNotFoundError: ds41f_mlx` in this environment. No fixture bypass,
installation repair, or false PASS. Existing lifecycle tests remain source evidence,
not newly executed acceptance.

## Remaining adoption conditions

1. Justify a pre-lookup all-history metadata/occurrence envelope or compatible
   bounded key construction, including queued/request/session retention. Resolve
   its cost rather than moving an active check past it.
2. Bound restore/append/capture coexistence conservatively and verify physical
   sharing/copying and allocator peaks on the authorized installed workload.
3. Only then qualify the narrow active-budget change: unchanged fresh/miss limit,
   all rejection/cleanup cases, exact paired/canonical numerical continuity and
   measured RSS/Metal/allocator/wired peaks under the chosen history envelope.

Numerical and real-model memory work requires separate authorization. Current
contracts/capabilities and historical receipts are unchanged. No automatic
follow-on work, package publication, or push is authorized by this decision.
