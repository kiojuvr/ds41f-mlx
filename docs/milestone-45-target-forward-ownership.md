# M45 — owned target-forward / all-layer decode transaction

## Selected boundary and moved authority

M42 R1, M43 promotion policy and M44 generation ownership remain closed baselines.
M45 replaces only M44's standard-off external LanguageModel target invocation:
**one unpadded row, one consumed token, the sole live P7-compatible 40-layer list**.
It does not replace general batched forward, prefill, vision, or DSpark verification.

`ds41f_mlx/runtime/target_forward.py::TargetForwardTransaction` is production
execution authority, entered exclusively by M44 `TargetGenerationSession._consume`.
There is no donor target call, scheduler, cache extraction/merge, or fallback:

1. Preflight token shape, layer count, compression layout, tokenizer-derived
   Engram map, admissibility and every row/frontier before any layer mutates.
2. Mark the entire lease pending. Embed/hash the token, initialize HC streams,
   create call-local shared CSA2 publications, and enter subordinate SSD Engram
   prefetch scope. Sequence Engram and all 40 blocks in qualified order directly
   on the original layer objects. Write each offset and Engram history; initialize
   absent empty slots using the qualified seven-slot representation only.
3. Collapse HC, normalize/project logits using unchanged numerical primitives.
   M44's sampler receives the same normalized logprobs. Materialize **every cache
   slot**, including compressor/history writes not necessarily dependencies of
   sampled logits, and synchronize the owned stream.
4. Validate every post-forward frontier, clear pending, then return consumed
   token and sampled lookahead. Only then does M44 publish history/frontier.

A failure at preflight, any layer/Engram operation, sampling, evaluation,
synchronization or final frontier validation invalidates **all 40 objects**.
Mutated objects are not rolled back, reconstructed or published as continuation.
Passive aliases fail existing P6 admission guards. Cancellation remains M44's
single-flight, between-transaction operation: finish the shielded step, discard
unconsumed lookahead, transfer the exact list at its committed frontier. There is
no new interruptible GPU, concurrent executor, retry, or public API contract.

## Retained subordinate substrate

The attributed loader still supplies checkpoint modules/weights and tokenizer
map. Numerical blocks retain HC/attention/CSA2/shared-KV/MoE math and mutations of
slots 1–5 as subordinate operations under the owned loop. HC collapse/head,
quantization/packed kernels, `DeepseekV41Cache` representation/methods, NgramHash,
SSD Engram and its I/O prefetch implementation remain external. These interfaces
receive only the token-local tensors/shared publications and the one live layer
cache; they do not schedule target steps or publish a committed continuation.

Decode sequencing is MIT-derived from the qualified oMLX single-row branch, not
claimed as independent model math. See `artifacts/m45/provenance.json` and
`ds41f_mlx/prefill_fp8_mlx/OMLX_MATH_LICENSE` (the subtree MIT license,
not oMLX's top-level Apache license). Keeping these primitives avoids simultaneous math,
representation, checkpoint and SSD redesign. Dependency count is not the goal.
The external general LanguageModel path remains for separately selected prefill,
legacy R1 fixtures/control comparisons and unchanged bounded MTP, never as an OFF
production fallback.

Official checkpoint, backend-local fidelity policy, DENSE_P0_P7/P5, standard-off
default, guarded bounded MTP, protocol/security/concurrency/vision/persistence scope
are unchanged. No executable cache clone, replay, repack, conversion, or speculative
duplicate target is introduced. Matched controls below use independent fresh P7
prefills and passive final arrays solely for comparison.

## Qualification and decision

Final pinned qualification is recorded in `artifacts/m45/decision.json`.

- Owned real-MLX lifecycle and affected P5/P6/continuation/tool/termination/
  transport/request checks: **77 passed, 32 subtests**. Owned-specific subset:
  **21 passed**. Failure injection covers layers 0/1/14/20/39, SSD Engram scope
  retirement, sampler, stream completion and final frontier mismatch. Tests prove
  no failed history publication, full-lease burn, stale-alias rejection,
  bootstrap/stop/length/cancel, wired-limit cleanup and exact-object resume.
- Immutable R1 **standard-off full qualification: PASS / CONFORMANT**, all 24
  isolated seam/protocol/preview/recovery/real-model gates. Manifest remains
  `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
  The real OFF gate covers three protocols, session capacity, tool/result/SSE
  re-entry, persistence corruption rejection, restore/continuation and shutdown.
  Bounded MTP was not independently requalified or broadened.
- Matched **4096 and 32768 tokens / 128 generated tokens**, frozen M44 consume
  control versus owned forward: **PASS**, identical tokens and every final cache
  slot at both contexts. Median throughput (M44 → owned) is **19.87 → 19.97 tok/s
  at 4K**, **19.56 → 19.64 tok/s at 32K** (ratios **1.0051 / 1.0042**). Instrumentation
  forbids donor forward and cache extract/merge on the owned path, verifies every
  40-layer frontier before/after each forward, exactly one input token per call,
  one P5 handoff, zero replay/full repack and same-list idle transfer. All 40×7
  final packed/tail/history slots are compared exactly. Same-backend exactness is
  regression evidence, not a new universal cross-backend numerical requirement.
- Practical decode floor remains >=15 tok/s; no optimization/speedup claim.
  `code-binding.json` verifies the frozen control's AST against authoritative
  pre-M45 HEAD and proves matched/final owned code identity apart from an
  attribution-only header correction.
  M44's 200K endpoint remains historical evidence, not a newly run M45 endpoint.

The existing development `.venv` initially lacked R1's native recipe binding and
installed guarded substrate. Only those missing R1 prerequisites were built/
installed in that existing environment from repository source exports; no fresh
OFF/MTP environment or runtime projection was made. Initial missing-import and
PYTHONPATH-rejection probes are preserved. A provenance probe also exposed the
pre-existing mlx-lm 0.32.0 package drift; final qualification restores the already
qualified `94cdcae13` pin. Earlier passing runs with 0.32.0 are exploratory evidence,
not the final qualification authority. Development Python is 3.13.14; the release
3.13.15 environment was not reconstructed or independently qualified here.

A historical M42/M43 mechanics-suite probe produced four expected stale-evidence/
receipt failures (six other checks passed); no projection completed and no receipt
was minted. This does not reopen those baselines or weaken their gates. M45 does
not run release acceptance, qualify/promote `ds41f-runtime`, or update existing
promotion receipts. Release-source templates/surface merely describe the new
implementation for a future explicit M43 checkpoint; previous receipts continue
to describe their bound historical trees.

**M45: PASS — qualified ds41f-mlx development state only.** No release promotion
claim or new release receipt is made. The canonical-doc static checker also
reports its pre-existing chronological `M2` token in `session-state.md`; this is
not an execution failure, and historical discussion/evidence was not removed to
make a broad old gate green.

## Evidence-based next frontier

The target loop and commit barrier are no longer missing seams. Matched results
show no practical decode regression requiring an optimization campaign. The
remaining architectural coupling is now **loaded block/attention state producers
and the packed-cache/SSD Engram primitive interface**, including shared CSA2
publications and compressor tails. A future change should first demonstrate a
specific ownership, stability or performance need at that subordinate seam; it
need not remove a dependency or change representation. No next implementation
milestone, kernel replacement, MTP expansion or release promotion is preassigned.
