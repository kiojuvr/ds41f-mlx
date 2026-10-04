# M44 — owned single-stream target generation

## Boundary and authority

M42 Reference Release R1 and M43 one-way deterministic extraction remain closed,
unchanged baselines. M44 transfers the complete standard-off generation lease:
terminal bootstrap, target-call scheduling, normalized-logprob sampling,
consumed-token history/frontiers, EOS/length/cancel, failure invalidation,
working-set policy, and coherent return to idle continuation.

`ds41f_mlx/runtime/target_generation.py` is the production engine. P5 and the
stateless/stateful serving selectors use it exclusively. It invokes the qualified
LanguageModel target call directly; it does not construct, wrap, delegate to, or
extract state through BatchGenerator/GenerationBatch. `omlx_generation.py` remains
an explicit legacy comparison/R1-fixture implementation, never a fallback.
Bounded explicit MTP is unchanged and remains on its separately qualified engine.

This is the largest coherent boundary transferable without simultaneously
requalifying target-forward math, every packed cache producer, and SSD Engram.
The existing native/reference decoder is not silently substituted: its different
state representation would require conversion or a second executable authority.
Kernel-by-kernel donor removal would not transfer this generation authority.

## Preserved contracts

- Official checkpoint, existing FP8/packed-KV/precision policy and target model
  math are unchanged. R1 remains the primary conformance oracle.
- DENSE_P0_P7 and P7 SSD Engram remain selected; P5 terminal holdout is consumed
  exactly once. No prefix logits, prompt replay, cache repack, publication-state
  reconstruction, or portable-state admission is introduced.
- The original list and layer objects remain the sole executable cache lease.
  Sampled lookahead is not consumed history. Every delivered response corresponds
  to a consumed target token, including EOS/length; cancel discards lookahead.
- Idle transfer checks all layer frontiers before returning the exact list.
  A target failure burns the lease; continuation extraction and retry fail closed.
- P6 capability retirement is now explicit. An early real-model R1 probe found
  that preserving the original cache objects also preserved a revoked P6 seal;
  the old scheduler's row extraction had implicitly discarded that metadata.
  On coherent generation-to-idle transfer only stale Python P6 capability markers
  are retired. P5's old runner/setup remains handoff-reserved/transferred and
  cannot execute or reuse its old certificate. No tensor is replaced by this
  transition. A later append establishes a new P6 owner capability.
- The engine leases/restores the qualified device recommended wired-memory limit;
  this is required operational policy formerly hidden inside BatchGenerator.
- Default standard-off, single-flight HTTP, protocol/parser authority, persistence
  schema, security/capability scope, and M43 promotion direction are unchanged.

## Qualification

**Execution ownership transition: PASS.** Evidence is recorded under
`artifacts/m44`; the final milestone/release decision is `decision.json`.

- Immutable R1 standard-off full verifier: **PASS / CONFORMANT**, all isolated
  seam checks, protocol/preview/recovery fixtures and real-model OFF lifecycle.
  R1 manifest remains `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
- Owned-engine real-MLX tests: **10 passed**. Affected lifecycle/transport/request
  tests: **37 passed, 8 subtests**. These include failure burn, stale-alias
  rejection, terminal-only bootstrap failure, seal retirement, idle frontier
  failure, wired-limit restoration and refusal to construct an external scheduler.
  The same 10 checks pass in fresh independent delivery. An initially copied
  legacy host probe assumed an optional donor fast-kernel symbol was mandatory;
  delivery/R1 do not promise that symbol. The check now executes the required
  packed representation rather than adding a new kernel-presence admission rule.
  No production math or dependency source was changed for this test correction.
- Matched fresh P7 prefixes at **4096 and 32768 tokens**, **128 output tokens**
  each: token identity and exact equality of every **40×7 cache slot**, including
  packed bytes, pending compressor state and Engram history. Each decode forward
  is exactly one token; zero replay/repack, one P5 handoff. The owned engine keeps
  the same list through idle. Observed exactness is a same-backend regression
  result, not a new cross-backend precision requirement.
- Median decode throughput (legacy → owned): **19.75 → 19.78 tok/s at 4K** and
  **19.48 → 19.46 tok/s at 32K**. Ratios **1.0016 / 0.9988**: practically flat,
  no claimed speedup. Bootstrap is measured separately from steady-state decode.
  Fresh independent installed delivery repeated the full matched test: **19.34 →
  19.38 tok/s at 4K**, **18.83 → 18.88 tok/s at 32K** (ratios **1.0020 / 1.0030**),
  again identical tokens/all cache slots and explicit rejection of stale prefill
  certificates. These are practical measurements, not a universal timing promise.
- Owned **200000-token / 32-output-token** endpoint: **PASS**; prefill **237.30 s**,
  bootstrap **9.03 s**, median decode **19.01 tok/s**, MLX peak **319.79 GB**,
  coherent frontier **200032**, zero replay/repack. The established >=15 tok/s
  practical floor is preserved. This is not an expanded soak or no-swap claim.
- Operational probe using immutable R1 OFF inputs: **PASS** for three protocols,
  four requested session ids/capacity rejection, tool/result re-entry, SSE,
  persistence/restore, three corrupt-artifact rejections, continuation, session
  close and graceful shutdown. Fresh independent operator acceptance additionally
  covers cancellation/recovery; see the current tree-bound release receipt.

The generic historical-suite run is not a current release gate: old source-hash
attestations deliberately detect development changes, and importing all profile
fixtures into one process mixes dependency authorities. Their evidence is not
rewritten to make that suite green. R1 runs its checks in isolated processes.
Early failed probes are retained as bounded development evidence, not release
claims; the first off-only environment also lacked R1's guarded preview substrate
and was replaced by the documented R1-capable environment, not by changing R1.

The release surface includes the owned engine and its regression checks. The
changed payload requires fresh independent OFF/MTP operator acceptance, native
checks, R1 for both existing profiles, source-origin checks and deterministic
projection using unchanged M43 mechanics. Promotion is qualified only when the
current tree matches the attached PASS receipt in
`release/promotion-qualification.json` / the generated promotion envelope.
Historical milestone artifacts and immutable R1 materials are not rewritten;
M43's superseded receipt is preserved, not relabeled as this new source proof.

## Remaining substrate and exact next frontier

Temporary attributed implementation still supplies official checkpoint loading,
LanguageModel target-forward/all-layer mutation, DeepseekV41Cache representation,
quantization/custom kernels, SSD Engram/prefetch, and the separate guarded MTP
scheduler. DeepSeek-recipe remains the official protocol authority.

**Next frontier:** transfer LanguageModel target-forward orchestration and its
all-40-layer cache mutation/commit boundary onto the same P7-compatible packed
representation and qualified math. Do not introduce a native/MLX cache adapter,
replay, a second cache authority, or broaden MTP/capability scope to do so.
