# Correctness-authority audit and repair plan

Audit target: `f068bb893ce4819ef02f98576520e8219a1ab5a6`.

## Authority hierarchy

1. **Model data authority:** official DeepSeek `DeepSeek-V4.1-Flash` checkpoint at `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`.
   Raw checkpoint tensors, config, tokenizer, shard/index identity, dtype, and directly derived raw-bit fixtures are authoritative model-data evidence.
2. **Model semantics authority:** reviewed official DeepSeek V4.1 Flash reference spans from the local official checkpoint snapshot. The snapshot contains `inference/model.py` with SHA-256 `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`. Each numerical fixture may use only the explicitly reviewed function/class span and declared operation contract recorded for that fixture; unreviewed spans are not oracles.
3. **Execution architecture authority:** DwarfStar `https://github.com/antirez/ds4.git` at `0aaea5a238fb41a35106a551e73c8409dfb751ac`, for prefill execution architecture, scheduling, ownership, lifetime, CED/deferred-decoder topology, and wait/drain behavior only.
4. **Legacy qualification archive:** `/Volumes/SDXC-512/deepseek-v41-flash-mlx`, useful only after each artifact is classified by provenance.
5. **oMLX:** implementation/runtime donor and compatibility/performance baseline only. Agreement with oMLX is not official model correctness.

## Audit decision

The repository is repairable in place. Contamination is mainly in semantic contracts, qualification evidence, and labels. Current native raw-checkpoint embedding and BF16 projection primitive fixtures are clean because their references are independently derived from official checkpoint data under explicit local arithmetic/data contracts.

## Current classification summary

| Component | Classification | Note |
| --- | --- | --- |
| M0 checkpoint/source identity | CLEAN | Provenance only. |
| M0 direct-vs-server oracle | VALID COMPATIBILITY ONLY | oMLX direct/server agreement, not official correctness. |
| M0.5 oMLX performance baselines | VALID PERFORMANCE / COMPATIBILITY ONLY | May be retained as baseline. |
| M1 checkpoint provenance | CLEAN | Official checkpoint hashes/counts. |
| M1 runtime identity/model settings | VALID COMPATIBILITY ONLY | Runtime provenance. |
| M1 bounded direct/server oracle inheritance | TAINTED if used as official correctness | Relabel as oMLX compatibility. |
| `ds41f_mlx/dwarfstar_semantics.py` historical semantics bridge | SUPERSEDED / TAINTED | Bound graph steps to oMLX as if official. |
| M2 state-publication fixtures | TAINTED correctness evidence | Useful oMLX internal-equivalence diagnostics only. |
| M2 layer-major correctness fixtures | TAINTED correctness evidence | Exactness against oMLX `_forward` only. |
| DwarfStar prefill smoke logits/cache gates | TAINTED for model semantics | Architecture plan is still clean. |
| DwarfStar sweep planner/topology | CLEAN — architecture only | Pinned DwarfStar authority. |
| Native arena/buffer/submission scaffold | CLEAN — architecture only | No model semantics claim. |
| Native official embedding fixture/validation | CLEAN — bounded primitive only | `ParallelEmbedding.forward` fixture and native BF16 gather validation; no logits/layer claim. |
| Native official BF16 linear fixture/validation | CLEAN — bounded primitive only | Non-quantized `linear()` / `F.linear` branch over official BF16 tensors + independent F32 arithmetic contract. |
| Native official RMSNorm fixture/validation | CLEAN — bounded primitive only | Reviewed `RMSNorm.forward`; native output is within the predeclared one-BF16-ULP tolerance. |
| Native official rotary no-YaRN fixture/validation | CLEAN — bounded primitive only | Reviewed `precompute_freqs_cis` / `apply_rotary_emb`; native output within predeclared F32 tolerance. |
| Native official rotary YaRN fixture/validation | CLEAN — bounded primitive only | Same reviewed rotary contract with bounded `original_seq_len=16`; no broader YaRN/model-shape claim. |
| Attention / Compressor / Indexer / Engram / MoE / HC / DSpark / cache semantics | NOT YET VALIDATED | Separate semantic review and fixtures required; oMLX/DwarfStar agreement is not sufficient. |

## Repair policy

- Do not delete historical artifacts.
- Do not rewrite git history.
- Preserve tainted evidence with explicit metadata: `historical_only`, `omlx_compatibility_reference`, `not_official_qualification`, `superseded`.
- No oMLX logits/cache/state/intermediate tensor may be called an official oracle unless independently revalidated against pinned official DeepSeek semantics.
- Split future gates into architecture, compatibility/performance, model-data, and model-semantics categories.

## Current freeze line

Native model-math expansion is frozen beyond the explicitly validated bounded primitive set: embedding gather, BF16 dense linear, RMSNorm, rotary no-YaRN, and rotary YaRN. Additional work is allowed only when it starts with a reviewed official-reference-derived or independent-arithmetic contract, a bounded fixture, and native validation for the exact new semantic boundary. Attention, Compressor/Indexer, Engram, MoE routing/expert math, Hyper-Connections, DSpark/MTP, sparse/quantized kernels, cache publication, logits, full layer, full prefill, and full model semantics remain outside the validated boundary.
