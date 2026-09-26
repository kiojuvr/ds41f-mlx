# HC and MoE runtime

HC and MoE are part of the local native model core and are implemented under `native/include/dsv41`, `native/src/mhc`, `native/src/moe`, `native/metal/mhc`, and `native/metal/moe`.

## HC ownership

HC wraps both attention and FFN/MoE subblocks:

```text
HC pre-mix
  attention or FFN/MoE subblock
HC post/collapse
```

`pre_mix`-style intermediates are call-local handoffs, not persistent session state.  Correctness authority remains the reviewed official HC semantics and local validators; imported source is the production implementation.

## MoE structure

The production MoE path preserves legacy arithmetic and ordering:

1. gate computation
2. route selection
3. tie policy and expert ordering
4. shared expert computation
5. routed expert computation
6. route reduction
7. shared/routed merge in the imported order

Do not simplify routing arithmetic, tie handling, cast boundaries, expert ordering, or shared/routed merge order without a new authority-backed qualification.

## Expert storage and residency

Implemented components include:

- expert bank / atlas
- expert backing
- grouped expert pipeline
- runtime residency infrastructure

Current production policy is the imported accepted residency behavior.  Late file-backed residency work is classified as non-default candidate/archival unless separately promoted by evidence.  Static expert weights/backing are model data, not session state.

## Known distinctions

Backend/tie differences are documented limitations.  They are not repaired opportunistically during migration or documentation cleanup.

## Not production

- fused mHC decode candidate: rejected/deferred and unavailable as a production selector
- experimental decode graphs/performance-only candidates: diagnostic/archive only
- late file-backed residency candidate: not canonical production by default
