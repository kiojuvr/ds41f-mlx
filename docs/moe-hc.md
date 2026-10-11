# HC and MoE runtime

This page owns **HC/MoE math, state classification and storage/dispatch boundaries**.
[Correctness](correctness.md) outranks implementation lineage;
[architecture](architecture.md) selects execution. The
[source edition](archive/editorial-baseline/moe-hc.md.txt) preserves the former
native-import production/residency descriptions and rejected candidates.

## HC ownership

HC wraps attention and FFN/MoE with pre-mix, subblock and post/collapse.
`pre_mix`, hidden values and related intermediates are call-local handoffs, not
independent persistent session authority. Preserve required normalization,
accumulation, cast and mixing semantics; a fused implementation must retain
necessary intermediate dtype boundaries, not merely similar final output.

Current OFF owners include `model_execution/language.py` and
`hyper_connection.py`; ordinary MTP retains admitted pinned model math.
Native HC headers/source/Metal remain executable reference and reusable material,
not the selected complete model engine.

## MoE structure

Preserve the selected qualified operation contract:

1. Gate computation and route selection.
2. Required tie handling and expert ordering on valid backend-produced values.
3. Shared and routed expert computation with admitted precision/cast boundaries.
4. Route weighting/reduction and shared/routed merge semantics.

Do not simplify routing arithmetic, ties, quantization, expert ordering or merge
boundaries without authority-backed affected qualification. Backend-local floating
trajectories need not share universal bit/logit/token identity; that is not license
to change a discrete algorithm or hide an invalid route.

OFF's first-party projection/routing/switch-layer owners choose admitted stateless
acceleration primitives. Kernel names, batch shapes, route counts and portable
primitive fallbacks do not select a donor model fallback or qualify another graph.

## Expert storage and residency

Static expert weights/backing are model data, not session state. Selected dense
prefill uses the full-resident backbone/SSD-Engram policy; OFF model lifetime and
allocator/wired admission are defined in [runtime strategy](runtime-strategy.md).
Do not inherit a native expert-bank/atlas/residency experiment as current OFF/MTP
policy or add file-backed expert dispatch merely because the reference contains it.
Checkpoint storage/precision and installed dependency/native identity remain
[admitted resources](runtime-admission.md), separately from host compatibility.

## Known distinctions

Historical backend/tie observations remain evidence, not semantic exceptions.
Review against official algorithms and the qualified backend configuration;
documentation cleanup neither repairs nor blesses numerical differences.

## Not production

The edition retains the rejected/deferred fused-mHC decode, experimental decode
graph and late file-backed residency classifications. Retained native/reference
or investigation material does not create a production selector. Any independently
justified future change needs affected math/state/performance qualification, not
a duplicate core or a new campaign requested by this page.
