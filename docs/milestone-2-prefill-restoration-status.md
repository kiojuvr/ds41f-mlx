# Milestone 2 prefill restoration status

Status: **COMPLETE — BOUNDED OFFICIAL-CHECKPOINT DWARFSTAR-DERIVED PREFILL PATH THROUGH FINAL LOGITS AND NEUTRAL HANDOFF IMPLEMENTED**.

This document records the current Milestone 2 state. It does not select a decode architecture; Milestone 3 begins from the neutral committed prefill/session handoff described below.

## Governing constraint

Milestone 2 required a real full-model prefill path for the official DeepSeek-V4.1-Flash checkpoint that executes with DwarfStar-derived `ds41_graph_prefill_sweep` execution/lifetime topology. A wrapper around the current native correctness/reference runtime was not sufficient if it preserved `TextBackboneReference` / `TextEncoderReference` / `TextDecoderReference` control flow.

Correctness authority remains official checkpoint/data, reviewed official DeepSeek semantics, official-source-derived `ds41f` validators/contracts, and implementation-scoped regression evidence. DwarfStar and oMLX remain architecture/implementation sources, not semantic authorities.

## Implemented full bounded production-prefill path

The repository now contains a vertically integrated DwarfStar-derived production-prefill executor path through all transformer layers, final model output, logits, transaction commit, and neutral handoff:

```text
ds41f_mlx/dwarfstar_prefill_slice.py
ds41f_mlx/official_model_math.py
ds41f_mlx/prefill_session.py
tools/run_m2_dwarfstar_prefill_vertical_slice.py
artifacts/m2/dwarfstar-prefill/full-prefill-layer0-39-final-logits-handoff.json
artifacts/native-engram-connected-deterministic-logits-validation.json
```

The completed bounded path demonstrates:

- native DwarfStar-derived sweep plan ownership and command ordering;
- executor-owned typed token, embedding, HC carry, pre-mix, shared publication, final-output, and transaction state;
- official checkpoint embedding gather through the native prefill library;
- complete real model execution for layers 0..39 using production-facing official model-math and Engram seams, without calling `TextBackboneReference`, `TextEncoderReference`, or `TextDecoderReference`;
- real Engram@1 and Engram@14 execution using regenerated token/ngram hashes, sparse SSD-backed Engram row reads, FP8 WKV projection, source-defined gate arithmetic, and residual updates;
- full source generations at layers 2, 8, 14, and 20;
- candidate publication at layer 20;
- index-only/top-k refresh generations at layers 24, 28, 32, and 36;
- per-field ownership tracking where compressed KV, index K, and candidates remain owned by source@20 while top-k ownership advances through the later index refreshes;
- downstream consumers through layer 39 validated against the correct per-field producer identity, not merely digest equality;
- final Hyper-Connection collapse, final RMSNorm, official output head, and final-position logits;
- transaction invalid during sweep, final output private until complete, and commit only after final output/session handoff is ready;
- architecture-neutral executable prefill-to-session/decode handoff via live `PrefillContinuationState` plus digest/provenance `PrefillSessionHandoff` artifact view, without selecting DwarfStar or oMLX decode;
- exact connected official-source-derived comparison from official tokens through final logits.

This is a bounded structural/correctness proof for the production-prefill path, not long-context performance qualification.

## Handoff contract

`ds41f_mlx/prefill_session.py` defines the neutral Milestone-3 starting contract. It separates live executable state from artifact evidence:

- `PrefillContinuationState` is the actual committed continuation state. It owns/detaches live arrays for token history, regenerated Engram hash history, all per-layer window KV states, source compressed KV generations, source index K generations, candidate state, top-k generations including index refreshes, shared publications, field ownership, and compressor-pending state classifications.
- `PrefillSessionHandoff` is the artifact/evidence view. It records token frontier, final logits digest, committed shared state digests, HC/pre-mix evidence digests, Engram history digests, per-category inventory, ownership/provenance, `decode_architecture_selected = false`, and `requires_no_prefill_recompute = true`.

The no-recompute handoff validation is recorded in `artifacts/m2/dwarfstar-prefill/prefill-continuation-state-validation.json`. It runs one bounded full prefill, discards executor scratch, and verifies required persistent state categories exclusively from the committed live continuation state.

Milestone 3 must choose/evaluate a decode architecture against this explicit live state contract.

## Remaining non-Milestone-2 work

Milestone 2 does not claim:

- final decode architecture selection;
- decode/MTP/DSpark implementation;
- API or multimodal integration;
- long-context performance qualification;
- replacement of every helper arithmetic import under `tools/` with native/runtime code.

`OfficialModelMath` remains an intermediate production-facing seam over reviewed official-source-derived helper arithmetic. This is an implementation dependency to continue reducing where it materially affects production ownership, but it no longer blocks the bounded full-model prefill path.

## Milestone 2 completion determination

Milestone 2 is **complete** for the bounded official-checkpoint production-prefill restoration gate:

- DwarfStar-derived V4.1 sweep/lifetime topology connected to real official model execution: **implemented for bounded full path**;
- official checkpoint data path preserved: **implemented**;
- executor-owned carry/state/publication/transaction ownership: **implemented**;
- real full-model prefill path through all 40 layers and final logits: **implemented**;
- correct executable state handoff into a neutral decode boundary without prompt recomputation: **implemented and no-recompute validated**;
- implementation-scoped correctness gates: **available and passing**.

Exact Milestone 3 starting point: decode architecture selection against `PrefillSessionHandoff`, comparing DwarfStar decode, oMLX decode, or a documented composition without recomputing prefill.
