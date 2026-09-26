# Correctness contract

The official checkpoint and pinned official DeepSeek model semantics are canonical. oMLX is an implementation donor, API/runtime compatibility reference, and performance baseline only. Agreement with oMLX logits/cache/state/intermediates is not official DeepSeek correctness.

See `docs/correctness-authority-audit.md` for the M0-M2 contamination audit and repair classification. See `docs/official-semantics-authority.md` for the candidate official reference source identity pin, `docs/semantic-oracle-plan.md` for the repaired oracle-generation sequence, and `docs/official-fixture-metadata-schema.md` for future fixture metadata requirements.

## Reference hierarchy

1. Official checkpoint/config/tokenizer/raw tensors and directly derived raw-bit fixtures.
2. Pinned official DeepSeek-V4.1-Flash reference implementation / published architecture for model semantics. Current candidate identity is recorded in `docs/official-semantics-authority.md`; it must be reviewed before broader model-math expansion.
3. Pinned DwarfStar architecture authority for execution topology only: `antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac`.
4. Legacy qualification archive artifacts only after per-artifact provenance classification.
5. oMLX known-good execution for compatibility/performance diagnostics only.
6. Optimized candidates, which must pass appropriately classified gates before promotion.

## Rules

- Do not modify official checkpoint files.
- Do not introduce unofficial quantization or approximation.
- Do not hide unsupported request features behind successful responses.
- Preserve invalid-request atomicity.
- Keep reference/exactness fixtures even after optimized paths exist.
- Optimized paths must match the appropriate authority at documented observable boundaries before performance claims.
- oMLX-derived exactness may be recorded as compatibility/regression evidence, but must be labeled `not_official_qualification` unless independently revalidated.
- Run `python3 tools/check_authority_labels.py` before commits that touch correctness/qualification language.

## M0 scope

M0 validates short bounded prompts, API wiring, tokenizer/prompt behavior, generation output, usage accounting, and error behavior. M0 does not qualify long context, DSpark production behavior, vision release behavior, or performance superiority.
