# Correctness model

The current correctness contract is a hierarchy, not a chronology.

## Authority hierarchy

1. **Official checkpoint/data** — immutable model data is the highest authority for weights and checkpoint identity.
2. **Reviewed official DeepSeek reference semantics** — reviewed source behavior defines model semantics where available.
3. **ds41f official-source-derived validators/contracts** — local fixtures and validators derived from official semantics qualify bounded domains.
4. **Imported implementation regression evidence** — historical native evidence is retained locally to protect lifecycle and integration behavior.
5. **Optimized production implementation** — production kernels and runtime code must conform to the higher authorities and regression evidence.

## Donor roles

- DwarfStar: architecture/design donor only.
- oMLX: compatibility, implementation, and performance donor only.
- Historical native repository: implementation origin only.  Required source and evidence are imported locally; it is not used for current qualification or as a dependency.

## Validated domains

Current local evidence covers official primitive semantics, selected connected prefill and incremental state contracts, sampling arithmetic seams, imported implementation integrity, native source closure, and checkpoint-free native build/tests.

## Non-claims

The repository does not currently claim:

- PyTorch RNG bitstream parity for native stochastic sampling.
- Full MLX-enabled imported-core build validation in this environment.
- Full native checkpoint execution qualification.
- Native HTTP serving path qualification.
- Long-context production qualification.
- Vision, DSpark/MTP production support, or release readiness unless explicitly added to `qualification.md`.

## Handling numerical differences

Historical CPU/oMLX observations are useful diagnostics but do not define official model semantics.  Cross-backend numerical differences are documented as observations unless tied to official-source-derived validators.
