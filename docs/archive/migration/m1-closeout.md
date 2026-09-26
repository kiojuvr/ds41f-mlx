# M1 closeout

M1 is closed as a qualification/provenance contract reconstruction milestone. It does not qualify long context and does not begin native performance architecture work.

## Artifacts

```text
artifacts/m1/checkpoint-provenance.json
artifacts/m1/runtime-identity.json
artifacts/m1/api-atomicity/run-20260923-065918/result.json
artifacts/m1/summary.json
```

The bounded direct/server comparison remains the reviewed M0 oMLX-compatibility artifact. It is not an official DeepSeek correctness oracle:

```text
artifacts/m0/oracle-compare/direct-vs-server.json
```

## Gates

| Gate | Result |
| --- | --- |
| Checkpoint provenance | passed |
| Runtime/source identity | passed |
| Bounded direct/server oMLX compatibility artifact availability | passed |
| Invalid-request atomicity | passed |

## Provenance reconstructed

- Official checkpoint path: `/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash`
- Checkpoint hashes matched the M0 identity and qualification-oracle verification for config, index, tokenizer, and tokenizer config.
- Tensor count: `96085`
- Shard count: `48`
- Total payload: `510286023000`
- Qualification oracle: `/Volumes/SDXC-512/deepseek-v41-flash-mlx`, commit `1b7d0a2c7d33602437dffd44e26a33f39f189661`
- Pinned oMLX: `b390b31e0c6831225fed0f24d278eb1db7fcb68b`
- Local oMLX parser patch SHA-256: `a38227306c3bff58760ad1732717c57c51295573dde80cd8de2ea6955503cf1d`
- Known-good model settings record `mtp_enabled=true`, `deepseek_v41_engram_ssd_offload=true`, `dflash_enabled=false`, `vlm_mtp_enabled=false`, and `moe_expert_offload_enabled=false`.

## Invalid-request atomicity

The M1 API atomicity runner starts the current thin server with an audit log. Only the valid request reached the backend and returned explicit `501 runtime_unavailable`. Unknown model, empty messages, invalid temperature, zero `max_tokens`, and `max_tokens` above the admission guard all failed before backend invocation.

## Correctness-authority repair note

After the M0-M2 authority audit, the M1 direct/server artifact is classified as `omlx_compatibility_reference` and `not_official_qualification`. The checkpoint provenance and API atomicity conclusions remain valid; any model-correctness inference from oMLX output agreement is superseded.

## Decision

The minimum provenance/API contracts are reconstructed in this repo. The next architectural decision may use the clean provenance/API parts as gates, but M1 does not authorize official model correctness claims, long-context qualification, DwarfStar kernel migration, broad profiling, server redesign, checkpoint conversion, or optimization of the rejected naive direct `mlx_lm.generate_step` diagnostic path.
