# M1 qualification/provenance contract reconstruction

> Correctness-authority repair note: this document is historical. The M1 checkpoint provenance and API atomicity gates remain valid, but the bounded direct/server artifact is now classified as oMLX compatibility only, not official DeepSeek qualification. See `docs/correctness-authority-audit.md`.

M1 reconstructs the minimum provenance/API contracts needed before native performance architecture work. M0 and M0.5 remain closed.

## Scope

M1 records and verifies:

1. checkpoint inventory, digests, and provenance against the official checkpoint and historical qualification archive;
2. source/runtime identity for this repo, pinned oMLX, local oMLX patch, historical archive, and known-good model settings;
3. bounded direct/server oMLX compatibility using the existing M0 comparison;
4. invalid-request atomicity for the thin API boundary;
5. DSpark/MTP configuration identity required for the known-good decode path.

## Gates

| Gate | Artifact | Pass condition |
| --- | --- | --- |
| Checkpoint provenance | `artifacts/m1/checkpoint-provenance.json` | Current checkpoint matches expected config/tokenizer/index hashes, tensor count, shard count, and total payload; historical archive identities are recorded. |
| Runtime identity | `artifacts/m1/runtime-identity.json` | Pinned oMLX head, local patch SHA, repo/archive identities, and model settings are recorded; `mtp_enabled=true` is present for known-good decode comparisons. |
| API invalid-request atomicity | `artifacts/m1/api-atomicity/run-*/result.json` | Invalid requests return errors before backend invocation; a valid request reaches the explicit unconnected backend. |
| Bounded oMLX compatibility comparison | existing/new `artifacts/m0/oracle-compare/direct-vs-server.json` or M1 copy | Direct oMLX and server bridge match generated IDs, decoded text, prompt digest, and usage for bounded prompt(s); not official model correctness. |

## Commands

```sh
python3 tools/run_m1_api_atomicity.py
python3 tools/record_m1_contracts.py
```

Run the atomicity gate first so `artifacts/m1/summary.json` can include the latest API result. The bounded oMLX compatibility artifact remains the existing M0 direct/server comparison unless new evidence requires a fresh connected run.

## Non-goals

- no DwarfStar kernel migration;
- no native runtime optimization;
- no long-context qualification;
- no broad profiling framework;
- no server redesign;
- no checkpoint rewrite, conversion, or quantization;
- no optimization of the rejected naive direct `mlx_lm.generate_step` diagnostic path.

## Decision boundary

After M1 passes, the project may choose the next architecture step with safe provenance, API evidence, and bounded oMLX compatibility evidence. Passing M1 does not qualify official model correctness, long context, or performance superiority.
