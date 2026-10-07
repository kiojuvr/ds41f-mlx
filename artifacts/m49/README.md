# M49 — stopped promotion attempt

**Decision: NO_PROMOTION / BLOCKED.** This is not a completed milestone or an
integrated-MTP qualification receipt. Runtime code, selectors and pins remain
unchanged at master `5e784d9e83a85c497cd192f87184e9b01a1343eb`.

See `../../docs/milestone-49-runtime-promotion-assessment.md` for the transaction,
state-production, admission and application ownership audit and required design.

These are historical runs at base `5e784d9`, not fresh current-tree qualification.
M51/M52 subsequently implemented the target transaction/generation boundaries;
M53 identifies the remaining proposal-side gap. Reproduce these historical claims
on the recorded base, not by interpreting the original probe as an M53 gate.

## Real-checkpoint blocker evidence

`ownership.json` / `ownership.log`: official 48-shard checkpoint loaded through
`DeepSeekRecipeRuntimeBackend.load` and normal M47 first-party admission; no
substituted model, policy bypass, donor loading or resource pin change.

- First-party MTP loading, canonical generation MTP configuration, four-row
  verification and verification state all reject at the current contract.
- Loaded first-party model has none of the six native MTP configuration/context
  ownership hooks tested by the probe.
- Actual dense/P5/OFF decode: **4,096 positions + 32 tokens**, idle frontier
  **4,128**, all 40 layers aligned, original cache list preserved, P5 once.
- Median **19.916 tok/s** (32-step control, not a speedup experiment); actual load
  and admission **336.346 s**.
- All **280 physical slot** shape/dtype/raw-byte hashes are equal before/after
  rejected interface probes. Replay **0**, full-cache repack **0**.
- Resource retirement rejects further execution with `execution resource lease retired`.

Unsupported transaction probes call validation only. Calling `execute` on an
unsupported input would burn the live lease by design; the tool does not weaken
that behavior to keep its control state. These observations prove the current
integration boundary, not an impossibility theorem for future implementation.

## Regression

`ownership-tests.log`: **96 tests PASS** across M44 generation, M45 target
transaction, M46 completion, M47 admission, M48 model execution, M29 MTP lifecycle
and MTP startup-resource lifetime.

`r1-off.json`: fresh **full R1 CONFORMANT / PASS, 24 gates**, existing admitted OFF
interpreter. Covers real standard local HTTP protocols, tool/result continuation,
SSE and persistence/restore as defined by the unchanged R1 fixture.

`r1-mtp.json`: fresh **full R1 CONFORMANT / PASS, 24 gates**, existing admitted
MTP interpreter. The real HTTP fixture passes **18 cases**, including retained
multi-turn/SSE/tool results, byte loss, partial-tool refusal, cancellation,
re-entry and retirement. Every settled observation has aligned idle frontiers,
zero replay/repack and zero settlement proposals/verify cycles.

Uninstrumented candidate-only performance (not canonical promotion evidence):

| Workflow | Prefix/handoff s | Decode tok/s | Accepted / considered drafts |
|---|---:|---:|---:|
| One weather call | 1.653 | 39.949 | 24 / 24 |
| Two weather calls | 1.615 | 38.458 | 46 / 47 |
| First stored-result continuation | 0.337 | 37.812 | 6 / 6 |
| Second stored-result continuation | 0.433 | 30.258 | 26 / 32 |

`summary.json` aggregates the final receipts. Draft counts establish real MTP
activity and nonaccepted drafts; they are not a new physical rollback observer
or a token-identical OFF/MTP speedup claim.
`r1-mtp-missing-checkpoint.json` is an **excluded failed launch**, not model
qualification: this attempt omitted the candidate's mandatory explicit
`DS41F_CHECKPOINT` environment variable and failed at pre-model identity admission.
The corrected run uses the explicit official checkpoint; no identity rule was
relaxed. OFF uses its normally admitted default checkpoint path.

Each lane uses its own admitted interpreter, sequentially with one full model
resident. R1 reference SHA256 is unchanged:
`45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
These are **unchanged baseline** regressions, not evidence that MTP now works via
standard Chat/Web. No expanded MTP context, generic tool declaration, Vision,
MTP persistence or ordinary application MTP path is claimed.

## Reproduction

```sh
.venv/bin/python -m tools.probe_m49_ownership --output artifacts/m49/ownership.json
.venv/bin/python -m pytest -q tests/test_m44_target_generation.py \
  tests/test_m45_target_forward.py tests/test_m46_completion.py \
  tests/test_m47_resource_admission.py tests/test_m48_model_execution.py \
  tests/test_m29_mtp_lifecycle.py tests/test_mtp_resources.py
.venv/bin/python -m ds41f_mlx.reference --profile standard-off --real-model \
  --output artifacts/m49/r1-off.json
DS41F_CHECKPOINT=/path/to/official/DeepSeek-V4.1-Flash \
  "$HOME/.venvs/ds41f-mtp-investigation/bin/python" -m ds41f_mlx.reference \
  --profile mtp-singleton-v1 --real-model --output artifacts/m49/r1-mtp.json
```

No source sealing is necessary for this audit-only change: admitted runtime
sources and dependency artifacts are unchanged. The new diagnostic tool and
assessment documentation do not select a production executor. No release work or
commit was performed during the original assessment. This record is now archived
in git with the user's authorization, without changing its NO_PROMOTION decision.
Resolve the archival commit with `git log -1 --format=%H -- artifacts/m49/README.md`;
`summary.json`'s original `commit: null` is retained as historical metadata.
The archive includes top-level receipts/logs, not transient gate directories or
generated persistence/cache payloads. Paths into those original local run
directories in R1 receipts are historical locators, not bundled artifacts.
The caller's existing `artifacts/web-application/chrome.log` modification is
unrelated and excluded.
