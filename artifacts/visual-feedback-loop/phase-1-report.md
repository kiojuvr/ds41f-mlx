# Phase 1 — Inline image classification

**PASS with correction.** The suspected cache-frontier defect is reproduced with the packaged Recipe 0.2 conversion and corrected at request preparation/enqueue. This is source-level acceptance, **not real-model qualification**. Phase 2 was not started.

## Identities and scope

- Branch: `Visual-Feedback-Loop`; starting HEAD: `3e910238f22f9c2ce3d4d31b90e38dc27aae09ef`; initially clean, one worktree. Initial process inspection found one pi process and no serving/test process; no other work was observed.
- Disposable environments provisioned through `ds41f_mlx.mtp_setup`, per `docs/verification.md`: initial `/tmp/ds41f-phase1-tests` (`flash`), final `/tmp/ds41f-phase1-1017-tests` (`flash-1017`), both `--glm-profile native`, with separate fresh build directories. Neither persistent operator installation was modified.
- Final lane: macOS 26.5.2 arm64, Python 3.13.15, Runtime 0.23.0, pytest 9.1.1, Recipe 0.2.0, oMLX 0.7.0, MLX 0.32.2, MLX-LM 0.31.4.dev132+g94cdcae13; numerics `flash-1017-pre-woa-fp8-v1`.
- Recipe authority: `third_party/mtp/normal-local.json`; source base `93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170` plus `docs/recipe-02-native-semantic-boundary.patch`; repaired wheel SHA-256 `11bda3355c2f48b99d04f6a2179b283add3f6d1218f34e4707bd5a29b304ab03`. No donor/upstream checkout substituted. Retained packaged Rust sources were inspected, and conversion executed through that wheel.
- [Environment](phase-1-evidence/environment.json) records source-file, manifest, tokenizer, patch and metadata identities. [Isolated installed admission](phase-1-evidence/installed-admission.json) passed; the pytest session also performs actual isolated admission before source tests. Checkpoint metadata only; no weight loading, full-model execution, Runtime rebuild in an operator installation, or checkpoint/Engram modification.

## Reproduction

[Before](phase-1-evidence/before.json) and [after](phase-1-evidence/after.json) retain the complete wire inputs (small generated 32×32 PNGs), opaque IDs, actual converted roles/tool IDs/source paths, **rendered** source ordering, actual multimodal digests/grids/token spans, frontier and real Scheduler/paired-authority admission results. Cache payloads are controlled Target/ring doubles, not model state.

Common wire structure: user input → assistant `tool_use` blocks → **one wire user message** containing completed `tool_result` blocks, optionally followed by ordinary text. IDs are `Read:Exact-A`, `Read:Exact-B`, `Read:Exact-C`, unchanged throughout. Historical cases put an image in the earlier user input. `U/A/T` below mean converted user/assistant/tool messages, not wire turns.

| Case | Converted roles | Expanded image spans (start, length), rendered order | Original frontier | Expected / corrected |
| --- | --- | --- | ---: | ---: |
| Single new image result | U A T | (267,184) | 0 | 0 |
| New image first of three results | U A T T T | (291,184) | 475 | 0 |
| New image middle of three results | U A T T T | (300,184) | 484 | 0 |
| New image last of three results | U A T T T | (309,184) | 0 | 0 |
| New image result then text-only result | U A T T | (279,184) | 463 | 0 |
| New image result then user text | U A T U | (267,184) | 451 | 0 |
| Historical red image then fresh blue screenshot | U U A T T | (236,184), (464,184) | 648 | 420 |
| Same bytes, distinct historical/new occurrences, trailing text | U U A T T U | (236,184), (464,184) | 648 | 420 |
| Historical image plus three differently colored new results; reversed tool-call order | U U A T T T U | (236,184), (476,184), (668,184), (860,184) | 1044 | 420 |

All these images have actual grid `[39,39]` and expanded length 184. Exact digests are in the JSON; byte equality is **not** the freshness criterion. Image-last cases retain six terminal text tokens (at least the required two); every tested span satisfies the existing P5 terminal-text condition.

Actual Recipe findings:

- One wire user turn expands into several Tool messages; an image-bearing user input can also split into multiple User messages. Consequently, `messages[:-1]` does not identify previous wire turns.
- Official `ImageSource.path` already records wire origin, e.g. `messages.0.content[0]` versus `messages.2.content.0.tool_result`. Several image occurrences inside one result share a path but remain separate ordered sources.
- Converted Tool messages retain return order, but Recipe **renders results in assistant tool-call order**. The reversed-call regression proves that origin flags and multimodal identities must follow rendered sources, not a traversal of converted messages or a byte-keyed map. In the final table row, rendered new sources are result indices 2, 1, 0.

At the affected cache boundary, the original code rejects first/middle/trailing-text fresh images even on a cache miss, and rejects historical-plus-fresh cases even with a compatible retained prefix ending immediately before the fresh image. The corrected code admits all fresh-only cases without a prior pair. Historical-plus-fresh cases admit only the compatible pair: missing, explicitly cleared/evicted, and incompatible historical identity reject before `_start`. These outcomes are recorded separately for each input in the JSON. A real wire-history byte edit is additionally tested through Recipe in the final regressions. Existing tests exercise upstream eviction and target-only payload loss.

The original-method reproduction loads only `_enqueue` from the exact starting Git blob into an in-memory test override; real current Recipe preparation, multimodal expansion, Scheduler admission and paired lookup remain in use. No converted conversation was fabricated to assume the defect. The baseline regression replay produces **13 failures / 5 passes** in the 18 new positive/boundary cases; [log](phase-1-evidence/reproduction-tests.log). This distinguishes the original source concern from an executed conversion and cache-frontier defect.

## Minimum correction and preserved owners

- `serving/server.py::prepare_request`: for ordinary Messages, retain one non-executable historical flag per **rendered Recipe source**, using its official wire-origin path and the latest wire message index. Unknown/out-of-range origins fail closed.
- `RecipePreparedRequest`: one optional ordered `image_is_historical` tuple. No raw Messages parser or extra Recipe conversion was introduced.
- `ProductionMTPBackend::_enqueue`: compute the frontier only over flagged identities, with strict source/identity count alignment. Missing Messages origin metadata rejects. Chat retains its previous classification path.
- No changes to Recipe, multimodal execution, Scheduler cache admission, paired checkpoint authority, historical digest/grid/span/order matching, Target/DSpark ownership, settlement, or replay prohibitions.

Existing owners were traced through `convert_messages` → Recipe conversion/render/encode → `prepare_multimodal` → prepared request → `_enqueue` → `ProductionScheduler._schedule_waiting` → `PairedCheckpointAuthority.acquire` / multimodal prefix validation → `_start`. Existing coverage was inspected before adding tests: native Messages/Chat semantics and exact IDs; multimodal identities/partial spans; metadata-first cached/new encoding and missing/target-only pairs; production paired identity/branch/eviction checks.

## Verification

Final admitted Flash-1017 command (from repository root, no server/environment overrides):

```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/ds41f-phase1-1017-tests/bin/python -m pytest \
  tests/test_recipe02_messages_serving.py tests/test_recipe02_chat_boundary.py \
  tests/test_multimodal_contract.py tests/test_metadata_first_vision.py \
  tests/test_production_mtp.py tests/test_multimodal_restore_ownership.py -q
```

**166 passed**, 53 existing deprecation warnings; [log](phase-1-evidence/regressions.log). New coverage includes first/middle/last results, trailing text/result, rendered ordering versus call/return ordering, multiple images within one result, identical-byte occurrences, real historical wire edits, retained/missing/cleared pairs, missing/misaligned origin metadata, exact IDs and two-terminal-text-token preservation. Existing text-only Messages HTTP/semantic, Chat, paired compatibility/eviction and cached-image-no-reencode tests pass unchanged. Earlier exploratory `flash` lane: original Messages baseline 44 passed; corrected six-file lane 163 passed before the three origin-metadata negative cases were added. Flash-1017 is the final acceptance lane.

All five independent repository integrity checks from `docs/verification.md` passed; their logs and isolated installed admission are in `phase-1-evidence/`. `git diff --check` passed. No skips/xfails or discovery changes were added; deselection is only the explicit original-method reproduction subset, not the final regression lane.

**Limits:** controlled source/cache-boundary tests and actual Recipe conversion, not model-generated screenshots, numerical full-model qualification, real-model paired continuation, HTTP image end-to-end qualification, Thinking ON Vision, a complete visual loop, Files API or Windows acceptance. Phase 0 was not repeated. No push; stop at Phase 1.
