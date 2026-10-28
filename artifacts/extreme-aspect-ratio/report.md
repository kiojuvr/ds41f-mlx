# Production bugfix: extreme-aspect-ratio image rejection

**PASS — 2026-10-28.** Independent post-completion correction, branched from
`ca9a813147d7dff977f3ec2ad942f2cb876e9882` as
`fix/extreme-aspect-ratio-images`. No Multimodal-Runtime completion record or
C1–C9 disposition changed. Implementation/wheel source:
`1059b8a727416aba07cb783bf278ee661b193b61`; subsequent changes are tests/documentation/evidence only.

## Failure and upstream comparison

The supplied Windows observation was successful original/face/torso/dessert reads,
then `crop_bg.png` rejected with
`invalid image: image aspect ratio outside supported 1:2..2:1 envelope` /
`INVALID_REQUEST`. The original bytes/dimensions are unavailable; no Windows
execution is claimed. Deterministic **1600×200 PNG** reproduces the exact inner
rejection against the completed baseline ([receipt](baseline-reproduction.json)).

Failure path: Harness `read_image` acquires an attachment; `prepareImages` obtains
its request variant and `prepareFileIds` resolves uploads, or the adapter uses
bounded inline fallback. Runtime Files upload calls `validate_encoded_image`;
resolved native Messages preserve ordered sources, Recipe renders placeholders,
then `server.prepare_request` calls `prepare_multimodal`, whose same validator
raised the reported `invalid image:` error, translated to HTTP 400. The error
text establishes an admission failure, **not a malformed tool call**. Without the
original wire trace the exact Windows Files-versus-inline leg is not established.

Official sources inspected:

- DeepInfer **`553741557a57df44a46fb3ab72b6aafeaf945337`**:
  `deepinfer/ve/processor/image_processor.py`, `ve/multimodal_processor.py`,
  `model_executor/models/deepseek_vl.py`. Also compared image geometry at the
  repository's earlier reference **`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`**.
  Both use arbitrary-aspect planning, patch-aligned nonzero dimensions, explicit
  single-row/single-column solver branches, and
  `tokens = n_llm_h*(n_llm_w+1)+2`. Request expansion uses the same layout as
  processing; fresh patch lengths are products of ViT grid dimensions. Fully
  covered cached image spans are skipped, not replayed.
- Official Harness **0.2.1-rc.2**, installed `dsh-tool-fs` and `dsh-llm-deepseek`
  image acquisition/Files-to-Messages preparation. Bundle hashes and paths:
  [identities](upstream-identities.json). No fixed 2:1 requirement in this path.
- Official Recipe **0.2.0**, **`93b0fd229cc9ffa6b1a8c63d42b0bf741bbec170`**:
  shared `protocol/image.rs`, Messages request conversion, image limits.
  Source validation and transport are distinct from model geometry. Recipe
  itself rejects unresolved File IDs; Runtime's existing acquisition resolves
  these before Recipe. No dependency modification or new dispatcher is needed.

Flash-1017 config has patch size **14**, downsample **3**, minimum pixels
**295936**, processor token budget **1024**, `max_wh_ratio: null`. Runtime
`ModelConfig` preserves those settings. Its existing planner agrees exactly
with both examined upstream geometry revisions for the 16 recorded cases
([comparison](upstream-geometry.json), [reproducer](compare-layout.py)); no
CUDA/Torch imports were used. MLX Aligner pads each patch-grid dimension modulo
3 and uses ceiling division, including one-row/one-column cases.

Pillow's contain/pad can round a very thin side to zero. Upstream catches this
and stretches to the planned target. The existing MLX processor already handles
it with a minimum-one-pixel contained resize and centered gray padding. Direct
packing checks of 100000×1 and 1×100000 complete with nonzero dimensions; these
shapes plan **1024 tokens**, so the unchanged Runtime **1014-token** bound rejects
them before execution. No newly exposed geometry defect justified a processor
change. No claim of pixel-identical upstream extreme fallback is made.

## Minimal correction and focused checks

Removed only the two-line fixed-ratio guard in `runtime/multimodal.py`.
Reuse existing planning/packing; no crop, square conversion, new image processor,
ceiling increase, kernel/Scheduler/cache change, or Harness modification.
Byte identity/SHA256, MIME/magic, complete verify/decode, single-frame checks,
all resource admission and paired-history protections remain unchanged.
CURRENT capability/API/Files docs now describe processor-planned aspect ratios.

**96 tests passed** ([log](tests.log)), using the existing disposable installed
Flash-1017 test closure `/tmp/ds41f-step4ar1-tests` with source-tree tests:
`test_multimodal_contract`, `test_image_files`, `test_harness_image_capacity`,
`test_cached_image_admission`, `test_multimodal_restore_ownership`.

- Square, ordinary landscape/portrait, 4:1, 8:1, 1:4, 1:8; 1×1,
  128×1/1×128, 512×1/1×512, 4096×1 admit. Planned dimensions, actual patch arrays,
  delimiters/token counts, immutable bytes/digests and finiteness agree.
- True one-row/one-column aligned layouts covered. No zero grid/invalid reshape.
- 1×4096 and 100000×1/1×100000 fail the actual token bound; 2049×2048 fails
  per-image pixels. Actual per-image/aggregate bytes, aggregate pixels and fresh
  patches still reject before MLX/patch allocation. Existing cached-history,
  hash/prefix, restore and failure protections pass.
- Wide/tall/thin inline and File-ID nested tool results produce identical
  canonical tokens/image identities, retaining original opaque tool IDs;
  existing repeated-image ordering/history tests pass.

Initial development `.venv` test invocation lacked an installed admitted package;
reran in the existing disposable test closure, without bypassing admission.
An inspect without explicit checkpoint was correctly refused; all recorded
successful inspect runs explicitly select the unchanged operational checkpoint.

## Native acceptance and Production promotion

Candidate wheel installed non-editably into that existing disposable closure,
then one focused native Messages regression ran on port 18057. Production ran
the same affected-workflow acceptance after promotion on port 8000. **Both PASS**:
[script](native-check.py), [candidate](candidate/native.json),
[Production](production/native.json), and respective Runtime logs.

Each run contains four bounded requests, Thinking OFF/native GLM/paired MTP:

1. Upload/read a genuine **1600×200 (8:1)** PNG through Files. Model: “From left
   to right, the image shows a red rectangle, a blue circle, and a green triangle.”
2. Model requests `read_image(ordinary.png)`.
3. Exactly one requested read returns a 384×384 PNG inline under the original
   tool ID. Model correctly identifies the centered red circle on white.
4. Compatible paired continuation answers “Red.” without re-encoding history.

Both runs restore **C=0,507,565,776** and settle/publish at
**507,565,776,796**. Every settlement reports `published=True`, no cancellation
or error, **replay=0/repack=0**. Health ends active=queued=0. This is image-grounded
native execution, not Windows Harness qualification or another C8 campaign.

Before mutation, verified target `kioju:staff`, installed source **977c57e3**,
RECORD hashes, successful ordinary admission and no active Runtime/listener/work.
Candidate was gracefully retired before Production replacement. Retained rollback:

`/Volumes/SDXC-512/ds41f-1017-aspect-rollback-1059b8a/`

All **136 RECORD-owned files** archived and independently rehashed against the
pre-install inventory. Archive `previous-package.tar.gz` SHA256:
**`71f3e39e34db49cd61de12db00259010524fe7ef5a19d9ace19b1bf548cd37af`**.
The directory retains previous/promoted package inventories and the promoted wheel.

Used existing `tools/build_execution_package.py --checkpoint-variant flash-1017`,
then `uv pip install --python /Volumes/SDXC-512/ds41f-1017/bin/python --no-deps
--reinstall <retained wheel>`. No provisioner, persistent second venv or dependency
update. Wheel SHA256:
**`e65023e07d4064eb83c6908d9699a31464b86b9bc90b083595012f70e260abc9`**.
Version remains **0.23.0**; source identity, not version, identifies the correction.

All **124 package wheel members** match installed bytes. Isolated imports resolve
inside `/Volumes/SDXC-512/ds41f-1017/lib/python3.13/site-packages/`.
Distribution names/versions/origins and every manifest field except provenance
are unchanged ([verification](installed-verification.json)). Checkpoint remains
`/Volumes/KIOXIA-PRO-2/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, numerical profile
`flash-1017-pre-woa-fp8-v1`, execution identity
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Previous/promoted/final ordinary installed inspection PASS; model readiness PASS.
Production acceptance process gracefully retired, Scheduler/application shutdown
complete, PID absent and port released ([receipt](shutdown.json)).

Rollback if required, **only after graceful stop/no work/port release**: verify
archive SHA above, `uv pip uninstall --python <env>/bin/python ds41f-mlx`, then
`tar -xpf <save>/previous-package.tar.gz -C <env>`. Rehash every restored member
against `previous-package.json` and run installed ordinary inspect with the
explicit checkpoint before restart. Archive recovery is independently verified;
no rollback over the accepted installation was necessary.

## Remaining scope

Native acceptance qualifies this 8:1 fixture and ordinary tool continuation only;
other recorded shapes have model-free admission/packing evidence, not new
numerical or visual-quality qualification. Token/pixel/byte/patch/context/cache
limits still independently bind; “arbitrary aspect planner” is not unconditional
support for every dimension or image. Original Windows content remains untested.
No historical completion edits, endurance campaign, push, merge or release tag.
