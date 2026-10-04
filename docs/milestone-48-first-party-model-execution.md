# M48 — first-party standard-OFF model execution

## Production implementation audit and priority decision

Stage 1 is complete for the supported text, standard-OFF path. Inspection of the
actual path, rather than the latest documented frontier, found no substantive
missing production implementation:

- The standard request policy selects the recipe backend, which prepares canonical
  tokens, rejects unsupported image/model/options, and owns single-flight inference.
- `DeepSeekRecipeRuntimeBackend.load` admits/binds the model lifetime. Its infer
  path uses dense P0–P7 prefill, the sole committed packed cache list, and exactly
  one held-out terminal token at P5. It neither selects bounded reference math nor
  replays/repackages the prompt.
- M44 generation, M45 all-layer forward/commit/failure, M46 subordinate state
  production and M47 resource permission/lifetime already implement the active
  production execution/lifecycle. Those ownership boundaries were not reopened.
- Idle continuation, SSD-backed Engram, persistence/restore, tool/session boundaries,
  cancellation/cleanup and serving request fencing are implemented. Their breadth
  and long-session performance are later-stage work, not missing implementation.

Release/distribution/provenance enhancements, startup-verification I/O optimization
and architectural renaming were explicitly not selected as production gaps. The
next substantive work was therefore Stage 2, not another Stage 1 micro-milestone.

## Boundary implemented

`ds41f_mlx/model_execution/` now owns the end-to-end model execution layer used
by standard-OFF: immutable official-checkpoint mapping and streamed construction,
model/config/parameter inventory, quantized projections and dispatch, activation
rounding/packing, CSA2 attention and index math, hyper-connections, MoE routing and
combination, head projection, Engram normalization/hashing, SSD tensor mapping/gather
and prefetch lifecycle, and the seven-slot cache specialization.

The existing dense executor, suffix decomposition, P5 handoff, M44 generation,
M45 transaction and M46 producer remain the orchestration foundation. Admission
now verifies/binds first-party numerical module implementations instead of donor
model modules, with the checkpoint, tokenizer pair, native profiles, dispatch and
backend-local fidelity policy unchanged. Decode receives stable first-party math
handles from the same M47 capability. Restore creates our packed-cache class;
existing artifact schema, physical tensors and compression metadata are unchanged.

`OmlxRuntime` retains its historical caller API but selects the first-party loader
for the actual standard-OFF configuration. Failure does not retry the donor.
Separate MTP/diagnostic loading remains explicitly separate and cannot acquire an
OFF capability. The suffix adapter still supports donor layers for that separate
path/oracle; admitted production models are exact first-party classes, and no
production fallback/selector can switch their numerical implementation.

This is not a series of wrappers around donor model methods. Qualified donor
algorithms are derived into first-party implementations and local model imports.
Ambient MTP dispatch, donor cache registration, offload/MTP loading branches and
export/shard-writing machinery are not part of the new production layer. Official
vision parameter construction remains necessary to account for the checkpoint's
complete inventory; image serving is still unsupported.

The existing native C++ reference model and `official_model_math` validation seam
were reviewed but not substituted: they are bounded/reference/qualification
implementations, not the selected all-40-layer packed MLX production foundation.
Reusing dense/suffix/producer code and qualified MLX arithmetic avoids creating a
second executable state representation or changing checkpoint math.

## What remains external

See the [model layer dependency/attribution inventory](../ds41f_mlx/model_execution/README.md).

MLX/Metal/OS frameworks; generic mlx-lm cache/module/activation/sampling primitives;
the admitted GLM native bundle and its stateless acceleration/portable primitive
implementations; NAX detection, generic fused gate/up and route sorting; NumPy,
tokenizers/Transformers/PIL and the mlx-vlm result container; official recipe/native
protocol mechanisms; immutable official checkpoint bytes; and OS SSD I/O/thread
mechanisms remain ordinary lower-level dependencies. Our model/projection/attention
implementation controls their invocation and dispatch. No donor numerical/model
executor owns the standard-OFF path.

Donor `tool_parser`/`encoding` still supply legacy tokenizer/processor metadata and
optional diagnostic prompt formatting, not production model math; production
protocol encoding/parsing remains recipe-owned. The donor MTP/diagnostic model
path and numerical oracle remain external intentionally, outside this scoped
production boundary. Their continued existence is not unfinished OFF ownership.
No genuine unfinished numerical/model-execution ownership was identified on the
supported production path. This does not claim source independence from MLX or
all-mode replacement of oMLX.

## Evidence

`artifacts/m48/donor.json` and `owned.json` are matched runs against the official
checkpoint, in separate processes with fresh independent dense prefills. The
oracle uses frozen M47 admission/lifetime/persistence code from the established
baseline commit, not a production selector. The first-party process rejects
imports **and calls** into displaced donor numerical/config/model/loading/storage
modules and the donor specialized switch layer throughout load and execution.

- Actual `DeepSeekRecipeRuntimeBackend.load` and `infer` execute successfully,
  with 16 identical serving tokens, one P5 handoff, no replay/repack and cleanup.
- At **4K and 32K contexts / 64 generated tokens**, tokens and every **40 × 7**
  final slot match the donor exactly (shape, dtype and raw-byte SHA256, including
  BF16 tails without casting). The same list survives P5 generation and idle
  extraction. Independent oracle arrays/digests are evidence only, never a
  production shadow cache or an execution input.
- Persistence/restore retains every slot exactly and reconstructs first-party
  cache classes. A subsequent real P6 suffix append and 8-token generation retains
  the restored list, zero replay/repack, identical tokens, and exact final 40 × 7
  slots at both contexts. SSD-backed Engram stays enabled throughout.
- The first-party profile records **182 implementation functions / 1,755,700 calls**,
  including checkpoint construction, model/block/projection/MoE/attention math,
  packing, HC/head operations, hashing and SSD store/prefetch execution. Production
  model and cache types are first-party; displaced donor imports/calls are forbidden,
  not merely counted as absent after the fact. Load-failure unit injection proves
  there is no retry/fallback to the donor loader.
- Affected numerical/admission/prefill/generation/state/SSD/lifecycle tests:
  **161 passed, 32 subtests**. Additional continuation/client/transport/recovery/
  admission/configuration tests: **87 passed**. Nonzero-weight tiny-block and
  MXFP4/MXFP8/affine projection/packing comparisons also match the donor exactly.
  M47 substitution, stale-bytecode, lease-retirement and resource-fault tests now
  target the first-party implementation; SSD future failure/drain tests use its
  actual store/prefetch classes.
- Incidental profiled median decode throughput: donor → first-party **19.192 →
  18.574 tok/s at 4K**, **18.987 → 18.312 at 32K**. Both remain above the existing
  practical floor. The first-party observer additionally counts its implementation
  calls; this is not an uninstrumented performance comparison or speedup claim.
  No optimization campaign followed.

An initial oracle attempt stopped in the evidence collector's NumPy BF16 buffer
conversion, not in model execution. Raw-byte views fixed the collector and both
final runs passed. A broader historical M20 donor-generation test attempt had
four tests blocked by the existing environment's missing `jsonschema`; no package
installation/environment work was undertaken, and those are not claimed passing.
The affected production/lifecycle suites above pass. Post-run EOF whitespace
cleanup in two model files was verified to leave compiled executable code identical;
source pins were updated and final admission/numerical/SSD checks passed (50 tests).
A loader API docstring was also corrected. R1 was not rerun.

## Checkpoint decision

Stage 1 can be considered complete for the supported production scope. Stage 2's
standard-OFF numerical/model-execution boundary is now implemented and established
by the real-path evidence above. The next task is ready for Stage 3
(performance/long-session/operational work), rather than
another tiny dependency-removal step. R1 re-verification remains later work; no
full R1 qualification or release claim is made here. No `ds41f-runtime` promotion,
packaging, distribution or clean-room work was performed.
