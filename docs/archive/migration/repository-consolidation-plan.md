# Repository consolidation plan

Status: planning only.  This plan does **not** perform bulk source import, runtime implementation change, mass documentation deletion, history rewrite, force push, new numerical Boundary work, or benchmark execution.

## Objective

Final repository goal:

```text
ds41f-mlx
  = canonical implementation
  + canonical runtime architecture
  + canonical correctness authority
  + required legacy implementation assets
  + required qualification/evidence artifacts
  + self-contained provenance
```

The old `kiojuvr/deepseek-v41-flash-mlx` repository should become a read-only historical archive.  After consolidation, ds41f build, tests, correctness gates, documentation, qualification, and development must not require the old repo to exist.

Do **not** merge unrelated histories blindly.  Import selected assets with explicit provenance:

```text
legacy commit SHA
legacy path
SHA256/content identity
evidence classification
import destination
```

## Current SHAs

- Current repo `kiojuvr/ds41f-mlx`: `559201eb8fefb839b3c8a734ecd59eb4e144a150`
- Legacy repo `kiojuvr/deepseek-v41-flash-mlx`: `1b7d0a2c7d33602437dffd44e26a33f39f189661`

Machine-readable plan: `artifacts/repository-consolidation-plan.json`.

## Phase A: legacy import inventory

### Runtime / session state — IMPORT_IMPLEMENTATION

Legacy paths include:

```text
include/dsv41/swa_state.hpp
include/dsv41/global_kv.hpp
include/dsv41/shared_attention.hpp
include/dsv41/engram.hpp
include/dsv41/text_backbone.hpp
src/attention/swa_state.cpp
src/cache/global_kv.cpp
src/attention/shared_attention.cpp
src/model/text_backbone.cpp
```

Purpose:

- SWA/window state
- compressed/global KV state
- Compressor pending state
- Indexer state
- SharedAttention publication
- Ngram/Engram state
- TextBackboneState
- generation continuation/reset/fork

Current proof status: state machinery is C/D legacy evidence and PROVEN_BY_COMPOSITION with ds41f Boundary13/source contracts.  Do not rebuild Boundary13-style state machinery from scratch; rebind this implementation to ds41f official contracts.

Import destination: `src/runtime_state/`, `include/ds41f/runtime_state/`, or equivalent canonical native namespace chosen before import.

Adaptation needed: remove old authority labels, replace old repo paths, add provenance headers/manifests, bind to ds41f semantic contracts.

### Attention — IMPORT_IMPLEMENTATION

Import only final production lineage, not rejected candidates.

Candidate import paths:

```text
include/dsv41/swa_attention.hpp
include/dsv41/swa_layer.hpp
include/dsv41/compressed_layer.hpp
include/dsv41/index_query.hpp
include/dsv41/compressor.hpp
include/dsv41/index_key.hpp
src/attention/swa_attention.cpp
src/attention/swa_layer.cpp
src/attention/compressed_layer.cpp
src/attention/index_query.cpp
src/attention/compressor.cpp
src/attention/index_key.cpp
src/attention/kv_quant.cpp
src/attention/packed_attention_worklist.hpp.in
src/attention/ragged_width_one_qk.hpp.in
src/attention/ragged_tail_qk.hpp.in
src/attention/ragged_tail_av.hpp.in
metal/attention/packed_attention_worklist.metal
metal/attention/ragged_width_one_qk.metal
metal/attention/ragged_tail_qk.metal
metal/attention/ragged_tail_av.metal
metal/attention/steel_attention_header.metal
metal/attention/kv_quant.metal
```

Covers:

- SWA
- compressed producer
- reuse consumer
- candidate source/consumer
- fixed-tile production attention
- request-boundary compact topology
- decode width-one paths

Reject/import-ban:

- old rank-3 padded attention candidate
- DwarfStar-style row-serial candidate
- direct packed MMA candidate
- rectangular packed work-list candidate
- wide attention candidate
- failing pre-`40ddbb6` fixed-tile forms

These remain archive-only rejection evidence via `docs/attention-reject-reaudit-40ddbb6.md`.

### HC / MoE — IMPORT_IMPLEMENTATION

Candidate import paths:

```text
include/dsv41/mhc.hpp
include/dsv41/moe.hpp
include/dsv41/expert_backing.hpp
include/dsv41/moe_pipeline.hpp
src/mhc/reference.cpp
src/mhc/split_sinkhorn.hpp.in
metal/mhc/split_sinkhorn.metal
src/moe/reference.cpp
src/moe/expert_bank.cpp
src/moe/expert_backing.cpp
src/moe/grouped_expert_pipeline.cpp
src/moe/route_select.hpp.in
src/moe/route_reduce.hpp.in
metal/moe/route_select.metal
metal/moe/route_reduce.metal
```

Covers:

- HC
- Gate/routing
- shared expert
- routed experts
- resident expert atlas
- expert-major work lists
- grouped expert pipeline

Adaptation: preserve route-tie policy and backend caveats; separate semantic correctness from optimized grouped implementation.

### Engram — IMPORT_IMPLEMENTATION

Candidate import paths:

```text
include/dsv41/engram.hpp
include/dsv41/engram_mlx.hpp
include/dsv41/engram_layer.hpp
src/engram/index.cpp
src/engram/store.cpp
src/engram/mlx.cpp
src/engram/layer.cpp
src/engram/kernel_source.hpp.in
src/engram/layer_kernels.hpp.in
metal/engram/projection.metal
metal/engram/activation_quant.metal
metal/engram/dequantize.metal
metal/engram/fp8_header.metal
```

Covers:

- SSD-backed Engram
- row lookup/cache
- Ngram state
- Engram@1/@14 integration

Adaptation: bind to ds41f Engram authority; preserve old CPU/oMLX mismatch caveats; import metadata/provenance.

### Full runtime — IMPORT_IMPLEMENTATION

Candidate import paths:

```text
include/dsv41/text_encoder.hpp
include/dsv41/text_decoder.hpp
include/dsv41/text_backbone.hpp
include/dsv41/text_generate.hpp
include/dsv41/generation_loop.hpp
include/dsv41/sampling.hpp
include/dsv41/runtime_bridge.h
src/model/text_encoder.cpp
src/model/text_decoder.cpp
src/model/text_backbone.cpp
src/model/text_generate.cpp
src/model/sampling.cpp
src/runtime/bridge_mlx.cpp
src/runtime/bridge_stub.cpp
src/runtime/bridge_executor.hpp
```

Covers:

- TextEncoder
- TextDecoder
- TextBackbone
- generation loop
- sampling abstraction
- runtime/session ownership
- bridge-relevant portions

Adaptation: replace legacy naming/reference terminology; integrate with ds41f API; remove dependence on old repo and eventually on oMLX worker for canonical runtime.

### Tests — IMPORT_TEST

Candidate paths:

```text
tests/attention/test_swa.cpp
tests/attention/test_attention.cpp
tests/attention/test_block.cpp
tests/attention/test_text_encoder.cpp
tests/attention/test_text_decoder.cpp
tests/attention/test_text_backbone.cpp
tests/attention/test_text_generate.cpp
tests/attention/test_generation_loop.cpp
tests/runtime/test_bridge.cpp
tests/engram/test_engram.cpp
```

Adaptation: remove old absolute paths, convert expected authorities to local imported artifacts/ds41f contracts, mark checkpoint-heavy tests as optional gates.

### Evidence — IMPORT_EVIDENCE

Reviewed evidence to physically preserve under ds41f, e.g. `artifacts/legacy-import/evidence/...`:

```text
artifacts/swa/layer-check.json
artifacts/global-kv/attention-check.json
artifacts/global-kv/consumer-check.json
artifacts/global-kv/consumer-range-check.json
artifacts/block/reviewed-result.json
artifacts/text-backbone/reviewed-prefill-129-route-ties-20260915.json
artifacts/attention/fixed-tile-backbone-20260919-203130-89172/test.log
artifacts/generation-lifecycle/reviewed-temperature-result-20260915.json
```

Representative legacy SHA256 values already captured in the JSON plan include:

```text
08af07f49d27accbaafe4ff15897e1eeded1ae52caf11de017ebea0eec4b734e  artifacts/swa/layer-check.json
cb61dbde67b53813b5afe33d01b94c1c2c0e5b9c726252e1fe0d5319e393fb40  artifacts/global-kv/attention-check.json
388c72be133a17e5c5eb04257aafbaa367d7ec7c0154bcb86e79c058fa590e80  artifacts/global-kv/consumer-check.json
587285c07d367393d7a2031fd0294fc2f4570340ab0ee691a1b3899ff1dce94a  artifacts/global-kv/consumer-range-check.json
c904c54a54c0fd0b0992ec826c60cd39a818cc5a6f03568d2c2d3bee6d11ba92  artifacts/block/reviewed-result.json
6fac8507c4a3c85ee3706013359a8c68d8a66739b959647e22eb25c7c92478b2  artifacts/text-backbone/reviewed-prefill-129-route-ties-20260915.json
2be4a0be98818b953ca0212d7809b9a22a6ae8f41b1340a930a965927c954963  artifacts/attention/fixed-tile-backbone-20260919-203130-89172/test.log
e4d7bd3b9edd1ff3c35135dbabaebf3f92c702ecf7cb43ff55cee78b3e4789df  artifacts/generation-lifecycle/reviewed-temperature-result-20260915.json
```

### Documented knowledge — IMPORT_DOCUMENTED_KNOWLEDGE

Absorb these into canonical docs rather than relying on old repo navigation:

```text
docs/swa-reference.md
docs/compressed-attention.md
docs/mhc.md
docs/moe-block.md
docs/text-decoder-validation.md
docs/attention-reject-reaudit-40ddbb6.md
docs/engram-forward.md
docs/sampling-reference.md
```

### API bridge — REIMPLEMENT_AS_ADAPTER

Do not blindly import old server topology.  Reuse request validation, lifecycle, cancel/stop, and bridge evidence, but implement the canonical ds41f adapter after the native runtime shape is fixed.

### Rejected material — REJECT / KEEP_ARCHIVE_ONLY

Do not import as implementation:

- oMLX-as-official-oracle artifacts
- DwarfStar-output-as-official-oracle artifacts
- superseded M0/M1/M2 correctness evidence
- rejected optimized attention candidates
- transient diagnostics and failed instrumentation artifacts

## Phase B: self-containment audit

A first pass found 16 ds41f files with explicit old-repo path/archive references.  They should be eliminated from production/correctness/canonical docs or replaced by local imported provenance.

Representative references:

```text
README.md
docs/correctness.md
docs/correctness-authority-audit.md
docs/m0-plan.md
docs/m1-plan.md
docs/m1-closeout.md
docs/boundary7-closeout.md
docs/boundary8-closeout.md
docs/boundary11-closeout.md
docs/legacy-evidence-boundary-reconciliation.md
artifacts/m0/source-identity.json
artifacts/m1/checkpoint-provenance.json
artifacts/m1/runtime-identity.json
artifacts/boundary7-closeout.json
artifacts/boundary8-closeout.json
artifacts/legacy-evidence-boundary-reconciliation.json
```

Classification:

- README / canonical docs: replace with local canonical content.
- M0/M1/M2 docs: archive/remove later after content absorption.
- Boundary closeouts: archive after proof graph absorbs evidence.
- Artifacts with old absolute paths: replace with local imported evidence/provenance manifests, or mark archive-only historical artifacts.

Goal: 0 old-repo references in runtime source, tests, correctness gates, and canonical docs.

## Phase C: canonical documentation design

Target structure:

```text
README.md
docs/
  architecture.md
  correctness.md
  session-state.md
  attention.md
  moe-hc.md
  engram.md
  generation.md
  api.md
  performance.md
  provenance.md
  qualification.md
```

README should describe only:

```text
what this runtime is
supported hardware
supported checkpoint
current capabilities
runtime architecture summary
build/run
API
correctness status
performance status
known limitations
canonical docs links
```

No “M0/M1/M2”, “next Boundary”, “oMLX reproduction objective”, or old qualification oracle narrative.

### Current-document stale narrative to rewrite

`README.md` currently contains:

- M0 goal as current project purpose.
- M0/M0.5/M1/M2 chronological progress.
- qualification oracle = `/Volumes/SDXC-512/deepseek-v41-flash-mlx`.
- oMLX reproduction as current objective.
- “next semantic boundary” language.
- DwarfStar/M2 repair narrative as current navigation.

`docs/correctness.md` currently contains:

- M0 scope as current section.
- legacy qualification archive as reference hierarchy item.
- pointers to contamination repair narrative as primary navigation.

`docs/runtime-boundary.md` currently contains:

- M0 runtime boundary and temporary API server purpose.
- explicit statement that production architecture is deferred.
- oMLX bridge as current purpose.

`docs/provenance.md` currently contains:

- required oMLX upstream/local patch identity as normal run identity.
- qualification archive commit identity as normal run identity.
- M0/M1 artifact lists as canonical provenance.

All of these are rewrite/archive targets, not final canonical specification.

## Phase D: evidence preservation

Add machine-readable manifests:

```text
artifacts/provenance/legacy-import.json
artifacts/provenance/proof-graph.json
```

Preserve:

- official semantic fixtures
- native validation artifacts
- imported reviewed legacy artifacts
- legacy import provenance
- proof composition graph
- qualification results

Copy old-repo-only valid artifacts into ds41f with:

```text
original repo
original commit
original path
original SHA256
imported path
classification
```

Do not rewrite legacy artifacts; copy immutably and classify.

## Phase E: remove duplicate/transient material later

Only after canonical docs and proof graph absorb content, consider archive/removal of:

```text
M0/M1/M2 planning docs
Boundary-specific planning docs
Boundary-specific closeout docs
superseded semantic plans
temporary repair plans
historical contamination repair narrative
duplicate fixture explanations
```

Removal condition:

```text
content fully absorbed into canonical docs == yes
evidence preserved elsewhere == yes
```

No mass deletion in this planning phase.

## Phase F: git history cleanup design

Do not rewrite history before import/consolidation is complete.

Options:

1. Keep existing history.
2. Interactive squash/rebase.
3. Create clean canonical-history branch.
4. Rebuild compact history from semantic milestones.

Recommendation: keep history during consolidation.  After the repository is self-contained and verified, if force-push is acceptable for this personal repo, create a clean canonical-history branch with semantic commits such as:

```text
1. Bootstrap checkpoint/provenance/runtime scaffold
2. Establish official semantic authority and validation primitives
3. Establish connected model semantics and state contracts
4. Reconcile and import proven legacy implementation/evidence
5. Integrate native runtime
6. Consolidate canonical documentation and qualification
```

The manifests must preserve imported legacy identities regardless of history rewrite.

## Self-containment exit criteria

Minimum final exit criteria:

```text
0 runtime source references to old repo
0 test dependencies on old repo
0 correctness gate dependencies on old repo
0 canonical-doc dependencies on old repo

all required legacy implementation copied/adapted locally
all required legacy evidence copied locally
all imported evidence has provenance

canonical docs describe current system without chronology
README contains no development-phase narrative
Boundary numbers unnecessary for understanding runtime

old repo can be unavailable and ds41f still builds/tests/runs/audits
```

## Ordered execution plan

1. Freeze/tag legacy source identity.
2. Build import/provenance manifest with per-file SHA256.
3. Import required evidence/tests.
4. Import/adapt proven runtime implementation.
5. Connect imported implementation to ds41f semantic contracts.
6. Eliminate old-repo runtime/test references.
7. Rewrite canonical docs from current state.
8. Archive/remove transient Boundary/M0-M2 docs only after absorption.
9. Verify self-contained build/test/audit.
10. Only then consider git history rewrite.
11. Freeze old repository as archive.

## First concrete implementation task

Create `artifacts/provenance/legacy-import.json` with exact per-file SHA256 for selected evidence/docs/source files, then copy only reviewed evidence artifacts into `artifacts/legacy-import/evidence/` without changing runtime code.
