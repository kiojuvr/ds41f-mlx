# M43 — Deterministic Release Repository Extraction

## Decision

**RELEASE_REPOSITORY_EXTRACTION — PASS**, for the existing qualified scope.
This is a source-delivery/extraction milestone, **not feature completeness**.
R1 remains `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
Official semantics remain higher authority than implementation qualification, R1
and the release projection. No R2 was created; R1 material/verifier is unchanged.

Canonical command, from a clean authoritative HEAD:

```sh
python3 tools/promote_release.py --source HEAD --reference R1 --output ../ds41f-runtime
```

`release/runtime-surface.json` is the explicit allowlist/ownership definition.
Its SHA-256 is
`3e6e98ffaef35407b0547f946d276221a157c7e0bab7b03da0ea40d9d13cf9f3`.
The qualified complete payload identity is
`adb4b5b02e8870d798c39cf30ca72f56726482599cc4d090f2da9f26b6de0c87`.
Semantic contract SHA-256:
`8856d0b29fafb49ef337af095cfe9593bbe43eeb3cbf0522d31433f9200cdac6`.
Promotion implementation SHA-256:
`b4b1ab1b0ac944164bc7202d27732941328762a8f1d9a5f74d604fb0ebff1f1f`.

The final fresh setup/model qualification input was source commit
`9d69a6e86f8125c0232c116b4ab1b5e60545b1d7`. Subsequent evidence/documentation commits
can inherit it only if the **entire projected payload**, including documentation
and surface definition, is exactly identical. The promotion envelope records the
actual source commit and the earlier tested commit separately. See
`artifacts/m43/determinism.json` for the final projection source/envelope/full-file
identities, and `release/promotion-qualification.json` for the committed PASS record.

## Deterministic authority and manifest

Promotion reads **committed Git blobs**, not an ambient filesystem copy. Declared
source must be the clean current HEAD. Each required source must exist as a regular
tracked blob with an approved Git mode. Closed owned source roots require explicit
include/exclude classification; new unclassified files reject. Reference manifest,
all bound reference material/verifier, and dependency source archives are checked
before destination creation. The output must be independently empty. No generated
binary is copied or compiled during promotion.

`release/promotion.json`, schema **ds41f.promotion.v1**, binds source repository and
commit, reference name/identity, contract schema/identity, surface/implementation
identities, every destination's content/size/mode/origin, source delivery/locks,
checkpoint/protocol/tokenizer compatibility, profiles/exclusions, M41/M42 receipts,
reference-scoped inheritance and the committed release qualification.

Payload identity is SHA-256 of canonical sorted JSON file entries. It excludes
only the **single derived promotion envelope**, avoiding recursive self-hashing;
this is declared by the schema from the outset. Determinism proof separately
compares the envelope byte-for-byte and hashes **all 274 delivered files**, including
it. Nothing changed is retrospectively excluded. Runtime-generated `.git`, caches,
artifacts, Cargo target and editable package metadata are declared generated data,
not projected source. Setup/build scratch is outside the source tree.

The PASS receipt must be committed in the authoritative source and bind the exact
payload/reference plus a tested ancestor commit. An optional receipt argument must
match that committed blob; arbitrary external PASS files cannot promote a release.
Without a committed receipt, output is explicitly NOT_RELEASE_QUALIFIED. A stale
receipt fails closed. This is a local provenance model, not a cryptographic signature
or protection against a malicious operator falsifying the authoritative repository.

## Surface and exclusions

273 allowlisted files plus the promotion envelope:

| Category | Files |
|---|---:|
| Current Python/native-in-package/runtime/operator resources | 67 |
| Native headers/source/Metal/build and structural source tests | 147 |
| Immutable R1 verification/material/private OFF export | 35 |
| Attributed MTP/recipe source exports, locks and licenses | 7 |
| Rust OFF boundary/acceptance/build source | 4 |
| Current runtime documentation | 4 |
| Release compatibility/surface/inherited compact receipts | 4 |
| Root build/package definitions, README and NOTICE | 5 |
| Derived promotion envelope | 1 |

Native source tests are retained to build/qualify the delivered native foundation,
not to imply a new production selector. `m2_state_publication.py` remains because
`native_prefill.py` imports it; DwarfStar-prefixed **runtime** ownership modules
remain because they own production interfaces. Ownership, not filename aesthetics,
determines selection. Ten known development source files are explicitly excluded
from otherwise closed roots; historical tools/tests/docs/artifact directories are
outside the allowlist entirely.

Excluded: CUDA oracle, donor architecture/comparison machinery, obsolete milestone
harnesses, old experiments and diagnostic traces, historical M1–M42 documentation
campaigns/archives except compact M41/M42 inherited receipts, whole-repository tests
and tools, M23 bundle builder, Git metadata, build products, checkpoint weights and
KV artifacts. Attributed licenses/exported sources are included, not replaced with
upstream names erased. Extraction grants no new public redistribution license for
project-authored code and performs no remote publication.

## Closing delivery debt without architecture replacement

M22/M23 version/resource lookup, Rust source acceptance and release compatibility
metadata remain useful. Their configured donor-checkout/prebuilt-bundle model is
historical, not mandatory or canonical for repository extraction. The builder is
preserved in ds41f-mlx but omitted from runtime. No historical result is rewritten.

OFF setup installs the existing immutable R1 attributed base oMLX export. MTP setup
installs the existing attributed M41 bounded export. Both build the repository-owned
recipe source with its Cargo.lock, full OpenCV linkage, locked Python dependencies
and pinned MLX-LM source revision, and deliver the tokenizer inside their separate
environments. Build directories are not runtime resources. Production prefill,
GenerationBatch execution, caches, protocol semantics and capability admission
remain the qualified implementation, not a new oMLX rewrite.

OFF now has fail-closed source-delivery provenance: installed source payload,
dependency versions/content, native binary/link closure, current source and official
checkpoint are bound. Projection defaults resolve provisioned resources, not old
donor paths; checkpoint must be explicit. OFF source overrides, validation bypass,
and accidental default-OFF launch using the MTP environment reject. R1's unchanged
verifier owns a **narrow private OFF export lane**, authenticated against its exact
archive and sealed shared dependencies. That lane is not an unqualified MTP-backed
OFF implementation or a normal operator dependency on donors.

MTP acceptance now runs the identical 64/77/28 owned R1 corpora, not milestone paths.
Projected OFF acceptance uses source-origin/Cargo gates and source-built Rust process
acceptance. Historical quick/full development campaigns are clearly unavailable in
the projection; ordinary R1 is the documented conformance route.

## Independent qualification

Actual local Git repository: `/Volumes/SDXC-512/ds41f-runtime`. Fresh independent
environments/builds: `/Volumes/SDXC-512/ds41f-m43-sealed`. Official checkpoint remains
external. Hardware and native dependency requirements are in the promoted README.

```sh
python3 tools/qualify_m43.py --runtime ../ds41f-runtime \
  --work /Volumes/SDXC-512/ds41f-m43-sealed \
  --checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
```

During actual qualification the entire ds41f-mlx checkout was renamed unavailable,
as were both historical `/tmp` candidates, OFF donor checkout, old recipe checkout,
M41 scratch and **both fresh dependency build trees**. All were restored in `finally`.
The external driver retained only pre-read control metadata; child processes ran
from runtime with independent interpreters and no PYTHONPATH fallback. Actual
Python/module/native origins, sealed inventories, native hashes/transitive OpenCV
links, checkpoint metadata/shards and build identity are recorded in the receipts
and inspector logs. Origins are runtime source + each environment, never hidden
checkout/build locations.

| Gate | Result |
|---|---|
| Fresh OFF setup from projected source | PASS; native built, not donor wheel |
| Fresh MTP setup from projected source | PASS; separate locked namespace |
| Native structural source configure/build/ctest | PASS, 5 tests |
| OFF source-origin and Cargo tests | PASS |
| OFF real supported operator/Rust/process acceptance | PASS, provenance PASS |
| MTP native corpora + real operator/application acceptance | PASS, 18 cases / 27 pre-mutation denials |
| Full ordinary R1 from runtime | CONFORMANT / PASS, both real-model lanes |
| Missing donor-origin override, OFF/MTP | rejected before execution |
| OFF no-validate and wrong-profile environment | rejected |
| Projection source-origin check | PASS |
| Independent promotion mechanics | 7 tests PASS |

Fresh R1 includes ordinary owner checks, native protocol/preview/reconstruction,
OFF Chat/Responses/Messages/tools/SSE/capacity/persist/restore and three corruption
rejections, and MTP living-client/tools/disconnect/negative partial-tool settlement.
The final R1 model lanes took 84.856 s (OFF) and 117.591 s (MTP); no performance-floor
claim follows. Unchanged numerical/native ownership and M20–M41 long campaigns are
inherited through R1 and compact source-bound receipts, not rerun or broadened.
No model math, native source, source export, dependency lock or R1 fixture changed.

Seven mechanics tests compare complete outputs across independent destinations,
source checkout relocation, changed mtimes and ignored build products. They reject
dirty input, missing/unclassified authoritative source, reference identity drift,
qualification mismatch, unexpected runtime files and runtime-only patches/nonempty
replacement targets. Final deterministic promotion additionally compares two clean
qualified output manifests/envelopes and complete per-file identities.

## One-way updates and remaining limits

For a behavior change: implement/investigate in ds41f-mlx, qualify affected owners
and reference semantics, then release-qualify a new projection and promote it.
Remove the stale `release/promotion-qualification.json` in the new implementation
commit to obtain an explicit qualification candidate. After fresh qualification,
archive/bind the receipt (`tools/record_m43.py`, same runtime as setup origin for a
fresh setup), commit it without changing qualified payload, and promote into an
empty destination. The receipt is source-owned, not manually duplicated runtime
implementation. Runtime-only patches are detected by `ds41f_mlx.projection` and are
unsupported; replacement never performs two-way merging.

R1 is declared through the surface definition rather than treated as the last
possible reference. New semantic authority requires official investigation and
possibly R2, not modification of R1 goldens. All bounded MTP exclusions remain,
including generalized/default MTP, concurrency, persistence/restart/rollover,
remote/browser/Rust applications, arbitrary schema/sampling/stops, distributed
crash-safe effects and long-context MTP.

Remaining delivery limits are explicit target-native/Homebrew/Xcode tool versions,
network/cache availability for pinned package/Cargo construction, external official
checkpoint/storage, local editable source-location installation and no remote
publication or new public-license grant. Setup seals are local build records, not
release signatures. No donor checkout or ds41f-mlx is required for supported build
and operation. The temporary attributed substrate is retained.

A setup attempt initially rejected MLX's namespace-package origin representation;
it was corrected to validate its unique provisioned package directory, without
accepting an external import. A later negative audit found the historical launcher
would start despite failed OFF provenance when the OFF seal was absent. M43 closes
that delivery defect and requalified both fresh profiles/R1. These are controller/
delivery changes, not model-semantic or execution-architecture replacement.
