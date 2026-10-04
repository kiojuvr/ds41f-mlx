# M47 — owned execution-resource admission

## Selected boundary

M42 R1, M43 promotion mechanics, M44 generation, M45 target-forward and M46
DecodeStateProducer remain closed baselines. M47 adds one compact **OFF
model-lifetime resource lease**, not another runtime implementation:

```text
repository-qualified content policy
  → ds41f startup verification
    → verified loader + checkpoint bytes
      → bound model / tokenizer / SSD descriptors
        → OFF prefill and idle P6 / TargetForwardTransaction / DecodeStateProducer
```

`runtime/resource_admission.py` owns verification, model binding, inspection and
retirement. `runtime/admitted_resources.json` identifies the allowed content.
`OmlxRuntime` verifies before calling the loader, binds the returned model before
publishing it, and retires the capability before closing model resources. The
serving backend does not retain a failed setup as a usable runtime.

The OFF dense facade and target constructor require this capability. They check
the bound model and exact admitted MLX backend/device handle before execution.
Target preflight requires the exact admitted packed-cache class and checks lease
liveness on every transaction. Admission exceptions at execution still invoke
M45's whole-lease burn; setup rejection occurs before cache mutation. An OFF
model retains its capability through idle P6 append: that entry checks resource
binding/backend/cache implementation before reserving or mutating the lease.
Generic P6 fixture/MTP primitives remain separate; they cannot bypass mandatory
OFF facade/target admission to become production executors.

DecodeStateProducer receives the admitted math handle from its parent, rather
than importing whichever numerical module wins path precedence. No checkpoint,
kernel, quantizer, numerical block or packed representation is replaced.

## What defines the compatible resource set

### Numerical modules and loaded model

The policy pins **62 selected executable modules**, not an entire external
repository: the loaded V4.1 numerical/loader closure, relevant projection/routing
helpers, packed-cache/base/sampling implementations, MLX neural-module definitions,
Ngram/NumPy primitives, tokenizer and recipe bindings. Each has content size/SHA256.
Dependency versions supplement, never replace, these checks.

For Python modules, ds41f verifies actual import spec/origin, the source loader,
source bytes, cached bytecode against freshly compiled source, and live local
functions/class methods/properties against that source. This rejects compatible
names with substituted implementations, stale `.pyc` files and replaced class
primitives. Stdlib-generated dataclass methods and imported aliases are not
misrepresented as source-local implementations.

After loading, ds41f verifies the concrete numerical module classes belong to
that admitted set, configuration equals the qualified checkpoint-derived
configuration, and the loaded tokenizer serialization matches its pin. The
lease binds the actual module tree, parameter object identities, configuration,
Ngram hashing vectors and model instance. These are admission metadata, not a
second tensor/model/cache representation. Rebinding a module/configuration or
changing the bound hashing resources fails before new execution.

Locations select candidates; they never confer authority. A relocated or
symlinked **identical** resource is compatible. A substituted resource with the
same import name, version label or filename is not. Verified file tokens and
logical checkpoint locators are rechecked at model binding/session setup;
redirected backing locations or changed files cannot silently become the new
admitted set. Stable numerical handles are retained for execution.

### Official checkpoint and tokenizer pair

The checkpoint is the unchanged source DeepSeek-V4.1-Flash checkpoint associated
with R1's revision `dba1be0a40aa45a94ad051997016db3960a90277`. M47 captures full
payload digests from the already-qualified M46 local resource set, anchored by
the historical config/index/tokenizer identities. This is local content identity,
not a new claim of remote cryptographic authenticity or checkpoint semantics.

Admission checks all **48 complete safetensor shards**, plus config, weight index,
checkpoint tokenizer and tokenizer configuration. Compatible filenames/headers,
unchanged index metadata, or shape-compatible alternative weights do not suffice.
The full payload scan precedes loading; conversion-in-progress is rejected.
The qualified loader's existing completeness/shape/conversion checks remain in
force. No conversion or repacking is performed by admission itself.

The **checkpoint tokenizer** (`c90dfa…`) and **official protocol tokenizer**
(`81f64d…`) are distinct qualified byte artifacts. R1 already records both; M47
admits the pair rather than incorrectly requiring equality. The OFF frontend
constructs its protocol tokenizer from verified bytes using the admitted recipe
native binding. The model tokenizer's resulting serialization and Engram hash
vectors are bound separately. Protocol behavior is unchanged; no new public API,
normalization policy or tokenizer transformation is introduced. The separate MTP
frontend retains its existing path.

### Native and kernel resources

The policy binds MLX core, its dylib/metallib, relevant NumPy/tokenizer native
artifacts and the recipe extension. GLM has exactly two previously qualified
profiles: M46's exact extension/dylib/metallib bundle, or R1's extension-absent
portable profile on the same admitted MLX implementation. A present but broken,
missing-member, altered or mixed bundle cannot silently downgrade to portable.

Actual dyld image locations are checked, not only files adjacent to a checkout.
Loaded Mach-O build UUIDs must match the verified on-disk images, including native
module extensions. This closes the ordinary stale-loaded-build seam where a
pathname has been overwritten with a newer binary. The GLM dylib's qualified
implementation locates its metallib through its own binary directory; the
metallib content is also pinned. Generated Metal math comes from the admitted
Python/native implementation. Effective routing thresholds, dynamic numerical
feature overrides, Python ABI and the qualified GPU architecture/device are
checked. Unknown builds/profiles fail closed rather than selecting a slower or
different ambient implementation.

This does not sign artifacts, attest GPU machine code, distrust the OS frameworks,
or protect against a malicious host forging process memory/build UUIDs. Existing
local operator/system trust is unchanged.

### SSD Engram

The full shard hashes include Engram payloads. The admitted converter's source
index mapping determines weight/scale/bias keys and files. Binding requires the
exact DiskEngramEmbedding and EngramPrefetch implementations, two expected
Engram stores, SSD/non-resident mode and no outstanding setup read.

Every opened weight/scale descriptor must refer to the verified backing inode,
size and metadata epoch. The lease also binds reader objects, mmap/file objects,
header/index metadata, data origin, tensor keys and quantization settings. Wrong
backing descriptors, altered index offsets, replacement stores, closed handles
and residency changes reject new execution. Ngram map/prime/multiplier/offset
vectors are bound at setup, not reconstructed during decode. Runtime close
retires authority before the attributed implementation drains/closes resources.

Page-touch tracking, read futures and OS I/O remain implementation mechanisms,
not model frontiers or continuation publication authorities. SSD-backed Engram
remains part of production architecture.

## Trust and lifetime limits

Resources, effective dispatch/environment settings and loaded implementation
namespaces must remain static for an admitted
model lifetime. Setup checks detect normal file/locator/model/store drift; active
transactions use stable handles and an O(1) liveness guard. This is not an atomic
concurrent-update protocol, in-process monkeypatch sandbox, signature service or
malicious-host attestation. Hot modification requires retiring/restarting the
model and fresh admission. No persistent verification cache based on timestamps,
operator-provided allowlist, automatic enrollment, or environment bypass exists.

The development capture tool is not imported by production. Updating the shipped
policy requires explicit qualification; import success cannot add a component.
`OmlxRuntime.admission.describe()` exposes logical identities, selected profile,
bound/active state and component digests without developer checkout paths.
The candidate paths remain local configuration/diagnostic details, not the public
compatibility contract. Diagnostic/offload/raw-loader and bounded MTP paths do
not receive an OFF capability merely because loading succeeded.

## Qualification and evidence

`artifacts/m47/decision.json` binds final evidence and source/policy identities.

- Positive numerical/native admission and negative unit/integration gates cover
  substituted source/import origin, live function/cache-class primitive, stale
  bytecode, same-size altered content, compatible-header altered official shard,
  package-version drift, symlink redirection, missing resources, each altered GLM
  artifact, dyld redirection, stale in-memory native UUID, dispatch override,
  backend-handle substitution, SSD index/file/descriptor replacement/retirement,
  absent model admission and protocol-tokenizer substitution.
- The official-shard negative uses only one bounded sparse fault artifact with
  the original header and file size, never a checkpoint/runtime projection. It
  rejects before loader invocation/authoritative mutation. SSD unit tests use
  real mmap/file resources; they do not claim a tiny fixture is an admitted
  production checkpoint.
- Real loaded-model faults reject changed configuration, numerical module,
  Engram key and backing descriptor while the sole committed P7 list's objects,
  slots and every frontier remain unchanged. Wrong packed-cache type rejects
  before forward and burns the complete lease; resource retirement revokes new
  target executors. Resource-file hashing is forbidden throughout post-startup
  qualification, including prefill/decode/idle transfer.
- Final matched M46 control versus admitted path: **4K and 32K contexts / 128
  generated tokens**, exact tokens and **all 40 × 7 final cache slots**. Every
  frontier, exact-list idle transfer, single terminal handoff and zero
  replay/repack remain checked. Frozen numerical/transaction controls are test
  evidence only, not production selectors or shadow-cache correctness crutches.
- Median throughput M46 → admitted: **19.883 → 19.941 tok/s at 4K** and
  **19.618 → 19.646 tok/s at 32K**, ratios **1.00292 / 1.00143**. No speedup claim;
  practical >=15 tok/s floor passes. Same-backend exactness is regression
  evidence, not a new global semantic requirement.
- Measured cold admission: **241.55 s**, dominated by reading/hashing roughly
  **510 GB** once; this cost is explicit, not called negligible. Post-startup
  resource setup median **4.00 ms**, token-liveness guard **35 ns** in its bounded
  CPU microprobe, and **no resource-file hashing after startup**. Decode itself
  shows no architectural regression. No stat/hash sweep occurs inside the token
  loop; small Ngram-resource fingerprints are setup-only.
- Affected admission/state/lifecycle suite: **146 passed, 32 subtests**. Full
  immutable R1 standard-off: **PASS / CONFORMANT**, all **24** isolated
  seam/protocol/preview/recovery/real-model gates, unchanged manifest
  `45653bd63c6a924c42dcdf0871cd950decb5efaabacce7b3c78ecf6b47f8b4ca`.
  R1's real gate exercises all three protocols, capacity/tool/result/SSE re-entry,
  persistence corruption rejection, restore/continuation and shutdown. These
  results include the final idle-P6 resource gate; the matched run binds the
  final target/producer/admission policy bytes. Final source/evidence bindings
  are in the decision artifact. Official checkpoint, numerical fidelity policy,
  DENSE_P0_P7/P5, M44–M46 authority, packed representation, standard-off default,
  SSD architecture and separate bounded MTP remain unchanged.

The initial probe incorrectly equated the two tokenizer files and failed closed
before allocation. Preliminary passing runs predate final identity guards and
are retained as exploratory evidence, not final authority. R1 uses its existing
immutable historical substrate fixture in the existing development environment;
no independent release environment or ds41f-runtime projection/promotion was
created. Package-data/surface declarations only include the new policy/source for
a future explicit M43 checkpoint. No promotion receipt is produced.

## Decision and evidence-based frontier

External implementations remain: numerical modules/weights loading, MLX/native/
Metal primitives, packed storage classes, tokenizer/recipe native mechanisms and
SSD I/O. They are **admitted subordinate resources**, not choices made implicitly
by ambient imports or filenames. ds41f owns permission to load/bind/use them,
what resource set is compatible, setup rejection, model-lifetime retirement and
how transaction failure affects continuation. Their numerical/storage algorithms
remain attributed; no source independence or dependency-removal claim is made.

**M47: PASS — qualified ds41f-mlx development state only; no runtime promotion.**

There is no unidentified resource blocker at this bounded OFF admission seam.
The measured remaining architectural cost is **full checkpoint/Engram startup
verification I/O**. The next frontier is a verified immutable-store/loader-read
binding that can avoid redundant startup reads without weakening payload identity
or descriptor guarantees—not a kernel/cache rewrite, dependency-count campaign,
persistent trust shortcut or preassigned release promotion. MTP admission and
additional hardware/artifact profiles require separate explicit qualification.
