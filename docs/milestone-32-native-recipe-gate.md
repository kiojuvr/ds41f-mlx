# M32 — real native recipe gate; uncovered MTP initialization commit

## Decision: MTP_SEMANTIC_CLAMP_BLOCKED

Started at ds41f `269c392664b384255f0ad627402346b1606fba20`.
**The native build and binding/parity prerequisites are solved. The live protocol
commit gate is not.** No production code or dependency manifest was changed.
Production MTP remains **OFF**, public MTP is disabled, persistence fails closed,
and token-exact immediate abort remains unsupported. **M33 is not authorized.**
No broad operational soak ran. M25–M31 evidence is unchanged.

This is a partial M32 result, not PROTOCOL_GATE_SOLVED. It does not establish
that the newly identified commit edge is fundamentally unfixable. It establishes
that the deferred chain-only hook does not safely cover pinned native MTP, and
that initialization guarding and terminal queue ownership remain to be implemented.

## Full native build, not protocol-only

- Recipe authority: `8cadfede7063c896b944e7bae05daa3549ae97ea`.
- M31 preview candidate: `066d2ef2ed0deb574a0e6b5e6136316d6f296d55`.
- M32 candidate: `29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`.
- oMLX authority: `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`.
- Isolated recipe: `/tmp/ds41f-m32-recipe`; isolated, **unmodified** oMLX:
  `/tmp/ds41f-m32-omlx`. No oMLX patch revision exists.
- Clean Python environment: `/tmp/ds41f-m32-qual`, CPython 3.13.15,
  no system-site-packages. Pristine native comparison environment:
  `/tmp/ds41f-m32-base-qual`.
- Real ARM64/abi3 native module:
  `/private/tmp/ds41f-m32-qual/lib/python3.13/site-packages/deepseek_recipe/_native.abi3.so`.
- Native SHA256:
  `454413afdcee2916795e1c5f7ce1b94a8346e76bffd6b45024f28f69f8d0a73f`.

The OpenCV dependency was the Python package's explicit image feature, not
semantic-preview architecture. Default Python/image behavior is unchanged; the
existing OpenCV-backed classes and image-resolution test execute successfully.
**No protocol-only feature gate, DOCS_RS qualification, or borrowed binary.**

Initial host audit found ARM64, Homebrew Rust 1.98.1/aarch64-apple-darwin, Apple
clang 21, CMake 4.3.3, pkgconf 3.0.7, and real libclang in Xcode, CLT and Homebrew
LLVM 22. No OpenCV formula, opencv4.pc, OpenCVConfig.cmake or development headers
were installed in normal `/opt/homebrew` or `/usr/local` locations. The actual
build environment had no OpenCV/pkg-config/libclang overrides.

With package-manager installation authorized for this isolated host:

1. `HOMEBREW_NO_AUTO_UPDATE=1 brew install opencv` installed **5.0.0_10** and
   dependencies. It provides opencv5.pc and an opencv5 CMake directory, not the
   required OpenCV 4 contract. Rust opencv 0.93.7 accepts only 3.2/3.4/4.x.
2. `HOMEBREW_NO_AUTO_UPDATE=1 brew install opencv@4` installed keg-only
   **4.14.0** and its dependencies. No global link switch was used.
3. The build explicitly selects that keg's pkg-config, CMake and headers,
   and Xcode libclang. Installation logs enumerate every installed dependency;
   `build-audit.log` records final versions, ABI, paths, headers and link names.

The intended maturin backend compiled successfully, but automatic wheel repair
failed locating transitive `@rpath` libraries (first abseil, then libgcc).
The successful maturin build uses `--skip-auditwheel`, **preserving real external
Homebrew linkage** rather than stubbing symbols or using documentation bindings.
Import and image preprocessing succeed without DYLD environment overrides.
This host-linked wheel is not a repaired/portable distribution. Neither
FULL_BINDING_QUALIFIED nor PROTOCOL_ONLY_BINDING_QUALIFIED is the final protocol
classification: the complete protocol gate has not passed.

`runtime-identities.json` records wheel/module SHA256, Cargo.lock equality to
pin, patch SHA256, source hashes, compiler, Python, Rust/Python tokenizers 0.23.2,
OpenCV 4.14.0 and resolved/hash identities of actually loaded Homebrew libraries.
The loaded-library list is for the qualification process, including oMLX tool
imports, not a claim that every listed library is directly linked by recipe.
`build-audit.log` separately records direct module linkage. The preserved release
native module hash still matches M30; release recipe/oMLX installations were not
modified. The candidate format-patch applies to the exact pin and reproduces the
candidate tree (`patch-apply.log`). It is not an approved production dependency.

## Native binding qualification

M32 adds a narrow read-only `semantic_snapshot()` diagnostic to the candidate:
opaque parser-field fingerprint, pending decoder IDs, pending count and source
byte count. It exposes neither private parser stages nor restoration/history
replay. The fingerprint is diagnostic only, not stable persisted state.
No parser, tokenizer, image feature or normal protocol behavior is forked.

- Recipe Rust semantic tests: **10 passed**.
- Real recipe Python binding suite: **30 passed**, including OpenCV images.
- Pristine-vs-candidate **native** canonical corpus: **64/64 comparisons** over
  all 22 M31 cases, token/text/character input. Protocol events, response bodies,
  finish and usage agree; only `created` timestamps are normalized.
- Native preview exactly agrees with **77/77 M31 Rust-source rows**. Each row
  repeats 200 previews and checks unchanged full fingerprint/pending IDs/count,
  determinism, exact terminal kind/index/safe count/byte span and canonical
  incremental observation. No rejected tail is fed canonically.
- Ordinary text, split Unicode, pending IDs, partial/full DSML, multiple calls,
  reasoning→DSML, single/multi-token/shared-prefix/Unicode stops and cross-window
  prefixes are covered. Already-terminal DSML returns mapping_exact=false and
  safe count zero; missing-tokenizer/closed input fails with errors.

These are genuine native binding results, not native **model acceptance** topology
qualification. The canonical corpus includes the preserved historical M11 fixture;
that fixture is not represented as fresh M32 DSML generation.

## Actual model commit-edge counterexample

After the native prerequisites passed, `run_m32_init_boundary_probe.py` loaded
fresh DeepSeek-V4.1 through pinned oMLX, enabled internal depth-5 native MTP,
encoded a real ordinary request through recipe, and instrumented initialization.
No production selector or source file was patched.

The model sampled ID **19923**, decoded as **Hello**. The diagnostic chose that
exact decoded string as a recipe stop before native initialization forwarded the
sampled token. This is a counterexample construction, not a user-selected stop
benchmark or independent tokenized-stop replacement. Matching/prediction still
runs through the real canonical recipe decoder/parser.

```text
recipe preview([19923]): STOP_SEQUENCE, index=0, safe=0, byte span=[0,5)

before _post_init_mtp:
  canonical model history = 11
  all 40 target cache frontiers = 11
  canonical parser bytes = 0

after _post_init_mtp, before emission:
  canonical model history = 11
  all 40 target cache frontiers = 12
  all 3 DSpark ring frontiers = 12
  queue = [(19923, init), (1031, init)]
  canonical parser fingerprint/pending IDs/count/bytes unchanged
  chain verify calls = 0
```

On actual emission, pushing 19923 canonically observes precisely the predicted
STOP_SEQUENCE/span. But the token had **already been target-committed**, and the
backend response still has no semantic finish reason. This violates the required
final-unforwarded-terminal invariant before the planned chain clamp can run.
The probe makes no further generation call and does not drain the second queued
token. It removes the diagnostic generation; that is **not** claimed as qualified
canonical interruption recovery or semantic quiescence.

Pinned `_post_init_mtp` forwards `main_tok`, clears rollback state, seeds two
responses and drafts before `_run_verify_cycle_chain`. Thus adding only the
M30/M31 chain seam cannot enforce arbitrary recipe stops at initialization.
A correct continuation must guard initialization **before** that forward, bind
terminal metadata to the queued response, suppress future verify/proposals and
boundary materialization, and handle canonical completion/quiescence using the
existing native machinery. It must not repair this by replaying parser history,
retokenizing stops or adding a separate target-cache rollback system.

No incomplete clamp or partial emission hook was installed as a release candidate.
Parts E–L remain blocked at this newly evidenced edge. Actual guarded DSML OFF/ON
parity, arbitrary-stop acceptance topologies, semantic interruptions, quiescence
frontiers, P6/P5 tool-result re-entry and replay/repack proofs are **not qualified**.
The prediction→observation match above does not qualify bounded queued metadata.

## Performance and regressions

Native Python preview: 15,400 timed calls, six-ID maximum diagnostic windows,
median **1,625 ns**, p95 **1,875 ns**, max **7,709 ns**. Pending group maxima and
candidate IDs are recorded per row. M31 source median/p95/max were
1,375/1,667/13,083 ns. These measurements include Python binding overhead and are
not live MTP-cycle costs. Separate clone latency and calls/cycle are unmeasured.
No guarded tok/s, acceptance, tokens/cycle or first-token latency is claimed;
M26 performance-class preservation is pending. No optimization was attempted.

Selected ds41f regressions: **65 passed, 24 subtests passed**, covering M31,
M29/M28 quiescence, M25 safety, P5/P6, qualified production-OFF selector,
M11/M14 and M20/runtime configuration. Four additional M32 evidence tests pass.
Real `omlx.api.tool_calling` imports in the new environment with genuine jsonschema.
One existing Pydantic class-config deprecation warning is separate from correctness.
The first live probe attempt used a nonexistent recipe Tokenizer.decode API; that
harness error was corrected with tokenizer-file decoding **only for selecting the
diagnostic stop string**, and its failed log is retained separately. No historical
Engram harness failure was repaired or silently reclassified; no repository-wide
soak/test-suite result is asserted.

The manifest uses null/unqualified replay, repack and quiescence counters rather
than converting “not exercised” into zero proof. Production policy is unchanged.
The next milestone must finish protocol architecture/qualification, **not** M33
operational soak.
