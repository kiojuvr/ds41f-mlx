# Supported source verification

## Environment and command

Use a **disposable** Apple Silicon macOS 26 environment provisioned by the current
repository workflow, never an operator installation. Python is **3.13.15**;
`third_party/mtp/requirements-normal-local.lock` pins pytest **9.1.1**, MLX
**0.32.2**, MLX-LM **0.31.4.dev132+g94cdcae13** and the remaining dependencies.
`third_party/mtp/normal-local.json` owns the exact repaired **Recipe 0.2.0 wheel**,
its source revision and native-boundary patch. oMLX **0.7.0** comes from the retained
source archive plus the four ordered overlays in `ds41f_mlx/mtp_setup.py`.
Do not substitute another build with the same version string.

From the repository root, for a new disposable environment (targets must be absent):

```sh
/opt/homebrew/opt/python@3.13/bin/python3.13 -m ds41f_mlx.mtp_setup \
  --venv /tmp/ds41f-source-tests --build-dir /tmp/ds41f-source-test-build \
  --checkpoint-variant flash --glm-profile native
TEST_ENV=/tmp/ds41f-source-tests
PYTHONDONTWRITEBYTECODE=1 "$TEST_ENV/bin/python" -m pytest tests native/tests --collect-only -q
PYTHONDONTWRITEBYTECODE=1 "$TEST_ENV/bin/python" -m pytest tests native/tests -q -ra
```

The recorded correction run reused the disposable environment
`/tmp/ds41f-unified-flash-install`, previously constructed by that workflow.
Its identities/results are recorded in `artifacts/verification-baseline/`.
Run from a shell without `DS41F_*`, `OMLX_*`, `DYLD_*`, `PYTHONPATH`, pytest addopts
or production-server overrides; admission tests intentionally reject these.
No additional test dependencies or donor checkout are needed. Tokenizers come
from `$TEST_ENV/share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json`.
Unit tests read official checkpoint **metadata** at the configured development
checkpoint path; metadata resources must be mounted. No test command above admits,
hashes or executes the checkpoint weight payload. Controlled MLX tensors and tiny
synthetic native/HTTP owners are not real-model qualification.

The explicit roots cover all active repository Python source tests, without
failure-based selection, deselection, skip/xfail additions, or discovery changes.
`reference/R1/checks` is the frozen release's own verification lane, not a second
copy of current-source tests. `artifacts/` contains imported upstream repositories
and historical test snapshots, not current source discovery roots. Native C++
structural/model-qualification tools retain their separately recorded build and
fixture prerequisites; these pytest commands do not claim to execute those lanes.
The corrected source baseline has no dependency-driven skips. Full-model serving,
long-context/image qualification, checkpoints and numerical benchmarks remain
separate affected-boundary acceptance procedures, not hidden skipped source tests.

The session fixture first asserts the Runtime under test is imported from this
checkout, then executes **actual isolated installed MTP admission**, including
native/ABI/dependency/tokenizer assertions, without loading a model. Source-tree
numerical tests use that installed package's authenticated immutable variant
selection; this is not a Python source seal or checkpoint admission. Tests isolate
configuration published during app construction so one test cannot configure the
next. Template reuse tests reconstruct their pre-overlay baseline from the retained
archive and the first two overlays, rather than skipping for an unmounted donor.

## Independent integrity checks

Run each command; a wording PASS is not a source-integrity PASS:

```sh
"$TEST_ENV/bin/python" tools/check_legacy_import_integrity.py
"$TEST_ENV/bin/python" tools/check_native_import_dependencies.py
"$TEST_ENV/bin/python" tools/check_source_identity_hashes.py
"$TEST_ENV/bin/python" tools/check_authority_labels.py
"$TEST_ENV/bin/python" tools/check_repository_self_containment.py
"$TEST_ENV/bin/python" -I -c 'from ds41f_mlx.execution_identity import inspect_payload; inspect_payload()'
```

The source-identity checker needs the original small official `inference/model.py`
and `inference/kernel.py` resources under its `--checkpoint` directory; it never
reads model shards. Missing sources, unknown extraction methods, mismatched
whole-file/span identities and zero executed source assertions fail closed.
Provisioning applies the Recipe patch and oMLX overlays forward-only. Source/patch
replay checks can extract those archives in a temporary directory and repeat the
same `git apply --check` / `apply_mtp_patches` calls, without modifying installations.
Both variant Runtime wheels use `tools/build_execution_package.py`; rebuilding
native Recipe/GLM libraries or requalifying unchanged model math is unnecessary.

## Source identity discrepancy resolution

No historical acceptance receipt, official source or release record was edited.
`tools/source_identity_historical_methods.json` supplies narrowly enumerated,
reproduced extraction metadata, not replacement expected digests. The checker also
requires these artifacts to equal their exact recorded Git revision before using
the external metadata.

For both compressed-KV receipts, at revision
`14ae7418fe3cc0b27fd10fa4916b8c7cd3d06ad0`, the official source file is
`inference/model.py`, SHA-256
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
`Attention._compress_kv`, lines **739–763**, records
`fa0b8a602b8d6e200131396219685902c525c48bfb1943740446c242352ea2d9`.
This **exactly reproduces `ast.get_source_segment` on that AST function node**.
Raw inclusive lines instead produce
`cf185e9832854ebe927c80eb619ad90b1b6b2603422f12100d6e46cab31a202b`;
merely removing the final newline produces
`a641a6863a0ad781b68f587d3dd5fd6c98a9b9a70cd921babccde998e55823bc`.
Thus this was not source-revision drift, nor only a newline difference: AST
extraction also excludes the leading indentation on the first line. The original
`972f998` generator contains the same AST digest; `cd6db30` normalized other
constants but left `_compress_kv` unchanged. The original authority is reproducible.

The other compressed-KV spans (model **429–485**, **458–485**, kernel **184–204**)
reproduce their recorded digests with raw inclusive lines. Kernel file identity is
`1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
Both candidate-consumer receipts, at revision
`5cd291a552c87aec50bf91c64dd2d9dad7454c52`, reproduce model spans **488–580**,
**496–525**, **527–580** using raw inclusive lines. These 14 entries lacked method
metadata; none requires a replacement digest. The historical Boundary10 closeout
omits a whole-file digest: its original revision
`3eebd769ba8dbe1d0a08c590a8ce5de7f826894c` and the same pinned model-file identity
are supplied explicitly, while its existing sample-span digest remains unchanged.
All **253** source-span assertions execute independently of documentation wording.
`audit_source_identity_hashes.py` is not a supported verification step: its old
normalization contract can rewrite receipts and must not be used to obtain green.

## Corrected and retired coverage

- Recipe 0.2 semantic, Chat/Messages and ordinary HTTP tests retain their real
  patched extension, exact IDs, speculative rejection, settlement and failure
  assertions. Missing development Recipe interfaces are not skipped.
- Capacity tests use actual converted ordinary requests and an admitted production
  Backend double. Recipe rejects out-of-range requests before capacity admission;
  both that rejection and the explicit Runtime capacity ceiling are asserted.
  Literal image spelling in escaped tool arguments remains text in Recipe 0.2;
  the test now asserts it cannot synthesize an unbacked image control token.
- Operator mocks include `glm_profile`; supported start/stop/TUI/interpreter,
  reasoning, readiness, admission/conflict and headless coverage remains. Retired
  commands are tested as rejected, not restored.
- Reconstruction tests check all archived bytes against their recorded manifest
  and historical Git blobs, not current source against an obsolete restart.
  Current producer, transaction, model ownership and admission tests remain.
- The old OFF/Recipe 0.1 generation-default test and three OFF live-turn suites
  (`test_stateful_live_delivery`, `backpressure`, `retirement`) were retired.
  They exercise the unselected OFF Backend, not production MTP. Current MTP
  cancellation, backpressure, lease, cleanup and exact-settlement tests remain.
- M30 blocked-gate diagnostic tests use Recipe 0.1 parser/copy contracts and old
  DSML spelling; current Recipe 0.2 semantic/Chat/Messages tests supersede them.
  The historical diagnostic tool is not a current verification entry point.
- The imported Engram benchmark-runner test references a benchmark driver that
  was never imported and is absent from current native build targets. Its exact
  bytes are retained at `docs/archive/validation/retired-tests/engram-test-runner.py.txt`.
  Provenance checks resolve that explicit relocation and still verify the original
  154 import digests; the immutable import receipt is unchanged.
- M47 OFF-only positive source seals and real-shard-header fault setup no longer
  define current admission. Current installed MTP native/version/tokenizer gates
  are exercised instead; generic source/bytecode fault primitives retain local
  fixture tests. Production pins were not regenerated or weakened.
- M36R Recipe 0.1 evidence is checked against original Git/evidence identities,
  not re-encoded with Recipe 0.2. Its negative admission test now parametrizes
  only negative cases: positive matrix rows no longer return without assertions.
- Sampler conformance previously directly called `apply_top_p` with zero, although
  the supported sampler treats zero as filter-disabled. The helper then computes
  `num_dropped == vocabulary_size` and gathers beyond the array; comparing two
  undefined GPU reads was intermittent, despite byte-identical source. Direct
  helper comparison now covers its valid positive domain. All existing sampler
  policy combinations including zero remain, and a new assertion verifies the
  out-of-range index calculation and that zero never dispatches that helper.
  No sampler or numerical source changed.
- Full-suite execution exposed a genuine singleton HTTP defect: two supported MTP
  profiles selected different Recipe conversion paths, leaving singleton requests
  on the retired 0.1 API. Supported HTTP preparation now uses the existing Recipe
  0.2 path in both profiles. Pre-mutation singleton conflict tests exercise it;
  ordinary behavior, capacity ceilings and model arithmetic are unchanged.
