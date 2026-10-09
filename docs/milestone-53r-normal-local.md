# M53R — reproducible normal-local dependency, admission and profile

**PASS. Release blockers: zero for the explicitly supported normal-local profile.**
Baseline: `1b1f867a3bb30a6a0f670c63e37b89fa4361413b` (M52R).
Stop here; no default-MTP or ds41f-runtime promotion, extraction, soak, new
capability qualification, execution architecture or optimization is included.

## Dependency and admission authority

`third_party/mtp/normal-local.json` is repository-owned production dependency
policy, not a qualification-only source allowance. It pins:

- The ds41f runtime/native source inventory, including M53R's setup/admission
  changes. Canonical JSON inventory SHA256:
  `e3605a37bc49cb46496a8d477f0cfb56f0812da87300d006c8ca42ba0b66547e`.
- Unchanged qualified oMLX 0.7.0 source/package contents. Source archive SHA256:
  `26cc224a5fa77d8576589764a56ba8053ac31fe9f4904d8ba60305ed841f33e6`;
  upstream base `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`.
- M52R recipe source export `recipe-m52r-source.tar.gz`, SHA256
  `1f4d42fd700b876ef36dd71a95db40cf88c3fa948dedc603e0676cd16e1db66b`.
  Its complete delta remains `docs/m52r-recipe-consuming-eof.patch`, SHA256
  `560a791ff684a60ad366c3f46b961cb004f7819cfab95b4400bd0aa0d43b3794`,
  against the unchanged delivered base recipe archive. Cargo.lock, default
  image/OpenCV features and original build inputs/provenance are retained.
- Exact **obtained**, repository-delivered M52R recipe 0.1.1 wheel, SHA256
  `439d9eaab77a0edb45ad250cb937ecf425e2b315370489085eb02e2315e592e2`;
  native SHA256
  `b4aef7e5749024f3dab8aac44c0dc10aa5fe3f8fe1068dd543a15493c34b567c`.
  This deliberately avoids substituting a path-dependent operator rebuild for
  the already-qualified binary. Original Rust 1.98.1, Cargo, CMake 4.3.3,
  Xcode 26.6/build 17F113, target and build flags are recorded. All 263 linked
  Homebrew dylib hashes and 50 exact package versions/arm64_tahoe bottle
  URLs/SHA256s are fixed. Actual loaded closure must match; versions alone do
  not admit a different build.
- Python 3.13.15, MLX/MLX Metal 0.32.2, mlx-lm
  `0.31.4.dev132+g94cdcae13` from
  `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`, the complete qualified package
  versions, and relevant executable/module payload hashes.
- Official checkpoint revision `dba1be0a40aa45a94ad051997016db3960a90277`,
  config/index/tokenizer/tokenizer-config fingerprints, all 48 shard names,
  lengths and qualified LFS hashes. Added tokenizer/configuration JSON or chat
  templates reject. Initial admission hashes every weight byte; its local cache
  is reusable only for the same expected hashes and unchanged resolved path,
  device/inode, size and nanosecond mtime/ctime.

`requirements-normal-local.lock` reproduces M52R's namespace, including its
already-present PDF helpers without enabling their capabilities. The original
`requirements.lock`, base archives, OFF resource policy and OFF startup path
remain unchanged. The existing setup/operator/server/client are reused.

A seal now requires the qualified wheel **and** matching repository-qualified
runtime, substrate, recipe/native, linked libraries, dependency versions/payloads,
platform and consuming capabilities. Resealing arbitrary drift cannot approve it.
The local seal additionally binds installation-specific files/paths; its digest
is not a portable package signature. The repository, interpreter, package manager,
local cache and operator-owned resources are trusted and must remain static for
the running lifetime; this is not hostile-host attestation.

## Ordinary operation

See [setup, checkpoint, API, client and shutdown instructions](mtp-local-release-candidate.md).
The explicit profile is still `mtp-singleton-v1`; no new launcher or server stack:

```sh
python3 -m ds41f_mlx.mtp_setup --profile mtp-singleton-v1 \
  --venv "$HOME/.venvs/ds41f-mtp-v1" --build-dir "$HOME/ds41f-mtp-build"
export DS41F_CHECKPOINT=/path/to/DeepSeek-V4.1-Flash
"$HOME/.venvs/ds41f-mtp-v1/bin/ds41f" inspect --profile mtp-singleton-v1
"$HOME/.venvs/ds41f-mtp-v1/bin/ds41f" start --profile mtp-singleton-v1
```

Use the unchanged `LocalMTPClient`/ordinary `RuntimeClient` application path:
certified text, JSON/SSE, one/two weather calls in one assistant turn, explicit
exactly-once effect reservations, actual stored-result re-entry and final summary.
SIGINT/SIGTERM performs existing owned shutdown. The legacy Web gateway/browser
and Rust OFF client are not silently promoted to MTP clients. The existing
`ds41f accept --profile mtp-singleton-v1` remains the operator acceptance command.

The independent `standard-off` control remains default; stop MTP and select its
preserved OFF environment/configuration, never candidate native dependencies or
an in-process profile switch.

## Final acceptance (one completed application acceptance and sanity run)

Final clean setup used an absent `/Volumes/SDXC-512/m53r-admitted-venv` and empty
`m53r-admitted-build` on the supported host, the repository setup command, normal
locked installs and repository wheel. No manual investigation-venv install,
identity-record override, process-local source allowance, server substitution or
bootstrap was used. Development setup iterations preceded this final receipt.

Receipts under `artifacts/m53r/`:

- `setup.log`: fresh setup and strict qualified seal succeeded.
- `admission-accepted.json.gz`: strict positive admission, same qualified native,
  correct installed origins and all 48 weight-byte hashes verified.
  `checkpoint-bytes.json.gz` retains that fresh byte-verification receipt.
- `negative-identities.json`: actual installed oMLX source and MLX version faults
  rejected by normal operator inspection **and** reseal; restoration returned the
  exact original installation identity, with its seal unchanged.
- `admission-without-build.json.gz`: admitted after removing the original build
  scratch path. Runtime does not require that path.
- `m41-accepted.log`: **83 passed**. Only two tests were added to this existing
  file: the new byte-verification cache boundary and unknown-wheel seal rejection.
- `m47-negatives.log`: **26 passed**, four old OFF-positive/positive-prerequisite
  cases deselected. The preliminary co-collected run was not a PASS receipt:
  legacy OFF fixtures alter import paths and M47's frozen OFF pin intentionally
  rejects the different installed recipe. No OFF pin, R1 material or admission
  allowance was changed to force those out-of-profile cases through.
- `r1.json`: existing fresh full MTP **PASS / CONFORMANT**, 24 gates;
  `r1-gates/mtp-model.json.gz`: **18 ordinary application cases and 27 admission
  assertions**, real canonical operator HTTP startup and orderly shutdown, no
  forced shutdown. Includes two calls, four total effects with duplicate reuse,
  real result re-entry/summary, retained JSON, SSE/UTF-8/argument transport loss,
  frozen retry, cancellation, unsupported requests and concurrency rejection.
  Immutable R1 verifier/material/contract/expected fixtures remain byte-identical.
  `r1.driver.py` reuses exactly M52R's five existing active synthetic-fixture
  adapters, retaining their assertions. Its receipt explicitly records
  `source_allowance: null`; production inspection and launch are unmodified.
- `performance.json.gz` and `performance-comparison.json`: the unmodified existing
  M51R/M52R matched workload, three measured sessions and one excluded warmup.
  Mean decode **40.383142 tok/s** versus M52R **40.321402** (**+0.1531%**);
  backbone **841.304847 ms** versus **842.454083** (**−0.1364%**). Same request,
  canonical IDs/frontier 283, depth drafted/accepted topology, zero history
  replay/repack and successful retirement; within the retained 1% sanity envelope.
  The raw tool receipt retains its original `CAPTURED_NOT_GATE_PASS` label;
  the comparison is a bounded sanity check, not a new qualification campaign.

Physical runtime/model execution/native sources and production application/client
sources are byte-identical to M52R. Only setup/admission changed. No semantic/runtime
defect was found or repaired; all completed proposal/verification, target cache,
DSpark, rollback, lifecycle, consuming EOF, certification/completion, effect,
retry/reconnect and publication authority remain untouched. No Metal/movement,
long-session, new long-context or broad performance campaign was rerun.

## Exit classification

- **Release blocker:** **none** for this supported normal-local source-clone profile.
- **Operational limitation but non-blocking:** M3 Ultra 512 GB, macOS 26.5.2 arm64,
  exact host-linked Homebrew closure and `/opt/homebrew` prefix; source-clone/editable
  location-bound install; large checkpoint/model memory and initial verification
  time; trusted local peers, loopback only, no authentication. Host/catalog drift
  rejects rather than silently substitutes dependencies. No portable release
  package or new-host/OS support is claimed.
- **Unsupported capability:** stochastic sampling, Vision/content parts, 1M or
  other unqualified long-context MTP, persistence/restore/restart recovery,
  concurrent sessions/requests, arbitrary tools/schema or more than two weather
  calls, remote/browser/Web gateway/Rust application use, stateless other APIs,
  diagnostics and inherited OFF features. Reject before request/model state
  mutation; no OFF fallback or default-MTP promotion.
- **Future optimization / research:** performance/ownership optimization,
  broader context/session/capability research. None is required for M53R PASS.

**M54R/M55R have no mandatory dependency/admission/feature work left for this
profile.** Reassess whether operational release decisions need either milestone:
choose audience/rollout, and only if desired authorize separate promotion,
packaging/extraction or broader capability work. Do not start another dependency
investigation or confidence-only test campaign to prolong M53R.
