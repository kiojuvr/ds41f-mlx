# ds41f-runtime

Canonical released source projection of **ds41f-mlx**, not a development fork.
Official semantics outrank implementation qualification, which anchors Reference
Release R1, which anchors this projection. Fix bugs/add features in ds41f-mlx,
qualify affected owners and reference/release behavior there, then promote again.
Runtime-only semantic patches and reverse synchronization are unsupported.

## Scope and requirements

Default **standard-off**: local text Chat Completions, Responses, Messages,
stateful exact-prefix Chat, SSE, idle same-backend persist/restore, Rust HTTP/SSE
boundary, local web client. Production prefill remains DENSE_P0_P7/P5/P7 and
GenerationBatch decode. Explicit **mtp-singleton-v1**: loopback only, one worker,
one session, depth 5, prompt+budget <=8192, output <=768, fixed weather tool and
living-client restrictive recovery. See `reference/R1/contract.json` for the exact
contract and `docs/mtp-local-release-candidate.md` for the bounded client API.

No generalized/default/concurrent MTP, remote/browser/Rust MTP applications,
persistence/restart/rollover, arbitrary MTP tools/schema/sampling/stops,
distributed/crash-safe effects or long-context MTP. Finite Reference conformance
is not all-input correctness or new-hardware/math qualification.

Qualified host: Apple M3 Ultra 512 GB class, Apple Silicon macOS, Python 3.13.15,
Rust 1.98.1, OpenCV 4.14.0, Xcode command tools/libclang, CMake, pkg-config, uv.
The setup currently expects Homebrew `/opt/homebrew` and Xcode installed at
`/Applications/Xcode.app`. Internet/package caches are needed for locked Python
and Cargo dependencies. `third_party/mtp/requirements.lock` pins Python packages
and the MLX-LM commit; attributed recipe export includes Cargo.lock and source.
Build locally; generated wheels/binaries are not promoted.

## Official checkpoint (not redistributed)

Provide the official deepseek-ai/DeepSeek-V4.1-Flash checkpoint, revision
`dba1be0a40aa45a94ad051997016db3960a90277`, all 48 safetensor shards plus config,
index, tokenizer and Hugging Face source/LFS metadata. Checkpoint metadata hashes
and recipe tokenizer identity are distinct; see `reference/R1/contract.json`.
Weight verification is source metadata/inventory, not a fresh full-weight rehash.
Large checkpoint and KV/Engram storage must be on suitably provisioned fast SSD.

## Fresh setup and operation

Use two **separate** fresh environments/build directories outside this source
repository. No historical oMLX/recipe checkout or ds41f-mlx is needed. Attributed
source remains an implementation dependency, not a claim of ds41f ownership.

```sh
# From this repository (a local Git repo is needed by R1's candidate recorder):
git init && git add . && git commit -m 'Import canonical promotion'
python3 -m ds41f_mlx.mtp_setup --profile standard-off --venv /absolute/off-env --build-dir /absolute/off-build
python3 -m ds41f_mlx.mtp_setup --profile mtp-singleton-v1 --venv /absolute/mtp-env --build-dir /absolute/mtp-build
export DS41F_CHECKPOINT=/absolute/official/checkpoint
export DS41F_KV_ROOT=/absolute/fast-ssd/kv
# Do not set PYTHONPATH, DS41F_OMLX_PATH or DS41F_RECIPE_PATH.
/absolute/off-env/bin/python -m ds41f_mlx.ops inspect
/absolute/off-env/bin/python -m ds41f_mlx.ops start
/absolute/off-env/bin/python -m ds41f_mlx.ops accept --output /absolute/results/off-acceptance.json
/absolute/mtp-env/bin/python -m ds41f_mlx.ops inspect --profile mtp-singleton-v1
/absolute/mtp-env/bin/python -m ds41f_mlx.ops start --profile mtp-singleton-v1
/absolute/mtp-env/bin/python -m ds41f_mlx.ops accept --profile mtp-singleton-v1 --output /absolute/results/mtp-acceptance.json
# R1 owns its private OFF source execution lane; both model gates run here:
/absolute/mtp-env/bin/python -m ds41f_mlx.reference --profile both --real-model --output artifacts/reference/qualification.json
/absolute/mtp-env/bin/python -m ds41f_mlx.projection
```

Acceptance starts/stops a real process; no separately running server required.
Rust acceptance builds from source with Cargo. Setup seals actual installed
source/native/dependency identities; a seal is not release qualification. Build
scratch can be removed after setup. Runtime uses installed packages/tokenizer,
not scratch. Inspect reports actual module/native origins and link closure.
OFF rejects source overrides and failed sealed provenance, including attempts
to bypass validation. MTP retains strict admission and identity rejection.

## Release identity and updates

`release/promotion.json` (`ds41f.promotion.v1`) binds origin commit, exact
reference/contract, surface, promotion implementation, source delivery, checkpoint,
profiles, exclusions and inherited/fresh qualification. Payload identity hashes
canonical sorted JSON file entries (path/content/size/mode/origin); the single
promotion envelope is derived and separately byte-comparable, not recursively
self-hashed. A candidate has NOT_RELEASE_QUALIFIED until a bound PASS receipt is
supplied to promotion. Never represent a candidate as a released runtime.

`python -m ds41f_mlx.projection` detects missing/modified/unexpected source and
unsupported runtime-only changes. Declared generated exclusions are `.git`,
`__pycache__`, `.pytest_cache`, `artifacts`, `target`, and editable package metadata.
They do not exclude arbitrary source mutations. Keep setup/build scratch outside
the tree. Promotion always targets an empty destination: replace the projection
from a newly qualified ds41f-mlx commit, not manual two-way reconciliation.

The M23 bundle builder is historical development machinery, superseded for this
source-delivery boundary; it is deliberately not delivered here. Historical
milestone campaigns and donor studies remain in ds41f-mlx. Current R1 fixtures
and compact inherited receipts are delivered; do not infer broader qualification
from them. Public redistribution licensing of ds41f-authored code is not newly
granted by extraction; see NOTICE and included attributed-source licenses.
