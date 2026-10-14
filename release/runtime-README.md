# ds41f-runtime

One-way release/source projection of **ds41f-mlx**, not a development fork.
Official semantics and Reference R1 remain above implementation qualification.
Fix features/bugs in ds41f-mlx; qualify affected owners and publish at an explicit
release checkpoint. Never reverse-synchronize runtime-only patches. Promotion
receipts bind their release payloads, not today's development HEAD.

## Production executable boundary

The projected repository is source/build provenance. Production runs a separately
installed, non-editable execution wheel. See `docs/runtime-admission.md` and
`docs/operations.md`: exact model/tokenizer and executable Python/native/kernels,
separate host ABI compatibility, and non-authoritative build/source provenance
and historical qualification evidence. No editable install, local seal, whole
Homebrew closure SHA equality, or source checkout equality is startup authority.

`release/execution-package.json` identifies the published **GLM native** MTP
wheel and its SHA, accepted execution digest and bounded acceptance. Native is
the supported M3 Ultra local default; `release/execution-package-portable.json`
identifies the separately admitted explicit portable alternative. The exact GLM
binary/metallib archive and build provenance ship in `third_party/mtp/glm-native.*`.
Native-selected admission requires actual load/ABI/symbol/UUID/GPU-resource
usability and refuses startup on failure; it never silently selects portable.
Installed identity inspection exposes `glm_profile`; MTP logs, health and operator
status also report it. OFF's legacy health reflects lazy model loading, not GLM
selected-path proof. Included wheels are published artifacts, not arbitrary
generated binaries. The subsequent bounded OFF publication is bound by
`release/execution-package-standard-off.json` (native default) and
`release/execution-package-standard-off-portable.json` (explicit alternative).
It reuses the same GLM bundle and the exact retained M45/M47 recipe wheel via
`third_party/mtp/standard-off-recipe.json`, without recipe re-admission or pin changes.
Installed native/P5/P6/selected-path acceptance is recorded in
`docs/standard-off-native-publication.md`. This is not a new full long-context or
general-runtime release; historical R1/OFF and singleton evidence is unchanged.
A subsequent projection still requires its own explicit promotion qualification.

## Operation

Provision locked dependencies and the recipe tokenizer using the source/build
setup route. `mtp_setup` constructs installed packages for separate MTP and OFF
environments; profile crossing is refused. Build/inspection is not acceptance.
The setup default deliberately provisions the exact frozen native overlay;
`--glm-profile portable` selects the explicit alternative. Do not install the
native execution wheel into an extension-absent portable environment: the wheel
alone does not provision its GLM dependency bundle. Alternatively install the
published wheel into an already **native-profile provisioned** environment:

```sh
uv pip install --python /absolute/mtp-env/bin/python --no-deps --reinstall \
  release/execution-packages/glm-native/ds41f_mlx-0.23.0-py3-none-any.whl
export DS41F_CHECKPOINT=/absolute/official/DeepSeek-V4.1-Flash
/absolute/mtp-env/bin/ds41f inspect --profile mtp-serving-v1
/absolute/mtp-env/bin/ds41f start --profile mtp-serving-v1
# In another shell, against that operator-owned ordinary process:
/absolute/mtp-env/bin/ds41f accept --profile mtp-serving-v1 \
  --runtime-url http://127.0.0.1:8000 --output /absolute/acceptance.json
```

MTP hardware remains M3 Ultra/512 GB. Host compatibility covers the supported
Apple Silicon/macOS/Python family and required native ABI/capabilities. First
inspection freshly hashes all official weight bytes; the file-identity cache only
avoids rescanning unchanged files. Checkpoint revision is
`dba1be0a40aa45a94ad051997016db3960a90277`, all 48 shards plus exact metadata.
Weights are not redistributed. Do not set import overrides or disable admission.
Stop the process before installing/updating payloads or assets.

Ordinary MTP has recipe-authoritative tools and paired target/DSpark reuse;
clients execute tools. See `docs/mtp-production-serving.md` for supported capacity,
controls and private-LAN scope. Singleton retains its bounded session/fence
contract. No Internet/multi-user isolation, general sampling, MTP persistence or
crash-safe effect guarantee is implied.

## Promotion and qualification

`release/promotion.json` and `python -m ds41f_mlx.projection` remain source-origin
and one-way promotion audit tools. Source mutation invalidates source audit or
build evidence, not an already installed executable whose bytes did not change.
Native recipe/image/numerical changes need affected semantic qualification before
publication. Historical R1 exports/fixtures remain source-owned qualification
consumers, never ordinary startup authorities. Compiler/SDK/locks/patches and
historical closure inventories remain reproducibility/evidence inputs only.
