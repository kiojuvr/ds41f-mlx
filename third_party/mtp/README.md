# Attributed source exports for the bounded local MTP profile

The base archives remain repository-owned **source** for the preserved OFF lane.
Normal-local MTP obtains the repository-delivered exact M52R native wheel, bound
by `normal-local.json`, not an operator's investigation environment. `sources.json` binds their exact SHA256, official upstream bases,
historical candidate evidence commits and delivered patch identities. No public
setup command clones/checks out those candidate commits or applies undocumented
patches. `ds41f_mlx.mtp_setup` verifies/extracts them in operator-owned build scratch,
installs oMLX and the pinned native wheel into a fresh venv. OFF still builds its
original recipe with the delivered Cargo.lock.

- oMLX: https://github.com/jundot/omlx (Apache-2.0; `omlx-LICENSE`, also inside
  archive). Full clean source export of the M33 candidate on the pinned 0.7.0 base;
  M33 generic semantic-horizon patch is included in exported source. ds41f owns the
  admission/lifecycle/recovery contracts. This remains a temporary implementation
  substrate, not the final ds41f execution-ownership destination.
- deepseek-recipe: https://github.com/deepseek-ai/deepseek-recipe (MIT, copyright
  DeepSeek; `recipe-LICENSE`, also inside archive). Full clean source export of
  the M32/M33 official-recipe semantic preview/native diagnostic candidate. No new
  guessed protocol parser or tokenizer has been substituted. Cargo.lock and the
  official V41 tokenizer are delivered in this export.
- `requirements.lock`: exact ordinary Python package versions from the qualified
  candidate namespace (including a normal pinned upstream mlx-lm package build).
  It contains **no local filesystem URLs/development checkouts**. Build tools and
  unused transitive packages remain explicitly locked for reproducibility; this
  does not promote their optional audio/vision/network/oMLX-server capabilities.

`normal-local.json` preserves the historical M52R/M53R qualification inventory:
source/patch/native/wheel/build, observed host closure, old source inventory and
checkpoint evidence. It is **not** read at production startup. Package construction
inherits its exact executable dependency and reachable semantic-native hashes;
observed unused host closure and checkout hashes are evidence only.
For standard-OFF only, `standard-off-dependencies.json` overrides the MTP
NumPy/Transformers inventory with the exact cached payloads used by the existing
200K OFF qualification (NumPy 2.5.3 / Transformers 5.18.0). Its provenance records
the historical environment receipt digest. All numerical module/native hashes
are cross-checked against the unchanged `runtime/admitted_resources.json` before
packaging; full installed payload equality remains required. Neither MTP lock
is an OFF numerical authority. No runtime enrollment or automatic reseal is added.

`runtime-support-payload.json` preserves previously sealed candidate hashes for
additional packages actually imported by the model/ordinary protocol path
(Pillow, regex/Jinja, HTTP support and their helpers), not a new observation seal.
See `docs/runtime-admission.md` for classification and installed authority.
`requirements-normal-local.lock` reproduces the complete qualified namespace,
including the already-present PDF helper packages; their presence does not admit
Web/binary/Vision capabilities to MTP. The native wheel and patched export are
MIT-licensed under `recipe-LICENSE`; oMLX remains Apache-2.0. The unchanged base
archives, base lock and OFF environment are not promoted to the M52R native.

Historical `/tmp` trees and candidate commit labels remain evidence/attribution,
never runtime lookup authorities. Archives can be inspected with `tar -tzf` or
extracted normally. OFF uses its preserved release dependencies and does not import
these candidates implicitly. See `docs/mtp-local-release-candidate.md` and M41's
identity/native-build/clean-clone evidence. New source/build/link identities require
affected qualification; equal package version numbers are not equal execution.

## Post-completion compressed growth overlay (#4365)

`compressed-growth.patch` is the isolated MIT-licensed DeepSeek V4.1 delta from
upstream `09be4f420c7409298656bcfa13699dee3a0ed938`, adapted to retain an unchanged
index-key handle on ratio=2 tokens without a publication. `compressed-growth.json`
binds the patch and the four before/after source identities. MTP provisioning
applies it to the unchanged 0.7.0 source export before building; packaging inherits
the old dependency inventory with only these four explicit replacements and
records the overlay in the new installed manifest. Neither historical
`normal-local.json` nor base archives nor OFF admission is rewritten.

This is post-completion performance improvement, not runtime completion or an
upstream release upgrade. Slots 2/3 keep logical views over two capacity buffers;
singleton extraction shares the advisory registry, partial native rollback keeps
its chain, external replacement breaks it, and paired cloning intentionally does
not export it. MLX copy-on-write preserves held snapshots. The extra resident
packed storage is deliberate, not a zero-memory-cost optimization. See
[performance evidence](../../docs/performance.md#post-completion-compressed-growth-4365).
No #4354/#4360/#4363 changes are included.
