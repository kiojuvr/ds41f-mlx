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

`normal-local.json` binds the M52R consuming-EOF patched recipe source export,
complete patch digest, native/wheel digests, build provenance, actual host-linked
dylib hashes, qualified runtime/module contents and official checkpoint inventory.
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
