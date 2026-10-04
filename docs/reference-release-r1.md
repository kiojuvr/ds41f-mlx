# Reference Release R1

## Authority and scope

Official checkpoint/model/protocol semantics → ds41f implementation and qualification
→ **Reference Release R1** → future implementation conformance. R1 is an executable
development oracle, not permission to preserve a discovered reference defect.
The [strategy](reference-release-and-promotion-strategy.md) governs; M43 is not done.

Normative material is `reference/R1/contract.json`, its content manifest, owned
fixtures/checks and the manifest-bound `ds41f_mlx/reference.py` verifier. The R1
identity is SHA-256 of `reference/R1/manifest.json`; every receipt emits it. The
contract names the official 48-shard checkpoint revision/metadata inventory,
**separate checkpoint and recipe tokenizer hashes**, precision, production prefill,
protocol, execution substrates, profiles and qualification environment. Weight
inventory is source/LFS metadata and size evidence, not a fresh full-weight rehash.

`standard-off` remains default. `mtp-singleton-v1` is separately explicit and
bounded by [M41's operator contract](mtp-local-release-candidate.md). No persistence,
restart, concurrency, browser/remote, arbitrary tools/sampling or default MTP is
admitted. R1 does not turn the internal generalized fixtures into public capabilities.

## Run conformance

Provision the normal M41 source environment as documented in the local MTP
contract (even when testing OFF: it supplies the reproducibly built official recipe
binding/tokenizer and ordinary Python/MLX dependencies). Install this candidate
source checkout, not a historical editable clone:

```sh
uv pip install --python /path/to/venv/bin/python --no-deps -e .
# After a source change, record the executable artifact normally, then requalify:
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig \
  /path/to/venv/bin/python -m ds41f_mlx.mtp_identity seal --wheel /path/to/built-recipe.whl
export DS41F_CHECKPOINT=/path/to/official/DeepSeek-V4.1-Flash
/path/to/venv/bin/python -m ds41f_mlx.reference \
  --profile both --real-model --output artifacts/r1/qualification.json
```

Run from the candidate repository root. Use `--profile standard-off` or
`--profile mtp-singleton-v1` to qualify only that real-model lane; shared owned
semantic checks still run. Without `--real-model`, the distinct decision is
`SEAM_CONFORMANT_NOT_FULL_QUALIFICATION`, **never full model qualification**.
Nonzero exit and durable FAIL receipt denote rejection; missing dependencies,
material drift, wrong tokenizer and skipped semantic tests fail closed.
Qualification needs the declared Apple Silicon environment, official checkpoint,
normal provisioned native dependencies, compiler for native prefill fixture builds,
and pytest. It needs no historical milestone artifacts or donor checkouts.

A replacement implements these Python/HTTP/cache-owner seams or supplies an
adapter with the same observations. Run the unchanged R1 material against the
candidate. Candidate source hashes are **recorded, not compared to old source**.
The verifier executes actual candidate imports via module entry points, isolating
import-mutating lifecycle tests in fresh processes. This avoids inadvertently
qualifying a previously installed editable checkout. The OFF source archive is
repository-owned, attributed execution delivery of the already-qualified v0.7.0
substrate, not a source hash used to decide semantic equality. A future backend
replaces its execution adapter and earns affected model/math qualification; it
must not masquerade as the old substrate or infer backend authority from fake-cache
fixtures alone.

## Equivalence and executable coverage

- Full-context prefill/publication, source-vs-query separation, incremental
  Engram/index/candidate publications, stale capability rejection, all-40 cache
  frontiers, held-out terminal bootstrap once, no replay/repack, and append state
  ownership have operation/state fixtures. These are not floating golden tensors.
- Integer Engram hashing has independent scalar signed-int64/XOR/modulo checks,
  official geometry/prime/RNG rules and chunked request-local history equivalence.
  Synthetic normalized IDs isolate the hash contract, not tokenizer normalization.
- 22 protocol inputs / 64 canonical token/text/character records, 77 preview rows
  with 15,400 nonmutating repeated previews, and 28 restrictive recovery cases
  bind exact canonical IDs and observable protocol/certificate predicates.
- MTP queue drain/rollback, aligned idle extraction, protected cancellation/cache
  transfer, leases, sequence/exact-byte fencing, finite issuer retirement,
  certified continuation, client once-owned effects and sticky ambiguity are
  executable owner fixtures. Forbidden settlement proposal/verify/replay/repack
  is explicitly rejected.
- Strict public admission, profile/config denial, negative certificates, partial
  DSML refusal and unsupported completed outputs are checks, not just prose.
- Real OFF Chat/Responses/Messages, weather/result re-entry, SSE, live capacity,
  same-backend idle persistence/restore/continuation and corrupt artifact denial;
  real MTP text/JSON/SSE, retained continuation, disconnects, tools/effects,
  negative partial tools, retirement/fresh prefill and pre-mutation admission.

Universal intermediate hidden/logit/float bit identity is **not** required.
Qualified backend-local numerical trajectories remain valid. Exact observable
routing/state/lifecycle/protocol decisions remain meaningful; deterministic recipe
outputs are exact. Real-model workflow predicates do not invent a universal prose
or greedy-token golden across different numerical backends.

## Inheritance, invalidation and limits

[M42](milestone-42-reference-release.md) records fresh gates and content-reconciled
inheritance. M33–M41 remain historical evidence, never rewritten as R1 fixtures by
changing their receipts. Promoted material now has an R1 owner. Historical source
receipt tests are not run by ordinary R1 verification.

The bounded executable campaigns deliberately do not repeat 200K, long-model soaks,
or every native cancellation phase. Unchanged owners/content legitimately inherit
those established results. Changes to prefill math, routing, normalization,
checkpoint/tokenizer/protocol, cache geometry, native settlement, platform/build,
or capability scope invalidate the relevant inheritance. Requalify that ownership
boundary against repository-owned contracts plus official authority as needed;
a seam PASS alone does not establish a new backend's mathematical fidelity.
Extend R1 through explicit authority review or create R2, not by editing expected
outputs to fit a failing candidate. The documented M41 exclusions remain exclusions.
R1 is finite representative executable evidence, not a proof for all possible
prompts, corruptions or floating implementations.
