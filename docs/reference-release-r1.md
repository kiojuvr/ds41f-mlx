# Reference Release R1

## Authority and scope

R1 is a **repository-owned executable development oracle**, subordinate to official
checkpoint/model/protocol semantics. It is not permission to preserve a discovered
reference defect or fresh release approval for later HEAD. [Promotion strategy](reference-release-and-promotion-strategy.md)
owns one-way authority; [qualification](qualification.md) separates proof scopes;
[runtime admission](runtime-admission.md) governs installed startup, not R1.

Normative material is `reference/R1/contract.json`, its content manifest, owned
fixtures/checks and manifest-bound `ds41f_mlx/reference.py`. Identity is SHA256 of
`reference/R1/manifest.json`, emitted in every verifier receipt. Material/verifier
bytes are unchanged by compilation. The contract binds checkpoint revision/metadata,
separate checkpoint/recipe tokenizer hashes, precision, prefill/state/protocol,
execution substrates, bounded profiles and qualification environment. Its weight
inventory is source/LFS/size evidence, not fresh full-weight-byte admission.

R1's real-model lanes are `standard-off` and bounded `mtp-singleton-v1`.
It does **not** include ordinary `mtp-serving-v1` as another verifier profile or
turn generalized internal fixtures into public APIs. R1 excludes broader MTP
persistence/restart/concurrency/images/arbitrary tools/sampling; these exclusions
are reference-scope limits, not today's blanket ordinary-serving capability policy.

## Run conformance

The unchanged verifier interface is:

```sh
/path/to/qualification-env/bin/python -m ds41f_mlx.reference \
  --profile standard-off --output /absolute/new-r1-receipt.json
```

Invoke from the candidate repository root in an explicitly selected qualification
environment; verify `ds41f_mlx.reference` imports that candidate and its owned R1
material, not another installed/editable checkout. This is not installed ordinary
startup. `--profile` accepts `standard-off`,
`mtp-singleton-v1` or `both`. Without `--real-model`, the decision is
`SEAM_CONFORMANT_NOT_FULL_QUALIFICATION`, never full model qualification.
`--real-model` additionally invokes selected model drivers, which can launch real
runtime processes; it needs explicit scope/environment reconciliation and a model
maintenance window, not another process beside an active server.

This is interface documentation, **not a claim that current HEAD/environment passes
all frozen R1 gates** or an instruction to run full R1 during compilation.
The verifier checks material/tokenizer identity and native recipe/protocol
requirements, runs owned checks in separate processes, records candidate source
hashes rather than requiring historical source equality, and writes durable
RUNNING/FAIL/PASS evidence. Missing dependencies, mismatches and skipped required
predicates cannot become full approval.

Recipe assets are provisioned under the qualification environment's
`share/ds41f-mtp/recipe`; frozen OFF real-model delivery uses the attributed owned
`off-source.tar.gz`. No external historical donor checkout is needed for those
owned fixtures. Frozen test/profile/native requirements are not automatically
satisfied by any current installed OFF/MTP package. A replacement must implement
or explicitly adapt observed seams and qualify affected math/state; no fake-cache
PASS establishes a new numerical backend.

The earlier R1 guide (`c1d1f716:docs/archive/editorial-baseline/reference-release-r1.md.txt` in Git)
retains original editable-checkout and `mtp_identity seal` commands for historical
reproduction. Seal is no longer a supported startup/qualification repair action;
do not reinstall, copy payloads, weaken pins or alter reference material to make
those old steps appear current. Present publication/setup belongs to operations;
R1 qualification and ordinary installed admission remain distinct lanes.

## Equivalence and executable coverage

Owned observations include:

- Full-context prefill/publication, source/query separation, all-layer frontiers,
  terminal bootstrap once, Engram/index/candidate state and append ownership.
- Independent integer hashing/geometry/history rules; synthetic IDs isolate that
  state predicate, not substitute tokenizer semantics.
- Exact canonical protocol records, nonmutating semantic previews, restricted
  representability/recovery and negative certificate predicates.
- MTP rollback/queue drain/quiescence, protected cancel/transfer, sequence/exact-body
  fencing, finite retirement and conservative client-effect ambiguity.
- Strict profile/admission refusal and partial/unsupported tool semantics.
- Bounded real OFF protocols/tools/idle artifacts and singleton MTP HTTP/continuation/
  interruption/retirement outcomes, only under their declared environment/profile.

The edition retains exact fixture/preview/recovery counts; manifest-owned material
remains authoritative. Universal intermediate float/hidden/logit bit equality and
cross-backend prose/greedy-token goldens are not required. Exact observable routing,
state, lifecycle and protocol decisions still matter; backend-local valid numerical
trajectories do not waive correctness or performance requirements.

## Inheritance, invalidation and limits

M42 closure/receipts remain historical input-bound evidence. A future projection may
deliver compact inherited receipt material, but later HEAD or today's old runtime
directory is not thereby qualified. Historical source SHA tests are not ordinary
R1 conformance; archive migration did not change their original receipts.

Changed math/routing/normalization, checkpoint/tokenizer/protocol, cache geometry,
native settlement, platform/build or capability scope requires affected authority
review and qualification. Stable owners may inherit explicitly reconciled evidence;
finite real-model R1 does not repeat every long-context/soak/native cancellation
campaign or cover all corruptions/prompts. Full re-verification at an appropriate
stable boundary is a separate decision, not every intermediate milestone.

A reference defect needs explicit review/versioning, potentially R2—not expected
outputs rewritten to fit a failing candidate. Preserve R1 and its negative evidence.
No material, verifier, fixture, receipt or model gate is regenerated by this edition.
