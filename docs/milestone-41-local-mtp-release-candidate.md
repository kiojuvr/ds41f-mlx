# M41 — explicit bounded local MTP release candidate

## Pre-change design (base 8813829)

Decision pending composed qualification. `standard-off` remains default and its
release manifest/dependency environment remains unchanged. M41 is one milestone,
not a selector-only promotion. Canonical destination: final-runtime-target.md;
current gaps: milestone-40-release-readiness.md.

Boundary: explicit process profile `mtp-singleton-v1`, literal 127.0.0.1, one
worker/backend/live session/response lease, guarded depth 5, stock source-JIT
kernels, DENSE_P0_P7/P5, 8192 encoded prompt+budget and 1..768 output. No stateless,
persistence, browser, remote, concurrent, arbitrary schema, raw-token or restart
recovery support. Finite IDs/sequences, exact-prefix continuation, shielded native
settlement/retirement, zero retained replay/repack, conservative negative/poison
outcomes and living-client effect ownership are unchanged.

M40's route/field/security/ingress allowlist is the initial design input. Remove
optional features rather than silently inherit upstream breadth. Mandatory exact
body sequence fencing, bounded receive/preparation and finite send stalls precede
model mutation. GET/outcome projection excludes traces and ordinary witness
placeholders. Public helper must validate the new bound certificate projection;
SSE/HTTP success never grants continuation or effect authority.

Dependency delivery: repository-owned exact source exports of the qualified oMLX
and recipe candidates, attribution/base/patch/content inventories; reproducible
ordinary package locks and target-native recipe build. No public runtime command
requires a donor Git checkout or historical candidate commit. oMLX remains a
transitional private substrate, not a rewrite. Source-clone setup, installed
runtime identity and relocated clone gates supersede M40's optional tarball plan.
Keep OFF dependency setup intact. Verify actual imports/native/link/build identities
fail closed for MTP. A rebuild is a new native artifact, not an assumption of byte
reproducibility. Checkpoint remains explicit external official asset.

Inherited evidence after content comparison: M33 semantic horizon/math/parity;
M34 native transfer; M35 response leases; M36 negative representability; M36R
certificate/fence; M37–38 living-client/ambiguous effects and lifecycle; M39 finite
identity/20,000-lifetime proof. Historical evidence is never rewritten. Rerun
changed boundary fixtures, native representation/preview on rebuild, installed
composed real text/tools/disconnect/negative/fresh workflows, ingress/security/
backpressure faults, short lifetime churn, source/import drift, matched 4096/128
OFF/MTP observation, preserved OFF real acceptance and cheap Rust/native/policy
regressions. No long-context or optimization campaign.

Qualification failures are retained and investigated. Native lifecycle contracts
cannot be relaxed to get PASS. Final decision, exact commands, source identities,
new versus inherited scopes, timings and remaining blockers will be recorded here
and in artifacts/m41/qualification.json before commit.
