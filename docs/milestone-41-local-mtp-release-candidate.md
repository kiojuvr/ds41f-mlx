# M41 — explicit bounded local MTP release candidate

## Decision

**`QUALIFIED_EXPLICIT_BOUNDED_LOCAL_MTP_RELEASE_CANDIDATE`** for the separately
selected `mtp-singleton-v1` source-clone profile on the recorded M3 Ultra 512 GB
macOS target. **`standard-off` remains the default qualified production profile.**
No profile parity, browser/remote/general-agent/persistence promotion follows.

Operator/application contract: [local MTP candidate](mtp-local-release-candidate.md).
Durable, content-bound receipt: `artifacts/m41/qualification.json`. Qualified
implementation source: `c5eb83f` (final evidence/docs commits do not change runtime).
Base: clean `master` `8813829`. No release blocker remains **within this declared
bounded opt-in scope**; deliberate exclusions below are not secretly enabled.

## Pre-change design (base 8813829)

The following design was recorded before implementation; its pending-decision
statements are historical, superseded by the decision and qualification below.

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
and in artifacts/m41/qualification.json before final qualification commit.

## Implemented owning boundaries

- `mtp_profile.py`: explicit capability/limits, strict UTF-8 JSON, duplicate and
  nonfinite/type/field rejection, pinned weather declaration and result envelope,
  canonical uint64 fence syntax, ordinary-message certificate binding.
- `serving/mtp_public.py` wraps—not replaces—the M33–M39 guarded native owner.
  Canonical history/cache, official recipe semantics, response reservation/lease,
  protected native settlement, finite server-issued lifetime and client effects
  remain distinct authorities. Scalar public outcomes omit native IDs/witnesses.
  Completed unsupported output is negatively classified, never repaired or used
  for effects. Explicit identity/history conflicts are bounded 409 codes.
- Local ASGI/h11 boundary: literal IPv4 loopback/Host/no-Origin, route/capability
  denial before mutation, one no-queue cumulative body/preparation slot, eight
  sockets, header/body/send deadlines, bounded buffers. HTTP expiry never revokes
  a native owner early. Protected model work may outlast transport deadlines.
- Normal `ds41f inspect/start/accept --profile mtp-singleton-v1`, one asyncio worker,
  no validation bypass/import-shadow/experiment environment or OFF fallback.
  Operator shutdown reaches uvicorn, waits for coherent cleanup and retirement.
  Ordinary browser gateway/legacy tool helper refuses this profile before mutation;
  Rust remains OFF transport, not a supported MTP application seam.
- `LocalMTPClient`: typed capability/dependency identity handshake, fresh issuer
  record validation, one frozen identity/stream/latest copy, certificate/body/message
  binding, explicit reconcile/retire/fresh and bounded effect reservation/results.
  A retained POST lost before admission leaves an older settled slot: retry only
  the same frozen **next** identity, never treat that observation as a new ACK.
  Unknown create/DELETE/effects and failed retirement still block reuse. No unsafe
  automatic tool execution, reset, witness-placeholder results or discovery.

No native mathematical lifecycle file, prefill/handoff layer, OFF configuration,
OFF release manifest/dependency pin or Rust source was changed. The internal
settlement certificate now carries the chosen conversion options (MTP thinking
none); inherited internal/default options remain unchanged. Request-fence authority
is untouched; public wrapping makes existing conflicts explicit, not weaker.

## Reproducible delivery / independent-clone gate

Exact attributed oMLX and recipe **source** exports/licenses/base/patch identities
are delivered in `third_party/mtp/`; normal package locks and full Cargo.lock native
build are owned by `mtp_setup.py`. oMLX remains a temporary private substrate.
The official V41 tokenizer is delivered into the fresh venv, not looked up in a
historical tree. No borrowed wheel, requested old revision checkout, undocumented
patch application or donor import-path override is an operator prerequisite.

Fresh `git clone --no-hardlinks` into `/Volumes/SDXC-512/ds41f-m41-final-clone`, then
normal setup in an absent `/Volumes/SDXC-512/ds41f-m41-final-venv` completed. During
inspection, installed operator/helper/native/composed/socket/performance gates,
**both historical `/tmp` candidates, both OFF donor checkouts and native build
scratch were renamed unavailable**. All selected imports were clone/venv/share
resources; those directories were restored only after MTP qualification.
`clean-clone.json` and `hidden-authorities.json` record exact paths/commands.
This is an independently located source-clone delivery gate, not a claim that an
editable install survives moving its source without reinstall/rebuild/reaccept.

Artifact identities:

- dependency/runtime/helper/payload seal:
  `9ae9a19d84e45c1ddebe5347d2860d3a2133813c1df72837f1fcac5fd2f406c2`;
- newly compiled full native recipe:
  `69480f1d8a3687f523c81c348adf617828f39f4b657dcaa0c78d351148e785c8`;
- newly built wheel:
  `9d288f7d00958e6b2c2e2d28ba8865a4cbffddaaf190b2fa4757e9ac5b48018a`.

`final-inspect.json` binds all 89 installed distributions/14,537 payload files,
actual module origins, exported sources, own runtime/native source files, native
binary and transitive non-system host dylibs. The seal is a local build record,
**not a signature or semantic qualification**. Checkpoint config/index/tokenizer
hashes and all 48 source revision/LFS metadata/size entries were checked, **not a
fresh full-weight rehash**. Official external checkpoint configuration is explicit.

## Fresh qualification (not inferred from M40/M39)

Exact commands/artifact hashes and source bindings are in the receipt. Normal
operator invocation was `ds41f accept --profile mtp-singleton-v1 --output ...`.

| Fresh gate | Result / evidence |
|---|---|
| Full host-native setup from independent source clone | PASS, `final-clone-setup.log`, `clean-clone.json` |
| Integrated native preview | PASS, 77 rows / 15,400 previews; exact source agreement, deterministic and nonmutating |
| Native official-base parity | PASS, all 64 records equal M32 official-base oracle |
| Protocol representability | PASS, 28 corpus canonical IDs and restrictive predicates equal M39; not broader RPC schema qualification |
| Installed operator + supported living helper + real checkpoint | PASS, **18 composed outcomes / 27 admission rejections**, `final-composed.json` |
| Real socket admission/header/body/send limits | PASS, **8 cases**, `final-sockets.json` |
| Actual startup identity/configuration faults | PASS, **26 cases**, `final-identity-negatives.json` |
| Changed public/client/lifecycle/lease/fence/config fixtures | **164 passed**, `final-boundary-client-tests.log` |
| Installed real-native/MLX semantic seams and native lifecycle/cache fixtures | **22 passed**, `final-native-lifecycle-tests.log` (synthetic target, not model math) |
| Fresh C++ native reference build/ctest | **5 passed**, `final-native-core-tests.log` |
| Rust workspace | **7 passed**, `final-rust-tests.log` |
| Preserved OFF cheap/config/metadata/boundary/client regression | **19 passed**, `final-off-cheap.log` |
| Preserved OFF real Rust/process/HTTP acceptance | PASS / provenance PASS, `final-off-real-acceptance.json` |
| OFF Chat/Responses/Messages, weather/results, persist/restore/continued turn | PASS, `final-off-protocols.json` |
| Canonical repository links/provenance/archive policy | PASS, `final-repository-policy.log` |

Composed outcomes include text JSON/SSE/retained continuation, real partial TCP
body loss after a prior slot, UTF-8 byte loss and certified continuation, one/two
completed weather calls with **four once-owned client effects** and stored results,
completed-tool delivery loss, partial-tool **negative** certificate with no effect,
explicit DELETE/fresh prefill, aliases/output edges, active duplicate/next/create/
DELETE conflicts, stream cancellation and retained continuation. Twenty further
empty lifetimes reject retired/foreign IDs; outcome expiry and changed exact bytes
reject without generation. All **18** settlements had aligned idle frontiers,
M28 new-proposal/new-verify/history-replay/repack counters **zero** and at most one
permitted canonical-token materialization. No transport packet/DONE was an ACK.

Socket gate uses the actual operator process for cumulative chunked/declared excess,
compression/duplicate fence/overlong Content-Length, preparation/body deadline and
eight-socket/header deadline. **Saturated-send pressure is separately synthetic**
bounded SSE comments (no model): producer plateau, 30.0089 s timeout, iterator
closed. Actual native cancellation/settlement belongs to composed acceptance and
preserved native phase fixtures; no fake SSE load is called model acceptance.

Identity faults include altered runtime/package, wrong/missing tokenizer,
missing/wrong same-version native module, an actually rewritten missing dylib link,
missing/wrong checkpoint metadata, forbidden source/import paths, host/session/
diagnostic/trace/P8/upstream row-exact/worker/profile/bypass overrides. Pristine
before/after passed; no shared host dylib or OFF environment was altered.

## Inherited authority, precisely scoped

M33 semantic horizon/math (full-context priming, non-delivery ACKs, response UID/
ordinal ownership, ordinary encoding); M34 transfer/protected-phase/quiet ownership;
M35–36 response cleanup/unstarted leases and negative representation; M36R
restrictive certificate/fence; M38 ambiguous effects/lifecycle; M39 finite namespace/
20,000-lifetime proof remain evidence, **not newly rerun long-model soaks**.
Whole `mtp_lifecycle.py`, `recipe_semantic_guard.py` and `request_fence.py`, prefill/
handoff tree and relevant issuer/get/retire/effect methods are content-compared
unchanged. Changed wrappers/client observation/binding get fresh affected fixtures
and composed tests. The delivered candidate model/horizon Python source is verified
against the exact exports; the **rebuilt** native gets fresh parity/preview/corpus.
Historical records are never rewritten to become public release qualification.

### Pre-existing historical receipt-test debt (not hidden as PASS)

A supplementary run of old M35/M36R/M37/M38/M39 whole-**current**-source receipt
assertions returned **5 failed / 58 passed / 137 subtests passed**. Every one of
those five already had stale source bindings on clean base `8813829` (respectively
9/7/6/7/2 mismatched source files). They are historical receipts, not today's
profile approval. `legacy-receipt-test-debt.json` records baseline/current comparisons;
`evidence-owner-tests.log` preserves failure output. Raw historical evidence was
not edited to manufacture green tests. The new M41 current-content receipt guard
passes, old raw/frontier/body checks pass, and unchanged hard owners/spans plus
fresh affected native/client/composed/OFF gates bind current qualification. General
historical-test authority maintenance remains repository debt, **not a native or
scoped MTP release blocker**; no blanket all-repository-green claim is made.

## Measurements / practicality

Fresh same-installed-candidate **stock-JIT greedy 4096/128** paired control:

| Mode | Decode tok/s | Decode seconds |
|---|---:|---:|
| Private OFF math control | **19.4721** | 6.5735 |
| Guarded MTP | **43.1998** | 2.9630 |

Ratio **2.21855x**, considered-draft acceptance **103/109 = 0.944954**; greedy token
identity observed in this pair, not a cross-backend contract. Warmup MTP 42.9517
tok/s, model load 67.9851 s. This is **not** preserved OFF HTTP throughput; its
separate real OFF protocol/lifecycle/tool/persistence outcomes passed. The composed
MTP process's first load was 92.0116 s; scalar per-turn load/prefill/decode/cleanup/
settlement data are stored, not hidden in aggregate speed claims. Short/tool turns
have different acceptance/overheads. No arbitrary performance floor, universal 2x,
long-context/concurrent quality or optimization campaign is claimed. The measured
bounded local lane is materially useful while preserving default-off behavior.

## Failures retained and corrected

Initial clean `uv --python 3.13` chose 3.13.14: mandatory identity rejected startup;
setup now selects/verifies the normal exact 3.13.15 interpreter. A final-clone full
native build succeeded but seal incorrectly searched an install-ID basename after
symlink resolution (`414` versus resolved `4.14.0`); corrected identity predicate,
then **rebuilt a fresh absent venv**, no borrowed old artifact/validation bypass.
The first performance command omitted `--output` and failed argparse before work;
corrected invocation produced the fresh pair. Logs preserve these excluded attempts.
Preparation/identity/body tests exposed and closed two public-edge gaps before final
gates: stable 409 classification for changed/expired bytes, and old-settled-slot
observation after a retained body loss. Same frozen retry—not a new identity or
absence ACK—closes that race. No native lifecycle contract was relaxed for PASS.

## Limits, blockers and final destination (separate)

**No remaining blocker within this bounded explicit local RC.** Still deliberately
unavailable: default/implicit MTP, arbitrary tools/schema/sampling/stop, persistence/
restart/context rollover, universal partial-DSML reconstruction, browser/remote/
Rust MTP application support, multi-user/authenticated/concurrent/batched serving,
crash-safe or distributed exactly-once effects, unsupported hosts/platforms/native
artifacts and long-context MTP. Client/runtime replacement cannot resolve unknown
external effects. These are limits, not silently working unsupported capabilities.

Longer-term [final-runtime target](final-runtime-target.md) remains canonical and
prospective: migrate temporary oMLX execution ownership into ds41f and complete
self-contained normal **OFF** delivery too. Optional artifact portability/dependency
reduction/kernel work needs affected ownership/parity gates. M41's coherent
source/native setup closes its selected profile's public checkout barrier; it does
not pretend legacy OFF donor setup or the temporary substrate is already removed,
or schedule dependency removal as an automatic next milestone.
