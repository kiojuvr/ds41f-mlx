# Singleton context-copy source closure (#4363)

## Subsequent current disposition

The source-closure account below retains its original payload identities and
“operator unchanged” / rebuild recommendation **at that stage**. The recommendation
was fulfilled by subsequent local MTP publication;
[current installed disposition](runtime-completeness.md#current-installed-disposition)
identifies the published payload and all four overlay provenance entries. It is
not a pending task. The cancellation mock's missing conversation has also been
corrected in repository tests; original failed evidence below remains unchanged.
No core requalification, new performance campaign or publication is requested here.

**ADOPT — CLOSED source performance optimization.** Runtime remains **COMPLETE**,
ordinary MTP remains default, and oMLX remains pinned to **0.7.0**. The installed
operator payload is unchanged. This closes the deferred boundaries from the
[initial evaluation](context-copy-evaluation.md), not a new runtime capability
or a rewrite of native MTP.

Source baseline: `4e2e862b4146be4bbc897511dd50a25cb65f7942`.
Installed baseline identity:
`927596fc6ab0a23030e459c0dd6130020636528de721bf3bf37433df87133d3f`.
Isolated admitted source-candidate identity:
`303077956acb20149b91de8f944dabad77853c1958f480772cf29e90be1317d4`.
Upstream idea: #4363 / `a2ee4a2c658d287cc6ff23ba1171b0f40119ddbf`.

## Legal authority and minimal integration

The request-local index is **derived metadata only**. The native paired
frontier and final target-verified prefix remain authoritative; no index is
exported into a reusable checkpoint or attached to the model.

For the singleton DSpark path, target rollback/commit completes **before**
`draft_next`: V4.1's `mtp_forward` does not advertise the generic cache-only
pre-commit drafting hook. Let C be the existing `state.hist_offset` and n the
final confirmed hidden-row count (`len(committed_ids)`). The consumed frontier
is **F = C+n = len(gen_batch.tokens[0])+n-1**. Native DSpark's existing append
advances its rings to exactly F. The index must contain exactly tokens `[0,F)`;
a disagreement fails closed with `SemanticGuardError`, never OFF fallback.

`committed_ids[-1]` is the **unconsumed prediction** at F. It is allowed only as
a transient **query** anchor: the query is the last <=63 consumed tokens plus
that anchor. It is never appended to searchable storage. Every selected
source continuation must fit **entirely** below F. The pending-anchor overlap
from the old prototype now returns no copy. A later cycle may index the old
anchor only after the target has actually consumed it.

The suffix rule retains the measured mechanism: indexed 3-grams, at most 32
recent candidates, longest backward match up to 64 tokens, >=12-token match,
most recent tie. The index builds once, then adds only newly consumed tokens;
there is no prompt replay, full-history rebuild per cycle, or KV repack.
Image-placeholder triples are not indexed, matches do not cross placeholders,
and copied spans cannot contain `image_token_id`. Image embeddings/identity
remain exclusively under the existing image-aware paired authority.

Only a **full configured DSpark draft block (3 tokens here)** can replace
native drafting, and only when the native controller currently selects that
same full width and sufficient output room remains. Smaller adaptive blocks
remain native. The copied branch uses the existing append-only depth-zero
DSpark hook: same target taps/rings, no decoder/Markov proposal. All drafts
still enter the existing target verifier/rollback/recipe horizon.

### Controller treatment

Copied cycles **do feed the ordinary controller observation**, using final
accepted count after cache/stop/recipe bounds. ds41f's
`AcceptanceDepthController` selects `cur=min(max_depth, accepted+1)` and ignores
latency; it is not upstream's generic drafter-cost estimator. Excluding copied
observations would leave stale target-shape feedback. Observation ordering is
unchanged (including existing asynchronous next-draft dispatch ordering).
No controller redesign, parked/OFF handoff, numerical kernel or batching policy
change is introduced. Actual small-model target/DSpark tests force rejection at
positions 0/1/2, preserve greedy output and verify frontier/controller updates.

### Lifetime and telemetry

Canonical quiescence clears the index and copy flag along with existing
transient proposals. Production qualification checks weak-reference release
after settlement. Cancellation telemetry borrows the already-existing native
state only to retain its counters across row removal; it does not export or
reuse that state. Normal and cancellation traces distinguish **copy/native
proposed, verified, accepted**, including generated proposals never verified.
Application acceptance totals use those complete verified-window counters.

No per-request priming/UID transport, interleaved-prefill capture, scheduler
architecture or continuous-batching changes are included. The existing
single-native-active topology and explicit paired priming remain sufficient.

## Previously missing production edges: closed, bounded evidence

[Receipt (gzip)](../artifacts/context-copy-closure/production.json.gz),
[comparison](../artifacts/context-copy-closure/comparison.json),
[provenance](../artifacts/context-copy-closure/provenance.json),
[memory measurement](../artifacts/context-copy-closure/memory.json).
The [driver](../tools/qualify_context_copy.py) uses real sockets, the installed
isolated source-built/admitted candidate and official external assets. There
are no target/proposal overrides in these production requests; instrumentation
only inspects state and gates cancellation timing.

- **Sampling:** ordinary temperature **1.0**, top-p **0.95**, seed **4363**;
  execution is stochastic under `omlx.utils.sampling.make_sampler`. Two rewrite
  runs have identical choices/canonical state and retained paired bytes. Each
  verifies 252 copied tokens and accepts 251. Copy q is one-hot; target filtering,
  Leviathan acceptance, rejection residual and bonus sampling are unchanged.
  Twenty seeded sparse-vs-dense one-hot reference controls agree and exclude the
  rejected copied token from the residual. No universal baseline seed/token
  identity or new distributional benchmark is claimed.
- **Termination:** actual production **EOS**, not a fabricated matcher or
  changed logits, occurs at the first position in the last copied window
  (depth 3, accepted 0). The two later proposals cannot commit. Output,
  canonical tokens and every retained target-slot/DSpark-ring byte match native
  baseline. Existing settlement consumes only the emitted EOS suffix (one
  target forward/DSpark append), not the rejected/copied future.
- **Cancellation:** disconnect after a copied target forward returns but
  **before acceptance/rollback**, while speculative cache rows exist. The
  serialized native cycle settles normally. Copies proposed/verified/accepted
  are **6/3/3**; the unverified next copy is discarded, as is the pending bonus.
  Frontier **1127** is coherent; quiescence adds no forward/proposal/replay or
  repack. A real continuation restores **all 1127 tokens** of that cancelled
  pair and remains healthy (63 further copied drafts accepted).
- **Vision:** the already-qualified bounded **corn.jpeg** conversation encodes
  once. Text-only code rewrite continuation accepts **251/252** copied tokens;
  historical-image encodes are **zero**, including repeat/reuse. Repeat choices
  and all paired bytes/image identities match. No new Vision owner, historical
  image policy, embedding path or OFF fallback is introduced.

All **19** requests have coherent target/DSpark frontiers and replay/repack
**zero**. Warm greedy rewrite/ordinary repeats have identical canonical tokens,
choices and all retained target/ring bytes between native and copy modes.
49 copy/lifecycle/serving/bind tests pass; another 44 sampling/termination/Vision
seam tests pass. One old cancellation mock lacks `conversation_request.conversation`
and fails identically on the installed baseline; it is not repaired or counted
as a pass. The earlier documented repository-doc closure failure is also
unrelated. Bounded production coverage is not a multimodal endurance campaign.

## Request-local memory

Host allocation tracking measures a 200K-token index at **4,800,800 bytes**
retained additional allocation; **11,401,404 bytes** peak construction. With
1025 generated tokens: **5,218,568 bytes** retained. Conservative object
attribution including token integers shared with canonical history is
**10,817,964 bytes**; RSS delta with allocation-tracking/allocator overhead is
~12.5 MB. Removing the index releases traced storage (512 bytes of measurement
bookkeeping remain). These numbers are reasonable on M3 Ultra 512GB, without a
new memory subsystem. Actual rewrite index attribution is ~146 KB; ordinary
~90 KB; post-image rewrite ~156 KB. No persistent/cross-request index or second
KV cache is retained; GPU peak was not separately measured.

## Performance recheck after edge closure

Same real edit as the prototype (`tools/bench_template_reuse.py`, rename
`prefix` to `kernel_prefix`), and the same ordinary agent-tradeoff explanation.
Both generate a bounded 384-token prefix. After edge closure, each has a warm
request followed by native/copy/native/copy in one process. Native control
disables only lookup in the candidate; numerical dependencies, sampler and
paired state are identical. Pair hashing/GC inspection are outside decode
measurement. No compiler-cache or broad benchmark campaign.

| Workload | Mean wall, native -> copy | Mean decode | Tokens/decode-s |
|---|---|---|---|
| Real-file rewrite | 10.356 -> 8.969 s | 10.140 -> 8.754 s | 37.87 -> 43.87 |
| Ordinary explanation | 16.835 -> 16.813 s | 16.611 -> 16.566 s | 23.12 -> 23.18 |

Safe rewrite decode improves **13.67%** (throughput **15.84%**), retaining the
prototype benefit. Per rewrite: native proposed/verified/accepted
**305/302/279 -> 39/36/35**; copies **252/252/251**. Overall verified acceptance
**92.38% -> 99.31%**, cycles **103 -> 96**. Ordinary proposes no copies;
native **383/382/172** is unchanged. Ordinary decode delta **-0.27%**, wall
**-0.13%**: no material overhead or speedup is attributed to this noise.

## Source delivery and final disposition

About **190 production LOC**, principally two oMLX files delivered by
`third_party/mtp/context-copy.patch` with exact before/after hash metadata:
`patches/mlx_lm_mtp/context_copy.py` and `batch_generator.py`. Small ds41f
changes retire the index (`runtime/mtp_lifecycle.py`) and expose complete
normal/cancellation counters (`serving/internal_mtp.py`). Provisioning/package
construction add the isolated overlay; the generated execution manifest admits
it. Test/qualification tools and evidence are not production LOC.

**Recommend a normal installed delivery rebuild/update after this CLOSED source
result; it was not performed here.** #4365, partial #4360, selective #4354 and
singleton context-copy #4363 are now closed for the **known identified
post-0.7.0 V4.1 performance backlog**. This does not adopt unrelated upstream
priming/batching work or authorize another optimization campaign. No new
milestones or other upstream optimization were begun.
