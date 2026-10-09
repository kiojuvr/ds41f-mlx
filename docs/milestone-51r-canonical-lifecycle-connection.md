# M51R — minimal canonical/lifecycle connection

**PASS — bounded M50R candidate, greedy sampling. M52R is not started.**
Explicit user authorization follows the [M49R doctrine](mtp-architecture-reset.md)
and [M50R freeze](milestone-50r-omlx-candidate-baseline.md). Evidence and exact
qualification adaptations: [artifacts/m51r](../artifacts/m51r/README.md).
This is not normal-local admission, default MTP, dependency promotion or a release.

## Working authority connection

`runtime/mtp_lifecycle.py` extends the existing native session, not its execution
architecture. `CanonicalPhysicalConnection` contains **host metadata only**:
request-owner lifetime UUID, revision, consumed IDs, queue-ahead suffix, known
pending prediction ID, disposition and qualified sampling draw count. It has no
model, cache, ring, journal, forward, sampler or scheduler implementation.
`serving/internal_mtp.py` connects the existing request/owner seam and records the
settled authority relation. No archived M50–M56 implementation is imported.

There is exactly one executable authority: the original oMLX `BatchGenerator`
and its 40-layer target cache, three native committed-context rings, verification
stash and asynchronous proposal state. The existing DwarfStar DENSE_P0_P7/P5
prefill and qualified MLX numerics remain authoritative. Native continuation
installation/extraction retains its original ownership hooks and row views.
The backend owns process-scoped model/SSD resources; request/session owners own
cache/ring lifetime. The connection never creates a second executable state.

Let E be the existing server-observed/emitted history length, F the native
committed ring/history frontier and Q the native response queue:

- F >= E: `len(Q) = F-E+1`. Consumed canonical history is emitted history plus
  `Q[:-1]`. The final Q entry is an **unconsumed** correction/bonus prediction.
- F = E-1, empty Q: the final emitted token is still a prediction, excluded from
  consumed history until the existing bounded settlement forward consumes it.
- Bootstrap/preactivation has no active MTP queue. Prompt consumption is bound
  to native priming/ring metadata; the native pending array stays native-owned.
  A missing host prediction ID is not a claim that no native prediction exists.
- Consumed history cannot change or shrink. Native `hist_offset` must equal the
  three ring offsets. At settlement, consumed IDs must exactly equal the retained
  canonical IDs, all 40 target offsets and all three ring offsets.

The older `CanonicalTransportHistory` is the **emission/observation projection**
while execution is ahead, not a second consumed frontier. `rec.canonical` is the
retained idle prefix until settlement. Socket delivery is not consumption or ACK.
Accepted drafts enter consumed history only after native prefix selection; draft
KV, rejected suffixes and the last queued prediction never enter it prematurely.
No target-offset `.item()` is added on next: target/ring correspondence is proven
by preserved native commit code, live diagnostics, byte oracle and the existing
all-layer settlement checks, not a new per-token device fence.

The frozen public profile admits temperature zero and no filter/seed controls.
Its sampling RNG draw count is zero, including discarded predictions and drafts.
The adapter makes no MLX RNG call; its UUID uses separate OS identity entropy.
Non-greedy internal callables have **unknown** draw state, not a stochastic PASS.
No non-greedy capability or serial-RNG equivalence is newly admitted here.

## Original-forward evidence and lifetime

Each original native response must match the active UID. Its original logprobs
(derived from same-forward logits) are bound to lifetime/revision and the exact
**conditioning consumed prefix**, separately from the predicted token. The latest
binding is replaced on next and cleared on retirement/fault. No external receipt
or stale observation is accepted into canonical state. Raw hidden taps remain
native-owned: the unchanged same-forward 37/38/39 path appends only selected
committed rows into the original rings. The existing byte oracle verifies taps,
full consumed-row logits, append inputs and ring contents on the new connection.
There is no recomputed hidden-state/logit surrogate.

Native regions still run on the singleton worker under the existing shield/lock.
Cancellation settles the already-committed suffix and retires prediction owners;
it does not start a proposal/verify or replay the prompt. Protected terminal uses
the existing UID/ordinal seam, drains its boundary and publishes after settlement.

A mutation/materialization/history/connection publication exception now closes
the native owner immediately, marks the connection faulted, and clears all
session-held native aliases, including original response/observation references.
It cannot subsequently quiesce into executable idle. Native removal errors are
no longer swallowed into a successful settlement. Serving cleanup poisons the
logical singleton, drops priming/cache/ring/processor ownership, synchronizes and
requires retirement. Native close failure still clears aliases; failed physical
retirement cannot advertise a reusable idle lifetime. Successful close is
idempotent. Process-scoped immutable model residency is not session cache leakage.

## Conformance evidence

- Five real workloads have **exactly equal native topology dictionaries and IDs**
  to M50R: ordinary full/clamped acceptance, tool terminal, early disconnect,
  actual numeric rejection/rollback, and committed-queue disconnect. Settled
  frontiers are 283, 546, 243, 359, 251. Depth 3, width-4 blocks, parallel proposal,
  mixed precision, capacity-128 rings and asynchronous ordering remain unchanged.
- Reused byte oracle: **16 primitive + 2 live cases PASS**, before 127/128/255/256,
  accepted 0..3. Exact all-layer state, integer history, compressor tails,
  index/candidate publications, full logits, taps and physical ring contents.
  Offline reference work begins after retirement; no competing live producer.
- New-boundary real fault: after cycle 1, F=247, E=245, two committed queued IDs
  and one unconsumed prediction. Injected publication failure burns all session
  aliases and serving retirement poisons/drops native ownership. No idle recovery.
  Target/connection rejection, partial native removal and foreign response failure
  assertions reuse the affected lifecycle tests rather than an archived producer.
- Boundary diagnostics match M50R **exactly**: 4,909 constructors, 1,467 item,
  15 tolist, 69 async eval, 7 eval, 46 sync, 1,024 concatenate regions; identical
  4,035,408 NumPy-source import bytes and 3,598,848 CPU selected-row copy bytes.
  These are API/bridge observations, not DMA or individual fence/kernel counts.
- No extra MLX operator, array import/export, dtype conversion, copy, staging,
  replay, repack or target re-execution is introduced. New Python tuple/history
  metadata and reference retention are necessary semantic binding, not physical
  tensor movement. Native observation reference lifetime already followed
  `last_response`; both references now retire explicitly. M50R's internal Metal
  opacity remains bounded by unchanged delegated source/binaries/geometry and
  fresh boundary evidence; no new Metal trace or opaque-kernel change is claimed.
- Affected tests **33 PASS**, profile tests **81 PASS**, oracle-tool tests **7 PASS**.
  Retained reset/source guards **3 PASS**. R1 assertions **24/24 PASS**, including
  18 real-model bounded cases. Two synthetic
  owner fixtures are adapted to new metadata, with original assertions retained;
  immutable R1 material is unchanged. Qualification-only source allowance is
  explicit and does not reseal installed admission.

## Matched rates, not diagnostic rates

Three separate loaded processes, one excluded warm-up plus three fresh sessions
per process; controls bracket the connection run. Exact checkpoint/dependencies,
precision, request bytes, context 242, budget 64, greedy IDs (41 including stop),
10 cycles/29 accepts/one stop clamp match. No profiler or extra sync in rate runs.

| Metric | Control before | Connection | Control after |
|---|---:|---:|---:|
| Next decode, canonical tok/s | 40.4312 | 40.4326 | 40.4224 |

Control decode drift: **-0.0218%**. Connection vs bracket control mean: **+0.0144%**
decode; existing backbone/verification wall telemetry **-0.0670%** (840.887 vs
841.451 ms). This telemetry overlaps lazy execution; it is not isolated GPU time.
Dispersion and residency/VM snapshots are retained. No material regression or
performance difference is deferred to optimization. The historical <1% envelope
is not subtracted as an allowance. Host metadata adds no observed physical region.

## Explicit residual boundary

PASS is the minimum coherent canonical/lifecycle connection in the frozen bounded
candidate, not completion of consuming EOF/tool semantic horizon, full JSON/SSE,
effect ledger, exact retry/re-entry or worker delivery races. Those are **M52R**;
existing bounded HTTP/R1 checks do not close that broader application contract.
M52R remains unstarted. Stochastic sampling, other capabilities, long-context/
soak/cold/pressure campaigns and normal-local/default/release admission are not
qualified. Installed admission remains unchanged and rejects the unpromoted source
delta; only exact qualification processes allow the two changed runtime files.
No dependency resealing, packaging or `ds41f-runtime` promotion occurs.
