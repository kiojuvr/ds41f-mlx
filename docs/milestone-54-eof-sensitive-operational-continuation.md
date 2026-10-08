# M54 — consuming-state EOF preview and operational continuation

**YES — M54 development operational integration PASS (bounded development
scope). No normal-runtime/default promotion. Full R1 is NOT PASS.**
Base: `fbca4bc` (M54 continuation), M51–M53 execution core unchanged.

## Native architecture

The incremental patch is recorded in
`artifacts/m54-eof/recipe-consuming-eof-preview.patch`; base/modified source
hashes, native binary, wheel and Cargo-lock identities are recorded in
`artifacts/m54-eof/native-provenance.json`. This is a development dependency
patch, not release/packaging or an upstream promotion.

The real Rust consuming processor optionally captures a clone at its **next
input read** boundary. Its snapshot contains the existing `SemanticSession`
(parser/KMP state and incremental decoder), **actual `StashedChunks`**, protocol
chunk generator and completion usage. Python's processor mutex serializes
capture, push/drain, EOF preview and close. No output history is decoded again;
no async stream is replayed; no target/model/cache operation occurs in preview.

`StreamProcessor.preview_eof_tokens(ids)` forks that consuming snapshot, feeds
the exact IDs through the existing `SemanticSession.token` and
`StashedChunks.apply_actions`, and clones the resulting state for Stop/EOF at
**every candidate prefix**. EOF uses the original parser's `finish`, original
stashing and original protocol generation. The consuming and preview paths
share `canonical_finish_reason`. EOF forks cannot affect later candidate rows,
the canonical parser, generator, response IDs/index, usage or input stream.

`ChatCompletionResponse.fork()` clones the native canonical accumulator.
`ToolEOFPreview` applies native token events to that fork, forks it again for
native EOF events, and invokes **unchanged M11 event/response predicates**.
There is no alternate DSML grammar, JSON completeness parser or approximate
terminal detector. The Python M11 predicate is the application authority for
interpreting the native protocol projection, not a second consuming parser.
Both M11's consuming JSON loop and `LiveRecipeTurn` use this same decision.
The historical full-prefix `_probe_tool_calls_complete` is used for parity
regression/legacy non-Chat bindings, never the development MTP Chat path.

The old raw `preview_tokens` continues to describe lexical DSML block end and
stop spans; it is **not** permission to cross an M11 EOF tool boundary. M54
intersects that permission with the earlier EOF decision. The completing input
is a protected canonical step; unauthorized proposals never enter target
verification. At completion, control goes to the existing application publisher.
No preview/proposal/generation method authorizes or executes effects.

Native EOF projection is currently Chat Completions only. Other protocols do
not acquire a new MTP capability. OFF remains default; development selection is
still the explicit strategy on the same standard backend/server.

## Exact provenance

The frozen `EOFProvenance` binds:

- the actual canonical processor object and monotonic native input revision;
- emitted canonical ordinal, with generation identity/frontier/pending anchor
  also bound by `SemanticAuthorization`;
- exact native accumulated-response serialization and last four canonical
  protocol events (M11's existing gate window);
- the complete requested candidate-ID tuple, selected observed-prefix count,
  explicit Stop/EOF inclusion and `M11_TOOL_EOF`/ordinary continuation decision.

The revision identifies a worker-local canonical state exactly; the old
64-bit diagnostic fingerprint is **not** used as a collision-free identity.
The state includes the immutable options/tokenizer and consuming generator via
processor ownership. A snapshot/projection/row-count mutation or inexact decoder
map rejects. Foreign processor, wrong prefix, wrong ordinal, changed response
projection, revision drift and finished state reject before target mutation.
Each authorization is recomputed; no preview cache or cross-prefix permission.

For an EOF protected step, whole-prefix M8 adoption and canonical recipe feed
precede delivery; a fresh native current-prefix EOF projection must agree with
the authorized ordinal. Any disagreement burns the lease/derived producer.
The actual agreement is recorded in `last_cycle_metrics.tool_eof_agreement`.

## Shared application publication

The first real JSON tool trial exposed an integration defect: JSON published a
turn/reserved bytes without SSE's reconstruction certificate or request ID.
Generation and retry worked, but the standard Web effect authority could not
admit this outcome. This was repaired, not bypassed:

`recipe_publication.publish_recipe_turn` is the extracted **existing SSE
publisher**, now called by JSON and SSE on the protected worker. It computes
the same existing reconstruction certificate against canonical history,
publishes one request identity/count and closes non-representable re-entry.
Only then may the existing response reservation serialize/freeze bytes. JSON
and SSE direct entry require the exact reserved body; fenced OFF as well as MTP
requires an active reservation. Publication failure burns retry/re-entry.

The Web request-count/request-ID effect ledger remains unchanged. Preview
cannot create a ledger entry or publish an effect. Replayed model responses and
replayed effect outcomes cannot execute a tool again. The probe's weather tool
is an actual registry effect with an observable append-only file, not fake model
generation. Web effect-response delivery loss is injected *after* the actual
application handler/ledger execution and reconciled through that same ledger.

## The next primitive discovered and implemented: atomic worker delivery

After the nominal tool/effect/socket matrix passed, an additional cancellation
audit found that `pending.extend(await worker(cursor.advance))` could lose a
successfully consumed recipe batch when the awaited future was cancelled. A
second race could rewrite already established tool EOF into Length if transport
cancellation became visible before `LiveRecipeTurn.finish`.

Both were repaired without another transcript/reservation/ledger:

- `StatefulStream.advance_worker` / `finish_worker` publish the complete batch
  into the **existing sole bounded pending deque on the protected worker**, before
  fulfilling its future. Async cancellation cannot discard its returned events.
- A known canonical tool/stop/length boundary wins over a concurrently arriving
  transport cancellation. Transport loss is not a new consuming-recipe decision.
- Failure before a complete batch still burns the uncertain reservation and
  retires the generation; no partially observed prefix becomes a completed turn.

The final real-checkpoint probe deliberately blocks at the native **consuming
EOF decision**, disconnects the socket, signals the standard cancellation route
while the worker is blocked, then releases the protected operation. It verifies
that frozen/retried SSE contains **every tool argument delta and tool_calls
finish**, exactly reconstructs the canonical JSON arguments, retains semantic
completion, executes one existing-ledger effect and continues with its result.
This is not a test of merely matching a prefix hash or a `[DONE]` sentinel.

## Qualification evidence

`artifacts/m54-eof/` contains native source/binary provenance, canonical parity,
regressions, standard socket receipts, matched correctness and measurement logs.

- Previous-to-modified Rust consuming protocol corpus: **22/22 exact**.
- Actual recipe/tokenizer EOF tests: full response JSON and SSE-event byte
  correspondence, Unicode, incomplete calls, braces in strings, empty invocation,
  ordinary text, every prefix, stale/foreign/projection-mismatched permissions.
- Fixture earliest EOF ordinal **31**, raw lexical block end **37** (0-based).
  Both new preview and consuming M11 select **31**. On the actual forced-tool
  checkpoint workload, both select **27** (28 consumed inputs), F=321 / F=423.
- Final affected regressions: **266 passed + 8 subtests**, 43.76 s.
- Official checkpoint MTP socket/application matrix: **17 turns**, **26 M52
  cycles**, **243 consumed inputs**, **260 target forwards = 17 bootstraps + 243
  planned decode inputs**; forbidden donor/diagnostic execution **0**.
- Tool JSON, SSE, completion-before-effect disconnect, effect-response loss,
  in-worker EOF/result-loss race, exact retry, tool-result submission/continuation,
  repeated assistant generation and a second weather tool cycle pass. Existing
  Web ledger executes **5 effects; duplicate effects 0**.
- Every response exact retry is byte-identical; same-sequence whitespace body
  changes and stale sequence/body retries reject without target/count mutation.
  Duplicate tool result under its original sequence replays; resubmission as a
  new generation rejects exact-prefix admission before mutation.
- Tool pre-completion disconnect closes the non-representable frontier and denies
  effect/re-entry, while its frozen response remains exactly retryable. It is
  **not** a certificate licensing speculative continuation of incomplete tools.
- Ordinary disconnect → exact retry → representable canonical re-entry → nine
  new generated inputs passes. New tool-result generations also pass. Short
  re-entry rings use explicit canonical warm-up, not history reconstruction or
  an exception-driven strategy switch; short turns may have **zero speculative
  cycles**. This is safe re-entry, not evidence of speculative continuation speedup.
- Executable idle states have all **40** offsets equal canonical history/frontier;
  the incomplete tool cancellation deliberately retires its cache, not an idle
  40-slot witness. Replay / full-cache repack / hidden target re-execution =
  **0 / 0 / 0**. Greedy application scope only; existing canonical stochastic RNG
  and failure/burn matrices pass in reduced M51–M53 regression coverage.
- Session DELETE retires frozen request/response payloads and existing Web effect
  entries. Final MTP parent/child inactive; receipts/producers **0/0**.
- OFF control passes the nominal matrix; **11 matched completed turns** have
  exact generated tokens, messages (excluding UUID/time) and frontiers. Different
  disconnect drain endpoints are not matched workloads; subsequent ordinary
  generated tokens/messages agree. The final worker race has additional actual
  MTP evidence and common-path regressions, not another OFF socket race trial.

## Full R1 — exactly one run

After the nominal operational and matched correctness PASS, the unchanged full
R1 command was run once (before the later matched performance runs): `--profile both --real-model`. **Overall FAIL:** 24/25
gates pass, including all shared owned checks/protocol/preview/recovery and real
OFF protocols/tool/re-entry/persistence regression. `mtp-singleton-v1` refuses
model execution at its existing Python/MLX identity gate: this environment is
Python **3.13.14**, whereas that historical public profile requires **3.13.15**
(MLX/package versions otherwise match). No contract/expected fixture was weakened.

The worker-batch/terminal race repair was finalized **after** this run, and is
qualified by the final 266-test affected matrix and official checkpoint race
probe, not a repeated full R1. Thus there is **no final-source full R1 PASS**.
Its inherited OFF persistence test is not a new first-party persistence capability.

## Performance and promotion boundary

`matched-performance.json` verifies the **same exact HTTP request bytes**, greedy
settings, generated **41 inputs**, final message/stop and **F=283** across all
three lanes. One warm-up is excluded; each result aggregates three sequential
fresh-session samples, no concurrent models. This is a fresh, tool-free ordinary
HTTP workload, not the tool/cancellation matrix or an unqualified continuation
benchmark. Native widths remain each implementation's existing setting.

| lane | integrated recipe decode tok/s | full HTTP request tok/s | offered / accepted (3 samples) |
|---|---:|---:|---:|
| OFF | **15.88** | **11.11** | n/a |
| first-party MTP development | **10.62** | **8.17** | 102 / 96 (**94.12%**) |
| existing candidate (internal development backend) | **31.79** | **14.94** | 90 / 87 (**96.67%**) |

Decode is the integrated recipe-to-idle/settled protocol interval, not just M53
verification. The separate full-request number includes actual prefill, handoff,
HTTP routing and final delivery. Model load is outside both rates. OFF normal
measurements remain valid because the subsequent repair changed SSE delivery and
MTP cancellation precedence, not the measured OFF JSON path. The pre-race MTP,
missing-candidate-dependency and candidate cleanup-harness attempts are excluded.
The latter two were setup/cleanup repairs only: install existing locked JSON
schema dependencies and close the historical executor on its caller, not itself.
No candidate arithmetic or implementation was changed for these measurements.

Mean inclusive observed host-wall phase seconds per 41-input request:

| phase | OFF | first-party MTP | candidate |
|---|---:|---:|---:|
| integrated decode | 2.582 | 3.862 | 1.290 |
| full HTTP latency | 3.690 | 5.016 | 2.744 |
| prefill/handoff receipt | 0.378 | 0.831 | 1.451 |
| proposal | — | 0.176 | 0.0422 |
| target verify | — | **3.229** | **0.996** |
| protected canonical target+sampling | 2.577 | 0.150 | inside candidate step |
| M51 journal setup / settlement | — | **0.0566 / 0.0474** | not M51 |
| semantic authorization/preview | — | 0.000440 | 0.000514 |
| recipe report | inside normal loop | 0.00143 | 0.000590 |
| M8 accounting | inside normal loop | 0.00103 | inside settled report |
| response/application worker publication + freezing | 0.0000194 | 0.0000185 | settled report includes it |
| candidate settle/report / native cache operations | — | — | 0.0669 / 0.00582 |

These **overlap and must not be summed**. Target-forward wall (OFF 1.317 s,
first-party 3.311 s) includes bootstrap and overlaps verification; it is not a
pure GPU-kernel timer. OFF target execution plus canonical sampling is measured
by its whole protected step. Final worker publication/freezing is not all
HTTP/transport overhead. The profiler observes original function frames and
existing timers without wrapping/re-running arithmetic or adding synchronization.
Tools-enabled EOF preview/probe timings are retained in operational receipts,
not claimed equal to this ordinary-text semantic preview phase.

No speedup claim: first-party integrated decode is ~**33.1% slower than OFF**,
and the existing candidate is faster on this matched workload. Verification is
the dominant measured first-party phase; reporting/semantic preview is tiny.
We found and repaired correctness/delivery integration gaps, not duplicated
numerical target execution or another avoidable performance loop. No core,
kernel, batching, repack or zero-copy redesign was attempted to change the result.
M53's **13.81 tok/s remains core-only evidence**, not this operational rate. No kernel/MMA/prefill/zero-copy research, concurrency,
long context, Vision, persistence expansion, defaults, candidate removal or
runtime promotion occurred. Remaining promotion gates include full R1 in its
qualified environment on final source, practical matched speed, and broader
sampling/application/capability qualification. Development scope remains text
Chat, greedy, explicit reasoning none, one session, <=64 output and <=512
prompt+output reservation. The native dependency patch is not an approved
upstream/release payload.
