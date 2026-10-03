# M33 — native semantic commit horizon qualification

## Decision

**PROTOCOL_GATE_SOLVED** for the isolated, fully primed native singleton runtime.
The recipe binding is **FULL_BINDING_QUALIFIED**, not a protocol-only build.
**M34 operational qualification is authorized.** No operational soak or public
serving promotion is claimed here.

Production MTP remains **OFF**; the public option remains disabled; MTP
persistence remains fail closed; token-exact immediate abort remains unsupported.
M34 must use the qualified candidate/runtime identities and test repeated
sessions without weakening these policies. Concurrent/shared MTP admission is
rejected, not promoted by this result.

The [design written before implementation](milestone-33-semantic-horizon.md)
selects the strong unforwarded-terminal rule. It was not weakened to an
initialization-only atomic alternative.

## Authorities and build scope

- ds41f base: `fff84bdd1acd814ae84b994bec4ed110f31f4bb1`.
- Recipe base: `8cadfede7063c896b944e7bae05daa3549ae97ea`.
- Real preview binding/source: `29dabb5a55b7b2c6a68e18bbb3eb14495623e81a`.
- oMLX base: `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`.
- Isolated oMLX candidate: `fbe18e8fe68e5bb7b9b1971652ed330f752b6afc`.
- Patch: `artifacts/m33/omlx-semantic-horizon.patch`; its SHA256 and all four
  modified upstream file hashes are in `runtime-identities.json`.
- Full ARM64 recipe module SHA256:
  `454413afdcee2916795e1c5f7ce1b94a8346e76bffd6b45024f28f69f8d0a73f`.

The native recipe module is the genuine M32 standard OpenCV 4.14.0 build with
Rust/Python tokenizers 0.23.2, not a borrowed release binary or a `DOCS_RS` build.
It remains host-linked; portable-wheel distribution qualification is not claimed.
No recipe parser, protocol generator or tokenizer behavior was changed for M33.

Model qualification uses the pinned model's stock MLX/source-JIT kernels in the
isolated candidate; it does not borrow the release C++ DSA extension. The
preserved release native-array ABI is tested separately in its proper namespace.
All 16 M20 release native artifacts, the release recipe module, the release source
checkout, and M25–M32 evidence remain unchanged. This is not qualification of a
new compiled C++ model-kernel variant.

The final ds41f result commit is resolved by
`git log -1 --format=%H -- artifacts/m33/qualification.json` and reported in the
completion receipt; embedding a commit's own SHA in its contents is impossible.

## Installed horizon and ownership

One optional generic callback serves initialization, chain verification, and
alignment materialization. oMLX contains no DSML grammar or independent matching
of arbitrary recipe stop strings. The pinned native processor is the only
semantic authority; previews clone its current incremental state, not its input
history or protocol generators.

1. **Initialization:** prove existing full prompt priming before forwarding main.
   A terminal main transfers those rings and queues only main: no main forward,
   successor sampling or drafts. A safe main uses upstream math. Previewing
   `[main, successor]` before proposals can queue a terminal successor while
   appending only main's safe target taps. No-boundary initialization retains
   upstream sampling, priming, draft construction, buffers and queue order.
2. **Chain:** after existing acceptance/layout/alignment/length/matcher limits,
   preview exactly `draft_ids[:m] + [emit_last_id]`, prefixed only by outstanding
   safe queued responses. A semantic cut reuses native model clamp/rollback;
   changed candidates are previewed again. Processor budgets and target/DSpark
   state use the final `m`. Terminal cycles append safe hidden taps without next
   drafts and without boundary-forwarding the terminal.
3. **Alignment:** prove the queued prefix before forwarding its last safe entry;
   preview its sampled successor before new proposals. This covers a real
   additional commit seam, not just the main verify-cycle seam.
4. **Emission:** a bounded sidecar binds UID, absolute response ordinal, token,
   native kind/span and candidate provenance. Earlier queue pops and repeated IDs
   cannot transfer ownership. Canonical native observation must match exactly;
   mismatch/inexactness is a hard failure, never history-based MTP fallback.
5. **Finish/idle:** completed DSML is finished through the authoritative recipe
   backend Stop path, producing `tool_calls`. Target and all DSpark rings remain
   strictly before the completing token until canonical emission. Existing M29
   bounded quiescence then forwards exactly that canonical token, without a
   successor sample, new verify, new proposal, history replay or full-cache
   repack. Every final target/ring frontier is checked against canonical history.

Backend EOS controls are not recipe text. Preview and canonical feed mirror the
qualified OFF backend's suppression policy; a user stop substring of a control's
printable spelling cannot create a false prediction. EOS IDs are backend control
metadata, never tokenizations of arbitrary user stops. Their native matchers are
installed explicitly. This preserves ordinary output and usage accounting.

An RLock serializes start/generation/canonical observation/cancellation/delivery
acknowledgement. Init/chain helpers have no await or response yield inside their
critical mutation phases. Injected phase exceptions poison the session: no resume
or quiescence certificate is returned. Old engines cannot silently ignore the
attached guard. The M29 mutable prompt-history alias exposed by actual re-entry
was fixed by passing a list copy at native insertion.

## Qualification matrix

| Evidence | Result and scope |
| --- | --- |
| `canonical-parity.json` | 64/64 exact native comparisons, 22 M31 cases, against unmodified pinned recipe; token/text/character inputs, events, finish and usage preserved |
| `native-preview.json` | 77/77 source rows agree; 15,400 repeated real-binding previews deterministic and non-mutating, including fingerprints/pending IDs |
| `rust-tests.log` | 10 real Rust semantic/decoder tests pass |
| `regressions.log` | 139 pass, 24 subtests; includes 29 new horizon fixtures, native bindings, M25/M29, P5/P6/P7, OFF/provenance and M30–M32 evidence |
| `release-off-runtime-regressions.log` | 5 pass in preserved release namespace, including actual native-array ABI/packing; one Pydantic deprecation warning |
| `evidence-tests.log` | 6 additional gate/identity/matrix/manifest checks pass |
| `init-live.json` | Actual sampled main `Hello` and successor ` there` string stops; ordinary text OFF/ON, correct backend EOS suppression |
| `tools-live.json` | Fresh OFF/ON one call, reasoning→call and two-call block; names/JSON arguments/tool counts and `tool_calls` finish agree |
| `stops-live.json` | Multi-token/shared-prefix, café-adjacent split-ID `🙂`, cross-cycle `HALT NOW`, and real aligned `!` completion |
| `interruptions.json` | 13 exact native idle transitions and 3 deliberate internal-phase fail-closed cases |
| `tool-reentry.json` | Real P7-enabled P6→P5 guarded call→quiescence→local result→P6 suffix→P5 final assistant→quiescence |
| `performance.json` | Paired M26-shape performance remains materially intact, same no-boundary ON tokens and acceptance |

The 29 controlled Python fixtures use the **real native recipe/MLX bindings and
actual oMLX functions with synthetic target fixtures**; they are not falsely
labelled checkpoint generations. They cover accepted-prefix terminal, full-accept
bonus terminal, partial-accept correction, rejection before proposed terminal,
zero-depth escape, repeated IDs, shorter layout re-clamping, immediate budget
priority, ownership/mapping disagreement, control suppression and serialization.
Actual checkpoint cycles additionally exercise full and partial native rollback,
including a zero-safe-prefix semantic cut.

Pending Unicode main/successor and prior stop-prefix initialization are explicitly
labelled forced-input fixtures through **real checkpoint target caches/DSpark**,
not sampler-distribution tests. Alignment-successor fixtures use the real native
materialization helper on a naturally sampled safe init queue. They prove the
canonical+queued-prefix fork without canonical replay.

Cross-cycle closure is observed, not inferred: the two-tool block's closing span
starts at byte 331 while its completing preview starts from canonical byte 343.
The cross-cycle recipe stop starts at 23 with canonical byte 27. Unicode completion
is ID 227 with exact stop span `[5,9)`. Each actual emission matches an independent
processor fed only canonical emitted inputs; verification leaves parser state
unchanged.

Interruption points include before init, first safe emission, a predicted
successor before emission, safe undelivered target prefix, inside DSML, before its
closure, after terminal emission, alignment prediction, and an already canonical
terminal not yet delivered. Un-emitted predicted terminals are discarded, never
drained. The last case repairs one canonical terminal and later acknowledges its
recovery suffix using ordinal-owned **metadata-only** delivery confirmation; no
model/parser work occurs during acknowledgement.

Tool re-entry preserves exact recipe prefix-extension admission. Both suffixes
use ordinary P6 ranges with same-forward layer-input taps; each records 174 P7
scheduling events. All target/ring frontiers finish at 327 and then 373. Re-entry
instrumentation forbids reconciliation/replay, native target-cache reconstruction
and full repack; observed replay/repack counts are **zero**, not inferred from
static source. Both quiescences add zero verify/proposal calls. No global or public
MTP selector was added: the P5 factory is an internal explicit opt-in seam.

## Practical cost and remaining scope

One warmup plus a paired 4096-token/128-output benchmark (same M26 prompt, seed and
sampler) measures:

- OFF **19.37 tok/s**; ON no guard **42.98**; guarded ON **42.84** (ratio **0.9966**).
- Native reported considered-draft acceptance **94.5%**, **4.478** accepted/cycle,
  identical ON no-guard/guarded tokens and acceptance. This is the same upstream
  counter metric as M26, not a new all-offered-draft acceptance definition.
- Guarded first response **67.7 ms** versus **67.4 ms** without guard.
- 25 live preview calls: two init, one per each of 23 verification cycles.
  Live median/p95/max **14.46/17.92/18.46 µs**; no hot-path fingerprint diagnostics.
- Repeated native-binding corpus median/p95/max **1.625/1.875/7.084 µs**.
  Same-core release Rust clone median/p95/max **84/167/7417 ns** (black-boxed;
  15,400 samples). The Python ABI has no separate clone timer; its inclusive
  native preview and this exact-core source clone are reported separately.

`metadata-overhead.json` measures reconstituted actual owner object graphs and
serialized sizes, excluding existing model/parser/output/cache objects: maximum
owner graph **1868 bytes**, serialized owner **276 bytes**, actual candidate IDs
at most six. There is
one current terminal owner, at most 16 retained native decisions, 64 optional
preview rows and 4096 bounded timing samples; no forecast attaches to an unrelated
request and no parser/whole-cache history is copied. This is performance evidence,
not a serving throughput promise or broad soak.

Known failed attempts are preserved and **excluded** from the final gate: omitted
EOS matcher/fixture stop not induced; the history-alias re-entry failure; a
supervisor timeout during benchmark loading; combined-test namespace mismatch
(the candidate intentionally lacks the release native DSA extension); and new
fixture/mock/config/name mistakes. In particular the missing guarded backend
matcher was corrected in the real internal session, not hidden by claiming an
EOS spelling was a recipe stop. Final live evidence uses the official control
suppression policy and the exact clean candidate above.

Remaining work is operational M34: repeated bounded singleton sessions/tool turns,
longer streams and interruption patterns using these identities. Portable wheels,
shared/concurrent serving, public MTP, MTP persistence and immediate token-exact
abort are not authorized. There is no remaining initialization/semantic-clamp/
emission/no-replay-quiescence architecture blocker in the qualified scope.
