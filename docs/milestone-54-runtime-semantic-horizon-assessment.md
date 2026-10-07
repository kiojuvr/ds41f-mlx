# M54 — canonical runtime / application semantic-horizon assessment

**NO / M54 NOT PASS.** Base: M53 execution-core PASS `bf8305d`, M52
`6d7c936`, M51 `40bd187`. This is a blocked integration assessment, not a
claim that first-party operational MTP is impossible. This is the historical
assessment. [M54 continuation](milestone-54-semantic-authorized-continuation.md)
implements both primitives and partially qualifies standard socket execution;
its next EOF-sensitive tool boundary remains NOT PASS. In this assessment, no
operational selector was enabled. M51–M53 are not redesigned or demoted. No promotion decision can
be inferred from this assessment.

## Actual application architecture

The existing standard surface is already suitable as the single application
boundary; a new server or transcript is unnecessary:

```text
Web / Chat canonical application boundary
  -> standard serving.server.create_app
  -> DeepSeekRecipeRuntimeBackend + StatefulSessionRecord
  -> StatefulStream (reservation, transport, cancellation)
  -> LiveRecipeTurn (canonical recipe processor / response publication)
  -> M11RecipeToolSession / M8LiveContinuationSession
  -> TargetGenerationSession (generation / sampler / live cache lease)
  -> M51 target transactions
```

Non-streaming stateful Chat uses `run_stateful_chat_turn` and
`M11RecipeToolSession.run_current_assistant_turn`; SSE uses `LiveRecipeTurn`.
Both currently consume `M8.next_token`, which consumes `generation.next_token`,
then updates M8 history/accounting. M11 and LiveRecipeTurn feed the recipe only
**after** this single-token commit, then detect tool completion using the
existing canonical-parser EOF probe. The probe reparses protocol tokens, not
model history; it is not target replay. It is also not speculative authorization.

The desired additive branch is an execution strategy **inside this chain**:
first-party proposal child -> DSpark producer -> M52 -> M51, returning a bounded
canonical report batch to the same M8/recipe owners. App code must not choose
acceptance, run a target, maintain proposal rings or reconstruct target history.

Current selection stays unchanged: `standard-off` default/control, and the
separate historical `mtp-singleton-v1` candidate. The candidate is neither
removed nor reused. An unknown `first-party-mtp-development` profile rejects
before backend allocation. This rejection is not a new supported selector.
A future explicit strategy should be fixed for a resource/session lifetime,
with OFF explicitly selectable and unsupported capabilities rejected before
reservation, image preparation, load, prefill or append. There must be no
exception-driven mid-request switch to OFF.

## Executable integration counterexamples

`tests/test_m54_semantic_boundary_assessment.py` runs real MLX against the
**reduced M52 target/state fixture**, not official checkpoint target math and
not DSpark proposal arithmetic:

- An unrestricted cycle emits/commits `[11,12,13,14]`. An application semantic
  boundary represented by token 12 requires `[11,12]`. Calling stop after the
  cycle preserves the committed tail; all target frontiers match that longer
  history. Tool/recipe-stop/response labels in this test are illustrative,
  **not real tool or stop workload qualification**.
- A naive queue replacing `generation.next_token` with cycle reports lets M8
  observe `[11,12]` while generation/cache already contain `[11,12,13,14]`.
  `ensure_idle` then adopts the unseen tail, with turn diagnostics still listing
  only two tokens. This breaks application-visible committed-frontier semantics.
- Static backend stop ID 12 correctly truncates M52. This is a positive control,
  not a solution for multi-token stop strings or dynamically parsed tools.
- Actual installed native `deepseek_recipe.StreamProcessor.preview_tokens`,
  using the actual tokenizer, previews `Hello HALT NOW after` with exact mapping,
  detects STOP_SEQUENCE before the tail, and leaves canonical state unchanged.
  Canonical feeding through its completing token observes that same terminal.
  **The preview/parser primitive exists. Do not implement another grammar.**

These are counterexamples to *naive wiring*, not an impossibility proof and not
an observed production corruption. No unsafe adapter was installed.

## Smallest missing runtime/application primitive

A **worker-confined semantic-authorized cycle adapter**, shared by JSON and SSE,
with coherent batch adoption by M8 and canonical recipe observation. It belongs
in runtime/application integration, not a replacement execution core:

1. Bind preview permission to the actual processor, generation identity,
   emitted ordinal, target frontier and remaining output budget. Exact
   non-mutating native preview is required; missing/inexact mapping fails closed.
2. Preview the pending canonical anchor **before proposal arithmetic**. A pending
   semantic terminal forbids proposing/verification. A protected ordinary
   canonical target transaction can materialize the terminal and preserve M52's
   existing consume/sample semantics; no successor proposal is permitted.
3. Only if the anchor is safe may the producer propose. Preview anchor + drafts
   before M52 verification and restrict the submitted span to the authorized
   prefix. Never submit a known terminal-crossing tail. M52 reports contain
   consumed anchor/accepted draft inputs, not an emitted correction/bonus;
   preview cannot turn unconsumed lookahead into application history.
4. Before another cycle, canonically feed every settled report, in order,
   through the *same* processor, checking terminal identity/provenance against
   preview. Adopt the complete settled prefix into M8 accounting coherently;
   buffering reports alone is insufficient. A consumed terminal must be final.
5. Serialize preview, proposal, verify/settlement, M8 adoption and recipe
   observation on the existing worker. Cancellation is checked at coherent
   boundaries. No proposal/verify entry during receipt transport, publication,
   terminal completion, cancellation settlement, reconciliation, idle extraction
   or retirement. Disagreement after target publication is unrecoverable; never
   replay, rollback application effects, repeat publication or continue as OFF.
6. Only the canonical response owner publishes terminal events/tool calls.
   `StatefulStream.publish` retains the existing reconstruction certificate and
   Web's existing tool-effect ledger remains authoritative. Standard SSE's
   application UUID is explicitly correlation only, **not a retry fence**.
   Exact-byte sequence fencing currently lives in `serving/request_fence.py`
   behind the candidate/internal `qualification_response` branch; it is not
   attached to `StatefulSessionRecord`. Its bounded identity/outcome contract
   must be integrated with the standard response reservation (without its native
   scheduler). Retry must observe the same completed reservation, not allocate
   a new model cycle or repeat tool execution. An expected request-count check
   alone is not an exact-byte retry certificate.

This contract permits ordinary assistant spans only where exact native preview
proves safety. Tool execution itself is outside generation. A tool-result turn
must pass existing exact-prefix admission/P6 append and obtain a fresh
generation-scoped proposal epoch through legitimate taps, never ring replay.
EOS/max-token are already M52 terminals but do not authorize recipe effects.
Response completion retires the producer; externally visible committed history
must equal generation/target history before exposure or re-entry. Transport
receipt is not client acknowledgement.

The adapter must also explicitly handle short continuation rings. M53 admits
warm-up through legitimate canonical forwards, but rejects early short-ring
proposals. Scheduling those protected canonical steps is not implicit failure
fallback. Loss/failure of an active producer must fail the selected strategy
closed unless an explicit coherent lifecycle operation is authorized; never
catch an uncertain cycle and retry OFF.

## Qualification and limitations

Command and output: [artifacts/m54](../artifacts/m54/README.md).
**77 passed, 20.54 s, exit 0**: new assessment probes, M52 reduced generation,
M53 reduced producer lifecycle, existing standard stateful delivery/backpressure/
retirement, application effect-boundary and recovery regressions.

The transport/application tests use execution doubles. They exercise OFF
contracts for reservation, streaming delivery, cancellation drain, recovery and
effect fencing, but are **not a fresh official-checkpoint OFF run**. Reduced M52
and M53 tests retain their original labels; no results are relabeled operational
MTP evidence.

| Requested result | M54 result |
|---|---|
| Official-checkpoint standard-path first-party execution | Not enabled / not run |
| Dialogue / multi-turn / JSON / SSE / real tool loop | MTP unqualified |
| Disconnect / cancellation / re-entry / duplicate effects | Existing standard regressions pass; MTP unqualified |
| Target/history/RNG | Reduced core regressions pass; naive adapter counterexample fails app coherence |
| Replay / repack / hidden target re-execution | No new operational measurement; do not inherit M53 counts |
| Receipt / journal / ring / resource retirement | Reduced core and standard retirement regressions pass; combined lifetime unqualified |
| Supported development MTP envelope | None at standard application boundary |

Vision, persistence/restore, Responses/Messages, long context, concurrency and
batching remain unauthorized for first-party operational MTP. No context limit
is established by these tests; the candidate's 8K cap is not copied into a new
specification. Existing OFF capabilities are unchanged, not inherited by MTP.

## Performance / promotion

No matched OFF/first-party/candidate application benchmark was run: the first-
party operational path is blocked before admission. Proposal, target verify,
settlement and app-overhead timings are therefore **N/A for M54**, not zero.
M53's historical 13.81 tok/s (proposal 0.1859 s; verify+settlement 4.4119 s;
receipt/ring 0.0376 s for 64 tokens) remains execution-core-only evidence.
Its combined verify/settlement measurement cannot supply separate M54 phases;
its candidate historical 38–40 tok/s is not a matched comparison.

Full R1 was **not run**: no integrated strategy is ready for qualification.
No kernel/MMA/prefill/zero-copy research, release, packaging, default change,
candidate removal or ds41f-runtime promotion. The immediate blocker is the
semantic-authorized cycle/batch-adoption primitive, standard exact-byte response
reservation fencing, and their combined resource lifetime qualification,
**not a new model execution primitive**. After that,
fresh official-checkpoint Chat/tool/recovery qualification and matched phase
measurements are required before deciding whether performance, capabilities
or further operational blockers prevent promotion.
