# M29 — stateful DSpark/MTP lifecycle integration

## Decision: PROTOCOL_BLOCKED

M29 promotes the M28 canonical-quiescence algorithm into a narrow internal runtime
primitive and adds an opt-in/native-MTP session substrate that is separate from the
qualified `OMLXGenerationSession` MTP-OFF path.  Production MTP remains **OFF** and
no public serving option is enabled.

The lifecycle target/DSpark ownership model is implemented, but actual DeepSeek
recipe tool/parser termination remains unsafe for bounded canonical drain unless
represented in the upstream token matcher before target commit.  Therefore M29
ends `PROTOCOL_BLOCKED`, not operational-ready.

## Implemented runtime seams

- `ds41f_mlx.runtime.mtp_lifecycle.NativeDSparkPriming` is the P7/P6 sidecar seam.
  It reads `language_model._config.dspark_target_layer_ids` and calls the native
  `return_dspark_hidden=True` / `dspark_append_context(...)` contract instead of
  reimplementing DSpark projection math.
- `DSparkCommittedContext` records bounded native DSpark ring ownership and
  validates all ring offsets against the canonical frontier.
- `CanonicalTransportHistory` stores `canonical_generated_tokens` separately from
  `transport_delivered_tokens`; `recovery_suffix_tokens` are canonical but not
  claimed delivered on an interrupted transport.
- `canonical_quiesce_native_singleton(...)` implements the M28 frontier relation:
  for target-ahead state it requires `queue_length == needed + 1`, commits exactly
  the first `needed` queued tokens to canonical history, and discards the final
  unforwarded future token.  For the one-token target-behind case it performs only
  the bounded native target forward and DSpark append.
- `OMLXMTPGenerationSession` is an internal MTP-ON class.  The existing
  `OMLXGenerationSession` still rejects speculation and remains the qualified
  MTP-OFF implementation.

## Idle ownership invariants

At a successful internal idle boundary:

```text
all 40 target cache offsets == len(canonical_history)
all DSpark ring offsets     == len(canonical_history)
queue                       == empty
rollback/draft/uid state    == absent/cleared
```

Quiescence counters explicitly require:

```text
new verify cycles = 0
new proposals     = 0
history replay    = 0
full-cache repack = 0
```

## P7/P6 priming continuity status

The narrow implementation seam is present and opt-in.  It is designed to harvest
native hidden taps from already-required target execution and append them through
upstream-owned DSpark logic.  The MTP-OFF prefill and P6 handoff paths do not
acquire DSpark ownership by default.

Full real-model P7-vs-stock numerical equivalence and repeated P6/MTP throughput
qualification are not promoted because the protocol gate below blocks lifecycle
agent-ready status.

## Recipe/protocol gate

M28 qualified length and token-matcher stop behavior.  M29 keeps the same line:
EOS or known stop tokens are MTP-safe only when represented in the upstream token
matcher before target commit.  Tool/DSML/parser-only boundaries are **not** MTP
safe in this milestone because they can be detected after text has crossed a
semantic boundary; canonical drain could then commit hidden protocol text.  Such
requests must use MTP-OFF.

MTP-safe modes for this commit:

- internal diagnostic text generation with no parser-only boundary;
- length termination;
- EOS/stop-token sequences registered with the upstream matcher before target
  verification.

MTP-ineligible modes:

- tool-call/tool-result loops whose boundary is parser-only;
- arbitrary user stop strings outside the upstream token matcher;
- requests requiring token-exact abort at the last transport-delivered token.

## Cancellation semantics

Stateful MTP cancellation means bounded canonical quiescence.  It does **not**
mean stop exactly at the final transport-delivered token.  A timeout may add up
to the bounded M28 suffix to canonical server history; clients recover that exact
suffix from `recovery_suffix_tokens` on the next interaction.

## Evidence

Machine-readable status is in `artifacts/m29/qualification.json`.  Structural
regressions include a fake-native unit suite for canonical-vs-delivered history,
cache-ahead drain, target-behind materialization, fail-closed topology handling,
and DSpark ring offset validation.

The historical M25-M28 artifacts are unchanged.  Persistence remains fail-closed;
M29 does not define a restart-serializable MTP idle representation.

## Recommended next milestone

Resolve the recipe/tool protocol gate by proving a pre-target-commit token clamp
for actual DeepSeek tool/DSML termination or by keeping those modes permanently
MTP-ineligible, then run full real-session repeated P7/P6/MTP/quiesce/re-entry
qualification and public serving promotion separately.
