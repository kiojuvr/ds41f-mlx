# M33 semantic commit horizon — design before implementation

Authority: ds41f fff84bdd1acd814ae84b994bec4ed110f31f4bb1;
recipe base 8cadfede7063c896b944e7bae05daa3549ae97ea;
recipe candidate 29dabb5a55b7b2c6a68e18bbb3eb14495623e81a;
oMLX base 4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40.

## Selected invariant (strong, no atomic weakening)

A predicted terminal-completing response is the final **unforwarded** response.
Before its canonical emission target and DSpark may commit only responses strictly
before it. After emission, exactly that now-canonical token may be materialized by
the existing one-token quiescence repair. No successor is sampled by that repair;
no verify, proposal, model-history replay or full-cache repack is allowed.

## State and horizons

- P: canonical recipe token-input ordinal for this turn; decoder-pending IDs count
  as canonical inputs even when source bytes have not yet been decoded.
- H: model history frontier, including prompt and canonically emitted responses.
  P = H - prompt length (plus any explicitly supplied canonical response prefix),
  after applying the official backend's protocol-input projection: its terminal
  control IDs are not recipe text. Decoder-pending protocol IDs still count.
- D: transport-delivered frontier, D <= H. Recovery suffix is history[D:H].
- T: all 40 aligned executable target cache frontiers.
- R: all aligned DSpark committed-context frontiers; not a second target cache.
- Q: bounded ordered future responses (token, distribution, source). Ownership is
  an absolute response ordinal, NOT token identity. Repeated token IDs are legal.
- B: optional earliest prediction (absolute response ordinal, exact token,
  terminal kind/source span, candidate origin and safe-prefix bound).

At a quiesce-capable external boundary T = R. In ordinary active operation:
T >= H implies Q consists of T-H safe committed responses followed by one
unforwarded future response. T = H-1 implies Q is empty and exactly the last
canonical token may be repaired. Idle requires H = T = R and empty Q.

With prediction B at absolute ordinal b, T/R <= prompt+b before emission.
Q must end at b, and all Q entries before b are semantically safe. P advances
only for canonical emission (including bounded safe undelivered recovery), never
for preview or rejected tokens. No verify/proposal admission with outstanding B.
After canonical emission at b: prediction must match actual parser observation;
no later response is allowed. H=prompt+b+1, T=R=H-1, Q empty, until one-token
repair establishes idle H=T=R. DSML canonical protocol finishes with tool_calls;
recipe stop with stop. The backend uses stop to finish the recipe protocol path.

## One generic semantic horizon

The optional session-owned guard previews canonical state + safe queued responses
+ candidate IDs in eventual order. It returns an exact safe prefix/prediction.
The oMLX code knows no DSML/stop grammar. A queue-ordinal sidecar binds prediction
to its eventual emission; pop of earlier responses cannot change its identity.
Inexact mapping or observation disagreement is a hard error, never MTP fallback
into the upstream history-rebuilding reconciliation path.

Initialization first previews main. Terminal main bypasses forward, successor
sampling and draft construction; prompt DSpark priming is transferred at its
already committed frontier. Safe main follows the original forward/sampler math.
Preview of [main, successor] uses still-canonical state. Terminal successor skips
proposals and appends only safe main's target taps. Nonterminal init preserves
upstream priming, sampling, draft math and queue order.

Chain preview follows existing acceptance/rollback/alignment/length/matcher
narrowing and precedes processor restoration, target rollback/commit and DSpark
append. Only draft[:m]+correction is a candidate. Re-narrowing reuses upstream
model clamp/rollback; the final candidate must be previewed again if changed.
Terminal cycles append safe target hidden taps but do not generate next drafts
or materialize an alignment terminal before emission.

## Atomicity and interruptions to prove, not assume

`BatchGenerator.next` and native initialization are synchronous, contain no await
or response yield between forward/sample/state construction and emission. The
session must serialize them with canonical parser operations and cancellation.
A timeout/cancel checked outside next can recover only a returned response boundary.
Exceptions or cross-thread callback interruption inside init must fail closed;
a half-constructed state must never be advertised as an idle/recoverable session.
This is not permission to forward a terminal internally: the strong rule remains.
Before activation, prompt priming plus standard pending main are an explicit state;
recovery must adopt committed priming and discard pending main without forwarding.
After constructed init or chain, bounded safe Q drain must feed the canonical
recipe processor and may never reach predicted B. If B is discarded as the sole
unforwarded future, prediction is retired, not reported as canonical terminal.
The implementation/tests must establish these properties before any solved claim.

## Qualification policy

Production OFF, public option disabled, MTP persistence fail closed, immediate
abort unsupported. No serving promotion or broad operational soak. This design
file is not itself qualification evidence; outcome is recorded separately in
[M33 protocol qualification](milestone-33-protocol-qualification.md).

## Implementation refinements within the selected invariant

Guarded activation first proves full committed prompt priming. Missing context,
old engines lacking semantic-horizon capability, and shared-batch handoffs fail
closed; they never enter history reconstruction. Ownership includes generation
UID as well as absolute response ordinal.

Alignment has its own pre-commit path: prove the queued safe prefix before its
last token is forwarded, then preview the newly sampled successor before any
new proposals. A completing recipe token is never boundary-materialized before
emission. The existing backend matcher also prevents guarded EOS boundary
materialization; EOS suppression is mirrored in native recipe preview and
canonical observation, not matched by a second recipe grammar.

Canonical emission and transport delivery are distinct. Internal callers can
use `next_token(transport_delivered=False)` and subsequently acknowledge the
exact canonical prefix with `confirm_delivery(..., start_ordinal=...)`. Delivery
acknowledgement is metadata only. A predicted but un-emitted terminal cannot be
drained; an already canonically observed terminal may be part of an undelivered
recovery suffix, never followed by a later response. Session operations hold an
RLock; an internal phase exception poisons the session rather than publishing a
half-built idle state. Quiescence synchronizes the owned generation stream.
