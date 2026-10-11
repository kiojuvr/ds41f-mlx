# M30 — recipe semantic preview audit

## Decision: RECIPE_PREVIEW_API_REQUIRED

Base: `e19eb817c33e848237758332c552e52e3461a9f4`.
Authority: clean `deepseek-recipe` revision
`8cadfede7063c896b944e7bae05daa3549ae97ea`.

**Protocol gate has NOT passed. Production MTP remains OFF.** No runtime clamp,
public option, dependency patch, or parser fork has been installed. The qualified
OFF path and M28/M29 quiescence implementation are unchanged. M31 operational
qualification is **not authorized** by this commit.

This is a blocked milestone, not an implementation/qualification success. The
requested fresh model-generated full DSML blocks, ON/OFF protocol parity,
rejection topology, live frontier proof, preview cycle cost, and tool-result
re-entry are not available. Every required live case is explicitly blocked in
`artifacts/m30/qualification.json`, with null frontiers rather than invented
measurements. No broad operational qualification was run.

## Authoritative parser audit

Paths relative to the pinned dependency:

- `deepseek-recipe/src/stream/state_machine.rs`: incremental KMP-based byte
  recognizer and reasoning/JSON/tool/stop stages.
- `deepseek-recipe/src/stream/decoder.rs`: incremental token decoder.
- `deepseek-recipe/src/stream/processor.rs`: source decoding, action stashing,
  protocol event generation and finish-reason mapping.
- `deepseek-recipe-python/src/response.rs`: Python `StreamProcessor` binding.

`ParsingOptions` is cloneable. **Active parser state is not exposed or cloneable
through the binding.** `StateMachine`, `State`, `MatchBranches`, `MatchBranch` and
`MatchState` do not implement Clone; their data naturally could. Merely deriving
Clone there is insufficient for serving: the actual canonical state machine,
`StreamDecoder` pending IDs, stashed actions/chunks, and completion state are
locals suspended inside `StreamProcessor.process()`'s async generator.

The Python object owns a mutex containing an input channel and
`Pin<Box<dyn Stream<...>>>`. `push()` drives that stream until its next pending
input read. Public methods/properties are `push`, `finish`, `finished`, `close`.
There is no snapshot/clone/preview/terminal-position interface. Copy, deepcopy
and pickle fail with TypeError. Copying a fresh parser's options cannot copy the
canonical parser's retained marker prefixes or UTF-8 pending IDs.

A thin adapter cannot extract these locals. A second parser fed canonical tokens
would only mirror state; it still cannot fork that state without history replay.
Destructive lookahead followed by replay/repair violates the requested ownership
contract. Full-output replay into a fresh parser every cycle is not proposed as
runtime policy. The existing M11 full-prefix EOF probe is not promoted.

## Actual tool completion is not `processor.finished`

In `Stage::ToolCalls`, either closing marker
`</｜DSML｜ calls>` or `</｜DSML｜tool_calls>` transitions to `Stage::Finished`
with **OutputAction::Skip**. The opening `<｜DSML｜` prefix does not finish a turn.
Individual invocation/parameter closures do not finish the block. Multiple
invocations must remain possible until the block closes.

There is no public action/event uniquely identifying that transition. Subsequent
bytes generate `SkipInvalid { ContentAfterFinished }`, but discovering that by
feeding a suffix is too late and destructive. Normal `Skip` occurs elsewhere.

The stream processor only breaks immediately for **StopSequence**. It does not
break on DSML Stage::Finished. Python `finished` remains false after a complete
DSML block; backend Stop/EOF must still finish the stream. Backend Stop maps to
ToolCalls when stashing has observed a tool name. EOF without backend Stop maps
to EndOfStream. Backend Length maps to Length. JSON fence completion transitions
to MatchedJson, **not** terminal; an upstream preview must not misclassify it.

Important diagnostic: a closed invocation **without the block closing marker**
already produces `finish_reason=tool_calls` when artificially finished with
backend Stop. Thus the OFF path's `_probe_tool_calls_complete` is an EOF heuristic,
not a detector of the exact DSML terminal transition. Leaving OFF unchanged does
not justify reusing that heuristic for speculative commitment.

`authoritative-source-actions.json` is produced by compiling the unmodified
pinned Rust state-machine file as a path module into a temporary diagnostic
executable. Its complete block accepts no tail, whereas the incomplete block has
not reached Finished. This source-level evidence does not rely on an independently
invented DSML grammar or on binary build provenance.

## Exact token → parser input contract

Serving already sends `InferenceChunk.token(id)` to the recipe processor, with
its recipe tokenizer attached (`serving/server.py`, `runtime/tool_boundary_session.py`).
There is no oMLX detokenizer in this protocol path to substitute.

The pinned StreamDecoder does the following for **each** token:

1. Append ID to pending `ids`; increment pending token count.
2. Decode **all pending IDs**, `skip_special_tokens=false`.
3. If output is empty or ends in U+FFFD, retain IDs and emit no parser text.
4. Otherwise clear IDs, reset pending count, and feed that complete decoded string
   to the state machine as one source chunk.

It does not decode the complete generated history on each push, nor decode every
ID independently. In particular `🙂` spans IDs; independent decoding produces
replacement characters. Markers remain present because special tokens are not
skipped. DSML and stop-prefix KMP states persist between pushes. Stop matching
applies in Common and Json, **not** reasoning, tools, or JSON surrounding text.

M30 records every pending-ID group/flush and compares these observed text chunks
against the token-input parser and character-chunk parser for the diagnostic
cases. This establishes the tested examples, not equivalence for arbitrary
custom tokenizer decoders. The Python tokenizers package is a diagnostic observer,
not a new runtime decoder. A production preview must invoke the **same Rust
StreamDecoder and tokenizer instance/contract**, copying its exact pending state.

A successful flush may contain bytes from multiple IDs, including canonical
pending IDs and newly speculative IDs. A terminal byte can occur inside that
flush. The desired API must report the completing candidate ID **and** the byte
span, including whether start bytes precede this preview. Do not convert a byte
index to an independently tokenized stop string. Ambiguous token-boundary mapping
must reject MTP eligibility, not round forward across a semantic boundary.

## Smallest useful upstream API

Desired Python shape (proposal, NOT implemented):

```python
preview = canonical_processor.preview_tokens(candidate_ids)
# read-only, at the same next-input boundary as push(); no emitted events
preview.terminal_kind       # None | dsml_block_end | stop_sequence
preview.completing_token_index  # zero-based within candidate_ids
preview.terminal_byte_span   # relative stream bytes; can start before this call
preview.safe_token_count    # tokens BEFORE terminal-completing candidate
preview.mapping_exact       # false => fail closed
```

The method must use a fork of the canonical **semantic parser + decoder** state,
not clone protocol IDs/generators/channels. Prefer refactoring the dependency's
existing stream implementation to hold an explicit synchronous incremental
semantic session, called by both the canonical stream and preview. That session
needs cloneable KMP positions, stage/leading flags, stashed size and pending
decoder IDs, and an explicit Finished transition/byte offset from the existing
recognizer. Tracking source-byte/token provenance belongs beside that decoder.
The live canonical session must be observable at Python push's input boundary.
Stop-sequence state and any buffered action bookkeeping needed to reproduce its
terminal decision must be included. Canonical state remains untouched on error.

A snapshot of semantic state plus preview over that snapshot is equivalent; an
API that only clones ParsingOptions, only exposes `finished`, or only performs
EOF parsing is insufficient. This does **not** require snapshotting an async
channel or duplicating the grammar. The necessary extraction of suspended state
belongs upstream; it is not a one-line Python compatibility shim. ds41f currently
carries **no temporary recipe patch**. Release needs an approved pinned upstream
extension with its own parser/decoder tests before enabling this path.

## Proposed pre-commit integration (not installed)

In the pinned oMLX `_chain_verify` in
`omlx/patches/mlx_lm_mtp/batch_generator.py`, the existing flow is:
model acceptance → rollback-capability clamp → commit alignment/model re-clamp →
remaining-length/token-matcher clamp → processor restoration/stats → `finish()`
queue population and target rollback/commit → DSpark drafting/commit handling.

The semantic hook belongs after the existing length/matcher scan and **before**
processor restoration, stats, `finish()`, `commit_cache(m)` or `_chain_rollback`.
It previews `draft_ids[:m] + [emit_last_id]`, never rejected draft tails. If it
lowers m, correction ID/logprob must be updated just as existing clamps do.
Any narrower prefix requiring a model rollback re-clamp must retain that safety
constraint. Alignment materialization must not commit a terminal correction.

```text
m_final <= m_model, m_length, m_token_stop, m_recipe_semantic
```

All are exclusive safe-prefix bounds in upstream's **accepted-draft count**
convention, not a count of every queued token. Upstream emits m+1 tokens and retains
the confirmed anchor plus m accepted target positions; the final correction is
normally unforwarded. Preview the correction as well. A terminal-completing token
must stay out of the target-committed drained prefix, like matcher-known stops.
Terminal finish reasons/events require explicit handling rather than blindly
continuing to the next verify or materializing the final correction on cancellation.

The preview starts from the canonical emitted parser, not from a shadow advanced
by draft existence. If previously committed queued tokens have not yet reached
that parser, feed that **bounded safe queue** into the temporary fork first, then
new candidates; preserve provenance and validate frontiers. Discard the fork.
Canonical parser push occurs only for actually emitted/canonical recovery tokens.
Rejected candidates must never become parser state. A semantic terminal beyond a
model rejection cannot influence canonical state; a terminal before rejection
must shorten the retained prefix before commit. All this remains unqualified.

## Stop strings and quiescence

The diagnostic parser correctly truncates a one-token stop, a multi-token stop
beside Unicode, and shared prefixes across token/character pushes. These are NOT
verify-cycle qualification. No arbitrary user stop is newly allowed under MTP.
Even a one-token encoding alone does not prove equivalence: recipe scope excludes
reasoning and tools, and neighboring decoded text can change matching positions.
No equivalence subset is promoted. Keep EOS, length and registered upstream
matcher clamps; add semantic preview as an additional minimum, not a replacement.

M28/M29 canonical quiescence remains structurally tested. No semantic guard is
installed, so tool/stop-boundary drains and interruption during DSML remain
ineligible. Live equality of history/target/DSpark frontiers and parser/history
semantics is **not proved** for these requests. The bounded tool-result loop was
not run, as required when the protocol gate fails.

## Evidence, environment and performance

`tools/run_m30_recipe_preview_audit.py` creates:

- `qualification.json`: decision, API/copy audit, all blocked live cases/counters.
- `authoritative-source-actions.json`: pinned-source action telemetry.
- `parser-diagnostics.json`: synthetic token/text/stop/tool diagnostic artifacts,
  token flushes, events and parsed arguments (null target/DSpark frontiers).
- `off-reference.json`: exact historical actual M11 OFF model tokens/response,
  source hash, current parser reparse. This example ends at a partial DSML end
  marker, not a fresh model-generated complete V4.1 block. Not an ON/OFF parity run.
- `runtime-identities.json`: package versions, binary/tokenizer/source hashes,
  source revision, real oMLX tool-path import identity.

Isolated `/tmp/ds41f-m30-qual` uses Python 3.13.14, no system-site-packages. Recipe
package and dist-info were copied byte-for-byte from the existing release venv;
no code patch was made. Distribution version 0.1.1/native version 0.1.0 match the
pinned pyproject/Cargo distinction. Binary build revision is not independently
attested; pinned-source probe is authoritative, binary tests corroborative.
A fresh offline source build was attempted but failed for absent OpenCV development
metadata. Initial exact pytest 9.0.2 resolution also failed offline; cached 9.1.1
was used instead. See `environment.md` for construction/reproduction.

Pinned oMLX genuinely declares `jsonschema>=4.0.0`; 4.26.0 is installed **only** in
the qualification environment. Real `omlx.api.tool_calling` imports successfully;
we did not exclude that import or modify ds41f dependencies. No repository-wide
pytest green claim, historical Engram repair, or unrelated runtime changes.

39 selected tests passed, plus 12 subtests. This includes OFF/tool parser,
M25/M29 lifecycle and P5 regressions, not model execution. Diagnostic parser
push timings are recorded separately; **preview cycle overhead is null**, because
there is no preview. No M26 performance preservation claim is possible yet.
The proposed algorithm forks bounded incremental state and processes newly
verified IDs plus a bounded queue, not all generated text every cycle.

Model replay/repack/verify/proposal counters are zero **because this audit executes
no model**; they are not live-MTP frontier proof. One historical parser replay is
explicitly counted. No parser repair under rejection was performed or claimed.

## Next scope

First obtain the upstream semantic-session/preview API and exact decoder
provenance interface. Then install the narrow pre-commit clamp, test acceptance,
rejection, terminal correction and quiescence frontiers, run fresh OFF/ON full
DSML and arbitrary-stop comparisons, measure incremental cost, and only after
that gate passes run bounded tool-result P6/P5/MTP re-entry. Operational soak
remains deferred until `PROTOCOL_GATE_SOLVED`.
