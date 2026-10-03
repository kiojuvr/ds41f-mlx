# M31 — candidate recipe semantic session and preview

## Decision: BUILD_OR_BINDING_BLOCKED

Started from ds41f `bd28b8186667e0942b333c22c574aa1a107b2328`.
Production MTP remains **OFF**, public MTP is disabled, and MTP persistence remains
fail closed. No live MTP protocol gate, tool-result re-entry, or operational soak
was run. M32 operational qualification is **not authorized**.

A narrow candidate upstream extension is implemented and source-tested. Its real
native Python build fails because OpenCV development metadata/headers are missing.
A compile-only check succeeds, but no new loadable native extension exists. The
Python runtime parity prerequisite is therefore unmet. In accordance with the
ordering requirement, **no oMLX semantic hook or live MTP session was installed**.

This decision does not mean the recipe extension is fundamentally insufficient,
nor that the MTP clamp is fundamentally blocked. It records the actual stopping
point: a runnable native binding cannot currently be qualified. All required live
cases have BLOCKED_NATIVE_BUILD status and null frontiers/topology in the manifest.

## Authorities and dependency ownership

- Recipe base: `8cadfede7063c896b944e7bae05daa3549ae97ea`.
- Candidate recipe commit: `066d2ef2ed0deb574a0e6b5e6136316d6f296d55`.
- oMLX: clean `4d4f5a280bc1739ba2cf39c1cee44fd5cc89cb40`.
- Isolated recipe source: `/tmp/ds41f-m31-recipe`, a separate detached worktree.
- Pristine comparison source: `/tmp/ds41f-m31-base`.
- Python environment: `/tmp/ds41f-m31-qual`, no system-site-packages.
- Build caches: `/tmp/ds41f-m31-cargo` and a separate compile-only target directory.

`artifacts/m31/recipe-semantic-preview.patch` is a git-format patch of the separate
recipe commit. It applies cleanly to the exact pin. Source/patch hashes and package,
compiler and tokenizer identities are in `runtime-identities.json`. The preserved
release native binary still matches M30's hash; it was **not** imported as a
substitute for the missing M31 build. Neither the original recipe source nor the
release installation was patched in place. The whole dependency is not vendored.

The candidate patch is **not an approved release dependency**. ds41f's dependency
manifest, production code, oMLX checkout and OFF path are unchanged. An eventual
release needs the extension upstream or an explicitly reviewed pinned patch and
provenance approval, not silent adoption of this local commit.

## Synchronous semantic state

The authoritative recipe files remain:

```text
deepseek-recipe/src/stream/state_machine.rs    existing recognizer + terminal observation
deepseek-recipe/src/stream/decoder.rs          exact decoder + forkable pending state
deepseek-recipe/src/stream/semantic.rs         synchronous session/handle
deepseek-recipe/src/stream/processor.rs        original protocol stashing
deepseek-recipe-python/src/response.rs        direct Rust preview binding
```

StateMachine and its nested state, match branches and KMP positions are Clone.
The clone includes stage, leading/newline flags, stashed_size, pattern strings and
KMP tables/positions, reasoning/JSON state, options, absolute source-byte counter,
and first semantic terminal. No private parser stage is exposed to Python.

`SemanticSession` owns that state machine, optional StreamDecoder and an input-end
flag. Decoder cloning copies pending IDs **and pending count**, while sharing the
immutable implementation through `Arc<Mutex<Box<dyn TokenizerDecoder>>>`. The mutex
keeps the original Send-only decoder trait contract rather than introducing a
breaking Sync bound. It is never held across protocol-generator awaits.

Canonical token decode and StateMachine.feed occur together as one synchronous
step. The async processor still owns its original protocol generator and
StashedChunks. They consume the same returned source and output actions. The
processor retains a handle to semantic state before becoming an async stream;
the Python binding retains that handle too. Its existing state mutex serializes
preview with push/drain at the exact next-input boundary. Rust callers must also
serialize input and preview at that boundary.

Preview clones **only** synchronous semantic state. It neither clones nor submits
to async channels, protocol generators or protocol stashing. It does not store,
reparse or repair canonical output history. Protocol finish behavior remains the
original behavior, including backend-Stop/EOF treatment of incomplete tools.

## Explicit terminal observations

The existing matched transition into Finished now records:

- STOP_SEQUENCE for OutputAction::StopSequence;
- DSML_TOOL_CALL_BLOCK_END for the existing DSML block-closing branches.

The record contains the absolute half-open decoded-source byte span. It is made
at the actual matched transition, not inferred from invalid tail, has_tool_calls,
opening marker, EOF, or an invocation close. Multiple invocations remain possible
until the block closes. JSON fence closure is MatchedJson, not terminal.

This metadata does **not** change `StreamProcessor.finished`, close the canonical
stream, or invent a backend finish reason. The M30 findings remain valid: DSML
semantic completion and Python stream completion are different; EOF/backend Stop
can still report tools from an incomplete block. Backend Stop after a real DSML
terminal still produces the ordinary ToolCalls finish through unchanged stashing.

## Preview API and exact token provenance

Candidate Python API (implemented in source; **not runnable locally yet**):

```python
result = processor.preview_tokens(candidate_ids)
result.terminal_kind           # None | STOP_SEQUENCE | DSML_TOOL_CALL_BLOCK_END
result.completing_token_index  # zero-based candidate ID that causes terminal
result.safe_token_count        # IDs before that completing candidate
result.source_start            # absolute decoded byte offset, or None
result.source_end              # exclusive offset, or None
result.mapping_exact
result.candidate_ids_processed
result.max_pending_ids
processor.semantic_terminal    # actual canonical observation, not prediction
```

Both canonical and preview execution invoke the **same Rust StreamDecoder**:
append ID and pending count, decode the complete pending group with special tokens
retained, hold empty text or text ending in U+FFFD, otherwise clear the group and
feed the decoded string through the same StateMachine. No Python detokenization
or stop-string re-tokenization is used for this API.

If candidate j flushes a group containing previously canonical pending IDs and
completes a terminal, the exact completion index is j and safe_token_count is j.
Earlier candidate IDs that produced no text remain safe; they have not completed
any semantic terminal. Span start can precede the preview boundary. A terminal
inside a flush still maps to its flush-causing ID, never a rounded byte/token
boundary. Full-flush feed may also see invalid tail, as canonical parsing does;
the observation preserves the first terminal and excludes the completing ID from
the speculative target-committed drained prefix.

Known immutable Tokenizer decoding opts in to preview. Custom TokenizerDecoder
implementations default to unsupported and must explicitly promise immutable,
deterministic decoding to opt in. Unsupported decoding or already-terminal/ended
semantic state yields mapping_exact=false and zero safe count. Missing tokenizer,
decode errors, and closed/finished Python input fail with errors. No canonical
repair or fallback reparse occurs.

## Source-level qualification

`cargo test --offline -p deepseek-recipe`: **10 passed**. Tests cover:

- complete DSML closing marker at every token split;
- every stop split, shared prefixes and adjacent Unicode;
- canonical pending UTF-8 IDs completed by speculative IDs;
- source spans starting before preview;
- repeated identical previews and exact parser/pending-state fingerprints;
- simulated full/partial/immediate-rejection emission prefixes;
- unsupported decoders and preview errors leaving canonical state untouched;
- reasoning scope, JSON fence nontermination, invocation-close nontermination;
- actual canonical StreamProcessor using the retained session and original finish.

These are parser/session tests, **not native target acceptance/rejection tests**.

`tools/run_m31_recipe_source_parity.py` compiles the same diagnostic driver
separately against pristine and patched recipe crates. **64 comparisons across
22 cases passed**: raw OutputChunks and chat protocol events are identical for
token, text and character chunks. Only chat `created` timestamps are normalized;
all other compared fields remain exact. Cases include text, reasoning, single and
multiple tools, string/non-string args, raw/fenced JSON, backend Stop/Length/EOF,
EOS literal, stops, Unicode, incomplete/complete DSML and invalid tail. One fixture
contains historical actual M11 OFF model tokens, explicitly not fresh M31 output.
Full reference and patched event artifacts are preserved.

Source preview telemetry covers 77 six-ID windows, including canonical pending
IDs and terminal spans crossing windows. Predictions agree with incremental
canonical token consumption. No terminal is reached canonically by a preview.
The source test fingerprints include all StateMachine state, decoder pending IDs
and count, and end flag. No complete-history repair is used.

## Native build and binding gate

Real build command, on the final committed recipe source:

```sh
CARGO_TARGET_DIR=/tmp/ds41f-m31-cargo uv pip install \
  --python /tmp/ds41f-m31-qual/bin/python --offline --no-build-isolation \
  /tmp/ds41f-m31-recipe/deepseek-recipe-python
```

Fails in opencv crate 0.93.7: `opencv4.pc` / OpenCVConfig.cmake / development headers
cannot be located by environment, pkg-config, CMake or vcpkg probes. Bundled
OpenCV dylibs in the preserved release wheel do not provide a development build
contract; we did not stub/remove image bindings or substitute a mismatched ABI.
`native-build.log` records the failure. No new Python native binary hash exists.

`DOCS_RS=1 cargo check -p deepseek-recipe-python` passes, validating the Rust/PyO3
source types using OpenCV's documentation bindings. It is **compile-only** and
cannot qualify linking, import, tokenizer/decoder behavior or Python ownership.
Its target directory is separate and no such artifact is installed.

New binding tests and the existing binding suite were both attempted. Collection
fails because no new deepseek_recipe module is installed. This is explicit in
`binding-tests.log`, not hidden by importorskip or by borrowing the old native
module. Python canonical/preview runtime parity remains **unqualified**.

The genuine jsonschema-dependent `omlx.api.tool_calling` import succeeds in the
isolated environment (jsonschema 4.26.0, satisfying pinned oMLX jsonschema>=4.0.0).
This is an actual tool-module import, not a fake. It does not make the small source
build environment a full MLX/live-model qualification environment.

Selected ds41f evidence/M29/M25/P5/P6 regressions: **51 passed, 24 subtests passed**,
no skips. M29 exercises the M28 canonical-quiescence safety primitive. There is no
standalone M28 test file. Repository-wide pytest and unrelated historical Engram
harness repair were not attempted.

## Intended MTP integration, explicitly deferred

No native MTP code was changed before the binding/runtime parity gate. After that
gate passes, install the narrow hook in pinned `_chain_verify` after rollback,
alignment and length/matcher clamps but before processor restoration, final stats,
`finish()`, target rollback/commit, and DSpark commit/draft-next finalization.

Preview `draft_ids[:m] + [emit_last_id]`. Bound m by safe_token_count, update the
correction ID/logprob exactly as existing narrowing does, and rerun the native
model rollback-capability clamp where required. The final accepted count must
satisfy all model, length, token-stop and semantic bounds. Reuse native rollback;
do not independently alter verification math. Terminal alignment materialization
must not commit the predicted terminal before its canonical emission.

Carry bounded metadata until emission: terminal kind, token ID, absolute emission
ordinal and expected source span. Do not mark the canonical parser terminal early.
On that exact emitted token, push canonically, validate actual observation against
prediction, block further verify admission, and issue backend Stop for DSML so
ordinary recipe output finishes ToolCalls. This mechanism is **not installed**.

If committed queued tokens precede the new candidates but have not reached the
canonical parser, preview that bounded safe queue followed by candidates in one
fork, retaining provenance. Never advance canonical parser from draft existence.
M29 recovery must push only truly canonical recovered tokens exactly once.
Unexpected terminal in the supposedly safe queue must fail closed.

After terminal emission, the existing one-token target-behind quiescence repair
can materialize that terminal without new verify/proposal, aligning canonical
history, parser, target and DSpark. This is a planned compatibility contract,
**not a live frontier proof**. Safe-prefix drain during interruption and P6/P5
DSpark continuity likewise remain unqualified for semantic tool/stop requests.

## Performance and replay scope

Source-only optimized Rust measurements: 15,400 preview samples over six-ID
windows. Median/p95/max: **1,375 / 1,667 / 13,083 ns**. State clone median/p95/max:
**84 / 167 / 2,375 ns**. Maximum observed decoder group: two IDs. Timing excludes
Python, MTP model execution and DSpark; exact data is in source-preview-mapping.
These figures do not prove guarded MTP remains in the M26 throughput class.

State-copy complexity depends on configured match patterns and pending IDs, not
generated history length. Pending groups are not theoretically fixed-size under
the existing U+FFFD-hold contract. The API reports their sizes; resource eligibility
bounds must not silently change decoding.

Preview performs zero parser-history replay. Historical fixtures are separately
parsed for diagnostic/reference comparison, not replayed per speculative cycle.
Model replay/repack/verify/proposal counters are zero because **no model or
quiescence was executed**. They are not live zero-replay/repack proof. Live MTP
throughput, acceptance, target/DSpark/parser frontiers and tool re-entry are null.

## Next scope

Restore a genuine native OpenCV development build environment and build this exact
candidate without stubs. Run existing/new binding tests and native canonical
parity first. Then integrate the pre-commit clamp and emission metadata, qualify
full/partial/rejected native MTP topologies, actual OFF/ON DSML and arbitrary stops,
interruption/quiescence frontiers, and one P6/P5/MTP tool-result re-entry with zero
model-history replay/repack. Only PROTOCOL_GATE_SOLVED permits operational soak.
