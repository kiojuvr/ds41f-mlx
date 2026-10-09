# M52R — application integration

**PASS. M53R not started.** Stop at this boundary; no soak, movement/Metal
campaign, new probe, test file, artifact schema or qualification framework.

## Production authority

The pinned recipe and actual model produce Paris and Berlin as two invocations
in **one assistant tool-call turn**. Result re-entry follows that complete turn;
there is no fixture-specific concatenation or synthetic second model request.
The immutable retained R1 `len(calls) == 2` assertion is unchanged.

Three levels now have different owners/transitions:

1. **Call certification:** `RecipeSemanticGuard` forks the official consuming
   parser/stashing/generator and accumulated response. At each canonical emit,
   it checks the exact projected prefix against canonical consumption. Official
   `ToolCallArgumentsEnd` completion certifies the call without pushing Stop into
   the canonical processor. Frozen call certificates bind processor identity,
   unique generation/guard lifetime, canonical revision, absolute consumed
   frontier, candidate prefix, prior response snapshot and consuming outcome.
   Identity/arguments and every preceding certificate remain immutable. Subsequent
   consuming outcomes must preserve the ordered certified prefix or fail closed.
2. **Effect permission:** certification alone neither executes a tool nor closes
   the task. Existing client reservations grant exactly-once ownership per
   session/request sequence/call ordinal/call ID after the supported full outcome
   is published. This policy defers effects until turn completion; it never
   authorizes an uncertified call. Conflicting duplicate authority rejects;
   completed duplicates reuse results; uncertain reservations never rerun.
3. **Assistant-turn completion:** the official recipe's semantic full-response
   transition is a physical generation boundary. Its consuming finished outcome
   must agree with canonical consumption and the immutable call certificates
   before canonical Stop/settlement. Raw DSML spelling is not an effect proof.
   `terminal_matches` retains the physical parser-boundary diagnostic separately
   from call certificates and consuming completion provenance.

The existing `last_turn` owner accumulates the ordered call segments while its
request fence is active. Settlement freezes their aggregate with the full
response, canonical events, certificate, body/sequence/session identity and
metrics. There is no second canonical state machine or parallel application
ledger. JSON, exact retained outcome retry, whole-outcome SSE reconnect and
client effects project this same frozen authority, without regeneration.

Existing canonical history and request fencing own result re-entry: exact
certified assistant history, lifetime/session, next request sequence, ordered
call IDs and bounded results must agree before any new canonical input is
admitted. The same supported local client executes both calls, admits their
stored results and obtains the final summary. Application history carries the
certified calls/results through continuation; no tool-call segment is relabeled
as an early end of the logical application task.

Preserved earlier M52R boundaries: pre-worker-completion `pending_delivery`,
known semantic terminal over cancellation, exact frozen JSON retry, shared
projection, exact SSE reconnect, duplicate/conflicting effect fencing,
certified result admission and uncertainty poison/retirement.

## Final qualification

Actual production HTTP Paris/Berlin execution first established these frontiers:
Paris consuming certificate **320**, Berlin **350**, full turn **356**. Both
effects ran once despite duplicate execution requests; certified result re-entry
produced the two-city summary. This was application completion, not a new probe.

Final source-matched retained model and fresh full **MTP-profile R1 PASS**:

- `artifacts/m52r/r1-final.json`: `CONFORMANT`, `level=full`,
  `profile=mtp-singleton-v1`.
- `artifacts/m52r/r1-final-gates/mtp-model.json`: 18 application cases and 27
  admission assertions, including retained two calls, duplicate effects, result
  re-entry, JSON retry and transport loss. No R1 model assertion changed.
- Existing affected tests: **181 passed** (`artifacts/m52r/affected-tests.log`).
- Existing official recipe matrix: `artifacts/m52r/recipe-matrix.json`; the
  two-call response reconstructs exactly. Existing preview test now asserts
  early certification without early turn closure and stable first-certificate
  identity through Berlin certification.

R1 material, verifier, manifest, protocol/recovery expected fixtures and contract
remain byte-identical. As in M51R qualification, existing active synthetic
fixtures supply current ownership metadata while preserving their assertions:
M34/M35, M36 completion flag, M39 explicit JSON-outcome observation, and M41
pre-publication profile validation. The R1 receipt records these adaptations.
Early seam-only attempts found those obsolete synthetic seams; they are not PASS
receipts. A both-profile launch also correctly rejected the unpromoted recipe
native in normal OFF admission. No admission bypass, reseal or OFF promotion was
introduced. M52R's full conformance scope is the changed MTP application path,
using the same explicit qualification-only source allowance as M51R.

Matched existing M50R/M51R performance workload, bracketing the candidate with
fresh M51R source controls, three measured sessions plus one excluded warmup in
each process, same installed dependencies/native and request:

| Process | Mean decode tok/s | Mean backbone ms |
| --- | ---: | ---: |
| M51R control before | 40.470616 | 840.155763 |
| M52R | 40.321402 | 842.454083 |
| M51R control after | 40.414329 | 841.280182 |

Candidate versus bracketed controls: decode **−0.2994%**, backbone wall
**+0.2065%**, within the retained matched 1% envelope. Canonical generated IDs,
frontier 283, depth-drafted/depth-accepted topology and zero history replay match
exactly. Receipts: `control-before.json.gz`, `performance.json.gz`,
`control-after.json.gz` under `artifacts/m52r`. The existing
`artifacts/m51r/connection-performance.tool.py` was reused unmodified.

Source/topology checks establish byte-identical tracked runtime physical sources
(except the host recipe guard), native sources, `_next` and `_retire`, and
AST-identical `_start` after removing only host guard/Ready construction, plus
the entire physical quiescence/cleanup prefix of `_settle`. Recipe projection
clones host semantic state only: no model invocation, tensor operation, copy,
synchronization, replay or repack. M51R physical topology is unchanged. No Metal
or movement campaign was needed or started.

## Qualification recipe identity and M53R boundary

`docs/m52r-recipe-consuming-eof.patch` is the complete recipe delta against
`third_party/mtp/recipe-source.tar.gz`, including consuming-state forks and the
completed-action/response-copy seams. No archived MTP execution architecture is
restored.

SHA256:

- Base recipe archive: `5b71ea6837ad3eb54a07da2b7ba1dd0a4ce658b585cdd2db24f4b83d9f879c22`
- Source patch: `560a791ff684a60ad366c3f46b961cb004f7819cfab95b4400bd0aa0d43b3794`
- Qualification native: `b4aef7e5749024f3dab8aac44c0dc10aa5fe3f8fe1068dd543a15493c34b567c`
- Qualification wheel: `439d9eaab77a0edb45ad250cb937ecf425e2b315370489085eb02e2315e592e2`

Native build: `/Volumes/SDXC-512/ds41f-mtp-investigation-build/recipe`, using
`PKG_CONFIG_PATH=/opt/homebrew/opt/opencv@4/lib/pkgconfig`, the investigation
interpreter's `maturin build --release --offline --skip-auditwheel` and
`CARGO_TARGET_DIR=/Volumes/SDXC-512/ds41f-mtp-investigation-build/cargo`.
Only the investigation interpreter received that wheel. Final qualification
identity and the matched control identity are retained compressed under
`artifacts/m52r`; compression preserves the original receipt bytes.

**M53R remaining scope only:** dependency/native reproducibility and normal-local
admission/profile. Source archives, requirements and installed admission remain
unpromoted and fail closed on the qualification delta. M53R has not begun.
