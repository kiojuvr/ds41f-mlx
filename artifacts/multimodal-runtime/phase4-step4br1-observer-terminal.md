# Phase 4 / Step 4B-R1 — terminal stream observer correction

**PASS — Observer qualified, model-free. Technically ready to recommend another
separately authorized Step 4B attempt; this report does not authorize it.**
Historical Step 4B remains **BLOCKED**, its first tool result remains failed, and
C8 remains unqualified. No model load, native Session, Vision execution, deployment,
push, Production change or official package edit occurred. Stop after Step 4B-R1.

## Authority and exact cause

Started clean on `Multimodal-Runtime` at expected/actual HEAD
`c9ed15daf3e98a4652a4807b9906ce979be549f5`. Read documentation policy, implementation
and CURRENT Multimodal Runtime plans, frozen Step 4A contract, Step 4A-R1 readiness
and historical Step 4B report. Inspected the retained observer, unchanged guard,
wire/lifecycle/gate/trace receipts and affected offline/registration tests. Reused
existing source identities, not another Harness/Recipe/DeepInfer/Runtime audit.

Evidence: [phase4-step4br1-evidence/](phase4-step4br1-evidence/).
`authority.json` binds starting authority and parent/guard hashes. Installed Node
is **v24.5.0**, official packages remain **0.2.1-rc.2**. Relevant installed adapter:

```text
/Users/kioju/.local/share/deepseek-harness/node_modules/@deepseek-ai/dsh-llm-deepseek/lib/index.js
SHA256 eba35025ed1eedda89c51f2da46a3a968f7be1d59719f2ceeadfc4518dff37d6
```

This equals the retained `phase4-step4b-evidence/observer-finding.json` identity;
package versions are not claims of exact upstream build Git SHA. Reverified the
Step 3B installed package receipt through the existing integrity inspector.

**Root cause:** the old clone observer unconditionally called
`phase4Gate.uncertain('native protocol observer failure')` on a reader exception,
even after persisting native `message_stop`. The installed translator returns
normally after a valid semantic terminal; `DeepSeekAdapter.generate` then calls
`consumer.abort()` in `finally`. That cleanup can abort the still-reading response
clone. The observer conflated that post-terminal transport cancellation with an
incomplete native outcome. Additionally, its old `captured=true` assignment
preceded tool-argument parsing, so that flag alone was not a trustworthy validator.

Preserved historical ordering, UTC:

```text
23:57:54.755 message_stop
23:57:54.757 native_message_stop receipt
23:57:54.761 clone AbortError (message_stop_already_captured=true)
23:57:54.770 sticky observer uncertainty
23:57:54.776 first read_image denied before execution
```

Native `00:1` independently completed N=571/C=0, selected a valid A read, settled
at 623 across Target40/DSpark3, published a complete pair and retired ownership.
This repair does not retrospectively execute that read or resume that Session.

## Small task-owned correction

New `observer.mjs` derives the retained Step 4A-R1 observer; original files remain
unchanged. Request inspection, HTTP charging, original `fetch(input, init)` call,
returned Response identity, tool guard and official lifecycle ownership are retained.
`observer.diff` records the source comparison.

| Case | Old classification | Corrected classification |
| --- | --- | --- |
| Validated complete terminal, then clone reader AbortError | Sticky uncertainty | **POST_TERMINAL_CLONE_ABORT**, observational transport cleanup only |
| Abort/timeout/transport failure before valid terminal | Sticky uncertainty | Sticky uncertainty; tool/request rejection |
| Normal EOF after valid terminal | Completion receipt | **VALIDATED_TERMINAL_EOF** |
| Missing terminal at EOF | Sticky uncertainty | **INCOMPLETE_SSE**, sticky uncertainty |
| Malformed terminal, JSON, duplicate sequence, logging or unexpected error | Potentially unsafe early flag | **OBSERVER_FAILURE**, sticky uncertainty |

Only `reader.read()` rejection with `name === 'AbortError'`, **after independent
validation and successful terminal receipt logging**, with no pending SSE/UTF-8
tail, qualifies for the benign classification. It never clears an existing sticky
stop. Parsing, terminal validation, record writing and arbitrary post-terminal
exceptions are outside that exemption. Further received events after terminal
(including duplicate terminals) fail closed. Unterminated frames are not promoted
to complete events at EOF. A bounded 16 MiB observation ceiling also fails closed.

### Validation without a competing Messages implementation

`terminal.mjs` verifies the complete installed adapter SHA256, then compiles its
**unchanged** `object`/`replayState` helpers and translator region into an isolated
function with the installed `LlmError`/`ToolCallId` dependencies. This is local
validation of recorded events, not another provider/HTTP submission, alternate
translator or official package patch. Source drift refuses qualification rather
than silently using different semantics. Framed event identity/JSON is checked
with the installed `eventsource-parser`.

At `message_stop`, the exact installed translator must exhaust normally and emit
one finish. It checks message start, supported stop reason, block identities/order,
open/closed state, compatible deltas, completed tool arguments and usage. The task
additionally refuses duplicate Tool Call IDs and unresolved/non-object tool JSON,
including the translator's permissive `max_tokens` tool-JSON escape. Only after
all validation **and terminal logging** does the observer mark completion. A raw
event name, preliminary flag or HTTP 200 is not acceptance.

The observer remains passive. No response rewriting/reencoding, original stream
replacement, tool dispatch/delay, Session mutation, retry-policy change, reservation
refund, cache/compaction/image-offload decision or shutdown ownership was added.
It consumes only the clone, continues observing its EOF/abort and releases its lock.
On observation failure it cancels only that branch; it does **not await** tee
cancellation, which waits on the original branch. The cancellation promise has a
rejection handler. Original-consumer EOF after clone failure is explicitly tested.
Detached SSE work has a handled completion task, retires from the diagnostic active
set, and introduces no timer or new request. `observerIdle` is test diagnostics,
never awaited by fetch or Agent. Existing external duration/transport ownership is
unchanged.

## Focused model-free results

`tests.mjs` uses original observer/global-gate interfaces, deterministic bounded
fixtures from the **33 genuine retained events**, actual installed adapter and
real installed Cordis/Agent/Tools/Session path. Final **23 regression groups PASS**;
`results.json`, `classifications.json`, `tests.log`, `fixture-wire.jsonl` retain results.
No second Messages translator was constructed for tests.

- **T1:** exact genuine text/tool sequence, completed JSON, `tool_use`, terminal,
  then stored-error-class AbortError: terminal accepted, benign classification,
  exact event/usage preservation, no uncertainty.
- **T2:** valid normal EOF, normal completion receipt, no uncertainty.
- **T3:** ordinary `end_turn` text works without any tool requirement.
- **T4:** actual installed Agent, real registered frozen guard, manifest A digest
  authentication and original Tool Call ID reach one successful **no-effect** tool
  fixture after adapter cleanup. No acquisition, attachment or Vision is claimed.
- **T5–T6:** preterminal abort and EOF are sticky and reject tools/new HTTP work.
- **T7–T10:** missing reason, unclosed block, incomplete tool JSON and malformed SSE
  JSON fail closed. Failed tool parsing never sets the validated-terminal flag.
- **T11:** duplicate terminal has no silent successful acceptance.
- **T12:** preterminal timeout/transport fixtures fail closed, keep reservation;
  an additional real installed idle-watchdog timeout proves one submission only.
- **T13:** unexpected post-terminal exception and injected terminal logging failure
  remain failures, not benign aborts.
- **T14:** earlier genuine sticky failure survives valid completion/benign abort.
- Additional groups cover terminal-before-start, unsupported stop reason, duplicate
  tool identity, malformed post-terminal JSON, max-token partial tool JSON, repeated
  block index, wrong block delta identity, SSE event-name mismatch and unterminated
  tail; original bytes/EOF after malformed-clone cancellation; task retirement and
  zero unhandled promise rejections.

There are also **29 PASS** unchanged installed guard registration/middleware checks
(`guard-tests.mjs`, `guard-registration.json`) and **6 PASS** original immutable
Step 4B offline regressions (`original-offline-tests.log`). No unrelated numerical,
Vision or full Phase 4 campaign was rerun. Early test-fixture setup failures are
retained in `tests-initial.log` and `tests-setup-*.log`: a fixture closed an already
cancelled stream, an adapter dependency lacked `prepareExtensions`, and a success
assertion expected undefined instead of actual `isError:false`. None was native.
The final Agent test deliberately rejects the *next* pre-step to avoid a second
provider request; the frozen guard correctly makes that later blocked turn sticky.
It is not observer uncertainty or a guard defect.

### Real installed adapter and asynchronous race

`adapter-lifecycle.json` proves actual `DeepSeekAdapter.stream/request/translate/
generate/finally` behavior over a controlled local **fetch transport**, with no
network, Runtime or subclass/reimplemented translator. A standalone consumer pauses
at the official finish yield until the clone terminal receipt exists, then exhausts
the generator: terminal receipt → official generator completion/cleanup abort →
clone benign classification. Both tool and text cases pass. This bounded scheduling
pause is only in the test consumer, never the observer or production Agent.

`guard-race.json` additionally exercises the **unpaused real Agent** schedule: its
adapter cleanup abort can arrive while the clone is still validating the already
received terminal; validation/receipt completes before the next clone read handles
the abort. The original Agent proceeds to the authenticated no-effect tool. In
both schedules there are **zero observer uncertainty calls**, no duplicated tool,
no false-positive asynchronous stop and exactly one adapter transport submission.
The Agent test then intentionally fences its next step, as explained above.
The receipt retains that distinction (`terminalBeforeCleanup`), not a fabricated
claim that the observer owns official semantic completion. Original fetch argument
and Response object identity assertions pass. No HTTP/generation refund or new
retry allowance exists; the charged state remains one request/reservation and 1,200
output allocation.

## Independent original-evidence regression and integrity

`replay-00-1.json` and `offline.json` bind the corrected replay to the immutable
wire event digest and exact native trace event list. Observed terminal is valid,
subsequent AbortError classifies **POST_TERMINAL_CLONE_ABORT**, and original usage
is preserved. Independent trace joins confirm settled frontier 623, all 40/3
positions, complete publication and retirement. Original persisted error result is
still `isError:true` with `PHASE4_STOP`; no historical C8 PASS, Session amendment,
resume or successful effect is asserted. Historical reports/evidence and compressed
official storage are unchanged relative to starting HEAD.

`integrity-before.json` / `integrity-after.json`: **817/817** original operational
Runtime paths match retained SHA256/size/mtime/inode; installed official package
identities match. Original `/Volumes/SDXC-512/ds41f-1017` was not executed or changed.
`offline.py` verifies the baseline-bound admitted checkpoint cache, all **48** original
Flash-1017 shard identities (resolved path, device, inode, size, mtime and ctime),
expected admitted hashes and checkpoint metadata hashes. Original checkpoint revision
remains `1bae0e1088993b9b82de08f59de5be5a34e42170`. No weights read/loaded or full
checkpoint re-admission was needed. No Production files changed.

New observer SHA256:
`41a9b1b35a4c30d65034ac33fb01f44567d3b48d6070b551731073507da6f766`.
Pinned translator-reuse module SHA256:
`94fd324362bada98aa9df7b4f666bba26b0b57d98b2bed385cd51850b07c4f03`.

Reproduction from repository root (model-free):

```sh
E=artifacts/multimodal-runtime/phase4-step4br1-evidence
node "$E/tests.mjs"
node "$E/guard-tests.mjs"
python3 -B "$E/offline.py"
python3 -B "$E/integrity.py" after
python3 -B artifacts/multimodal-runtime/phase4-step4b-evidence/offline_test.py
git diff --check
python3 -B tools/check_repository_self_containment.py
```

## Frozen contract and disposition

No frozen contract or C1–C9 criterion changed: nine distinct sources/ten real reads;
image budget **3**, quantum **1**; pressure **8,192**, retention **1,024**, summary
cap **2,048**, ordinary cap **1,200**; **one** summary attempt, **19** ordinary
generations, **20** Runtime requests, **27** adapter attempts; prepared+output
**32,768**, Round-3 entry **4,608**. All Resource Supervisor thresholds, one-session
requirements, no-retry/uncertain-effect stops and original readiness scope remain.
The Guard and official lifecycle mechanism are unchanged. Only this next task-owned
observer version/identity is qualified for future separately authorized execution.

**PASS — Observer qualified. No remaining narrow observer blocker.** Recommend a
separately authorized Step 4B attempt with this new observer identity and all normal
fresh installed/checkpoint/source/host preflight gates; do not reuse the exhausted
Session or its allowance. This is technical readiness only, not automatic execution
authorization or a promise of integrated native workload success. Existing C2/C7
and broader C6/C9 qualification boundaries remain. No push/deploy, new native
attempt or Phase 4C. **Stop after Phase 4 / Step 4B-R1.**
