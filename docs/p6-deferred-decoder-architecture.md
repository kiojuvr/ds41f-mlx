# P6 deferred decoder: authority reconstruction and complete architecture plan

**Status: implemented and real-qualified in the ds41f P6 package.**

The authority reconstruction below remains the historical design/audit record from `5cea84d`. Since that audit, ds41f implemented P6 as `DeferredPrefillAppend` and qualified it on the target Mac MLX environment. This document still does **not** change production runtime selection, implement P7/P8, or make performance claims.

Implemented/qualified scope:

- package: `ds41f_mlx.prefill_fp8_mlx.p6_append` plus P5 same-cache handoff integration;
- explicit sealed-commit continuation API: `DeferredPrefillAppend.continue_from_commit(...)`;
- canonical public slot0: MLX int32 array shape `(1,)`, not Python int;
- ordinary-only small append seal allowed for bounded rebuilds;
- production selector remains unchanged.

Real qualification environment:

```text
Python 3.13.15
MLX 0.32.2
NumPy 2.3.5
oMLX 0.7.0.dev2
oMLX revision b390b31e0c6831225fed0f24d278eb1db7fcb68b
checkpoint /Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash
preserve_mtp=False
engram_ssd_offload=True
```

Successful evidence cases:

```text
complete-16384
pending-16384
tiny-16385
A-24577
matched-geometry non-deferred control
B-fresh-49155
B-continued C24578 -> T49155
failure/rebuild
P5 bootstrap/decode
```

Observed P5 handoff evidence across completed real cases: same live cache list, `prompt_replay_count = 0`, `full_cache_repack_count = 0`, `PrefillContinuationState` export count/equivalent exported flag = 0, bounded decode succeeded. P6 qualification supports the P6-local status `deferred_decoder_suffix_lifetime = qualified` and `dwarfstar_carry_lifetime = qualified`; it does **not** promote P7 overlap/scheduling, P8 graph reuse, production selector, performance gate, or global runtime completion.

Watchdog truth correction: the completed matrix runner recorded post-segment coarse timings and enforced only the post-segment pathological `<10 effective tokens/sec after >=4096 tokens` guard. It did not enforce an in-segment 180-second no-progress watchdog, and the 4x normalized precursor guard was not active because no baselines were supplied. Evidence should therefore be read as: no observed pathological slowdown, no `<10 tok/s` segment abort, and no manually observed long stall—not as a full watchdog pass.

## 1. Pinned authority and production trace

Execution authority: **`antirez/ds4@0aaea5a238fb41a35106a551e73c8409dfb751ac`**, not the local native planner. Source was obtained with `git show` at that revision, rather than inferred from test names. Reproducibility digests:

- `ds4.c`: SHA256 `8f546837c0032a24b381052253244a34e10b7eea2647e97862cb4967b8604b89`.
- `tests/test_deepseek41_graph.c`: SHA256 `2846dff692e179d74794e41132b879c7ac944abafadf16e6dd54551b6e349d03`.

Line numbers below refer to those pinned files. [Pinned ds4.c](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/ds4.c) and [pinned graph test](https://github.com/antirez/ds4/blob/0aaea5a238fb41a35106a551e73c8409dfb751ac/tests/test_deepseek41_graph.c) are the external references. Mathematical operations for the future MLX implementation remain the reviewed oMLX baseline/CED donor used in P3/P4; DwarfStar tensor formats and GGUF kernels are not to be ported.

### Actual call path

```text
test main --deferred-decoder
  -> check_deferred_decoder
  -> ds4_session_sync
  -> ds4_session_sync_internal, DeepSeek-V4.1 GPU branch
  -> request-level loop selecting a token range and whether decoder is pending
     -> ds41_graph_prefill_sweep(..., encoder_only, resume_encoder)
        -> full encoder layers 0..19 for the NEW token range
        -> layer20 full-source publication
        -> either stop invalid, or exact suffix query layers20..39
     -> ordinary graph prefill/step for the final non-pending tail
  -> ds41_graph_logits
  -> checkpoint_valid = true
```

This is not a decoder-only replay of stored old full-prefix HC. The decisive unit is the **session append/sync loop**, with subordinate layer-major sweeps.

### Evidence map

| Behavior | Production implementation | Pinned test evidence |
|---|---|---|
| Capacity-bounded multiple sweeps | `ds41_prefill_count`, `ds4.c:41382–41417`: count bounded by `carry_cap`, rounded down to a 2048-token boundary in wide mode; outer sync loop `75413–75461` repeats | `check_deferred_decoder`, test `1061–1145`: ctx57344, streaming Metal, both graphs' effective carry cap forced to16384, targets24577 and49155 |
| Defer eligibility | Sync `75426–75430`: `!encoder_resident && count >= 16384 && remaining-count >= 8192`, suffix enabled, defer not disabled | Control sets `DS4_METAL_DISABLE_V41_DEFER_DECODER=1`; candidate does not, test `1092–1106` |
| Encoder sweep really leaves decoder unexecuted | Sweep `41701–41709`: at layer20, map only its decode/source weights as needed, call `ds41_decoder_prepare(... offset=0, rows=total_count, publish=true)`, then **break** | Invalid completed encoder sweep observed by `note_deferred_progress`, test `1053–1059`, cancellation `1130–1138` |
| Pending state and later completion | Sync passes `(defer_decoder, decoder_pending)` to the sweep at `75436–75440`; sets `decoder_pending=defer_decoder` only after success, `75452–75456` | Initial and continued cases; candidate reaches final valid frontier and exact state/logits |
| Admission while graph invalid is internal only | Sweep `41667–41679`: invalid graph rejected unless `resume_encoder`; both flags require suffix mode. Resume still runs layers from0, with new tokens, not a decoder-only loop | Continued encoder execution inside the same sync invocation |
| Encoder-ahead position/history | Sweep `41971–41975`: on success, `g->pos=initial_start+total_count`, `g->history=next_history`, `g->valid=!encoder_only`; sync appends processed tokens to `s->checkpoint`, but sets `checkpoint_valid=false` | Progress callback observes invalid state; no valid snapshot |
| Source and local-window preparation are distinct | `ds41_decoder_prepare`, `41525–41589`; sweep `41748–41755`: full layer20 source, then 127 immediately preceding input rows per decoder layer | Final owner state/windows equal control |
| Exact final dependency cone | Sweep `41752`: query rows `1+(39-layer)*127`; comment `41522–41524`: no old decoder output needed | State spans, history and logits compared byte-for-byte at final frontier, test `1107–1119` |
| Final output and valid session checkpoint | Sweep `41966–41975` copies last residual/pre only on non-encoder-only success; sync `75472–75483` computes final head and then makes checkpoint valid | Callback snapshot becomes ready; test `1107`, save/load `1120–1122` |
| Invalid snapshot forbidden | `ds4_session_payload_bytes`, `62472ff`, requires valid checkpoint/complete graph; `ds41_save_payload`, `62393ff`, requires `g->valid` and `g->pos==checkpoint.len`; snapshot entry `63837ff` rejects zero payload | `note_progress`, test `245–267`, tries save during invalid graph; pending cancellation save fails, `1135–1138` |
| Decode requires a synchronized checkpoint | `ds4_session_eval_probe_tp`, `77476ff`; DeepSeek eval branch `77320ff`; graph step also requires valid graph | Next eval after save/restore matches control logits, `1123–1125` |
| Interruption/recovery | Sync failure invalidates checkpoint; pending exit `75463–75468` explicitly sets both invalid. Next sync entry `75403–75408` resets invalid/mismatched graph and token checkpoint | Cancel at pending frontier16384, then sync fresh129 and compare control, test `1130–1144` |
| Tiny final tail does not manufacture pending work | Eligibility includes an **8192-token remaining-tail** test; short path disabled only when already pending, sync `75421–75434` |16385 succeeds without the callback ever observing a completed-sweep pending decoder, test `1081–1090` |

`note_progress` checks the first **in-sweep display** event while `g->valid=false`, not just a completed encoder-only sweep. Its later successful snapshot check occurs on a non-display event with `checkpoint_valid=true`. `note_deferred_progress` arms cancellation only on a **completed `prefill_chunk` whose graph is invalid**. Distinguish these assertions.

### What is persisted in DwarfStar

`ds41_gpu_graph`, `40182ff`, contains one graph position, history,40 window rings,4 owner compressed/index caches, pending pair buffers, and reusable batch/carry workspace. There are **not**40 independent committed offset counters in that implementation.

`ds41_state_spans`, `62368–62383`, explicitly excludes selection masks: they are reconstructed by index sources on the next token. It saves40 windows,4 compressed/index owners, and unfinished compression pairs only where present. Snapshot payload adds tokens and logits; restoration reconstructs normalized Engram history from the last up-to-three token IDs (`62456–62465`). HC carry/decoder intermediates are not snapshot state.

A math/storage distinction matters: native publication writes fixed absolute cache positions (`40620ff`, `40951ff`); its suffix attention may project/write the same source rows again (`41049ff`). That is not oMLX's append-on-source path. Future ds41f keeps the already-qualified source/query split: one new-range source prepare, no source regeneration/append from query chunks. "Once" below refers to ds41f source production and logical source coverage, not a claim that the native code invokes a compressor only once.

## 2. Purpose of deferral

**Primary purpose: avoid repeated decoder query/attention/HC/MoE sweeps, and their weight-streaming work, while advancing a long request through multiple full encoder sweeps.** Each intermediate sweep still produces every new layer20 global source key. The final decoder only needs recent input dependencies plus those global keys.

Evidence, not inference from names:

- The layer20 branch actually skips queries/MoE for20..39; ordinary control runs them after every sweep.
- Encoder-only publication uses `metal_graph_stream_map_layer_decode` instead of the ordinary full layer mapping. Later encoder sweeps continue writing their own source caches at absolute positions.
- The final-sweep comment promises an exact **2541-token input suffix**, and explicitly says no old decoder output is needed.
- Encoder residency is a **different mechanism** (`ds41_encoder_acquire/release`, `41425–41504`). Wide carry mode causes acquire to return without pinning encoder experts (`41458`), and deferral explicitly requires `!encoder_resident`. It is therefore wrong to describe this tested deferral as primarily keeping encoder weights resident. Encoder weights may be streamed again per encoder sweep.
- Engram overlap, layer preparation and expert-cache seeding coexist in the code, but are not the state-machine reason for deferral. Encoder-only skips `ds41_prefill_seed` (`41933`); none of those P7 mechanisms is part of this P6 implementation package.

Minimal semantic carry retention is enabled by the mechanism, but pinned physical workspace allocations stay reusable at carry capacity. Do not claim the native implementation frees/compacts every allocation to2541 rows: it **reuses** its carry arrays. The important port requirement is not to preserve obsolete full-prefix HC as deferred state.

## 3. Reconstructed state machine

Terminology: `encoder_only`, `resume_encoder`, `decoder_pending`, `g->valid`, `checkpoint_valid`, and `pending_logits` are actual pinned names. States below describe their meaning; they are not new executable ds41f enums.

Let:

- **C** = last successfully exposed checkpoint frontier before this append;
- **E** = latest completely encoded and layer20-source-published frontier;
- **D** = latest completely reconstructed decoder frontier (logical completeness, not all old rows physically replayed);
- **T** = requested target frontier.

During a chunk/layer, physical frontiers can be mixed; E advances only after the full encoder/source sweep succeeds. C is historical metadata once in-place work begins, **not a promise that the old cache remains usable**. C=0 in fresh examples denotes the reset/empty append origin, not a nonempty snapshot or P5-admissible checkpoint (`ds41_graph_reset` makes the empty graph append-ready; session checkpoint validity still gates external use).

```text
VALID_CHECKPOINT (C=E=D)
    -> BEGIN_APPEND / invalidate current checkpoint capability
    -> ENCODER_SWEEP_INVALID (new tokens; mixed layer progress)
       -> if defer eligible:
          LAYER20_SOURCE_COMPLETE / DECODER_PENDING (E increases; D does not)
          -> next NEW-token ENCODER_SWEEP_INVALID, admitted internally
             by resume_encoder=true
          -> repeat while defer eligible
       -> last eligible-wide sweep executes its NEW encoder range,
          publishes layer20 source, and runs DEFERRED_DECODER_EXECUTION
          on that range's exact dependency suffix
    -> GRAPH_COMPLETE / OUTPUT_PENDING (D=E; g.valid=true,
       checkpoint_valid=false)
    -> optional ordinary non-pending tail (small sweep or token steps)
    -> FINAL_OUTPUT (native final head; only final requested output)
    -> VALID_CHECKPOINT (C=E=D=T)

Any incomplete/failed/pending interruption -> INVALID_REBUILD_REQUIRED
    -> reset and replay the requested token history, not external warm resume
```

### Transition ledger (native meaning)

| Transition/state | Frontiers and validity | Tokens / Engram | Persistent sources/pending | Carry ownership / snapshot |
|---|---|---|---|---|
| Valid entry | C=E=D; graph and checkpoint valid | `checkpoint` owns synchronized tokens; history atC | Owner/window/pair state consistent withC | No old HC required for append; snapshot eligible |
| Begin new sweep | C remains last exposure; current per-layer work starts at priorE; `g.valid=false` | Hashes for new range computed using a copy `next_history` of current history | Encoder windows/sources may be mutated in place | Current sweep owns full encoder workspace; snapshot unavailable |
| Successful encoder-only/source boundary | E+=count, D unchanged, graph invalid; checkpoint invalid | `g.history=next_history` atE; processed tokens pushed to `checkpoint` atE | Encoder0..19 state and owner20 global source reachE; decoder windows are stale/mixed; pending pairs for encoder sources preserved | Old sweep HC/pre/idx are no longer needed for a later encoder range. Workspace can be overwritten. No snapshot or decode |
| Next internal sweep | `resume_encoder` permits invalid entry, starts atE; no external validity granted | Hash next new token range using history atE | Continue encoder caches and20 source at their actual positions, not atC | New embeddings/HC, not resumed old HC |
| Completing decoder suffix | Final new range has enough rows to contain the cone; E advances, D converges toE | New-range hashes belong to encoder only; history ends atE | Full cumulative20 source consumed; per-layer127 local preparation replaces needed dependency window; query selections recreated | Preserve only final-range dependencies until consumed; no snapshot mid-layer |
| Graph complete, head pending | `g.valid=true`, D=E; checkpoint still invalid | `checkpoint.len` is working frontierE | Persistent continuation state complete atE | Native retains last residual/pre for output; snapshot still unavailable |
| Ordinary tail | Pending flag already false; per-token/small-sweep E and D advance together | Tokens/history advance | All layers update normally, including pending pair completion | No forced extra encoder-only sweep; validity exposed only after final head in this sync |
| Final native output/commit | C=E=D=T; checkpoint valid after logits | Synchronized token checkpoint and history atT | Snapshot spans coherent | Snapshot/save/load permitted; decoder intermediates not persistent |
| Failure/interruption while pending | No usable checkpoint; mixed physical state cannot be interpreted asC orE | Complete requested tokens still supplied by caller; in-flight checkpoint may contain processed prefix | Mutated caches/pairs may no longer represent any complete prefix | No snapshot; no decode; next sync resets and rebuilds |

**Important native nuance:** `s->checkpoint.len` and `g->pos` advance even while invalid. `ds4_session_pos` returns `checkpoint.len` (`85233–85235`); the field name does not imply a valid committed prefix. Native gates snapshot/decode by validity, not by keeping a numeric position frozen. After mutation there is no automatic rollback to oldC. On a continued append, `checkpoint_valid` can still carry its entry value during the first in-flight sweep; `g->valid=false` already blocks snapshot/decode, and the session flag is cleared after successful range processing. Eligibility is the conjunction of complete graph/session state, not either flag alone. ds41f's separate public slot0 design below is a deliberate representation adaptation, not a claim about native positions.

Cancellation after a **non-pending, fully completed** sweep may be finalized with logits and a valid partial checkpoint before returning interrupted (`pending_logits` processing precedes the final `interrupted` return). Cancellation while `decoder_pending` is true instead exits invalid before logits. Preserve that distinction; do not infer that every interruption yields a reusable partial checkpoint.

## 4. Concrete multi-sweep geometry

Assumptions match the pinned test: single-session Metal streaming, ctx57344, effective carry capacity16384, normal wide/suffix/defer policy enabled, no encoder residency. `DS41_PREFILL_CAP=8192`; a16384 sweep uses8192 encoder tiles, an8192 sweep uses4096 tiles. Wide sweep counts are rounded to2048. Tiny tails1/3 are below every relevant short-prefill threshold and are token steps. Capacity here is the effective field used for planning; the test reduces it after graph allocation.

Ranges are half-open token indices. "Decoder completes" means logical state at the range end using the exact suffix, not replay of every preceding decoder row.

### A: carry16384, target24577, initial C=0

| Range | Count / policy | Work / state after successful range |
|---|---|---|
| `[0,16384)` |16384; remaining tail8193 >=8192, **defer** | Encoder0..19 + full20 source; E16384, D0, invalid. Retain encoder windows/owners/pairs,20 KV/index source and Engram history; no old HC cone is needed for next sweep |
| `[16384,24576)` |8192; `resume_encoder=true`, `encoder_only=false` | New encoder range + new20 source, then20..39 suffix reconstruction; E=D24576, graph valid but session output pending |
| `[24576,24577)` |1; no pending decoder | Ordinary all-layer step; E=D24577 |
| Final output | Native head | C=E=D24577; snapshot eligible |

The final wide decoder uses encoder-final inputs `[22035,24576)` (2541 rows), layer20 queries `[22162,24576)` (2414 rows), and layer39 queries `[24575,24576)` (one row). Its127 preceding local inputs are prepared at each layer. The final one-token step then extends all caches to24577. It does **not** require an encoder-final full history of24577 rows.

Two wide20-source preparations account for16384+8192 rows, then normal source production for the one-token tail. "Exactly one source prepare" from the single8192 P3/P4 smoke must become **once per encoder range**, not once per entire long append.

### B: carry16384, target49155 — the actual continued test

The test does **not** start this case at zero. After case A it evaluates one more token on both sessions (`1123`), so **C=24578** and append length is24577.

| Range | Count / policy | State/work |
|---|---|---|
| `[24578,40962)` |16384; tail8193, **defer** | Encoder/source ahead toE40962; D=C24578; invalid |
| `[40962,49154)` |8192; internal resume, **complete decoder** | Fresh encoder/source range plus final suffix; E=D49154 |
| `[49154,49155)` |1; ordinary step | E=D49155 |
| Final output | Commit | ValidC49155 |

Final wide input cone is `[46613,49154)`; layer20 query span `[46740,49154)`; layer39 query `[49153,49154)`. Previous deferred HC is not replayed. The test then save/restores and evaluates the next token, reaching49156 and comparing logits.

### B from a fresh checkpoint (additional derived example)

For initialC0 and target49155: `[0,16384)` defer; `[16384,32768)` defer (remaining after it16387); `[32768,49152)` completes encoder+suffix decoder (tail3 is too small to defer); then three ordinary steps to49155 and final output. E advances16384→32768→49152; D remains0 during the first two ranges, then catches up49152 and advances with the tail. Three wide source preparations, not two. Final wide input cone `[46611,49152)`; queries20 `[46738,49152)`, queries39 `[49151,49152)`.

This fresh example follows the production conditions but is **not** the49155 initial state exercised by `check_deferred_decoder`.

### Tiny tail16385

`[0,16384)` leaves remaining1, so `remaining-count >=8192` is false: execute encoder **and decoder** in that sweep. Graph becomes valid, though session head is still pending. Run one ordinary token step, then final output and valid frontier16385. No extra encoder sweep is added just to manufacture a resumable tail. The callback would cancel if a completed sweep had an invalid graph; its `cancel_at` remains-1 in the test.

### Why no old cone crosses a pending boundary

Defer only if at least8192 tokens remain. With fixed admissible carry capacity >=16384, the next rounded wide sweep contains at least8192 new tokens. That exceeds the2541 input cone. The last pending sweep therefore always has enough **new** encoder-final inputs to reconstruct all required decoder layers. This proof, including stable capacity and fallback-policy preconditions, belongs in the planner. Reject unsupported pending-to-small-range transitions rather than inventing decoder replay from missing HC.

## 5. Retained state and exact lifetime

For decoder layerL,20<=L<=39:

```text
query rows Q(L) = 1 + (39-L)*127
input dependency rows R(L) = Q(L) + 127 = 1 + (40-L)*127
max Q = 2414 (layer20)
max R = 2541 (layer20)
last layer input R(39) = 128, query Q(39) = 1
```

The current local `decoder_suffix_rows=2414` is **query count**, not the maximum input cone. Local prepare requires the additional127 preceding h/pre rows. These2541 rows are needed **inside the completing sweep**, not as an old full-prefix tensor retained across successful encoder-only boundaries.

| Item | Classification | Must survive where / release rule |
|---|---|---|
| Full HC carry for current encoder range | Arena-only working state; must survive between encoder layers | Needed through0..19 and full20-source publication for this range only. **Must release/overwrite at successful encoder-only boundary**; zero old HC rows are required by the next new encoder sweep |
| `pre` / native FFN split mix | Arena-only working state | Same lifetime as its matching HC rows; do not retain pre without the row's h. Final completing range keeps corresponding2541-row input mix, then shrinks with cone |
| Encoder-final h/pre | Must survive until full source is produced and, for completing sweep, until cone is detached | **Not cross-sweep persistent.** Old encoder-only source-complete range can be dropped entirely. Final range retains exactlyR(20)=2541 rows, not `plan.count` |
| Decoder h/pre input/output cone | Arena-only deferred-completion state | During final decoder: bounded2541 input rows, outputs2414 and subsequently shrinking. Release consumed prefixes; final one-row h/pre only for optional final output diagnostics |
| Full encoder source input | Must survive until source operation/materialization completes | Cannot shrink to2541 **before** all new20 keys are generated. May process range as reviewed source batches, but partition/dtype semantics must be fixed and qualified, not opportunistically optimized |
| Encoder local window KV0..19 | Live-cache persistent; must survive | Actual latest encoder window atE, including non-source layers; next encoder range consumes it |
| Decoder local window KV20..39 | Live-cache persistent allocation; old contents **not authoritative atE** while pending | Keep qualified cache objects, but final127 prepare replaces stale dependency rows. Do not consume stale previous-window rows for a skipped range. Final state becomes complete atD=E |
| Compressed KV owners2/8/14/20 | Live-cache persistent; must survive | All accepted encoder-derived keys up toE (floor(E/ratio));20 produced once for each new range even while decoder pending. Future decoder uses cumulative owner20 |
| Index K owners2/8/14/20 | Live-cache persistent; must survive | Same cumulative coverage/dtype as KV. Refresh layers24/28/32/36 do not create a new persistent key owner |
| Compressor pending KV/gates | Live-cache persistent; must survive | Official ratio/absolute-position remainder at source's physical frontier. Native first3 owners ratio2, owner20 ratio1; use loaded oMLX ratios, never hardcode native storage into MLX. Ratio1 pending state remains official empty representation |
| Encoder row `idx`, decoder `idx` and candidates/masks | Arena-only row spans; can reconstruct future ranges/tokens | Needed only for actual downstream queries in the same range/cone. Source-only20 creates **no full-range query idx/candidates**. Release old range spans; decoder candidates need at mostQ(20)=2414 rows, aligned to absolute spans, and shrink with consumers |
| Engram hashes for current range | Arena-only; must survive until layers1/14 consume | Recompute from complete tokens plus correct preceding normalized history; release after encoder consumption/source boundary. Not an old-prefix tensor to retain |
| Normalized Engram history | Live-cache persistent/provisional while invalid; must survive | Next encoder range needs history atE, notC. Native next-history is installed only after successful sweep. Snapshot reconstruction from token history is possible; hash generation must not become a second persistent-history authority |
| Token history | Request-owned; must survive | Complete desired request prefix for deterministic reset/replay and final P5 seed. Separate committed-lengthC from processed encoder-lengthE. Last arena slice is not full history |
| Publication coverage/frontiers/producer identities | Metadata only; must survive | Persist source coverage/readiness across sweeps. Handles refer to live-cache tensors; no second KV/index store. Row-span metadata scoped to current range |
| Public cache slot0 / checkpoint capability | Metadata only | See section8: retain last advertisedC during active invalid append, independently track actual positions; converge all40 toT only at final seal. Matching integers alone are not validity |
| Native final residual/pre and logits | Last-row/output state, not owner-cache persistence | Native needs them for final head and logits snapshot. ds41f serving retains its qualified suppressed-prefix-logits contract; last-row HC may be released after optional diagnostic output/cache seal |
| Weight residency/prefetch workspaces | Not minimal deferred model state | Outside P6; existing loaded modules remain authority. No P7 residency or overlap work in this package |

**Physical-lifetime requirement for MLX:** a suffix view/alias can pin a full parent buffer. Merely naming a2541-row view does not satisfy bounded retention. At the source-complete/final-cone boundary, materialize persistent source outputs at an explicit boundary, detach only the bounded h/pre cone into arena-owned storage if necessary, and drop full parent/encoder-final/graph aliases. This is an arena lifetime operation, not a cache repack or new persistent authority. Qualification must observe actual backing/graph ownership, not just `shape[1]`. Native reusable capacity is workspace, not proof that prior HC is semantically live.

## 6. Current ds41f gap matrix

Local paths refer to the implementation at `5cea84d`. These gaps are **not fixed by this document**.

| Current concept | Real pinned behavior | Status | Required P6 change |
|---|---|---|---|
| Native `ds41f_prefill_count(ctx,remaining)` (`native/...c:88ff`) | Uses carry capacity, not just encoder tile cap; dynamic request policy | **Insufficient:** local count bounded by prefill cap, normally8192; no actual16384 carry sweep | Explicit capacity/absolute-origin/policy input and faithful count selection; qualify16K sweeps before using deferral |
| `encoder_only` | Encoder0..19 **then full20-source publication**, stop invalid | **Incomplete:** local C breaks at20 with `ENCODER_ONLY_COMPLETE_INVALID`, no20 prepare/publication command | Source-only20 command and acknowledged producer boundary before stop; never expose commit |
| `resume_encoder` | Internal permission to run another new-token encoder sweep from an invalid graph in the same sync | **Misleading:** stored in metadata/eligibility only; doesn't change emitted work; Python correctly blocks it | Replace raw flag with coordinator-owned pending cursor/segment modes; no unconditional legacy guard removal |
| `defer_decoder_candidate` | Determines actual skip, thresholdcount>=16384 and tail>=8192, with capability/environment checks | **Misleading:** local thresholdcount>=8192; full0..39 still emitted unless independently encoder_only | Make decision determine encoder-source-only vs completing sweep before command emission, using pinned thresholds/capability |
| `ENCODER_ONLY_COMPLETE_INVALID` | Reached only after20 cumulative keys have been produced; advances private encoder position/history | **Incomplete marker** | Coupled source/readiness/cursor transition and release boundary, not just an event |
| `DECODER_PENDING_INVALID` | Decoder20..39 was not executed for this sweep; next sweep joins pending append | **Contradictory scaffold:** emitted after decoder work and may be followed by output/commit | Emit only at true source-only boundary; no output/final commit while pending |
| `SweepPhase.DEFERRED_DECODER` | Real decoder work occurs in final new-range sweep after its encoder | **Label only:** native phase4 denotes pending marker, not deferred query commands | Typed completion subphase with exact source/cone/position dependencies; no invented replay of all old ranges |
| `survives_encoder_only` | Persistent caches/history survive; obsolete range HC/masks need not | **Overbroad:** full `batch_cur_hc`, carry residual/pre/block_mask marked surviving | Separate working capacity, semantic live rows and persistent ownership; no old full HC/masks across source-complete boundary |
| `survives_deferred_decoder` | Final-cone rows/row spans live during completion; global sources persist in caches | **Not evidence:** flags don't release tensor references | Enforce last-use/alias/backing/graph retirement at actual transitions |
| `decoder_suffix_rows` | Input2541 including127 prepare; query2414 | **Wrong as maximum retained input size**, currently2414; metadata allocation/view isn't actual compact store | Explicit h AND pre input cone with absolute origin, shrinking output/candidate spans |
| `RequestArena.encoder_final_h/pre` (`arena.py:349ff`) | Same-range transient source input; no old encoder-final history needed | **Full-range aliases** currently held; no cross-sweep retirement | Consume source, detach final cone only when completing, clear parents/aliases at boundary |
| Publication transaction lifetime (`publications.py:147ff`) | One append can have several provisional encoder sweeps before externally valid checkpoint | **Per-sweep:** `begin_transaction` clears visibility; metadata fail does not undo mutated caches | Append transaction + scoped sweep visibility + persistent producer coverage; final one external seal |
| `LivePrefillContinuation` (`executor.py:25ff`) | Only valid checkpoints externally continue; pending resume is private | **Valid-complete substrate only:** checks equal offsets, no pending lifecycle/cursor authority | Retain valid-entry role; never use it to smuggle pending state as committed continuation |
| Slot0 / `_advance_cache_layer` (`block_runner.py:222ff`) | Native one internal pos may be encoder-ahead while validity false | **Unsafe for P6:** currently updates each layer's public offset on every command | Separate actual physical cursors from advertised offsets; all40 final convergence only at whole-append seal |
| Setup/history (`executor.py:94ff`) | Hash next-history installed after successful encoder/source sweep | **Eager:** setup updates cache0 history before work; arena holds one slice | Provisional next-history with explicitE ownership; install at encoder success, external validity only at final seal |
| Per-sweep final commit (`block_runner.py:145ff`) | Session checkpoint valid only after all pending decoder work and required final output | **Too small a unit** | Coordinator suppresses intermediate external commits; final readiness verifies all physical coverage, not just slot0 numbers |
| Failure bookkeeping | Native invalid graph resets/replays; no checkpoint rollback by truncation | **Incomplete:** publication `fail()` discards metadata but cache mutation remains | Invalidate whole append and every old capability; deterministic reset/discard/rebuild from request tokens |
| Existing P5 admission | Valid committed full cache only, exact prefix length, one transfer | **Reusable final boundary** | Deliver one final committed setup/cache with complete history. Pending setup never admitted; no P5 code changed here |

Local native planner source proof: `native/ds41f_prefill_native.c:153–171` records flags/metadata; `190–194` stops encoder_only without20-source prepare; `196–239` otherwise emits normal layers0..39; `241–252` emits pending marker then non-encoder-only output/commit. Therefore toggling existing flags is **not** an implementation of P6. `RequestArena.apply_command` only checks invalidity for those two markers; it doesn't perform deferred math or cursor management.

## 7. Architecture decision: Option C, matching the actual nested authority

**Choose a request/session-level append transaction coordinating real capacity-bounded layer sweeps, with a completing sweep that includes its new encoder range and decoder suffix.** This is a refined coordinator design, not multiple *unchanged* current SweepPlans plus an arbitrary decoder-only resume.

Reasons:

1. Native session sync owns target history, capacity policy, pending status, interruption and final checkpoint. One current `SweepPlan.count` is not that authority.
2. A final pending sweep still executes a new encoder range. No old full-prefix HC archive or independent decoder catch-up loop exists.
3. True source-only sweeps, their20-key publication, and physical cursors are missing from current native plans. Existing plans must be **extended/replaced for P6 segments**, not enabled by flag passthrough.
4. A monolithic enriched native plan (Option A) could precompute geometry, but cannot alone supply request ownership/validity/recovery. The native component remains a command/geometry authority; the Python owner controls the one append transaction and execution of those commands.

### Complete target package

Conceptual components (names are proposed, not added in this task):

- **Append owner / `DeferredPrefillAppend`**: one live request cache, complete request prefix, last committedC, current physicalE/D and per-layer/source coverage, normalized history position, pending/valid/failed state, exclusive execution/admission lease.
- **`AppendPlan` envelope**: targetT, capacity, validated topology/policy, ordered segment boundaries/modes. Absolute positions and pending-to-final-cone proof are part of the plan; not inferred from cache slot0.
- **Accurate native segment command plans**:
  - `ENCODER_SOURCE_ONLY`:0..19,20 persistent source prepare, internal source publication/readiness, invalid completion, arena retirement;
  - `FINAL_ENCODER_DECODER`: new0..19, new20 persistent source, bounded cone transition,20..39 shrinking queries/prepares, complete physical state;
  - `ORDINARY_COMPLETE_RANGE`: normal non-pending small range or token step, following the actual count policy;
  - one append-level final seal, not a commit for every internal range.
- **Command runner extension**: same loaded oMLX HC/attention/MoE math; commands remain the only transformer execution loop. Accept explicit private absolute origin and per-layer cursor ownership. Never invoke whole `LanguageModel._forward` to process an invalid-state tail: it would interpret public offsets as physical positions.
- **Arena/deferred lifetime implementation**: full current encoder working range; no old HC retention at source-only boundary; only final2541-row h/pre input cone and needed query/candidate rows during completion. No second persistent tensor store.
- **Append publication owner**: metadata references to cache2/3 and row spans; phase/sweep visibility distinguished from external commit.
- **Final bridge**: final committed setup compatible with existing `LivePrefillResult.from_committed` and `handoff_to_generation`. Its final arena records the actual last range's base/count, whose sum isT, while its committed transaction represents the entire completed append. The private range origin must no longer be inferred from frozen public slot0. Complete prefix history comes from append owner, not this final arena slice; terminal held out exactly as P5. Do not fabricate cache state or commit flags merely to pass P5 validation.

Initial P6 qualification is for the intended text-only, MTP-OFF, single-request DeepSeek-V4.1-Flash topology with a fixed validated capacity/policy. Unsupported topology, multimodal/history requirements or capacity changes must be rejected before entering pending state (or use an explicitly chosen complete non-deferred path from a valid entry). There is no fallback from an arbitrary pending state to a short decoder replay.

One cache list is preserved on every successful append through P5 admission. Only failure recovery may discard an invalid list and reinitialize a request cache; this is not handoff repacking and must not leave another active cache authority.

The native `encoder_resident` policy flag describes the separate borrowed-expert-budget mechanism, not simply that model weights are loaded in MLX memory. The P6 capacity/policy contract must preserve that distinction; it must not silently disable deferral because the official MLX modules are resident.

The legacy `resume_encoder=True` interface is not the future authorization mechanism. Keep it guarded now; future P6 admission must require an owner-issued pending cursor matching the active append, source coverage, next range and stable capacity. A boolean alone cannot authorize execution from arbitrary invalid state.

## 8. ds41f checkpoint/frontier semantics

**Representation decision:** keep all public slot0 offsets at last advertisedC during an active invalid append, and track actual per-layer positions/coverage in owner metadata. This differs from native `g->pos`, which advances internally. It preserves native validity semantics without falsely advertising encoder-ahead work as a decodable MLX cache.

Requirements:

1. BEGIN_APPEND revokes the previous valid capability and marks the whole append/its setups invalid before the first cache mutation. Old committed setup/result/continuation handles cannot remain executable. C is only last-valid historical metadata; in-place state atC is not guaranteed to remain recoverable.
2. New range absolute starts come from privateE/per-layer cursors, **not** `arena.base_frontier` blindly derived from public slot0. Compressor remainders, prefix trim/appends, RoPE, row publications and local prepares all use actual positions.
3. P6 source-only completion does not run `_advance_cache_layer` as currently written; records successful encoder/source coverage instead. Decoder pending retains invalid transaction status. Equal slot0=C is insufficient for continuation/admission.
4. While invalid: **P5 forbidden, decode forbidden, snapshot/export forbidden, no externally committed cache capability**. Access to the cache for internal math is owner-private. Any public continuation/admission boundary must consult validity, not just40 equal integers. Existing P5's transaction check is retained as a final barrier; stale old setups must be revoked so they cannot pass it.
5. Final readiness requires actual encoder/source coverageT, decoder completenessT, all latest windows, legitimate pending state and Engram historyT. Do not manufacture completeness by merely setting slot0.
6. At one request-exclusive final seal, publish all40 slot0=T and committed metadata/history after acknowledged MLX work. Only after the complete seal becomes visible may the valid capability/P5 setup be issued. Partial offset installation or evaluation failure remains invalid and requires rebuild; no observer may treat mixed offset writes as a checkpoint.
7. Serving final-prefix logits remain suppressed. Native FINAL_OUTPUT includes a head because native checkpoint/sampling/snapshot includes logits; ds41f's corresponding final barrier is **cache readiness**, with optional bounded output diagnostics outside the serving path. Do not reintroduce a serving prefix-logits requirement or include P5's held-out terminal inT.

An ownership fence is required even if a raw cache list has old matching offsets. Do not expose raw pending lists through `LivePrefillContinuation.from_cache` or direct generation admission; passive aliases are not capabilities. A future P6 implementation must test stale-handle rejection, not assume new state labels make old APIs safe.

## 9. Publication across several sweeps

Two visibility levels are required inside one append, plus external final validity:

- **Producer-private pending**: loaded source operations updating their owner cache; inaccessible before the corresponding planned source boundary.
- **Append-internal published**: same live-cache tensor references, with producer and physical coverage. Encoder consumers see the current encoder producer at normal layer frontiers. Owner20 cumulative state remains available across source-only ranges to its own producer and, at the completing sweep's true source frontier, decoder consumers.
- **Externally committed**: only after full decoder/tail readiness and final seal. Internal publication is not snapshot/P5 eligibility.

Retain cache1/2/3/4/5 and source-coverage metadata across encoder sweeps. A scoped new sweep may reset ephemeral `idx`/candidate spans and current-layer shared visibility, but must not forget owner20's cumulative source or reappend it. Metadata can reference/recover owner handles from the cache; never rebuild or duplicate their tensors. A map of latest producer generations is metadata, not another persistent KV store. An old layer20 `kv` alias must not leak into an encoder consumer that requires the current layer2/8/14 producer.

Source-only20 publishes KV/index K without idx/candidates. Its internal source-ready boundary does not authorize layer21 consumers: layer20's own query path may consume that producer state, but layer21 must wait for the completing layer20 query publication frontier, including its required idx/candidate spans. Final20 query commands generate only their exact row spans; refresh layers consume KV/index K and relevant candidates and replace idx as in qualified P3/P4. Row-span coordinates must include the absolute range origin (or explicit origin+relative offsets), so reused sweep offsets cannot collide across ranges. Old range selections are not checkpoint-persistent and must be retired.

`PublicationManager.begin_transaction()` as currently called at every `BEGIN_INVALIDATE` is not enough. P6 needs an append begin/final commit/abort and subordinate visibility scopes; `committed_*` references are not an undo log. `fail()` discarding pointers/metadata does **not** restore windows, pending pairs or overwritten tensors. An aborted append invalidates all its state authority; recovery resets/discards and replays rather than claiming old committed pointers are intact.

## 10. Token and Engram history reconciliation

Request ownership retains the full **desired** prefix token sequence toT (terminal excluded), a committed lengthC, and processed encoder lengthE. Native `checkpoint` grows provisionally toE despite its name; copy that meaning, not the misleading name. Each range arena owns only its token slice/hashes.

The qualified tokenizer-derived hasher processes new tokens from normalized history atE. Construct `next_history` provisionally; install it as private working history only after the entire encoder/source boundary succeeds. That history may live in cache0 slot6 as the one current working representation, tagged by privateE and invalid owner status. It is not externally committed history until final seal atT. No separate persistent history tensor archive is required; request tokens allow deterministic reconstruction after reset.

The current eager history write during setup must therefore be separated from successful encoder completion. On a failed partial sweep, neither precomputed hashes nor an advanced history array imply a valid prefix. OldC history cannot be recovered by assuming slot6 was never changed. On final success historyT and complete request prefix become externally coherent together, and P5 receives **all** prefix tokens, never the last range's slice.

## 11. Cancellation, failure and recovery

- Poll cancellation at existing semantic/command boundaries; no output/admission while a source or decoder layer is partial.
- If cancellation occurs at a source-only boundary or during pending completion: mark failed/invalid, stop future commands, expose no decoder commit, revoke old setup/continuation/P5 capabilities, retire all transient h/pre/hashes/idx/candidate references after in-flight work is drained.
- Source/window/pair buffers may already have been changed. **No rollback by trimming lengths, rewinding slot0, or discarding PublicationManager records.** Native reset explicitly relies on validity/length visibility (`40303–40309`), and rewind warns compressors cannot be rolled back by row counts (`85218ff`).
- Deterministic recovery: invalidate/discard the failed cache authority, reset/reinitialize the request's single working cache, reset normalized history/cursors/publications, then rebuild the desired prefix from complete request token ownership through the validated planner. A retry with a different/shorter prompt is supported; reproduce the pinned failed24577→fresh129 case. Rebuild replay is explicit recovery, not zero-replay P5 bootstrap, and must be labeled accordingly.
- Pending state is **not** an externally resumable checkpoint. Native `decoder_pending` is local to a sync invocation; the next sync resets invalid state, even if the previous encoder-only sweep completed. Do not promise warm pending resume after cancellation/crash or implement disk persistence in P6.
- To preserve native completed non-pending interruption behavior, cancellation at an acknowledged physically complete non-pending range will seal that partial prefix (with token history exactly through its frontier) before returning interrupted, provided all final-state checks succeed. A failed seal or cancellation inside an incomplete range uses invalid/rebuild. The result must say **valid partial checkpoint, target not completed**; it must not automatically invoke P5 as though the requested targetT were reached. Never seal merely because an encoder boundary or progress counter was reached.

No checkpoint disk save/restore subsystem is required for ds41f P6. Its **eligibility** and rejection semantics are required; any existing/future export entry must refuse pending state. Native snapshot assertions remain explanatory authority, not a mandate to add persistence now.

## 12. Complete implementation package / sequence

Implement the whole path before any executable resume promotion:

```text
valid exclusive live-cache capability + complete request history
 -> append geometry/capacity plan and transaction invalidation
 -> one or more full new-token encoder/source sweeps
 -> persistent cache sources/history only across pending boundaries
 -> final new-token encoder range + minimal2541-row input cone
 -> exact deferred suffix layers20..39 + legitimate ordinary tail
 -> final actual-state readiness and40-frontier seal
 -> same committed live cache and complete prefix passed to existing P5
 -> held-out terminal bootstrap, no prefix replay
```

Package-level delivery order (tests may be staged; these are not partially enabled features):

1. **Authority-complete planning and owner protocol:** new append envelope/capacity geometry, segment modes, privateE/D/source cursors, valid/invalid/failure capability protocol; exact native fixture plans for A/B/tiny/continued cases. Existing legacy guard remains until the complete package is accepted.
2. **Connected execution and lifetimes:** source-only20 math boundary, reuse reviewed encoder/suffix modules with explicit positions, full-source-before-compaction rule, zero old HC across pending ranges, bounded final h/pre and selection storage, acknowledged retirement of parents/graphs. Correct ordinary tail command ordering is part of this step, not later optimization.
3. **One append publication/history/frontier lifecycle:** subordinate scopes, producer-private vs internally published coverage, provisional Engram history, actual-state final readiness, externally atomic commit into the same list, and final P5-compatible setup. No second persistent tensor authority.
4. **Interruption/recovery and all admission fences:** drain/invalidate/discard/rebuild protocol, stale handle rejection, no arbitrary invalid resume, no handoff/decode/export while pending; validated partial-prefix sealing on completed non-pending cancellation only. These are mandatory before real promotion.
5. **Complete package qualification:** structural operation-recording tests, then real non-deferred geometry controls, real candidate and interruption/tiny/continued runs, finally P5 terminal bootstrap/short decode. Only after all pass expose the owner-mediated P6 entry in the new package; production selector and P7 still remain separate.

Do not ship “enable the flag, try partial resume, then fix lifetime later.” Any math mismatch stops at the first demonstrated boundary under the current correctness policy; no sequence of speculative optimizations.

## 13. Structural acceptance criteria

The P6 package is accepted only if all of the following are observed:

- Count/skip/resume/final-tail geometry derives from the pinned request loop, including the16384/8192 predicate and carry capacity, not local candidate flags.
- Source-only segments execute exactly0..19 plus20 source; **zero decoder query/MoE/output work** before true completion. Keys cover every accepted encoder token once; decoder queries never append/regenerate source ranges.
- Each completing wide range contains its2541-row dependency input cone; layerL local prepare uses127 exact absolute predecessors and queriesQ(L). No ordinary Block fallback for suffix queries.
- Old encoder-range HC/pre/hashes/selections do not survive a pending boundary. Completing decoder live data is bounded by input2541, query2414 and shrinking subsequent cones. Views cannot secretly retain full-range parents/graphs.
- One cache list/owner persists across successful ranges; private actual positions are independent of public slot0; legitimate pending compressed state/history reachesE without advertising a checkpoint.
- Provisional source publication persists internally across sweeps but does not permit external admission. Exact spans do not collide when relative offsets restart.
- No P5/decode/export/snapshot capability from pending, partial, failed, or stale-handle state; all40 public offsetsT and final physical geometry/history are coherent before final commit.
- Deterministic reset/rebuild after cancellation is demonstrated, not rollback by metadata. Tiny-tail16385 never leaves completed-sweep pending work.
- Complete request token history is the P5 seed; terminal held out and forwarded once. Successful handoff uses the same list/tensors, zero export/repack/prefix replay, as in P5 qualification.
- Current51 regression tests and all P3–P5 real evidence remain valid within their original scope. No P7 or production completion field is promoted merely by P6 correctness.

## 14. Exact real-MLX qualification plan (future, not run here)

Environment must be recorded before every model run: imported oMLX version/path/language digest/revision/local changes, MLX version/device, checkpoint config/index digests, sampler, token fixture, segment geometry and any enabled capability policy. Use the qualified official loader (`preserve_mtp=False`, `engram_ssd_offload=True`), no runtime selector or throughput measurement.

### Fixed fixtures

Use the previous compatibility fixture (`prefix=[1]*T`, terminal3) for the2048/8192 regression runs. For every long/tiny/recovery/continued matrix case, define a varied deterministic stream `token[i] = 16 + ((37*i) % 4096)` for prefix/held-out terminal positions and optional teacher-forced diagnostic next-inputs (normal P5 generation still uses its greedy sampler). These IDs lie within the intended129280-token vocabulary and avoid the pad/image special IDs; verify that against the recorded loaded config before running. Candidate and controls receive identical slices; hold out `token[T]` for P5. Record the formula and a canonical int32 token-stream digest in qualification diagnostics. This makes history/range mistakes observable rather than hiding them behind identical repeated tokens. A pinned-test-style tokenized chat fixture may be added as bounded secondary evidence with its exact input digest, not as a broad evaluation suite.

### Controls and comparisons

**Control 1 — already-qualified non-deferred path:** same complete request prefix through current P3/P4 complete sweeps and `LivePrefillContinuation`, with existing serving prefix-logits suppression. This establishes final geometry/history/P5 behavior against the connected baseline. Current planner caps8192 and may choose different tail partitions, so do not silently require bit equality for every floating trajectory against this control.

**Control 2 — matched-geometry non-deferred control:** corrected native/append geometry with deferral disabled, same carry16384, encoder tile sizes, absolute starts, token steps, loaded operators and evaluation policy as candidate. Every sweep completes decoder. This mirrors pinned test control. Qualify its new16384 non-deferred geometry first using reviewed P3/P4 math; it is **not automatically qualified** by the previous8192 smoke.

**Candidate:** same matched geometry with actual encoder-source-only segments and final decoder completion. No independent layer loop. Comparisons occur at final valid prefix, not during mixed frontiers.

Required final checks:

- all40 offsets exactlyT, cache compression layout and packed widths/lengths, all local128-row geometry and proper remainder slots;
- Engram normalized history and complete token-history seed exact (discrete state);
- same-input encoder-side source/window state and layer20 KV/index K: exact packed bytes where identical loaded operators/inputs/partitions make equality meaningful; compare coverage/generation counts and physical source preservation through decoder queries;
- matched-control encoder boundaries should be independent of skipped decoder work. Capture the earliest boundary if they diverge; distinguish HC, compressor, index K, local prepare, ranking, sparse attention, MoE, publication or lifecycle. Do not fit tolerances to observed differences;
- query idx/candidates obey reviewed causal/ranking semantics on backend-computed values and correct row spans; no full-range idx side effect at source-only20;
- optional final-token hidden/logit diagnostics outside serving, using captured equivalent inputs and the existing `docs/correctness.md` policy. Do not impose universal equality against a different physical replay/partition trajectory;
- existing P5 terminal bootstrap after final commit: one held-out terminal, same-cache identity, zero prefix replay/export/repack, offsetsT→T+1, first generated token and two bounded consistent next steps. Optional independent same-backend terminal `_forward` comparison is meaningful only from independently reproduced equivalent candidate cache.

### Required run matrix

| Case | Setup and execution | Required observation |
|---|---|---|
| Small compatibility | Prefix2048 and8192, existing P3–P5 smokes | No regression; baseline admission and exact suffix source split intact |
| Tiny tail | C0, capacity16384, prefixT16385 (full prompt16386 with holdout) |16384 completes encoder+decoder, one ordinary tail step; never a completed-sweep pending state; final valid and P5 bootstrap |
| A | C0, capacity16384, prefixT24577 (full prompt24578) |16384 source-only,8192 new encoder+decoder, one normal token; final matched-control/source/history/geometry and P5 |
| B fresh / more than two ranges | C0, capacity16384, prefixT49155 (full prompt49156) | Two source-only16384 ranges, completing16384 range, three normal tokens; only final bounded cone retained |
| B continued (pinned initial condition) | Start from a valid cache atC24578; prefixT49155 |16384 source-only +8192 complete +one token; preserve pre-C source/pending/history correctly. Construct C using qualified non-deferred append before P5, not by reusing a P5-revoked producer |
| Odd/remainder valid entry | ValidC with loaded source compression remainder, then defer-eligible append | Correct pending KV/gates across new encoder ranges; historyatE, notC; exact source reference checks where meaningful |
| Interrupt at first pending boundary | A, cancel when E16384 and D0 | No valid/P5/decode/export capability; public validity false; no falsely advertised encoder frontier |
| Interrupt during encoder/source/final decoder | Inject cancellation at representative command boundaries, including20 source and a later decoder layer | No partial external publication; invalidate/drain/transient release; stale-handle admission rejected |
| Recovery | From failed24577 attempt, retry requested prefix129, then repeat a long target | Reset/discard/rebuild, equivalent valid control state; no reuse of polluted pair/window/publication history |
| Completed non-pending interruption | Cancel at acknowledged complete non-pending segment | Validated partial-prefix seal with coherent history (native-equivalent); a failed readiness/seal remains invalid/rebuild. Requested target is still incomplete; never encoder-only commit |

Every matrix row includes explicit owner-validity, physicalE/D/source coverage, public offsets, token/history positions and retained backing/graph references at boundaries. Operation-recording adapters can establish structure first, but cannot substitute for these real model runs. Real row-frontier metadata must be taken from actual cache/operations, not only optimistic counters.

No disk snapshot implementation is required for these runs. Exercise the eligibility guard through admission/export entry points; a nonexistent snapshot path must not be invented just to manufacture a test pass. No tokens/sec, speed deltas, cache-budget tuning, scheduling changes or speculative kernel work.

## 15. Decision / stop

The local `resume_encoder` scaffolding is **not a usable P6 execution model**. Keep its guard and replace the conceptual flag-driven design with the owner-mediated append package above, extending/replacing the native segment planner where necessary. Preserve the real mechanism: cumulative encoder/source state across several new-token sweeps, no old HC archive, one final-range exact decoder cone, legitimate tail, one valid cache seal, existing P5.

Only this document and a short main-plan clarification are changed in this task. P6 remains unimplemented; P7, runtime selection, P5 code, and benchmarking remain untouched.
