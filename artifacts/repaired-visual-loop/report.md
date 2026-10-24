# Repaired official Harness visual loop — PARTIAL

Evaluation restrictions were removed and meaningful image-grounded correction
succeeded. **Successful eight-image cached admission was not qualified:** the
first eight-occurrence request hit the unchanged paired-byte fallback. No retry,
limit adjustment, conversation reconstruction, push or deployment followed.

## Evaluation repairs and preflight

Started on `Visual-Feedback-Loop` at `61e69ee103a1bbd8e21449192fa5ef4c7a640367`.
Only evaluation infrastructure/evidence changed.

- Restored **2,600 tokens** in both official provider/model configurations and the
  observer; controller checks the effective configuration before loading. Raised
  aggregate output from 18,000 to **52,000**, independently bounded by 50 requests,
  70 tools, 12 renders, 5,400 seconds, 600-second generation fences, 32,768 prepared
  positions and 8 MiB wire bodies. Actual output was 4,298 tokens; neither fence
  interrupted correction. The initial successful edit used **1,241 tokens**,
  exceeding the former 1,000-token restriction without exhausting 2,600.
- Presentation-only tag/CSS-rule line breaks make official `read` show the entire
  165-line source accurately. Original and formatted renders are byte-identical
  to the previous challenge PNG: SHA-256
  `468e14f7490ec77e4ad583f762bda73a5d499bb23cd5801ec7b3c710b077fe7b`.
  No CSS correction or solution was supplied before execution.
- Guard admits only the approved renderer plus `pwd`, `ls index.html`,
  `wc -lc index.html`, and bounded literal `sed` line ranges on `index.html`.
  Official tools still own edits and screenshot acquisition. No unrestricted shell,
  new tool framework or orchestration. Compound commands, mutation, network,
  escalation, background execution and outside-workspace paths remain denied.

[Model-free preflight](model-free-preflight.json), [actual config dump](composed-config.yml)
and [targeted guard/installed-sandbox checks](policy-check.json) passed before model
loading. Source inspection succeeded during the real session; official `read`
returned all source, without the former 2,000-character single-line truncation.

Reused official **Harness 0.2.1-rc.2**, renderer, native Messages/Files, unchanged
registry, paired authority and admitted disposable Flash-1017 Runtime. All 17
Harness package identities and **114 installed Runtime sources** match previous
provenance. Existing wheel base is `dbbaa7289e4b16c89b2a9d437b029bd57ee549e7`;
execution identity remains
`9c439d3fb9ad730996ecca821234bba1ce8817d7b46a2744e171c10a4d7d2ca2`.
Installed/checkpoint admission preceded numerical backend import/loading. No
Runtime, model, kernel, deployed installation or production configuration changed.

## Actual workflow and independent decision

One **fresh** official conversation, `session-381c9db0-c205-4a4c-bc4f-70e2aa6a7f17`,
ran 171.1 seconds after startup. No historical conversation was resumed.

| Outcome | Evidence/result |
|---|---|
| Harness / Files / Vision transport | 8 distinct PNG uploads and File-ID joins; 7 actual fresh encodes |
| Image-grounded corrective edits | Explicit diagnoses, official edits, revised screenshots and model verification |
| Independent visual improvement | Original 4/9 rubric checks → final 8/9; not perfect completion |
| Cached-image continuation | 85 historical occurrence reuses; success through 7 retained screenshots |
| Beyond-seven discount | Attempted at request 28, rejected; **zero successful discounted generations** |

G6 identifies uneven cards, clipped titles/names and missing footer from the
initial image, then completes an official edit. The revised image is consumed at
G9, which confirms full text, aligned cards and visible footer. Independently,
all eight complete session titles/names and both footer messages are now readable.

There are seven executed revisions and six model-verified edit/render cycles, not
seven independent improvements. A registration-spacing experiment introduces a
real regression: its panel drops to about y340. G15 explicitly diagnoses that
image-specific displacement, edits the auto margin, and G18 verifies restored
alignment. Later spacing/density refinements preserve readable content. The final
pill line-height change is minor, not credited as another meaningful defect fix.
The eighth screenshot is independently inspected but never consumed by the model.

[Independent review](independent-review.json) records each image, grounded chains
and remaining defect: the content grid extends **29px** beyond the header/cards/
footer right edge, although everything remains inside 1280×720. All fixed body
text is preserved. The legacy renderer's unrelated quality score is **not** used.
One compound inspection command was denied; one ambiguous edit failed explicitly
before mutation, followed by a more specific edit—not an uncertain-effect retry.

## Terminal admission limit and evidence gap

Request 28 / wire 36 has **17,471 prepared positions**, 2,600 output reservation,
7 historical images plus 1 fresh image: **38,272 total / 4,784 fresh patches**.
Files acquisition and preparation succeed. Before paired restoration/native work,
the existing authority rejects:

`aggregate image patch budget exceeded (paired-byte qualification fallback)`

This identifies the existing **12 MiB projected-pair or 192 MiB retained-root**
qualification gate, not a generation-budget, Files or Vision encoding failure.
The exact two byte values were not retained, so which ceiling failed cannot be
separated. No discount, eighth fresh encode, native generation or publication is
claimed for that request. The 36,864 fresh-patch budget, conditional discount,
both byte guards and all other Runtime constraints remain unchanged.

The existing evaluator archiver incorrectly treats this pre-generation rejection
as an incomplete trace. It retains **all 27 completed raw traces**, but not the
28th in-memory rejection trace; its fail-closed assertion causes Runtime exit 1.
This is a disclosed evaluation-retention gap, not evidence of numerical or paired
state corruption. [Terminal receipt](terminal.json), [stack](runtime.log),
[wire](wire.jsonl) and [retention failure](trace-retention.json) preserve the
available evidence. No recovery or second run was attempted.

## Frontiers, safety and cleanup

[Recomputed summary](summary.json) joins every completed generation, image identity,
fresh/reuse flag, upload, File-ID and canonical frontier. **27 native completions**
publish settled pairs: all 40 Target + 3 DSpark offsets match their canonical
frontier, prediction retired, queue empty, zero replay/repack; every restore equals
the previous frontier. Maximum successful prompt/frontier: **16,523 / 16,609**.
The eighth-image request does not advance that frontier. Maximum wire body:
45,608 bytes. Passive clone-stream aborts after captured `message_stop` are
separated from the single actual terminal failure.

Fresh host reserve was **492.075 GiB**, allowing for the prior measured
359.226 GiB peak plus 100 GiB reserve. Reused finite safety controller unchanged:
360 GiB process/allocator ceiling, 100 GiB reserve floor, zero swap and sustained
pressure checks. Lifetime physical high-water **300.566 GiB**; sampled MLX
active+cache maximum **299.074 GiB**; minimum host reserve **192.099 GiB**;
pressure free minimum 40%; swap usage/ins/outs **zero**. No safety stop occurred.
These are separate overlapping measurements, not additive or allocatable-memory
promises. Graceful serving shutdown/close ran despite the archive assertion;
owned processes exited, renderer exit 0, port freed and disposable root removed.
Both deployed manifests remain unchanged ([final host](host-after.json)).

**PARTIAL:** useful image-grounded correction and cached continuation are established;
the extended admission attempt was refused within the unchanged qualification
envelope. No full task completion or eight-image execution PASS is asserted.
Model-free evidence recomputation: `python3 artifacts/repaired-visual-loop/audit.py`.
Raw evidence and evaluated helper hashes are retained; no follow-on work authorized.
