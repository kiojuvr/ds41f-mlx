# History-preserving repository cleanup

## Scope and findings

Baseline: `0ba382a` (438 reachable commits). No history, tags, remote refs,
Runtime/Harness source, release payloads, dependency locks or fixtures were replaced.
This is an ordinary deletion commit, not a reduction of historical Git objects.

The baseline tracked tree contained 6,674 files / 591,881,207 bytes (564.46 MiB).
`artifacts/` contributed 397,707,609 bytes; `third_party/` 166,696,530 bytes;
`release/` 13,904,393 bytes. The largest files were the 86,664,942-byte Recipe
wheel, 54,537,175-byte oMLX source archive and 32,984,126-byte M50r trace.
The wheels, source archives and reference fixtures remain because release manifests,
identity checks and reproducibility depend on them.

Read-only history inspection found 7,805 reachable blobs totaling 821,962,570
uncompressed bytes, including three approximately 86.6 MB versions of the Recipe
0.2 wheel. These are historical object totals, not current checkout size or wire
transfer size. Before cleanup, `git count-objects -vH` reported 397.72 MiB loose
objects and a 99.23 MiB pack; local storage includes objects outside current HEAD
and cannot predict server-side packing.

## Deletions and retention decisions

Removed 426 tracked files totaling 1,882,412 bytes:

- Six private Harness session-storage files (62,290 bytes), including stale locks
  and compressed state whose encoded host workspace names caused long paths.
- 374 unreferenced `.log` files, eight `.log.gz` files, 36 `.exit` files and two
  `.pid` files. These are disposable execution output, not test fixtures.

Log/status deletion was deliberately conservative: candidates were limited to
`artifacts/`, and retained if their basename appeared in any other tracked
Markdown, Python, JavaScript-module, JSON, TOML, text, YAML, Rust, shell, CFG or INI
file. This preserves referenced logs even when they may be redundant. No Runtime
or Harness integration uses the deleted session files. Three historical reports
now distinguish captured session state from currently retained evidence. Their
immutable cleanup manifests/digests, projections, inspectors and receipts remain.

The two archived diagnostic preflights previously read the removed compaction
session directory. For historical replay, export
`artifacts/multimodal-runtime/phase3-step3br3-evidence/session-storage/` from
commit `0ba382a` to an **external, short-root directory**, and set
`DS41F_REPLAY_SESSION_STORAGE` to that directory. These historical scripts still
require their original external Harness environment; they are not current CI.
Do not restore private session state to the tracked tree.

Retained workspace-final HTML/screenshots, authenticated disposable inputs,
compressed official-session replay fixtures, milestone reports, failed-case
receipts and numerical reference fixtures have continuing evidentiary value.
Their names alone do not make them safe deletion candidates. In particular,
`artifacts/` was not deleted wholesale.

## Generation and retention convention

- Generate runs under ignored `/artifacts/`; use external temporary directories
  for live Harness homes, sessions and workspaces, as current tools already do.
- Never `git add -f` an entire run directory. Review and explicitly add only
  durable reports, minimal fixtures, source identities and required receipts.
- Exclude logs, PID/exit markers, locks, session stores, render handoff state and
  temporary workspace copies from promoted evidence. Preserve a digest/projection
  instead of live private storage when replay of that storage is unnecessary.
- The existing artifact ignore remains; narrow ignores now also cover
  `session-storage/`, `.render-handoff/` and `.pytest_cache/` wherever generated.
  Ignore rules do not protect tracked files or prevent deliberate force-adds.

## Quantitative result

Final tracked count: 6,249 files.
Final tracked content: 590,006,720 bytes (sum of indexed file contents, not disk
allocation or `.git`). The total includes this report and reference repairs.

Largest remaining immediate artifact groups (bytes):

| Group | Bytes |
| --- | ---: |
| `artifacts/m50r` | 54,723,266 |
| `artifacts/multimodal-runtime` | 43,311,737 |
| `artifacts/m4` | 19,990,914 |
| `artifacts/very-long-context` | 13,614,267 |
| `artifacts/m38` | 12,718,595 |
| `artifacts/m35` | 12,328,295 |
| `artifacts/p8-attribution` | 12,125,366 |
| `artifacts/tool-capability-audit` | 10,933,945 |

## Windows assessment

Static inspection of all indexed paths found no reserved Windows device names,
illegal/control characters, trailing spaces/dots, case-insensitive path collisions
(including directory components), or components exceeding 255 UTF-16 code units.
Before cleanup, six paths were 225–238 characters: a 238-character path exceeds
legacy MAX_PATH with a checkout prefix of only 21 characters (including separator).
After cleanup the longest relative path is 114 characters; all fit below 260
characters with the representative root `C:\Users\example\source\repos\ds41f-mlx`.
No inherently Windows-incompatible paths remain. Arbitrarily deep checkout roots
can still exceed Windows limits; no repository can guarantee otherwise.

This environment is macOS (Darwin). These are static checkout checks, **not an
executed Windows checkout**, nor evidence that the Apple Silicon runtime runs on
Windows. `core.longpaths=true` is not relied upon.

## Validation

- Repository self-containment checker: PASS; 35 current owners, 154 Markdown
  files, 703 live links; retained provenance validated.
- Model-free focused pytest run with `--noconftest`: **45 passed** (documentation
  authority, archived build-release documents, Phase 3 evaluation, release metadata,
  and retained OFF Recipe artifact identity tests).
- Normal pytest invocation: blocked at session setup for all 45 tests because the
  existing venv cannot import installed `ds41f_mlx` in isolated Python. The
  `--noconftest` run explicitly bypassed installed-payload admission; it does not
  constitute installed Runtime or model qualification.
- Ignore probes for new artifact logs, nested session storage, render handoff state
  and pytest cache: PASS. Staged/unstaged whitespace checks: PASS.

## Limitations and unresolved candidates

Existing history and published references remain intact. Full clones still fetch
historical objects, so the reported 321.12 MiB transfer can remain approximately
unchanged (server packing varies). Only the current tree and future accumulation
are improved. Old commits retain the original long paths and checkout limitations.

Referenced logs, large milestone traces, duplicated qualified-source snapshots,
and older release wheels remain conservative follow-up review candidates. Removing
them requires a separate reference/identity audit; this cleanup does not assert
that all remaining evidence is minimal. No history rewriting or push was performed.
