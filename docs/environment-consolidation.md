# Local environment consolidation

## Current layout after cleanup (2026-10-22)

Only `/Volumes/SDXC-512/ds41f-mlx` is the development repository and registered
worktree; `master` is its sole local development branch. Its internal `.venv`
remains the development/tooling environment, including the running wire observer.

| Variant | Retained installed Runtime | Observed state at cleanup verification |
| --- | --- | --- |
| Original Flash | `/Volumes/SDXC-512/ds41f-0910` | Populated; no model process observed |
| Flash-1017 | `/Volumes/SDXC-512/ds41f-1017` | Populated; ordinary Runtime remained active |

[Operations](operations.md#local-environment-layout) owns current startup and
publication paths. This snapshot is not fresh installed admission, model acceptance
or qualification. Neither Runtime was rebuilt, reinstalled or restarted.

## Approved development cleanup (2026-10-22)

The cleanup was evaluated against `master` at `900291a` and executed only after
explicit operator approval. It removed 1,078 non-overlapping filesystem entries:

- Legacy original-Flash MTP/OFF, Flash-1017 branch and dedicated DeepInfer
  environments, plus obsolete dependency builds and candidate fixtures.
- Temporary test installations, native/package build outputs, logs and abandoned
  Recipe worktrees, including their uncommitted work.
- Local artifact extras: full GPU traces/operand exports, the abandoned packed
  Flash-1017 DSpark candidate, imported scratch clones and duplicate outputs.
- Old project-specific investigation environments, acceptance homes, retired Web
  state and legacy USB KV test/qualification snapshots.
- The stale `/private/tmp/ds41f-stable-e8c719c` worktree skeleton and registration.

All master-tracked files, current test/verification fixtures, required provenance,
release payloads and documentation-link targets were retained. Three additional
Boundary 12c/12d/12e JSON checker inputs and the active observer's wire/status logs
were also retained. Historical receipt identities and Git tags were not rewritten.
Both official checkpoints and Engram assets were protected; their filesystem
metadata was unchanged. Shared caches and other projects were outside cleanup scope.

The Flash-1017 provisioning build directory was removed. Original Flash retains
only `ds41f-0910-build/runtime-wheels`; the active Flash-1017 publication wheel
remains at `/private/tmp/ds41f-harness-inventory-wheel.NxZ0BY`. Installed execution
uses its own packages, native assets and tokenizer, not dependency build exports.
Missing scratch resources must be recreated separately from retained repository
build inputs if needed; do not reuse or mutate production installations for tests.

The six non-master local branches and their obsolete remote-tracking references
were removed. At final remote verification, GitHub still contained
`DeepSeek-V4.1-Flash` alongside `master`: deleting that remote branch was blocked
by unavailable HTTPS credentials. The other four targeted remote branches were
already absent on recheck; the failed authenticated push did not establish their
removal. This is a dated observation, not permission to recreate obsolete branches.

Available space increased by approximately **50.54 GiB** during cleanup:
30.63 GiB on SDXC, 10.74 GiB on internal storage and 9.18 GiB on USB SSD. The
59.87 GiB allocated-size estimate was not a promise of freed APFS extents.
Detailed local records are `/private/tmp/ds41f-cleanup-review-master-900291a.json`,
`/private/tmp/ds41f-cleanup-local-execution-900291a.jsonl` and
`/private/tmp/ds41f-cleanup-local-result-900291a.json`. These temporary audit files
are not repository build inputs or portable qualification evidence.

## Historical environment relocation (before unified cleanup)

The following table and migration results describe the earlier relocation only.
**Both destination environments below have since been removed.**

| Historical role | Former environment | Adopted installed payload |
| --- | --- | --- |
| MTP | `/Volumes/SDXC-512/ds41f-mlx_mtp-venv` | former `ds41f-generation-policy-venv`, non-editable frontend-complete wheel |
| Standard-OFF / bounded Vision | `/Volumes/SDXC-512/ds41f-mlx_off-venv` | former `ds41f-generation-policy-off-venv`, non-editable off-final wheel |

That relocation preserved existing installed execution resources; it was not a
publication of the latest checkout or an expansion of qualified runtime capabilities.

### Historical migration and checks

Fresh uv virtual environments were created with Python 3.13.15 at the new paths.
The adopted environments' `lib` and `share` resources were copied without changing
payload bytes. Text executable launchers were regenerated with the new
interpreter prefix, including 61 launchers per environment that still referenced
an earlier `m53r-normal-local-venv`; binary launchers were copied unchanged. Activation scripts
and Python links come from the newly created environments, not the old ones.

All regular `lib`/`share` files were compared byte-for-byte: 16,140 MTP files and
15,989 OFF files matched. Both new environments passed installed
`inspect_payload()`, `bin/ds41f --help` and `bin/pytest --version`; a final scan
found no obsolete environment paths in either new `bin` directory. No native rebuild, package upgrade,
manifest reseal, real-model acceptance or server restart was performed.
Migration records and inspection results are in `artifacts/venv-consolidation/`.

The following inactive environments were removed after checking open files with
`lsof`; their package inventories were recorded first:

- `m53r-admitted-venv`
- `m53r-fresh-venv`
- `m53r-local-venv`
- `m53r-normal-local-venv`
- `ds41f-generation-policy-venv`

At that migration, launch instructions and the frontend acceptance tool were updated
to the relocated MTP path. That path is no longer current. Original paths in historical
qualification documents, receipts and package build provenance remain unchanged;
they do not imply an existing environment.

### Historical OFF retirement

At migration, PID 25544 (`/tmp/ds41f-document-off-observer.py`) had loaded native
libraries from `ds41f-generation-policy-off-venv`, so deletion was initially
deferred. With explicit operator approval, that process was stopped with SIGTERM.
After confirming process exit and no remaining open files in the old environment,
its package inventory was recorded and the environment removed.

At completion of that earlier migration, the two relocated runtime environments
remained alongside the development repository. No replacement model server was
started and no real-model acceptance was run.
The separate Web client process was not stopped. Retirement details are recorded
in `artifacts/venv-consolidation/off-retirement.json`.

Build directories, `ds41f-runtime`, other projects, checkpoints and runtime state
were not removed or modified by that earlier relocation/retirement. The later
unified cleanup above had its own explicitly approved scope.
