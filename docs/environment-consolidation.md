# Local environment consolidation

## Current layout

Only `/Volumes/SDXC-512/ds41f-mlx` is the development repository. Its internal
`.venv` remains a development/tooling environment. Installed runtime environments:

| Role | Environment | Adopted installed payload |
| --- | --- | --- |
| MTP | `/Volumes/SDXC-512/ds41f-mlx_mtp-venv` | former `ds41f-generation-policy-venv`, non-editable frontend-complete wheel |
| Standard-OFF / bounded Vision | `/Volumes/SDXC-512/ds41f-mlx_off-venv` | former `ds41f-generation-policy-off-venv`, non-editable off-final wheel |

This is a relocation of existing installed execution resources, not a publication
of the latest checkout or an expansion of qualified runtime capabilities.

## Migration and checks

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

Current launch instructions and the frontend acceptance tool now use the new MTP
path. Original paths in historical qualification documents, receipts and package
build provenance remain unchanged; they do not imply an existing environment.

## Completed OFF retirement

At migration, PID 25544 (`/tmp/ds41f-document-off-observer.py`) had loaded native
libraries from `ds41f-generation-policy-off-venv`, so deletion was initially
deferred. With explicit operator approval, that process was stopped with SIGTERM.
After confirming process exit and no remaining open files in the old environment,
its package inventory was recorded and the environment removed.

Only the two new runtime environments remain alongside the development repository.
No replacement model server was started and no real-model acceptance was run.
The separate Web client process was not stopped. Retirement details are recorded
in `artifacts/venv-consolidation/off-retirement.json`.

Build directories, `ds41f-runtime`, other projects, checkpoints and runtime state
were not removed or modified by this environment cleanup.
