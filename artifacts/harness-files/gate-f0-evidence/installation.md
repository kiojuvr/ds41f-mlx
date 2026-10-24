# Disposable installation execution record

Tested source: `6766b652fd3293f7523533decc102a9697168af1`.
No persistent installation was replaced or modified.

1. Built current checkout with `/tmp/ds41f-phase1-1017-tests/bin/python tools/build_execution_package.py --checkpoint-variant flash-1017 --glm-profile native --output /tmp/ds41f-f0-wheels` (`build.log`).
2. Created `/tmp/ds41f-f0-runtime` using `uv venv --python /opt/homebrew/opt/python@3.13/bin/python3.13`.
3. Froze the existing disposable Phase 1 environment; removed only its Runtime line and installed every other exact package with `uv pip install --offline --no-deps`. Installed the newly built Runtime wheel with the same flags (`install.log`). `dependencies-before.txt` and `dependencies-after.txt` compare equal excluding the Runtime wheel location. No dependency upgrade or solver-selected substitution.
4. Initial bootstrap was rejected at payload admission: native GLM bundle not yet provisioned. No model loading/conversation. Archived in `startup-unprovisioned/`.
5. Called the existing checkout `ds41f_mlx.glm_profile.provision('/tmp/ds41f-f0-runtime/lib/python3.13/site-packages', 'native')`, the exact native resource provisioner used by `mtp_setup.py`.
6. Next bootstrap was rejected at tokenizer admission: wheel installation does not provision the external tokenizer asset. No model loading/conversation. Archived in `startup-missing-tokenizer/`.
7. Copied `/tmp/ds41f-phase1-1017-build/recipe/static/tokenizers/v41/tokenizer.json` to `/tmp/ds41f-f0-runtime/share/ds41f-mtp/recipe/static/tokenizers/v41/tokenizer.json`, matching `mtp_setup.py`'s resource staging. Ran isolated `execution_identity.inspect_payload()` successfully (`payload-admission.json`); normal admission verifies its exact publisher hash, native bundle, dependencies and installed origins.
8. Created another fresh isolated Harness root after each rejected bootstrap. No attachment or upload-index state is shared. The final bootstrap performs unchanged checkpoint full-byte admission, with no copied identity-cache exemption, before backend construction and model loading.

`source-package-match.json` independently compares every checkout Runtime Python file to the installed wheel. The execution bootstrap additionally repeats this check in-process before numerical backend imports. Bootstrap source changes are solely revision assertion, complete source check and the existing passive preparation/trace retention evaluation instrumentation.

The observer's file `native_block` projection was added before Harness launch, during checkpoint admission. It records the actual image source object without changing the request. `prepare.py` records the initial derivation; executed `.py`/`.mjs` files are authoritative.
