#!/bin/bash
# Sequential same-fixture production requalification after passive P5 retirement.
# Requires admitted PYTHONPATH, DS41F_KV_ROOT and DS41F_QUALIFICATION_PYTHON.
set -euo pipefail
P="${DS41F_QUALIFICATION_PYTHON:-python}"
R="${1:-artifacts/very-long-context}"
OLD="$R/before-certificate-retirement"
for spec in '512k-wired 512k 524288' '768k-initial 768k 786432' '1m-initial 1m 1040090'; do
    read -r initial label context <<< "$spec"
    "$P" tools/qualify_standard_off_long_session.py --context "$context" \
        --fixture-revision 31a7e1d --turns 8 --decode-tokens 128 \
        --compare "$OLD/$initial.json" --output "$R/$initial.json" \
        --artifact-root "${DS41F_KV_ROOT:?}" > "$R/$initial.log" 2>&1
    "$P" tools/qualify_standard_off_long_session.py --restore "$R/$initial.json" \
        --turns 4 --decode-tokens 128 --compare "$OLD/$label-restored.json" \
        --output "$R/$label-restored.json" --artifact-root "$DS41F_KV_ROOT" \
        > "$R/$label-restored.log" 2>&1
    "$P" tools/summarize_very_long_context.py "$R/$initial.json" "$R/$label-restored.json" \
        --output "$R/$label-summary.json"
done
"$P" tools/qualify_standard_off_long_session.py --restore "$R/1m-restored.json" \
    --restore-probe-only --output "$R/1m-ceiling-restored.json" \
    --artifact-root "$DS41F_KV_ROOT" > "$R/1m-ceiling-restored.log" 2>&1
"$P" tools/summarize_very_long_context.py "$R/1m-initial.json" "$R/1m-restored.json" \
    "$R/1m-ceiling-restored.json" --output "$R/1m-summary.json"
"$P" tools/close_very_long_context.py --root "$R"
