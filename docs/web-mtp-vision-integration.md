# Browser and PDF ordinary MTP Vision integration

Browser image picker, paste and drop now consume READY
`capabilities.chat_vision: bounded-inline-v1` instead of treating production
sessions as text-only. Both ordinary MTP and standard-OFF advertise this existing
Chat Completions capability. Unknown/missing capability disables acquisition;
there is no profile-based inference or automatic OFF fallback.

Original bounded attachment preflight, previews and byte-preserving submission
are unchanged. The runtime remains the authority for formats, animation rejection,
encoded size, source geometry, image count, expanded multimodal positions, total
request capacity and executable retained image state. Application previews are
not executable state. Historical-image edits or lost eligible pairs still reject;
no replay, re-encoding, migration or persistence is added.

PDF observation defaults to the connected Chat runtime (including `--runtime-url`).
The existing operator `DS41F_DOCUMENT_VISION_URL` override remains explicit, not a
fallback. Availability uses the same advertised Vision capability. Each selected
page still follows:

```text
retained PDF -> transient raster -> isolated stateless Chat Vision request
             -> retained non-authoritative observation -> main-chat tool text
```

Exactly one page per observation; no session/history/tools in its inference
request. Temporary raster bytes are not retained in the observation or inserted
into main-chat image history. Source PDF/parser text, raster identity, model
observation provenance and executable model state stay separate. Reservation,
uncertainty fences and no-automatic-retry semantics are unchanged. Provenance now
also records the executing profile and Vision capability. The explicit page-dialog
“Add this page to ordinary chat Vision” action remains distinct and uses a normal
bounded attachment slot.

## Focused acceptance

`tools/accept_web_mtp_vision.py` is a small real-Chrome application delta, using
current checkout application/runtime source and separately admitted installed
MTP dependencies/assets, not new release or Vision qualification. It starts no
OFF runtime and provisions no separate PDF Vision endpoint. It checks text chat,
original-image submission/previews, native MTP Vision execution, then a PDF
`observe` tool loop with a stateless image execution and only derived text in the
main conversation. See the retained [receipt](web-mtp-vision-acceptance.json).

**PASS:** text chat returned normally; Browser identified the original corn
attachment. That image encoded once, with 22 native drafts considered / 14
accepted, settled C=1149. The PDF observation encoded one page, with 60 drafts
considered / 44 accepted, settled C=792; main chat reported 17 inventory units
from the labelled derived observation and contained no page raster/image slots.
Both Vision traces have all 40 target and 3 DSpark offsets aligned and zero
prompt replay / full-cache repack. UI controls and PDF capability presentation
were checked against the advertised ordinary MTP capability.

Run with the installed MTP interpreter, an explicit official `DS41F_CHECKPOINT`,
and the existing development Chrome/Playwright/PDF dependencies available. The
collector uses ports 18073/18074; services there must be stopped. On this checkout
those development-only dependencies were appended from `.venv` to the installed
interpreter's search path; isolated installed admission still runs with `-I`.

Focused regressions: **75 Python tests passed**, covering document observations,
reservation/uncertainty, application effects, PDF resources, capability policy,
ordinary MTP/sessions and multimodal rejection/state contracts. All **26 Node
tests passed**, including picker/paste/drop, unchanged bounds and unknown/unready
capability rejection. Broader legacy fixture attempts also encountered the absent
retired default recipe directory and test-order import-path pollution (oMLX outside
the selected interpreter); these were not repaired as part of this integration.
Documentation tests: 46 passed; the repository-wide closure check remains blocked
by three pre-existing unclassified files (`implementation-ownership-feasibility.md`,
`ordinary-output-budget-acceptance.json`, `shared-sampling-transfer.md`). This
change registers its own new contract/receipt and does not repair unrelated docs.
The original [runtime Vision evidence](mtp-vision-integration.md) remains execution
authority; historical OFF-only PDF receipts are explicitly marked historical.
