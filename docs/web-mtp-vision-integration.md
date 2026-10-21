# Browser and PDF ordinary MTP Vision integration — retired

Browser Chat, PDF observation and their collectors are retired. There is no
application Vision router, acquisition tool or page workflow in the bundled
product. No replacement Harness exists. Direct bounded Runtime Vision remains
supported for external clients; Runtime code, Recipe 0.2 semantics and logical
sessions are unchanged. Successful inference is not comprehensive Agent/Harness
compatibility.

## Focused acceptance

Historical evidence only: the unchanged excerpt below describes a removed
collector and former application, not instructions to run it. Original source,
full measurements and removed tests remain in Git history. Receipt inputs are
not current application or Runtime/Harness qualification.

`tools/accept_web_mtp_vision.py` is a small real-Chrome application delta, using
current checkout application/runtime source and separately admitted installed
MTP dependencies/assets, not new release or Vision qualification. It starts no
OFF runtime and provisions no separate PDF Vision endpoint. It checks text chat,
original-image submission/previews, native MTP Vision execution, then a PDF
`observe` tool loop with a stateless image execution and only derived text in the
main conversation. See the retained [receipt](web-mtp-vision-acceptance.json).
