# PDF application workflow — retired

The bundled Browser PDF acquisition, retained documents, page observation and
application tool execution are retired. No PDF parser/acquisition workflow was
moved into Runtime. Bounded inline images directly supported by Runtime remain
available to external clients; that is not PDF/Files or Harness qualification.
No replacement Harness is bundled.

## Historical evidence

The following unchanged excerpt describes the former collector and application,
not current instructions. The referenced collectors were removed with Browser
Chat; source remains in Git history and receipts remain archival evidence only.

See [PDF acceptance](pdf-document-acceptance.md) for unchanged historical source
scope and original receipts. The old Python Chat frontend/qualification workflow
has been removed; `tools/qualify_pdf_documents.py` retains only shared PDF fixtures.
Current real-browser/operator multi-conversation acceptance includes a retained
PDF tool loop:
