# Source identity hash contract

This project uses source identity metadata only to identify reviewed official reference source. It is not model math and is not a numerical oracle.

## Cause of previous hash divergence

The same official file SHA and line spans had two different `source_sha256` values because two hash methods were mixed:

1. **Raw inclusive line span**: SHA-256 over the exact UTF-8 text of the 1-indexed inclusive line span, preserving indentation and line endings.
2. **`ast.get_source_segment`**: SHA-256 over the Python AST node source segment. For these node spans it omits the final line ending, so the digest differs even though the source content and line span are the same.

No source-content divergence was found for the audited pinned files.

## Canonical metadata contract

Every source identity entry should record:

```json
{
  "file_sha256": "...",
  "source_lines": [start, end],
  "source_hash_method": "raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1",
  "source_sha256": "..."
}
```

The canonical `source_sha256` is computed by `tools/source_identity.py` as SHA-256 over the raw inclusive line span from the pinned official source file.

## Audited examples

For official `model.py` SHA `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`:

| Span | Canonical raw inclusive hash | Previous AST-segment hash |
|---|---|---|
| `Attention` 613-789 | `86d80f5cdaa6435cacd56ce5be796c3f0155a7f92cebdb12ffe6743ac974d110` | `5d0ae1d7eb0225130ebe382aa7dc2bfd82bbfd033c3038950bf510f7f6025325` |
| `Compressor` 429-485 | `dcd32a8debcf46c4d19d0347a3bc982e7aa70bba9746845d0b1555a7a73c8d67` | `a41d9e96f58a21559d8cac5b3356d50201597251809c3c72905a70de0163a21f` |
| `Indexer` 488-580 | `cb9d882d1701f3e62892e7730fe6901658e39886c55af65ece1b830858ca0a75` | `e4d1a9931037796b7a3a2ec62745ae4f27279d18f8fa1c483fb366a9e5a3aa32` |

For official `kernel.py`:

| Span | Canonical raw inclusive hash | Previous AST-segment hash |
|---|---|---|
| `sparse_attn` 392-403 | `42208bc5467f5a29efd18020b62162fa3177614293f3669d3d0b8800d6e5d704` | `e8438cf9b0bc0dba1cf3ceb4b571271b161d69ad9de417f80c6904d06935a8bc` |
| `fp4_act_quant` 184-204 | `1066960c1da76f484a12ae2b456daa4734eabe618f77fd781f3c98d80b05db99` | `4dc2c2e60c261c8a9a224d6c88a1a99de757716e64c55b330b3129593892602a` |

## Guard

`tools/check_source_identity_hashes.py` scans artifact JSON and fails if any source identity entry has a missing/non-canonical `source_hash_method` or a digest that does not reproduce from `{file, source_lines, source_hash_method}`.

Audit record: `artifacts/source-identity-hash-audit.json`.
