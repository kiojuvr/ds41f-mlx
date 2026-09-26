#!/usr/bin/env python3
"""Canonical source identity hashing for official reference spans.

Contract v1: SHA-256 over the raw UTF-8 text of the inclusive 1-indexed
line span as read from the pinned source file, preserving indentation and line
endings. This intentionally differs from ast.get_source_segment, which omits the
trailing newline after the final line of a node.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_CHECKPOINT = Path('/Volumes/KIOXIA-PRO-1/models/deepseek-ai/DeepSeek-V4.1-Flash')
SOURCE_HASH_METHOD = 'raw_utf8_inclusive_1indexed_line_span_preserve_line_endings_v1'


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_inclusive_line_span(path: str | Path, start_line: int, end_line: int) -> str:
    if start_line < 1 or end_line < start_line:
        raise ValueError(f'invalid source line span: {start_line}-{end_line}')
    lines = Path(path).read_text().splitlines(keepends=True)
    if end_line > len(lines):
        raise ValueError(f'end line {end_line} beyond file length {len(lines)}: {path}')
    return ''.join(lines[start_line - 1:end_line])


def source_sha256(path: str | Path, start_line: int, end_line: int) -> str:
    return hashlib.sha256(raw_inclusive_line_span(path, start_line, end_line).encode('utf-8')).hexdigest()


def official_source_path(file_name: str, checkpoint: str | Path = DEFAULT_CHECKPOINT) -> Path:
    p = Path(file_name)
    if p.is_absolute():
        return p
    return Path(checkpoint) / file_name


def source_identity(file_name: str, start_line: int, end_line: int, checkpoint: str | Path = DEFAULT_CHECKPOINT) -> dict[str, object]:
    path = official_source_path(file_name, checkpoint)
    return {
        'file': file_name,
        'file_sha256': file_sha256(path),
        'source_lines': [int(start_line), int(end_line)],
        'source_hash_method': SOURCE_HASH_METHOD,
        'source_sha256': source_sha256(path, int(start_line), int(end_line)),
    }
