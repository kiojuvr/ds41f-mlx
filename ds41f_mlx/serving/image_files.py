"""Bounded process-local image bytes, not a Vision or KV owner.

The ordinary server's trusted deployment is the ownership scope. No header is
an authentication mechanism. Instances never share IDs or storage. Immutable
bytes survive removal only in requests that already acquired a snapshot.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
import re
from threading import RLock
from time import time
from uuid import uuid4

from ds41f_mlx.runtime.multimodal import (
    MAX_IMAGE_BYTES, MAX_REQUEST_IMAGE_BYTES, MAX_REQUEST_IMAGES, MAX_REQUEST_PIXELS,
    validate_encoded_image,
)

UPLOAD_BODY_BYTES = MAX_IMAGE_BYTES + 64 * 1024
MAX_OCCURRENCES = MAX_REQUEST_IMAGES


class FileError(ValueError):
    def __init__(self, message, status=400, code='invalid_file_request'):
        super().__init__(message)
        self.status_code, self.code = status, code


@dataclass(frozen=True)
class FileLimits:
    file_bytes: int = MAX_IMAGE_BYTES
    count: int = 128
    storage_bytes: int = MAX_REQUEST_IMAGE_BYTES

    def __post_init__(self):
        for value, ceiling in ((self.file_bytes, MAX_IMAGE_BYTES),
                               (self.count, 128), (self.storage_bytes, MAX_REQUEST_IMAGE_BYTES)):
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError('Files limits may only lower the deployment ceilings')


@dataclass(frozen=True)
class StoredImage:
    data: bytes
    mime: str
    filename: str
    created: int
    expires: int
    pixels: int


class ImageFiles:
    def __init__(self, limits=None, *, clock=time):
        self.limits = limits or FileLimits()
        self.clock = clock
        self._files = {}
        self._lock = RLock()

    def sweep(self):
        with self._lock:
            now = self.clock()
            for key in list(self._files):
                if self._files[key].expires <= now:
                    del self._files[key]

    def clear(self):
        with self._lock:
            self._files.clear()

    def upload(self, body, content_type):
        # The HTTP boundary serializes uploads/preparation and bounds ingress
        # before buffering. This parser also stays bounded for direct callers.
        if len(body) > min(UPLOAD_BODY_BYTES, self.limits.file_bytes + 64 * 1024):
            raise FileError('file upload too large', 413)
        if not isinstance(content_type, str) or len(content_type) > 1024 or '\r' in content_type or '\n' in content_type:
            raise FileError('invalid multipart content type')
        values = parse_upload(body, content_type, self.limits.file_bytes)
        anchor = values[b'expires_after[anchor]'][1]
        seconds = values[b'expires_after[seconds]'][1]
        if anchor != b'created_at' or not re.fullmatch(rb'[0-9]{4,7}', seconds) or not 3600 <= int(seconds) <= 2592000:
            raise FileError('unsupported file expiry')
        headers, data, filename = values[b'file']
        mime = headers.get(b'content-type', b'').decode('ascii', errors='replace')
        if not filename or len(filename) > 512 or any(ord(c) < 32 or ord(c) == 127 for c in filename):
            raise FileError('invalid file filename')
        if mime not in ('image/png', 'image/jpeg', 'image/webp'):
            raise FileError('unsupported image MIME')
        if len(data) > self.limits.file_bytes:
            raise FileError('file upload too large', 413)
        try:
            metadata = validate_encoded_image(data, mime)
        except Exception as exc:
            raise FileError(f'invalid image: {exc}') from exc
        with self._lock:
            self.sweep()
            if len(self._files) >= self.limits.count or sum(len(f.data) for f in self._files.values()) + len(data) > self.limits.storage_bytes:
                raise FileError('file storage quota exceeded', 429, 'file_storage_quota')
            created = int(self.clock())
            key = 'file_' + uuid4().hex
            self._files[key] = StoredImage(data, mime, filename, created, created + int(seconds),
                                            metadata['width'] * metadata['height'])
        return dict(id=key, type='file', mime_type=mime, size_bytes=len(data),
                    created_at=datetime.fromtimestamp(created, timezone.utc).isoformat().replace('+00:00', 'Z'),
                    filename=filename)

    def snapshot(self, key):
        with self._lock:
            self.sweep()
            if not isinstance(key, str) or not re.fullmatch(r'file_[0-9a-f]{32}', key) or key not in self._files:
                # Matches rc.2 providerRejectedFileId; do not reveal other scopes.
                raise FileError(f'file not found or expired: {key}', 404, 'file_not_found')
            return self._files[key]

    def delete(self, key):
        with self._lock:
            self.snapshot(key)
            del self._files[key]
        return dict(id=key, type='file_deleted')


def parse_upload(body, content_type, file_limit):
    # Reuse the already pinned HTTP multipart dependency. Limit part/header
    # counts as well as bytes: a small body must not create unbounded objects.
    from python_multipart import MultipartParser
    from python_multipart.multipart import parse_options_header
    kind, options = parse_options_header(content_type)
    boundary = options.get(b'boundary', b'')
    if kind != b'multipart/form-data' or not boundary or len(boundary) > 70 or not re.fullmatch(rb"[0-9A-Za-z'()+_,./:=? -]+", boundary):
        raise FileError('invalid multipart boundary')
    if not body.startswith(b'--' + boundary + b'\r\n') or not body.endswith((b'\r\n--' + boundary + b'--', b'\r\n--' + boundary + b'--\r\n')):
        raise FileError('incomplete multipart upload')
    values, headers = {}, {}
    field, value, data = bytearray(), bytearray(), bytearray()
    name = filename = None
    complete = False
    parts = 0

    def begin():
        nonlocal parts
        parts += 1
        if parts > 3:
            raise FileError('exactly three multipart fields required')
        headers.clear()
        data.clear()

    def header_begin():
        field.clear()
        value.clear()

    def header_end():
        key = bytes(field).lower()
        if key in headers or key not in (b'content-disposition', b'content-type'):
            raise FileError('unknown or duplicate multipart header')
        headers[key] = bytes(value)

    def headers_finished():
        nonlocal name, filename
        disposition, params = parse_options_header(headers.get(b'content-disposition'))
        name = params.get(b'name')
        if disposition != b'form-data' or name not in (b'file', b'expires_after[anchor]', b'expires_after[seconds]') or name in values:
            raise FileError('unknown or duplicate multipart field')
        filename = params.get(b'filename')
        if set(params) - {b'name', b'filename'} or (name != b'file' and filename is not None):
            raise FileError('invalid multipart field schema')

    def part_data(chunk, start, end):
        limit = file_limit if name == b'file' else 32
        if len(data) + end - start > limit:
            raise FileError('file upload too large' if name == b'file' else 'invalid expiry field',
                            413 if name == b'file' else 400)
        data.extend(chunk[start:end])

    def part_end():
        try:
            text = filename.decode('utf-8') if filename is not None else None
        except UnicodeError as exc:
            raise FileError('invalid filename encoding') from exc
        values[name] = (dict(headers), bytes(data), text)

    def end():
        nonlocal complete
        complete = True

    parser = MultipartParser(boundary, callbacks={
        'on_part_begin': begin, 'on_header_begin': header_begin,
        'on_header_field': lambda chunk, start, end: field.extend(chunk[start:end]),
        'on_header_value': lambda chunk, start, end: value.extend(chunk[start:end]),
        'on_header_end': header_end, 'on_headers_finished': headers_finished,
        'on_part_data': part_data, 'on_part_end': part_end, 'on_end': end,
    }, max_size=UPLOAD_BODY_BYTES, max_header_count=2, max_header_size=2048)
    try:
        for offset in range(0, len(body), 65536):
            parser.write(body[offset:offset + 65536])
        parser.finalize()
    except FileError:
        raise
    except ValueError as exc:
        raise FileError('malformed multipart upload') from exc
    if not complete or len(values) != 3:
        raise FileError('incomplete multipart upload')
    return values


def resolve_messages(payload, store):
    """Validate original native grammar, then replace only user image sources.

    All other fields remain untouched for the existing strict capability gate.
    Never traverse arbitrary objects (e.g. tool inputs or extension metadata).
    """
    def has_file(content):
        return isinstance(content, list) and any(
            isinstance(b, dict) and (
                b.get('type') == 'image' and isinstance(b.get('source'), dict) and b['source'].get('type') == 'file'
                or b.get('type') == 'tool_result' and has_file(b.get('content')))
            for b in content)
    if not (has_file(payload.get('system')) or any(
            isinstance(m, dict) and has_file(m.get('content')) for m in payload.get('messages', []) or [])):
        return payload
    import deepseek_recipe as d
    from .messages_admission import fields
    from .generation_policy import with_sampling_defaults
    original = dict(payload)
    original.pop('dsh_plugin_packages', None)
    original = with_sampling_defaults(original)
    if original.get('max_tokens') == 'auto':
        original['max_tokens'] = 1
    d.MessagesRequest(json.dumps(original).encode())  # original structure, not a repaired request
    # JSON owns this request-local tree; no shared mutable objects are published.
    count = encoded = pixels = 0

    def visit(content, *, allowed, nested=False):
        nonlocal count, encoded, pixels
        if not isinstance(content, list):
            return
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_result':
                visit(block.get('content'), allowed=allowed and not nested, nested=True)
            elif block.get('type') == 'image':
                source = block.get('source')
                if not isinstance(source, dict):
                    continue
                count += 1
                if count > MAX_OCCURRENCES:
                    raise FileError('image occurrence budget exceeded')
                if source.get('type') == 'file':
                    fields(block, 'type source')
                    fields(source, 'type file_id')
                    if not allowed or set(source) != {'type', 'file_id'} or not isinstance(source['file_id'], str):
                        raise FileError('file image in unsupported location or schema')
                    image = store.snapshot(source['file_id'])
                    encoded += len(image.data)
                    pixels += image.pixels
                    # Physical all-history guards remain independent of Harness
                    # request budgets and actual fresh encoder work.
                    # Check BEFORE allocating the base64 request-local copy.
                    if encoded > MAX_REQUEST_IMAGE_BYTES or pixels > MAX_REQUEST_PIXELS:
                        raise FileError('aggregate image budget exceeded')
                    block['source'] = dict(type='base64', media_type=image.mime,
                                           data=base64.b64encode(image.data).decode('ascii'))
                elif source.get('type') == 'base64':
                    data = source.get('data', '')
                    if isinstance(data, str):
                        encoded += (len(data) // 4) * 3 - min(2, len(data) - len(data.rstrip('=')))
                if encoded > MAX_REQUEST_IMAGE_BYTES:
                    raise FileError('aggregate encoded image byte budget exceeded')

    visit(payload.get('system'), allowed=False)
    for message in payload.get('messages', []):
        visit(message.get('content'), allowed=message.get('role') == 'user')
    return payload
