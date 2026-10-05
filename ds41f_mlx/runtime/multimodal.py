"""Request-owned image inputs; no generation or cache authority.

Only inline encoded images are accepted: no implicit filesystem/network access.
Committed images are identified by encoded-byte digest + expanded span, never
re-encoded on continuation/restore. Their executable representation is the KV.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import base64
import re
import warnings
from time import perf_counter
from typing import Any

MAX_IMAGES = 4
MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 2048 * 2048
MAX_IMAGE_TOKENS = 1014
MAX_MULTIMODAL_CONTEXT = 8192


@dataclass(frozen=True)
class ImageSpan:
    start: int
    length: int
    digest: str
    grid: tuple[int, int]
    kinds: tuple[int, ...]
    patches: Any

    def identity(self) -> dict:
        return dict(start=self.start, length=self.length, sha256=self.digest,
                    grid=list(self.grid))


@dataclass(frozen=True)
class MultimodalInput:
    token_ids: tuple[int, ...]
    spans: tuple[ImageSpan, ...]

    def identities(self) -> list[dict]:
        return [s.identity() for s in self.spans]

    def validate_prefix(self, frontier: int, committed: list[dict]) -> None:
        old = [s.identity() for s in self.spans if s.start < frontier]
        if any(s.start < frontier < s.start + s.length for s in self.spans):
            raise ValueError('committed frontier intersects an image span')
        if old != committed:
            raise ValueError('continuation changed committed image identity')

    def encode_new(self, model: Any, frontier: int = 0) -> 'ImageEmbeddings':
        from ds41f_mlx.runtime.resource_admission import resources_for
        resources_for(model.language_model).validate_binding(model.language_model)
        import mlx.core as mx
        start = perf_counter()
        rows = []
        for span in self.spans:
            if span.start < frontier:
                continue
            h, w = span.grid
            patches = mx.array(span.patches).astype(mx.bfloat16)
            value = model.encode_image_span(patches, h, w, span.kinds)
            mx.eval(value)
            rows.append((span.start, value))
        return ImageEmbeddings(tuple(rows), model.config.image_token_id, perf_counter() - start)


@dataclass(frozen=True)
class ImageEmbeddings:
    """Short-lived sparse image rows, indexed in absolute token coordinates."""
    rows: tuple[tuple[int, Any], ...]
    image_token_id: int
    seconds: float = 0.0

    def merge(self, h: Any, start: int) -> Any:
        end = start + h.shape[1]
        for origin, values in self.rows:
            left, right = max(start, origin), min(end, origin + values.shape[0])
            if left < right:
                h[:, left-start:right-start] = values[left-origin:right-origin].astype(h.dtype)
        return h


def image_bytes(source: Any) -> bytes:
    if getattr(source, 'detail', 'high') not in ('high', 'auto'):
        raise ValueError('low-detail image resizing is not qualified; use high/auto')
    if source.kind == 'bytes':
        data = source.data
    elif source.kind == 'data_url':
        url = source.data_url
        if not isinstance(url, str) or len(url) > MAX_IMAGE_BYTES * 4 // 3 + 1024:
            raise ValueError('invalid or oversized image data URL')
        header, sep, payload = url.partition(',')
        if not sep or not re.fullmatch(r'data:image/(png|jpeg|webp);base64', header):
            raise ValueError('only base64 PNG/JPEG/WebP inline images are supported')
        try:
            data = base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise ValueError('invalid image base64') from exc
    else:
        raise ValueError('external image URLs/files are unsupported; supply inline image data')
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError('empty or oversized image bytes')
    return data


def prepare_multimodal(token_ids: list[int], sources: list[Any], config: Any) -> MultimodalInput:
    from PIL import Image
    from ds41f_mlx.model_execution.processing import image_patch_array
    if not config.vision_enabled:
        raise ValueError('checkpoint has no vision encoder')
    if not 0 < len(sources) <= MAX_IMAGES:
        raise ValueError(f'inline image count must be 1..{MAX_IMAGES}')
    if token_ids.count(config.image_token_id) != len(sources):
        raise ValueError('image placeholder/source count mismatch')
    expanded, spans = [], []
    sources = iter(sources)
    for token in token_ids:
        if token != config.image_token_id:
            expanded.append(token)
            continue
        data = image_bytes(next(sources))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(BytesIO(data)) as image:
                    if image.format not in ('PNG', 'JPEG', 'WEBP') or getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('unsupported or animated image')
                    if not 0 < image.width * image.height <= MAX_PIXELS:
                        raise ValueError('image pixel limit exceeded')
                    if not 0.5 <= image.width / image.height <= 2.0:
                        raise ValueError('image aspect ratio outside supported 1:2..2:1 envelope')
                    patches, h, w, kinds = image_patch_array(image, config)
                    if len(kinds) > MAX_IMAGE_TOKENS:
                        raise ValueError('expanded image exceeds qualified token envelope')
        except Exception as exc:
            raise ValueError(f'invalid image: {exc}') from exc
        spans.append(ImageSpan(len(expanded), len(kinds), sha256(data).hexdigest(),
                               (h, w), tuple(kinds), patches))
        expanded.extend([token] * len(kinds))
    if len(expanded) > MAX_MULTIMODAL_CONTEXT:
        raise ValueError('expanded multimodal context exceeds qualified envelope')
    if expanded[-1] == config.image_token_id:
        raise ValueError('held-out terminal must be a text token')
    return MultimodalInput(tuple(expanded), tuple(spans))
